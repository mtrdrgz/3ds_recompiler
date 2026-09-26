// HLE of the Horizon (3DS) kernel: objects, handles, a strictly
// single-core priority scheduler and the SVC table.
//
// Every guest thread is a host thread, but exactly one of them runs guest
// code at a time (the one in `g_k.current`). Switches happen inside SVCs
// and on the ~1 ms kernel_poll tick: a strictly higher-priority thread
// preempts immediately, and equal-priority threads round-robin on the tick
// (the real kernel time-slices them, and without it a spin-polling thread
// starves same-priority workers that on hardware would run on the other
// core).
#pragma once
#include "cpu.h"
#include <memory>
#include <vector>
#include <map>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <string>
#include <functional>

using Result = u32;
constexpr Result RES_OK = 0;
constexpr Result RES_TIMEOUT = 0x09401BFE;
constexpr Result RES_INVALID_HANDLE = 0xD8E007F7;
constexpr Result RES_NOT_FOUND = 0xD88007FA;
constexpr Result RES_INVALID_COMBINATION = 0xE0E01BEE;

enum class KType { Thread, Event, Mutex, Semaphore, Timer, Arbiter, SharedMem, Session, Port, Process, ResLimit, Dummy };

struct Thread;
struct KObject : std::enable_shared_from_this<KObject> {
    KType type;
    std::string name;
    std::vector<Thread *> waiters;
    explicit KObject(KType t) : type(t) {}
    virtual ~KObject() {}
    virtual bool waitable() const { return false; }
    virtual bool should_wait(Thread *) { return true; }
    virtual void acquire(Thread *) {}
    void wake_waiters();   // kernel lock held
};
using KPtr = std::shared_ptr<KObject>;

enum class TState { Ready, Running, WaitSync, WaitSleep, WaitArb, WaitIpc, Dead };

struct Thread : KObject {
    Cpu cpu{};
    u32 id = 0;
    s32 prio = 0x30;
    u32 entry = 0, arg = 0, stack_top = 0, tls = 0;
    s32 proc_id = 0;
    TState state = TState::Ready;
    u64 seq = 0;                        // FIFO order among equal priorities
    std::condition_variable cv;
    // wait bookkeeping
    std::vector<KPtr> wait_objs;
    bool wait_all = false;
    s32 wait_index = -1;
    u64 wake_ns = ~0ull;
    Result wait_result = 0;
    u32 arb_addr = 0;
    Thread() : KObject(KType::Thread) {}
    bool waitable() const override { return true; }
    bool should_wait(Thread *) override { return state != TState::Dead; }
};

struct Event : KObject {
    u32 reset;   // 0 oneshot, 1 sticky, 2 pulse
    bool signaled = false;
    Event(u32 r) : KObject(KType::Event), reset(r) {}
    bool waitable() const override { return true; }
    bool should_wait(Thread *) override { return !signaled; }
    void acquire(Thread *) override { if (reset == 0) signaled = false; }
    void signal();
    void clear() { signaled = false; }
};

struct Timer : KObject {
    u32 reset;
    bool signaled = false;
    u64 next_ns = ~0ull, interval_ns = 0;
    Timer(u32 r) : KObject(KType::Timer), reset(r) {}
    bool waitable() const override { return true; }
    bool should_wait(Thread *) override { return !signaled; }
    void acquire(Thread *) override { if (reset == 0) signaled = false; }
};

struct Mutex : KObject {
    Thread *owner = nullptr;
    u32 count = 0;
    Mutex() : KObject(KType::Mutex) {}
    bool waitable() const override { return true; }
    bool should_wait(Thread *t) override { return owner && owner != t; }
    void acquire(Thread *t) override { owner = t; count++; }
};

struct Semaphore : KObject {
    s32 count, max;
    Semaphore(s32 c, s32 m) : KObject(KType::Semaphore), count(c), max(m) {}
    bool waitable() const override { return true; }
    bool should_wait(Thread *) override { return count <= 0; }
    void acquire(Thread *) override { count--; }
};

struct SharedMem : KObject {
    u32 addr = 0;          // backing guest address (0 = kernel allocated)
    u32 size = 0;
    u32 my_perm = 0, other_perm = 0;
    std::vector<u32> mapped_at;
    SharedMem() : KObject(KType::SharedMem) {}
};

struct Session;
struct Service;
struct Port : KObject {
    Service *svc;
    Port(Service *s) : KObject(KType::Port), svc(s) {}
};

struct Kernel {
    std::mutex mtx;
    std::condition_variable idle_cv;
    Thread *current = nullptr;
    std::vector<Thread *> ready;
    std::vector<std::shared_ptr<Thread>> threads;
    std::map<u32, KPtr> handles;
    u32 next_handle = 0x100;
    u64 seq_counter = 0;
    u32 next_tid = 1;
    u32 tls_next = TLS_VA_START;
    std::vector<Thread *> arb_waiters;
    std::vector<std::shared_ptr<Timer>> timers;
    struct Periodic { u64 next_ns, period_ns; std::function<void()> fn; };
    std::vector<Periodic> periodic;
    bool exiting = false;
    static constexpr u32 TLS_VA_START = 0x1FF82000;

    u32 new_handle(KPtr o);
    KPtr get(u32 h);
    template <class T> std::shared_ptr<T> get_as(u32 h, KType t) {
        KPtr o = get(h);
        if (!o || o->type != t) return nullptr;
        return std::static_pointer_cast<T>(o);
    }

    std::shared_ptr<Thread> create_thread(u32 entry, u32 arg, u32 stack_top, s32 prio, s32 proc, const char *name);
    void make_ready(Thread *t, bool front = false);
    void block_current(std::unique_lock<std::mutex> &lk);   // current has set its wait state
    void yield_current(std::unique_lock<std::mutex> &lk);
    void maybe_preempt(std::unique_lock<std::mutex> &lk);
    void timeslice(std::unique_lock<std::mutex> &lk);       // RR among equal priorities on the poll tick
    void switch_away(std::unique_lock<std::mutex> &lk, Thread *self);
    void wake_thread(Thread *t, Result res, s32 index);
    void process_time(u64 now);
    u64 next_deadline();
    void add_periodic(u64 period_ns, std::function<void()> fn);
    void start_main(u32 entry, u32 stack_top, s32 prio);
};

extern Kernel g_k;
u64 now_ns();
u64 now_ticks();
Thread *cur_thread();

// helpers for services
void kernel_signal_event(const std::shared_ptr<Event> &e);   // takes no lock (caller in svc)
