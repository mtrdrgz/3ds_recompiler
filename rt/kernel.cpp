#include <pthread.h>
#include "kernel.h"
#include "platform.h"
#include "mem.h"
#include "services.h"
#include "lift_rt.h"
#include "hwr.h"
#include <chrono>
#include <algorithm>
#include <cstdarg>
#include <atomic>

Kernel g_k;
int g_log_level = 1;
int g_svc_log = 0;
struct ThreadExit {};

static const auto T0 = std::chrono::steady_clock::now();
// Monotonic across threads. In the browser every thread is a worker whose
// clock is timeOrigin + performance.now(), and Safari's workers can disagree
// slightly: a value read on one thread could be earlier than one read before
// on another. The game then saw time run backwards (svcGetSystemTick), which
// jolts its movement, and interval arithmetic in the runtime wrapped around.
u64 now_ns() {
    static std::atomic<u64> last{0};
    u64 t = (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - T0).count();
    u64 prev = last.load(std::memory_order_relaxed);
    while (t > prev && !last.compare_exchange_weak(prev, t, std::memory_order_relaxed)) {}
    return t > prev ? t : prev;
}
u64 now_ticks() { return (u64)((unsigned __int128)now_ns() * 268111856u / 1000000000u); }
Thread *cur_thread() { return g_k.current; }

void fatal(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[fatal] ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    fflush(stderr);
    plat_exit(1);
}

void dump_cpu(const Cpu &c) {
    fprintf(stderr, "  r0=%08x r1=%08x r2=%08x r3=%08x r4=%08x r5=%08x r6=%08x r7=%08x\n"
                    "  r8=%08x r9=%08x r10=%08x r11=%08x r12=%08x sp=%08x lr=%08x pc=%08x %s\n",
            c.r[0], c.r[1], c.r[2], c.r[3], c.r[4], c.r[5], c.r[6], c.r[7], c.r[8], c.r[9],
            c.r[10], c.r[11], c.r[12], c.r[13], c.r[14], c.r[15], c.thumb ? "T" : "A");
    fprintf(stderr, "  stack:");
    u32 sp = c.r[13];
    int shown = 0;
    for (u32 p = sp; p < sp + 0x800 && shown < 24 && mem_is_mapped(p); p += 4) {
        u32 w = rd32(p);
        if (w >= IMG_BASE && w < TEXT_END) { fprintf(stderr, " %08x", w); shown++; }
    }
    fprintf(stderr, "\n");
}

// ---------------------------------------------------------------- objects
u32 Kernel::new_handle(KPtr o) {
    u32 h = next_handle++;
    handles[h] = std::move(o);
    return h;
}

KPtr Kernel::get(u32 h) {
    if (h == 0xFFFF8000) {
        for (auto &t : threads) if (t.get() == current) return t;
        return nullptr;
    }
    auto it = handles.find(h);
    return it == handles.end() ? nullptr : it->second;
}

void Kernel::wake_thread(Thread *t, Result res, s32 index) {
    for (auto &o : t->wait_objs) {
        auto &w = o->waiters;
        w.erase(std::remove(w.begin(), w.end(), t), w.end());
    }
    t->wait_objs.clear();
    arb_waiters.erase(std::remove(arb_waiters.begin(), arb_waiters.end(), t), arb_waiters.end());
    t->wait_result = res;
    t->wait_index = index;
    t->wake_ns = ~0ull;
    make_ready(t);
}

void KObject::wake_waiters() {
    std::vector<Thread *> ws = waiters;
    std::stable_sort(ws.begin(), ws.end(), [](Thread *a, Thread *b) { return a->prio < b->prio; });
    for (Thread *w : ws) {
        if (w->state != TState::WaitSync) continue;
        if (w->wait_all) {
            bool ok = true;
            for (auto &o : w->wait_objs) if (o->should_wait(w)) { ok = false; break; }
            if (!ok) continue;
            for (auto &o : w->wait_objs) o->acquire(w);
            g_k.wake_thread(w, RES_OK, -1);
        } else {
            if (should_wait(w)) continue;
            acquire(w);
            s32 idx = -1;
            for (size_t i = 0; i < w->wait_objs.size(); i++)
                if (w->wait_objs[i].get() == this) { idx = (s32)i; break; }
            g_k.wake_thread(w, RES_OK, idx);
        }
    }
}

void Event::signal() {
    signaled = true;
    wake_waiters();
    if (reset == 2) signaled = false;
}
void kernel_signal_event(const std::shared_ptr<Event> &e) { if (e) e->signal(); }

// -------------------------------------------------------------- scheduler
void Kernel::make_ready(Thread *t, bool front) {
    t->state = TState::Ready;
    t->seq = front ? 0 : ++seq_counter;
    if (std::find(ready.begin(), ready.end(), t) == ready.end()) ready.push_back(t);
    idle_cv.notify_all();
}

static Thread *pick_next() {
    auto &r = g_k.ready;
    if (r.empty()) return nullptr;
    auto it = std::min_element(r.begin(), r.end(), [](Thread *a, Thread *b) {
        return a->prio != b->prio ? a->prio < b->prio : a->seq < b->seq;
    });
    Thread *t = *it;
    r.erase(it);
    return t;
}

u64 Kernel::next_deadline() {
    u64 d = ~0ull;
    for (auto &t : threads) if (t->state != TState::Dead && t->state != TState::Ready && t->state != TState::Running) d = std::min(d, t->wake_ns);
    for (auto &tm : timers) d = std::min(d, tm->next_ns);
    for (auto &p : periodic) d = std::min(d, p.next_ns);
    return d;
}

void Kernel::process_time(u64 now) {
    for (auto &t : threads) {
        if (t->wake_ns > now) continue;
        if (t->state == TState::WaitSleep) wake_thread(t.get(), RES_OK, -1);
        else if (t->state == TState::WaitSync || t->state == TState::WaitArb) wake_thread(t.get(), RES_TIMEOUT, -1);
    }
    for (auto &tm : timers) {
        if (tm->next_ns > now) continue;
        tm->signaled = true;
        tm->wake_waiters();
        if (tm->reset == 2) tm->signaled = false;
        if (tm->interval_ns) { tm->next_ns += tm->interval_ns; if (tm->next_ns < now) tm->next_ns = now + tm->interval_ns; }
        else tm->next_ns = ~0ull;
    }
    for (size_t i = 0; i < periodic.size(); i++) {
        if (periodic[i].next_ns > now) continue;
        periodic[i].next_ns += periodic[i].period_ns;
        if (periodic[i].next_ns < now) periodic[i].next_ns = now + periodic[i].period_ns;
        periodic[i].fn();
    }
}

void Kernel::add_periodic(u64 period_ns, std::function<void()> fn) {
    periodic.push_back({now_ns() + period_ns, period_ns, std::move(fn)});
}

void Kernel::switch_away(std::unique_lock<std::mutex> &lk, Thread *self) {
    for (;;) {
        process_time(now_ns());
        Thread *next = pick_next();
        if (!next) {
            u64 dl = next_deadline();
            if (dl == ~0ull) fatal("deadlock: no runnable threads and no timers");
            u64 now = now_ns();
            if (dl > now)
            {   // no guest thread can run: the emulated CPU is idle (diagnostics)
                idle_cv.wait_for(lk, std::chrono::nanoseconds(dl - now));
                g_hwr_dbg[HD_CPU_IDLE_MS] += (now_ns() - now) / 1e6;
            }
            continue;
        }
        next->state = TState::Running;
        current = next;
        if (next == self) return;
        g_hwr_dbg[HD_CTX_SWITCHES]++;   // a hand-off between host threads (diagnostics)
        next->cv.notify_one();
        break;
    }
    if (self->state == TState::Dead) return;
    self->cv.wait(lk, [&] { return current == self; });
}

void Kernel::block_current(std::unique_lock<std::mutex> &lk) { switch_away(lk, current); }

void Kernel::yield_current(std::unique_lock<std::mutex> &lk) {
    Thread *self = current;
    make_ready(self);
    switch_away(lk, self);
}

void Kernel::maybe_preempt(std::unique_lock<std::mutex> &lk) {
    Thread *self = current;
    for (Thread *t : ready) {
        if (t->prio < self->prio) {
            make_ready(self, true);
            switch_away(lk, self);
            return;
        }
    }
}

static void thread_host_main(Thread *t) {
    {
        std::unique_lock<std::mutex> lk(g_k.mtx);
        t->cv.wait(lk, [&] { return g_k.current == t; });
    }
    host_fpscr(t->cpu.fpscr);
    try {
        disp_run(t->cpu, t->entry);
    } catch (ThreadExit &) {
    }
}

std::shared_ptr<Thread> Kernel::create_thread(u32 entry, u32 arg, u32 stack_top, s32 prio, s32 proc, const char *name) {
    auto t = std::make_shared<Thread>();
    t->id = next_tid++;
    t->name = name;
    t->entry = entry; t->arg = arg; t->stack_top = stack_top; t->prio = prio; t->proc_id = proc;
    t->tls = tls_next; tls_next += 0x200;
    if (!mem_is_mapped(t->tls)) mem_map(t->tls & ~0xFFFu, 0x1000, MEM_RW, "tls");
    memset(gp(t->tls), 0, 0x200);
    t->cpu.r[0] = arg;
    t->cpu.r[13] = stack_top & ~7u;
    t->cpu.r[15] = entry & ~1u;
    t->cpu.thumb = entry & 1;
    t->cpu.tpidruro = t->tls;
    t->cpu.fpscr = 0x03C00000;
    t->cpu.thread = t.get();
    threads.push_back(t);
    make_ready(t.get());
    std::thread(thread_host_main, t.get()).detach();
    INFO("[kern] thread %u '%s' entry=%08x arg=%08x sp=%08x prio=%d core=%d tls=%08x", t->id, name, entry, arg, stack_top, prio, proc, t->tls);
    return t;
}

void Kernel::start_main(u32 entry, u32 stack_top, s32 prio) {
    std::unique_lock<std::mutex> lk(mtx);
    auto t = create_thread(entry, 0, stack_top, prio, 0, "main");
    ready.erase(std::remove(ready.begin(), ready.end(), t.get()), ready.end());
    t->state = TState::Running;
    current = t.get();
    t->cv.notify_one();
    extern void kernel_start_timer();
    kernel_start_timer();
}

// ------------------------------------------------------------------- SVCs
static u64 s64ns(u32 lo, u32 hi) { return ((u64)hi << 32) | lo; }
static u64 deadline_from(s64 ns) {
    if (ns < 0) return ~0ull;
    return now_ns() + (u64)ns;
}

struct Linear { u32 next = LINEAR_BASE; std::map<u32, u32> blocks; } g_linear;
u64 g_mem_used = 0;

static u32 linear_alloc(u32 size) {
    size = (size + 0xFFF) & ~0xFFFu;
    // first fit over gaps
    u32 a = LINEAR_BASE;
    for (auto &b : g_linear.blocks) {
        if (b.first - a >= size) break;
        a = b.first + b.second;
    }
    if (a + size > LINEAR_BASE + LINEAR_SIZE) return 0;
    g_linear.blocks[a] = size;
    mem_map(a, size, MEM_RW, "linear");
    memset(gp(a), 0, size);
    return a;
}

bool mem_query(u32 va, u32 &base, u32 &size, u32 &perm);

static void svc_wait_objs(std::unique_lock<std::mutex> &lk, Cpu &c, std::vector<KPtr> objs, bool wait_all, s64 timeout) {
    Thread *t = g_k.current;
    // immediate check
    if (wait_all) {
        bool ok = true;
        for (auto &o : objs) if (o->should_wait(t)) { ok = false; break; }
        if (ok) {
            for (auto &o : objs) o->acquire(t);
            c.r[0] = RES_OK; c.r[1] = 0;
            return;
        }
    } else {
        for (size_t i = 0; i < objs.size(); i++) {
            if (!objs[i]->should_wait(t)) {
                objs[i]->acquire(t);
                c.r[0] = RES_OK; c.r[1] = (u32)i;
                return;
            }
        }
    }
    if (timeout == 0) { c.r[0] = RES_TIMEOUT; c.r[1] = 0xFFFFFFFF; return; }
    t->wait_objs = objs;
    t->wait_all = wait_all;
    t->wait_index = -1;
    t->wake_ns = deadline_from(timeout);
    t->state = TState::WaitSync;
    for (auto &o : objs) o->waiters.push_back(t);
    g_k.block_current(lk);
    c.r[0] = t->wait_result;
    c.r[1] = t->wait_result == RES_OK ? (u32)t->wait_index : 0xFFFFFFFF;
}

struct ResLimit : KObject { ResLimit() : KObject(KType::ResLimit) {} };

static const char *svc_name(u32 i);

extern std::atomic<int> g_irq_pending;
std::atomic<u32> g_excl_epoch{0};
void kernel_poll(Cpu &c) {
    std::unique_lock<std::mutex> lk(g_k.mtx);
    excl_clear_all();   // interrupt / possible context switch: CLREX
    g_irq_pending = 0;
    g_k.process_time(now_ns());
    g_k.maybe_preempt(lk);
}

void interp_kick();
void kernel_start_timer() {
    std::thread([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
            g_irq_pending = 1;
            interp_kick();
        }
    }).detach();
}

void rt_svc(Cpu &c, u32 imm) {
    std::unique_lock<std::mutex> lk(g_k.mtx);
    excl_clear_all();   // exception entry; the kernel may switch threads: CLREX
    Thread *t = g_k.current;
    if (t->cpu.thread != t) fatal("svc from non-current thread");
    g_k.process_time(now_ns());
    if (g_svc_log) LOG("[svc] t%u %02x %s r0=%08x r1=%08x r2=%08x r3=%08x lr=%08x", t->id, imm, svc_name(imm), c.r[0], c.r[1], c.r[2], c.r[3], c.r[14]);
    switch (imm) {
    case 0x01: {  // ControlMemory
        u32 op = c.r[0], a0 = c.r[1], a1 = c.r[2], size = c.r[3], perm = c.r[4];
        u32 kind = op & 0xFF;
        bool linear = op & 0x10000;
        u32 out = 0; Result res = RES_OK;
        if (kind == 3) {
            if (linear) {
                out = linear_alloc(size);
                if (!out) res = 0xD86007F3;
            } else {
                mem_map(a0, size, MEM_RW, "heap");
                out = a0;
            }
            g_mem_used += size;
        } else if (kind == 1) {
            if (a0 >= LINEAR_BASE && a0 < LINEAR_BASE + LINEAR_SIZE) {
                auto it = g_linear.blocks.find(a0);
                if (it != g_linear.blocks.end()) {
                    if (it->second <= size) g_linear.blocks.erase(it);
                    else { u32 rest = it->second - ((size + 0xFFF) & ~0xFFFu); g_linear.blocks.erase(it); g_linear.blocks[a0 + ((size + 0xFFF) & ~0xFFFu)] = rest; }
                }
            }
            mem_unmap(a0, size);
            out = a0;
            g_mem_used -= std::min<u64>(g_mem_used, size);
        } else if (kind == 6) {
            mem_protect(a0, size, perm);
            out = a0;
        } else {
            LOG("[svc] ControlMemory op=%x a0=%08x a1=%08x size=%x unsupported", op, a0, a1, size);
            res = 0xE0E01BF4;
        }
        INFO("[svc] ControlMemory op=%05x a0=%08x a1=%08x size=%x perm=%x -> %08x res=%08x", op, a0, a1, size, perm, out, res);
        c.r[0] = res; c.r[1] = out;
        break;
    }
    case 0x02: {  // QueryMemory
        u32 base, size, perm;
        bool m = mem_query(c.r[2], base, size, perm);
        c.r[0] = 0; c.r[1] = base; c.r[2] = size; c.r[3] = perm; c.r[4] = m ? 5 : 0; c.r[5] = 0;
        break;
    }
    case 0x03:
        LOG("[kern] ExitProcess");
        fflush(stderr);
        plat_exit(0);
    case 0x04: case 0x06: // GetProcessAffinityMask / GetProcessIdealProcessor
        if (imm == 0x04) wr8(c.r[0], 1);
        c.r[0] = 0; c.r[1] = 0; break;
    case 0x08: {  // CreateThread
        s32 prio = (s32)c.r[0];
        u32 entry = c.r[1], arg = c.r[2], sp = c.r[3];
        s32 proc = (s32)c.r[4];
        auto nt = g_k.create_thread(entry, arg, sp, prio, proc, "guest");
        c.r[0] = 0; c.r[1] = g_k.new_handle(nt);
        g_k.maybe_preempt(lk);
        break;
    }
    case 0x09: {  // ExitThread
        INFO("[kern] thread %u exits", t->id);
        t->state = TState::Dead;
        t->wake_waiters();
        g_k.switch_away(lk, t);
        lk.unlock();
#ifdef __EMSCRIPTEN__
        pthread_exit(nullptr);   // unwinds the worker without C++ exception support
#else
        throw ThreadExit();
#endif
    }
    case 0x0A: {  // SleepThread
        s64 ns = (s64)s64ns(c.r[0], c.r[1]);
        if (ns <= 0) { g_k.yield_current(lk); break; }
        t->state = TState::WaitSleep;
        t->wake_ns = now_ns() + ns;
        g_k.block_current(lk);
        break;
    }
    case 0x0B: {  // GetThreadPriority
        auto th = g_k.get_as<Thread>(c.r[1], KType::Thread);
        c.r[0] = th ? 0 : RES_INVALID_HANDLE; c.r[1] = th ? th->prio : 0; break;
    }
    case 0x0C: {  // SetThreadPriority
        auto th = g_k.get_as<Thread>(c.r[0], KType::Thread);
        if (th) th->prio = (s32)c.r[1];
        c.r[0] = th ? 0 : RES_INVALID_HANDLE;
        g_k.maybe_preempt(lk);
        break;
    }
    case 0x0D: case 0x0E: case 0x0F: case 0x10: c.r[0] = 0; break;
    case 0x11: c.r[0] = 0; break;  // GetCurrentProcessorNumber
    case 0x13: {  // CreateMutex
        auto m = std::make_shared<Mutex>();
        if (c.r[1]) m->acquire(t);
        c.r[0] = 0; c.r[1] = g_k.new_handle(m); break;
    }
    case 0x14: {  // ReleaseMutex
        auto m = g_k.get_as<Mutex>(c.r[0], KType::Mutex);
        if (!m) { c.r[0] = RES_INVALID_HANDLE; break; }
        if (m->owner != t) { c.r[0] = 0xD8E0041F; break; }
        if (--m->count == 0) { m->owner = nullptr; m->wake_waiters(); }
        c.r[0] = 0;
        g_k.maybe_preempt(lk);
        break;
    }
    case 0x15: {  // CreateSemaphore
        auto s = std::make_shared<Semaphore>((s32)c.r[1], (s32)c.r[2]);
        c.r[0] = 0; c.r[1] = g_k.new_handle(s); break;
    }
    case 0x16: {  // ReleaseSemaphore
        auto s = g_k.get_as<Semaphore>(c.r[1], KType::Semaphore);
        if (!s) { c.r[0] = RES_INVALID_HANDLE; break; }
        u32 prev = s->count;
        s->count += (s32)c.r[2];
        s->wake_waiters();
        c.r[0] = 0; c.r[1] = prev;
        g_k.maybe_preempt(lk);
        break;
    }
    case 0x17: {  // CreateEvent
        auto e = std::make_shared<Event>(c.r[1]);
        c.r[0] = 0; c.r[1] = g_k.new_handle(e); break;
    }
    case 0x18: {  // SignalEvent
        auto e = g_k.get_as<Event>(c.r[0], KType::Event);
        if (!e) { c.r[0] = RES_INVALID_HANDLE; break; }
        e->signal(); c.r[0] = 0;
        g_k.maybe_preempt(lk);
        break;
    }
    case 0x19: {  // ClearEvent
        auto e = g_k.get_as<Event>(c.r[0], KType::Event);
        if (e) e->clear();
        c.r[0] = e ? 0 : RES_INVALID_HANDLE; break;
    }
    case 0x1A: {  // CreateTimer
        auto tm = std::make_shared<Timer>(c.r[1]);
        g_k.timers.push_back(tm);
        c.r[0] = 0; c.r[1] = g_k.new_handle(tm); break;
    }
    case 0x1B: {  // SetTimer(h, initial r2:r3, interval r1:r4)
        auto tm = g_k.get_as<Timer>(c.r[0], KType::Timer);
        if (!tm) { c.r[0] = RES_INVALID_HANDLE; break; }
        s64 initial = (s64)s64ns(c.r[2], c.r[3]), interval = (s64)s64ns(c.r[1], c.r[4]);
        tm->next_ns = now_ns() + (initial > 0 ? initial : 0);
        tm->interval_ns = interval > 0 ? interval : 0;
        c.r[0] = 0; break;
    }
    case 0x1C: {  // CancelTimer
        auto tm = g_k.get_as<Timer>(c.r[0], KType::Timer);
        if (tm) tm->next_ns = ~0ull;
        c.r[0] = 0; break;
    }
    case 0x1D: {  // ClearTimer
        auto tm = g_k.get_as<Timer>(c.r[0], KType::Timer);
        if (tm) tm->signaled = false;
        c.r[0] = 0; break;
    }
    case 0x1E: {  // CreateMemoryBlock(r0=other perm, r1=addr, r2=size, r3=my perm)
        auto sm = std::make_shared<SharedMem>();
        sm->addr = c.r[1]; sm->size = c.r[2]; sm->my_perm = c.r[3]; sm->other_perm = c.r[0];
        if (sm->addr) sm->mapped_at.push_back(sm->addr);
        c.r[0] = 0; c.r[1] = g_k.new_handle(sm); break;
    }
    case 0x1F: {  // MapMemoryBlock(h, addr, myperm, otherperm)
        auto sm = g_k.get_as<SharedMem>(c.r[0], KType::SharedMem);
        if (!sm) { c.r[0] = RES_INVALID_HANDLE; break; }
        c.r[0] = shm_map(sm.get(), c.r[1]);
        break;
    }
    case 0x20: c.r[0] = 0; break;  // UnmapMemoryBlock
    case 0x21: {  // CreateAddressArbiter
        auto a = std::make_shared<KObject>(KType::Arbiter);
        c.r[0] = 0; c.r[1] = g_k.new_handle(a); break;
    }
    case 0x22: {  // ArbitrateAddress(h, addr, type, value, timeout r4:r5)
        u32 addr = c.r[1], type = c.r[2];
        s32 value = (s32)c.r[3];
        s64 timeout = (s64)s64ns(c.r[4], c.r[5]);
        c.r[0] = 0;
        if (type == 0) {  // signal
            std::vector<Thread *> ws;
            for (Thread *w : g_k.arb_waiters) if (w->arb_addr == addr) ws.push_back(w);
            std::stable_sort(ws.begin(), ws.end(), [](Thread *a, Thread *b) { return a->prio < b->prio; });
            s32 n = value < 0 ? (s32)ws.size() : std::min<s32>(value, (s32)ws.size());
            for (s32 i = 0; i < n; i++) g_k.wake_thread(ws[i], RES_OK, -1);
            g_k.maybe_preempt(lk);
            break;
        }
        s32 cur = (s32)rd32(addr);
        bool wait = cur < value;
        if (wait && (type == 2 || type == 4)) wr32(addr, (u32)(cur - 1));
        if (!wait) break;
        t->arb_addr = addr;
        t->state = TState::WaitArb;
        t->wake_ns = (type == 3 || type == 4) ? deadline_from(timeout) : ~0ull;
        g_k.arb_waiters.push_back(t);
        g_k.block_current(lk);
        c.r[0] = t->wait_result;
        break;
    }
    case 0x23: {  // CloseHandle
        g_k.handles.erase(c.r[0]);
        c.r[0] = 0; break;
    }
    case 0x24: {  // WaitSynchronization1(h, timeout r2:r3)
        KPtr o = g_k.get(c.r[0]);
        if (!o || !o->waitable()) { LOG("[svc] WaitSync1 bad handle %08x", c.r[0]); c.r[0] = RES_INVALID_HANDLE; break; }
        svc_wait_objs(lk, c, {o}, false, (s64)s64ns(c.r[2], c.r[3]));
        break;
    }
    case 0x25: {  // WaitSynchronizationN(r1 handles, r2 count, r3 all, timeout r0:r4)
        u32 hp = c.r[1], n = c.r[2];
        bool all = c.r[3] != 0;
        s64 timeout = (s64)s64ns(c.r[0], c.r[4]);
        std::vector<KPtr> objs;
        bool bad = false;
        for (u32 i = 0; i < n; i++) {
            KPtr o = g_k.get(rd32(hp + i * 4));
            if (!o || !o->waitable()) { LOG("[svc] WaitSyncN bad handle %08x", rd32(hp + i * 4)); bad = true; break; }
            objs.push_back(o);
        }
        if (bad) { c.r[0] = RES_INVALID_HANDLE; break; }
        if (n == 0) {  // pure sleep
            if (timeout) { t->state = TState::WaitSleep; t->wake_ns = deadline_from(timeout); g_k.block_current(lk); }
            c.r[0] = RES_TIMEOUT; break;
        }
        svc_wait_objs(lk, c, objs, all, timeout);
        break;
    }
    case 0x27: {  // DuplicateHandle
        KPtr o = g_k.get(c.r[1]);
        if (!o) { c.r[0] = RES_INVALID_HANDLE; break; }
        c.r[0] = 0; c.r[1] = g_k.new_handle(o); break;
    }
    case 0x28: {  // GetSystemTick
        u64 tk = now_ticks();
        c.r[0] = (u32)tk; c.r[1] = (u32)(tk >> 32); break;
    }
    case 0x2A: {  // GetSystemInfo(type r1, param r2:r3)
        u64 v = 0;
        if (c.r[1] == 0) v = g_mem_used;
        else if (c.r[1] == 26) v = 5;
        c.r[0] = 0; c.r[1] = (u32)v; c.r[2] = (u32)(v >> 32); break;
    }
    case 0x2B: {  // GetProcessInfo(h r1, type r2)
        u64 v = 0;
        switch (c.r[2]) {
        case 0: case 2: v = g_mem_used; break;
        case 20: v = FCRAM_PA - LINEAR_BASE; break;
        default: LOG("[svc] GetProcessInfo type %u", c.r[2]);
        }
        c.r[0] = 0; c.r[1] = (u32)v; c.r[2] = (u32)(v >> 32); break;
    }
    case 0x2D: {  // ConnectToPort(r1 name)
        std::string nm((const char *)gp(c.r[1]));
        u32 h = 0;
        Result r = svc_connect_port(nm, h);
        c.r[0] = r; c.r[1] = h; break;
    }
    case 0x32: {  // SendSyncRequest
        KPtr o = g_k.get(c.r[0]);
        if (!o || o->type != KType::Session) { LOG("[svc] SendSyncRequest bad handle %08x", c.r[0]); c.r[0] = RES_INVALID_HANDLE; break; }
        c.r[0] = session_request(std::static_pointer_cast<Session>(o), t, lk);
        g_k.maybe_preempt(lk);
        break;
    }
    case 0x35: c.r[0] = 0; c.r[1] = 0x28; break;   // GetProcessId
    case 0x37: {  // GetThreadId
        auto th = g_k.get_as<Thread>(c.r[1], KType::Thread);
        c.r[0] = 0; c.r[1] = th ? th->id : t->id; break;
    }
    case 0x38: {  // GetResourceLimit
        c.r[0] = 0; c.r[1] = g_k.new_handle(std::make_shared<ResLimit>()); break;
    }
    case 0x39: case 0x3A: {  // GetResourceLimit{Limit,Current}Values(out, h, names, count)
        u32 out = c.r[0], names = c.r[2], n = c.r[3];
        for (u32 i = 0; i < n; i++) {
            u32 nm = rd32(names + i * 4);
            u64 v = 0;
            if (imm == 0x39) {
                static const u64 lim[] = {0x18, 0x04000000, 32, 32, 32, 8, 8, 16, 2, 0};
                v = nm < 10 ? lim[nm] : 0;
            } else {
                if (nm == 1) v = g_mem_used;
                else if (nm == 2) v = g_k.threads.size();
            }
            wr64(out + i * 8, v);
        }
        c.r[0] = 0; break;
    }
    case 0x3C: {  // Break
        LOG("[kern] svcBreak reason=%u by thread %u", c.r[0], t->id);
        dump_cpu(c);
        fatal("guest break");
    }
    case 0x3D: {  // OutputDebugString
        std::string s((const char *)gp(c.r[0]), c.r[1]);
        LOG("[guest] %s", s.c_str());
        c.r[0] = 0; break;
    }
    default:
        LOG("[svc] unimplemented svc 0x%02x (%s) t%u pc=%08x", imm, svc_name(imm), t->id, c.r[15]);
        dump_cpu(c);
        c.r[0] = 0xF8C007F4;
        break;
    }
}

bool mem_query_impl(u32 va, u32 &base, u32 &size, u32 &perm);
bool mem_query(u32 va, u32 &base, u32 &size, u32 &perm) { return mem_query_impl(va, base, size, perm); }

static const char *svc_name(u32 i) {
    static const char *n[0x80] = {};
    static bool init = false;
    if (!init) {
        n[0x01] = "ControlMemory"; n[0x02] = "QueryMemory"; n[0x03] = "ExitProcess"; n[0x08] = "CreateThread";
        n[0x09] = "ExitThread"; n[0x0A] = "SleepThread"; n[0x0B] = "GetThreadPriority"; n[0x0C] = "SetThreadPriority";
        n[0x13] = "CreateMutex"; n[0x14] = "ReleaseMutex"; n[0x15] = "CreateSemaphore"; n[0x16] = "ReleaseSemaphore";
        n[0x17] = "CreateEvent"; n[0x18] = "SignalEvent"; n[0x19] = "ClearEvent"; n[0x1A] = "CreateTimer";
        n[0x1B] = "SetTimer"; n[0x1C] = "CancelTimer"; n[0x1D] = "ClearTimer"; n[0x1E] = "CreateMemoryBlock";
        n[0x1F] = "MapMemoryBlock"; n[0x20] = "UnmapMemoryBlock"; n[0x21] = "CreateAddressArbiter"; n[0x22] = "ArbitrateAddress";
        n[0x23] = "CloseHandle"; n[0x24] = "WaitSynchronization1"; n[0x25] = "WaitSynchronizationN"; n[0x27] = "DuplicateHandle";
        n[0x28] = "GetSystemTick"; n[0x2A] = "GetSystemInfo"; n[0x2B] = "GetProcessInfo"; n[0x2D] = "ConnectToPort";
        n[0x32] = "SendSyncRequest"; n[0x35] = "GetProcessId"; n[0x37] = "GetThreadId"; n[0x38] = "GetResourceLimit";
        n[0x39] = "GetResourceLimitLimitValues"; n[0x3A] = "GetResourceLimitCurrentValues"; n[0x3C] = "Break"; n[0x3D] = "OutputDebugString";
        init = true;
    }
    return (i < 0x80 && n[i]) ? n[i] : "?";
}

static const char *tstate(TState s) {
    switch (s) { case TState::Ready: return "ready"; case TState::Running: return "RUN"; case TState::WaitSync: return "waitsync";
    case TState::WaitSleep: return "sleep"; case TState::WaitArb: return "waitarb"; case TState::WaitIpc: return "waitipc"; default: return "dead"; }
}
void kernel_dump_threads() {
    std::unique_lock<std::mutex> lk(g_k.mtx, std::try_to_lock);
    extern u32 display_frame_count();
    fprintf(stderr, "[status] frames=%u mem_used=%llx\n", display_frame_count(), (unsigned long long)g_mem_used);
    for (auto &t : g_k.threads) {
        fprintf(stderr, "  t%u prio=%d %-8s pc=%08x lr=%08x sp=%08x", t->id, t->prio, tstate(t->state), t->cpu.r[15], t->cpu.r[14], t->cpu.r[13]);
        if (t->state == TState::WaitSync) {
            fprintf(stderr, " objs:");
            for (auto &o : t->wait_objs) fprintf(stderr, " %d", (int)o->type);
        }
        if (t->state == TState::WaitArb) fprintf(stderr, " arb=%08x val=%08x", t->arb_addr, rd32(t->arb_addr));
        fprintf(stderr, "\n");
    }
    if (const char *pk = getenv("R3DS_PEEK")) {
        std::string s = pk; size_t p = 0;
        while (p < s.size()) {
            u32 a = (u32)strtoul(s.c_str() + p, nullptr, 16);
            fprintf(stderr, "  [peek %08x]", a);
            for (int i = 0; i < 8; i++) fprintf(stderr, " %08x", mem_is_mapped(a + i * 4) ? rd32(a + i * 4) : 0xDEADDEAD);
            fprintf(stderr, "\n");
            size_t q = s.find(',', p); if (q == std::string::npos) break; p = q + 1;
        }
    }
}

u32 thread_id_of(Thread *t) { return t ? t->id : 0; }
#include <unordered_map>
extern std::unordered_map<u32, u64> g_prof[64];
void prof_dump(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    for (int t = 0; t < 64; t++)
        for (auto &kv : g_prof[t]) fprintf(f, "%d %08x %llu\n", t, kv.first, (unsigned long long)kv.second);
    fclose(f);
}
