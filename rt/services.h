// HLE services: IPC plumbing shared by all service implementations.
#pragma once
#include "kernel.h"
#include "mem.h"

struct Service;
struct Session : KObject {
    std::shared_ptr<Service> svc;
    Session(std::shared_ptr<Service> s) : KObject(KType::Session), svc(std::move(s)) {}
};

// Request context: wraps the calling thread's command buffer (TLS+0x80).
struct Ipc {
    Thread *t;
    u32 buf;
    std::unique_lock<std::mutex> *lk;
    u32 req[64];   // snapshot of the request: replies overwrite the buffer
    Ipc(Thread *t_, u32 b, std::unique_lock<std::mutex> *l) : t(t_), buf(b), lk(l) { memcpy(req, gp(b), sizeof req); }
    u32 hdr() const { return req[0]; }
    u32 cmd() const { return req[0] >> 16; }
    u32 p(int i) const { return req[i & 63]; }
    void w(int i, u32 v) const { wr32(buf + 4 * i, v); }
    void reply(u32 normal, u32 translate, Result res = RES_OK) const {
        w(0, (cmd() << 16) | (normal << 6) | translate);
        w(1, res);
    }
    // client's receive static buffer #id
    u32 static_buf_addr(int id) const { return rd32(t->tls + 0x180 + id * 8 + 4); }
    u32 static_buf_size(int id) const { return rd32(t->tls + 0x180 + id * 8) >> 14; }
    void static_desc(int slot, int id, u32 size, u32 addr) const {
        w(slot, (size << 14) | (id << 10) | 2);
        w(slot + 1, addr);
    }
};

struct Service {
    std::string name;
    Service(std::string n) : name(std::move(n)) {}
    virtual ~Service() {}
    virtual void handle(Ipc &ipc);          // default: log + OK
    void unknown(Ipc &ipc, u32 normal = 1);
};

using ServiceFactory = std::function<std::shared_ptr<Service>()>;
void services_register(const std::string &name, ServiceFactory f);
void services_init();
Result svc_connect_port(const std::string &name, u32 &handle);
Result session_request(const std::shared_ptr<Session> &s, Thread *t, std::unique_lock<std::mutex> &lk);
u32 new_session_handle(std::shared_ptr<Service> svc);
Result shm_map(SharedMem *sm, u32 addr);
std::shared_ptr<SharedMem> shm_create(u32 size, const char *name);
u32 shm_addr(const std::shared_ptr<SharedMem> &sm);   // guest address where mapped (0 if not)
extern int g_ipc_log;

std::string romfs_path();
