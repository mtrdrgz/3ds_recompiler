#include "services.h"
#include <map>

int g_ipc_log = 0;
static std::map<std::string, ServiceFactory> g_factories;

void services_register(const std::string &name, ServiceFactory f) { g_factories[name] = std::move(f); }

#include <set>
void Service::unknown(Ipc &ipc, u32 normal) {
    static std::map<std::pair<std::string, u32>, int> seen;
    int &n = seen[{name, ipc.cmd()}];
    if (n++ < 3) LOG("[ipc] %s: unhandled cmd 0x%04x hdr=%08x (%08x %08x %08x %08x)", name.c_str(), ipc.cmd(), ipc.hdr(), ipc.p(1), ipc.p(2), ipc.p(3), ipc.p(4));
    ipc.reply(normal, 0);
    for (u32 i = 1; i < normal; i++) ipc.w(1 + i, 0);
}
void Service::handle(Ipc &ipc) { unknown(ipc); }

u32 new_session_handle(std::shared_ptr<Service> svc) {
    return g_k.new_handle(std::make_shared<Session>(std::move(svc)));
}

Result session_request(const std::shared_ptr<Session> &s, Thread *t, std::unique_lock<std::mutex> &lk) {
    Ipc ipc(t, t->tls + 0x80, &lk);
    if (g_ipc_log) LOG("[ipc] t%u -> %s cmd %08x (%08x %08x %08x)", t->id, s->svc->name.c_str(), ipc.hdr(), ipc.p(1), ipc.p(2), ipc.p(3));
    s->svc->handle(ipc);
    return RES_OK;
}

// ---------------------------------------------------------------- shared mem
static u32 g_shm_next = 0x10000000 - 0x800000;   // kernel-chosen mappings (below main stack region)
std::shared_ptr<SharedMem> shm_create(u32 size, const char *name) {
    auto sm = std::make_shared<SharedMem>();
    sm->size = (size + 0xFFF) & ~0xFFFu;
    sm->name = name;
    // kernel-owned block: give it a fixed home in guest space right away so
    // the service side can use it before the app maps it
    sm->addr = 0x1E800000 + 0;   // replaced below
    static u32 home = 0x1E800000;
    sm->addr = home; home += sm->size;
    mem_map(sm->addr, sm->size, MEM_RW, name);
    memset(gp(sm->addr), 0, sm->size);
    return sm;
}

u32 shm_addr(const std::shared_ptr<SharedMem> &sm) { return sm->addr; }

Result shm_map(SharedMem *sm, u32 addr) {
    // Flat memory can't alias; kernel blocks live at their home address and
    // an app-requested address is served by remapping the block there.
    if (!addr) { addr = g_shm_next; g_shm_next += sm->size; }
    if (addr != sm->addr) {
        if (sm->addr && (sm->addr >= 0x1E800000 && sm->addr < 0x1F000000)) {
            mem_map(addr, sm->size, MEM_RW, sm->name.c_str());
            memcpy(gp(addr), gp(sm->addr), sm->size);
            mem_unmap(sm->addr, sm->size);
            sm->addr = addr;
        } else if (!mem_is_mapped(addr)) {
            mem_map(addr, sm->size, MEM_RW, "shm");
        }
    }
    sm->mapped_at.push_back(addr);
    INFO("[kern] MapMemoryBlock %s size=%x at %08x", sm->name.c_str(), sm->size, addr);
    return RES_OK;
}

// ---------------------------------------------------------------------- srv
struct SrvService : Service {
    std::shared_ptr<Event> notif;
    SrvService() : Service("srv:") {}
    void handle(Ipc &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001:  // RegisterClient
            ipc.reply(1, 0); break;
        case 0x0002: {  // EnableNotification
            if (!notif) notif = std::make_shared<Event>(0);
            ipc.reply(1, 2); ipc.w(2, 0x04000000); ipc.w(3, g_k.new_handle(notif)); break;
        }
        case 0x0005: {  // GetServiceHandle(name[8], len, flags)
            char nm[9] = {};
            memcpy(nm, gp(ipc.buf + 4), 8);
            std::string n(nm, strnlen(nm, 8));
            auto it = g_factories.find(n);
            if (it == g_factories.end()) {
                LOG("[srv] GetServiceHandle('%s') -> not found", n.c_str());
                ipc.reply(1, 0, 0xD8E06406);
                break;
            }
            u32 h = new_session_handle(it->second());
            INFO("[srv] GetServiceHandle('%s') -> %08x", n.c_str(), h);
            ipc.reply(1, 2); ipc.w(2, 0x04000000); ipc.w(3, h);
            break;
        }
        case 0x0009: case 0x000A: case 0x000B:  // Subscribe/Unsubscribe/ReceiveNotification
            ipc.reply(2, 0); ipc.w(2, 0); break;
        default: unknown(ipc);
        }
    }
};

Result svc_connect_port(const std::string &name, u32 &handle) {
    if (name == "srv:") {
        handle = new_session_handle(std::make_shared<SrvService>());
        return RES_OK;
    }
    auto it = g_factories.find(name);
    if (it != g_factories.end()) {
        handle = new_session_handle(it->second());
        return RES_OK;
    }
    LOG("[kern] ConnectToPort('%s') not found", name.c_str());
    return 0xD88007FA;
}

void services_register_all();
void services_init() { services_register_all(); }
