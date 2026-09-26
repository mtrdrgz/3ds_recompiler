// APT: applet manager. Only the application side of the lifecycle: the
// app is told to wake up once and is never suspended.
#include "services.h"
#include <deque>

struct AptParam { u32 sender, command; std::vector<u8> buf; u32 handle; };
static std::shared_ptr<Mutex> g_apt_lock;
static std::shared_ptr<Event> g_apt_notif, g_apt_param;
static std::deque<AptParam> g_apt_params;
static std::shared_ptr<SharedMem> g_apt_font;
static u32 g_notification = 0;

static void post_param(u32 sender, u32 cmd) {
    g_apt_params.push_back({sender, cmd, {}, 0});
    g_apt_param->signal();
}

struct AptService : Service {
    AptService(std::string n) : Service(n) {}
    void handle(Ipc &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001:  // GetLockHandle
            if (!g_apt_lock) g_apt_lock = std::make_shared<Mutex>();
            ipc.reply(3, 2); ipc.w(2, 0); ipc.w(3, 0); ipc.w(4, 0x04000000); ipc.w(5, g_k.new_handle(g_apt_lock));
            break;
        case 0x0002:  // Initialize(appid, attr)
            if (!g_apt_notif) g_apt_notif = std::make_shared<Event>(0);
            if (!g_apt_param) g_apt_param = std::make_shared<Event>(0);
            ipc.reply(1, 3); ipc.w(2, 0x04000000);
            ipc.w(3, g_k.new_handle(g_apt_notif)); ipc.w(4, g_k.new_handle(g_apt_param));
            break;
        case 0x0003:  // Enable
            post_param(0, 1);  // APTCMD_WAKEUP
            ipc.reply(1, 0);
            break;
        case 0x0006:  // GetAppletInfo(appid)
            ipc.reply(7, 0); ipc.w(2, 0); ipc.w(3, 0x00040000); ipc.w(4, 0); ipc.w(5, 1); ipc.w(6, 1); ipc.w(7, 0);
            break;
        case 0x0009:  // IsRegistered
            ipc.reply(2, 0); ipc.w(2, 1); break;
        case 0x000B:  // InquireNotification
            ipc.reply(2, 0); ipc.w(2, g_notification); g_notification = 0; break;
        case 0x000C:  // SendParameter
            ipc.reply(1, 0); break;
        case 0x000D: case 0x000E: {  // ReceiveParameter / GlanceParameter(appid, size)
            u32 cap = ipc.p(2);
            AptParam pr{0, 0, {}, 0};
            bool have = !g_apt_params.empty();
            if (have) { pr = g_apt_params.front(); if (ipc.cmd() == 0x0D) g_apt_params.pop_front(); }
            ipc.reply(4, 4, have ? RES_OK : 0xC8A0CFEF);
            ipc.w(2, pr.sender); ipc.w(3, pr.command); ipc.w(4, (u32)std::min<size_t>(pr.buf.size(), cap));
            ipc.w(5, 0); ipc.w(6, pr.handle);
            ipc.static_desc(7, 0, (u32)std::min<size_t>(pr.buf.size(), cap), ipc.static_buf_addr(0));
            INFO("[apt] %sParameter -> cmd=%u", ipc.cmd() == 0x0D ? "Receive" : "Glance", pr.command);
            break;
        }
        case 0x000F:  // CancelParameter
            ipc.reply(2, 0); ipc.w(2, 1); break;
        case 0x0043:  // NotifyToWait
        case 0x004F:  // SetAppCpuTimeLimit
            ipc.reply(1, 0); break;
        case 0x0050:  // GetAppCpuTimeLimit
            ipc.reply(2, 0); ipc.w(2, 30); break;
        case 0x0044: {  // GetSharedFont
            if (!g_apt_font) g_apt_font = shm_create(0x332000, "sharedfont");
            ipc.reply(2, 2); ipc.w(2, shm_addr(g_apt_font)); ipc.w(3, 0x04000000); ipc.w(4, g_k.new_handle(g_apt_font));
            LOG("[apt] GetSharedFont: no system font data available (block left empty)");
            break;
        }
        case 0x004B:  // AppletUtility(id, insize, outsize, ...)
            ipc.reply(2, 2); ipc.w(2, 0); ipc.static_desc(3, 0, 0, ipc.static_buf_addr(0)); break;
        case 0x0055:  // GetStartupArgument? (actually 0x51 on some fw)
        case 0x0051: {  // GetStartupArgument(size, type)
            u32 sz = ipc.p(1);
            u32 dst = ipc.static_buf_addr(0);
            if (dst && sz) memset(gp(dst), 0, std::min<u32>(sz, 0x1000));
            ipc.reply(2, 2); ipc.w(2, 0); ipc.static_desc(3, 0, sz, dst); break;
        }
        case 0x0101: case 0x0102:  // CheckNew3DSApp / CheckNew3DS
            ipc.reply(2, 0); ipc.w(2, 0); break;
        default: unknown(ipc);
        }
    }
};

void register_apt() {
    for (const char *n : {"APT:U", "APT:A", "APT:S"}) {
        std::string s = n;
        services_register(s, [s] { return std::make_shared<AptService>(s); });
    }
}
