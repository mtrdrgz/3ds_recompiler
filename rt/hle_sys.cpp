// cfg, hid, ptm, ndm, nim, err:f, ac, frd, dsp and friends.
#include "services.h"
#include "input.h"
#include <map>

// ------------------------------------------------------------------ cfg
static std::vector<u8> cfg_block(u32 id, u32 size) {
    std::vector<u8> b(size, 0);
    auto put16 = [&](size_t o, u16 v) { if (o + 2 <= b.size()) memcpy(&b[o], &v, 2); };
    switch (id) {
    case 0x000A0002: b[0] = 1; break;                    // language: English
    case 0x000B0000: if (size >= 4) { b[2] = 110; b[3] = 0; } break;  // country: UK
    case 0x00070001: b[0] = 1; break;                    // sound: stereo
    case 0x000A0000: {                                   // username
        const char *n = "Player";
        for (int i = 0; n[i]; i++) put16(i * 2, (u8)n[i]);
        break;
    }
    case 0x000A0001: if (size >= 2) { b[0] = 1; b[1] = 1; } break;  // birthday
    case 0x00050005: {                                   // stereo camera calib
        float v[8] = {62.0f, 289.0f, 76.8f, 46.08f, 10.0f, 5.0f, 55.58f, 21.57f};
        memcpy(b.data(), v, std::min<size_t>(size, sizeof v));
        break;
    }
    case 0x000F0004: b[0] = 0; break;                    // system model: CTR
    default: break;
    }
    return b;
}

struct CfgService : Service {
    CfgService(std::string n) : Service(n) {}
    void handle(Ipc &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001: case 0x0401: case 0x0801: {  // GetConfigInfoBlk2(size, id, desc, ptr)
            u32 size = ipc.p(1), id = ipc.p(2), dst = ipc.p(4);
            auto b = cfg_block(id, size);
            memcpy(gp(dst), b.data(), size);
            INFO("[cfg] GetConfigInfoBlk id=%08x size=%x", id, size);
            ipc.reply(1, 2); ipc.w(2, (size << 4) | 0xC); ipc.w(3, dst);
            break;
        }
        case 0x0002: ipc.reply(2, 0); ipc.w(2, 2); break;             // GetRegion: EUR
        case 0x0003: ipc.reply(3, 0); ipc.w(2, 0x12345678); ipc.w(3, 0x9abcdef0); break;
        case 0x0004: ipc.reply(2, 0); ipc.w(2, 0); break;             // GetRegionCanadaUSA
        case 0x0005: ipc.reply(2, 0); ipc.w(2, 0); break;             // GetSystemModel
        case 0x0006: ipc.reply(2, 0); ipc.w(2, 1); break;             // GetModelNintendo2DS (1 = not 2DS)
        case 0x0009: ipc.reply(2, 0); ipc.w(2, 'G' | ('B' << 8)); break;
        case 0x000A: ipc.reply(2, 0); ipc.w(2, 110); break;
        default: unknown(ipc);
        }
    }
};

// ------------------------------------------------------------------ hid
static std::shared_ptr<SharedMem> g_hid_shm;
static std::shared_ptr<Event> g_hid_ev[5];
static u32 g_hid_index = 0, g_touch_index = 0, g_hid_prev = 0;

void hid_update() {
    if (!g_hid_shm) return;
    excl_clear_all();   // shared memory written from a host thread
    u32 base = shm_addr(g_hid_shm);
    InputState in = input_get();
    u64 tick = now_ticks();
    // pad
    u32 cur = in.buttons;
    g_hid_index = (g_hid_index + 1) & 7;
    wr64(base + 0x08, rd64(base + 0x00));
    wr64(base + 0x00, tick);
    wr32(base + 0x10, g_hid_index);
    wr32(base + 0x1C, cur);
    wr16(base + 0x20, (u16)in.cx); wr16(base + 0x22, (u16)in.cy);
    u32 e = base + 0x28 + g_hid_index * 0x10;
    wr32(e + 0, cur);
    wr32(e + 4, cur & ~g_hid_prev);
    wr32(e + 8, ~cur & g_hid_prev);
    wr16(e + 12, (u16)in.cx); wr16(e + 14, (u16)in.cy);
    g_hid_prev = cur;
    // touch
    u32 tb = base + 0xA8;
    g_touch_index = (g_touch_index + 1) & 7;
    wr64(tb + 0x08, rd64(tb + 0x00));
    wr64(tb + 0x00, tick);
    wr32(tb + 0x10, g_touch_index);
    wr16(tb + 0x18, in.tx); wr16(tb + 0x1A, in.ty); wr32(tb + 0x1C, in.touch);
    u32 te = tb + 0x20 + g_touch_index * 8;
    wr16(te + 0, in.tx); wr16(te + 2, in.ty); wr32(te + 4, in.touch);
    // accelerometer (flat, 1g down)
    u32 ab = base + 0x108;
    wr64(ab + 0x08, rd64(ab)); wr64(ab, tick);
    u32 ai = (rd32(ab + 0x10) + 1) & 7;
    wr32(ab + 0x10, ai);
    u32 ae = ab + 0x28 + ai * 6;
    wr16(ae, 0); wr16(ae + 2, (u16)-512); wr16(ae + 4, 0);
    for (int i = 0; i < 2; i++) if (g_hid_ev[i]) g_hid_ev[i]->signal();
}

struct HidService : Service {
    HidService(std::string n) : Service(n) {}
    void handle(Ipc &ipc) override {
        switch (ipc.cmd()) {
        case 0x000A: {  // GetIPCHandles
            if (!g_hid_shm) {
                g_hid_shm = shm_create(0x2B0, "hid shm");
                for (int i = 0; i < 5; i++) { g_hid_ev[i] = std::make_shared<Event>(0); g_hid_ev[i]->name = "hid ev" + std::to_string(i); }
                g_k.add_periodic(16666667, hid_update);
            }
            ipc.reply(1, 7); ipc.w(2, 0x14000000);
            ipc.w(3, g_k.new_handle(g_hid_shm));
            for (int i = 0; i < 5; i++) ipc.w(4 + i, g_k.new_handle(g_hid_ev[i]));
            break;
        }
        case 0x0011: case 0x0012: case 0x0013: case 0x0014: ipc.reply(1, 0); break;
        case 0x0015: { float f = 14.375f; u32 v; memcpy(&v, &f, 4); ipc.reply(2, 0); ipc.w(2, v); break; }
        case 0x0016: { ipc.reply(1, 0); for (int i = 0; i < 6; i++) ipc.w(2 + i, 0); ipc.reply(7, 0); break; }
        case 0x0017: ipc.reply(2, 0); ipc.w(2, 0x3F); break;
        default: unknown(ipc);
        }
    }
};

// ------------------------------------------------------------------ ptm
struct PtmService : Service {
    PtmService(std::string n) : Service(n) {}
    void handle(Ipc &ipc) override {
        switch (ipc.cmd()) {
        case 0x0005: ipc.reply(2, 0); ipc.w(2, 1); break;   // GetAdapterState
        case 0x0006: ipc.reply(2, 0); ipc.w(2, 1); break;   // GetShellState (open)
        case 0x0007: ipc.reply(2, 0); ipc.w(2, 5); break;   // GetBatteryLevel
        case 0x0008: ipc.reply(2, 0); ipc.w(2, 0); break;   // GetBatteryChargeState
        case 0x000B: ipc.reply(3, 0); ipc.w(2, 0); ipc.w(3, 0); break;   // GetStepHistory etc
        case 0x000C: ipc.reply(2, 0); ipc.w(2, 0); break;   // GetTotalStepCount
        default: unknown(ipc);
        }
    }
};

// ------------------------------------------------------------------ dsp
// Minimal DSP: component loads succeed; the audio pipe answers the init
// handshake with structure addresses in DSP RAM. Audio frames are not
// generated yet (silent).
static std::shared_ptr<Event> g_dsp_sem_ev, g_dsp_irq_ev[3][4];
static std::vector<u16> g_dsp_pipe2;
static std::shared_ptr<Event> g_dsp_audio_ev;
static bool g_dsp_audio_on = false;
static int g_dsp_frame = 0;

void dsp_audio_frame();
void dsp_audio_dump();
static void dsp_tick() {
    excl_clear_all();
    if (!g_dsp_audio_on) return;
    g_dsp_frame++;
    dsp_audio_frame();
    if (getenv("R3DS_DSP_DUMP") && g_dsp_frame % 2000 == 0) dsp_audio_dump();
    if (g_dsp_irq_ev[2][2]) g_dsp_irq_ev[2][2]->signal();
    if (g_dsp_audio_ev && g_dsp_audio_ev != g_dsp_irq_ev[2][2]) g_dsp_audio_ev->signal();
}

struct DspService : Service {
    DspService() : Service("dsp::DSP") {}
    void handle(Ipc &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001: ipc.reply(2, 0); ipc.w(2, 1); break;      // RecvData
        case 0x0002: ipc.reply(2, 0); ipc.w(2, 1); break;      // RecvDataIsReady
        case 0x0007: ipc.reply(1, 0); break;                   // SetSemaphore
        case 0x000C: ipc.reply(2, 0); ipc.w(2, 0x1FF40000 + (ipc.p(1) << 1)); break;
        case 0x000D: {  // WriteProcessPipe(channel, size, desc, ptr)
            u32 ch = ipc.p(1), sz = ipc.p(2), src = ipc.p(4);
            INFO("[dsp] WriteProcessPipe ch=%u size=%u first=%04x", ch, sz, sz >= 2 ? rd16(src) : 0);
            if (ch == 2 && sz >= 4) {
                u32 state = rd16(src);
                if (state == 0) {  // initialize: reply with 15 struct addresses per region
                    g_dsp_pipe2.clear();
                    g_dsp_pipe2.push_back(15);
                    // DSP word addresses of the shared structures in region 0
                    static const u16 addrs[15] = {0xBFFF, 0x9E92, 0x8680, 0xA792, 0x9430, 0x8400, 0x8540, 0x9492,
                                                  0x8710, 0x8410, 0xA912, 0xAA12, 0xAAD2, 0xAC52, 0xAC5C};
                    for (u16 a : addrs) g_dsp_pipe2.push_back(a);
                    g_dsp_audio_on = true;
                } else if (state == 1) {
                    g_dsp_audio_on = false;
                }
            }
            ipc.reply(1, 0);
            break;
        }
        case 0x0010: {  // ReadPipeIfPossible(channel, peer, size)
            u32 ch = ipc.p(1), sz = ipc.p(3) & 0xFFFF;
            u32 dst = ipc.static_buf_addr(0);
            u32 n = 0;
            if (ch == 2) {
                n = std::min<u32>(sz, (u32)g_dsp_pipe2.size() * 2);
                for (u32 i = 0; i < n / 2; i++) wr16(dst + i * 2, g_dsp_pipe2[i]);
                g_dsp_pipe2.erase(g_dsp_pipe2.begin(), g_dsp_pipe2.begin() + n / 2);
            }
            ipc.reply(2, 2); ipc.w(2, n); ipc.static_desc(3, 0, n, dst);
            break;
        }
        case 0x000E: {  // ReadPipe? (same shape)
            ipc.reply(1, 2); ipc.static_desc(2, 0, 0, ipc.static_buf_addr(0)); break;
        }
        case 0x0011: {  // LoadComponent(size, pmask, dmask, desc, ptr)
            u32 csize = ipc.p(1), cptr = ipc.p(5);
            ipc.reply(2, 2); ipc.w(2, 1); ipc.w(3, (csize << 4) | 0xA); ipc.w(4, cptr);
            if (!mem_is_mapped(0x1FF00000)) mem_map(0x1FF00000, 0x80000, MEM_RW, "dsp ram");
            static bool tick = false;
            if (!tick) { tick = true; g_k.add_periodic(4880000, dsp_tick); }
            INFO("[dsp] LoadComponent size=%x", csize);
            break;
        }
        case 0x0012: ipc.reply(1, 0); break;
        case 0x0013: case 0x0014: ipc.reply(1, 0); break;
        case 0x0015: {  // RegisterInterruptEvents(interrupt, channel, desc, handle)
            u32 irq = ipc.p(1), ch = ipc.p(2), h = ipc.p(4);
            auto e = g_k.get_as<Event>(h, KType::Event);
            if (irq < 3 && ch < 4) g_dsp_irq_ev[irq][ch] = e;
            if (irq == 2 && ch == 2) g_dsp_audio_ev = e;
            INFO("[dsp] RegisterInterruptEvents irq=%u ch=%u h=%08x", irq, ch, h);
            ipc.reply(1, 0);
            break;
        }
        case 0x0016: {
            if (!g_dsp_sem_ev) { g_dsp_sem_ev = std::make_shared<Event>(0); g_dsp_sem_ev->name = "dsp sema"; }
            ipc.reply(1, 2); ipc.w(2, 0); ipc.w(3, g_k.new_handle(g_dsp_sem_ev)); break;
        }
        case 0x0017: ipc.reply(1, 0); break;
        case 0x001F: ipc.reply(2, 0); ipc.w(2, 0); break;
        case 0x0021: case 0x0022: ipc.reply(1, 0); break;
        default: unknown(ipc);
        }
    }
};

// ------------------------------------------------------------------ cecd
// StreetPass. No CEC system module runs, so no hits ever arrive: the two
// events are real (games WaitSync on them) but simply stay unsignaled.
struct CecdService : Service {
    std::shared_ptr<Event> info_ev, state_ev;
    CecdService() : Service("cecd:u") {}
    void handle(Ipc &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001: case 0x0005: case 0x0008:          // OpenRawFile / WriteRawFile / Delete
        case 0x000B: case 0x000C:                       // Start / Stop
            ipc.reply(1, 0); break;
        case 0x000E:                                    // GetCecdState
            ipc.reply(2, 0); ipc.w(2, 2); break;        // CEC_STATE_ABBREV_INACTIVE
        case 0x000F: case 0x0010: {                     // GetCecInfoEventHandle / GetChangeStateEventHandle
            auto &e = ipc.cmd() == 0x000F ? info_ev : state_ev;
            if (!e) { e = std::make_shared<Event>(1); e->name = ipc.cmd() == 0x000F ? "cecd info" : "cecd state"; }
            ipc.reply(1, 2); ipc.w(2, 0x04000000); ipc.w(3, g_k.new_handle(e));
            break;
        }
        default: unknown(ipc);
        }
    }
};

// ------------------------------------------------------------------ ndm
// Network daemon manager: there are no daemons — commands are accepted and
// state queries describe an offline console.
struct NdmService : Service {
    u32 excl = 0, suspended = 0, defaults = 0xF;
    NdmService() : Service("ndm:u") {}
    void handle(Ipc &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001: excl = ipc.p(1); ipc.reply(1, 0); break;                 // EnterExclusiveState
        case 0x0002: excl = 0; ipc.reply(1, 0); break;                        // LeaveExclusiveState
        case 0x0003: ipc.reply(2, 0); ipc.w(2, excl); break;                  // QueryExclusiveMode
        case 0x0004: case 0x0005: ipc.reply(1, 0); break;                     // LockState / UnlockState
        case 0x0006: suspended |= ipc.p(1); ipc.reply(1, 0); break;           // SuspendDaemons(mask)
        case 0x0007: suspended &= ~ipc.p(1); ipc.reply(1, 0); break;          // ResumeDaemons(mask)
        case 0x0008: case 0x0009: ipc.reply(1, 0); break;                     // SuspendScheduler / ResumeScheduler
        case 0x000A: case 0x000B: ipc.reply(2, 0); ipc.w(2, 0); break;        // GetCurrentState / GetTargetState: STATE_INITIAL
        case 0x000D: {  // QueryStatus(daemon) -> DaemonStatus
            u32 d = ipc.p(1);
            ipc.reply(2, 0); ipc.w(2, (suspended & (1u << (d & 7))) ? 3u : 1u); break;
        }
        case 0x000E: case 0x000F: ipc.reply(2, 0); ipc.w(2, 0); break;        // daemon/scheduler disable counts
        case 0x0010: case 0x0012: ipc.reply(1, 0); break;                     // SetScanInterval / SetRetryInterval
        case 0x0011: case 0x0013: ipc.reply(2, 0); ipc.w(2, 0); break;        // GetScanInterval / GetRetryInterval
        case 0x0014: defaults = ipc.p(1); ipc.reply(1, 0); break;             // OverrideDefaultDaemons
        case 0x0015: defaults = 0xF; ipc.reply(1, 0); break;                  // ResetDefaultDaemons
        case 0x0016: ipc.reply(2, 0); ipc.w(2, defaults); break;              // GetDefaultDaemons
        case 0x0017: ipc.reply(1, 0); break;                                  // ClearHalfAwakeMacFilter
        default: unknown(ipc);
        }
    }
};

// ------------------------------------------------------------------ frd
// Friends: an offline console — never logged in, empty friend list, no
// notifications. Event handles the client passes are real but never signaled.
struct FrdService : Service {
    std::shared_ptr<Event> notif_ev;
    bool logged_in = false;
    FrdService() : Service("frd:u") {}
    void handle(Ipc &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001: ipc.reply(2, 0); ipc.w(2, logged_in); break;             // HasLoggedIn
        case 0x0002: ipc.reply(2, 0); ipc.w(2, 0); break;                     // IsOnline
        case 0x0003: logged_in = true;                                        // Login(desc, event)
            if (auto e = g_k.get_as<Event>(ipc.p(2), KType::Event)) { notif_ev = e; e->name = "frd notif"; }
            ipc.reply(1, 0); break;
        case 0x0004: logged_in = false; ipc.reply(1, 0); break;               // Logout
        case 0x0005: {  // GetMyFriendKey {principal_id, pad, local_friend_code}
            ipc.reply(5, 0);
            for (int i = 2; i <= 5; i++) ipc.w(i, 0); break;
        }
        case 0x0006: ipc.reply(2, 0); ipc.w(2, 0); break;                     // GetMyPreference
        case 0x0007: {  // GetMyProfile: FriendProfile 0x14 bytes
            ipc.reply(6, 0);
            for (int i = 2; i <= 6; i++) ipc.w(i, 0); break;
        }
        case 0x0008: {  // GetMyPresence: FriendPresence 0x1C bytes
            ipc.reply(8, 0);
            for (int i = 2; i <= 8; i++) ipc.w(i, 0); break;
        }
        case 0x0009: {  // GetMyScreenName: u16[11], empty
            ipc.reply(7, 0);
            for (int i = 2; i <= 7; i++) ipc.w(i, 0); break;
        }
        case 0x000A: {  // GetMyMii: zeroed mii data
            ipc.reply(24, 0);
            for (int i = 2; i <= 24; i++) ipc.w(i, 0); break;
        }
        case 0x000B: ipc.reply(2, 0); ipc.w(2, 0); break;                     // GetMyLocalAccountId
        case 0x000C: case 0x000D:                                             // GetMyPlayingGame / GetMyFavoriteGame (u64)
            ipc.reply(3, 0); ipc.w(2, 0); ipc.w(3, 0); break;
        case 0x000E: ipc.reply(2, 0); ipc.w(2, 0); break;                     // GetMyNcPrincipalId
        case 0x000F: {  // GetMyComment: u16[33], empty
            ipc.reply(18, 0);
            for (int i = 2; i <= 18; i++) ipc.w(i, 0); break;
        }
        case 0x0011: {  // GetFriendKeyList(offset, count) -> keys in static buf 0; empty list
            u32 dst = ipc.static_buf_addr(0);
            ipc.reply(2, 2); ipc.w(2, 0); ipc.static_desc(3, 0, 0, dst);
            break;
        }
        case 0x0012: {  // GetFriendPresence(count, desc0, keybuf): 0x30B placeholder entries (found=0) in static buf 0
            u32 dst = ipc.static_buf_addr(0);
            u32 n = dst ? std::min<u32>(ipc.p(1), ipc.static_buf_size(0) / 0x30) : 0;
            if (n) memset(gp(dst), 0, n * 0x30);
            ipc.reply(1, 2); ipc.static_desc(2, 0, n * 0x30, dst);
            break;
        }
        case 0x0013: {  // GetFriendScreenName(count...): names in static buf 0, charsets in buf 1; empty placeholders
            u32 n = ipc.p(3);
            u32 a0 = ipc.static_buf_addr(0), a1 = ipc.static_buf_addr(1);
            u32 nb = a0 ? std::min<u32>(n * 0x16, ipc.static_buf_size(0)) : 0;
            u32 cb = a1 ? std::min<u32>(n, ipc.static_buf_size(1)) : 0;
            if (nb) memset(gp(a0), 0, nb);
            if (cb) memset(gp(a1), 0, cb);
            ipc.reply(1, 4);
            ipc.static_desc(2, 0, nb, a0);
            ipc.static_desc(4, 1, cb, a1);
            break;
        }
        case 0x0014: {  // GetFriendMii(count, desc0, keybuf, descW, outbuf): zeroed placeholder MiiData
            u32 out = ipc.p(5), n = out ? std::min<u32>(ipc.p(1) * 0x60, ipc.p(4) >> 4) : 0;
            if (n) memset(gp(out), 0, n);
            ipc.reply(1, 2); ipc.w(2, ipc.p(4)); ipc.w(3, out);
            break;
        }
        case 0x0015: case 0x0016: case 0x0017: case 0x0018: case 0x0019:
        case 0x001A: case 0x001C: {  // GetFriendProfile/Relationship/AttributeFlags/PlayingGame/FavoriteGame/Info/UnscrambleLocalFriendCode
            u32 dst = ipc.static_buf_addr(0), sz = ipc.static_buf_size(0);   // zeroed placeholder entries
            if (dst && sz) memset(gp(dst), 0, sz);
            ipc.reply(1, 2); ipc.static_desc(2, 0, sz, dst);
            break;
        }
        case 0x001B: ipc.reply(2, 0); ipc.w(2, 0); break;                     // IsIncludedInFriendList
        case 0x001D: case 0x001E: ipc.reply(1, 0); break;                     // UpdateGameModeDescription / UpdateMyPresence
        case 0x0020:                                                        // AttachToEventNotification(desc, event)
            if (auto e = g_k.get_as<Event>(ipc.p(2), KType::Event)) { notif_ev = e; e->name = "frd notif"; }
            ipc.reply(1, 0); break;
        case 0x0021: ipc.reply(1, 0); break;                                  // SetNotificationMask
        case 0x0023: ipc.reply(2, 0); ipc.w(2, 0); break;                     // GetLastResponseResult
        case 0x0025: ipc.reply(2, 0); ipc.w(2, 0); break;                     // FriendCodeToPrincipalId
        case 0x0026: ipc.reply(2, 0); ipc.w(2, 0); break;                     // IsValidFriendCode
        case 0x0027: ipc.reply(2, 0); ipc.w(2, 0); break;                     // ResultToErrorCode
        case 0x002D: ipc.reply(3, 0); ipc.w(2, 0); ipc.w(3, 0); break;        // GetNatProperties
        case 0x002E: ipc.reply(3, 0); ipc.w(2, 0); ipc.w(3, 0); break;        // GetServerTimeDifference
        case 0x002F: ipc.reply(1, 0); break;                                  // AllowHalfAwake
        case 0x0030: ipc.reply(2, 0); ipc.w(2, 0); break;                     // GetServerTypes
        case 0x0032: ipc.reply(1, 0); break;                                  // SetClientSdkVersion
        default: unknown(ipc);
        }
    }
};

// ------------------------------------------------------------------ boss
// SpotPass: the session opens, no storage is configured and no tasks ever
// run — an arrival event the client registers is real but never signaled.
struct BossService : Service {
    std::shared_ptr<Event> arrival_ev;
    BossService() : Service("boss:U") {}
    void handle(Ipc &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001: ipc.reply(1, 0); break;                                  // InitializeSession(u64 programID)
        case 0x0002: case 0x0003: ipc.reply(1, 0); break;                     // SetStorageInfo / UnregisterStorage
        case 0x0005: case 0x0006: ipc.reply(1, 0); break;                     // RegisterPrivateRootCa / ClientCert
        case 0x0007: ipc.reply(2, 0); ipc.w(2, 0); break;                     // GetNewArrivalFlag
        case 0x0008: {  // RegisterNewArrivalEvent(desc, event): client hands us the event
            arrival_ev = g_k.get_as<Event>(ipc.p(2), KType::Event);
            if (arrival_ev) arrival_ev->name = "boss arrival";
            ipc.reply(1, 0, arrival_ev ? RES_OK : RES_INVALID_HANDLE); break;
        }
        case 0x0009: ipc.reply(1, 0); break;                                  // SetOptoutFlag
        case 0x000A: ipc.reply(2, 0); ipc.w(2, 0); break;                     // GetOptoutFlag
        case 0x000E: ipc.reply(2, 0); ipc.w(2, 0); break;                     // GetTaskIdList -> empty
        default: unknown(ipc);
        }
    }
};

// ------------------------------------------------------------------ misc
struct ErrF : Service {
    ErrF() : Service("err:f") {}
    void handle(Ipc &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001:  // Throw(ERRF_FatalErrInfo): the game is dead from here on
            LOG("[err:f] fatal error: type=%u result=%08x pc=%08x proc=%08x",
                ipc.p(1) & 0xFF, ipc.p(2), ipc.p(3), ipc.p(4));
            fatal("guest fatal error");
        case 0x0002: {  // SetUserString(size, desc, ptr)
            u32 s = ipc.p(3), n = std::min<u32>(ipc.p(1), 64);
            std::string txt;
            for (u32 i = 0; i < n && mem_is_mapped(s + i); i++) { char c = (char)rd8(s + i); if (!c) break; txt += c; }
            LOG("[err:f] user string '%s'", txt.c_str());
            ipc.reply(1, 0);
            break;
        }
        default: unknown(ipc);
        }
    }
};

struct GenericOk : Service {
    std::map<u32, std::vector<u32>> canned;
    GenericOk(std::string n) : Service(n) {}
    void handle(Ipc &ipc) override {
        auto it = canned.find(ipc.cmd());
        if (it != canned.end()) {
            ipc.reply(1 + (u32)it->second.size(), 0);
            for (size_t i = 0; i < it->second.size(); i++) ipc.w(2 + (int)i, it->second[i]);
            return;
        }
        unknown(ipc, 2);
    }
};

void register_apt(); void register_fs(); void register_gsp(); void register_y2r();

void services_register_all() {
    register_apt();
    register_fs();
    register_gsp();
    register_y2r();
    services_register("cfg:u", [] { return std::make_shared<CfgService>("cfg:u"); });
    services_register("hid:USER", [] { return std::make_shared<HidService>("hid:USER"); });
    services_register("ptm:u", [] { return std::make_shared<PtmService>("ptm:u"); });
    services_register("dsp::DSP", [] { return std::make_shared<DspService>(); });
    services_register("err:f", [] { return std::make_shared<ErrF>(); });
    services_register("cecd:u", [] { return std::make_shared<CecdService>(); });
    services_register("ndm:u", [] { return std::make_shared<NdmService>(); });
    services_register("frd:u", [] { return std::make_shared<FrdService>(); });
    services_register("boss:U", [] { return std::make_shared<BossService>(); });
    const char *stubs[] = {"ac:u", "cam:u", "dlp:FKCL", "dlp:SRVR", "http:C", "mic:u",
                           "news:u", "nwm::UDS", "soc:U", "ssl:C", "ldr:ro", "ir:USER", "nim:aoc",
                           "am:app", "pxi:dev"};
    for (const char *s : stubs) {
        std::string n = s;
        services_register(n, [n] { return std::make_shared<GenericOk>(n); });
    }
}
