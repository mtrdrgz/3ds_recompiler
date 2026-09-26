#include <memory>
#include <vector>
// gsp::Gpu — GX command queue, interrupt relay, framebuffer swaps.
#include "services.h"
#include "gpu.h"
#include "hwr.h"
#include "pica_snap.h"
#include <deque>
#include <algorithm>
#include <thread>
#include <condition_variable>
void pica_texcache_invalidate(u32 va, u32 size);

// data: for DMA from CPU-written memory, the source bytes captured when the
// game submitted the command (see gx_submit)
struct GxCmd { u32 w[8]; std::shared_ptr<std::vector<u8>> data; std::shared_ptr<GpuSnap> snap; u64 t = 0; };
enum { GX_INVALIDATE = 0xF0, GX_PRESENT = 0xF1 };
static void gx_submit(const GxCmd &c);
static int g_gx_sync = -1;   // R3DS_SYNC_GPU: run GX commands on the submitting thread
static bool gx_defer_vblank();
static std::shared_ptr<SharedMem> g_gsp_shm;
static std::shared_ptr<Event> g_gsp_ev;
static u32 g_gsp_regs[0x10000 / 4];      // GPU external regs, offset 0x400000
static FbInfo g_fb[2];

enum { IRQ_PSC0 = 0, IRQ_PSC1, IRQ_VBLANK_TOP, IRQ_VBLANK_BOT, IRQ_PPF, IRQ_P3D, IRQ_DMA };

void gsp_irq(u32 id) {
    excl_clear_all();   // writes the interrupt relay queue in shared memory
    if (!g_gsp_shm) return;
    u32 q = shm_addr(g_gsp_shm) + 0 * 0x40;
    u32 hdr = rd32(q);
    u32 idx = hdr & 0xFF, cnt = (hdr >> 8) & 0xFF;
    if (cnt >= 0x34) { wr32(q + 4, rd32(q + 4) + 1); }
    else {
        u32 slot = (idx + cnt) % 0x34;
        wr8(q + 0xC + slot, (u8)id);
        cnt++;
        wr32(q, (hdr & 0xFFFF00FF) | (cnt << 8));
    }
    if (g_gsp_ev) g_gsp_ev->signal();
}

// the game presented a top-screen frame (diagnostics: rate and the longest gap)
static void note_game_frame() {
    static u64 last = 0;
    u64 now = now_ns();
    g_hwr_dbg[HD_GAME_FRAMES]++;
    if (last) g_hwr_dbg[HD_GAME_GAP_MAX_MS] = std::max(g_hwr_dbg[HD_GAME_GAP_MAX_MS], (now - last) / 1e6);
    last = now;
}
static void read_fb_info(int screen) {
    if (!g_gsp_shm) return;
    u32 base = shm_addr(g_gsp_shm) + 0x200 + screen * 0x40;
    u32 h = rd32(base);
    u8 idx = h & 0xFF, dirty = (h >> 8) & 0xFF;
    if (!dirty) return;
    // FramebufferInfo: active_fb, address_left, address_right, stride, format,
    // shown_fb, unk. active_fb picks which of the LCD's two register sets the
    // addresses go to (hardware double buffering), not an eye: the picture is
    // always address_left. address_right is the 3D right eye and is 0 on the
    // bottom screen, so taking it made every other bottom frame vanish.
    if (screen == 0) note_game_frame();
    u32 info = base + 4 + (idx & 1) * 0x1C;
    u32 addr = rd32(info + 4);
    g_fb[screen] = {addr, rd32(info + 12), rd32(info + 16) & 7, true};
    wr32(base, h & 0xFFFF00FF);
}

#include "pica.h"
extern u64 g_frag_count;
static void vblank() {
    if (g_gsp_shm) {   // commands left in the shared-memory GX queue without a trigger?
        static u32 stale = 0;
        u32 hdr = rd32(shm_addr(g_gsp_shm) + 0x800);
        if ((hdr >> 8) & 0xFF) { if (++stale == 3) LOG("[gsp] GX queue holds %u untriggered command(s) (hdr %08x)", (hdr >> 8) & 0xFF, hdr); }
        else stale = 0;
    }
    static u32 vb = 0;
    if (++vb % 60 == 0) { GxCmd c{}; c.w[0] = GX_INVALIDATE; c.w[1] = 0; c.w[2] = 0xFFFFFFFFu; gx_submit(c); }
    read_fb_info(0);
    read_fb_info(1);
    static u32 n = 0;
    static u64 last_t = 0;
    if (++n % 120 == 0) {
        u64 t = now_ns();
        double fps = last_t ? 120.0 / ((t - last_t) / 1e9) : 0;
        last_t = t;
        INFO("[gsp] %.1f fps ", fps);
    }
    if (n % 120 == 0)
        INFO("[gsp] vblank %u: top fb=%08x fmt=%u stride=%u valid=%d | bot fb=%08x fmt=%u | draws=%llu frags=%llu",
             n, g_fb[0].addr_va, g_fb[0].format, g_fb[0].stride, g_fb[0].valid, g_fb[1].addr_va, g_fb[1].format,
             (unsigned long long)g_pica->draws, (unsigned long long)g_frag_count);
    // What the screens show at this vblank is what the GPU work submitted
    // before it produced. With the GPU on its own thread that is known only
    // when the thread gets here in its queue, so the frame is captured there
    // (GX_PRESENT) and stamped with this vblank's time: the display then
    // paces frames by vblank time, independent of how far behind the GPU
    // thread runs or how unevenly (hwr_gpu.cpp).
    if (g_gx_sync == 0) {
        GxCmd c{};
        c.w[0] = GX_PRESENT;
        for (int s = 0; s < 2; s++) { c.w[1 + s * 3] = g_fb[s].addr_va; c.w[2 + s * 3] = g_fb[s].stride; c.w[3 + s * 3] = g_fb[s].format | (g_fb[s].valid ? 0x100 : 0); }
        c.t = now_ns();
        gx_submit(c);
    } else display_present(g_fb[0], g_fb[1]);
    if (!gx_defer_vblank()) {
        gsp_irq(IRQ_VBLANK_TOP);
        gsp_irq(IRQ_VBLANK_BOT);
    }
}

// GX commands run on a dedicated GPU thread, like the real (asynchronous)
// GPU: TriggerCmdReqQueue only drains the shared-memory queue; completion
// interrupts are raised when the GPU thread finishes each command.
static std::deque<GxCmd> g_gx_q;
static std::mutex g_gx_m;
static std::condition_variable g_gx_cv, g_gx_idle_cv;
static bool g_gx_busy = false;
// work queued or running on the GPU thread: display transfers (about one per
// screen per frame) and command lists. See gx_throttle.
static u32 g_gx_pend_xfer = 0, g_gx_pend_lists = 0;
// VBlank waits for the GPU. The game starts each frame on the vblank
// interrupt and then rewrites vertex data it computes on the CPU (skinned and
// animated meshes), trusting that the real GPU finished the previous frame
// long before. When the GPU thread is still working at vblank time the
// interrupt is held back and raised when it goes idle, so the game never
// rewrites buffers still being read (which broke geometry and textures more
// the longer the player moved) while the GPU keeps running in parallel with
// the game building the current frame. R3DS_SYNC_GPU=1 is the exact, slower
// alternative. R3DS_VBLANK_WAIT=0 turns this off.
static bool g_vb_deferred = false;
static u64 g_vb_deferred_ns = 0;
static void exec_gx(const GxCmd &cmd);

static void irq_locked(u32 id) {
    std::lock_guard<std::mutex> lk(g_k.mtx);
    gsp_irq(id);
}

static void gx_thread_main() {
    for (;;) {
        GxCmd c;
        {
            std::unique_lock<std::mutex> lk(g_gx_m);
            g_gx_cv.wait(lk, [] { return !g_gx_q.empty(); });
            c = g_gx_q.front(); g_gx_q.pop_front();
            g_gx_busy = true;
        }
        const u64 t0 = now_ns();
        exec_gx(c);
        g_hwr_dbg[HD_GPU_BUSY_MS] += (now_ns() - t0) / 1e6;
        bool vb = false;
        {
            std::lock_guard<std::mutex> lk(g_gx_m);
            g_gx_busy = false;
            u32 id = c.w[0] & 0xFF;
            if (id == 1) g_gx_pend_lists--;
            else if (id == 3) g_gx_pend_xfer--;
            if (g_gx_q.empty()) { g_gx_idle_cv.notify_all(); vb = g_vb_deferred; g_vb_deferred = false; }
        }
        if (vb) {   // the vblank held back for this work (see gx_defer_vblank)
            g_hwr_dbg[HD_VBLANK_WAIT_MS] += (now_ns() - g_vb_deferred_ns) / 1e6;
            std::lock_guard<std::mutex> lk(g_k.mtx);
            gsp_irq(IRQ_VBLANK_TOP);
            gsp_irq(IRQ_VBLANK_BOT);
        }
    }
}

// called at vblank with the kernel lock held: true = the GPU thread is busy,
// it raises the vblank interrupts itself when idle
static bool gx_defer_vblank() {
    static int on = getenv("R3DS_VBLANK_WAIT") ? atoi(getenv("R3DS_VBLANK_WAIT")) : 0;
    if (g_gx_sync != 0 || !on) return false;
    std::lock_guard<std::mutex> lk(g_gx_m);
    if (g_gx_q.empty() && !g_gx_busy) return false;
    u64 now = now_ns();
    if (!g_vb_deferred) { g_vb_deferred = true; g_vb_deferred_ns = now; return true; }
    // already one held back: it covers this one too (the game sees a slower
    // display), unless the GPU thread is stuck
    if (now - g_vb_deferred_ns < 200000000ull) return true;
    g_vb_deferred = false;
    return false;
}

static int g_gx_log = -1;   // R3DS_GX_LOG=1: log every GX command at submit and at execution
static void gx_submit(const GxCmd &c) {
    if (g_gx_log < 0) g_gx_log = getenv("R3DS_GX_LOG") ? 1 : 0;
    if (g_gx_log && (c.w[0] & 0xFF) < GX_INVALIDATE) LOG("[gxsub] %u %08x %08x %08x", c.w[0] & 0xFF, c.w[1], c.w[2], c.w[3]);
    if (g_gx_sync < 0) {
        g_gx_sync = getenv("R3DS_SYNC_GPU") ? 1 : 0;
        if (!g_gx_sync) std::thread(gx_thread_main).detach();
    }
    if (g_gx_sync) { exec_gx(c); return; }
    GxCmd q = c;
    // a command list: copy what it reads from main memory now (pica_snap.cpp)
    static int snap_on = getenv("R3DS_GX_SNAP") ? atoi(getenv("R3DS_GX_SNAP")) : 1;
    if ((c.w[0] & 0xFF) == 1 && snap_on && c.w[2]) {
        std::vector<std::pair<u32, u32>> skip;   // main memory that queued GPU work still writes
        {
            std::lock_guard<std::mutex> lk(g_gx_m);
            for (const GxCmd &o : g_gx_q) {
                u32 id = o.w[0] & 0xFF, dst = 0, len = 0;
                if (id == 0) { dst = o.w[2]; len = o.w[3]; }
                else if (id == 2) { if (o.w[1]) skip.push_back({o.w[1], o.w[3]}); if (o.w[4]) skip.push_back({o.w[4], o.w[6]}); }
                else if (id == 3) { dst = o.w[2]; len = (o.w[4] & 0xFFFF) * (o.w[4] >> 16) * 4; }
                else if (id == 4) { dst = o.w[2]; len = o.w[3] * 2; }
                if (len) skip.push_back({dst, dst + len});
            }
        }
        const u64 t0 = now_ns();
        q.snap = gpu_capture(c.w[1], c.w[2], skip);
        g_hwr_dbg[HD_SNAP_MS] += (now_ns() - t0) / 1e6;
        if (q.snap) g_hwr_dbg[HD_SNAP_BYTES] += (double)q.snap->data.size();
    }
    // The GPU thread may run behind the game. Real hardware finishes a DMA
    // long before the game touches its staging buffer again, so read the
    // source now, in program order, unless it is VRAM (GPU-produced data,
    // which must be read after the GPU work queued before it).
    u32 src = c.w[1], size = c.w[3];
    if ((c.w[0] & 0xFF) == 0 && size && size <= 0x1000000 && !(src >= VRAM_VA && src < VRAM_VA + VRAM_SIZE) && mem_is_mapped(src)) {
        q.data = std::make_shared<std::vector<u8>>(gp(src), gp(src) + size);
    }
    std::lock_guard<std::mutex> lk(g_gx_m);
    u32 id = c.w[0] & 0xFF;
    if (id == 1) g_gx_pend_lists++;
    else if (id == 3) g_gx_pend_xfer++;
    g_gx_q.push_back(std::move(q));
    g_hwr_dbg[HD_GX_CMDS]++;
    if (g_gx_q.size() > g_hwr_dbg[HD_GX_QUEUE_MAX]) g_hwr_dbg[HD_GX_QUEUE_MAX] = (double)g_gx_q.size();
    g_gx_cv.notify_one();
}

static void process_gx_queue() {
    u32 q = shm_addr(g_gsp_shm) + 0x800;
    for (;;) {
        u32 hdr = rd32(q);
        u32 idx = hdr & 0xFF, num = (hdr >> 8) & 0xFF;
        if (!num) break;
        u32 c = q + 0x20 + idx * 0x20;
        GxCmd cmd;
        for (int i = 0; i < 8; i++) cmd.w[i] = rd32(c + i * 4);
        gx_submit(cmd);
        idx = (idx + 1) % 15;
        num--;
        wr32(q, (hdr & 0xFFFF0000) | idx | (num << 8));
    }
}

// Backpressure. The GPU thread can fall behind the game (heavy scenes: the
// vertex work is on the CPU). The game only double-buffers its command lists
// and vertex data, as the real GPU is never a frame late, so a GPU thread
// running further behind executes lists the game is already rewriting:
// geometry and textures break up more the longer the backlog, and the
// picture lags the game (rubber-banding). Hold the submitting thread here,
// like a stalled GPU would, until at most ~one frame of work is queued.
// R3DS_GX_AHEAD: display transfers allowed in flight (default 3; 0 = off).
static void gx_throttle(Thread *t, std::unique_lock<std::mutex> &lk) {
    static int ahead = getenv("R3DS_GX_AHEAD") ? atoi(getenv("R3DS_GX_AHEAD")) : 3;
    if (g_gx_sync || ahead <= 0) return;
    const u64 t0 = now_ns();
    for (;;) {
        {
            std::lock_guard<std::mutex> g(g_gx_m);
            if (g_gx_pend_xfer <= (u32)ahead && g_gx_pend_lists <= (u32)ahead * 4 && g_gx_q.size() <= 96) break;
        }
        if (now_ns() - t0 > 250000000ull) break;   // never hang the game on a stuck GPU thread
        t->state = TState::WaitSleep;
        t->wake_ns = now_ns() + 500000;
        g_k.block_current(lk);
    }
    const double held = (now_ns() - t0) / 1e6;
    g_hwr_dbg[HD_GX_WAIT_MS] += held;
    g_hwr_dbg[HD_GX_WAIT_MAX_MS] = std::max(g_hwr_dbg[HD_GX_WAIT_MAX_MS], held);
}

u32 display_frame_count();
static void exec_gx(const GxCmd &cmd) {
    if (g_gx_log > 0 && (cmd.w[0] & 0xFF) != GX_INVALIDATE) LOG("[gxexe] %u %08x %08x %08x", cmd.w[0] & 0xFF, cmd.w[1], cmd.w[2], cmd.w[3]);
    static u32 wlo = getenv("R3DS_GX_WATCH") ? (u32)strtoul(getenv("R3DS_GX_WATCH"), 0, 16) : 0;
    if (wlo) {
        u32 id = cmd.w[0] & 0xFF, dst = 0, len = 0;
        if (id == 0) { dst = cmd.w[2]; len = cmd.w[3]; }
        else if (id == 2) { dst = cmd.w[1]; len = cmd.w[3] - cmd.w[1]; }
        else if (id == 3) { dst = cmd.w[2]; len = (cmd.w[4] & 0xFFFF) * (cmd.w[4] >> 16) * 4; }
        else if (id == 4) { dst = cmd.w[2]; len = cmd.w[3] * 2; }
        static int all = getenv("R3DS_GX_WATCH_ALL") != nullptr;
        if (len && ((dst <= wlo && wlo < dst + len) || (all && id == 0 && dst >= VRAM_VA)))
        {
            u32 h = 2166136261u;
            if (cmd.data) for (u8 b : *cmd.data) { h ^= b; h *= 16777619u; }
            LOG("[gxwatch] cmd %u src %08x dst %08x len %x snap %d hash %08x vblank %u", id, cmd.w[1], dst, len, cmd.data ? 1 : 0, h, display_frame_count());
        }
    }
    {
        u32 id = cmd.w[0] & 0xFF;
        u32 a1 = cmd.w[1], a2 = cmd.w[2], a3 = cmd.w[3], a4 = cmd.w[4], a5 = cmd.w[5], a6 = cmd.w[6], a7 = cmd.w[7];
        auto gsp_irq = [](u32 v) { if (g_gx_sync) ::gsp_irq(v); else irq_locked(v); };
        switch (id) {
        case 0: {  // RequestDma
            DBG("[gx] dma %08x -> %08x size %x", a1, a2, a3);
            if (cmd.data && mem_is_mapped(a2)) memcpy(gp(a2), cmd.data->data(), cmd.data->size());
            else if (mem_is_mapped(a1) && mem_is_mapped(a2)) memmove(gp(a2), gp(a1), a3);
            pica_texcache_invalidate(a2, a3);
            hwr_mem_written(a2, a3);
            gsp_irq(IRQ_DMA);
            break;
        }
        case 1:  // SubmitGpuCmdList
            DBG("[gx] cmdlist %08x size %x", a1, a2);
            g_gpu_snap = cmd.snap.get();   // read memory as it was at submit
            pica_run_cmdlist(a1, a2, true);
            g_gpu_snap = nullptr;
            gsp_irq(IRQ_P3D);
            break;
        case 2:  // MemoryFill
            DBG("[gx] fill %08x-%08x=%08x / %08x-%08x=%08x ctl %08x", a1, a3, a2, a4, a6, a5, a7);
            if (a1) { gpu_memory_fill(a1, a3, a2, a7 & 0xFFFF); pica_texcache_invalidate(a1, a3 - a1); gsp_irq(IRQ_PSC0); }
            if (a4) { gpu_memory_fill(a4, a6, a5, a7 >> 16); pica_texcache_invalidate(a4, a6 - a4); gsp_irq(IRQ_PSC1); }
            break;
        case 3:  // DisplayTransfer
            DBG("[gx] xfer %08x -> %08x in %08x out %08x flags %08x", a1, a2, a3, a4, a5);
            gpu_display_transfer(a1, a2, a3, a4, a5);
            pica_texcache_invalidate(a2, (a4 & 0xFFFF) * (a4 >> 16) * 4);
            gsp_irq(IRQ_PPF);
            break;
        case 4:  // TextureCopy
            DBG("[gx] texcopy %08x -> %08x size %x", a1, a2, a3);
            gpu_texture_copy(a1, a2, a3, a4, a5, a6);
            pica_texcache_invalidate(a2, a3 * 2);
            gsp_irq(IRQ_PPF);
            break;
        case 5: break;  // FlushCacheRegions
        case GX_INVALIDATE: pica_texcache_invalidate(a1, a2); break;
        case GX_PRESENT: {
            FbInfo fb[2];
            for (int s = 0; s < 2; s++) fb[s] = {cmd.w[1 + s * 3], cmd.w[2 + s * 3], cmd.w[3 + s * 3] & 0xFF, (cmd.w[3 + s * 3] & 0x100) != 0};
            hwr_publish();   // the frame's images reach the executor before the frame does
            display_present(fb[0], fb[1], cmd.t);
            break;
        }
        default: LOG("[gx] unknown command %u", id);
        }
    }
    hwr_publish();
}

struct GspService : Service {
    GspService() : Service("gsp::Gpu") {}
    void handle(Ipc &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001: case 0x0002: {  // WriteHWRegs(reg, size, desc, ptr) / WithMask
            u32 reg = ipc.p(1), size = ipc.p(2);
            u32 src = ipc.cmd() == 1 ? ipc.p(4) : ipc.p(4);
            for (u32 i = 0; i < size; i += 4) {
                u32 off = reg + i - 0x400000;
                if (off < sizeof g_gsp_regs) g_gsp_regs[off / 4] = rd32(src + i);
            }
            ipc.reply(1, 0);
            break;
        }
        case 0x0004: {  // ReadHWRegs(reg, size)
            u32 reg = ipc.p(1), size = ipc.p(2), dst = ipc.static_buf_addr(0);
            for (u32 i = 0; i < size; i += 4) {
                u32 off = reg + i - 0x400000;
                wr32(dst + i, off < sizeof g_gsp_regs ? g_gsp_regs[off / 4] : 0);
            }
            ipc.reply(1, 2); ipc.static_desc(2, 0, size, dst);
            break;
        }
        case 0x0005: {  // SetBufferSwap(screen, info[7])
            u32 s = ipc.p(1) & 1;
            // info = active_fb, address_left, address_right, stride, format, ...:
            // the picture is address_left whichever register set is active
            g_fb[s] = {ipc.p(3), ipc.p(5), ipc.p(6) & 7, true};
            if (s == 0) note_game_frame();
            ipc.reply(1, 0);
            break;
        }
        case 0x0008: case 0x0009: {  // Flush/InvalidateDataCache(addr, size): ordered with GPU work
            GxCmd c{}; c.w[0] = GX_INVALIDATE; c.w[1] = ipc.p(1); c.w[2] = ipc.p(2);
            gx_submit(c);
            ipc.reply(1, 0); break;
        }
        case 0x000B: case 0x0010: case 0x0016: case 0x0017:
        case 0x0019: case 0x001A: case 0x001F: case 0x001E:
            ipc.reply(1, 0); break;
        case 0x000C:  // TriggerCmdReqQueue
            if (g_gsp_shm) process_gx_queue();
            ipc.reply(1, 0);
            gx_throttle(ipc.t, *ipc.lk);
            break;
        case 0x0013: {  // RegisterInterruptRelayQueue(flags, desc, event)
            g_gsp_ev = g_k.get_as<Event>(ipc.p(3), KType::Event);
            if (!g_gsp_shm) {
                g_gsp_shm = shm_create(0x1000, "gsp shm");
                // 60.00 Hz rather than the LCD's 59.83: frames then line up
                // with a 60 (or 120) Hz display instead of slipping one
                // every ~6 s (a visible hitch); the game runs 0.3% faster.
                // R3DS_VBLANK_HZ overrides.
                const double hz = getenv("R3DS_VBLANK_HZ") ? atof(getenv("R3DS_VBLANK_HZ")) : 60.0;
                g_k.add_periodic((u64)(1e9 / (hz > 1 ? hz : 60.0)), vblank);
            }
            ipc.reply(2, 2, 0x2A07);  // first registration
            ipc.w(2, 0);   // thread index
            ipc.w(3, 0x04000000); ipc.w(4, g_k.new_handle(g_gsp_shm));
            if (g_gsp_ev) g_gsp_ev->signal();
            break;
        }
        case 0x0014: ipc.reply(1, 0); break;
        case 0x0018: {  // ImportDisplayCaptureInfo
            ipc.reply(9, 0);
            for (int i = 0; i < 8; i++) ipc.w(2 + i, 0);
            break;
        }
        default: unknown(ipc);
        }
    }
};

void register_gsp() {
    services_register("gsp::Gpu", [] { return std::make_shared<GspService>(); });
    pica_init();
    display_init();
}
