#include "kernel.h"
#include "mem.h"
#include "services.h"
#include <vector>
#include <ctime>
#include <thread>
#include <chrono>

void interp_init();
void disp_init();
#include "rom.h"
#ifdef R3DS_HAVE_LIFTED
void lifted_register_all();   // generated
extern const u32 g_lifted_code_hash, g_lifted_code_size;
#else
static void lifted_register_all() {}
static const u32 g_lifted_code_hash = 0, g_lifted_code_size = 0;
#endif
extern int g_interp_only;
extern int g_svc_log;
#ifdef __EMSCRIPTEN__
std::string g_rom_path = "stream:rom";   // served by the page (web/rom_worker.js)
std::string g_save_root = "/save";       // IndexedDB-backed mount, see web/app.js
#else
std::string g_rom_path = "game.3ds";
std::string g_save_root = "save";
#endif

static void setup_config_mem() {
    mem_map(CFGMEM_VA, 0x1000, MEM_RW, "configmem");
    wr8(CFGMEM_VA + 0x02, 0x37); wr8(CFGMEM_VA + 0x03, 2);
    wr64(CFGMEM_VA + 0x08, 0x0004013000008002ull);
    wr32(CFGMEM_VA + 0x10, 2);
    wr8(CFGMEM_VA + 0x14, 1);
    wr32(CFGMEM_VA + 0x30, 0);
    wr32(CFGMEM_VA + 0x40, 0x04000000);
    wr32(CFGMEM_VA + 0x44, 0x02C00000);
    wr32(CFGMEM_VA + 0x48, 0x01400000);
    wr8(CFGMEM_VA + 0x62, 0x37); wr8(CFGMEM_VA + 0x63, 2);
    wr32(CFGMEM_VA + 0x64, 2);

    mem_map(SHPAGE_VA, 0x1000, MEM_RW, "sharedpage");
    wr8(SHPAGE_VA + 0x04, 1);
    u64 ms = ((u64)time(nullptr) + 2208988800ull) * 1000ull;
    wr64(SHPAGE_VA + 0x20, ms);
    wr64(SHPAGE_VA + 0x28, now_ticks());
    wr64(SHPAGE_VA + 0x30, 0xFFFFFFFFull << 32);
    wr64(SHPAGE_VA + 0x40, ms);
    wr64(SHPAGE_VA + 0x48, now_ticks());
    wr8(SHPAGE_VA + 0x67, 3);          // wifi link level
    wr8(SHPAGE_VA + 0x84, 0);          // 3D LED
    wr8(SHPAGE_VA + 0x86, 1);          // battery: charging/adapter
    wr8(SHPAGE_VA + 0x87, 5);
}

int g_host_fp_plain = 0;
int main(int argc, char **argv) {
    if (getenv("R3DS_FP_PLAIN")) g_host_fp_plain = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--interp") g_interp_only = 1;
        else if (a == "--svclog") g_svc_log = 1;
        else if (a == "--ipclog") g_ipc_log = 1;
        else if (a == "-v") g_log_level = 2;
        else if (a == "--save" && i + 1 < argc) g_save_root = argv[++i];
        else g_rom_path = a;
    }
    if (getenv("R3DS_INTERP")) g_interp_only = 1;
    if (const char *sd = getenv("R3DS_SAVE_DIR")) g_save_root = sd;
    { extern int g_profiling; if (getenv("R3DS_PROFILE")) g_profiling = 1; }
    setvbuf(stderr, nullptr, _IOLBF, 0);

    mem_init();
    interp_init();

    if (!rom_open(g_rom_path))
        fatal("cannot use ROM '%s' (pass the path to a decrypted .3ds as the first argument)", g_rom_path.c_str());
    std::vector<u8> img;
    if (!rom_load_code(img)) fatal("cannot read code from the ROM's ExeFS");
    disp_init();   // sizes the dispatch table from the exheader layout
    size_t n = img.size();
    if (n > IMG_END - IMG_BASE) fatal("code.bin too large (%zx)", n);
    u32 h = code_hash(img.data(), n);
    if (g_lifted_code_size && (h != g_lifted_code_hash || n != g_lifted_code_size)) {
        LOG("[boot] WARNING: this ROM's code (hash %08x, %zx bytes) differs from the one the lifted code was", h, n);
        LOG("[boot]          generated from (hash %08x, %x bytes). Re-run the lifter for this ROM.", g_lifted_code_hash, g_lifted_code_size);
        if (!getenv("R3DS_FORCE")) fatal("ROM/lift mismatch (set R3DS_FORCE=1 to run anyway)");
    }
    mem_map(IMG_BASE, TEXT_END - IMG_BASE, MEM_RX, "text");
    mem_map(TEXT_END, RO_END - TEXT_END, MEM_R, "rodata");
    mem_map(RO_END, IMG_END - RO_END, MEM_RW, "data");
    mem_protect(IMG_BASE, RO_END - IMG_BASE, MEM_RW);
    memcpy(gp(IMG_BASE), img.data(), n);
    mem_protect(IMG_BASE, TEXT_END - IMG_BASE, MEM_RX);
    mem_protect(TEXT_END, RO_END - TEXT_END, MEM_R);
    INFO("[boot] code.bin %zu bytes", n);

    mem_map(STACK_TOP - g_stack_size, g_stack_size, MEM_RW, "main stack");
    mem_map(VRAM_VA, VRAM_SIZE, MEM_RW, "vram");
    setup_config_mem();

    if (!g_interp_only) lifted_register_all();
    extern void hle_hooks_register(); hle_hooks_register();
    services_init();

    g_k.start_main(g_entry, STACK_TOP, 0x30);
    extern bool frontend_run();
    frontend_run();   // returns only when running headless
#if defined(R3DS_WEBGPU) && !defined(__EMSCRIPTEN__)
    // R3DS_HWR_TEST=dir: run the WebGPU renderer off screen and write PNGs of it
    if (const char *dir = getenv("R3DS_HWR_TEST")) {
        std::thread([dir] {
            extern void hwr_test_main(const char *dir);
            hwr_test_main(dir);
        }).detach();
    }
#endif
    extern void kernel_dump_threads();
    int wd = getenv("R3DS_WATCHDOG") ? atoi(getenv("R3DS_WATCHDOG")) : 0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(wd ? wd : 3600));
        if (wd) kernel_dump_threads();
        extern int g_profiling; extern void prof_dump(const char *);
        if (g_profiling) prof_dump("profile.txt");
    }
}
