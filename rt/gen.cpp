#include "gen.h"
#include "mem.h"
#include <atomic>
#include <cstddef>

u32 *g_gen_table = nullptr;
u32 g_gen_span = 0;

#ifndef __EMSCRIPTEN__
bool gen_init() { return false; }
bool gen_active() { return false; }
u32 gen_run(Cpu &, u32 pc) { return pc; }
u32 gen_entry(u32) { return 0; }
void gen_mark_host(u32) {}
#else
#include <emscripten.h>

// The generated code addresses the Cpu struct by these offsets.
static_assert(offsetof(Cpu, r) == 0 && offsetof(Cpu, n) == 64 && offsetof(Cpu, q) == 80 && offsetof(Cpu, ge) == 84 &&
              offsetof(Cpu, fpscr) == 88 && offsetof(Cpu, f) == 112, "Cpu layout changed: update web/translator/emit.js CPU");

u32 armint_step(Cpu &c, u32 pc, int *svc);
extern std::atomic<int> g_irq_pending;

// one instruction on the interpreter (the translator's fallback for anything without a native lowering)
extern "C" EMSCRIPTEN_KEEPALIVE u32 gh_step(Cpu *c, u32 pc) {
    c->thumb = pc & 1;
    int svc;
    u32 next = armint_step(*c, pc, &svc);
    if (svc >= 0) {
        c->r[15] = next & ~1u; c->thumb = next & 1;
        rt_svc(*c, (u32)svc);
    }
    c->thumb = next & 1;
    return next;
}
extern "C" EMSCRIPTEN_KEEPALIVE void gh_svc(Cpu *c, u32 imm, u32 next) {
    c->r[15] = next & ~1u; c->thumb = next & 1;
    rt_svc(*c, imm);
}

struct GenBlob { u32 ptr, len, nchunks, ok; };
static GenBlob g_blob;

// Runs in the calling thread's JS context (a pthread worker): translation is CPU bound
// and the main browser thread stays responsive.
EM_JS(int, gen_js_translate, (u32 code, u32 codeLen, u32 base, u32 textEnd, u32 roEnd, u32 dataEnd,
                              u32 gbase, u32 irqAddr, u32 tableAddr, u32 out), {
    if (typeof R3T === 'undefined') return 0;
    try {
        const t0 = performance.now();
        const bytes = HEAPU8.subarray(code, code + codeLen);
        const r = R3T.translate(bytes, { base, textEnd, roEnd, dataEnd },
            { gbase, irqAddr, tableAddr }, (s) => out('[gen] ' + s));
        HEAPU32.set(r.table, tableAddr >>> 2);
        const p = _malloc(r.wasm.length);
        HEAPU8.set(r.wasm, p);
        HEAPU32[out >>> 2] = p; HEAPU32[(out >>> 2) + 1] = r.wasm.length; HEAPU32[(out >>> 2) + 2] = r.nchunks;
        out('[gen] translated ' + r.nchunks + ' chunks, ' + (r.wasm.length / 1048576).toFixed(1) + ' MiB in ' + (performance.now() - t0).toFixed(0) + ' ms');
        return 1;
    } catch (e) { err('[gen] translation failed: ' + e + '\n' + (e && e.stack)); return 0; }
});

// Instantiate the generated module in this thread and put its `run` into this thread's table.
EM_JS(int, gen_js_bind, (u32 ptr, u32 len), {
    try {
        const mod = new WebAssembly.Module(new Uint8Array(HEAPU8.buffer, ptr, len));
        const inst = new WebAssembly.Instance(mod, { env: { memory: wasmMemory, step: wasmExports['gh_step'], svc: wasmExports['gh_svc'] } });
        const idx = wasmTable.length;
        wasmTable.grow(1);
        wasmTable.set(idx, inst.exports.run);
        return idx;
    } catch (e) { err('[gen] instantiate failed: ' + e); return 0; }
});

typedef u32 (*GenRunFn)(Cpu *, u32);
static GenRunFn g_gen_fn0;              // main thread's bound function, for gen_active()
static thread_local GenRunFn tls_run = nullptr;

bool gen_init() {
    if (getenv("R3DS_GEN") && getenv("R3DS_GEN")[0] == '0') return false;
    u32 nHalf = ((TEXT_END - IMG_BASE) >> 1) + 2;
    g_gen_table = (u32 *)calloc(nHalf, 4);
    g_gen_span = TEXT_END - IMG_BASE;
    if (!g_gen_table) return false;
    u32 n = IMG_END - IMG_BASE;   // the loaded image: .text .rodata .data (+ bss)
    LOG("[gen] translating the code image before boot...");
    g_blob = {};
    int ok = gen_js_translate((u32)(uintptr_t)gp(IMG_BASE), n, IMG_BASE, TEXT_END, RO_END, IMG_END,
                              (u32)(uintptr_t)g_base, (u32)(uintptr_t)&g_irq_pending, (u32)(uintptr_t)g_gen_table, (u32)(uintptr_t)&g_blob);
    if (!ok || !g_blob.ptr) { free(g_gen_table); g_gen_table = nullptr; LOG("[gen] unavailable: running on the interpreter"); return false; }
    return true;
}

bool gen_active() { return g_gen_table != nullptr && g_blob.ptr; }

u32 gen_run(Cpu &c, u32 pc) {
    if (__builtin_expect(!tls_run, 0)) {
        int idx = gen_js_bind(g_blob.ptr, g_blob.len);
        if (!idx) fatal("cannot instantiate the generated code in this thread");
        tls_run = (GenRunFn)(uintptr_t)idx;
    }
    return tls_run(&c, pc);
}

u32 gen_entry(u32 pc) {
    u32 d = (pc & ~1u) - IMG_BASE;
    if (!g_gen_table || d >= g_gen_span) return 0;
    return g_gen_table[d >> 1];
}
void gen_mark_host(u32 pc) {
    u32 d = (pc & ~1u) - IMG_BASE;
    if (g_gen_table && d < g_gen_span) g_gen_table[d >> 1] = 1;
}
#endif
