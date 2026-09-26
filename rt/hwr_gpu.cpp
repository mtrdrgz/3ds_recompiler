// Hardware renderer, executor side: replays the recorder's command stream
// with WebGPU (Dawn in the browser via Emscripten, wgpu-native on PC) at a
// multiple of the 3DS resolution, and composes the screens for the window.
// Runs on the frontend's thread (the browser's main thread on the web).
#ifdef R3DS_WEBGPU
#include "hwr.h"
#include "hwr_shaders.h"
#include "hwr_fonts.h"
#include "kernel.h"
#if __has_include(<webgpu/webgpu.h>)
#include <webgpu/webgpu.h>
#else
#include <webgpu.h>
#endif
#if !defined(__EMSCRIPTEN__) && __has_include(<wgpu.h>)
#include <wgpu.h>
#define R3DS_WGPU_NATIVE 1
#endif
#include <unordered_map>
#include <map>
#include <vector>
#include <deque>
#include <string>
#include <cmath>
#include <algorithm>
#include <thread>

static WGPUStringView sv(const char *s) { return WGPUStringView{s, WGPU_STRLEN}; }
static std::string svs(WGPUStringView v) { return v.data ? std::string(v.data, v.length == WGPU_STRLEN ? strlen(v.data) : v.length) : std::string(); }

struct Tex {
    WGPUTexture tex = nullptr; WGPUTextureView view = nullptr; u32 w = 0, h = 0;
    explicit operator bool() const { return tex != nullptr; }
};
struct GSurf { u32 kind = 0, lw = 0, lh = 0, pw = 0, ph = 0; Tex color, cov, depth; };
struct GLcd { u32 id; u64 hash; u32 bytes; };
struct PipeKeyHash { size_t operator()(const HwrPipeKey &k) const { return (size_t)k.blend * 31 ^ (size_t)k.misc * 131 ^ (size_t)k.depth * 7919 ^ (size_t)k.stencil * 104729; } };
struct Tri3 { u32 a, b, c; bool operator==(const Tri3 &o) const { return a == o.a && b == o.b && c == o.c; } };
struct Tri3Hash { size_t operator()(const Tri3 &t) const { return t.a * 73856093u ^ t.b * 19349663u ^ t.c * 83492791u; } };

static struct Gpu {
    int state = 0;   // 0 idle, 1 requesting, 2 ready, 3 failed
    WGPUInstance inst = nullptr; WGPUAdapter adapter = nullptr; WGPUDevice dev = nullptr; WGPUQueue queue = nullptr;
    WGPUSurface surface = nullptr; WGPUTextureFormat sfmt = WGPUTextureFormat_BGRA8Unorm;
    u32 sw = 0, sh = 0;
    WGPUBindGroupLayout bgl_uni = nullptr, bgl_tex = nullptr, bgl_blit = nullptr, bgl_depth = nullptr, bgl_comp = nullptr;
    WGPUPipelineLayout pl_pica = nullptr, pl_blit = nullptr, pl_depth = nullptr, pl_comp = nullptr;
    WGPUShaderModule sm_pica = nullptr, sm_blit = nullptr, sm_depth = nullptr, sm_comp = nullptr;
    std::unordered_map<HwrPipeKey, WGPURenderPipeline, PipeKeyHash> pipes;
    WGPURenderPipeline blit_surf = nullptr, blit_lcd = nullptr, depth_pipe = nullptr, stencil_pipe[8] = {};
    std::map<std::pair<int, WGPUTextureFormat>, WGPURenderPipeline> comp_pipes;   // (blend, format)
    WGPUSampler samp_lin = nullptr, samp_near = nullptr;
    std::unordered_map<u32, GSurf> surfs;
    std::unordered_map<u32, Tex> texs;
    std::unordered_map<u32, std::vector<GLcd>> lcds;
    std::map<std::pair<u32, u32>, Tex> dummy_depth;
    Tex dummy, frame1x, blur;
    std::unordered_map<Tri3, WGPUBindGroup, Tri3Hash> bg_tex;
    // per submission
    WGPUCommandEncoder enc = nullptr;
    WGPURenderPassEncoder pass = nullptr; u32 pass_color = 0, pass_depth = 0;
    WGPUBuffer vbuf = nullptr, ubuf = nullptr; u64 vcap = 0, ucap = 0;
    WGPUBindGroup bg_uni = nullptr;
    std::vector<u8> vdata, udata;
    std::vector<WGPUTexture> dead_tex; std::vector<WGPUTextureView> dead_view; std::vector<WGPUBindGroup> dead_bg;
    // presentation state
    std::vector<u32> swframe; HwrFrameInfo info; bool have_frame = false; u32 frames_shown = 0;
    int bmode_cur = 1, bmode_votes = 0; float cur[4] = {0, 0, 0, 0}; bool have_cur = false;
    u32 want_scale = 0, scale_votes = 0; float want_wide = 1.0f; u32 wide_votes = 0;
    u32 errors = 0;
    HwrOut out;
    // R3DS_HWR_STATS=1: per-second counters
    struct { u64 draws, verts, loads, texs, tex_bytes, blits, chunks, bytes, frames, pipes, soft_top, soft_bot; double t_replay, t_submit, t_present; } st{};
} G;
static double now_ms() { return now_ns() / 1e6; }

// ------------------------------------------------------------ helpers
static Tex make_tex(u32 w, u32 h, WGPUTextureFormat f, WGPUTextureUsage usage, const char *label) {
    WGPUTextureDescriptor d = WGPU_TEXTURE_DESCRIPTOR_INIT;
    d.label = sv(label); d.usage = usage; d.dimension = WGPUTextureDimension_2D;
    d.size = {std::max(1u, w), std::max(1u, h), 1}; d.format = f;
    Tex t; t.tex = wgpuDeviceCreateTexture(G.dev, &d); t.w = d.size.width; t.h = d.size.height;
    t.view = wgpuTextureCreateView(t.tex, nullptr);
    return t;
}
static void drop_tex(Tex &t) {
    if (t.view) G.dead_view.push_back(t.view);
    if (t.tex) G.dead_tex.push_back(t.tex);
    t = Tex{};
}
static void write_tex(const Tex &t, const void *data, u32 w, u32 h, u32 bpp) {
    WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    dst.texture = t.tex;
    WGPUTexelCopyBufferLayout lay = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
    lay.bytesPerRow = w * bpp; lay.rowsPerImage = h;
    WGPUExtent3D ext = {w, h, 1};
    wgpuQueueWriteTexture(G.queue, &dst, data, (size_t)w * h * bpp, &lay, &ext);
}
static WGPUShaderModule shader(const char *src, const char *label) {
    WGPUShaderSourceWGSL w = WGPU_SHADER_SOURCE_WGSL_INIT;
    w.code = sv(src);
    WGPUShaderModuleDescriptor d = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    d.nextInChain = &w.chain; d.label = sv(label);
    return wgpuDeviceCreateShaderModule(G.dev, &d);
}
static u32 align256(u32 v) { return (v + 255) & ~255u; }
static u32 push_uni(const void *p, size_t n) {
    u32 off = align256((u32)G.udata.size());
    G.udata.resize(off + std::max<size_t>(n, 256));
    memcpy(&G.udata[off], p, n);
    return off;
}
static void end_pass() {
    if (!G.pass) return;
    wgpuRenderPassEncoderEnd(G.pass);
    wgpuRenderPassEncoderRelease(G.pass);
    G.pass = nullptr; G.pass_color = G.pass_depth = 0;
}
static const Tex &dummy_depth(u32 w, u32 h) {
    auto &t = G.dummy_depth[{w, h}];
    if (!t) t = make_tex(w, h, WGPUTextureFormat_Depth24PlusStencil8, WGPUTextureUsage_RenderAttachment, "dummy depth");
    return t;
}

// ------------------------------------------------------------ pipelines
static WGPUBlendFactor bf(u32 f) {
    switch (f) {
    case 0: return WGPUBlendFactor_Zero; case 1: return WGPUBlendFactor_One;
    case 2: return WGPUBlendFactor_Src; case 3: return WGPUBlendFactor_OneMinusSrc;
    case 4: return WGPUBlendFactor_Dst; case 5: return WGPUBlendFactor_OneMinusDst;
    case 6: return WGPUBlendFactor_SrcAlpha; case 7: return WGPUBlendFactor_OneMinusSrcAlpha;
    case 8: return WGPUBlendFactor_DstAlpha; case 9: return WGPUBlendFactor_OneMinusDstAlpha;
    case 10: case 12: return WGPUBlendFactor_Constant; case 11: case 13: return WGPUBlendFactor_OneMinusConstant;
    case 14: return WGPUBlendFactor_SrcAlphaSaturated;
    }
    return WGPUBlendFactor_One;
}
static WGPUBlendOperation bop(u32 e) {
    switch (e) { case 0: return WGPUBlendOperation_Add; case 1: return WGPUBlendOperation_Subtract;
    case 2: return WGPUBlendOperation_ReverseSubtract; case 3: return WGPUBlendOperation_Min; default: return WGPUBlendOperation_Max; }
}
static WGPUBlendComponent bcomp(u32 eq, u32 sf, u32 df) {
    WGPUBlendComponent c;
    c.operation = bop(eq);
    bool mm = c.operation == WGPUBlendOperation_Min || c.operation == WGPUBlendOperation_Max;
    c.srcFactor = mm ? WGPUBlendFactor_One : bf(sf);
    c.dstFactor = mm ? WGPUBlendFactor_One : bf(df);
    return c;
}
static WGPUCompareFunction cmpfn(u32 f) {
    static const WGPUCompareFunction m[8] = {WGPUCompareFunction_Never, WGPUCompareFunction_Always, WGPUCompareFunction_Equal, WGPUCompareFunction_NotEqual,
                                             WGPUCompareFunction_Less, WGPUCompareFunction_LessEqual, WGPUCompareFunction_Greater, WGPUCompareFunction_GreaterEqual};
    return m[f & 7];
}
static WGPUStencilOperation sop(u32 o) {
    static const WGPUStencilOperation m[8] = {WGPUStencilOperation_Keep, WGPUStencilOperation_Zero, WGPUStencilOperation_Replace, WGPUStencilOperation_IncrementClamp,
                                              WGPUStencilOperation_DecrementClamp, WGPUStencilOperation_Invert, WGPUStencilOperation_IncrementWrap, WGPUStencilOperation_DecrementWrap};
    return m[o & 7];
}
static int g_cull_flip = -1;
static WGPURenderPipeline pica_pipe(const HwrPipeKey &k) {
    auto it = G.pipes.find(k);
    if (it != G.pipes.end()) return it->second;
    WGPUVertexAttribute attrs[6];
    for (int i = 0; i < 5; i++) { attrs[i] = WGPU_VERTEX_ATTRIBUTE_INIT; attrs[i].format = WGPUVertexFormat_Float32x4; attrs[i].offset = 16 * i; attrs[i].shaderLocation = i; }
    attrs[5] = WGPU_VERTEX_ATTRIBUTE_INIT; attrs[5].format = WGPUVertexFormat_Float32; attrs[5].offset = 80; attrs[5].shaderLocation = 5;
    WGPUVertexBufferLayout vbl = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
    vbl.stepMode = WGPUVertexStepMode_Vertex; vbl.arrayStride = HWR_VFLOATS * 4; vbl.attributeCount = 6; vbl.attributes = attrs;

    bool blend_on = k.misc & 1;
    u32 lclass = (k.misc >> 1) & 15, wmask = (k.misc >> 8) & 15, cov = (k.misc >> 12) & 3, cull = (k.misc >> 14) & 3;
    WGPUBlendState bs;
    if (blend_on) {
        u32 b = k.blend;
        bs.color = bcomp(b & 7, (b >> 16) & 15, (b >> 20) & 15);
        bs.alpha = bcomp((b >> 8) & 7, (b >> 24) & 15, b >> 28);
    } else if (lclass == 5) {   // invert: 1 * (1 - dst)
        bs.color = {WGPUBlendOperation_Add, WGPUBlendFactor_OneMinusDst, WGPUBlendFactor_Zero};
        bs.alpha = bs.color;
    }
    WGPUBlendState cbs;   // coverage: max(alpha) or replace
    cbs.color = {WGPUBlendOperation_Max, WGPUBlendFactor_One, WGPUBlendFactor_One};
    cbs.alpha = cbs.color;
    WGPUColorTargetState ct[2] = {WGPU_COLOR_TARGET_STATE_INIT, WGPU_COLOR_TARGET_STATE_INIT};
    ct[0].format = WGPUTextureFormat_RGBA8Unorm;
    ct[0].blend = (blend_on || lclass == 5) ? &bs : nullptr;
    ct[0].writeMask = (wmask & 1 ? WGPUColorWriteMask_Red : 0) | (wmask & 2 ? WGPUColorWriteMask_Green : 0) |
                      (wmask & 4 ? WGPUColorWriteMask_Blue : 0) | (wmask & 8 ? WGPUColorWriteMask_Alpha : 0);
    ct[1].format = WGPUTextureFormat_R8Unorm;
    ct[1].blend = cov == 1 ? &cbs : nullptr;
    ct[1].writeMask = cov ? WGPUColorWriteMask_Red : WGPUColorWriteMask_None;

    WGPUDepthStencilState ds = WGPU_DEPTH_STENCIL_STATE_INIT;
    ds.format = WGPUTextureFormat_Depth24PlusStencil8;
    ds.depthCompare = (k.depth & 1) ? cmpfn((k.depth >> 1) & 7) : WGPUCompareFunction_Always;
    ds.depthWriteEnabled = (k.depth & 16) ? WGPUOptionalBool_True : WGPUOptionalBool_False;
    WGPUStencilFaceState sf;
    if (k.stencil & 1) {
        sf.compare = cmpfn((k.stencil >> 1) & 7);
        sf.failOp = sop((k.stencil >> 4) & 7); sf.depthFailOp = sop((k.stencil >> 7) & 7); sf.passOp = sop((k.stencil >> 10) & 7);
        ds.stencilReadMask = (k.stencil >> 16) & 0xFF; ds.stencilWriteMask = k.stencil >> 24;
    } else {
        sf = {WGPUCompareFunction_Always, WGPUStencilOperation_Keep, WGPUStencilOperation_Keep, WGPUStencilOperation_Keep};
        ds.stencilReadMask = 0; ds.stencilWriteMask = 0;
    }
    ds.stencilFront = sf; ds.stencilBack = sf;

    WGPUFragmentState fs = WGPU_FRAGMENT_STATE_INIT;
    fs.module = G.sm_pica; fs.entryPoint = sv("fs"); fs.targetCount = 2; fs.targets = ct;
    WGPURenderPipelineDescriptor d = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    d.label = sv("pica"); d.layout = G.pl_pica;
    d.vertex.module = G.sm_pica; d.vertex.entryPoint = sv("vs"); d.vertex.bufferCount = 1; d.vertex.buffers = &vbl;
    d.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    // PICA cull 1 removes triangles that are counter-clockwise in its screen
    // space (y up); the GPU clip space has the same orientation
    if (g_cull_flip < 0) g_cull_flip = getenv("R3DS_HWR_CULLFLIP") ? 1 : 0;
    if (cull) {
        bool ccw_culled = (cull == 1) ^ (g_cull_flip == 1);
        d.primitive.frontFace = ccw_culled ? WGPUFrontFace_CW : WGPUFrontFace_CCW;
        d.primitive.cullMode = WGPUCullMode_Back;
    } else {
        d.primitive.frontFace = WGPUFrontFace_CCW; d.primitive.cullMode = WGPUCullMode_None;
    }
    d.depthStencil = &ds;
    d.fragment = &fs;
    WGPURenderPipeline p = wgpuDeviceCreateRenderPipeline(G.dev, &d);
    G.pipes[k] = p;
    G.st.pipes++;
    return p;
}

static WGPUBindGroupLayout bgl(std::initializer_list<WGPUBindGroupLayoutEntry> es) {
    std::vector<WGPUBindGroupLayoutEntry> v(es);
    WGPUBindGroupLayoutDescriptor d = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    d.entryCount = v.size(); d.entries = v.data();
    return wgpuDeviceCreateBindGroupLayout(G.dev, &d);
}
static WGPUBindGroupLayoutEntry e_uni(u32 b, bool dyn, u64 size, WGPUShaderStage vis) {
    WGPUBindGroupLayoutEntry e = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    e.binding = b; e.visibility = vis;
    e.buffer.type = WGPUBufferBindingType_Uniform; e.buffer.hasDynamicOffset = dyn; e.buffer.minBindingSize = size;
    return e;
}
static WGPUBindGroupLayoutEntry e_tex(u32 b, WGPUTextureSampleType t) {
    WGPUBindGroupLayoutEntry e = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    e.binding = b; e.visibility = WGPUShaderStage_Fragment;
    e.texture.sampleType = t; e.texture.viewDimension = WGPUTextureViewDimension_2D;
    return e;
}
static WGPUBindGroupLayoutEntry e_samp(u32 b) {
    WGPUBindGroupLayoutEntry e = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    e.binding = b; e.visibility = WGPUShaderStage_Fragment; e.sampler.type = WGPUSamplerBindingType_Filtering;
    return e;
}
static WGPUPipelineLayout playout(std::initializer_list<WGPUBindGroupLayout> ls) {
    std::vector<WGPUBindGroupLayout> v(ls);
    WGPUPipelineLayoutDescriptor d = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    d.bindGroupLayoutCount = v.size(); d.bindGroupLayouts = v.data();
    return wgpuDeviceCreatePipelineLayout(G.dev, &d);
}
static WGPURenderPipeline simple_pipe(WGPUShaderModule m, const char *fs_entry, WGPUPipelineLayout pl,
                                      std::vector<WGPUColorTargetState> targets, const WGPUDepthStencilState *ds, WGPUPrimitiveTopology topo) {
    WGPUFragmentState fs = WGPU_FRAGMENT_STATE_INIT;
    fs.module = m; fs.entryPoint = sv(fs_entry); fs.targetCount = targets.size(); fs.targets = targets.data();
    WGPURenderPipelineDescriptor d = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    d.layout = pl;
    d.vertex.module = m; d.vertex.entryPoint = sv("vs");
    d.primitive.topology = topo;
    d.depthStencil = ds;
    d.fragment = &fs;
    return wgpuDeviceCreateRenderPipeline(G.dev, &d);
}
static WGPURenderPipeline comp_pipe(bool blend, WGPUTextureFormat f) {
    auto &p = G.comp_pipes[{blend ? 1 : 0, f}];
    if (p) return p;
    WGPUBlendState bs;
    bs.color = {WGPUBlendOperation_Add, WGPUBlendFactor_One, WGPUBlendFactor_OneMinusSrcAlpha};
    bs.alpha = bs.color;
    WGPUColorTargetState ct = WGPU_COLOR_TARGET_STATE_INIT;
    ct.format = f; ct.blend = blend ? &bs : nullptr;
    p = simple_pipe(G.sm_comp, "fs", G.pl_comp, {ct}, nullptr, WGPUPrimitiveTopology_TriangleStrip);
    return p;
}

static void create_pipelines() {
    G.sm_pica = shader(WGSL_PICA, "pica");
    G.sm_blit = shader(WGSL_BLIT, "blit");
    G.sm_depth = shader(WGSL_DEPTH, "depth");
    G.sm_comp = shader(WGSL_COMP, "comp");
    const WGPUShaderStage VF = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
    G.bgl_uni = bgl({e_uni(0, true, HWR_UVEC4 * 16, VF)});
    G.bgl_tex = bgl({e_tex(0, WGPUTextureSampleType_UnfilterableFloat), e_tex(1, WGPUTextureSampleType_UnfilterableFloat), e_tex(2, WGPUTextureSampleType_UnfilterableFloat)});
    G.bgl_blit = bgl({e_uni(0, true, 32, VF), e_samp(1), e_tex(2, WGPUTextureSampleType_Float), e_tex(3, WGPUTextureSampleType_Float)});
    G.bgl_depth = bgl({e_tex(0, WGPUTextureSampleType_Uint), e_uni(1, true, 16, WGPUShaderStage_Fragment)});
    G.bgl_comp = bgl({e_uni(0, true, 80, VF), e_samp(1), e_tex(2, WGPUTextureSampleType_Float)});
    G.pl_pica = playout({G.bgl_uni, G.bgl_tex});
    G.pl_blit = playout({G.bgl_blit});
    G.pl_depth = playout({G.bgl_depth});
    G.pl_comp = playout({G.bgl_comp});

    WGPUColorTargetState c0 = WGPU_COLOR_TARGET_STATE_INIT, c1 = WGPU_COLOR_TARGET_STATE_INIT;
    c0.format = WGPUTextureFormat_RGBA8Unorm; c1.format = WGPUTextureFormat_R8Unorm;
    G.blit_surf = simple_pipe(G.sm_blit, "fs_surf", G.pl_blit, {c0, c1}, nullptr, WGPUPrimitiveTopology_TriangleList);
    G.blit_lcd = simple_pipe(G.sm_blit, "fs_lcd", G.pl_blit, {c0}, nullptr, WGPUPrimitiveTopology_TriangleList);
    WGPUDepthStencilState ds = WGPU_DEPTH_STENCIL_STATE_INIT;
    ds.format = WGPUTextureFormat_Depth24PlusStencil8;
    ds.depthCompare = WGPUCompareFunction_Always; ds.depthWriteEnabled = WGPUOptionalBool_True;
    ds.stencilReadMask = 0; ds.stencilWriteMask = 0;
    ds.stencilFront = ds.stencilBack = {WGPUCompareFunction_Always, WGPUStencilOperation_Keep, WGPUStencilOperation_Keep, WGPUStencilOperation_Keep};
    G.depth_pipe = simple_pipe(G.sm_depth, "fs_depth", G.pl_depth, {}, &ds, WGPUPrimitiveTopology_TriangleList);
    for (int b = 0; b < 8; b++) {
        WGPUDepthStencilState s = ds;
        s.depthWriteEnabled = WGPUOptionalBool_False;
        s.stencilWriteMask = 1u << b; s.stencilReadMask = 0xFF;
        s.stencilFront = s.stencilBack = {WGPUCompareFunction_Always, WGPUStencilOperation_Keep, WGPUStencilOperation_Keep, WGPUStencilOperation_Replace};
        G.stencil_pipe[b] = simple_pipe(G.sm_depth, "fs_stencil", G.pl_depth, {}, &s, WGPUPrimitiveTopology_TriangleList);
    }
    WGPUSamplerDescriptor sd = WGPU_SAMPLER_DESCRIPTOR_INIT;
    sd.addressModeU = sd.addressModeV = sd.addressModeW = WGPUAddressMode_ClampToEdge;
    sd.magFilter = sd.minFilter = WGPUFilterMode_Linear;
    G.samp_lin = wgpuDeviceCreateSampler(G.dev, &sd);
    sd.magFilter = sd.minFilter = WGPUFilterMode_Nearest;
    G.samp_near = wgpuDeviceCreateSampler(G.dev, &sd);
    G.dummy = make_tex(1, 1, WGPUTextureFormat_RGBA8Unorm, WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst, "dummy");
    u32 z = 0; write_tex(G.dummy, &z, 1, 1, 4);
    G.frame1x = make_tex(400, 480, WGPUTextureFormat_RGBA8Unorm, WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst, "software frame");
    G.blur = make_tex(160, 96, WGPUTextureFormat_RGBA8Unorm, WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment, "blur");
}

// ------------------------------------------------------------ resources
struct Src { WGPUTextureView view; float tw, th; bool rotated; float u0, v0, uw, vh; float aspect; };   // aspect of the screen image
// The last GPU image shown on each screen. When the surface behind it goes
// away (a reset: the recorder starts over after the executor stalled; or the
// game freeing it) its texture is kept here for a moment, so the screen keeps
// the full-resolution picture while the surfaces rebuild instead of dropping
// to the 1x software image for a few frames (a visible low-res flicker).
struct Held { u32 id = 0; Src src{}; Tex tex; double t = 0; };
static Held g_held[2];
static void release_surf(GSurf &s, u32 id = 0) {
    for (auto &h : g_held)
        if (id && h.id == id && !h.tex && s.color) { h.tex = s.color; s.color = Tex{}; h.t = now_ns() / 1e6; }
    drop_tex(s.color); drop_tex(s.cov); drop_tex(s.depth);
}
static void forget_bind_groups(u32 id) {
    for (auto it = G.bg_tex.begin(); it != G.bg_tex.end();) {
        if (it->first.a == id || it->first.b == id || it->first.c == id) { G.dead_bg.push_back(it->second); it = G.bg_tex.erase(it); }
        else ++it;
    }
}
static void reset_all() {
    end_pass();
    for (auto &kv : G.surfs) release_surf(kv.second, kv.first);
    for (auto &kv : G.texs) drop_tex(kv.second);
    for (auto &kv : G.bg_tex) G.dead_bg.push_back(kv.second);
    G.surfs.clear(); G.texs.clear(); G.lcds.clear(); G.bg_tex.clear();
}
// view to sample for id (a surface's color or an uploaded texture)
static WGPUTextureView tex_view(u32 id) {
    if (!id) return G.dummy.view;
    auto s = G.surfs.find(id);
    if (s != G.surfs.end()) return s->second.color ? s->second.color.view : G.dummy.view;
    auto t = G.texs.find(id);
    return t != G.texs.end() ? t->second.view : G.dummy.view;
}

static void begin_pass(u32 color_id, u32 depth_id) {
    if (G.pass && G.pass_color == color_id && G.pass_depth == depth_id) return;
    end_pass();
    GSurf &c = G.surfs[color_id];
    WGPURenderPassColorAttachment ca[2] = {WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT, WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT};
    ca[0].view = c.color.view; ca[0].loadOp = WGPULoadOp_Load; ca[0].storeOp = WGPUStoreOp_Store;
    ca[1].view = c.cov.view; ca[1].loadOp = WGPULoadOp_Load; ca[1].storeOp = WGPUStoreOp_Store;
    WGPURenderPassDepthStencilAttachment da = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
    auto d = depth_id ? G.surfs.find(depth_id) : G.surfs.end();
    bool real = d != G.surfs.end() && d->second.depth && d->second.pw == c.pw && d->second.ph == c.ph;
    da.view = real ? d->second.depth.view : dummy_depth(c.pw, c.ph).view;
    da.depthLoadOp = WGPULoadOp_Load; da.depthStoreOp = WGPUStoreOp_Store;
    da.depthClearValue = 1.0f;   // unused with Load, but browsers reject the default NaN
    da.stencilLoadOp = WGPULoadOp_Load; da.stencilStoreOp = WGPUStoreOp_Store;
    WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    pd.colorAttachmentCount = 2; pd.colorAttachments = ca; pd.depthStencilAttachment = &da;
    G.pass = wgpuCommandEncoderBeginRenderPass(G.enc, &pd);
    G.pass_color = color_id; G.pass_depth = depth_id;
}

// ------------------------------------------------------------ command replay
struct Reader { const u8 *p, *e; };

static void op_surf_new(const u32 *a) {
    u32 id = a[0], kind = a[1];
    GSurf s; s.kind = kind; s.lw = a[2]; s.lh = a[3]; s.pw = a[4]; s.ph = a[5];
    const WGPUTextureUsage RT = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
    if (kind == HSK_DEPTH) s.depth = make_tex(s.pw, s.ph, WGPUTextureFormat_Depth24PlusStencil8, WGPUTextureUsage_RenderAttachment, "depth");
    else if (kind == HSK_LCD) s.color = make_tex(s.pw, s.ph, WGPUTextureFormat_RGBA8Unorm, RT, "lcd");
    else { s.color = make_tex(s.pw, s.ph, WGPUTextureFormat_RGBA8Unorm, RT, "color"); s.cov = make_tex(s.pw, s.ph, WGPUTextureFormat_R8Unorm, WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding, "cov"); }
    auto old = G.surfs.find(id);
    if (old != G.surfs.end()) release_surf(old->second);
    G.surfs[id] = s;
}
static void op_surf_free(u32 id) {
    auto it = G.surfs.find(id);
    if (it == G.surfs.end()) return;
    if (G.pass_color == id || G.pass_depth == id) end_pass();
    if (it->second.kind == HSK_LCD)
        for (auto &kv : G.lcds) kv.second.erase(std::remove_if(kv.second.begin(), kv.second.end(), [&](const GLcd &l) { return l.id == id; }), kv.second.end());
    forget_bind_groups(id);
    release_surf(it->second, id);
    G.surfs.erase(it);
}
static void op_clear(u32 id, u32 v, bool cov) {
    auto it = G.surfs.find(id);
    if (it == G.surfs.end()) return;
    end_pass();
    GSurf &s = it->second;
    WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    WGPURenderPassColorAttachment ca[2] = {WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT, WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT};
    WGPURenderPassDepthStencilAttachment da = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
    if (s.kind == HSK_DEPTH) {
        da.view = s.depth.view;
        da.depthLoadOp = WGPULoadOp_Clear; da.depthStoreOp = WGPUStoreOp_Store; da.depthClearValue = (v & 0xFFFFFF) / 16777215.0f;
        da.stencilLoadOp = WGPULoadOp_Clear; da.stencilStoreOp = WGPUStoreOp_Store; da.stencilClearValue = v >> 24;
        pd.depthStencilAttachment = &da;
    } else {
        ca[0].view = s.color.view; ca[0].loadOp = WGPULoadOp_Clear; ca[0].storeOp = WGPUStoreOp_Store;
        ca[0].clearValue = {(v & 0xFF) / 255.0, ((v >> 8) & 0xFF) / 255.0, ((v >> 16) & 0xFF) / 255.0, (v >> 24) / 255.0};
        pd.colorAttachmentCount = 1;
        if (s.cov) {
            ca[1].view = s.cov.view; ca[1].loadOp = cov ? WGPULoadOp_Clear : WGPULoadOp_Load; ca[1].storeOp = WGPUStoreOp_Store;
            ca[1].clearValue = {0, 0, 0, 0};
            pd.colorAttachmentCount = 2;
        }
        pd.colorAttachments = ca;
    }
    WGPURenderPassEncoder p = wgpuCommandEncoderBeginRenderPass(G.enc, &pd);
    wgpuRenderPassEncoderEnd(p); wgpuRenderPassEncoderRelease(p);
}
// blit src (color [+ cov]) into dst: surface (color + cov) or LCD image
static void blit(u32 src_id, u32 dst_id, float sx, float sy, float sw, float sh, bool flip, bool cov, bool noalpha, bool nearest,
                 WGPUTextureView src_color = nullptr, WGPUTextureView src_cov = nullptr) {
    auto di = G.surfs.find(dst_id);
    if (di == G.surfs.end()) return;
    GSurf &d = di->second;
    if (!src_color) {
        auto si = G.surfs.find(src_id);
        if (si == G.surfs.end() || !si->second.color) return;
        GSurf &s = si->second;
        src_color = s.color.view; src_cov = s.cov ? s.cov.view : G.dummy.view;
        sx /= s.lw; sy /= s.lh; sw /= s.lw; sh /= s.lh;
    }
    end_pass();
    struct { float r[4]; u32 f[4]; } bu = {{sx, sy, sw, sh}, {flip ? 1u : 0u, cov ? 1u : 0u, noalpha ? 1u : 0u, 0}};
    u32 off = push_uni(&bu, sizeof bu);
    WGPUBindGroupEntry e[4] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
    e[0].binding = 0; e[0].buffer = G.ubuf; e[0].size = 32;
    e[1].binding = 1; e[1].sampler = nearest ? G.samp_near : G.samp_lin;
    e[2].binding = 2; e[2].textureView = src_color;
    e[3].binding = 3; e[3].textureView = src_cov ? src_cov : G.dummy.view;
    WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bd.layout = G.bgl_blit; bd.entryCount = 4; bd.entries = e;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(G.dev, &bd);
    G.dead_bg.push_back(bg);
    WGPURenderPassColorAttachment ca[2] = {WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT, WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT};
    ca[0].view = d.color.view; ca[0].loadOp = WGPULoadOp_Clear; ca[0].storeOp = WGPUStoreOp_Store;
    ca[1].view = d.cov ? d.cov.view : nullptr; ca[1].loadOp = WGPULoadOp_Clear; ca[1].storeOp = WGPUStoreOp_Store;
    WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    bool lcd = d.kind == HSK_LCD;
    pd.colorAttachmentCount = lcd ? 1 : 2; pd.colorAttachments = ca;
    WGPURenderPassEncoder p = wgpuCommandEncoderBeginRenderPass(G.enc, &pd);
    wgpuRenderPassEncoderSetPipeline(p, lcd ? G.blit_lcd : G.blit_surf);
    wgpuRenderPassEncoderSetBindGroup(p, 0, bg, 1, &off);
    wgpuRenderPassEncoderDraw(p, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(p); wgpuRenderPassEncoderRelease(p);
}
static void op_load(const u32 *a, const u8 *data) {
    G.st.loads++;
    u32 id = a[0], lw = a[1], lh = a[2], has_cov = a[3];
    auto it = G.surfs.find(id);
    if (it == G.surfs.end()) return;
    GSurf &s = it->second;
    if (s.kind == HSK_COLOR) {
        Tex c = make_tex(lw, lh, WGPUTextureFormat_RGBA8Unorm, WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst, "load");
        write_tex(c, data, lw, lh, 4);
        Tex v;
        if (has_cov) {
            v = make_tex(lw, lh, WGPUTextureFormat_R8Unorm, WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst, "load cov");
            // rows of 1-byte texels: bytesPerRow must be a multiple of 256
            u32 bpr = (lw + 255) & ~255u;
            std::vector<u8> pad((size_t)bpr * lh);
            for (u32 y = 0; y < lh; y++) memcpy(&pad[(size_t)y * bpr], data + (size_t)lw * lh * 4 + (size_t)y * lw, lw);
            WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT; dst.texture = v.tex;
            WGPUTexelCopyBufferLayout lay = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT; lay.bytesPerRow = bpr; lay.rowsPerImage = lh;
            WGPUExtent3D ext = {lw, lh, 1};
            wgpuQueueWriteTexture(G.queue, &dst, pad.data(), pad.size(), &lay, &ext);
        }
        blit(0, id, 0, 0, 1, 1, false, has_cov != 0, false, true, c.view, has_cov ? v.view : G.dummy.view);
        drop_tex(c); if (v) drop_tex(v);
    } else if (s.kind == HSK_DEPTH) {
        end_pass();
        Tex t = make_tex(lw, lh, WGPUTextureFormat_R32Uint, WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst, "load depth");
        write_tex(t, data, lw, lh, 4);
        u32 offs[9];
        for (u32 b = 0; b < 9; b++) { u32 pu[4] = {b, 0, s.pw, s.ph}; offs[b] = push_uni(pu, 16); }
        WGPUBindGroupEntry e[2] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
        e[0].binding = 0; e[0].textureView = t.view;
        e[1].binding = 1; e[1].buffer = G.ubuf; e[1].size = 16;
        WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bd.layout = G.bgl_depth; bd.entryCount = 2; bd.entries = e;
        WGPUBindGroup bg = wgpuDeviceCreateBindGroup(G.dev, &bd);
        G.dead_bg.push_back(bg);
        WGPURenderPassDepthStencilAttachment da = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
        da.view = s.depth.view;
        da.depthLoadOp = WGPULoadOp_Clear; da.depthStoreOp = WGPUStoreOp_Store; da.depthClearValue = 1.0f;
        da.stencilLoadOp = WGPULoadOp_Clear; da.stencilStoreOp = WGPUStoreOp_Store; da.stencilClearValue = 0;
        WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        pd.depthStencilAttachment = &da;
        WGPURenderPassEncoder p = wgpuCommandEncoderBeginRenderPass(G.enc, &pd);
        wgpuRenderPassEncoderSetPipeline(p, G.depth_pipe);
        wgpuRenderPassEncoderSetBindGroup(p, 0, bg, 1, &offs[8]);
        wgpuRenderPassEncoderDraw(p, 3, 1, 0, 0);
        wgpuRenderPassEncoderSetStencilReference(p, 0xFF);
        for (u32 b = 0; b < 8; b++) {
            wgpuRenderPassEncoderSetPipeline(p, G.stencil_pipe[b]);
            wgpuRenderPassEncoderSetBindGroup(p, 0, bg, 1, &offs[b]);
            wgpuRenderPassEncoderDraw(p, 3, 1, 0, 0);
        }
        wgpuRenderPassEncoderEnd(p); wgpuRenderPassEncoderRelease(p);
        drop_tex(t);
    }
}
bool hwr_icon_build(const u32 *px, u32 w, u32 h, u32 s, std::vector<u32> &out);
// texture replacements (tools/texlab.py): a replacement that is still being
// loaded waits here; the original is shown until it is ready
struct ReplPending { u32 id; u64 hash; };
static std::vector<ReplPending> g_repl_pending;
static bool repl_upload(u32 id, u64 hash, u32 w, u32 h) {
    if (!w || !h || w > 8192 || h > 8192) return false;
    std::vector<u32> px((size_t)w * h);
    hwr_texrepl_copy(hash, px.data());
    Tex t = make_tex(w, h, WGPUTextureFormat_RGBA8Unorm, WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst, "replaced tex");
    write_tex(t, px.data(), w, h, 4);
    static int logged = 0;
    if (logged++ < 4) LOG("[hwr] texture %016llx replaced (%ux%u)", (unsigned long long)hash, w, h);
    auto old = G.texs.find(id);
    if (old != G.texs.end()) { forget_bind_groups(id); drop_tex(old->second); }
    G.texs[id] = t;
    return true;
}
static void repl_poll() {
    for (size_t i = 0; i < g_repl_pending.size();) {
        ReplPending r = g_repl_pending[i];
        u32 w = 0, h = 0;
        int q = G.texs.count(r.id) ? hwr_texrepl_query(r.hash, &w, &h) : 0;
        if (q == 1) { i++; continue; }
        if (q == 2) repl_upload(r.id, r.hash, w, h);
        g_repl_pending[i] = g_repl_pending.back(); g_repl_pending.pop_back();
    }
}

static void op_tex(const u32 *a, const u8 *data) {
    G.st.texs++; G.st.tex_bytes += (u64)a[1] * a[2] * 4;
    u32 id = a[0], w = a[1], h = a[2];
    const u64 hash = (u64)a[3] | (u64)a[4] << 32;
    {
        u32 rw = 0, rh = 0;
        int q = hwr_texrepl_query(hash, &rw, &rh);
        if (q == 2 && repl_upload(id, hash, rw, rh)) return;
        if (q == 1) g_repl_pending.push_back({id, hash});
    }
    // a button prompt icon: redrawn as keyboard keys (R3DS_HWR_BUTTONS=0 keeps the 3DS buttons)
    static bool keys = !getenv("R3DS_HWR_BUTTONS") || strcmp(getenv("R3DS_HWR_BUTTONS"), "0") != 0;
    std::vector<u32> hi;
    u32 s = std::clamp<u32>(g_hwr_req_scale.load(), 2, 6);
    static bool dbg = getenv("R3DS_HWR_ICONDBG") != nullptr;
    if (dbg && w <= 64 && h <= 64) LOG("[hwr] small texture %u: %ux%u", id, w, h);
    if (keys && w <= 64 && h <= 64 && hwr_icon_build((const u32 *)data, w, h, s, hi)) {
        Tex t = make_tex(w * s, h * s, WGPUTextureFormat_RGBA8Unorm, WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst, "button icon");
        write_tex(t, hi.data(), w * s, h * s, 4);
        auto old = G.texs.find(id);
        if (old != G.texs.end()) drop_tex(old->second);
        G.texs[id] = t;
        return;
    }
    Tex t = make_tex(w, h, WGPUTextureFormat_RGBA8Unorm, WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst, "tex");
    write_tex(t, data, w, h, 4);
    auto old = G.texs.find(id);
    if (old != G.texs.end()) drop_tex(old->second);
    G.texs[id] = t;
}
bool hwr_font_build(const u32 *src, u32 w, u32 h, const HwrSheet &S, u32 s, std::vector<u32> &out);
// a glyph sheet: redraw it at a higher resolution from a vector font
static void op_tex_font(const u32 *a, const u8 *data) {
    u32 id = a[0], w = a[1], h = a[2], s = a[3], f = a[4];
    HwrSheet S;
    S.cw = a[5]; S.ch = a[6]; S.cols = a[7]; S.cells = a[8];
    S.font = f < (u32)g_hwr_font_count ? &g_hwr_fonts[f] : nullptr;
    std::vector<u32> hi;
    double t0 = now_ms();
    if (!S.font || !hwr_font_build((const u32 *)data, w, h, S, s, hi)) {
        u32 plain[3] = {id, w, h};
        op_tex(plain, data);
        return;
    }
    static int logged = 0;
    if (logged++ < 8) LOG("[hwr] glyph sheet %u redrawn at %ux (%ux%u) in %.0f ms", id, s, w * s, h * s, now_ms() - t0);
    Tex t = make_tex(w * s, h * s, WGPUTextureFormat_RGBA8Unorm, WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst, "glyph sheet");
    write_tex(t, hi.data(), w * s, h * s, 4);
    auto old = G.texs.find(id);
    if (old != G.texs.end()) drop_tex(old->second);
    G.texs[id] = t;
    G.st.texs++;
}
static void op_tex_free(u32 id) {
    auto it = G.texs.find(id);
    if (it == G.texs.end()) return;
    forget_bind_groups(id);
    drop_tex(it->second);
    G.texs.erase(it);
}

static void op_draw(const HwrDraw &d, const float *verts) {
    G.st.draws++; G.st.verts += d.nverts;
    auto ci = G.surfs.find(d.color_id);
    if (ci == G.surfs.end() || !ci->second.color || !d.nverts) return;
    GSurf &c = ci->second;
    // a draw sampling its own target needs a snapshot of it
    u32 tid[3] = {d.tex_id[0], d.tex_id[1], d.tex_id[2]};
    WGPUTextureView snap = nullptr;
    for (int i = 0; i < 3; i++)
        if (tid[i] == d.color_id) {
            if (!snap) {
                end_pass();
                Tex t = make_tex(c.pw, c.ph, WGPUTextureFormat_RGBA8Unorm, WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst, "snapshot");
                WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT, dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
                src.texture = c.color.tex; dst.texture = t.tex;
                WGPUExtent3D ext = {c.pw, c.ph, 1};
                wgpuCommandEncoderCopyTextureToTexture(G.enc, &src, &dst, &ext);
                snap = t.view;
                G.dead_tex.push_back(t.tex); G.dead_view.push_back(t.view);
            }
        }
    begin_pass(d.color_id, d.depth_id);
    // scissor in physical pixels
    float s[4]; memcpy(s, d.scissor, 16);
    float rx = (float)c.pw / c.lw, ry = (float)c.ph / c.lh;
    int x0 = std::clamp((int)lroundf(s[0] * rx), 0, (int)c.pw), y0 = std::clamp((int)lroundf(s[1] * ry), 0, (int)c.ph);
    int x1 = std::clamp((int)lroundf(s[2] * rx), 0, (int)c.pw), y1 = std::clamp((int)lroundf(s[3] * ry), 0, (int)c.ph);
    if (x1 <= x0 || y1 <= y0) return;
    WGPUBindGroup tg;
    if (snap) {
        WGPUBindGroupEntry e[3] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
        for (int i = 0; i < 3; i++) { e[i].binding = i; e[i].textureView = tid[i] == d.color_id ? snap : tex_view(tid[i]); }
        WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bd.layout = G.bgl_tex; bd.entryCount = 3; bd.entries = e;
        tg = wgpuDeviceCreateBindGroup(G.dev, &bd);
        G.dead_bg.push_back(tg);
    } else {
        Tri3 key{tid[0], tid[1], tid[2]};
        auto it = G.bg_tex.find(key);
        if (it == G.bg_tex.end()) {
            WGPUBindGroupEntry e[3] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
            for (int i = 0; i < 3; i++) { e[i].binding = i; e[i].textureView = tex_view(tid[i]); }
            WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
            bd.layout = G.bgl_tex; bd.entryCount = 3; bd.entries = e;
            it = G.bg_tex.emplace(key, wgpuDeviceCreateBindGroup(G.dev, &bd)).first;
        }
        tg = it->second;
    }
    u32 uoff = push_uni(d.u, sizeof d.u);
    u64 voff = G.vdata.size();
    G.vdata.insert(G.vdata.end(), (const u8 *)verts, (const u8 *)verts + (size_t)d.nverts * HWR_VFLOATS * 4);
    WGPURenderPassEncoder p = G.pass;
    wgpuRenderPassEncoderSetPipeline(p, pica_pipe(d.pipe));
    wgpuRenderPassEncoderSetBindGroup(p, 0, G.bg_uni, 1, &uoff);
    wgpuRenderPassEncoderSetBindGroup(p, 1, tg, 0, nullptr);
    wgpuRenderPassEncoderSetVertexBuffer(p, 0, G.vbuf, voff, (u64)d.nverts * HWR_VFLOATS * 4);
    wgpuRenderPassEncoderSetScissorRect(p, x0, y0, x1 - x0, y1 - y0);
    wgpuRenderPassEncoderSetStencilReference(p, d.stencil_ref);
    u32 k = d.blend_const, b = d.pipe.blend;
    bool const_alpha = ((b >> 16) & 15) >= 12 || ((b >> 20) & 15) >= 12;
    bool const_color = ((b >> 16) & 15) == 10 || ((b >> 16) & 15) == 11 || ((b >> 20) & 15) == 10 || ((b >> 20) & 15) == 11;
    WGPUColor bc = {(k & 0xFF) / 255.0, ((k >> 8) & 0xFF) / 255.0, ((k >> 16) & 0xFF) / 255.0, (k >> 24) / 255.0};
    if (const_alpha && !const_color) bc.r = bc.g = bc.b = bc.a;
    wgpuRenderPassEncoderSetBlendConstant(p, &bc);
    wgpuRenderPassEncoderDraw(p, d.nverts, 1, 0, 0);
}

// size the per-submission buffers for everything in chunks
static void reserve_buffers(const std::vector<std::vector<u8>> &chunks, u64 extra_uni) {
    u64 vb = 0, ub = extra_uni + 64 * 256;
    for (auto &c : chunks)
        for (size_t at = 0; at + 8 <= c.size();) {
            u32 op, len; memcpy(&op, &c[at], 4); memcpy(&len, &c[at + 4], 4);
            if (op == HOP_DRAW) { const HwrDraw *d = (const HwrDraw *)&c[at + 8]; vb += (u64)d->nverts * HWR_VFLOATS * 4; ub += align256(sizeof d->u); }
            else if (op == HOP_BLIT || op == HOP_SURF_LOAD || op == HOP_LCD) ub += 256 * 10;
            at += 8 + len;
        }
    auto grow = [](WGPUBuffer &buf, u64 &cap, u64 need, WGPUBufferUsage usage, const char *label) {
        if (buf && cap >= need) return false;
        if (buf) wgpuBufferRelease(buf);
        cap = std::max<u64>(need + need / 2, 1 << 20);
        WGPUBufferDescriptor d = WGPU_BUFFER_DESCRIPTOR_INIT;
        d.label = sv(label); d.usage = usage | WGPUBufferUsage_CopyDst; d.size = cap;
        buf = wgpuDeviceCreateBuffer(G.dev, &d);
        return true;
    };
    grow(G.vbuf, G.vcap, vb, WGPUBufferUsage_Vertex, "vertices");
    if (grow(G.ubuf, G.ucap, ub, WGPUBufferUsage_Uniform, "uniforms") || !G.bg_uni) {
        if (G.bg_uni) wgpuBindGroupRelease(G.bg_uni);
        WGPUBindGroupEntry e = WGPU_BIND_GROUP_ENTRY_INIT;
        e.binding = 0; e.buffer = G.ubuf; e.size = HWR_UVEC4 * 16;
        WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bd.layout = G.bgl_uni; bd.entryCount = 1; bd.entries = &e;
        G.bg_uni = wgpuDeviceCreateBindGroup(G.dev, &bd);
    }
}

static void replay(const std::vector<u8> &c) {
    for (size_t at = 0; at + 8 <= c.size();) {
        u32 op, len; memcpy(&op, &c[at], 4); memcpy(&len, &c[at + 4], 4);
        const u8 *p = &c[at + 8];
        const u32 *a = (const u32 *)p;
        switch (op) {
        case HOP_RESET: reset_all(); break;
        case HOP_SURF_NEW: op_surf_new(a); break;
        case HOP_SURF_FREE: op_surf_free(a[0]); break;
        case HOP_SURF_LOAD: op_load(a, p + 16); break;
        case HOP_SURF_CLEAR: op_clear(a[0], a[1], a[2] != 0); break;
        case HOP_TEX: op_tex(a, p + 20); break;
        case HOP_TEX_FONT: op_tex_font(a, p + 36); break;
        case HOP_TEX_FREE: op_tex_free(a[0]); break;
        case HOP_DRAW: { const HwrDraw *d = (const HwrDraw *)p; op_draw(*d, (const float *)(p + sizeof(HwrDraw))); break; }
        case HOP_BLIT: {
            G.st.blits++;
            HwrBlit b; memcpy(&b, p, sizeof b);
            auto dst = G.surfs.find(b.dst_id);
            if (b.src_id != b.dst_id && dst != G.surfs.end())
                blit(b.src_id, b.dst_id, (float)b.sx, (float)b.sy, (float)b.sw, (float)b.sh, b.flip, b.cov, b.noalpha, false);
            break;
        }
        case HOP_LCD: {
            HwrLcd l; memcpy(&l, p, sizeof l);
            auto &v = G.lcds[l.va];
            v.push_back({l.id, l.hash, l.bytes});
            if (v.size() > 4) v.erase(v.begin());
            break;
        }
        default: break;
        }
        at += 8 + len;
    }
}

// ------------------------------------------------------------ composition
static bool lcd_src(const HwrScreen &sc, u32 rows, Src &s) {
    if (!sc.valid) return false;
    auto it = G.lcds.find(sc.va);
    if (it == G.lcds.end()) return false;
    for (auto l = it->second.rbegin(); l != it->second.rend(); ++l) {
        if (l->hash != sc.hash || l->bytes != sc.bytes) continue;
        auto si = G.surfs.find(l->id);
        if (si == G.surfs.end() || si->second.lh != rows) continue;
        GSurf &g = si->second;
        // LCD image: 240-texel rows, one per screen column; its aspect is
        // rows * (ph/lh) : 240 * (pw/lw)
        s = {g.color.view, (float)g.pw, (float)g.ph, true, 0, 0, 1, 1, ((float)g.ph / g.lh * rows) / ((float)g.pw / g.lw * 240.0f)};
        Held &h = g_held[rows == 400 ? 0 : 1];
        drop_tex(h.tex); h.id = l->id;   // live again: a held copy is no longer needed
        h.src = s;
        return true;
    }
    return false;
}
static Src soft_src(bool top) {
    if (top) return {G.frame1x.view, 400, 480, false, 0, 0, 1, 0.5f, 400.0f / 240.0f};
    return {G.frame1x.view, 400, 480, false, 40.0f / 400, 0.5f, 320.0f / 400, 0.5f, 320.0f / 240.0f};
}
struct Quad { float x, y, w, h; };   // pixels, y down
static void draw_quad(WGPURenderPassEncoder p, WGPUTextureFormat fmt, float W, float H, const Src &s, Quad q, int filter,
                      float alpha, int mode, bool blank, float blur, float dim, bool blend) {
    // uv = M * (lx, ly) + t, local (0..1) over the quad
    float m0[4], m1[4];
    if (s.rotated) {   // screen x -> texture v (rows), screen y -> 1 - u
        m0[0] = 0; m0[1] = 1; m0[2] = -1; m0[3] = 0; m1[0] = 1; m1[1] = 0;
    } else {
        m0[0] = s.uw; m0[1] = 0; m0[2] = 0; m0[3] = s.vh; m1[0] = s.u0; m1[1] = s.v0;
    }
    m1[2] = s.tw; m1[3] = s.th;
    // texels per screen pixel along the texture's axes, for the sharp filter
    float qx = s.rotated ? q.h / s.tw : q.w / (s.tw * s.uw), qy = s.rotated ? q.w / s.th : q.h / (s.th * s.vh);
    float cu[20] = {q.x / W * 2 - 1, 1 - q.y / H * 2, q.w / W * 2, q.h / H * 2,
                    m0[0], m0[1], m0[2], m0[3], m1[0], m1[1], m1[2], m1[3],
                    alpha, blur, dim, (float)filter,
                    (float)mode, blank ? 1.0f : 0.0f, std::max(1.0f, qx), std::max(1.0f, qy)};
    u32 off = push_uni(cu, sizeof cu);
    WGPUBindGroupEntry e[3] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
    e[0].binding = 0; e[0].buffer = G.ubuf; e[0].size = 80;
    e[1].binding = 1; e[1].sampler = filter == 0 && blur == 0 ? G.samp_lin : G.samp_lin;
    e[2].binding = 2; e[2].textureView = s.view;
    WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bd.layout = G.bgl_comp; bd.entryCount = 3; bd.entries = e;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(G.dev, &bd);
    G.dead_bg.push_back(bg);
    wgpuRenderPassEncoderSetPipeline(p, comp_pipe(blend, fmt));
    wgpuRenderPassEncoderSetBindGroup(p, 0, bg, 1, &off);
    wgpuRenderPassEncoderDraw(p, 4, 1, 0, 0);
}
static Quad fit(Quad a, float aspect) {
    float w = a.w, h = w / aspect;
    if (h > a.h) { h = a.h; w = h * aspect; }
    return {a.x + (a.w - w) / 2, a.y + (a.h - h) / 2, w, h};
}

// lay out and draw both screens into target (W x H pixels)
static void compose(WGPUTextureView target, WGPUTextureFormat fmt, u32 W, u32 H, const HwrView &v, bool anim_step) {
    Src top, bot;
    bool gtop = lcd_src(G.info.top, 400, top), gbot = lcd_src(G.info.bot, 320, bot);
    auto held = [](int i, Src &s) {
        Held &h = g_held[i];
        if (!h.tex) return false;
        if (now_ms() - h.t > 600) { drop_tex(h.tex); h.id = 0; return false; }
        s = h.src;
        return true;
    };
    bool htop = !gtop && held(0, top), hbot = !gbot && held(1, bot);
    if (!gtop && !htop) top = soft_src(true);
    if (!gbot && !hbot) bot = soft_src(false);
    gtop = gtop || htop; gbot = gbot || hbot;
    float wide = 1.0f; { u32 b = g_hwr_req_wide.load(); memcpy(&wide, &b, 4); }
    float top_aspect = v.aspect == HWA_STRETCH ? 16.0f / 9.0f : v.aspect == HWA_NATIVE ? 5.0f / 3.0f : 5.0f / 3.0f * wide;
    Quad full{0, 0, (float)W, (float)H};
    Quad tq, bq{0, 0, 0, 0};
    bool bon = true, premul = false;
    float balpha = 1.0f;
    if (v.layout == HWL_CLASSIC) {
        Quad r = fit(full, 400.0f / 480.0f);
        float s = r.w / 400;
        tq = {r.x, r.y, r.w, 240 * s}; bq = {r.x + 40 * s, r.y + 240 * s, 320 * s, 240 * s};
    } else if (v.layout == HWL_SIDE) {
        Quad r = fit(full, (400 + 320 * 0.6f) / 240);
        float s = r.h / 240, bs = s * 0.6f;
        tq = {r.x, r.y, 400 * s, 240 * s}; bq = {r.x + 400 * s, r.y + 240 * s - 240 * bs, 320 * bs, 240 * bs};
    } else if (v.layout == HWL_LARGE) {   // Citra's large screen: big top, small bottom beside it
        const float ta = 5.0f / 3.0f, k = 0.5f;   // bottom at half the top's height
        float th = std::min((float)H, W / (ta + k * 4.0f / 3.0f));
        float tw = th * ta, bw = th * k * 4.0f / 3.0f, bh = th * k;
        float x = (W - tw - bw) / 2, y = (H - th) / 2;
        tq = {x, y, tw, th}; bq = {x + tw, y + th - bh, bw, bh};
    }
    // does the bottom screen show a menu? (same guess as the automatic UI / minimap mode)
    {
        int guess = G.info.masked && G.info.cover > 0.72f ? HWB_MINIMAP : HWB_UI;
        if (guess != G.bmode_cur) { if (anim_step && ++G.bmode_votes > 20) { G.bmode_cur = guess; G.bmode_votes = 0; } } else G.bmode_votes = 0;
    }
    G.out.bottom_focus = G.info.masked && G.bmode_cur == HWB_UI;
    if (v.layout == HWL_REMASTER || v.layout == HWL_TOP) {
        tq = fit(full, top_aspect);
        float s = tq.h / 240;
        int want = v.bottom_mode == HWB_AUTO ? G.bmode_cur : v.bottom_mode;
        premul = v.remove_bg && G.info.masked;
        float m = tq.h * 0.02f;
        if (v.show_full) {
            float bs = s * std::max(v.ui_scale, 0.9f);
            bq = {tq.x + (tq.w - 320 * bs) / 2, tq.y + tq.h - 240 * bs, 320 * bs, 240 * bs};
            premul = false;
        } else if (v.layout == HWL_TOP) {
            bon = false;
        } else if (want == HWB_MINIMAP) {
            float bs = s * v.mini_scale;
            float x = (v.mini_corner & 1) ? tq.x + m : tq.x + tq.w - 320 * bs - m;
            float y = (v.mini_corner & 2) ? tq.y + m : tq.y + tq.h - 240 * bs - m;
            bq = {x, y, 320 * bs, 240 * bs}; balpha = v.mini_alpha;
        } else {
            float bs = s * v.ui_scale;
            float bw = 320 * bs, bh = 240 * bs;
            // 0 bottom right, 1 bottom centre, 2 bottom left, 3 top right, 4 top left
            bool left = v.ui_anchor == 2 || v.ui_anchor == 4, top = v.ui_anchor >= 3;
            float x = v.ui_anchor == 1 ? tq.x + (tq.w - bw) / 2 : left ? tq.x + m : tq.x + tq.w - bw - m;
            float y = top ? tq.y + m : tq.y + tq.h - bh - (v.ui_anchor == 1 ? 0 : m);
            bq = {x, y, bw, bh}; balpha = v.ui_alpha;
        }
        if (bon && G.have_cur && !v.show_full) {
            if (anim_step) {
                const float k = 0.25f;
                G.cur[0] += (bq.x - G.cur[0]) * k; G.cur[1] += (bq.y - G.cur[1]) * k;
                G.cur[2] += (bq.w - G.cur[2]) * k; G.cur[3] += (bq.h - G.cur[3]) * k;
            }
            G.out.animating = fabsf(G.cur[0] - bq.x) > 0.5f || fabsf(G.cur[1] - bq.y) > 0.5f || fabsf(G.cur[2] - bq.w) > 0.5f;
            bq = {G.cur[0], G.cur[1], G.cur[2], G.cur[3]};
        } else if (bon) { G.cur[0] = bq.x; G.cur[1] = bq.y; G.cur[2] = bq.w; G.cur[3] = bq.h; G.out.animating = false; }
        G.have_cur = bon && !v.show_full;
    }
    // the classic / side layouts show the software frame's own aspect for the top
    if (v.layout == HWL_CLASSIC || v.layout == HWL_SIDE) {}

    // side bars: blurred top screen, drawn small then stretched
    bool bars = v.side_blur && v.layout != HWL_CLASSIC && (tq.x > 1 || tq.y > 1);
    if (bars) {
        WGPURenderPassColorAttachment ca = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        ca.view = G.blur.view; ca.loadOp = WGPULoadOp_Clear; ca.storeOp = WGPUStoreOp_Store; ca.clearValue = {0, 0, 0, 1};
        WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        pd.colorAttachmentCount = 1; pd.colorAttachments = &ca;
        WGPURenderPassEncoder p = wgpuCommandEncoderBeginRenderPass(G.enc, &pd);
        float step = std::max(1.0f, (top.rotated ? top.th : top.tw) / 160.0f) * 1.5f;
        draw_quad(p, WGPUTextureFormat_RGBA8Unorm, 160, 96, top, {0, 0, 160, 96}, 1, 1.0f, 0, false, step, 0.45f, false);
        wgpuRenderPassEncoderEnd(p); wgpuRenderPassEncoderRelease(p);
    }
    WGPURenderPassColorAttachment ca = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    ca.view = target; ca.loadOp = WGPULoadOp_Clear; ca.storeOp = WGPUStoreOp_Store; ca.clearValue = {0, 0, 0, 1};
    WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    pd.colorAttachmentCount = 1; pd.colorAttachments = &ca;
    WGPURenderPassEncoder p = wgpuCommandEncoderBeginRenderPass(G.enc, &pd);
    if (bars) {
        Quad cover{0, 0, (float)W, W / top_aspect};
        if (cover.h < H) { cover.h = (float)H; cover.w = H * top_aspect; }
        cover.x = (W - cover.w) / 2; cover.y = (H - cover.h) / 2;
        Src b{G.blur.view, 160, 96, false, 0, 0, 1, 1, 160.0f / 96};
        draw_quad(p, fmt, W, H, b, cover, 1, 1.0f, 0, false, 0, 1, false);
    }
    draw_quad(p, fmt, W, H, top, tq, v.filter, 1.0f, 0, false, 0, 1, false);
    if (bon) draw_quad(p, fmt, W, H, bot, bq, v.filter, balpha, premul ? 1 : 0, premul && G.info.blank, 0, 1, true);
    wgpuRenderPassEncoderEnd(p); wgpuRenderPassEncoderRelease(p);
    G.out.bx = bq.x; G.out.by = bq.y; G.out.bw = bq.w; G.out.bh = bq.h; G.out.bottom_on = bon; G.out.hit = premul;
    G.out.top_gpu = gtop; G.out.bot_gpu = gbot;
    if (!gtop) { G.st.soft_top++; g_hwr_dbg[HD_EXE_SOFT_TOP]++; }
    if (!gbot && bon) { G.st.soft_bot++; g_hwr_dbg[HD_EXE_SOFT_BOT]++; }
}

// ------------------------------------------------------------ frame
static void choose_scale(u32 W, u32 H, const HwrView &v) {
    // resolution multiplier from the displayed size of the top screen
    float top_h = std::min((float)H, W / (5.0f / 3.0f));
    u32 s = v.res > 0 ? (u32)v.res : (u32)ceilf(top_h / 240.0f - 0.15f);
    s = std::clamp(s, 1u, (u32)std::clamp(v.max_res, 1, 8));
    float wide = 1.0f;
    if (v.aspect == HWA_WIDE) wide = 16.0f / 15.0f;
    else if (v.aspect == HWA_AUTO) wide = std::clamp(((float)W / H) / (5.0f / 3.0f), 1.0f, 1.4f);
    if (v.layout != HWL_REMASTER && v.layout != HWL_TOP) wide = 1.0f;
    if (fabsf(wide - G.want_wide) < 0.004f) wide = G.want_wide;   // ignore jitter while resizing
    // follow a changed size only once it has settled (resizing a window)
    if (s != G.want_scale) { if (++G.scale_votes > 20 || !g_hwr_req_scale) { G.want_scale = s; G.scale_votes = 0; } } else G.scale_votes = 0;
    if (fabsf(wide - G.want_wide) > 1e-3f) { if (++G.wide_votes > 20) { G.want_wide = wide; G.wide_votes = 0; } } else G.wide_votes = 0;
    if (G.want_scale) g_hwr_req_scale = G.want_scale;
    u32 b; memcpy(&b, &G.want_wide, 4); g_hwr_req_wide = b;
}

// ---- frame pacing
// Each frame info carries the time of its vblank. The one shown is the newest
// whose vblank is at least D old: D is the observed latency from vblank to
// the frame reaching the executor (the GPU thread runs behind, unevenly), so
// frames come out on the vblank rhythm and a 30 fps game shows each image for
// exactly two 60 Hz refreshes, instead of one or three depending on when the
// GPU thread happened to finish. When the display refreshes at the vblank
// rate, D is also placed so the choice falls mid-way between vblanks, away
// from the timing noise of either clock. R3DS_PACE=0: show the newest frame.
static struct Pace {
    std::deque<HwrFrameInfo> pend;
    std::deque<std::pair<u64, double>> lat;   // (arrival, latency ms), last 2 s
    double period = 1000.0 / 60, raf = 0, D = 0, cx = 0, cy = 0;
    u64 last_now = 0, last_t = 0;
    int shrink = 0;
} P;
static bool pace(const std::vector<HwrFrameInfo> &hist, u64 now) {
    const double TAU = 6.283185307179586;
    for (const HwrFrameInfo &h : hist) {
        if (P.last_t && h.t_ns > P.last_t) {
            double d = (h.t_ns - P.last_t) / 1e6;
            if (d > 5 && d < 40) P.period += (d - P.period) * 0.05;
        }
        P.last_t = h.t_ns;
        P.pend.push_back(h);
        P.lat.push_back({now, now > h.t_ns ? (now - h.t_ns) / 1e6 : 0.0});
    }
    while (!P.lat.empty() && now - P.lat.front().first > 2000000000ull) P.lat.pop_front();
    while (P.pend.size() > 12) P.pend.pop_front();
    if (P.last_now) { double d = (now - P.last_now) / 1e6; if (d > 2 && d < 60) P.raf = P.raf ? P.raf + (d - P.raf) * 0.05 : d; }
    P.last_now = now;
    double want = 0;
    for (auto &l : P.lat) want = std::max(want, l.second);
    want = std::min(want + 1.0, 80.0);
    double target = want;
    if (P.last_t && P.raf && fabs(P.raf - P.period) < P.period * 0.1) {   // display at the vblank rate
        double ph = fmod((now - P.last_t) / 1e6, P.period) / P.period * TAU;
        P.cx += (cos(ph) - P.cx) * 0.05; P.cy += (sin(ph) - P.cy) * 0.05;
        double psi = fmod(atan2(P.cy, P.cx) / TAU * P.period + P.period, P.period);
        target = want + fmod(fmod(psi - want - P.period / 2, P.period) + P.period, P.period);
    }
    // grow at once (a late frame); shrink only when it has been too long for a while
    if (target > P.D) { P.D = target; P.shrink = 0; }
    else if (P.D - target > 4 && ++P.shrink > 120) { P.D = target; P.shrink = 0; }
    else if (P.D - target <= 4) P.shrink = 0;
    int pick = -1;
    for (int i = 0; i < (int)P.pend.size(); i++)
        if (P.pend[i].t_ns + (u64)(P.D * 1e6) <= now) pick = i;
    g_hwr_dbg[HD_PACE_MS] = P.D;
    if (pick < 0) return false;
    g_hwr_dbg[HD_PACE_NEW]++;
    G.info = P.pend[pick];
    P.pend.erase(P.pend.begin(), P.pend.begin() + pick + 1);
    return true;
}

static bool process(bool *new_frame) {
    std::vector<std::vector<u8>> chunks;
    static int pacing = getenv("R3DS_PACE") ? atoi(getenv("R3DS_PACE")) : 1;
    std::vector<HwrFrameInfo> hist;
    if (pacing) display_frame_history(hist);   // first: the images of every frame read here are queued already
    hwr_take(chunks);
    if (!g_repl_pending.empty()) repl_poll();
    HwrFrameInfo latest;
    bool nf = display_get_frame_info(G.swframe, latest);
    if (!pacing && nf) G.info = latest;
    *new_frame = nf;
    double t0 = now_ms();
    G.enc = wgpuDeviceCreateCommandEncoder(G.dev, nullptr);
    G.vdata.clear(); G.udata.clear();
    reserve_buffers(chunks, 64 * 256);
    for (auto &c : chunks) { replay(c); G.st.chunks++; G.st.bytes += c.size(); }
    end_pass();
    g_hwr_alive_ns = now_ns();   // a long replay is not a stalled executor
    G.st.t_replay += now_ms() - t0; g_hwr_dbg[HD_EXE_REPLAY_MS] += now_ms() - t0;
    if (nf) {
        G.have_frame = true;
        std::vector<u32> px(G.swframe);
        write_tex(G.frame1x, px.data(), 400, 480, 4);
    }
    if (pacing) *new_frame = pace(hist, now_ns());
    return true;
}
static void submit() {
    double t0 = now_ms();
    if (!G.vdata.empty()) wgpuQueueWriteBuffer(G.queue, G.vbuf, 0, G.vdata.data(), G.vdata.size());
    if (!G.udata.empty()) wgpuQueueWriteBuffer(G.queue, G.ubuf, 0, G.udata.data(), G.udata.size());
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(G.enc, nullptr);
    wgpuQueueSubmit(G.queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(G.enc);
    G.enc = nullptr;
    for (auto b : G.dead_bg) wgpuBindGroupRelease(b);
    for (auto v : G.dead_view) wgpuTextureViewRelease(v);
    for (auto t : G.dead_tex) { wgpuTextureDestroy(t); wgpuTextureRelease(t); }
    G.dead_bg.clear(); G.dead_view.clear(); G.dead_tex.clear();
    G.st.t_submit += now_ms() - t0; g_hwr_dbg[HD_EXE_SUBMIT_MS] += now_ms() - t0;
    static int stats = getenv("R3DS_HWR_STATS") ? 1 : 0;
    static double last = now_ms();
    if (stats && now_ms() - last >= 1000) {
        last = now_ms();
        auto &S = G.st;
        LOG("[hwr] /s: frames %llu draws %llu verts %llu loads %llu tex %llu (%.1f MB) blits %llu chunks %llu (%.1f MB) new pipelines %llu software-image frames top %llu bottom %llu | cpu replay %.1f ms submit %.1f ms present %.1f ms",
            (unsigned long long)S.frames, (unsigned long long)S.draws, (unsigned long long)S.verts, (unsigned long long)S.loads,
            (unsigned long long)S.texs, S.tex_bytes / 1e6, (unsigned long long)S.blits, (unsigned long long)S.chunks, S.bytes / 1e6,
            (unsigned long long)S.pipes, (unsigned long long)S.soft_top, (unsigned long long)S.soft_bot, S.t_replay, S.t_submit, S.t_present);
        S = {};
    }
}

// ------------------------------------------------------------ device setup
static void on_error(WGPUDevice const *, WGPUErrorType type, WGPUStringView msg, void *, void *) {
    if (G.errors++ < 20) LOG("[hwr] WebGPU error %d: %s", (int)type, svs(msg).c_str());
}
static void on_lost(WGPUDevice const *, WGPUDeviceLostReason reason, WGPUStringView msg, void *, void *) {
    if (reason != WGPUDeviceLostReason_Destroyed) LOG("[hwr] device lost (%d): %s", (int)reason, svs(msg).c_str());
    G.state = 3; g_hwr_on = false;
}
static void on_device(WGPURequestDeviceStatus st, WGPUDevice dev, WGPUStringView msg, void *, void *) {
    if (st != WGPURequestDeviceStatus_Success) { LOG("[hwr] no WebGPU device: %s", svs(msg).c_str()); G.state = 3; return; }
    G.dev = dev;
    G.queue = wgpuDeviceGetQueue(dev);
    create_pipelines();
    G.state = 2;
    g_hwr_alive_ns = now_ns();
    g_hwr_on = true;
    LOG("[hwr] WebGPU renderer ready");
}
static void on_adapter(WGPURequestAdapterStatus st, WGPUAdapter ad, WGPUStringView msg, void *, void *) {
    if (st != WGPURequestAdapterStatus_Success) { LOG("[hwr] no WebGPU adapter: %s", svs(msg).c_str()); G.state = 3; return; }
    G.adapter = ad;
    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    if (wgpuAdapterGetInfo(ad, &info) == WGPUStatus_Success) {
        LOG("[hwr] adapter: %s (%s)", svs(info.device).c_str(), svs(info.description).c_str());
        wgpuAdapterInfoFreeMembers(info);
    }
    WGPULimits lim = WGPU_LIMITS_INIT;
    wgpuAdapterGetLimits(ad, &lim);
    WGPULimits req = WGPU_LIMITS_INIT;
    req.maxTextureDimension2D = std::min<u32>(lim.maxTextureDimension2D, 8192);
    WGPUDeviceDescriptor dd = WGPU_DEVICE_DESCRIPTOR_INIT;
    dd.requiredLimits = &req;
    dd.uncapturedErrorCallbackInfo.callback = on_error;
    dd.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    dd.deviceLostCallbackInfo.callback = on_lost;
    WGPURequestDeviceCallbackInfo cb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous; cb.callback = on_device;
    wgpuAdapterRequestDevice(ad, &dd, cb);
}

bool hwr_gpu_start(const HwrSurfaceTarget &t) {
    if (G.state) return G.state != 3;
    G.state = 1;
    G.inst = wgpuCreateInstance(nullptr);
    if (!G.inst) { G.state = 3; return false; }
    if (t.kind != HWT_NONE) {
        WGPUSurfaceDescriptor sd = WGPU_SURFACE_DESCRIPTOR_INIT;
#ifdef __EMSCRIPTEN__
        WGPUEmscriptenSurfaceSourceCanvasHTMLSelector src = WGPU_EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR_INIT;
        src.selector = sv(t.canvas);
        sd.nextInChain = &src.chain;
#else
        WGPUSurfaceSourceMetalLayer ml = WGPU_SURFACE_SOURCE_METAL_LAYER_INIT;
        WGPUSurfaceSourceWindowsHWND wh = WGPU_SURFACE_SOURCE_WINDOWS_HWND_INIT;
        WGPUSurfaceSourceXlibWindow xw = WGPU_SURFACE_SOURCE_XLIB_WINDOW_INIT;
        WGPUSurfaceSourceWaylandSurface wl = WGPU_SURFACE_SOURCE_WAYLAND_SURFACE_INIT;
        if (t.kind == HWT_METAL) { ml.layer = t.a; sd.nextInChain = &ml.chain; }
        else if (t.kind == HWT_WIN32) { wh.hinstance = t.a; wh.hwnd = t.b; sd.nextInChain = &wh.chain; }
        else if (t.kind == HWT_X11) { xw.display = t.a; xw.window = (u64)(uintptr_t)t.b; sd.nextInChain = &xw.chain; }
        else if (t.kind == HWT_WAYLAND) { wl.display = t.a; wl.surface = t.b; sd.nextInChain = &wl.chain; }
#endif
        G.surface = wgpuInstanceCreateSurface(G.inst, &sd);
        if (!G.surface) { LOG("[hwr] cannot create a WebGPU surface"); G.state = 3; return false; }
    }
    WGPURequestAdapterOptions ao = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    ao.powerPreference = WGPUPowerPreference_HighPerformance;
    ao.compatibleSurface = G.surface;
    WGPURequestAdapterCallbackInfo cb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous; cb.callback = on_adapter;
    wgpuInstanceRequestAdapter(G.inst, &ao, cb);
#ifndef __EMSCRIPTEN__
    for (int i = 0; i < 2000 && G.state == 1; i++) { wgpuInstanceProcessEvents(G.inst); std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
#endif
    return G.state != 3;
}
int hwr_gpu_state() { return G.state; }

static bool configure(u32 w, u32 h) {
    if (!G.surface || !w || !h) return false;
    if (w == G.sw && h == G.sh) return true;
    WGPUSurfaceCapabilities caps = WGPU_SURFACE_CAPABILITIES_INIT;
    wgpuSurfaceGetCapabilities(G.surface, G.adapter, &caps);
    G.sfmt = caps.formatCount ? caps.formats[0] : WGPUTextureFormat_BGRA8Unorm;
    for (size_t i = 0; i < caps.formatCount; i++)
        if (caps.formats[i] == WGPUTextureFormat_BGRA8Unorm || caps.formats[i] == WGPUTextureFormat_RGBA8Unorm) { G.sfmt = caps.formats[i]; break; }
    wgpuSurfaceCapabilitiesFreeMembers(caps);
    WGPUSurfaceConfiguration c = WGPU_SURFACE_CONFIGURATION_INIT;
    c.device = G.dev; c.format = G.sfmt; c.usage = WGPUTextureUsage_RenderAttachment;
    c.width = w; c.height = h; c.alphaMode = WGPUCompositeAlphaMode_Opaque; c.presentMode = WGPUPresentMode_Fifo;
    wgpuSurfaceConfigure(G.surface, &c);
    G.sw = w; G.sh = h;
    return true;
}

bool hwr_gpu_frame(u32 W, u32 H, const HwrView &v, HwrOut &out, bool force_present) {
    if (G.state != 2) { out.ready = false; return false; }
#ifndef __EMSCRIPTEN__
    wgpuInstanceProcessEvents(G.inst);
#endif
    choose_scale(W, H, v);
    bool nf = false;
    process(&nf);
    static std::string last_view;
    std::string vk((const char *)&v, sizeof v);
    bool redraw = nf || force_present || G.out.animating || vk != last_view || W != G.sw || H != G.sh;
    last_view = vk;
    bool shown = false;
    if (redraw && G.have_frame && configure(W, H)) {
        WGPUSurfaceTexture st = WGPU_SURFACE_TEXTURE_INIT;
        double tp = now_ms();
        wgpuSurfaceGetCurrentTexture(G.surface, &st);
        G.st.t_present += now_ms() - tp; g_hwr_dbg[HD_EXE_PRESENT_MS] += now_ms() - tp;
        if (st.status == WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal || st.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
            WGPUTextureView tv = wgpuTextureCreateView(st.texture, nullptr);
            compose(tv, G.sfmt, W, H, v, true);
            submit();
            wgpuTextureViewRelease(tv);
#ifndef __EMSCRIPTEN__
            tp = now_ms();
            wgpuSurfacePresent(G.surface);
            G.st.t_present += now_ms() - tp;
#endif
            wgpuTextureRelease(st.texture);
            shown = true;
        } else {
            if (st.texture) wgpuTextureRelease(st.texture);
            G.sw = G.sh = 0;   // reconfigure next time
            submit();
        }
    } else submit();
    if (nf) { G.frames_shown++; G.st.frames++; g_hwr_dbg[HD_EXE_FRAMES]++; }
    G.out.ready = true;
    out = G.out;
    return shown && nf;
}

// copy of the software frame the executor last showed (touch hit tests)
u32 hwr_gpu_soft_pixel(int x, int y) {
    if (x < 0 || y < 0 || x >= 400 || y >= 480 || G.swframe.size() < 400 * 480) return 0;
    return G.swframe[y * 400 + x];
}

#ifdef R3DS_WGPU_NATIVE
// ------------------------------------------------------------ headless test (R3DS_HWR_TEST=dir)
// Renders the composition off screen and reads back PNGs, plus the GPU's
// top / bottom LCD images and their difference to the software frame.
static void png_write(const std::string &path, const u8 *rgba, u32 W, u32 H) {
    std::vector<u8> raw; raw.reserve((W * 4 + 1) * H);
    for (u32 y = 0; y < H; y++) { raw.push_back(0); raw.insert(raw.end(), rgba + (size_t)y * W * 4, rgba + (size_t)(y + 1) * W * 4); }
    std::vector<u8> z = {0x78, 0x01};
    u32 a = 1, b = 0;
    for (u8 v : raw) { a = (a + v) % 65521; b = (b + a) % 65521; }
    for (size_t off = 0; off < raw.size(); off += 65535) {
        u32 n = (u32)std::min<size_t>(65535, raw.size() - off);
        z.push_back(off + n == raw.size() ? 1 : 0);
        z.push_back(n & 0xFF); z.push_back(n >> 8); z.push_back(~n & 0xFF); z.push_back((~n >> 8) & 0xFF);
        z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
    }
    u32 ad = (b << 16) | a;
    z.push_back(ad >> 24); z.push_back(ad >> 16); z.push_back(ad >> 8); z.push_back(ad);
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return;
    auto crc = [](const u8 *d, size_t n) { static u32 t[256]; if (!t[1]) for (u32 k = 0; k < 256; k++) { u32 c = k; for (int i = 0; i < 8; i++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1))); t[k] = c; }
                                          u32 c = ~0u; for (size_t i = 0; i < n; i++) c = t[(c ^ d[i]) & 0xFF] ^ (c >> 8); return ~c; };
    auto be = [&](u32 v) { u8 x[4] = {(u8)(v >> 24), (u8)(v >> 16), (u8)(v >> 8), (u8)v}; fwrite(x, 1, 4, f); };
    auto chunk = [&](const char *type, const u8 *d, u32 n) {
        be(n); std::vector<u8> td(type, type + 4); td.insert(td.end(), d, d + n);
        fwrite(td.data(), 1, td.size(), f); be(crc(td.data(), td.size()));
    };
    static const u8 sig[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10};
    fwrite(sig, 1, 8, f);
    u8 ih[13] = {(u8)(W >> 24), (u8)(W >> 16), (u8)(W >> 8), (u8)W, (u8)(H >> 24), (u8)(H >> 16), (u8)(H >> 8), (u8)H, 8, 6, 0, 0, 0};
    chunk("IHDR", ih, 13); chunk("IDAT", z.data(), (u32)z.size()); chunk("IEND", nullptr, 0);
    fclose(f);
}
static std::vector<u8> readback(WGPUTexture tex, u32 w, u32 h) {
    u32 bpr = (w * 4 + 255) & ~255u;
    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead; bd.size = (u64)bpr * h;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(G.dev, &bd);
    WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(G.dev, nullptr);
    WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT; src.texture = tex;
    WGPUTexelCopyBufferInfo dst = WGPU_TEXEL_COPY_BUFFER_INFO_INIT; dst.buffer = buf; dst.layout.bytesPerRow = bpr; dst.layout.rowsPerImage = h;
    WGPUExtent3D ext = {w, h, 1};
    wgpuCommandEncoderCopyTextureToBuffer(e, &src, &dst, &ext);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(e, nullptr);
    wgpuQueueSubmit(G.queue, 1, &cb);
    wgpuCommandBufferRelease(cb); wgpuCommandEncoderRelease(e);
    bool done = false;
    WGPUBufferMapCallbackInfo mi = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    mi.mode = WGPUCallbackMode_AllowProcessEvents;
    mi.callback = [](WGPUMapAsyncStatus, WGPUStringView, void *u, void *) { *(bool *)u = true; };
    mi.userdata1 = &done;
    wgpuBufferMapAsync(buf, WGPUMapMode_Read, 0, bd.size, mi);
    while (!done) { wgpuDevicePoll(G.dev, true, nullptr); wgpuInstanceProcessEvents(G.inst); }
    const u8 *p = (const u8 *)wgpuBufferGetConstMappedRange(buf, 0, bd.size);
    std::vector<u8> out((size_t)w * h * 4);
    for (u32 y = 0; y < h; y++) memcpy(&out[(size_t)y * w * 4], p + (size_t)y * bpr, w * 4);
    wgpuBufferUnmap(buf); wgpuBufferRelease(buf);
    return out;
}
// GPU LCD image vs the software frame: mean absolute difference per channel
// after box-filtering the GPU image down to 1x
static double lcd_diff(const GSurf &g, u32 rows, int oy, const std::vector<u8> &img) {
    double sum = 0; u64 n = 0;
    u32 fx = g.pw / g.lw, fy = g.ph / g.lh;
    if (!fx || !fy || g.pw % g.lw || g.ph % g.lh) return -1;   // widened: not comparable
    for (u32 r = 0; r < rows; r++)
        for (u32 c = 0; c < 240; c++) {
            u32 acc[3] = {0, 0, 0};
            for (u32 dy = 0; dy < fy; dy++) for (u32 dx = 0; dx < fx; dx++) {
                const u8 *p = &img[(((size_t)(r * fy + dy)) * g.pw + c * fx + dx) * 4];
                acc[0] += p[0]; acc[1] += p[1]; acc[2] += p[2];
            }
            // LCD texel (c, r) is screen pixel (x = r, y = 239 - c)
            u32 sx = r + (rows == 320 ? 40 : 0), sy = 239 - c + oy;
            u32 s = G.swframe[sy * 400 + sx];
            for (int k = 0; k < 3; k++) sum += fabs(acc[k] / (double)(fx * fy) - ((s >> (8 * k)) & 0xFF));
            n += 3;
        }
    return n ? sum / n : -1;
}
void hwr_gpu_test_loop(const char *dir, u32 every, u32 W, u32 H, const HwrView &v) {
    if (!hwr_gpu_start(HwrSurfaceTarget{})) { LOG("[hwr-test] WebGPU unavailable"); return; }
    while (G.state == 1) { wgpuInstanceProcessEvents(G.inst); std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    if (G.state != 2) return;
    Tex target = make_tex(W, H, WGPUTextureFormat_RGBA8Unorm, WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc, "test target");
    u32 last = 0;
    for (;;) {
        choose_scale(W, H, v);
        bool nf = false;
        process(&nf);
        if (nf && G.info.frame >= last + every) {
            last = G.info.frame - G.info.frame % every;
            compose(target.view, WGPUTextureFormat_RGBA8Unorm, W, H, v, true);
            submit();
            char name[64];
            snprintf(name, sizeof name, "/g%05u.png", G.info.frame);
            auto img = readback(target.tex, W, H);
            png_write(dir + std::string(name), img.data(), W, H);
            std::string rep;
            for (int s = 0; s < 2; s++) {
                Src src; const HwrScreen &sc = s ? G.info.bot : G.info.top;
                if (!lcd_src(sc, s ? 320 : 400, src)) { rep += s ? " bottom=software" : " top=software"; continue; }
                for (auto &l : G.lcds[sc.va]) if (l.hash == sc.hash) {
                    GSurf &g = G.surfs[l.id];
                    auto li = readback(g.color.tex, g.pw, g.ph);
                    double d = lcd_diff(g, s ? 320 : 400, s ? 240 : 0, li);
                    char b[96]; snprintf(b, sizeof b, " %s=gpu %ux%u diff %.2f", s ? "bottom" : "top", g.pw, g.ph, d);
                    rep += b;
                    snprintf(name, sizeof name, "/g%05u_%s.png", G.info.frame, s ? "bot" : "top");
                    png_write(dir + std::string(name), li.data(), g.pw, g.ph);
                    break;
                }
            }
            LOG("[hwr-test] frame %u scale %u%s", G.info.frame, g_hwr_req_scale.load(), rep.c_str());
        } else submit();
        if (!nf) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}
void hwr_test_main(const char *dir) {
    u32 every = getenv("R3DS_HWR_TEST_EVERY") ? atoi(getenv("R3DS_HWR_TEST_EVERY")) : 300;
    u32 W = 1920, H = 1080;
    if (const char *sz = getenv("R3DS_HWR_TEST_SIZE")) sscanf(sz, "%ux%u", &W, &H);
    HwrView v = hwr_view_from_env();
    hwr_gpu_test_loop(dir, every ? every : 300, W, H, v);
}
#endif
// display settings from the environment (native): R3DS_LAYOUT, R3DS_BOTTOM, R3DS_ASPECT, R3DS_FILTER, R3DS_RES ...
HwrView hwr_view_from_env() {
    HwrView v;
    auto pick = [](const char *env, std::initializer_list<const char *> names, int def) {
        const char *x = getenv(env);
        int i = 0;
        if (x) for (const char *n : names) { if (!strcmp(x, n)) return i; i++; }
        return def;
    };
    v.layout = pick("R3DS_LAYOUT", {"remaster", "classic", "side", "top", "large"}, v.layout);
    v.bottom_mode = pick("R3DS_BOTTOM", {"auto", "ui", "minimap"}, v.bottom_mode);
    v.aspect = pick("R3DS_ASPECT", {"5:3", "auto", "16:9", "stretch"}, v.aspect);
    v.filter = pick("R3DS_FILTER", {"pixel", "smooth", "hq"}, v.filter);
    v.ui_anchor = pick("R3DS_UI_POS", {"right", "centre", "left", "top-right", "top-left"}, v.ui_anchor);
    if (getenv("R3DS_KEEP_BG")) v.remove_bg = 0;
    if (const char *x = getenv("R3DS_RES")) v.res = atoi(x);
    if (const char *x = getenv("R3DS_MAX_RES")) v.max_res = atoi(x);
    if (const char *x = getenv("R3DS_UI_SCALE")) v.ui_scale = (float)atof(x);
    return v;
}
#endif   // R3DS_WEBGPU
