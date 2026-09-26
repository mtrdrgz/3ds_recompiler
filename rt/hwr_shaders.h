// WGSL for the hardware renderer (hwr_gpu.cpp).
#pragma once

// PICA fragment pipeline. Integer math follows pica_fast.cpp (shade_rows,
// sample_fast, pica_lighting) so that at 1x the output matches the software
// renderer; at higher resolutions it is the same computation per sample.
static const char *WGSL_PICA = R"(
struct Uni { d: array<vec4<u32>, 32> };
@group(0) @binding(0) var<uniform> U: Uni;
@group(1) @binding(0) var T0: texture_2d<f32>;
@group(1) @binding(1) var T1: texture_2d<f32>;
@group(1) @binding(2) var T2: texture_2d<f32>;

const F_WBUF: u32 = 1u; const F_LIGHT: u32 = 2u; const F_TEX2_TC1: u32 = 4u; const F_ALPHA_TEST: u32 = 8u;
const F_EXCLUDE: u32 = 16u; const F_NOALPHA: u32 = 32u; const F_SQUEEZE: u32 = 64u;
const F_TEX0: u32 = 256u; const F_TEX1: u32 = 512u; const F_TEX2: u32 = 1024u;

fn uf(v: u32) -> f32 { return bitcast<f32>(v); }

struct VIn {
  @location(0) pos: vec4f,
  @location(1) col: vec4f,
  @location(2) t01: vec4f,
  @location(3) t2v: vec4f,
  @location(4) quat: vec4f,
  @location(5) vz: f32,
};
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) col: vec4f,
  @location(1) t01: vec4f,
  @location(2) t2w: vec4f,
  @location(3) quat: vec4f,
  @location(4) view: vec4f,
};

@vertex fn vs(v: VIn) -> VOut {
  let a = U.d[0]; let b = U.d[1]; let flags = U.d[2].x;
  var x = uf(a.x) * v.pos.x + uf(a.y) * v.pos.w;
  var y = uf(a.z) * v.pos.y + uf(a.w) * v.pos.w;
  if ((flags & F_SQUEEZE) != 0u) {
    if (b.w == 1u) { x = x * uf(b.z); } else if (b.w == 2u) { y = y * uf(b.z); }
  }
  var o: VOut;
  o.pos = vec4f(x, y, -v.pos.z, v.pos.w);
  o.col = v.col;
  o.t01 = v.t01;
  o.t2w = vec4f(v.t2v.xy, v.pos.w, 0.0);
  o.quat = v.quat;
  o.view = vec4f(v.t2v.zw, v.vz, 0.0);
  return o;
}

fn unpack8(c: u32) -> vec4i { return vec4i(i32(c & 255u), i32((c >> 8u) & 255u), i32((c >> 16u) & 255u), i32(c >> 24u)); }

fn wrapc(c: i32, size: i32, mode: u32, border: ptr<function, bool>) -> i32 {
  if (mode == 2u) { var m = c % size; if (m < 0) { m = m + size; } return m; }
  if (mode == 3u) { let p = size * 2; var m = c % p; if (m < 0) { m = m + p; } return select(p - 1 - m, m, m < size); }
  if (mode == 1u) { if (c < 0 || c >= size) { *border = true; } }
  return clamp(c, 0, size - 1);
}
fn texel(t: texture_2d<f32>, s: i32, tt: i32, h: i32) -> vec4i {
  return vec4i(round(textureLoad(t, vec2i(s, h - 1 - tt), 0) * 255.0));
}
fn sample(t: texture_2d<f32>, unit: u32, uv: vec2f) -> vec4i {
  let p = U.d[10u + unit];
  let param = p.x;
  let border = unpack8(p.y);
  let ws = (param >> 12u) & 7u; let wt = (param >> 8u) & 7u;
  let dims = vec2i(textureDimensions(t));
  var fu = clamp(uv.x * f32(dims.x), -8388608.0, 8388608.0);
  var fv = clamp(uv.y * f32(dims.y), -8388608.0, 8388608.0);
  if (((param >> 1u) & 1u) == 0u) {
    var bd = false;
    let s = wrapc(i32(floor(fu)), dims.x, ws, &bd);
    let tt = wrapc(i32(floor(fv)), dims.y, wt, &bd);
    if (bd) { return border; }
    return texel(t, s, tt, dims.y);
  }
  fu = fu - 0.5; fv = fv - 0.5;
  let s0 = i32(floor(fu)); let t0 = i32(floor(fv));
  let wu = i32((fu - f32(s0)) * 256.0); let wv = i32((fv - f32(t0)) * 256.0);
  var b0 = false; var b1 = false; var b2 = false; var b3 = false;
  let sa = wrapc(s0, dims.x, ws, &b0); let sb = wrapc(s0 + 1, dims.x, ws, &b1);
  let ta = wrapc(t0, dims.y, wt, &b2); let tb = wrapc(t0 + 1, dims.y, wt, &b3);
  let c00 = select(texel(t, sa, ta, dims.y), border, b0 || b2);
  let c10 = select(texel(t, sb, ta, dims.y), border, b1 || b2);
  let c01 = select(texel(t, sa, tb, dims.y), border, b0 || b3);
  let c11 = select(texel(t, sb, tb, dims.y), border, b1 || b3);
  let top = c00 * (256 - wu) + c10 * wu;
  let bot = c01 * (256 - wu) + c11 * wu;
  return min((top * (256 - wv) + bot * wv + 32768) >> vec4u(16u), vec4i(255));
}

fn cmod(op: u32, v: vec4i) -> vec3i {
  switch (op) {
    case 1u: { return 255 - v.rgb; }
    case 2u: { return vec3i(v.a); }
    case 3u: { return vec3i(255 - v.a); }
    case 4u: { return vec3i(v.r); }
    case 5u: { return vec3i(255 - v.r); }
    case 8u: { return vec3i(v.g); }
    case 9u: { return vec3i(255 - v.g); }
    case 12u: { return vec3i(v.b); }
    case 13u: { return vec3i(255 - v.b); }
    default: { return v.rgb; }
  }
}
fn amod(op: u32, v: vec4i) -> i32 {
  switch (op) {
    case 0u: { return v.a; }
    case 1u: { return 255 - v.a; }
    case 2u: { return v.r; }
    case 3u: { return 255 - v.r; }
    case 4u: { return v.g; }
    case 5u: { return 255 - v.g; }
    case 6u: { return v.b; }
    default: { return 255 - v.b; }
  }
}
fn comb(op: u32, a: vec3i, b: vec3i, c: vec3i) -> vec3i {
  switch (op) {
    case 1u: { return (a * b + 127) / 255; }
    case 2u: { return min(a + b, vec3i(255)); }
    case 3u: { return clamp(a + b - 128, vec3i(0), vec3i(255)); }
    case 4u: { return (a * c + b * (255 - c) + 127) / 255; }
    case 5u: { return max(a - b, vec3i(0)); }
    case 8u: { return min((a * b + 255 * c) / 255, vec3i(255)); }
    case 9u: { return min(a + b, vec3i(255)) * c / 255; }
    default: { return a; }
  }
}
fn cmpf(f: u32, a: i32, b: i32) -> bool {
  switch (f) {
    case 0u: { return false; }
    case 1u: { return true; }
    case 2u: { return a == b; }
    case 3u: { return a != b; }
    case 4u: { return a < b; }
    case 5u: { return a <= b; }
    case 6u: { return a > b; }
    default: { return a >= b; }
  }
}
fn col10(c: u32) -> vec3f { return vec3f(f32((c >> 20u) & 255u), f32((c >> 10u) & 255u), f32(c & 255u)) / 255.0; }

// pica_lighting() of pica_raster.cpp
fn lighting(view: vec3f, qin: vec4f, prim: ptr<function, vec4i>, sec: ptr<function, vec4i>) {
  var q = qin;
  let ql = length(q);
  if (ql > 0.0) { q = q / ql; }
  let n = vec3f(2.0 * (q.x * q.z + q.w * q.y), 2.0 * (q.y * q.z - q.w * q.x), 1.0 - 2.0 * (q.x * q.x + q.y * q.y));
  let vl = length(view);
  var v = vec3f(0.0, 0.0, 1.0);
  if (vl > 0.0) { v = view / vl; }
  var p = col10(U.d[2].w);
  var s = vec3f(0.0);
  let nl = U.d[3].x;
  for (var i = 0u; i < 8u; i = i + 1u) {
    if (i >= nl) { break; }
    let L = U.d[14u + i * 2u];
    let P = U.d[15u + i * 2u];
    var l = vec3f(uf(P.x), uf(P.y), uf(P.z));
    if ((L.w & 1u) == 0u) { l = l + view; }
    let ll = length(l);
    if (ll > 0.0) { l = l / ll; }
    let ndl = dot(n, l);
    p = p + col10(L.z) + col10(L.y) * max(ndl, 0.0);
    let h = l + v; let hl = length(h);
    if (hl > 0.0 && ndl > 0.0) {
      let nh = max(dot(n, h) / hl, 0.0);
      s = s + col10(L.x) * pow(nh, 16.0);
    }
  }
  *prim = vec4i(vec3i(clamp(p, vec3f(0.0), vec3f(1.0)) * 255.0), 255);
  *sec = vec4i(vec3i(clamp(s, vec3f(0.0), vec3f(1.0)) * 255.0), 255);
}

struct TevIn { prim: vec4i, fp: vec4i, fs: vec4i, t0: vec4i, t1: vec4i, t2: vec4i, buf: vec4i, k: vec4i, prev: vec4i };
fn pick(id: u32, e: TevIn) -> vec4i {
  switch (id) {
    case 1u: { return e.fp; }
    case 2u: { return e.fs; }
    case 3u: { return e.t0; }
    case 4u: { return e.t1; }
    case 5u: { return e.t2; }
    case 6u: { return vec4i(0); }
    case 13u: { return e.buf; }
    case 14u: { return e.k; }
    case 15u: { return e.prev; }
    default: { return e.prim; }
  }
}

struct FOut {
  @location(0) c: vec4f,
  @location(1) cov: vec4f,
  @builtin(frag_depth) depth: f32,
};

@fragment fn fs(i: VOut) -> FOut {
  let flags = U.d[2].x;
  let dims = U.d[13];
  if ((flags & F_EXCLUDE) != 0u) {
    let lw = uf(dims.x); let lh = uf(dims.y); let pw = uf(dims.z); let ph = uf(dims.w);
    let fx = i32(floor(i.pos.x * lw / pw));
    let fy = i32(lh) - 1 - i32(floor(i.pos.y * lh / ph));
    let e0 = U.d[3].z; let e1 = U.d[3].w;
    if (fx >= i32(e0 & 0xFFFFu) && fx < i32(e1 & 0xFFFFu) && fy >= i32(e0 >> 16u) && fy < i32(e1 >> 16u)) { discard; }
  }
  let prim = clamp(vec4i(i.col * 255.0 + 0.5), vec4i(0), vec4i(255));
  var e: TevIn;
  e.prim = prim; e.fp = vec4i(0); e.fs = vec4i(0);
  e.t0 = vec4i(0); e.t1 = vec4i(0); e.t2 = vec4i(0);
  if ((flags & F_TEX0) != 0u) { e.t0 = sample(T0, 0u, i.t01.xy); }
  if ((flags & F_TEX1) != 0u) { e.t1 = sample(T1, 1u, i.t01.zw); }
  if ((flags & F_TEX2) != 0u) {
    var uv = i.t2w.xy;
    if ((flags & F_TEX2_TC1) != 0u) { uv = i.t01.zw; }
    e.t2 = sample(T2, 2u, uv);
  }
  if ((flags & F_LIGHT) != 0u) {
    var lp: vec4i; var ls: vec4i;
    lighting(i.view.xyz, i.quat, &lp, &ls);
    e.fp = lp; e.fs = ls;
  }
  e.buf = vec4i(0);
  e.prev = vec4i(0);
  var outc = vec4i(0);
  var nbuf = unpack8(U.d[2].y);
  let ntev = (U.d[3].y >> 16u) & 7u;
  for (var s = 0u; s < ntev; s = s + 1u) {
    let st = U.d[4u + s];
    e.k = unpack8(st.w);
    let src = st.x; let op = st.y; let cmb = st.z;
    let c0 = cmod(op & 15u, pick(src & 15u, e));
    let c1 = cmod((op >> 4u) & 15u, pick((src >> 4u) & 15u, e));
    let c2 = cmod((op >> 8u) & 15u, pick((src >> 8u) & 15u, e));
    let a0 = amod((op >> 12u) & 7u, pick((src >> 16u) & 15u, e));
    let a1 = amod((op >> 16u) & 7u, pick((src >> 20u) & 15u, e));
    let a2 = amod((op >> 20u) & 7u, pick((src >> 24u) & 15u, e));
    let cop = cmb & 15u; let aop = (cmb >> 16u) & 15u;
    var rc: vec3i;
    if (cop == 6u || cop == 7u) {
      let d = clamp(((c0.r - 128) * (c1.r - 128) + (c0.g - 128) * (c1.g - 128) + (c0.b - 128) * (c1.b - 128)) * 4 / 255, 0, 255);
      rc = vec3i(d);
    } else {
      rc = comb(cop, c0, c1, c2);
    }
    var ra: i32;
    if (cop == 7u) { ra = rc.x; }
    else { ra = comb(select(aop, 0u, aop == 6u || aop == 7u), vec3i(a0), vec3i(a1), vec3i(a2)).x; }
    let cs = 1 << ((cmb >> 8u) & 3u); let as_ = 1 << ((cmb >> 24u) & 3u);
    outc = vec4i(min(rc * cs, vec3i(255)), min(ra * as_, 255));
    e.prev = outc;
    e.buf = nbuf;
    if ((cmb & 16u) != 0u) { nbuf = vec4i(outc.rgb, nbuf.a); }
    if ((cmb & 32u) != 0u) { nbuf.a = outc.a; }
  }
  let at = U.d[2].z;
  if ((flags & F_ALPHA_TEST) != 0u && !cmpf(at & 7u, outc.a, i32((at >> 8u) & 255u))) { discard; }

  var o: FOut;
  let z = -i.pos.z;
  var depth = z * uf(U.d[1].x) + uf(U.d[1].y);
  if ((flags & F_WBUF) != 0u) { depth = depth * i.t2w.z; }
  o.depth = clamp(depth, 0.0, 1.0);
  var c = vec4f(outc) / 255.0;
  let lclass = U.d[3].y & 255u;
  if (lclass == 1u) { c = vec4f(0.0); }
  else if (lclass == 2u || lclass == 5u) { c = vec4f(1.0); }
  else if (lclass == 3u) { c = vec4f(1.0) - c; }
  if ((flags & F_NOALPHA) != 0u) { c.a = 1.0; }
  o.c = c;
  let covm = (U.d[3].y >> 8u) & 3u;
  var cv = 0.0;
  if (covm == 1u) { cv = f32(outc.a) / 255.0; } else if (covm == 2u) { cv = 1.0; }
  o.cov = vec4f(cv, 0.0, 0.0, 1.0);
  return o;
}
)";

// Copies between surfaces: src rect (normalized) -> whole target, optional
// row flip; color + coverage, or color with coverage in alpha (LCD images).
static const char *WGSL_BLIT = R"(
struct BU { rect: vec4f, flags: vec4u };
@group(0) @binding(0) var<uniform> B: BU;
@group(0) @binding(1) var S: sampler;
@group(0) @binding(2) var TC: texture_2d<f32>;
@group(0) @binding(3) var TV: texture_2d<f32>;
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VO;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  var t = vec2f(p.x, 1.0 - p.y);          // target texel space, y down
  if (B.flags.x != 0u) { t.y = 1.0 - t.y; }
  o.uv = B.rect.xy + t * B.rect.zw;
  return o;
}
struct Two { @location(0) c: vec4f, @location(1) v: vec4f };
@fragment fn fs_surf(i: VO) -> Two {
  var o: Two;
  o.c = textureSampleLevel(TC, S, i.uv, 0.0);
  if (B.flags.z != 0u) { o.c.a = 1.0; }
  var cv = 0.0;
  if (B.flags.y != 0u) { cv = textureSampleLevel(TV, S, i.uv, 0.0).r; }
  o.v = vec4f(cv, 0.0, 0.0, 1.0);
  return o;
}
@fragment fn fs_lcd(i: VO) -> @location(0) vec4f {
  let c = textureSampleLevel(TC, S, i.uv, 0.0);
  let cv = textureSampleLevel(TV, S, i.uv, 0.0).r;
  return vec4f(c.rgb, cv);
}
)";

// depth / stencil load from a r32uint staging texture (depth24 | stencil << 24)
static const char *WGSL_DEPTH = R"(
@group(0) @binding(0) var D: texture_2d<u32>;
struct DU { v: vec4u };   // x: stencil bit, zw: target size in pixels
@group(0) @binding(1) var<uniform> P: DU;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
fn fetch(pos: vec4f) -> u32 {
  let d = vec2f(textureDimensions(D));
  let q = vec2i(floor(pos.xy * d / vec2f(P.v.zw)));
  return textureLoad(D, q, 0).r;
}
@fragment fn fs_depth(@builtin(position) pos: vec4f) -> @builtin(frag_depth) f32 {
  return f32(fetch(pos) & 0xFFFFFFu) / 16777215.0;
}
@fragment fn fs_stencil(@builtin(position) pos: vec4f) {
  if ((((fetch(pos) >> 24u) >> P.v.x) & 1u) == 0u) { discard; }
}
)";

// Compositor: one textured quad per call.
static const char *WGSL_COMP = R"(
struct CU {
  dst: vec4f,      // x, y, w, h in NDC (y up), of the quad
  m0: vec4f,       // uv = m0.xy * local.x + m0.zw * local.y + m1.xy
  m1: vec4f,       // m1.zw: texture size in texels
  p: vec4f,        // x alpha, y blur step, z dim, w filter (0 pixel 1 smooth 2 hq)
  q: vec4f,        // x mode (0 opaque, 1 premultiplied mask), y blank keying, z pixel scale.x, w pixel scale.y
};
@group(0) @binding(0) var<uniform> C: CU;
@group(0) @binding(1) var S: sampler;
@group(0) @binding(2) var T: texture_2d<f32>;
struct VO { @builtin(position) pos: vec4f, @location(0) l: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32(i & 1u), f32((i >> 1u) & 1u));
  var o: VO;
  o.pos = vec4f(C.dst.x + p.x * C.dst.z, C.dst.y - p.y * C.dst.w, 0.0, 1.0);
  o.l = p;
  return o;
}
fn tap(t: vec2f) -> vec4f { return textureSampleLevel(T, S, t / C.m1.zw, 0.0); }
fn bicubic(t: vec2f) -> vec4f {
  let c = floor(t - 0.5) + 0.5; let f = t - c;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f)); let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f)); let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2; let o12 = w2 / w12;
  let t0 = c - 1.0; let t3 = c + 2.0; let t12 = c + o12;
  let r = tap(vec2f(t0.x, t0.y)) * w0.x * w0.y + tap(vec2f(t12.x, t0.y)) * w12.x * w0.y + tap(vec2f(t3.x, t0.y)) * w3.x * w0.y
        + tap(vec2f(t0.x, t12.y)) * w0.x * w12.y + tap(vec2f(t12.x, t12.y)) * w12.x * w12.y + tap(vec2f(t3.x, t12.y)) * w3.x * w12.y
        + tap(vec2f(t0.x, t3.y)) * w0.x * w3.y + tap(vec2f(t12.x, t3.y)) * w12.x * w3.y + tap(vec2f(t3.x, t3.y)) * w3.x * w3.y;
  return max(r, vec4f(0.0));
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let uv = C.m0.xy * i.l.x + C.m0.zw * i.l.y + C.m1.xy;
  let t = uv * C.m1.zw;   // texel coordinates
  var c: vec4f;
  if (C.p.y > 0.0) {
    c = vec4f(0.0);
    for (var y = -2; y <= 2; y = y + 1) { for (var x = -2; x <= 2; x = x + 1) { c = c + tap(t + vec2f(f32(x), f32(y)) * C.p.y); } }
    c = c / 25.0; c = vec4f(c.rgb * C.p.z, 1.0);
    return c;
  } else if (C.p.w < 0.5) {
    let f0 = fract(t - 0.5); let ii = floor(t - 0.5);
    let f = clamp((f0 - 0.5) * C.q.zw + 0.5, vec2f(0.0), vec2f(1.0));
    c = tap(ii + 0.5 + f);
  } else if (C.p.w < 1.5) {
    c = tap(t);
  } else {
    c = bicubic(t);
  }
  let alpha = C.p.x;
  if (C.q.x > 0.5) {
    var a = clamp(c.a, 0.0, 1.0);
    if (C.q.y > 0.5) {
      let l = (c.r + c.g + c.b) / 3.0 * 255.0;
      a = min(a, select(min(1.0, (l - 12.0) * 8.0 / 255.0), 0.0, l < 12.0));
    }
    return vec4f(c.rgb * a * alpha, a * alpha);
  }
  return vec4f(c.rgb * alpha, alpha);
}
)";
