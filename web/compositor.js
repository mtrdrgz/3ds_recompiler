// WebGL2 compositor: lays out the two 3DS screens for a PC/phone display.
//
// Input each frame: the runtime's 400x480 RGBA image (top screen in rows
// 0..239, bottom screen at x 40..359 of rows 240..479). The bottom screen's
// alpha is the renderer's content mask (0 = backdrop), so it can be drawn
// over the top screen with its background removed.
//
// Layouts
//   remaster  top screen fills the window; the bottom screen floats over it:
//             "ui" (large, bottom-centre) for menus, "minimap" (small corner)
//             for mostly-opaque screens like the field map, or "auto"
//             (switches on how much of the bottom screen is content)
//   classic   the original stacked screens
//   side      top screen left, bottom screen right
//   top       top screen only (hold Tab to see the bottom screen)
// Filters: pixel (sharp, anti-aliased pixel edges), smooth (bilinear),
// hq (Catmull-Rom bicubic). Aspect: native 5:3 or 16:9 stretch; the side
// bars can be black or a blurred extension of the picture.
'use strict';

const R3DS_DEFAULTS = {
  layout: 'remaster', bottomMode: 'auto', removeBg: true,
  uiScale: 0.62, uiOpacity: 0.96, uiAnchor: 'br', miniScale: 0.36, miniOpacity: 0.9, miniCorner: 'br',
  aspect: 'auto', sideFill: 'blur', filter: 'hq', showFps: true, renderer: 'auto', res: 'auto', swapKeys: true, romMode: 'download', autoFs: true, hdTextures: true,
};

class R3DSCompositor {
  constructor(canvas) {
    this.cv = canvas;
    const gl = this.gl = canvas.getContext('webgl2', { alpha: false, antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 is not available in this browser');
    this.frame = new Uint8Array(400 * 480 * 4);      // raw frame (copied out of shared wasm memory)
    this.bottomPM = new Uint8Array(320 * 240 * 4);   // bottom screen, premultiplied by its mask
    this.coverage = 1; this.masked = false;
    this.mode = 'ui'; this.modeVotes = 0; this.cur = null; // animated bottom rect
    this.showFull = false;                            // Tab held
    this.bottomRect = null;                           // last bottom rect in CSS px (for touch)
    this.texFrame = this._tex(400, 480);
    this.texBottom = this._tex(320, 240);
    // the blurred side bars are drawn at low resolution into this, then stretched
    this.texBlur = this._tex(160, 96);
    this.fboBlur = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboBlur);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texBlur, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const vs = `#version 300 es
      in vec2 p; out vec2 uv; uniform vec4 dst; uniform vec4 src;
      void main() { uv = src.xy + p * src.zw; gl_Position = vec4(dst.xy + p * dst.zw, 0.0, 1.0); }`;
    const fs = `#version 300 es
      precision highp float;
      in vec2 uv; out vec4 o;
      uniform sampler2D tex; uniform vec2 tsz; uniform int mode; uniform vec2 scale;
      uniform float alpha; uniform int useAlpha; uniform float blur; uniform float dim;
      vec4 tap(vec2 t) { return texture(tex, t / tsz); }
      vec4 bicubic(vec2 t) {   // Catmull-Rom with 9 bilinear taps
        vec2 c = floor(t - 0.5) + 0.5, f = t - c;
        vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f)), w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
        vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f)), w3 = f * f * (-0.5 + 0.5 * f);
        vec2 w12 = w1 + w2, o12 = w2 / w12;
        vec2 t0 = c - 1.0, t3 = c + 2.0, t12 = c + o12;
        vec4 r = tap(vec2(t0.x, t0.y)) * w0.x * w0.y + tap(vec2(t12.x, t0.y)) * w12.x * w0.y + tap(vec2(t3.x, t0.y)) * w3.x * w0.y
               + tap(vec2(t0.x, t12.y)) * w0.x * w12.y + tap(vec2(t12.x, t12.y)) * w12.x * w12.y + tap(vec2(t3.x, t12.y)) * w3.x * w12.y
               + tap(vec2(t0.x, t3.y)) * w0.x * w3.y + tap(vec2(t12.x, t3.y)) * w12.x * w3.y + tap(vec2(t3.x, t3.y)) * w3.x * w3.y;
        return max(r, 0.0);
      }
      void main() {
        vec2 t = uv;   // texel coordinates
        vec4 c;
        if (blur > 0.0) {
          c = vec4(0.0);
          for (int y = -2; y <= 2; y++) for (int x = -2; x <= 2; x++) c += tap(t + vec2(x, y) * blur);
          c /= 25.0; c.rgb *= dim;
        } else if (mode == 0) {   // sharp pixels: nearest with 1-screen-pixel anti-aliased edges
          vec2 f = fract(t - 0.5), i = floor(t - 0.5);
          f = clamp((f - 0.5) * scale + 0.5, 0.0, 1.0);
          c = tap(i + 0.5 + f);
        } else if (mode == 1) {
          c = tap(t);
        } else {
          c = bicubic(t);
        }
        if (useAlpha == 1) o = vec4(c.rgb * alpha, c.a * alpha);   // premultiplied input
        else o = vec4(c.rgb * alpha, alpha);
      }`;
    this.prog = this._program(vs, fs);
    this.u = {};
    for (const n of ['dst', 'src', 'tex', 'tsz', 'mode', 'scale', 'alpha', 'useAlpha', 'blur', 'dim']) this.u[n] = gl.getUniformLocation(this.prog, n);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const loc = gl.getAttribLocation(this.prog, 'p');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  }
  _tex(w, h) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  _program(vs, fs) {
    const gl = this.gl;
    const sh = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  // copy a new frame from wasm memory (view over the shared buffer)
  upload(view, coverage, masked) {
    this.frame.set(view);
    this.coverage = coverage; this.masked = masked;
    const f = this.frame, b = this.bottomPM;
    for (let y = 0; y < 240; y++) {
      let si = ((240 + y) * 400 + 40) * 4, di = y * 320 * 4;
      for (let x = 0; x < 320; x++, si += 4, di += 4) {
        const a = masked ? f[si + 3] : 255;
        b[di] = f[si] * a / 255; b[di + 1] = f[si + 1] * a / 255; b[di + 2] = f[si + 2] * a / 255; b[di + 3] = a;
      }
    }
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.texFrame);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 400, 480, gl.RGBA, gl.UNSIGNED_BYTE, this.frame);
    gl.bindTexture(gl.TEXTURE_2D, this.texBottom);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 320, 240, gl.RGBA, gl.UNSIGNED_BYTE, this.bottomPM);
  }

  // alpha of the bottom screen at bottom-screen pixel (x, y), for touch hit tests
  bottomAlpha(x, y) {
    if (x < 0 || y < 0 || x >= 320 || y >= 240) return 0;
    return this.bottomPM[(y * 320 + x) * 4 + 3];
  }

  _draw(tex, tw, th, src, dst, opts, W = this.cv.width, H = this.cv.height) {
    const gl = this.gl;
    gl.uniform4f(this.u.dst, dst.x / W * 2 - 1, 1 - dst.y / H * 2, dst.w / W * 2, -dst.h / H * 2);
    gl.uniform4f(this.u.src, src.x, src.y, src.w, src.h);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(this.u.tex, 0);
    gl.uniform2f(this.u.tsz, tw, th);
    gl.uniform1i(this.u.mode, { pixel: 0, smooth: 1, hq: 2 }[opts.filter] ?? 2);
    gl.uniform2f(this.u.scale, Math.max(1, dst.w / src.w), Math.max(1, dst.h / src.h));
    gl.uniform1f(this.u.alpha, opts.alpha ?? 1);
    gl.uniform1i(this.u.useAlpha, opts.premul ? 1 : 0);
    gl.uniform1f(this.u.blur, opts.blur || 0);
    gl.uniform1f(this.u.dim, opts.dim ?? 1);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // fit a w:h box inside area a (all in device px)
  static fit(a, aspect) {
    let w = a.w, h = w / aspect;
    if (h > a.h) { h = a.h; w = h * aspect; }
    return { x: a.x + (a.w - w) / 2, y: a.y + (a.h - h) / 2, w, h };
  }

  render(S, dpr) {
    const gl = this.gl, W = this.cv.width, H = this.cv.height;
    gl.useProgram(this.prog); gl.bindVertexArray(this.vao);
    gl.disable(gl.BLEND);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    // true widescreen needs the GPU renderer; here 'wide' stretches and 'auto' keeps 5:3
    const topAspect = S.aspect === 'stretch' || S.aspect === 'wide' ? 16 / 9 : 5 / 3;
    const topSrc = { x: 0, y: 0, w: 400, h: 240 }, botSrc = { x: 0, y: 0, w: 320, h: 240 }, botRawSrc = { x: 40, y: 240, w: 320, h: 240 };
    const full = { x: 0, y: 0, w: W, h: H };
    let top, bot = null, botOpacity = 1, botPremul = S.removeBg && this.masked, botHit = botPremul;

    if (S.layout === 'classic') {
      const r = R3DSCompositor.fit(full, 400 / 480);
      const s = r.w / 400;
      top = { x: r.x, y: r.y, w: r.w, h: 240 * s };
      bot = { x: r.x + 40 * s, y: r.y + 240 * s, w: 320 * s, h: 240 * s };
      botPremul = false; botHit = false;
    } else if (S.layout === 'side') {
      const r = R3DSCompositor.fit(full, (400 + 320 * 0.6) / 240);
      const s = r.h / 240;
      top = { x: r.x, y: r.y, w: 400 * s, h: 240 * s };
      const bs = s * 0.6;
      bot = { x: r.x + 400 * s, y: r.y + (240 * s - 240 * bs), w: 320 * bs, h: 240 * bs };
      botPremul = false; botHit = false;
    } else if (S.layout === 'large') {   // Citra's large screen: big top, small bottom beside it
      const th = Math.min(H, W / (5 / 3 + 2 / 3)), tw = th * 5 / 3, bw = th * 2 / 3, bh = th / 2;
      const x = (W - tw - bw) / 2, y = (H - th) / 2;
      top = { x, y, w: tw, h: th };
      bot = { x: x + tw, y: y + th - bh, w: bw, h: bh };
      botPremul = false; botHit = false;
    } else {   // remaster / top
      top = R3DSCompositor.fit(full, topAspect);
      const s = top.h / 240;
      // automatic UI / minimap choice, with hysteresis
      let want = S.bottomMode;
      if (want === 'auto') {
        const guess = this.masked && this.coverage > 0.72 ? 'minimap' : 'ui';
        if (guess !== this.mode) { if (++this.modeVotes > 20) { this.mode = guess; this.modeVotes = 0; } } else this.modeVotes = 0;
        want = this.mode;
      }
      if (this.showFull) {   // Tab: the whole bottom screen, opaque, large
        const bs = s * Math.max(S.uiScale, 0.9);
        bot = { x: top.x + (top.w - 320 * bs) / 2, y: top.y + top.h - 240 * bs, w: 320 * bs, h: 240 * bs };
        botPremul = false; botHit = false; botOpacity = 1;
      } else if (S.layout === 'top') {
        bot = null;
      } else if (want === 'minimap') {
        const bs = s * S.miniScale, m = 12 * dpr;
        const x = S.miniCorner.includes('l') ? top.x + m : top.x + top.w - 320 * bs - m;
        const y = S.miniCorner.includes('t') ? top.y + m : top.y + top.h - 240 * bs - m;
        bot = { x, y, w: 320 * bs, h: 240 * bs };
        botOpacity = S.miniOpacity;
      } else {
        const bs = s * S.uiScale, bw = 320 * bs, bh = 240 * bs, m = top.h * 0.02;
        const left = S.uiAnchor === 'bl' || S.uiAnchor === 'tl', upper = S.uiAnchor === 'tr' || S.uiAnchor === 'tl';
        const x = S.uiAnchor === 'bc' ? top.x + (top.w - bw) / 2 : left ? top.x + m : top.x + top.w - bw - m;
        bot = { x, y: upper ? top.y + m : top.y + top.h - bh - (S.uiAnchor === 'bc' ? 0 : m), w: bw, h: bh };
        botOpacity = S.uiOpacity;
      }
      // animate between placements
      if (bot && this.cur && !this.showFull) {
        const k = 0.25;
        for (const key of ['x', 'y', 'w', 'h']) this.cur[key] += (bot[key] - this.cur[key]) * k;
        bot = { ...this.cur };
      }
      if (bot) this.cur = { ...bot };
    }
    // side bars: blurred, darkened extension of the top screen
    if (S.sideFill === 'blur' && (top.x > 1 || top.y > 1) && S.layout !== 'classic') {
      const cover = { w: W, h: W / topAspect };
      if (cover.h < H) { cover.h = H; cover.w = H * topAspect; }
      // blur at 160x96, then stretch it (a full-window blur pass is too slow on software GL)
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboBlur); gl.viewport(0, 0, 160, 96);
      this._draw(this.texFrame, 400, 480, topSrc, { x: 0, y: 0, w: 160, h: 96 }, { filter: 'smooth', blur: 4.0, dim: 0.45 }, 160, 96);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, W, H);
      this._draw(this.texBlur, 160, 96, { x: 0, y: 96, w: 160, h: -96 }, { x: (W - cover.w) / 2, y: (H - cover.h) / 2, w: cover.w, h: cover.h },
                 { filter: 'smooth' });
    }
    this._draw(this.texFrame, 400, 480, topSrc, top, { filter: S.filter });
    if (bot) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      if (botPremul) this._draw(this.texBottom, 320, 240, botSrc, bot, { filter: S.filter, premul: true, alpha: botOpacity });
      else this._draw(this.texFrame, 400, 480, botRawSrc, bot, { filter: S.filter, alpha: botOpacity });
      gl.disable(gl.BLEND);
    }
    this.bottomRect = bot ? { x: bot.x / dpr, y: bot.y / dpr, w: bot.w / dpr, h: bot.h / dpr, hit: botHit } : null;
  }
}
window.R3DSCompositor = R3DSCompositor;
window.R3DS_DEFAULTS = R3DS_DEFAULTS;
