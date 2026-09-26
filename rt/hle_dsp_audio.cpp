// DSP audio HLE: the application drives 24 voices through two double-
// buffered shared-memory regions in DSP RAM (layout from the DSP firmware's
// structures). Every audio frame (160 samples @ 32728 Hz) we parse the
// region the app most recently finished, decode + resample each voice, mix,
// write voice statuses and the final mix into the other region, and hand the
// stereo frame to the host audio device.
#include "services.h"
#include <deque>
#include <queue>
#include <vector>
#include <mutex>

static const u32 REGION0 = 0x1FF50000, REGION1 = 0x1FF70000;
static inline u32 dspva(u16 dsp_addr, u32 region) { return region + (dsp_addr - 0x8000) * 2; }
enum : u16 { A_FRAME = 0xBFFF, A_SRCCFG = 0x9E92, A_SRCSTAT = 0x8680, A_ADPCM = 0xA792, A_DSPCFG = 0x9430,
             A_DSPSTAT = 0x8400, A_FINAL = 0x8540, A_INTERMIX = 0x9492 };
static const int NSRC = 24, CFG_SIZE = 192, STAT_SIZE = 12, SPF = 160;

static inline u32 rd_dsp32(u32 a) { return ((u32)rd16(a) << 16) | rd16(a + 2); }
static inline void wr_dsp32(u32 a, u32 v) { wr16(a, (u16)(v >> 16)); wr16(a + 2, (u16)v); }
static inline float rdf(u32 a) { u32 v = rd32(a); float f; memcpy(&f, &v, 4); return f; }


struct Buf {
    u32 pa, len; u16 ps; s16 yn1, yn2; bool adpcm_dirty, looping; u16 id;
    int channels, format; bool from_queue, played; u32 play_pos;
    bool operator<(const Buf &o) const { return id > o.id; }   // min-heap by id
};

struct Voice {
    bool enabled = false; u16 sync = 0;
    float rate = 1.0f; int interp = 0;
    float gain[3][4] = {};
    int format = 0, channels = 1;
    s16 coef[16] = {};
    s16 yn1 = 0, yn2 = 0;
    std::priority_queue<Buf> queue;
    std::vector<s16> cur;      // decoded samples of the current buffer (interleaved when stereo)
    int cur_ch = 1;
    double pos = 0;            // fractional position into cur (frames)
    u32 cur_id = 0, sample_no = 0;
    bool buffer_update = false;
    float out[SPF][2];
};
static Voice g_voice[NSRC];
static std::mutex g_audio_mtx;
static std::deque<s16> g_audio_out;   // interleaved stereo for the host device

void dsp_audio_pull(s16 *dst, int frames) {
    std::lock_guard<std::mutex> lk(g_audio_mtx);
    for (int i = 0; i < frames * 2; i++) {
        if (g_audio_out.empty()) { dst[i] = 0; continue; }
        dst[i] = g_audio_out.front(); g_audio_out.pop_front();
    }
}
size_t dsp_audio_queued() { std::lock_guard<std::mutex> lk(g_audio_mtx); return g_audio_out.size() / 2; }

static void decode(Voice &v, const Buf &b) {
    v.cur.clear();
    v.cur_ch = b.channels;
    u32 va = pa_to_va(b.pa);
    if (!va || !b.len || b.len > 0x1000000) return;
    if (b.format == 0) {          // PCM8
        v.cur.resize((size_t)b.len * b.channels);
        for (size_t i = 0; i < v.cur.size(); i++) v.cur[i] = (s16)((s8)rd8(va + i) << 8);
    } else if (b.format == 1) {   // PCM16
        v.cur.resize((size_t)b.len * b.channels);
        for (size_t i = 0; i < v.cur.size(); i++) v.cur[i] = (s16)rd16(va + i * 2);
    } else {                      // DSP-ADPCM, mono
        v.cur_ch = 1;
        v.cur.resize(b.len);
        if (b.adpcm_dirty) { v.yn1 = b.yn1; v.yn2 = b.yn2; }
        s32 y1 = v.yn1, y2 = v.yn2;
        u32 n = 0, off = 0;
        while (n < b.len) {
            u8 hdr = rd8(va + off);
            int scale = 1 << (hdr & 0xF), idx = (hdr >> 4) & 7;
            s32 c1 = v.coef[idx * 2], c2 = v.coef[idx * 2 + 1];
            for (int k = 0; k < 14 && n < b.len; k++, n++) {
                u8 byte = rd8(va + off + 1 + k / 2);
                s32 nib = (k & 1) ? (byte & 0xF) : (byte >> 4);
                if (nib >= 8) nib -= 16;
                s32 s = ((nib * scale) << 11) + c1 * y1 + c2 * y2 + 1024;
                s >>= 11;
                s = std::clamp(s, -32768, 32767);
                v.cur[n] = (s16)s;
                y2 = y1; y1 = s;
            }
            off += 8;
        }
        v.yn1 = (s16)y1; v.yn2 = (s16)y2;
    }
}

static bool dequeue(Voice &v) {
    if (v.queue.empty()) return false;
    Buf b = v.queue.top();
    v.queue.pop();
    decode(v, b);
    v.pos = (!b.played) ? b.play_pos : 0;
    v.sample_no = (u32)v.pos;
    v.cur_id = b.id;
    v.buffer_update = b.from_queue && !b.played;
    if (b.looping) { b.played = true; v.queue.push(b); }
    return true;
}

static void parse_config(Voice &v, u32 cfg, u32 coef_va) {
    u32 dirty = rd32(cfg);
    if (!dirty) return;
    // embedded buffer: pa(dsp32) len(dsp32) flags1{mono_or_stereo:2, format:2} ps yn[2] flags2{dirty:1, loop:1} buffer_id
    u32 emb = cfg + 0xAC;
    u16 flags1 = rd16(emb + 0x08);
    auto bit = [&](int b) { return (dirty >> b) & 1; };
    if (bit(29)) { v = Voice(); }                               // reset
    if (bit(4)) { while (!v.queue.empty()) v.queue.pop(); }      // partial reset
    if (bit(16)) v.enabled = rd8(cfg + 0xA0) != 0;
    if (bit(28)) v.sync = rd16(cfg + 0xA2);
    if (bit(18)) { v.rate = rdf(cfg + 0x34); if (!(v.rate > 0)) v.rate = 1.0f; }
    if (bit(2)) for (int i = 0; i < 16; i++) v.coef[i] = (s16)rd16(coef_va + i * 2);
    for (int m = 0; m < 3; m++)
        if (bit(25 + m)) for (int k = 0; k < 4; k++) v.gain[m][k] = rdf(cfg + 4 + (m * 4 + k) * 4);
    if (bit(17)) v.interp = rd8(cfg + 0x38);
    if (bit(0) || bit(30)) v.format = (flags1 >> 2) & 3;
    if (bit(1) || bit(30)) v.channels = (flags1 & 3) == 2 ? 2 : 1;
    u32 play_pos = 0;
    if (bit(21)) play_pos = rd_dsp32(cfg + 0xA4);
    if (bit(30)) {
        u16 f2 = rd16(emb + 0x10);
        Buf b{rd_dsp32(emb + 0), rd_dsp32(emb + 4), rd16(emb + 0x0A), (s16)rd16(emb + 0x0C), (s16)rd16(emb + 0x0E),
              (f2 & 1) != 0, (f2 & 2) != 0, rd16(emb + 0x12), v.channels, v.format, false, false, play_pos};
        v.queue.push(b);
    }
    if (bit(19)) {
        u16 bd = rd16(cfg + 0x4A);
        for (int i = 0; i < 4; i++) {
            if (!((bd >> i) & 1)) continue;
            u32 q = cfg + 0x4C + i * 20;
            u16 id = rd16(q + 0x10);
            if (!id) continue;
            u16 f2 = rd16(q + 14);
            Buf b{rd_dsp32(q), rd_dsp32(q + 4), rd16(q + 8), (s16)rd16(q + 10), (s16)rd16(q + 12),
                  (f2 & 1) != 0, (f2 & 2) != 0, id, v.channels, v.format, true, false, 0};
            v.queue.push(b);
        }
        wr16(cfg + 0x4A, 0);
    }
    wr32(cfg, 0);
}

static void gen_frame(Voice &v) {
    memset(v.out, 0, sizeof v.out);
    if (!v.enabled) return;
    if (v.cur.empty() && !dequeue(v)) { v.enabled = false; v.buffer_update = true; v.cur_id = 0; return; }
    int n = 0;
    while (n < SPF) {
        size_t frames = v.cur.size() / v.cur_ch;
        if (v.pos >= frames) {
            v.cur.clear();
            if (!dequeue(v)) break;
            continue;
        }
        size_t i = (size_t)v.pos;
        double fr = v.pos - i;
        auto smp = [&](size_t k, int ch) -> float {
            if (k >= frames) k = frames - 1;
            return v.cur[k * v.cur_ch + (v.cur_ch == 2 ? ch : 0)];
        };
        for (int ch = 0; ch < 2; ch++) {
            float a = smp(i, ch);
            if (v.interp != 2) { float b = smp(i + 1, ch); a = a + (b - a) * (float)fr; }
            v.out[n][ch] = a;
        }
        v.pos += v.rate;
        n++;
    }
    v.sample_no = (u32)v.pos;
}

static u32 read_region() {
    u16 f0 = rd16(dspva(A_FRAME, REGION0)), f1 = rd16(dspva(A_FRAME, REGION1));
    if (f0 == 0xFFFF && f1 != 0xFFFE) return REGION1;
    if (f1 == 0xFFFF && f0 != 0xFFFE) return REGION0;
    return f0 > f1 ? REGION0 : REGION1;
}

static FILE *g_wav;
static u32 g_wav_bytes;
static void wav_write_header() {
    if (!g_wav) return;
    u32 rate = 32728, byterate = rate * 4;
    fseek(g_wav, 0, SEEK_SET);
    fwrite("RIFF", 1, 4, g_wav); u32 v = 36 + g_wav_bytes; fwrite(&v, 4, 1, g_wav);
    fwrite("WAVEfmt ", 1, 8, g_wav); v = 16; fwrite(&v, 4, 1, g_wav);
    u16 h = 1; fwrite(&h, 2, 1, g_wav); h = 2; fwrite(&h, 2, 1, g_wav);
    fwrite(&rate, 4, 1, g_wav); fwrite(&byterate, 4, 1, g_wav);
    h = 4; fwrite(&h, 2, 1, g_wav); h = 16; fwrite(&h, 2, 1, g_wav);
    fwrite("data", 1, 4, g_wav); fwrite(&g_wav_bytes, 4, 1, g_wav);
    fseek(g_wav, 0, SEEK_END);
}

void dsp_audio_frame() {
    static bool init = false;
    if (!init) {
        init = true;
        if (const char *w = getenv("R3DS_WAV")) { g_wav = fopen(w, "wb"); wav_write_header(); }
    }
    if (!mem_is_mapped(REGION0)) return;
    u32 rr = read_region(), wr = rr == REGION0 ? REGION1 : REGION0;
    float mix[3][SPF][2] = {};
    for (int s = 0; s < NSRC; s++) {
        Voice &v = g_voice[s];
        parse_config(v, dspva(A_SRCCFG, rr) + s * CFG_SIZE, dspva(A_ADPCM, rr) + s * 32);
        gen_frame(v);
        u32 st = dspva(A_SRCSTAT, wr) + s * STAT_SIZE;
        wr8(st + 0, v.enabled);
        wr8(st + 1, v.buffer_update ? 1 : 0);
        v.buffer_update = false;
        wr16(st + 2, v.sync);
        wr_dsp32(st + 4, v.sample_no);
        wr16(st + 8, (u16)v.cur_id);
        if (!v.enabled && v.cur.empty()) continue;
        for (int m = 0; m < 3; m++) {
            float gl = v.gain[m][0] + v.gain[m][2], gr = v.gain[m][1] + v.gain[m][3];
            if (gl == 0 && gr == 0) continue;
            for (int i = 0; i < SPF; i++) { mix[m][i][0] += v.out[i][0] * gl; mix[m][i][1] += v.out[i][1] * gr; }
        }
    }
    // final mix: per-mixer volume from the DSP configuration
    u32 dc = dspva(A_DSPCFG, rr);
    static float vol[3] = {1, 0, 0};
    u32 ddirty = rd32(dc);
    for (int m = 0; m < 3; m++) if ((ddirty >> (16 + m)) & 1) vol[m] = rdf(dc + 4 + m * 4);
    if (ddirty) wr32(dc, 0);
    s16 outf[SPF * 2];
    u32 fin = dspva(A_FINAL, wr);
    for (int i = 0; i < SPF; i++) {
        for (int ch = 0; ch < 2; ch++) {
            float sum = mix[0][i][ch] * vol[0] + mix[1][i][ch] * vol[1] + mix[2][i][ch] * vol[2];
            s32 o = (s32)std::clamp(sum, -32768.0f, 32767.0f);
            outf[i * 2 + ch] = (s16)o;
            wr16(fin + (i * 2 + ch) * 2, (u16)o);
        }
    }
    if (g_wav) {
        fwrite(outf, 2, SPF * 2, g_wav); g_wav_bytes += SPF * 4;
        static int n = 0; if (++n % 200 == 0) { wav_write_header(); fflush(g_wav); }
    }
    std::lock_guard<std::mutex> lk(g_audio_mtx);
    if (g_audio_out.size() < 32728 * 2 / 4)   // cap latency at ~250ms
        g_audio_out.insert(g_audio_out.end(), outf, outf + SPF * 2);
}

void dsp_audio_dump() {
    u32 rr = read_region();
    for (u32 r : {REGION0, REGION1}) {
        int nz = 0; u32 first = 0;
        for (u32 a = r; a < r + 0x8000; a += 4) if (rd32(a)) { if (!nz) first = a; nz++; }
        fprintf(stderr, "[dsp] region %08x frame=%04x nonzero words=%d first=%08x\n", r, rd16(dspva(A_FRAME, r)), nz, first);
    }
    for (int s = 0; s < NSRC; s++) {
        u32 c = dspva(A_SRCCFG, rr) + s * CFG_SIZE;
        bool any = false;
        for (int i = 0; i < CFG_SIZE; i += 4) if (rd32(c + i)) any = true;
        if (!any) continue;
        fprintf(stderr, "[dsp] src %d cfg:", s);
        for (int i = 0; i < CFG_SIZE; i += 2) fprintf(stderr, "%s%04x", i % 32 ? " " : "\n   ", rd16(c + i));
        fprintf(stderr, "\n");
        u32 pa = rd_dsp32(c + 0x4C), va = pa_to_va(pa);
        if (va) { fprintf(stderr, "   data@%08x:", pa); for (int i = 0; i < 48; i++) fprintf(stderr, " %02x", rd8(va + i)); fprintf(stderr, "\n"); }
        u32 co = dspva(A_ADPCM, rr) + s * 32;
        fprintf(stderr, "   coef:"); for (int i = 0; i < 16; i++) fprintf(stderr, " %d", (s16)rd16(co + i * 2)); fprintf(stderr, "\n");
        if (s > 3) break;
    }
}
