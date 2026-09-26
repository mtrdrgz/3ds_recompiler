// y2r:u — YUV to RGB conversion unit (used for movie playback). The whole
// conversion is performed synchronously at StartConversion.
#include "services.h"
#include "pica_fmt.h"

struct Y2rBuf { u32 addr = 0, size = 0, unit = 0, gap = 0; };
static struct {
    u32 in_fmt = 0, out_fmt = 0, rotation = 0, block = 0, line_width = 0, lines = 0, std_coef = 0, alpha = 0xFF;
    bool end_irq = false;
    s16 coef[8] = {0x100, 0x166, 0xB6, 0x58, 0x1C5, -0x166F, 0x10EE, -0x1C5B};
    Y2rBuf y, u, v, yuyv, out;
    std::shared_ptr<Event> end_ev;
} Y;

static const s16 STD_COEF[4][8] = {
    {0x100, 0x166, 0xB6, 0x58, 0x1C5, -0x166F, 0x10EE, -0x1C5B},
    {0x100, 0x193, 0x77, 0x2F, 0x1DB, -0x1933, 0xA7C, -0x1D51},
    {0x12A, 0x198, 0xD0, 0x64, 0x204, -0x1BDE, 0x10F2, -0x229B},
    {0x12A, 0x1CA, 0x88, 0x36, 0x21C, -0x1F04, 0x99C, -0x2421},
};

// read byte `i` of a transfer stream honoring unit/gap
static u8 rd_stream(const Y2rBuf &b, u32 i) {
    if (!b.unit || !b.gap) return rd8(b.addr + i);
    u32 blk = i / b.unit, off = i % b.unit;
    return rd8(b.addr + blk * (b.unit + b.gap) + off);
}
static void wr_stream(const Y2rBuf &b, u32 i, u8 v) {
    if (!b.unit || !b.gap) { wr8(b.addr + i, v); return; }
    u32 blk = i / b.unit, off = i % b.unit;
    wr8(b.addr + blk * (b.unit + b.gap) + off, v);
}

static void convert() {
    u32 w = Y.line_width, h = Y.lines;
    if (!w || !h || !Y.out.addr) return;
    const s16 *k = Y.coef;
    u32 obpp = Y.out_fmt == 0 ? 4 : Y.out_fmt == 1 ? 3 : 2;
    for (u32 yy = 0; yy < h; yy++) {
        for (u32 xx = 0; xx < w; xx++) {
            int Yv, U, V;
            switch (Y.in_fmt) {
            case 0: Yv = rd_stream(Y.y, yy * w + xx); U = rd_stream(Y.u, yy * (w / 2) + xx / 2); V = rd_stream(Y.v, yy * (w / 2) + xx / 2); break;
            case 1: Yv = rd_stream(Y.y, yy * w + xx); U = rd_stream(Y.u, (yy / 2) * (w / 2) + xx / 2); V = rd_stream(Y.v, (yy / 2) * (w / 2) + xx / 2); break;
            case 2: Yv = rd_stream(Y.y, (yy * w + xx) * 2 + 1); U = rd_stream(Y.u, (yy * (w / 2) + xx / 2) * 2 + 1); V = rd_stream(Y.v, (yy * (w / 2) + xx / 2) * 2 + 1); break;
            case 3: Yv = rd_stream(Y.y, (yy * w + xx) * 2 + 1); U = rd_stream(Y.u, ((yy / 2) * (w / 2) + xx / 2) * 2 + 1); V = rd_stream(Y.v, ((yy / 2) * (w / 2) + xx / 2) * 2 + 1); break;
            default: {
                u32 base = (yy * w + (xx & ~1u)) * 2;
                Yv = rd_stream(Y.yuyv, base + (xx & 1) * 2);
                U = rd_stream(Y.yuyv, base + 1);
                V = rd_stream(Y.yuyv, base + 3);
            }
            }
            int yc = k[0] * Yv;
            int r = (yc + k[1] * V) >> 3; r += k[5]; r >>= 5;
            int g = (yc - k[2] * V - k[3] * U) >> 3; g += k[6]; g >>= 5;
            int b = (yc + k[4] * U) >> 3; b += k[7]; b >>= 5;
            r = std::clamp(r, 0, 255); g = std::clamp(g, 0, 255); b = std::clamp(b, 0, 255);
            u32 c = r | (g << 8) | (b << 16) | (Y.alpha << 24);
            u8 px[4];
            u32 fmt = Y.out_fmt == 0 ? 0 : Y.out_fmt == 1 ? 1 : Y.out_fmt == 2 ? 3 : 2;
            fb_encode(px, fmt, c);
            u32 off;
            if (Y.block) off = (((yy & ~7u) * w) + ((xx & ~7u) * 8) + morton8(xx & 7, yy & 7)) * obpp;
            else off = (yy * w + xx) * obpp;
            for (u32 i = 0; i < obpp; i++) wr_stream(Y.out, off + i, px[i]);
        }
    }
}

struct Y2rService : Service {
    Y2rService() : Service("y2r:u") {}
    void handle(Ipc &ipc) override {
        u32 a1 = ipc.p(1), a2 = ipc.p(2), a3 = ipc.p(3), a4 = ipc.p(4);
        auto setbuf = [&](Y2rBuf &b) { b = {a1, a2, a3, a4}; ipc.reply(1, 0); };
        auto get = [&](u32 v) { ipc.reply(2, 0); ipc.w(2, v); };
        switch (ipc.cmd()) {
        case 0x01: Y.in_fmt = a1 & 0xFF; ipc.reply(1, 0); break;
        case 0x02: get(Y.in_fmt); break;
        case 0x03: Y.out_fmt = a1 & 0xFF; ipc.reply(1, 0); break;
        case 0x04: get(Y.out_fmt); break;
        case 0x05: Y.rotation = a1 & 0xFF; ipc.reply(1, 0); break;
        case 0x06: get(Y.rotation); break;
        case 0x07: Y.block = a1 & 0xFF; ipc.reply(1, 0); break;
        case 0x08: get(Y.block); break;
        case 0x09: case 0x0B: case 0x24: ipc.reply(1, 0); break;
        case 0x0A: case 0x0C: get(0); break;
        case 0x0D: Y.end_irq = a1 & 1; ipc.reply(1, 0); break;
        case 0x0E: get(Y.end_irq); break;
        case 0x0F:
            if (!Y.end_ev) Y.end_ev = std::make_shared<Event>(0);
            ipc.reply(1, 2); ipc.w(2, 0); ipc.w(3, g_k.new_handle(Y.end_ev)); break;
        case 0x10: setbuf(Y.y); break;
        case 0x11: setbuf(Y.u); break;
        case 0x12: setbuf(Y.v); break;
        case 0x13: setbuf(Y.yuyv); break;
        case 0x14: case 0x15: case 0x16: case 0x17: case 0x19: get(1); break;
        case 0x18: setbuf(Y.out); break;
        case 0x1A: Y.line_width = a1 & 0xFFFF; ipc.reply(1, 0); break;
        case 0x1B: get(Y.line_width); break;
        case 0x1C: Y.lines = a1 & 0xFFFF; ipc.reply(1, 0); break;
        case 0x1D: get(Y.lines); break;
        case 0x1E: memcpy(Y.coef, gp(ipc.buf + 4), 16); ipc.reply(1, 0); break;
        case 0x1F: ipc.reply(5, 0); memcpy(gp(ipc.buf + 8), Y.coef, 16); break;
        case 0x20: Y.std_coef = a1 & 3; memcpy(Y.coef, STD_COEF[Y.std_coef], 16); ipc.reply(1, 0); break;
        case 0x21: ipc.reply(5, 0); memcpy(gp(ipc.buf + 8), STD_COEF[a1 & 3], 16); break;
        case 0x22: Y.alpha = a1 & 0xFF; ipc.reply(1, 0); break;
        case 0x23: get(Y.alpha); break;
        case 0x25: ipc.reply(9, 0); break;
        case 0x26:  // StartConversion
            convert();
            if (Y.end_ev) Y.end_ev->signal();
            ipc.reply(1, 0);
            break;
        case 0x27: ipc.reply(1, 0); break;
        case 0x28: get(0); break;
        case 0x29: {  // SetPackageParameter
            u8 b[0x20]; memcpy(b, gp(ipc.buf + 4), sizeof b);
            Y.in_fmt = b[0]; Y.out_fmt = b[1]; Y.rotation = b[2]; Y.block = b[3];
            Y.line_width = b[4] | (b[5] << 8); Y.lines = b[6] | (b[7] << 8);
            Y.std_coef = b[8] & 3; memcpy(Y.coef, STD_COEF[Y.std_coef], 16);
            Y.alpha = b[10] | (b[11] << 8);
            ipc.reply(1, 0);
            break;
        }
        case 0x2A: get(1); break;
        case 0x2B: case 0x2C: ipc.reply(1, 0); break;
        default: unknown(ipc);
        }
    }
};

void register_y2r() { services_register("y2r:u", [] { return std::make_shared<Y2rService>(); }); }
