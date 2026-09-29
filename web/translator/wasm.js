// Minimal WebAssembly binary writer for the ARM -> WASM translator: growable
// byte buffers, LEB128, the opcodes the emitter uses, and module assembly.
// No dependency on any toolchain, so it runs unchanged in a browser worker.

export class ByteBuf {
    constructor(cap = 256) { this.a = new Uint8Array(cap); this.n = 0; }
    grow(extra) {
        if (this.n + extra <= this.a.length) return;
        let cap = this.a.length * 2;
        while (cap < this.n + extra) cap *= 2;
        const b = new Uint8Array(cap);
        b.set(this.a.subarray(0, this.n));
        this.a = b;
    }
    u8(x) { if (this.n >= this.a.length) this.grow(1); this.a[this.n++] = x; return this; }
    u8s(x, y) { this.grow(2); this.a[this.n++] = x; this.a[this.n++] = y; return this; }
    // unsigned LEB128 (value < 2^32)
    uleb(x) {
        this.grow(5);
        x >>>= 0;
        do {
            let b = x & 0x7f;
            x >>>= 7;
            if (x) b |= 0x80;
            this.a[this.n++] = b;
        } while (x);
        return this;
    }
    // signed LEB128 (32-bit two's complement value)
    sleb(x) {
        this.grow(5);
        x |= 0;
        for (;;) {
            const b = x & 0x7f;
            x >>= 7;
            if ((x === 0 && !(b & 0x40)) || (x === -1 && (b & 0x40))) { this.a[this.n++] = b; return this; }
            this.a[this.n++] = b | 0x80;
        }
    }
    sleb64(x) {   // BigInt
        this.grow(10);
        x = BigInt.asIntN(64, x);
        for (;;) {
            const b = Number(x & 0x7fn);
            x >>= 7n;
            if ((x === 0n && !(b & 0x40)) || (x === -1n && (b & 0x40))) { this.a[this.n++] = b; return this; }
            this.a[this.n++] = b | 0x80;
        }
    }
    f32(x) { this.grow(4); new DataView(this.a.buffer).setFloat32(this.n, x, true); this.n += 4; return this; }
    f64(x) { this.grow(8); new DataView(this.a.buffer).setFloat64(this.n, x, true); this.n += 8; return this; }
    bytes(src, len = src.length) {
        this.grow(len);
        this.a.set(src.subarray ? src.subarray(0, len) : src.slice(0, len), this.n);
        this.n += len;
        return this;
    }
    buf(other) { return this.bytes(other.a, other.n); }
    name(s) { const e = new TextEncoder().encode(s); this.uleb(e.length); return this.bytes(e); }
    get length() { return this.n; }
    view() { return this.a.subarray(0, this.n); }
    reset() { this.n = 0; }
}

export const T = { i32: 0x7f, i64: 0x7e, f32: 0x7d, f64: 0x7c };

// opcodes
export const OP = {
    unreachable: 0x00, nop: 0x01, block: 0x02, loop: 0x03, if: 0x04, else: 0x05, end: 0x0b,
    br: 0x0c, br_if: 0x0d, br_table: 0x0e, return: 0x0f, call: 0x10, call_indirect: 0x11,
    drop: 0x1a, select: 0x1b,
    local_get: 0x20, local_set: 0x21, local_tee: 0x22, global_get: 0x23, global_set: 0x24,
    i32_load: 0x28, i64_load: 0x29, f32_load: 0x2a, f64_load: 0x2b,
    i32_load8_s: 0x2c, i32_load8_u: 0x2d, i32_load16_s: 0x2e, i32_load16_u: 0x2f,
    i32_store: 0x36, i64_store: 0x37, f32_store: 0x38, f64_store: 0x39, i32_store8: 0x3a, i32_store16: 0x3b,
    i32_const: 0x41, i64_const: 0x42, f32_const: 0x43, f64_const: 0x44,
    i32_eqz: 0x45, i32_eq: 0x46, i32_ne: 0x47, i32_lt_s: 0x48, i32_lt_u: 0x49, i32_gt_s: 0x4a, i32_gt_u: 0x4b,
    i32_le_s: 0x4c, i32_le_u: 0x4d, i32_ge_s: 0x4e, i32_ge_u: 0x4f,
    i64_eqz: 0x50, i64_eq: 0x51, i64_ne: 0x52, i64_lt_s: 0x53, i64_lt_u: 0x54,
    i32_clz: 0x67, i32_ctz: 0x68, i32_popcnt: 0x69,
    i32_add: 0x6a, i32_sub: 0x6b, i32_mul: 0x6c, i32_div_s: 0x6d, i32_div_u: 0x6e, i32_rem_s: 0x6f, i32_rem_u: 0x70,
    i32_and: 0x71, i32_or: 0x72, i32_xor: 0x73, i32_shl: 0x74, i32_shr_s: 0x75, i32_shr_u: 0x76,
    i32_rotl: 0x77, i32_rotr: 0x78,
    i64_add: 0x7c, i64_sub: 0x7d, i64_mul: 0x7e, i64_and: 0x83, i64_or: 0x84, i64_xor: 0x85,
    i64_shl: 0x86, i64_shr_s: 0x87, i64_shr_u: 0x88,
    i32_wrap_i64: 0xa7, i64_extend_i32_s: 0xac, i64_extend_i32_u: 0xad,
    i32_extend8_s: 0xc0, i32_extend16_s: 0xc1,
    atomic_prefix: 0xfe,
};

export const EMPTY = 0x40;   // empty block type

// module assembly ------------------------------------------------------------
function section(out, id, body) {
    out.u8(id).uleb(body.length).buf(body);
}

// A function type: params / results are arrays of T.*
function functype(b, params, results) {
    b.u8(0x60).uleb(params.length);
    for (const p of params) b.u8(p);
    b.uleb(results.length);
    for (const r of results) b.u8(r);
}

// spec:
//   types:   [{params, results}]
//   imports: [{module, name, kind:'func'|'global'|'memory', ...}]   (order defines index spaces)
//   funcs:   [{type, body: ByteBuf (locals vector + code + end)}]
//   tableSize, exports: [{name, kind:'func'|'table', index}]
//   data: [{offset, bytes}]  (active segments, memory 0)
export function assemble(spec) {
    const out = new ByteBuf(1 << 20);
    out.bytes([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);

    let b = new ByteBuf(); b.uleb(spec.types.length);
    for (const t of spec.types) functype(b, t.params, t.results);
    section(out, 1, b);

    b = new ByteBuf(); b.uleb(spec.imports.length);
    for (const im of spec.imports) {
        b.name(im.module).name(im.name);
        if (im.kind === 'func') b.u8(0x00).uleb(im.type);
        else if (im.kind === 'global') b.u8(0x03).u8(im.valtype).u8(0);   // immutable
        else if (im.kind === 'memory') {   // shared memory with a maximum
            b.u8(0x02).u8(0x03).uleb(im.min).uleb(im.max);
        }
    }
    section(out, 2, b);

    b = new ByteBuf(); b.uleb(spec.funcs.length);
    for (const f of spec.funcs) b.uleb(f.type);
    section(out, 3, b);

    b = new ByteBuf(); b.uleb(1).u8(0x70).u8(0x00).uleb(spec.tableSize);   // one funcref table
    section(out, 4, b);

    b = new ByteBuf(); b.uleb(spec.exports.length);
    for (const e of spec.exports) b.name(e.name).u8(e.kind === 'func' ? 0 : 1).uleb(e.index);
    section(out, 7, b);

    // active element segment: table[0..n) = funcs [firstFunc .. firstFunc+n)
    if (spec.elem) {
        b = new ByteBuf(); b.uleb(1).uleb(0).u8(OP.i32_const).sleb(0).u8(OP.end).uleb(spec.elem.length);
        for (const fi of spec.elem) b.uleb(fi);
        section(out, 9, b);
    }

    // DataCount before code is only needed for passive data; we use active data only.
    b = new ByteBuf(1 << 20); b.uleb(spec.funcs.length);
    for (const f of spec.funcs) b.uleb(f.body.length).buf(f.body);
    section(out, 10, b);

    if (spec.data && spec.data.length) {
        b = new ByteBuf(); b.uleb(spec.data.length);
        for (const d of spec.data) b.uleb(0).u8(OP.i32_const).sleb(d.offset).u8(OP.end).uleb(d.bytes.length).bytes(d.bytes);
        section(out, 11, b);
    }
    return out.view();
}
