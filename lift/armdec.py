"""ARMv6K ARM/Thumb decoder that emits C++ statements for the recompiler.

Every decode returns an Insn:
  size     2 or 4
  cond     0..14 (14 = always)
  body     C++ statement(s) executed when the condition holds. Control
           transfers are written with the placeholders
             JMP(<pc>)   static target (pc bit0 = thumb)
             EXIT(<expr>) dynamic target expression (bit0 = thumb)
           which the chunk emitter lowers to goto / return.
  targets  static successor pcs (for discovery), not incl. fall-through
  falls    whether execution can continue at the next instruction
  call     True for BL/BLX (return site = next instruction)
  flags_w  whether it writes NZCV (informational)
  bad      unsupported: the emitter falls back to the interpreter
"""

class Insn:
    __slots__ = ('addr', 'thumb', 'size', 'cond', 'body', 'targets', 'falls', 'call', 'bad', 'text', 'table', 'ptr_loads')
    def __init__(self, addr, thumb, size):
        self.addr = addr; self.thumb = thumb; self.size = size
        self.cond = 14; self.body = ''; self.targets = []; self.falls = True
        self.call = False; self.bad = None; self.text = ''; self.table = None
        self.ptr_loads = []

CONDS = ['eq','ne','cs','cc','mi','pl','vs','vc','hi','ls','ge','lt','gt','le','al']
COND_EXPR = ['fz', '!fz', 'fc', '!fc', 'fn', '!fn', 'fv', '!fv', '(fc && !fz)', '(!fc || fz)',
             '(fn == fv)', '(fn != fv)', '(!fz && fn == fv)', '(fz || fn != fv)', '1']

def R(n):
    return 'r%d' % n

def sx(v, bits):
    m = 1 << (bits - 1)
    return (v & (m - 1)) - (v & m)

def hx(v):
    return '0x%xu' % (v & 0xffffffff)

def ror(v, s):
    s &= 31
    return ((v >> s) | (v << (32 - s))) & 0xffffffff if s else v

# ------------------------------------------------------------------ helpers
def reg_read(n, pcval):
    return hx(pcval) if n == 15 else R(n)

def write_reg(ins, n, expr, interwork=False, arm_alu=False):
    """assign expr to register n; writes to PC become exits"""
    if n == 15:
        if interwork:
            return 'EXIT(%s);' % expr
        # ARMv6 ALU/LDR-without-interwork writes: stay in current state
        if ins.thumb:
            return 'EXIT((%s) | 1u);' % expr
        return 'EXIT((%s) & ~3u);' % expr
    return '%s = %s;' % (R(n), expr)

def nz(res):
    return 'fn = %s >> 31; fz = %s == 0;' % (res, res)

SHIFT_NAMES = ['lsl', 'lsr', 'asr', 'ror']

def imm_shift(rm_expr, typ, amt, want_carry):
    """shift by immediate. returns (expr, carry_expr or None)"""
    if typ == 0:  # LSL
        if amt == 0:
            return rm_expr, None
        return '(%s << %d)' % (rm_expr, amt), ('((%s >> %d) & 1)' % (rm_expr, 32 - amt))
    if typ == 1:  # LSR
        if amt == 0: amt = 32
        if amt == 32:
            return '0u', '(%s >> 31)' % rm_expr
        return '(%s >> %d)' % (rm_expr, amt), '((%s >> %d) & 1)' % (rm_expr, amt - 1)
    if typ == 2:  # ASR
        if amt == 0: amt = 32
        if amt == 32:
            return '(u32)((s32)%s >> 31)' % rm_expr, '(%s >> 31)' % rm_expr
        return '(u32)((s32)%s >> %d)' % (rm_expr, amt), '((%s >> %d) & 1)' % (rm_expr, amt - 1)
    # ROR / RRX
    if amt == 0:
        return '((fc << 31) | (%s >> 1))' % rm_expr, '(%s & 1)' % rm_expr
    return 'ror32(%s, %d)' % (rm_expr, amt), '((%s >> %d) & 1)' % (rm_expr, amt - 1)

def reg_shift(rm_expr, typ, rs_expr):
    """register-specified shift: returns (expr using helper that updates `sc`)"""
    fn = ['lsl_rc', 'lsr_rc', 'asr_rc', 'ror_rc'][typ]
    return '%s(%s, %s, sc)' % (fn, rm_expr, rs_expr)

# ======================================================================= ARM
def dec_arm(addr, w):
    ins = Insn(addr, False, 4)
    cond = w >> 28
    pc = addr + 8
    ins.text = '%08x' % w
    if cond == 15:
        return dec_arm_uncond(ins, w)
    ins.cond = cond
    op = (w >> 25) & 7
    try:
        if op == 0:
            if (w & 0x90) == 0x90:
                if (w >> 5) & 3 == 0:
                    dec_arm_mul_swp(ins, w, pc)
                else:
                    dec_arm_extra_ls(ins, w, pc)
            elif (w >> 23) & 3 == 2 and not (w >> 20) & 1:
                dec_arm_misc(ins, w, pc)
            else:
                dec_arm_dp(ins, w, pc)
        elif op == 1:
            if (w >> 23) & 3 == 2 and not (w >> 20) & 1:
                if (w >> 21) & 1:   # MSR imm / hints
                    mask = (w >> 16) & 0xF
                    if mask == 0:
                        ins.body = ';'   # nop / yield / wfe / wfi / sev
                    else:
                        v = ror(w & 0xFF, ((w >> 8) & 0xF) * 2)
                        ins.body = msr_body(mask, hx(v))
                else:
                    ins.bad = 'movw/movt'
            else:
                dec_arm_dp(ins, w, pc)
        elif op == 2 or (op == 3 and not (w >> 4) & 1):
            dec_arm_ls(ins, w, pc)
        elif op == 3:
            dec_arm_media(ins, w, pc)
        elif op == 4:
            dec_arm_ldm(ins, w, pc)
        elif op == 5:
            off = sx(w & 0xFFFFFF, 24) << 2
            t = (pc + off) & 0xffffffff
            if (w >> 24) & 1:
                ins.call = True
                ins.body = 'r14 = %s; JMP(%s);' % (hx(addr + 4), hx(t))
            else:
                ins.body = 'JMP(%s);' % hx(t)
                if cond == 14: ins.falls = False
            ins.targets = [t]
        elif op == 6:
            dec_arm_ldc(ins, w, pc)
        else:
            if (w >> 24) & 1:
                ins.body = 'SVC(%s, %s);' % (hx(w & 0xFFFFFF), hx(addr + 4))
            else:
                dec_arm_cp(ins, w, pc)
    except NotImplementedError as e:
        ins.bad = str(e) or 'unimpl'
    if ins.cond == 14 and ins.body.startswith('EXIT(') and ins.body.count(';') == 1:
        ins.falls = False
    return ins

def msr_body(mask, v):
    out = []
    if mask & 8:
        out.append('fn = (%s) >> 31; fz = ((%s) >> 30) & 1; fc = ((%s) >> 29) & 1; fv = ((%s) >> 28) & 1; c.q = ((%s) >> 27) & 1;' % (v, v, v, v, v))
    if mask & 4:
        out.append('c.ge = ((%s) >> 16) & 0xF;' % v)
    return ' '.join(out) or ';'

def dec_arm_uncond(ins, w):
    ins.cond = 14
    if (w & 0x0E000000) == 0x0A000000:   # BLX imm
        off = (sx(w & 0xFFFFFF, 24) << 2) | (((w >> 24) & 1) << 1)
        t = (ins.addr + 8 + off) & 0xffffffff
        ins.call = True
        ins.body = 'r14 = %s; JMP(%s);' % (hx(ins.addr + 4), hx(t | 1))
        ins.targets = [t | 1]
        return ins
    if (w & 0x0D70F000) == 0x0550F000:   # PLD
        ins.body = ';'; return ins
    if w == 0xF57FF01F:                  # CLREX
        ins.body = 'c.excl_valid = 0;'; return ins
    if (w & 0xFFFFFFF0) in (0xF57FF040, 0xF57FF050, 0xF57FF060):   # DSB/DMB/ISB
        ins.body = ';'; return ins
    if (w & 0xFFF1FE20) == 0xF1000000:   # CPS
        ins.body = ';'; return ins
    ins.bad = 'uncond %08x' % w
    return ins

def operand2(ins, w, pc, want_carry):
    """returns (setup_code, expr, carry_expr or None)"""
    if (w >> 25) & 1:
        rot = ((w >> 8) & 0xF) * 2
        v = ror(w & 0xFF, rot)
        carry = ('%du' % (v >> 31)) if rot else None
        return '', hx(v), carry
    rm = w & 0xF
    typ = (w >> 5) & 3
    if (w >> 4) & 1:  # register shift
        rs = (w >> 8) & 0xF
        rm_e = reg_read(rm, pc + 4)
        rs_e = reg_read(rs, pc + 4) + ' & 0xFF'
        if want_carry:
            return 'u32 sc = fc; u32 so = %s;' % reg_shift(rm_e, typ, '(' + rs_e + ')'), 'so', 'sc'
        fn = ['lsl_r', 'lsr_r', 'asr_r', 'ror_r'][typ]
        return '', '%s(%s, %s)' % (fn, rm_e, '(' + rs_e + ')'), None
    amt = (w >> 7) & 0x1F
    rm_e = reg_read(rm, pc)
    e, c = imm_shift(rm_e, typ, amt, want_carry)
    return '', e, c

def dec_arm_dp(ins, w, pc):
    opc = (w >> 21) & 0xF
    s = (w >> 20) & 1
    rn = (w >> 16) & 0xF
    rd = (w >> 12) & 0xF
    logical = opc in (0, 1, 8, 9, 12, 13, 14, 15)
    reg_shift_form = not ((w >> 25) & 1) and ((w >> 4) & 1)
    setup, op2, carry = operand2(ins, w, pc, s and logical)
    rn_e = reg_read(rn, pc + 4 if reg_shift_form else pc)
    body = []
    if setup: body.append(setup)
    if s and rd == 15 and opc not in (8, 9, 10, 11):
        raise NotImplementedError('exception return')
    if logical:
        expr = {0: '%s & %s' % (rn_e, op2), 1: '%s ^ %s' % (rn_e, op2), 8: '%s & %s' % (rn_e, op2),
                9: '%s ^ %s' % (rn_e, op2), 12: '%s | %s' % (rn_e, op2), 13: op2,
                14: '%s & ~(%s)' % (rn_e, op2), 15: '~(%s)' % op2}[opc]
        if opc in (8, 9):
            body.append('{ u32 res = %s; %s%s }' % (expr, nz('res'), (' fc = %s;' % carry) if carry else ''))
        elif s:
            body.append('{ u32 res = %s; %s%s %s }' % (expr, nz('res'), (' fc = %s;' % carry) if carry else '', write_reg(ins, rd, 'res')))
        else:
            body.append(write_reg(ins, rd, expr))
    else:
        # arithmetic
        a, b, cin = {2: (rn_e, '~(%s)' % op2, '1'), 3: (op2, '~(%s)' % rn_e, '1'), 4: (rn_e, op2, '0'),
                     5: (rn_e, op2, 'fc'), 6: (rn_e, '~(%s)' % op2, 'fc'), 7: (op2, '~(%s)' % rn_e, 'fc'),
                     10: (rn_e, '~(%s)' % op2, '1'), 11: (rn_e, op2, '0')}[opc]
        if opc in (10, 11):
            body.append('{ adds(%s, %s, %s, fn, fz, fc, fv); }' % (a, b, cin))
        elif s:
            body.append('{ u32 res = adds(%s, %s, %s, fn, fz, fc, fv); %s }' % (a, b, cin, write_reg(ins, rd, 'res')))
        else:
            if cin == '0': e = '%s + %s' % (a, b)
            elif cin == '1': e = '%s + %s + 1u' % (a, b)
            else: e = '%s + %s + fc' % (a, b)
            body.append(write_reg(ins, rd, e))
    ins.body = ' '.join(body)
    if rd == 15 and opc not in (8, 9, 10, 11):
        # computed jump: detect add pc, pc, rX, lsl #2 jump tables
        if opc == 4 and rn == 15 and not (w >> 25) & 1 and not (w >> 4) & 1 and ((w >> 5) & 3) == 0 and ((w >> 7) & 0x1F) == 2:
            ins.table = ('addpc', w & 0xF)
        if ins.cond == 14: ins.falls = False
        if opc == 13 and not (w >> 25) & 1 and (w & 0xFF0) == 0 and (w & 0xF) == 14:
            pass  # mov pc, lr: plain return

def dec_arm_misc(ins, w, pc):
    op = (w >> 21) & 3
    op2 = (w >> 4) & 0xF
    rd = (w >> 12) & 0xF
    rm = w & 0xF
    if op2 == 0:
        if op & 1:   # MSR reg
            mask = (w >> 16) & 0xF
            if (w >> 22) & 1: raise NotImplementedError('msr spsr')
            ins.body = msr_body(mask, R(rm))
        else:        # MRS
            if (w >> 22) & 1: raise NotImplementedError('mrs spsr')
            ins.body = '%s = (fn << 31) | (fz << 30) | (fc << 29) | (fv << 28) | (c.q << 27) | (c.ge << 16) | 0x10u;' % R(rd)
        return
    if op2 == 1 and op == 1 or op2 == 2 and op == 1:   # BX / BXJ
        ins.body = 'EXIT(%s);' % reg_read(rm, pc)
        if ins.cond == 14: ins.falls = False
        return
    if op2 == 1 and op == 3:   # CLZ
        ins.body = '%s = clz32(%s);' % (R(rd), R(rm)); return
    if op2 == 3 and op == 1:   # BLX reg
        ins.call = True
        ins.body = '{ u32 t = %s; r14 = %s; EXIT(t); }' % (reg_read(rm, pc), hx(ins.addr + 4)); return
    if op2 == 5:   # QADD/QSUB/QDADD/QDSUB
        rn = (w >> 16) & 0xF
        f = ['qadd', 'qsub', 'qdadd', 'qdsub'][op]
        ins.body = '%s = %s(c, %s, %s);' % (R(rd), f, R(rm), R(rn)); return
    if op2 == 7 and op == 1:
        ins.body = 'TRAP("bkpt");'; return
    if (op2 & 9) == 8:   # signed multiplies halfword
        x = (w >> 5) & 1; y = (w >> 6) & 1
        rdh = (w >> 16) & 0xF; rn = (w >> 12) & 0xF; rs = (w >> 8) & 0xF
        hm = '((s32)(s16)(%s%s))' % (R(rm), ' >> 16' if x else '')
        hs = '((s32)(s16)(%s%s))' % (R(rs), ' >> 16' if y else '')
        if op == 0:   # SMLAxy
            ins.body = '%s = smla_q(c, (u32)(%s * %s), %s);' % (R(rdh), hm, hs, R(rn))
        elif op == 1:
            hs2 = '((s64)(s16)(%s%s))' % (R(rs), ' >> 16' if y else '')
            if x:   # SMULWy
                ins.body = '%s = (u32)(((s64)(s32)%s * %s) >> 16);' % (R(rdh), R(rm), hs2)
            else:   # SMLAWy
                ins.body = '%s = smla_q(c, (u32)(((s64)(s32)%s * %s) >> 16), %s);' % (R(rdh), R(rm), hs2, R(rn))
        elif op == 2:   # SMLALxy
            ins.body = '{ s64 acc = (s64)(((u64)%s << 32) | %s); acc += (s64)(%s * %s); %s = (u32)acc; %s = (u32)((u64)acc >> 32); }' % (
                R(rdh), R(rn), hm, hs, R(rn), R(rdh))
        else:   # SMULxy
            ins.body = '%s = (u32)(%s * %s);' % (R(rdh), hm, hs)
        return
    raise NotImplementedError('misc %x %x' % (op, op2))

def dec_arm_mul_swp(ins, w, pc):
    if (w >> 24) & 1:   # swap / exclusive
        rn = (w >> 16) & 0xF; rt = (w >> 12) & 0xF; rt2 = w & 0xF
        if (w >> 23) & 1:
            kind = (w >> 21) & 3
            load = (w >> 20) & 1
            sz = {0: 32, 1: 64, 2: 8, 3: 16}[kind]
            if load:
                if sz == 64:
                    ins.body = 'c.excl_addr = %s; c.excl_valid = 1; c.excl_epoch = excl_epoch_now(); %s = RD32(%s); %s = RD32(%s + 4);' % (R(rn), R(rt), R(rn), R(rt + 1), R(rn))
                else:
                    ins.body = 'c.excl_addr = %s; c.excl_valid = 1; c.excl_epoch = excl_epoch_now(); %s = RD%d(%s);' % (R(rn), R(rt), sz, R(rn))
            else:
                if sz == 64:
                    st = 'WR32(%s, %s); WR32(%s + 4, %s);' % (R(rn), R(rt2), R(rn), R(rt2 + 1))
                else:
                    st = 'WR%d(%s, %s);' % (sz, R(rn), R(rt2))
                ins.body = 'if (c.excl_valid && c.excl_addr == %s && c.excl_epoch == excl_epoch_now()) { %s %s = 0; } else %s = 1; c.excl_valid = 0;' % (R(rn), st, R(rt), R(rt))
            return
        b = (w >> 22) & 1
        if b: ins.body = '{ u32 a = %s; u32 t = RD8(a); WR8(a, %s); %s = t; }' % (R(rn), R(rt2), R(rt))
        else: ins.body = '{ u32 a = %s; u32 t = RD32(a); WR32(a, %s); %s = t; }' % (R(rn), R(rt2), R(rt))
        return
    op = (w >> 21) & 7
    s = (w >> 20) & 1
    rd = (w >> 16) & 0xF; rn = (w >> 12) & 0xF; rs = (w >> 8) & 0xF; rm = w & 0xF
    if op == 0:
        ins.body = '%s = %s * %s;' % (R(rd), R(rm), R(rs)) + (' ' + nz(R(rd)) if s else '')
    elif op == 1:
        ins.body = '%s = %s * %s + %s;' % (R(rd), R(rm), R(rs), R(rn)) + (' ' + nz(R(rd)) if s else '')
    elif op == 2:   # UMAAL
        ins.body = '{ u64 r = (u64)%s * %s + %s + %s; %s = (u32)r; %s = (u32)(r >> 32); }' % (R(rm), R(rs), R(rn), R(rd), R(rn), R(rd))
    elif op == 3:   # MLS
        ins.body = '%s = %s - %s * %s;' % (R(rd), R(rn), R(rm), R(rs))
    else:
        signed = op >= 6
        acc = op & 1
        mul = '(u64)((s64)(s32)%s * (s64)(s32)%s)' % (R(rm), R(rs)) if signed else '((u64)%s * %s)' % (R(rm), R(rs))
        accs = ' + (((u64)%s << 32) | %s)' % (R(rd), R(rn)) if acc else ''
        ins.body = '{ u64 r = %s%s; %s = (u32)r; %s = (u32)(r >> 32);%s }' % (mul, accs, R(rn), R(rd),
            (' fn = (u32)(r >> 63); fz = r == 0;' if s else ''))

def ea_code(ins, rn_e, off_e, p, u, wback, rn):
    """returns (addr_expr_setup, writeback_code) as strings using var a"""
    sign = '+' if u else '-'
    if p:
        setup = 'u32 a = %s %s %s;' % (rn_e, sign, off_e)
        wb = ('%s = a;' % R(rn)) if wback else ''
    else:
        setup = 'u32 a = %s;' % rn_e
        wb = '%s = a %s %s;' % (R(rn), sign, off_e)
    return setup, wb

def dec_arm_extra_ls(ins, w, pc):
    p = (w >> 24) & 1; u = (w >> 23) & 1; i = (w >> 22) & 1; wb = (w >> 21) & 1; l = (w >> 20) & 1
    rn = (w >> 16) & 0xF; rt = (w >> 12) & 0xF
    sh = (w >> 5) & 3
    off = hx(((w >> 4) & 0xF0) | (w & 0xF)) if i else R(w & 0xF)
    rn_e = reg_read(rn, pc)
    setup, wbc = ea_code(ins, rn_e, off, p, u, wb, rn)
    if rn == 15 and i and p:
        lit = (pc + (1 if u else -1) * (((w >> 4) & 0xF0) | (w & 0xF))) & 0xffffffff
        ins.ptr_loads.append(lit)
    if l:
        if sh == 1: ld = '%s = RD16(a);' % R(rt)
        elif sh == 2: ld = '%s = (u32)(s32)(s8)RD8(a);' % R(rt)
        else: ld = '%s = (u32)(s32)(s16)RD16(a);' % R(rt)
        if rt == 15: raise NotImplementedError('ldrh pc')
        ins.body = '{ %s %s %s }' % (setup, wbc if rn != rt else '', ld) if rn != rt else '{ %s %s %s }' % (setup, '', ld)
        # writeback before load (load wins if rn==rt)
        ins.body = '{ %s %s %s }' % (setup, wbc, ld) if rn != rt else '{ %s %s }' % (setup, ld)
    else:
        if sh == 1:
            ins.body = '{ %s WR16(a, %s); %s }' % (setup, reg_read(rt, pc), wbc)
        elif sh == 2:   # LDRD
            if rt & 1 or rt == 14: raise NotImplementedError('ldrd odd')
            ld = 'u32 v0 = RD32(a), v1 = RD32(a + 4); %s %s = v0; %s = v1;' % (wbc, R(rt), R(rt + 1))
            ins.body = '{ %s %s }' % (setup, ld)
        else:           # STRD
            if rt & 1 or rt == 14: raise NotImplementedError('strd odd')
            ins.body = '{ %s WR32(a, %s); WR32(a + 4, %s); %s }' % (setup, R(rt), R(rt + 1), wbc)

def dec_arm_ls(ins, w, pc):
    reg = (w >> 25) & 1
    p = (w >> 24) & 1; u = (w >> 23) & 1; b = (w >> 22) & 1; wb = (w >> 21) & 1; l = (w >> 20) & 1
    rn = (w >> 16) & 0xF; rt = (w >> 12) & 0xF
    if reg:
        rm = w & 0xF
        e, _ = imm_shift(reg_read(rm, pc), (w >> 5) & 3, (w >> 7) & 0x1F, False)
        off = e
    else:
        off = hx(w & 0xFFF)
        if rn == 15 and p:
            ins.ptr_loads.append((pc + (w & 0xFFF) * (1 if u else -1)) & 0xffffffff)
    rn_e = reg_read(rn, pc)
    wback = wb or not p
    setup, wbc = ea_code(ins, rn_e, off, p, u, wb, rn)
    if l:
        ld = ('RD8(a)' if b else 'RD32(a)')
        if rt == 15:
            if b: raise NotImplementedError('ldrb pc')
            ins.body = '{ %s u32 v = %s; %s EXIT(v); }' % (setup, ld, wbc if wback else '')
            if ins.cond == 14: ins.falls = False
            if reg and rn == 15 and p and ((w >> 5) & 3) == 0 and ((w >> 7) & 0x1F) == 2:
                ins.table = ('ldrpc', w & 0xF)
        elif wback and rn == rt:
            ins.body = '{ %s %s = %s; }' % (setup, R(rt), ld)
        else:
            ins.body = '{ %s %s %s = %s; }' % (setup, wbc if wback else '', R(rt), ld)
    else:
        v = reg_read(rt, pc)
        ins.body = '{ %s %s(a, %s); %s }' % (setup, 'WR8' if b else 'WR32', v, wbc if wback else '')

def dec_arm_ldm(ins, w, pc):
    p = (w >> 24) & 1; u = (w >> 23) & 1; s = (w >> 22) & 1; wb = (w >> 21) & 1; l = (w >> 20) & 1
    rn = (w >> 16) & 0xF
    regs = [i for i in range(16) if (w >> i) & 1]
    n = len(regs)
    if n == 0: raise NotImplementedError('ldm empty')
    if s and not (l and 15 in regs): raise NotImplementedError('ldm user regs')
    if s: raise NotImplementedError('ldm exc return')
    if u: start = '%s + 4' % R(rn) if p else R(rn)
    else: start = '%s - %d' % (R(rn), 4 * n) if p else '%s - %d' % (R(rn), 4 * n - 4)
    newbase = '%s %s %d' % (R(rn), '+' if u else '-', 4 * n)
    body = ['u32 a = %s;' % start]
    if l:
        loads = []
        for k, r in enumerate(regs):
            if r == 15: loads.append('u32 tpc = RD32(a + %d);' % (4 * k))
            elif r == rn: loads.append('u32 nb = RD32(a + %d);' % (4 * k))
            else: loads.append('%s = RD32(a + %d);' % (R(r), 4 * k))
        body += loads
        if wb and rn not in regs: body.append('%s = %s;' % (R(rn), newbase))
        if rn in regs: body.append('%s = nb;' % R(rn))
        if 15 in regs:
            body.append('EXIT(tpc);')
            if ins.cond == 14: ins.falls = False
    else:
        for k, r in enumerate(regs):
            body.append('WR32(a + %d, %s);' % (4 * k, reg_read(r, pc)))
        if wb: body.append('%s = %s;' % (R(rn), newbase))
    ins.body = '{ ' + ' '.join(body) + ' }'

def dec_arm_media(ins, w, pc):
    op1 = (w >> 20) & 0x1F
    op2 = (w >> 5) & 7
    rd = (w >> 12) & 0xF; rn = (w >> 16) & 0xF; rm = w & 0xF; rs = (w >> 8) & 0xF
    if (op1 >> 3) == 0:   # parallel add/sub
        kind = op1 & 7
        pre = {1: 's', 2: 'q', 3: 'sh', 5: 'u', 6: 'uq', 7: 'uh'}.get(kind)
        opn = {0: 'add16', 1: 'asx', 2: 'sax', 3: 'sub16', 4: 'add8', 7: 'sub8'}.get(op2)
        if not pre or not opn: raise NotImplementedError('parallel')
        ins.body = '%s = par_%s%s(c, %s, %s);' % (R(rd), pre, opn, R(rn), R(rm))
        return
    if (op1 >> 3) == 1:
        o = op1 & 7
        rot = ((w >> 10) & 3) * 8
        rme = 'ror32(%s, %d)' % (R(rm), rot) if rot else R(rm)
        if o == 0 and not (op2 & 1):   # PKH
            sh = (w >> 7) & 0x1F
            if (w >> 6) & 1:   # PKHTB: top from Rn, bottom from Rm ASR
                e, _ = imm_shift(R(rm), 2, sh, False)
                ins.body = '%s = (%s & 0xFFFF0000u) | ((%s) & 0xFFFFu);' % (R(rd), R(rn), e)
            else:
                e = '(%s << %d)' % (R(rm), sh) if sh else R(rm)
                ins.body = '%s = (%s & 0xFFFFu) | ((%s) & 0xFFFF0000u);' % (R(rd), R(rn), e)
            return
        if o == 0 and op2 == 3:   # SXTAB16 / SXTB16
            base = '0u' if rn == 15 else R(rn)
            ins.body = '{ u32 x = %s; u32 lo = (u32)((s32)(s8)(x & 0xFF) + (s32)(s16)(%s & 0xFFFF)) & 0xFFFF; u32 hi = (u32)((s32)(s8)((x >> 16) & 0xFF) + (s32)(s16)(%s >> 16)) & 0xFFFF; %s = lo | (hi << 16); }' % (rme, base, base, R(rd))
            return
        if o == 0 and op2 == 5:   # SEL
            ins.body = '%s = sel_ge(c, %s, %s);' % (R(rd), R(rn), R(rm)); return
        if (o >> 1) == 1 and not (op2 & 1):   # SSAT
            sat = ((w >> 16) & 0x1F) + 1
            e, _ = imm_shift(R(rm), 2 if (w >> 6) & 1 else 0, (w >> 7) & 0x1F, False)
            ins.body = '%s = ssat(c, (s32)(%s), %d);' % (R(rd), e, sat); return
        if o == 2 and op2 == 1:   # SSAT16
            sat = ((w >> 16) & 0xF) + 1
            ins.body = '%s = ssat16(c, %s, %d);' % (R(rd), R(rm), sat); return
        if (o >> 1) == 3 and not (op2 & 1):   # USAT
            sat = (w >> 16) & 0x1F
            e, _ = imm_shift(R(rm), 2 if (w >> 6) & 1 else 0, (w >> 7) & 0x1F, False)
            ins.body = '%s = usat(c, (s32)(%s), %d);' % (R(rd), e, sat); return
        if o == 6 and op2 == 1:   # USAT16
            sat = (w >> 16) & 0xF
            ins.body = '%s = usat16(c, %s, %d);' % (R(rd), R(rm), sat); return
        if op2 == 3:
            ext = {2: '(u32)(s32)(s8)', 3: '(u32)(s32)(s16)', 4: None, 6: '(u32)(u8)', 7: '(u32)(u16)'}
            if o == 4:   # UXTAB16/UXTB16
                base = '0u' if rn == 15 else R(rn)
                ins.body = '{ u32 x = %s; u32 lo = ((x & 0xFF) + (%s & 0xFFFF)) & 0xFFFF; u32 hi = (((x >> 16) & 0xFF) + (%s >> 16)) & 0xFFFF; %s = lo | (hi << 16); }' % (rme, base, base, R(rd))
                return
            if o in ext and ext[o]:
                v = '%s(%s)' % (ext[o], rme)
                ins.body = '%s = %s;' % (R(rd), v if rn == 15 else '%s + %s' % (R(rn), v))
                return
        if o == 3 and op2 == 1: ins.body = '%s = __builtin_bswap32(%s);' % (R(rd), R(rm)); return
        if o == 3 and op2 == 5: ins.body = '%s = rev16(%s);' % (R(rd), R(rm)); return
        if o == 7 and op2 == 5: ins.body = '%s = (u32)(s32)(s16)__builtin_bswap16((u16)%s);' % (R(rd), R(rm)); return
        raise NotImplementedError('media pack %x %x' % (o, op2))
    if (op1 >> 3) == 2:   # signed multiplies
        o = op1 & 7
        rdd = (w >> 16) & 0xF; ra = (w >> 12) & 0xF
        swap = (op2 & 1)
        rs_e = 'ror32(%s, 16)' % R(rs) if swap else R(rs)
        if o == 0:
            lo = '((s32)(s16)%s * (s32)(s16)%s)' % (R(rm), rs_e)
            hi = '((s32)(s16)(%s >> 16) * (s32)(s16)(%s >> 16))' % (R(rm), rs_e)
            sub = (op2 >> 1) & 1
            if (op2 >> 2) & 1: raise NotImplementedError('smlad op2')
            acc = '0' if ra == 15 else '(s64)(s32)%s' % R(ra)
            ins.body = '{ u32 rs2 = %s; (void)rs2; s64 r = (s64)%s %s (s64)%s + %s; %s = (u32)r; if (r != (s64)(s32)r) c.q = 1; }' % (
                rs_e, lo, '-' if sub else '+', hi, acc, R(rdd))
            return
        if o == 4:   # SMLALD / SMLSLD
            lo = '((s64)(s16)%s * (s16)%s)' % (R(rm), rs_e)
            hi = '((s64)(s16)(%s >> 16) * (s16)(%s >> 16))' % (R(rm), rs_e)
            sub = (op2 >> 1) & 1
            ins.body = '{ s64 acc = (s64)(((u64)%s << 32) | %s); acc += %s %s %s; %s = (u32)acc; %s = (u32)((u64)acc >> 32); }' % (
                R(rdd), R(ra), lo, '-' if sub else '+', hi, R(ra), R(rdd))
            return
        if o == 5:   # SMMLA/SMMUL/SMMLS
            rnd = '0x80000000ll' if (op2 & 1) else '0'
            prod = '((s64)(s32)%s * (s32)%s)' % (R(rm), R(rs))
            if (op2 >> 1) == 0:
                acc = '0' if ra == 15 else '((s64)(s32)%s << 32)' % R(ra)
                ins.body = '%s = (u32)((u64)(%s + %s + %s) >> 32);' % (R(rdd), acc, prod, rnd)
            elif (op2 >> 1) == 3:
                ins.body = '%s = (u32)((u64)(((s64)(s32)%s << 32) - %s + %s) >> 32);' % (R(rdd), R(ra), prod, rnd)
            else: raise NotImplementedError('smm')
            return
        raise NotImplementedError('smul %x' % o)
    if op1 == 0x18 and op2 == 0:   # USAD8/USADA8
        rdd = (w >> 16) & 0xF; ra = (w >> 12) & 0xF
        acc = '' if ra == 15 else ' + %s' % R(ra)
        ins.body = '%s = usad8(%s, %s)%s;' % (R(rdd), R(rm), R(rs), acc)
        return
    raise NotImplementedError('media %x %x' % (op1, op2))

# --------------------------------------------------------------- VFP (ARM)
def sreg(v, bit):   # single register number
    return (v << 1) | bit
def dreg(v, bit):
    return (bit << 4) | v

def dec_arm_ldc(ins, w, pc):
    cp = (w >> 8) & 0xF
    if cp not in (10, 11): raise NotImplementedError('ldc cp%d' % cp)
    if ((w >> 21) & 0x7F) == 0x62:   # VMOV two core regs <-> D / two S
        l = (w >> 20) & 1; rt = (w >> 12) & 0xF; rt2 = (w >> 16) & 0xF
        m = (w >> 5) & 1; vm = w & 0xF
        if cp == 11:
            a, b = dreg(vm, m) * 2, dreg(vm, m) * 2 + 1
        else:
            a = sreg(vm, m); b = (a + 1) & 31
        if l: ins.body = '%s = c.f.sw[%d]; %s = c.f.sw[%d];' % (R(rt), a, R(rt2), b)
        else: ins.body = 'c.f.sw[%d] = %s; c.f.sw[%d] = %s;' % (a, R(rt), b, R(rt2))
        return
    p = (w >> 24) & 1; u = (w >> 23) & 1; d = (w >> 22) & 1; wb = (w >> 21) & 1; l = (w >> 20) & 1
    rn = (w >> 16) & 0xF; vd = (w >> 12) & 0xF; imm8 = w & 0xFF
    dbl = cp == 11
    base = hx(pc & ~3) if rn == 15 else R(rn)
    if p and not wb:   # VLDR/VSTR
        a = '%s %s %d' % (base, '+' if u else '-', imm8 * 4)
        if rn == 15:
            ins.ptr_loads.append(((pc & ~3) + (imm8 * 4 if u else -imm8 * 4)) & 0xffffffff)
        if dbl:
            r = dreg(vd, d)
            ins.body = ('{ u32 a = %s; c.f.dw[%d] = RD64(a); }' if l else '{ u32 a = %s; WR64(a, c.f.dw[%d]); }') % (a, r)
        else:
            r = sreg(vd, d)
            ins.body = ('{ u32 a = %s; c.f.sw[%d] = RD32(a); }' if l else '{ u32 a = %s; WR32(a, c.f.sw[%d]); }') % (a, r)
        return
    # VLDM/VSTM
    if dbl:
        n = imm8 // 2
        first = dreg(vd, d)
    else:
        n = imm8
        first = sreg(vd, d)
    if n == 0: raise NotImplementedError('vldm 0')
    size = imm8 * 4
    if p == 0 and u == 1: start = R(rn)
    elif p == 1 and u == 0: start = '%s - %d' % (R(rn), size)
    else: raise NotImplementedError('vldm mode')
    body = ['u32 a = %s;' % start]
    for k in range(n):
        if dbl:
            if l: body.append('c.f.dw[%d] = RD64(a + %d);' % ((first + k) & 15, 8 * k))
            else: body.append('WR64(a + %d, c.f.dw[%d]);' % (8 * k, (first + k) & 15))
        else:
            body.append(('c.f.sw[%d] = RD32(a + %d);' % ((first + k) & 31, 4 * k)) if l else 'WR32(a + %d, c.f.sw[%d]);' % (4 * k, (first + k) & 31))
    if wb:
        body.append('%s = %s %s %d;' % (R(rn), R(rn), '+' if u else '-', size))
    ins.body = '{ ' + ' '.join(body) + ' }'

def dec_arm_cp(ins, w, pc):
    cp = (w >> 8) & 0xF
    if (w >> 4) & 1:   # register transfer
        l = (w >> 20) & 1
        rt = (w >> 12) & 0xF
        opc1 = (w >> 21) & 7
        crn = (w >> 16) & 0xF; crm = w & 0xF; opc2 = (w >> 5) & 7
        if cp == 15:
            if l and crn == 13 and crm == 0 and opc1 == 0 and opc2 == 3:
                ins.body = '%s = c.tpidruro;' % R(rt); return
            if l and crn == 13 and crm == 0 and opc1 == 0 and opc2 == 2:
                ins.body = '%s = c.tpidrurw;' % R(rt); return
            if not l and crn == 13 and crm == 0 and opc2 == 2:
                ins.body = 'c.tpidrurw = %s;' % R(rt); return
            if not l and crn == 7:
                ins.body = ';'; return   # barriers / cache ops
            raise NotImplementedError('cp15 %d c%d c%d %d' % (opc1, crn, crm, opc2))
        if cp == 10:
            if opc1 == 0:   # VMOV core <-> single
                sn = sreg(crn, (w >> 7) & 1)
                ins.body = ('%s = c.f.sw[%d];' % (R(rt), sn)) if l else ('c.f.sw[%d] = %s;' % (sn, R(rt)))
                return
            if opc1 == 7:   # VMRS / VMSR
                if l:
                    if crn == 1:
                        if rt == 15:
                            ins.body = 'fn = c.fpscr >> 31; fz = (c.fpscr >> 30) & 1; fc = (c.fpscr >> 29) & 1; fv = (c.fpscr >> 28) & 1;'
                        else:
                            ins.body = '%s = c.fpscr;' % R(rt)
                    elif crn == 8: ins.body = '%s = 0x40000000u;' % R(rt)
                    elif crn == 0: ins.body = '%s = 0x410120B4u;' % R(rt)
                    else: raise NotImplementedError('vmrs')
                else:
                    if crn == 1: ins.body = 'c.fpscr = %s; host_fpscr(c.fpscr);' % R(rt)
                    else: ins.body = ';'
                return
        if cp == 11 and (opc1 & 6) == 0:   # VMOV.32 Dd[x] <-> Rt
            x = opc1 & 1
            dd = dreg(crn, (w >> 7) & 1)
            ins.body = ('%s = c.f.sw[%d];' % (R(rt), dd * 2 + x)) if l else ('c.f.sw[%d] = %s;' % (dd * 2 + x, R(rt)))
            return
        raise NotImplementedError('cp reg transfer cp%d' % cp)
    if cp not in (10, 11): raise NotImplementedError('cdp cp%d' % cp)
    ins.body = vfp_dp(w, cp == 11)

def vfp_dp(w, dbl):
    p = (w >> 23) & 1; q = (w >> 21) & 1; r = (w >> 20) & 1; s = (w >> 6) & 1
    D = (w >> 22) & 1; N = (w >> 7) & 1; M = (w >> 5) & 1
    vn = (w >> 16) & 0xF; vd = (w >> 12) & 0xF; vm = w & 0xF
    if dbl:
        d, n, m = 'c.f.d[%d]' % dreg(vd, D), 'c.f.d[%d]' % dreg(vn, N), 'c.f.d[%d]' % dreg(vm, M)
        T = 'double'
    else:
        d, n, m = 'c.f.s[%d]' % sreg(vd, D), 'c.f.s[%d]' % sreg(vn, N), 'c.f.s[%d]' % sreg(vm, M)
        T = 'float'
    opc = (p << 3) | (q << 2) | (r << 1) | s
    # FPSCR.DN is set by the game (RunFast): any NaN result is the default NaN.
    # Arithmetic goes through the lift_rt.h helpers, which honour FPSCR
    # rounding / flush-to-zero in software where the host can't (wasm).
    x = 'd' if dbl else 'f'
    F = 'c.fpscr'
    mul = 'vmul_%s(%s, %s, %s)' % (x, F, n, m)
    if opc == 0: return '%s = vadd_%s(%s, %s, %s);' % (d, x, F, d, mul)          # VMLA
    if opc == 1: return '%s = vsub_%s(%s, %s, %s);' % (d, x, F, d, mul)          # VMLS
    if opc == 2: return '%s = vsub_%s(%s, %s, %s);' % (d, x, F, mul, d)          # VNMLS: -d + n*m
    if opc == 3: return '%s = vsub_%s(%s, -%s, %s);' % (d, x, F, d, mul)         # VNMLA: -d - n*m
    if opc == 4: return '%s = %s;' % (d, mul)                                     # VMUL
    if opc == 5: return '%s = -%s;' % (d, mul)                                    # VNMUL: FPNeg after the DN product
    if opc == 6: return '%s = vadd_%s(%s, %s, %s);' % (d, x, F, n, m)            # VADD
    if opc == 7: return '%s = vsub_%s(%s, %s, %s);' % (d, x, F, n, m)            # VSUB
    if opc == 8: return '%s = vdiv_%s(%s, %s, %s);' % (d, x, F, n, m)            # VDIV
    if (opc & 0xE) == 0xE:   # extension, opc2 = vn
        o3 = (w >> 6) & 3
        if vn == 0 and o3 == 1: return '%s = %s;' % (d, m)
        if vn == 0 and o3 == 3: return '%s = fabs%s(%s);' % (d, '' if dbl else 'f', m)
        if vn == 1 and o3 == 1: return '%s = -%s;' % (d, m)
        if vn == 1 and o3 == 3: return '%s = vsqrt_%s2(%s, %s);' % (d, x, F, m)
        if vn in (4, 5):
            rhs = ('(%s)0' % T) if vn == 5 else 'fz_cmp_%s(%s, %s)' % (x, m, F)
            return 'c.fpscr = vfp_cmp(c.fpscr, (double)fz_cmp_%s(%s, %s), (double)%s);' % (x, d, F, rhs)
        if vn == 7 and o3 == 3:   # VCVT f32<->f64
            if dbl: return 'c.f.s[%d] = vcvt_f_d(%s, %s);' % (sreg(vd, D), F, m)
            return 'c.f.d[%d] = vcvt_d_f(%s, %s);' % (dreg(vd, D), F, 'c.f.s[%d]' % sreg(vm, M))
        if vn == 8:   # int -> float (source single reg)
            src = 'c.f.sw[%d]' % sreg(vm, M)
            conv = '(double)(s32)%s' % src if (w >> 7) & 1 else '(double)(u32)%s' % src
            if dbl: return '%s = %s;' % (d, conv)
            return '%s = vcvt_f_i(c.fpscr, %s);' % (d, conv)
        if vn in (12, 13):   # float -> int (dest single reg)
            dst = 'c.f.sw[%d]' % sreg(vd, D)
            rz = (w >> 7) & 1
            fn = 'f2s' if vn == 13 else 'f2u'
            return '%s = %s%s((double)%s, c.fpscr);' % (dst, fn, '_rz' if rz else '', m)
        raise NotImplementedError('vfp ext %d %d' % (vn, o3))
    raise NotImplementedError('vfp dp %x' % opc)

def dec_arm_mcrr(ins, w):
    pass

# ===================================================================== Thumb
def dec_thumb(addr, h, h2=None):
    ins = Insn(addr, True, 2)
    pc = addr + 4
    ins.text = '%04x' % h
    try:
        _dec_thumb(ins, addr, h, h2, pc)
    except NotImplementedError as e:
        ins.bad = str(e) or 'unimpl'
    if ins.cond == 14 and ins.body.startswith('EXIT(') and ins.body.count(';') == 1:
        ins.falls = False
    return ins

def _dec_thumb(ins, addr, h, h2, pc):
    top5 = h >> 11
    lo = lambda s: (h >> s) & 7
    if top5 < 3:   # shift imm
        op = top5; imm = (h >> 6) & 0x1F; rm = lo(3); rd = lo(0)
        if op == 0 and imm == 0:
            ins.body = '%s = %s; %s' % (R(rd), R(rm), nz(R(rd)))
            return
        e, c = imm_shift(R(rm), op, imm, True)
        ins.body = '{ u32 cc = %s; %s = %s; %s fc = cc; }' % (c, R(rd), e, nz(R(rd)))
        return
    if top5 == 3:  # add/sub reg/imm3
        imm = (h >> 10) & 1; sub = (h >> 9) & 1
        rn = lo(3); rd = lo(0)
        b = hx((h >> 6) & 7) if imm else R((h >> 6) & 7)
        if sub: ins.body = '%s = adds(%s, ~(%s), 1, fn, fz, fc, fv);' % (R(rd), R(rn), b)
        else: ins.body = '%s = adds(%s, %s, 0, fn, fz, fc, fv);' % (R(rd), R(rn), b)
        return
    if top5 < 8:   # mov/cmp/add/sub imm8
        op = top5 - 4; rd = lo(8); imm = hx(h & 0xFF)
        if op == 0: ins.body = '%s = %s; %s' % (R(rd), imm, nz(R(rd)))
        elif op == 1: ins.body = 'adds(%s, ~(%s), 1, fn, fz, fc, fv);' % (R(rd), imm)
        elif op == 2: ins.body = '%s = adds(%s, %s, 0, fn, fz, fc, fv);' % (R(rd), R(rd), imm)
        else: ins.body = '%s = adds(%s, ~(%s), 1, fn, fz, fc, fv);' % (R(rd), R(rd), imm)
        return
    if (h >> 10) == 0x10:   # ALU
        op = (h >> 6) & 0xF; rm = lo(3); rd = lo(0)
        D, M = R(rd), R(rm)
        if op == 0: ins.body = '%s &= %s; %s' % (D, M, nz(D))
        elif op == 1: ins.body = '%s ^= %s; %s' % (D, M, nz(D))
        elif op in (2, 3, 4, 7):
            fn = {2: 'lsl_rc', 3: 'lsr_rc', 4: 'asr_rc', 7: 'ror_rc'}[op]
            ins.body = '{ u32 sc = fc; %s = %s(%s, %s & 0xFF, sc); fc = sc; %s }' % (D, fn, D, M, nz(D))
        elif op == 5: ins.body = '%s = adds(%s, %s, fc, fn, fz, fc, fv);' % (D, D, M)
        elif op == 6: ins.body = '%s = adds(%s, ~(%s), fc, fn, fz, fc, fv);' % (D, D, M)
        elif op == 8: ins.body = '{ u32 res = %s & %s; %s }' % (D, M, nz('res'))
        elif op == 9: ins.body = '%s = adds(0u, ~(%s), 1, fn, fz, fc, fv);' % (D, M)
        elif op == 10: ins.body = 'adds(%s, ~(%s), 1, fn, fz, fc, fv);' % (D, M)
        elif op == 11: ins.body = 'adds(%s, %s, 0, fn, fz, fc, fv);' % (D, M)
        elif op == 12: ins.body = '%s |= %s; %s' % (D, M, nz(D))
        elif op == 13: ins.body = '%s = %s * %s; %s' % (D, M, D, nz(D))
        elif op == 14: ins.body = '%s &= ~%s; %s' % (D, M, nz(D))
        else: ins.body = '%s = ~%s; %s' % (D, M, nz(D))
        return
    if (h >> 10) == 0x11:   # hi reg ops / bx
        op = (h >> 8) & 3
        rd = ((h >> 4) & 8) | (h & 7)
        rm = (h >> 3) & 0xF
        rm_e = hx(pc) if rm == 15 else R(rm)
        if op == 0:
            if rd == 15:
                ins.body = 'EXIT((%s + %s) | 1u);' % (hx(pc), rm_e)
                ins.falls = False
            else:
                ins.body = '%s = %s + %s;' % (R(rd), R(rd), rm_e)
        elif op == 1:
            rd_e = hx(pc) if rd == 15 else R(rd)
            ins.body = 'adds(%s, ~(%s), 1, fn, fz, fc, fv);' % (rd_e, rm_e)
        elif op == 2:
            if rd == 15:
                ins.body = 'EXIT((%s) | 1u);' % rm_e
                ins.falls = False
            else:
                ins.body = '%s = %s;' % (R(rd), rm_e)
        else:
            if (h >> 7) & 1:   # BLX reg
                ins.call = True
                ins.body = '{ u32 t = %s; r14 = %s; EXIT(t); }' % (rm_e, hx((addr + 2) | 1))
            else:
                ins.body = 'EXIT(%s);' % rm_e
                ins.falls = False
        return
    if top5 == 9:   # ldr literal
        a = ((pc & ~3) + (h & 0xFF) * 4) & 0xffffffff
        ins.body = '%s = RD32(%s);' % (R(lo(8)), hx(a))
        ins.ptr_loads.append(a)
        return
    if (h >> 12) == 5:   # load/store reg offset
        op = (h >> 9) & 7; rm = lo(6); rn = lo(3); rt = lo(0)
        a = '%s + %s' % (R(rn), R(rm))
        ins.body = [
            'WR32(%s, %s);', 'WR16(%s, %s);', 'WR8(%s, %s);', '%s = (u32)(s32)(s8)RD8(%s);',
            '%s = RD32(%s);', '%s = RD16(%s);', '%s = RD8(%s);', '%s = (u32)(s32)(s16)RD16(%s);'][op]
        ins.body = ins.body % ((a, R(rt)) if op < 3 else (R(rt), a))
        return
    if (h >> 13) == 3:   # ldr/str imm5
        b = (h >> 12) & 1; l = (h >> 11) & 1
        imm = ((h >> 6) & 0x1F) * (1 if b else 4)
        rn = lo(3); rt = lo(0)
        a = '%s + %d' % (R(rn), imm)
        if l: ins.body = '%s = %s(%s);' % (R(rt), 'RD8' if b else 'RD32', a)
        else: ins.body = '%s(%s, %s);' % ('WR8' if b else 'WR32', a, R(rt))
        return
    if (h >> 12) == 8:   # ldrh/strh imm5
        l = (h >> 11) & 1; imm = ((h >> 6) & 0x1F) * 2
        a = '%s + %d' % (R(lo(3)), imm)
        ins.body = ('%s = RD16(%s);' % (R(lo(0)), a)) if l else ('WR16(%s, %s);' % (a, R(lo(0))))
        return
    if (h >> 12) == 9:   # sp relative
        l = (h >> 11) & 1; rt = lo(8); a = 'r13 + %d' % ((h & 0xFF) * 4)
        ins.body = ('%s = RD32(%s);' % (R(rt), a)) if l else ('WR32(%s, %s);' % (a, R(rt)))
        return
    if (h >> 12) == 10:  # add rd, pc/sp
        rd = lo(8); imm = (h & 0xFF) * 4
        if (h >> 11) & 1: ins.body = '%s = r13 + %d;' % (R(rd), imm)
        else: ins.body = '%s = %s;' % (R(rd), hx((pc & ~3) + imm))
        return
    if (h >> 12) == 11:  # misc
        if (h >> 8) == 0xB0:
            imm = (h & 0x7F) * 4
            ins.body = 'r13 %s= %d;' % ('-' if (h >> 7) & 1 else '+', imm)
            return
        if (h & 0x0600) == 0x0400:   # push/pop
            l = (h >> 11) & 1; rbit = (h >> 8) & 1
            regs = [i for i in range(8) if (h >> i) & 1]
            if l:
                if rbit: regs.append(15)
                body = ['u32 a = r13;']
                for k, r in enumerate(regs):
                    body.append(('u32 tpc = RD32(a + %d);' if r == 15 else R(r) + ' = RD32(a + %d);') % (4 * k))
                body.append('r13 = a + %d;' % (4 * len(regs)))
                if rbit:
                    body.append('EXIT(tpc);'); ins.falls = False
                ins.body = '{ ' + ' '.join(body) + ' }'
            else:
                if rbit: regs.append(14)
                n = len(regs)
                body = ['u32 a = r13 - %d;' % (4 * n)]
                for k, r in enumerate(regs):
                    body.append('WR32(a + %d, %s);' % (4 * k, R(r)))
                body.append('r13 = a;')
                ins.body = '{ ' + ' '.join(body) + ' }'
            return
        if (h >> 8) == 0xB2:
            op = (h >> 6) & 3; rm = R(lo(3)); rd = R(lo(0))
            ins.body = '%s = %s(%s);' % (rd, ['(u32)(s32)(s16)', '(u32)(s32)(s8)', '(u32)(u16)', '(u32)(u8)'][op], rm)
            return
        if (h >> 8) == 0xBA:
            op = (h >> 6) & 3; rm = R(lo(3)); rd = R(lo(0))
            if op == 0: ins.body = '%s = __builtin_bswap32(%s);' % (rd, rm)
            elif op == 1: ins.body = '%s = rev16(%s);' % (rd, rm)
            elif op == 3: ins.body = '%s = (u32)(s32)(s16)__builtin_bswap16((u16)%s);' % (rd, rm)
            else: raise NotImplementedError('rev?')
            return
        if (h >> 8) == 0xBE:
            ins.body = 'TRAP("bkpt");'; return
        if (h >> 8) == 0xBF or (h & 0xFFE8) == 0xB660 or (h & 0xFFF7) == 0xB650:
            ins.body = ';'; return
        raise NotImplementedError('thumb misc %04x' % h)
    if (h >> 12) == 12:  # ldmia/stmia
        l = (h >> 11) & 1; rn = lo(8)
        regs = [i for i in range(8) if (h >> i) & 1]
        if not regs: raise NotImplementedError('ldm empty')
        body = ['u32 a = %s;' % R(rn)]
        if l:
            for k, r in enumerate(regs): body.append('%s = RD32(a + %d);' % (R(r), 4 * k))
            if rn not in regs: body.append('%s = a + %d;' % (R(rn), 4 * len(regs)))
        else:
            for k, r in enumerate(regs): body.append('WR32(a + %d, %s);' % (4 * k, R(r)))
            body.append('%s = a + %d;' % (R(rn), 4 * len(regs)))
        ins.body = '{ ' + ' '.join(body) + ' }'
        return
    if (h >> 12) == 13:  # b cond / svc
        cond = (h >> 8) & 0xF
        if cond == 15:
            ins.body = 'SVC(%s, %s);' % (hx(h & 0xFF), hx((addr + 2) | 1)); return
        if cond == 14:
            ins.body = 'TRAP("udf");'; return
        t = (pc + (sx(h & 0xFF, 8) << 1)) & 0xffffffff
        ins.cond = cond
        ins.body = 'JMP(%s);' % hx(t | 1)
        ins.targets = [t | 1]
        return
    if top5 == 0x1C:     # b
        t = (pc + (sx(h & 0x7FF, 11) << 1)) & 0xffffffff
        ins.body = 'JMP(%s);' % hx(t | 1)
        ins.targets = [t | 1]
        ins.falls = False
        return
    if top5 == 0x1E:     # bl/blx prefix
        if h2 is None or (h2 >> 11) not in (0x1F, 0x1D):
            raise NotImplementedError('lone bl prefix')
        ins.size = 4
        ins.text += ' %04x' % h2
        off = (sx(h & 0x7FF, 11) << 12) | ((h2 & 0x7FF) << 1)
        base = (pc + off) & 0xffffffff
        ret = (addr + 4) | 1
        ins.call = True
        if (h2 >> 11) == 0x1F:
            t = base | 1
        else:
            t = base & ~3
        ins.body = 'r14 = %s; JMP(%s);' % (hx(ret), hx(t))
        ins.targets = [t]
        return
    raise NotImplementedError('thumb %04x' % h)
