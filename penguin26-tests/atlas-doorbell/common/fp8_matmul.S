/*
 * fp8_matmul.S - the 64x64x64 FP8 matmul on the host's Saturn vector unit, in three routines that
 * main.c calls and times:
 *   fp8_matmul_pre(in, bscr)        transpose the B tiles to row-major
 *   fp8_matmul_mac(in, bscr, c16)   the multiply-accumulate loop (BF16 results)
 *   fp8_matmul_cvt(c16, c8)         BF16 -> E4M3, laid out like the golden data
 *
 * Data layout (generator: gen_perf_mm_mxu0_64x64x64.py in tapeout-chipyard
 * generators/sp26-atlas-acc/baremetal/generators, lines 21-33 and 148-186):
 *   The 64x64 matrices are cut into four 32x32 FP8 (E4M3) tiles, each stored row-major, 1 KiB per tile.
 *   in + 0x0000: A00 A01 A10 A11   (A_ik[m][kk], i = row tile, k = inner tile)
 *   in + 0x1000: B00_T B01_T B10_T B11_T   (B_kj TRANSPOSED: B_T[n][kk] = B_kj[kk][n]; Atlas's
 *                VMATPUSH.W wants that)
 *   golden:      C00 C01 C10 C11, each 32x32 row-major, E4M3 bytes (not BF16): the generator runs the
 *                BF16 systolic result through bf16_to_e4m3_scaled_bits (lines 93-131, scale 0).
 *                Tile (i,j) element (m,n) is at golden[(2i+j)*1024 + m*32 + n], one byte.
 *   Golden accumulation (lines 164-177): BF16 accumulator, K = 64 summed in order, first the inner
 *   tile k=0 (kk 0..31), then k=1.
 *
 * Credit: the vfwmacc loop, the VSETVLI_ALTFMT encodings (.insn) and the vmv.v.x splat come from
 * Saturn's benchmarks:
 *   repo ucb-bar/saturn (tapeout-chipyard/generators/saturn) commit de04946
 *   (de04946ac1ce878c79a4d6e74e493c5f440908e2)
 *   benchmarks/vec-mx-matmul/main.c (TEST_VECTOR_OUTER), benchmarks/common/rvv_mx.h
 *   (VSETVLI_ALTFMT*), benchmarks/vec-mx-gemv/main.c (vfwmacc.vv at SEW 8 with altfmt).
 * The BF16 -> E4M3 steps are vectorized from the generator above.
 * The altfmt encodings are copied from rvv_mx.h; they are not checked against the RTL decode.
 *
 * Vector state: glossy's crt0.S already turns on mstatus.FS and mstatus.VS (vector FP instructions are
 * illegal while FS is off), so this file does not touch mstatus.
 */

/* The default build is rv64imafd without the V extension. Turn V on for this file only, so the test
 * builds with the default ARCH (no RVV_TYPE needed). Running it still needs a core with the vector unit. */
    .option arch, +v

/* VSET rd, avl, sew, lmul, alt: vsetvli with the altfmt bit (vtype bit 8; 0 = E4M3, 1 = E5M2), as
 * Saturn's rvv_mx.h VSETVLI_ALTFMT: .insn i 0x57, 0x7, rd, avl, (alt<<8)|(sew<<3)|lmul
 * (sew code 0 = e8, 1 = e16; lmul code 0 = m1, 1 = m2). The assembler has no mnemonic for altfmt. */
    .macro VSET rd, avl, sew, lmul, alt
    .insn i 0x57, 0x7, \rd, \avl, ((\alt) << 8) | ((\sew) << 3) | (\lmul)
    .endm

    .text

/*
 * fp8_matmul_pre(a0 = in, a1 = bscr)
 * B_T tiles (in + 0x1000, [n][kk]) -> bscr tiles ([kk][n]), tile index 2k+j kept. The outer-product
 * form below wants the B rows contiguous; Atlas gets B_T for free, the host has to transpose.
 * One strided load per column of B_T (stride 32 bytes) gives one row of B.
 */
    .globl fp8_matmul_pre
fp8_matmul_pre:
    li    t0, 32
    VSET  zero, t0, 0, 0, 0          # e8, m1, vl 32 (one 32-byte vector register)
    li    t1, 0x1000
    add   a2, a0, t1                 # B_T tiles
    mv    a3, a1                     # bscr
    li    t2, 32                     # stride between n rows
    li    t5, 4                      # tiles
1:  mv    a4, a2
    mv    a5, a3
    li    t6, 32
2:  vlse8.v v0, (a4), t2             # column kk of B_T = row kk of B (32 values over n)
    vse8.v  v0, (a5)
    addi  a4, a4, 1
    addi  a5, a5, 32
    addi  t6, t6, -1
    bnez  t6, 2b
    addi  a2, a2, 1024
    addi  a3, a3, 1024
    addi  t5, t5, -1
    bnez  t5, 1b
    ret

/*
 * fp8_matmul_mac(a0 = in, a1 = bscr, a2 = c16)
 * C16 (tile order, BF16, 32 per row) = A x B, accumulating k = 0 then 1 in one BF16 accumulator.
 * Saturn's vec-mx-matmul loop: 8 output rows x 32 columns per block. Per kk: one vle8 of the B row,
 * 8 scalar A bytes splat with vmv.v.x, 8 widening FMAs vfwmacc.vv (E4M3 x E4M3 -> BF16
 * accumulators v16..v31). One BF16 rounding per product, in the golden's order.
 * Registers: acc v16..v31, A splats v8..v15, B row v0.
 */
    .globl fp8_matmul_mac
fp8_matmul_mac:
    addi  sp, sp, -80
    sd    s0, 0(sp)
    sd    s1, 8(sp)
    sd    s2, 16(sp)
    sd    s3, 24(sp)
    sd    s4, 32(sp)
    sd    s5, 40(sp)
    sd    s6, 48(sp)
    sd    s7, 56(sp)
    sd    s8, 64(sp)
    mv    s3, a0                     # in
    mv    s4, a1                     # bscr
    mv    s5, a2                     # c16
    li    s0, 0                      # tile index tc = 2i + j
tile:
    srli  t0, s0, 1
    slli  t0, t0, 11
    add   s6, s3, t0                 # A tiles of row tile i: in + i*2048 (A_i0, A_i1)
    andi  t0, s0, 1
    slli  t0, t0, 10
    add   s7, s4, t0                 # B_0j row-major tile: bscr + j*1024 (B_1j is +2048)
    slli  t0, s0, 11
    add   s8, s5, t0                 # C16 tile: 1024 BF16 = 2048 bytes
    li    s1, 0                      # block of 8 rows
block:
    li    t0, 128
    vsetvli zero, t0, e16, m8, ta, ma
    vmv.v.i v16, 0                   # v16..v23
    vmv.v.i v24, 0                   # v24..v31
    slli  t0, s1, 8
    add   a2, s6, t0                 # A_i0 + 8*mb rows * 32 B
    mv    a3, s7
    li    s2, 2                      # inner tiles k
    li    t1, 32
    VSET  zero, t1, 0, 0, 0          # e8, m1, E4M3 (altfmt 0), vl 32
ktile:
    li    t2, 32                     # kk
kk:
    vle8.v v0, (a3)                  # B_kj[kk][0..31]
    lbu   t3, 0(a2)                  # A[row r][kk], rows are 32 B apart
    lbu   t4, 32(a2)
    lbu   t5, 64(a2)
    lbu   t6, 96(a2)
    lbu   a4, 128(a2)
    lbu   a5, 160(a2)
    lbu   a6, 192(a2)
    lbu   a7, 224(a2)
    vmv.v.x v8, t3
    vmv.v.x v9, t4
    vmv.v.x v10, t5
    vmv.v.x v11, t6
    vmv.v.x v12, a4
    vmv.v.x v13, a5
    vmv.v.x v14, a6
    vmv.v.x v15, a7
    vfwmacc.vv v16, v8, v0
    vfwmacc.vv v18, v9, v0
    vfwmacc.vv v20, v10, v0
    vfwmacc.vv v22, v11, v0
    vfwmacc.vv v24, v12, v0
    vfwmacc.vv v26, v13, v0
    vfwmacc.vv v28, v14, v0
    vfwmacc.vv v30, v15, v0
    addi  a2, a2, 1
    addi  a3, a3, 32
    addi  t2, t2, -1
    bnez  t2, kk
    addi  a2, a2, 992                # next inner tile: A_i1 = A_i0 + 1024 (a2 already advanced 32)
    addi  a3, a3, 1024               # B_1j = B_0j + 2048 (a3 already advanced 1024)
    addi  s2, s2, -1
    bnez  s2, ktile
    li    t1, 32
    vsetvli zero, t1, e16, m2, ta, ma
    slli  t0, s1, 9                  # 8 rows * 64 B per block
    add   t0, s8, t0
    vse16.v v16, (t0)
    addi  t0, t0, 64
    vse16.v v18, (t0)
    addi  t0, t0, 64
    vse16.v v20, (t0)
    addi  t0, t0, 64
    vse16.v v22, (t0)
    addi  t0, t0, 64
    vse16.v v24, (t0)
    addi  t0, t0, 64
    vse16.v v26, (t0)
    addi  t0, t0, 64
    vse16.v v28, (t0)
    addi  t0, t0, 64
    vse16.v v30, (t0)
    addi  s1, s1, 1
    li    t0, 4
    blt   s1, t0, block
    addi  s0, s0, 1
    blt   s0, t0, tile
    ld    s0, 0(sp)
    ld    s1, 8(sp)
    ld    s2, 16(sp)
    ld    s3, 24(sp)
    ld    s4, 32(sp)
    ld    s5, 40(sp)
    ld    s6, 48(sp)
    ld    s7, 56(sp)
    ld    s8, 64(sp)
    addi  sp, sp, 80
    ret

/*
 * fp8_matmul_cvt(a0 = c16, a1 = c8)
 * c16 (BF16) -> c8 (E4M3), 32 elements per step, 128 steps, same order (so c8 is in golden layout).
 * No FP8 narrowing instruction exists on the host, so each element is converted with integer ops,
 * mirroring bf16_to_e4m3_scaled_bits (scale 0):
 *   e = bits[14:7], f = bits[6:0], q = (f + 7 + bit4) >> 4  (= round-to-nearest-even of f/16, 0..8),
 *   carry = q >> 3, mant = q & 7, x = e - 120 + carry  (E4M3 exponent field)
 *   x <= 0 -> 0 (flush: BF16 zero/subnormal/small), x > 15 -> 0x7E | sign<<7 (saturate),
 *   else sign<<7 | x<<3 | mant.
 * BF16 NaN/Inf are not special-cased (a finite FP8 product sum cannot produce them; Inf would
 * saturate anyway).
 */
    .globl fp8_matmul_cvt
fp8_matmul_cvt:
    mv    a2, a0                     # c16
    mv    a3, a1                     # c8
    li    a4, 128
    li    a5, 0x7F
    li    a6, -120
    li    a7, 0x7E
    li    t1, 32
1:  vsetvli zero, t1, e16, m2, ta, ma
    vle16.v v16, (a2)
    vsll.vi v2, v16, 1
    vsrl.vi v2, v2, 8                # e
    vand.vx v4, v16, a5              # f
    vsrl.vi v6, v16, 4
    vand.vi v6, v6, 1                # bit 4 (round-to-even tie bit)
    vadd.vv v4, v4, v6
    vadd.vi v4, v4, 7
    vsrl.vi v4, v4, 4                # q
    vsrl.vi v6, v4, 3                # carry
    vand.vi v4, v4, 7                # mant
    vadd.vv v2, v2, v6
    vadd.vx v2, v2, a6               # x
    vmsle.vi v10, v2, 0              # flush mask
    vmsgt.vi v11, v2, 15             # saturate mask
    vsll.vi v2, v2, 3
    vor.vv  v2, v2, v4
    vsrl.vi v6, v16, 15
    vsll.vi v6, v6, 7                # sign << 7
    vor.vv  v2, v2, v6               # normal code
    vor.vx  v8, v6, a7               # saturated code
    vmv1r.v v0, v11
    vmerge.vvm v2, v2, v8, v0
    vmv1r.v v0, v10
    vmerge.vim v2, v2, 0, v0
    vsetvli zero, t1, e8, m1, ta, ma
    vnsrl.wi v12, v2, 0              # 16 -> 8 bits
    vse8.v  v12, (a3)
    addi  a2, a2, 64
    addi  a3, a3, 32
    addi  a4, a4, -1
    bnez  a4, 1b
    ret
