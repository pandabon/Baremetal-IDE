/**
 * @file fp8_matmul.h
 * @brief The 64x64x64 FP8 matmul on the host's Saturn vector unit (fp8_matmul.S), in three phases.
 *
 * Buffers are passed in because the data arrays in matmul_data.h are static.
 *   in    the matmul inputs, laid out like mm_in (A tiles, then B tiles stored transposed)
 *   bscr  4 KiB scratch: B back in row-major order
 *   c16   8 KiB: the BF16 sums, tile order (64 x 64 values)
 *   c8    4 KiB: the E4M3 result, in mm_golden's layout
 */
#ifndef __FP8_MATMUL_H
#define __FP8_MATMUL_H

#include <stdint.h>

void fp8_matmul_pre(const void *in, void *bscr);              /* transpose B to row-major */
void fp8_matmul_mac(const void *in, const void *bscr, void *c16);  /* FP8 multiply-accumulate into BF16 */
void fp8_matmul_cvt(const void *c16, void *c8);               /* BF16 -> E4M3 */

#endif /* __FP8_MATMUL_H */
