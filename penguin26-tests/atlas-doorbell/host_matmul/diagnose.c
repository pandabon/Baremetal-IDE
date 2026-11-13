/*
 * diagnose.c - when host_matmul's result is wrong, say where.
 *
 * It recomputes the BF16 sums in software and compares them with what the vector unit left in c16,
 * then compares the final E4M3 bytes with the golden data. The counts are split by output tile, by
 * row within the 8-row block the kernel computes at once (each row has its own accumulator register,
 * v16..v30), and by groups of 8 columns. A wrong pattern in one of those splits points at one part of
 * the kernel: a whole tile at the addressing, one row at one accumulator, one column group at a part
 * of the vector register.
 *
 * The software sums follow Saturn's reference model for the FP8 widening multiply-accumulate
 * (ucb-bar/saturn @de04946, benchmarks/common-data-gen/mx_data_gen.c, e4m3_wmacc: the product
 * rounded to BF16, then a BF16 add), in the kernel's order: K tile 0 (kk 0..31), then K tile 1.
 * E4M3 decoding follows the OCP FP8 format (bias 7, subnormals, no infinities).
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* E4M3 byte to float (exact). 0x7F/0xFF are NaN in E4M3; the test data has none. */
static float e4m3_to_float(uint8_t b) {
  int e = (b >> 3) & 15, m = b & 7;
  float v = (e == 0) ? (float)m / 8.0f / 64.0f                     /* subnormal: m/8 * 2^-6 */
                     : (1.0f + (float)m / 8.0f) * (float)(1 << e) / 128.0f;  /* (1 + m/8) * 2^(e-7) */
  return (b & 0x80) ? -v : v;
}

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/* Round a float to BF16 (round to nearest, ties to even); returns the BF16 bits. */
static uint16_t bf16_bits(float f) {
  uint32_t u = f2u(f);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
static float bf16_val(uint16_t h) { return u2f((uint32_t)h << 16); }

/* Rows of A and B as the kernel reads them: A tiles (2i+k) row-major, B tiles (2k+j) stored transposed. */
static float ref_sum(const uint8_t *in, int i, int j, int m, int n) {
  float acc = 0.0f;
  for (int k = 0; k < 2; k++) {
    for (int kk = 0; kk < 32; kk++) {
      float a = e4m3_to_float(in[(2 * i + k) * 1024 + m * 32 + kk]);
      float b = e4m3_to_float(in[4096 + (2 * k + j) * 1024 + n * 32 + kk]);
      float prod = bf16_val(bf16_bits(a * b));       /* the product, rounded to BF16 */
      acc = bf16_val(bf16_bits(acc + prod));         /* the BF16 add */
    }
  }
  return acc;
}

void host_matmul_diagnose(const uint8_t *in, const uint16_t *c16, const uint8_t *c8, const uint8_t *golden) {
  unsigned s_tile[4] = {0}, s_row[8] = {0}, s_col[4] = {0}, s_total = 0;   /* BF16 sums wrong */
  unsigned b_tile[4] = {0}, b_row[8] = {0}, b_col[4] = {0}, b_total = 0;   /* E4M3 bytes wrong */
  int shown = 0;
  for (int t = 0; t < 4; t++) {
    int i = t >> 1, j = t & 1;
    for (int m = 0; m < 32; m++) {
      for (int n = 0; n < 32; n++) {
        int idx = t * 1024 + m * 32 + n;
        uint16_t want = bf16_bits(ref_sum(in, i, j, m, n));
        int sum_bad = (c16[idx] != want);
        int byte_bad = (c8[idx] != golden[idx]);
        s_total += sum_bad; s_tile[t] += sum_bad; s_row[m % 8] += sum_bad; s_col[n / 8] += sum_bad;
        b_total += byte_bad; b_tile[t] += byte_bad; b_row[m % 8] += byte_bad; b_col[n / 8] += byte_bad;
        if (sum_bad && shown < 8) {
          printf("diag first_bad tile=%d m=%d n=%d c16=0x%04x ref=0x%04x c8=0x%02x golden=0x%02x\n",
                 t, m, n, c16[idx], want, c8[idx], golden[idx]);
          shown++;
        }
      }
    }
  }
  printf("diag sums_wrong=%u bytes_wrong=%u of 4096\n", s_total, b_total);
  printf("diag by_tile sums=%u,%u,%u,%u bytes=%u,%u,%u,%u\n", s_tile[0], s_tile[1], s_tile[2], s_tile[3],
         b_tile[0], b_tile[1], b_tile[2], b_tile[3]);
  printf("diag by_row_in_block sums=%u,%u,%u,%u,%u,%u,%u,%u bytes=%u,%u,%u,%u,%u,%u,%u,%u\n",
         s_row[0], s_row[1], s_row[2], s_row[3], s_row[4], s_row[5], s_row[6], s_row[7],
         b_row[0], b_row[1], b_row[2], b_row[3], b_row[4], b_row[5], b_row[6], b_row[7]);
  printf("diag by_column_group sums=%u,%u,%u,%u bytes=%u,%u,%u,%u\n", s_col[0], s_col[1], s_col[2], s_col[3],
         b_col[0], b_col[1], b_col[2], b_col[3]);
}
