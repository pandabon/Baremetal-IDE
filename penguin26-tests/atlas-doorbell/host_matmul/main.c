/*
 * main.c - atlas_doorbell_host_matmul: the same 64x64x64 FP8 matmul the Atlas tests run, computed on
 * the host with Saturn's FP8 vector FMA, to compare its cycle count against Atlas.
 *
 * How: the host reads A and B from mm_in (the same bytes Atlas gets) and computes C in three phases,
 * each timed with mcycle (get_cycles()) and a fence before the clock is read, so the vector stores
 * are done when the time is taken. The vector code is in common/fp8_matmul.S; this file only times,
 * compares and prints.
 *   T_pre  transpose B to row-major (the host needs it, Atlas gets B transposed for free)
 *   T_mac  the multiply-accumulate loop: widening FP8 FMAs into BF16 accumulators
 *   T_cvt  BF16 -> E4M3, with the generator's rounding, flush and saturation
 *   T_host_mm = T_pre + T_mac + T_cvt
 * The whole matmul runs RUNS (5) times. Each run clears the result first, so a stale result cannot
 * pass. Per run one line of key=value:
 *   host_matmul T_host_mm=.. exact=.. total=4096 maxdiff=.. T_pre=.. T_mac=.. T_cvt=.. run=.. fail=..
 * exact = result bytes equal to mm_golden (of total 4096); maxdiff = largest difference in E4M3 code
 * steps (see below).
 *
 * Pass (return 0): in every run maxdiff <= TOL (2 code steps). The golden data is the BF16 systolic
 * result of the Atlas generator converted to E4M3. With the same accumulation order the result is
 * exact (4096/4096) in a Python model; a different order (exact dot product per K tile) gives 3434
 * exact and 4024 within 2. So a tolerance of 2 forgives FMA rounding detail but not a wrong layout.
 *
 * Credit: the vfwmacc loop, the altfmt vsetvli encodings and the vmv.v.x splat come from Saturn's
 * benchmarks: repo ucb-bar/saturn (tapeout-chipyard/generators/saturn) commit de04946
 * (de04946ac1ce878c79a4d6e74e493c5f440908e2), benchmarks/vec-mx-matmul/main.c, benchmarks/common/rvv_mx.h,
 * benchmarks/vec-mx-gemv/main.c. Data layout and golden: see fp8_matmul.S and common/matmul_data.h.
 * Start-up and UART set-up follow penguin26-tests/atlas_example.
 *
 * Spike cannot run the FP8 FMA (it supports vector FP only at SEW 16/32/64), so this does not pass there.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "chip_config.h"
#include "rocketcore.h"
#include "matmul_data.h"
#include "fp8_matmul.h"   /* the vector kernel, common/fp8_matmul.S */

#define RUNS 5
#define TOL  2          /* allowed |difference| in E4M3 code steps */
#define N    4096       /* result bytes: 64 x 64 */

void host_matmul_diagnose(const uint8_t *in, const uint16_t *c16, const uint8_t *c8, const uint8_t *golden);

static uint8_t bscr[4096] __attribute__((aligned(64)));  /* B row-major tiles [kk][n] */
static uint16_t c16[4096] __attribute__((aligned(64)));  /* BF16 result, tile order: 64 x 64 sums, 8 KiB */
static uint8_t c8[N] __attribute__((aligned(64)));       /* E4M3 result, golden layout */

/* Position of an E4M3 code on a number line: sign-magnitude to signed, so that codes are monotonic in value. */
static inline int key(uint8_t c) {
  return (c & 0x80) ? -(int)(c & 0x7F) : (int)(c & 0x7F);
}

void app_init() {
#if defined(TERMINAL_DEVICE_UART0)
  /* PLATFORM=CHIP: stdio goes to UART0, whose divisor comes from SYS_CLK_FREQ. */
  UART_InitType UART_init_config;
  UART_init_config.baudrate = 115200;
  UART_init_config.mode = UART_MODE_TX_RX;
  UART_init_config.stopbits = UART_STOPBITS_2;
  uart_init(UART0, &UART_init_config);
#endif
}

int app_main() {
  const uint8_t *golden = (const uint8_t *)mm_golden;
  int fail = 0;

  for (int run = 0; run < RUNS; run++) {
    memset(c8, 0, sizeof(c8));
    asm volatile("fence" ::: "memory");

    uint64_t t0 = get_cycles();
    fp8_matmul_pre(mm_in, bscr);
    asm volatile("fence" ::: "memory");
    uint64_t t1 = get_cycles();
    fp8_matmul_mac(mm_in, bscr, c16);
    asm volatile("fence" ::: "memory");
    uint64_t t2 = get_cycles();
    fp8_matmul_cvt(c16, c8);
    asm volatile("fence" ::: "memory");
    uint64_t t3 = get_cycles();

    unsigned exact = 0, maxdiff = 0;
    for (int i = 0; i < N; i++) {
      int d = key(c8[i]) - key(golden[i]);
      unsigned a = (unsigned)(d < 0 ? -d : d);
      exact += (a == 0);
      if (a > maxdiff) {
        maxdiff = a;
      }
    }
    int run_fail = maxdiff > TOL;
    fail += run_fail;

    printf("host_matmul T_host_mm=%lu exact=%u total=%d maxdiff=%u T_pre=%lu T_mac=%lu T_cvt=%lu run=%d fail=%d\n",
           (unsigned long)(t3 - t0), exact, N, maxdiff, (unsigned long)(t1 - t0),
           (unsigned long)(t2 - t1), (unsigned long)(t3 - t2), run, run_fail);
  }

  if (fail) {
    host_matmul_diagnose((const uint8_t *)mm_in, c16, c8, golden);   /* where is it wrong? (diagnose.c) */
  }
  printf(fail ? "*** FAILED ***\n" : "*** PASSED ***\n");
  return fail != 0;
}

int main(void) {
  app_init();
  return app_main();
}
