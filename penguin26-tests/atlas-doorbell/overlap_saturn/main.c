/*
 * main.c - atlas_doorbell_overlap_saturn: Atlas and the host's vector unit (Saturn) each compute the same
 * 64x64x64 FP8 matmul at the same time. The doorbell tells the host when Atlas is done; nothing waits.
 *
 * Start-up and UART set-up follow penguin26-tests/atlas_example/src/main.c. Atlas's kernel and the data are
 * the Atlas team's (credited in common/matmul_data.h); the Saturn kernel is common/fp8_matmul.S (credited there).
 *
 * How, 10 repeats of four runs:
 *   saturn_alone  Saturn's matmul alone: transpose B (pre), FP8 multiply-accumulate into BF16 (mac), BF16 to
 *                 E4M3 (cvt). Times per phase; exact = output bytes equal to the golden data.
 *   atlas_alone   Atlas's matmul alone, polled (interrupts off): T_atlas.
 *   serial        today's way: start Atlas, poll until done, then Saturn's matmul. T_serial.
 *   overlap       start Atlas, turn interrupts on, run Saturn's matmul; the doorbell interrupts Saturn
 *                 wherever it is, the doorbell layer clears msip and counts, Saturn continues. T_total.
 *                 speedup = T_serial / T_total of the same repeat.
 * Both units read the same inputs: Atlas from DRAM at MM_IN (its DMA), Saturn from the program's copy, so the
 * overlap also shows whether the two slow each other down (compare its phases with saturn_alone).
 *
 * Timeline stamps (mcycle, cycles from `start`, the stamp just before the start command), one "stamps" line per
 * overlap run: launched, irq_on, pre_end, mac_end, cvt_end (Saturn's phases end), cb_entry / cb_acked / cb_exit
 * (the doorbell callback started / cleared msip / finished), end (interrupts off). The doorbell can land in any
 * phase; the stamps show which.
 *
 * Interrupting vector code: glossy's trap entry saves the 32 integer registers only, not the vector or FP
 * registers or vl/vtype. That is safe here because the doorbell callback is plain C built without the vector
 * extension and uses no FP; a callback that used either would need to save them itself.
 *
 * Pass (return 0): in every overlap run exactly one doorbell, Atlas's output golden with STATUS 5 (ECALL), and
 * Saturn's output exact (4096 of 4096 bytes); the alone and serial runs correct too. A watchdog (1 s) exits with
 * code 3 if the doorbell never comes.
 * Output: 10 x (saturn_alone, atlas_alone, serial, overlap, stamps) lines of key=value, then *** PASSED *** or
 * *** FAILED ***.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "chip_config.h"
#include "rocketcore.h"
#include "atlas_doorbell.h"
#include "atlas_run.h"
#include "matmul_data.h"
#include "fp8_matmul.h"

#define REPEATS        10
#define WATCHDOG_TICKS 500000   /* 1 s: mtime runs at core clock / 1000 */
#define N              4096     /* result bytes: 64 x 64 */

static uint8_t bscr[4096] __attribute__((aligned(64)));  /* B in row-major order */
static uint16_t c16[4096] __attribute__((aligned(64)));  /* BF16 sums, 8 KiB */
static uint8_t c8[N] __attribute__((aligned(64)));       /* E4M3 result */

/* Times of Saturn's three phases, as absolute mcycle values. */
typedef struct {
  uint64_t start, pre_end, mac_end, cvt_end;
} saturn_times;

/* Saturn's whole matmul; the fences make each phase's vector stores complete before its time is taken. */
static void saturn_matmul(saturn_times *t) {
  memset(c8, 0, sizeof(c8));   /* a stale result cannot pass */
  asm volatile("fence" ::: "memory");
  t->start = get_cycles();
  fp8_matmul_pre(mm_in, bscr);
  asm volatile("fence" ::: "memory");
  t->pre_end = get_cycles();
  fp8_matmul_mac(mm_in, bscr, c16);
  asm volatile("fence" ::: "memory");
  t->mac_end = get_cycles();
  fp8_matmul_cvt(c16, c8);
  asm volatile("fence" ::: "memory");
  t->cvt_end = get_cycles();
}

/* Number of Saturn output bytes equal to the golden data (4096 = exact). */
static unsigned saturn_exact(void) {
  const uint8_t *golden = (const uint8_t *)mm_golden;
  unsigned exact = 0;
  for (int i = 0; i < N; i++) {
    exact += (c8[i] == golden[i]);
  }
  return exact;
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
  int failures = 0;
  saturn_times st;

  atlas_doorbell_init();          /* doorbell callback only counts and stamps (no hook) */
  atlas_run_load_doorbell();

  for (int rep = 0; rep < REPEATS; rep++) {
    /* saturn_alone */
    saturn_matmul(&st);
    unsigned exact = saturn_exact();
    int ok = (exact == N);
    failures += !ok;
    printf("saturn_alone T_sat=%lu T_pre=%lu T_mac=%lu T_cvt=%lu exact=%u ok=%d\n",
           (unsigned long)(st.cvt_end - st.start), (unsigned long)(st.pre_end - st.start),
           (unsigned long)(st.mac_end - st.pre_end), (unsigned long)(st.cvt_end - st.mac_end), exact, ok);

    /* atlas_alone: polled, interrupts off; the doorbell kernel's msip write is cleared afterwards */
    mm_setup();
    uint64_t t0 = get_cycles();
    atlas_run_start();
    uint32_t status = atlas_run_poll(NULL);
    uint64_t t_atlas = get_cycles() - t0;
    ok = mm_check() && (status == ATLAS_STATUS_DONE);
    atlas_doorbell_ack();
    failures += !ok;
    printf("atlas_alone T_atlas=%lu ok=%d\n", (unsigned long)t_atlas, ok);

    /* serial: poll Atlas until done, then Saturn's matmul */
    mm_setup();
    t0 = get_cycles();
    atlas_run_start();
    status = atlas_run_poll(NULL);
    saturn_matmul(&st);
    uint64_t t_serial = get_cycles() - t0;
    exact = saturn_exact();
    ok = mm_check() && (status == ATLAS_STATUS_DONE) && (exact == N);
    atlas_doorbell_ack();
    failures += !ok;
    printf("serial T_serial=%lu exact=%u ok=%d\n", (unsigned long)t_serial, exact, ok);

    /* overlap */
    mm_setup();
    uint64_t count_before = atlas_doorbell.count;
    watchdog_arm(WATCHDOG_TICKS);
    uint64_t start = get_cycles();
    atlas_run_start();
    uint64_t launched = get_cycles();
    enable_global_interrupt();
    uint64_t irq_on = get_cycles();
    saturn_matmul(&st);
    disable_global_interrupt();
    uint64_t end = get_cycles();
    watchdog_off();

    exact = saturn_exact();
    uint64_t doorbells = atlas_doorbell.count - count_before;
    ok = (doorbells == 1) && mm_check() && (atlas_run_status() == ATLAS_STATUS_DONE) && (exact == N);
    failures += !ok;
    uint64_t total = end - start;
    printf("overlap T_total=%lu T_sat=%lu T_pre=%lu T_mac=%lu T_cvt=%lu exact=%u doorbells=%lu speedup_x100=%lu ok=%d\n",
           (unsigned long)total, (unsigned long)(st.cvt_end - st.start), (unsigned long)(st.pre_end - st.start),
           (unsigned long)(st.mac_end - st.pre_end), (unsigned long)(st.cvt_end - st.mac_end), exact,
           (unsigned long)doorbells, (unsigned long)(t_serial * 100 / total), ok);
    printf("stamps launched=%lu irq_on=%lu pre_end=%lu mac_end=%lu cvt_end=%lu cb_entry=%lu cb_acked=%lu cb_exit=%lu "
           "end=%lu\n",
           (unsigned long)(launched - start), (unsigned long)(irq_on - start), (unsigned long)(st.pre_end - start),
           (unsigned long)(st.mac_end - start), (unsigned long)(st.cvt_end - start),
           (unsigned long)(atlas_doorbell.last_entry - start), (unsigned long)(atlas_doorbell.last_acked - start),
           (unsigned long)(atlas_doorbell.last_exit - start), (unsigned long)(end - start));
  }

  printf(failures ? "*** FAILED ***\n" : "*** PASSED ***\n");
  return failures ? 1 : 0;
}

int main(void) {
  app_init();
  return app_main();
}
