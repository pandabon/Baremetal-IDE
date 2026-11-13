/*
 * main.c - atlas_doorbell_overlap: the host runs its own job while Atlas computes, and the doorbell
 * interrupt only acknowledges. Nothing ever waits.
 *
 * Start-up and UART set-up follow penguin26-tests/atlas_example/src/main.c. The kernel and the matmul data
 * are the Atlas team's (credited in common/matmul_data.h).
 *
 * How it works: the doorbell kernel is loaded once. For each job size (4000 and 8000 steps, both longer than
 * the matmul) and 10 repeats: run the job alone (T_job, and its checksum), then start Atlas and run the same
 * job with interrupts on. The job never looks at Atlas. The doorbell interrupts it mid-job; the doorbell interrupt (hal_atlas_irq)
 * clears msip and counts, then the hook only records which job step was running. Global interrupts are on
 * only around the job itself, and a watchdog (1 s) exits with code 3 if no doorbell ever comes.
 *
 * Per line: job = steps, T_job = job alone, T_total = launch + job with interrupts, overhead = T_total - T_job
 * (launch plus the interruption), T_launch = cycles for the start command, rang_at = cycles from before the
 * start until the doorbell callback began, step = job step when the doorbell hit, handler = the hook's own
 * length, ok = the run was correct.
 *
 * Timeline timestamps (mcycle, cycles from `start`, the timestamp taken just before the start command), for drawing
 * the run: launched = the start command returned; irq_on = global interrupts enabled (the job starts right
 * after); cb_entry = doorbell callback started (= rang_at); cb_acked = msip cleared; hook_in / hook_out = the
 * hook's start and end; cb_exit = callback finished (glossy's register restore and mret follow, not stamped);
 * job_end = the job returned; end = interrupts off (= T_total). Each timestamp costs a few cycles, which T_total
 * includes.
 *
 * Differences from a hand-written trap handler, forced by using glossy's trap path:
 *   - rang_at is stamped at the start of the doorbell callback (atlas_irq_timestamps.entry), which is after
 *     glossy has saved all registers, so it is later than a timestamp at the very first trap instruction.
 *   - glossy has no exit timestamp, so "handler" is the hook's own length (timestamp at the hook's start and end);
 *     it leaves out clearing msip, counting, glossy's register save/restore and mret.
 *   - the job keeps its step in the volatile job_step (host_job_tracked: one extra store per step), because
 *     a C hook cannot read the job's register. T_job uses the same function, so overhead stays comparable.
 *   - mcause 3 is implied: glossy calls the doorbell callback only for a machine software interrupt.
 *
 * Pass (return 0): in every run exactly one doorbell came, the job checksum equals the job alone, the result
 * is golden and Atlas ended with STATUS 5 (ECALL).
 * Output: 20 x (one overlap line + one timestamps line) of key=value, then *** PASSED *** or *** FAILED ***.
 */
#include <stdio.h>
#include <stdint.h>

#include "chip_config.h"
#include "rocketcore.h"
#include "hal_atlas_irq.h"
#include "atlas_run.h"

#define REPEATS        10
#define WATCHDOG_TICKS 500000   /* 1 s: mtime runs at core clock / 1000 */

static const uint64_t job_sizes[] = {4000, 8000};
#define N_SIZES (sizeof(job_sizes) / sizeof(job_sizes[0]))

/* written by the hook, read by the test */
static volatile uint64_t irq_step;
static volatile uint64_t hook_start;
static volatile uint64_t hook_end;

/* Runs inside the doorbell interrupt (msip is already cleared and counted). */
static void on_doorbell(void *context) {
  (void)context;
  hook_start = get_cycles();
  irq_step = job_step;
  hook_end = get_cycles();
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

  atlas_irq_init();
  atlas_irq_set_handler(on_doorbell, NULL);
  atlas_run_load_doorbell();

  for (unsigned s = 0; s < N_SIZES; s++) {
    uint64_t steps = job_sizes[s];
    for (int rep = 0; rep < REPEATS; rep++) {
      /* the job alone */
      uint64_t t0 = get_cycles();
      uint64_t reference = host_job_tracked(steps);
      uint64_t t_job = get_cycles() - t0;

      mm_setup();
      uint64_t count_before = atlas_irq_count();
      watchdog_arm(WATCHDOG_TICKS);

      /* ---- timed: launch + job with interrupts on ---- */
      uint64_t start = get_cycles();
      atlas_run_start();
      uint64_t launched = get_cycles();
      enable_global_interrupt();
      uint64_t irq_on = get_cycles();
      uint64_t checksum = host_job_tracked(steps);
      uint64_t job_end = get_cycles();
      disable_global_interrupt();
      uint64_t end = get_cycles();
      /* ---- end ---- */

      watchdog_off();

      int ok = (checksum == reference);
      ok &= (atlas_irq_count() - count_before == 1);
      ok &= mm_check();
      ok &= (atlas_run_status() == ATLAS_STATUS_DONE);
      if (!ok) {
        failures++;
      }

      uint64_t total = end - start;
      printf("overlap job=%lu T_job=%lu T_total=%lu overhead=%ld T_launch=%lu rang_at=%lu step=%lu handler=%lu ok=%d\n",
             (unsigned long)steps, (unsigned long)t_job, (unsigned long)total, (long)(total - t_job),
             (unsigned long)(launched - start), (unsigned long)(atlas_irq_timestamps.entry - start),
             (unsigned long)irq_step, (unsigned long)(hook_end - hook_start), ok);
      printf("timestamps job=%lu launched=%lu irq_on=%lu cb_entry=%lu cb_acked=%lu hook_in=%lu hook_out=%lu cb_exit=%lu "
             "job_end=%lu end=%lu\n",
             (unsigned long)steps, (unsigned long)(launched - start), (unsigned long)(irq_on - start),
             (unsigned long)(atlas_irq_timestamps.entry - start), (unsigned long)(atlas_irq_timestamps.acked - start),
             (unsigned long)(hook_start - start), (unsigned long)(hook_end - start),
             (unsigned long)(atlas_irq_timestamps.exit - start), (unsigned long)(job_end - start),
             (unsigned long)(end - start));
    }
  }

  printf(failures ? "*** FAILED ***\n" : "*** PASSED ***\n");
  return failures ? 1 : 0;
}

int main(void) {
  app_init();
  return app_main();
}
