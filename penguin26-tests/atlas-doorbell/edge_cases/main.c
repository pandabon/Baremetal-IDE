/*
 * main.c - atlas_doorbell_edge_cases: what the doorbell does when things do not go the usual way.
 *
 * Start-up and UART set-up follow penguin26-tests/atlas_example/src/main.c. The kernel and the matmul data
 * are the Atlas team's (credited in common/matmul_data.h).
 *
 * The host always runs its job (STEPS steps, far longer than two matmuls) with interrupts on; it never waits
 * or sleeps. Each case checks what the doorbell layer and the hooks saw. 3 repeats of four cases:
 *   no_ring  the team kernel (no doorbell epilogue): 0 doorbells; the watchdog fires mid-job (after 50 mtime
 *            ticks, about 50k cycles) and is counted; Atlas finished (STATUS 5) with a golden result.
 *   error    a kernel that hits an illegal instruction before any doorbell: 0 doorbells, the watchdog fires,
 *            STATUS 3 (halted, reason 1 = illegal) and the illegal-instruction PC is 1 (Atlas PCs count
 *            words: ScalarCore.scala:279 link = pc + 1). Halt codes: sp26-atlas-acc 2ae0bef,
 *            src/main/scala/diplomatic/memory/CSRFile.scala:8-17.
 *   rering   the hook rings msip again inside the handler: exactly 2 doorbells, the second one right after
 *            mret (interrupts are off inside the handler); the watchdog (1 s) must not fire.
 *   chain    divide the work: the hook checks the first result, clears it and starts Atlas again from the
 *            handler; the second doorbell's result is checked too. 2 doorbells, both results golden, and the
 *            two doorbells landed mid-job, in order (0 < step1 < step2 < STEPS).
 * In every case the job checksum must equal the job run alone (the interrupts did not disturb the host).
 *
 * Line format: label, then mode (65 'A' no_ring, 66 'B' error, 67 'C' rering, 68 'D' chain), doorbells, timer
 * (watchdog fires), step1/step2 (job step at doorbell 1/2), status (Atlas STATUS), T_total (cycles of
 * launch + job), [ill_pc for error], ok.
 *
 * Differences from a hand-written trap handler, forced by using glossy's trap path:
 *   - mcause 3 is implied: glossy calls the doorbell callback only for a machine software interrupt, so the
 *     explicit mcause check of case C is dropped.
 *   - the job keeps its step in the volatile job_step (host_job_tracked: one extra store per step), because
 *     a C hook cannot read the job's register.
 *   - the watchdog reaction (count it, stay disarmed, carry on) is installed with watchdog_set_hook().
 * This version targets the RTL only.
 *
 * Pass (return 0): every line has ok=1.
 * Output: 3 x 4 lines of key=value, then *** PASSED *** or *** FAILED ***.
 */
#include <stdio.h>
#include <stdint.h>

#include "chip_config.h"
#include "rocketcore.h"
#include "atlas_doorbell.h"
#include "atlas_run.h"
#include "matmul_data.h"  /* team_kernel, doorbell_kernel */

#define REPEATS    3
#define STEPS      100000   /* job length, far longer than two matmuls */
#define WD_SHORT   50       /* no_ring, error: watchdog after 50 mtime ticks (~50k cycles), mid-job */
#define WD_LONG    500000   /* rering, chain: 1 s, must not fire */

#define MODE_NO_RING 'A'
#define MODE_ERROR   'B'
#define MODE_RERING  'C'
#define MODE_CHAIN   'D'

/* An Atlas program that faults: x5 = 0, then word 0 (no Atlas opcode 0000000: illegal); the ecall is never
 * reached. */
static const uint32_t error_kernel[] = {0x00000293, 0x00000000, 0x00000073};

/* state of the current run, written by the hooks */
static volatile uint64_t mode;
static volatile uint64_t n_db;      /* doorbells in this run */
static volatile uint64_t n_timer;   /* watchdog fires in this run */
static volatile uint64_t step1;     /* job step at doorbell 1 */
static volatile uint64_t step2;     /* job step at doorbell 2 */
static volatile uint64_t ok1;       /* chain: result check after doorbell 1 */
static volatile uint64_t ok2;       /* chain: result check after doorbell 2 */

/* The doorbell hook (msip is already cleared and counted by the doorbell layer). */
static void on_doorbell(void *context) {
  (void)context;
  uint64_t n = n_db + 1;
  n_db = n;
  if (n == 1) {
    step1 = job_step;
  } else {
    step2 = job_step;
  }
  if (mode == MODE_RERING) {
    /* Ring once more. Interrupts are off inside the handler, so the second trap is taken right after mret. */
    if (n == 1) {
      atlas_doorbell_ring_self();
    }
  } else if (mode == MODE_CHAIN) {
    /* Result 1: check it, clear it and launch Atlas again. Result 2: check it. */
    uint64_t ok = (uint64_t)mm_check();
    if (n == 1) {
      ok1 = ok;
      mm_setup();   /* clear the output, so result 2 must be fresh */
      atlas_run_start();
    } else {
      ok2 = ok;
    }
  }
}

/* The watchdog reaction: count it and carry on (the watchdog is already disarmed). */
static void on_timer(void) {
  n_timer = n_timer + 1;
}

/* Launch a kernel, run the job with interrupts on; returns 1 if the job checksum equals `reference`.
 * The counters are left in the globals above. */
static int run_case(uint64_t m, const uint32_t *kernel, uint32_t n_words, uint64_t watchdog_ticks,
                    uint64_t reference, uint64_t *t_total) {
  mode = m;
  atlas_run_load(kernel, n_words);
  mm_setup();
  n_db = 0;
  n_timer = 0;
  step1 = 0;
  step2 = 0;
  ok1 = 0;
  ok2 = 0;
  watchdog_arm(watchdog_ticks);
  uint64_t start = get_cycles();
  atlas_run_start();
  enable_global_interrupt();
  uint64_t checksum = host_job_tracked(STEPS);
  disable_global_interrupt();
  *t_total = get_cycles() - start;
  watchdog_off();
  return checksum == reference;
}

/* Print the fields every line has. ill_pc < 0: leave that field out. */
static void print_line(const char *label, uint32_t status, uint64_t t_total, int64_t ill_pc, int ok) {
  printf("%s mode=%lu doorbells=%lu timer=%lu step1=%lu step2=%lu status=%lu T_total=%lu", label,
         (unsigned long)mode, (unsigned long)n_db, (unsigned long)n_timer, (unsigned long)step1,
         (unsigned long)step2, (unsigned long)status, (unsigned long)t_total);
  if (ill_pc >= 0) {
    printf(" ill_pc=%lu", (unsigned long)ill_pc);
  }
  printf(" ok=%d\n", ok);
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
  uint64_t t_total;

  atlas_doorbell_init();
  atlas_doorbell_set_hook(on_doorbell, NULL);
  watchdog_set_hook(on_timer);

  uint64_t reference = host_job_tracked(STEPS);   /* the job alone */

  for (int rep = 0; rep < REPEATS; rep++) {
    int ok;
    uint32_t status;

    /* no_ring: no doorbell kernel -> only the watchdog */
    ok = run_case(MODE_NO_RING, team_kernel, sizeof(team_kernel) / sizeof(team_kernel[0]), WD_SHORT,
                  reference, &t_total);
    status = atlas_run_status();
    ok &= (n_db == 0);
    ok &= (n_timer == 1);
    ok &= (status == ATLAS_STATUS_DONE);
    ok &= mm_check();
    print_line("no_ring", status, t_total, -1, ok);
    failures += !ok;

    /* error: the kernel faults before the doorbell */
    ok = run_case(MODE_ERROR, error_kernel, sizeof(error_kernel) / sizeof(error_kernel[0]), WD_SHORT,
                  reference, &t_total);
    status = atlas_run_status();
    uint32_t ill_pc = ATLAS->ILLEGAL_PC;
    ok &= (n_db == 0);
    ok &= (n_timer == 1);
    ok &= (status == 3);          /* halted, reason 1 (illegal) */
    ok &= (ill_pc == 1);
    print_line("error", status, t_total, (int64_t)ill_pc, ok);
    failures += !ok;

    /* rering: a second ring from inside the handler */
    ok = run_case(MODE_RERING, doorbell_kernel, sizeof(doorbell_kernel) / sizeof(doorbell_kernel[0]),
                  WD_LONG, reference, &t_total);
    status = atlas_run_status();
    ok &= (n_db == 2);
    ok &= (n_timer == 0);
    ok &= (status == ATLAS_STATUS_DONE);
    ok &= mm_check();
    print_line("rering", status, t_total, -1, ok);
    failures += !ok;

    /* chain: the handler dispatches Atlas's next job */
    ok = run_case(MODE_CHAIN, doorbell_kernel, sizeof(doorbell_kernel) / sizeof(doorbell_kernel[0]),
                  WD_LONG, reference, &t_total);
    status = atlas_run_status();
    ok &= (n_db == 2);
    ok &= (n_timer == 0);
    ok &= (status == ATLAS_STATUS_DONE);
    ok &= (ok1 == 1);
    ok &= (ok2 == 1);
    ok &= (step1 != 0) && (step1 < step2) && (step2 < STEPS);   /* both doorbells mid-job, in order */
    print_line("chain", status, t_total, -1, ok);
    failures += !ok;
  }

  printf(failures ? "*** FAILED ***\n" : "*** PASSED ***\n");
  return failures ? 1 : 0;
}

int main(void) {
  app_init();
  return app_main();
}
