/*
 * main.c - atlas_doorbell_poll_baseline: how today's runtimes wait for Atlas. Interrupts stay off; the host
 * polls Atlas's STATUS register until the kernel halts.
 *
 * Start-up and UART set-up follow penguin26-tests/atlas_example/src/main.c. The kernel and the matmul data
 * are the Atlas team's (credited in common/matmul_data.h); the launch and poll rule is in common/atlas_run.h.
 *
 * Ten repeats of three kinds of run:
 *   team    the Atlas team's matmul, unchanged, polled. T_atlas = cycles from the start command until the
 *           poll sees the halt; polls = number of STATUS reads; ok = result equals golden.
 *   doorbell the same matmul with the doorbell epilogue, polled. It shows what the epilogue costs: msip is
 *           set but nobody listens (interrupts are off), so ok also needs msip to be 1 afterwards. The host
 *           clears it with atlas_doorbell_ack().
 *   serial  poll until Atlas is done, then run the host job (job sizes shorter than, about equal to and
 *           longer than the matmul; 4000 and 8000 are the overlap test's sizes), with the doorbell kernel. job = steps, T_job = the job alone,
 *           T_serial = start + poll + job, ok = same checksum as the job alone and golden result.
 *   serial_team  the same with the team's kernel, unchanged: exactly what runs today, with no doorbell
 *           anywhere. The serial time is what a doorbell can win back by overlapping the two (see the
 *           overlap test).
 * The job is host_job_tracked, the same job code the overlap test runs, so T_serial and the overlap
 * test's T_total time the same host work.
 *
 * Pass (return 0): in every line the result is golden and Atlas halted with STATUS 5 (ECALL).
 * Output: 10 x (1 team + 1 doorbell + 4 serial + 4 serial_team) lines of key=value, then *** PASSED *** or
 * *** FAILED ***.
 */
#include <stdio.h>
#include <stdint.h>

#include "chip_config.h"
#include "rocketcore.h"
#include "atlas_doorbell.h"
#include "atlas_run.h"

#define REPEATS 10

static const uint64_t job_sizes[] = {1000, 2700, 4000, 8000};
#define N_SIZES (sizeof(job_sizes) / sizeof(job_sizes[0]))

static int failures;

/* A line is ok only if its result is right AND Atlas ended with STATUS 5; count the bad ones. */
static int verify(int ok, uint32_t status) {
  if (status != ATLAS_STATUS_DONE) {
    ok = 0;
  }
  if (!ok) {
    failures++;
  }
  return ok;
}

/* Serial runs of the loaded kernel: poll until Atlas is done, then the job; one line per job size. */
static void run_serial(const char *label) {
  for (unsigned i = 0; i < N_SIZES; i++) {
    uint64_t steps = job_sizes[i];

    uint64_t t0 = get_cycles();
    uint64_t reference = host_job_tracked(steps);
    uint64_t t_job = get_cycles() - t0;

    mm_setup();
    t0 = get_cycles();
    atlas_run_start();
    uint32_t status = atlas_run_poll(NULL);
    uint64_t checksum = host_job_tracked(steps);
    uint64_t t_serial = get_cycles() - t0;

    int ok = (checksum == reference);
    ok &= mm_check();
    atlas_doorbell_ack();   /* the doorbell kernel set msip; nothing listens here */
    ok = verify(ok, status);
    printf("%s job=%lu T_job=%lu T_serial=%lu ok=%d\n", label, (unsigned long)steps,
           (unsigned long)t_job, (unsigned long)t_serial, ok);
  }
}

/* Launch the loaded kernel and poll it; returns the cycles it took. */
static uint64_t run_polled(uint32_t *status, uint64_t *polls) {
  uint64_t t0 = get_cycles();
  atlas_run_start();
  *status = atlas_run_poll(polls);
  return get_cycles() - t0;
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
  /* Only to start from a clean msip and stats. Global interrupts stay off for the whole test, so no
   * doorbell is ever taken here: the doorbell kernel's msip write just stays pending until we clear it. */
  atlas_doorbell_init();

  for (int rep = 0; rep < REPEATS; rep++) {
    uint32_t status;
    uint64_t polls;
    uint64_t t_atlas;
    int ok;

    /* team kernel, polled */
    atlas_run_load_team();
    mm_setup();
    t_atlas = run_polled(&status, &polls);
    ok = mm_check();
    atlas_doorbell_ack();
    ok = verify(ok, status);
    printf("team T_atlas=%lu polls=%lu ok=%d\n", (unsigned long)t_atlas, (unsigned long)polls, ok);

    /* doorbell kernel, polled */
    atlas_run_load_doorbell();
    mm_setup();
    t_atlas = run_polled(&status, &polls);
    ok = mm_check();
    int rang = (int)(CLINT->MSIP[get_hart_id()] & 1U);   /* the epilogue set msip */
    atlas_doorbell_ack();
    ok = verify(ok & rang, status);
    printf("doorbell T_atlas=%lu polls=%lu ok=%d\n", (unsigned long)t_atlas, (unsigned long)polls, ok);

    /* serial: poll until done, then the job; the doorbell kernel is still loaded */
    run_serial("serial");

    /* serial_team: the same with the team's kernel, no doorbell */
    atlas_run_load_team();
    run_serial("serial_team");
  }

  printf(failures ? "*** FAILED ***\n" : "*** PASSED ***\n");
  return failures ? 1 : 0;
}

int main(void) {
  app_init();
  return app_main();
}
