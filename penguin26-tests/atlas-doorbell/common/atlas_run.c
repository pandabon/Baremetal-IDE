/**
 * @file atlas_run.c
 * @brief Launch/poll helpers, matmul data helpers and the host job. See atlas_run.h.
 */
#include "atlas_run.h"

#include "chip_config.h"
#include "hal_atlas.h"
#include "matmul_data.h"  /* kernels, mm_in, mm_golden, MM_IN, MM_OUT */

/* ---- Atlas: load, start, wait ---- */

/* Retired-instruction count noted at start; the poll uses it to tell "ran" from "stale halt". */
static uint32_t retired_at_start;

void atlas_run_load(const uint32_t *words, uint32_t n_words) {
  atlas_reset_for_load(ATLAS);                 /* stop Atlas, clear DBG0/DBG1, fence */
  atlas_load_program(ATLAS_IMEM, words, n_words);  /* word-by-word copy + fence */
  (void)ATLAS_IMEM[0];                         /* read one word back: the IMEM writes are done */
  atlas_fence();
}

void atlas_run_load_team(void) {
  atlas_run_load(team_kernel, sizeof(team_kernel) / sizeof(team_kernel[0]));
}

void atlas_run_load_doorbell(void) {
  atlas_run_load(doorbell_kernel, sizeof(doorbell_kernel) / sizeof(doorbell_kernel[0]));
}

void atlas_run_start(void) {
  ATLAS->DBG0 = 0;
  retired_at_start = ATLAS->INSTCNT;
  atlas_fence();
  ATLAS->EXEC_CONTROL = ATLAS_EXEC_START;
}

uint32_t atlas_run_status(void) {
  return ATLAS->STATUS;
}

uint32_t atlas_run_poll(uint64_t *polls) {
  uint32_t status;
  uint64_t n = 0;
  int seen_running = 0;  /* Atlas has visibly started */
  for (;;) {
    n++;
    status = ATLAS->STATUS;
    if (!(status & ATLAS_STATUS_HALTED_MSK)) {
      seen_running = 1;                         /* not halted: it is running */
      continue;
    }
    if (((status & ATLAS_STATUS_REASON_MSK) >> ATLAS_STATUS_REASON_POS) == ATLAS_HALT_NONE) {
      seen_running = 1;                         /* halted with no reason yet: not a real halt */
      continue;
    }
    if (seen_running) {
      break;                                    /* halted after we saw it start: done */
    }
    if (ATLAS->INSTCNT != retired_at_start) {
      seen_running = 1;                         /* the counter moved: it ran, so this halt counts */
    }
  }
  if (polls) {
    *polls = n;
  }
  return status;
}

/* ---- the matmul's data in DRAM ---- */

void mm_setup(void) {
  volatile uint32_t *in = (volatile uint32_t *)MM_IN;
  volatile uint32_t *out = (volatile uint32_t *)MM_OUT;
  for (uint32_t i = 0; i < sizeof(mm_in) / sizeof(mm_in[0]); i++) {
    in[i] = mm_in[i];
  }
  for (uint32_t i = 0; i < 4096 / 4; i++) {
    out[i] = 0;
  }
  atlas_fence();
}

int mm_check(void) {
  volatile uint32_t *out = (volatile uint32_t *)MM_OUT;
  atlas_fence();
  for (uint32_t i = 0; i < sizeof(mm_golden) / sizeof(mm_golden[0]); i++) {
    if (out[i] != mm_golden[i]) {
      return 0;
    }
  }
  return 1;
}

/* ---- the host's own work ---- */

volatile uint64_t job_step;

/* xorshift64 (13, 7, 17) from seed 88172645463325252, summed into a checksum. The empty asm after each
 * step hands x to the compiler as "changed by something unknown", so it can neither vectorize the loop nor
 * replace it with a closed form; the cost per step stays that of the plain scalar instructions. */
static inline __attribute__((always_inline)) uint64_t job_body(uint64_t steps, int track) {
  uint64_t x = 88172645463325252ULL;
  uint64_t sum = 0;
  uint64_t n = 0;
  if (track) {
    job_step = 0;
  }
  while (n < steps) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    asm volatile("" : "+r"(x));
    sum += x;
    n++;
    if (track) {
      job_step = n;   /* the one extra store per step that lets an interrupt hook see how far the job got */
    }
  }
  return sum;
}

uint64_t host_job(uint64_t steps) {
  return job_body(steps, 0);
}

uint64_t host_job_tracked(uint64_t steps) {
  return job_body(steps, 1);
}
