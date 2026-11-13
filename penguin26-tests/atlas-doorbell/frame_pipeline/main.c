/*
 * main.c - atlas_doorbell_frame_pipeline: Atlas runs a matmul chained into a GELU, step after step, scheduled from
 * the doorbell interrupt, while the host prepares camera-frame patches. A light proof of concept of the split in
 * SmolVLA's vision path: Atlas does the layer math, the host does what Atlas cannot (pixel bytes in DRAM).
 *
 * Start-up and UART set-up follow penguin26-tests/atlas_example/src/main.c. The two Atlas jobs and their data
 * are the Atlas team's (credited in atlas_jobs.h and common/matmul_data.h).
 *
 * Atlas's work, per step: the team's 64x64x64 FP8 matmul, then the team's GELU (tanh form) on one 32x32 BF16
 * tile on Atlas's vector unit, chained inside Atlas: the matmul (at IMEM word 0x100) ends with a jump into the
 * GELU (word 0x200) instead of halting, the way the team's tri_phase test chains kernels; only the GELU rings
 * the doorbell and halts. A 3-word jump at IMEM[0] (the team's trampoline) enters the matmul on every start, so
 * one start and one doorbell per step. Proof-of-concept simplifications: the GELU runs on the team's own test
 * tile, not on the matmul's output (that needs one fused kernel), every step repeats the same data, and
 * Atlas's steps do not consume the host's patches.
 *
 * The host's work: SigLIP input preparation for SmolVLA's camera images. A 512x512 RGB frame of bytes (made up
 * here, already at SigLIP's size, so no resize) becomes patch vectors for the patch-embedding matmul: each 16x16
 * patch is 768 BF16 values in the conv weight's (channel, row, column) order, each pixel scaled to [-1, 1]
 * (x / 255 * 2 - 1). A 256-entry table maps a pixel byte to its BF16 value, so the loop is integer code only
 * and the doorbell may interrupt it (glossy saves the integer registers only). PATCHES_PER_STEP (2) patches per
 * step: on FireSim 8 patches took ~70k cycles and one Atlas step ~18k, so 2 patches roughly balance the two sides.
 *
 * Scheduling, the point of the test: the doorbell hook is Atlas's scheduler. On every doorbell it notes the
 * time, checks STATUS and starts Atlas's next step. The host's main loop never looks at
 * Atlas until all its own steps are done; then it sleeps in wfi until the last doorbell.
 *
 * Runs, REPEATS (5) times:
 *   serial    today's way: for each step, start Atlas, poll until done, then the step's patches.
 *   pipeline  start the first step, interrupts on, prepare all patches; the hook starts the other steps.
 *             speedup = T_serial / T_pipeline of the same repeat.
 * Before them, each piece alone: one Atlas step (matmul + GELU), one step of patches.
 *
 * Per pipeline run, one "step" line per Atlas step (launch and doorbell in cycles from the run's start, and how
 * many instructions Atlas retired, as a check that both kernels ran: about 210, the two
 * jumps plus the matmul's 130 and the GELU's 75 executed instructions; the run before chaining counted 150 per
 * matmul and 77 per GELU, jump included) and one "prep" line per host
 * step.
 * Before the first start the test also prints msip, to see a stale doorbell if there is one.
 *
 * Pass (return 0): STATUS 5 (ECALL) after every step, STEPS doorbells per pipeline run, the matmul and GELU
 * outputs golden after every run, every patch value equal to a direct computation (float, round to nearest
 * even). A watchdog (1 s) exits with code 3 if a doorbell never comes.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "chip_config.h"
#include "rocketcore.h"
#include "hal_atlas.h"
#include "hal_atlas_irq.h"
#include "atlas_run.h"
#include "matmul_data.h"
#include "atlas_jobs.h"

#define REPEATS          5
#define STEPS            8
#define PATCHES_PER_STEP 2
#define WATCHDOG_TICKS   500000   /* 1 s: mtime runs at core clock / 1000 */

#define FRAME_W   512
#define FRAME_H   512
#define PATCH     16
#define PATCH_LEN (3 * PATCH * PATCH)        /* 768 values */
#define PATCHES   (STEPS * PATCHES_PER_STEP)  /* the first 16 of the frame's 1024 patches */

static uint8_t frame[FRAME_H * FRAME_W * 3] __attribute__((aligned(64)));  /* RGB bytes, row-major (HWC) */
static uint16_t patches[PATCHES][PATCH_LEN] __attribute__((aligned(64)));  /* BF16 patch vectors */
static uint16_t pixel_to_bf16[256];

/* ---- Atlas: one step = matmul chained into GELU ---- */

#define MATMUL_BODY 130   /* team_kernel without its final "dbg0 = 1; ECALL" (3 words) */

static void load_jobs(void) {
  atlas_reset_for_load(ATLAS);
  atlas_load_program(ATLAS_IMEM, jump_to_matmul, 3);
  atlas_load_program(ATLAS_IMEM + MATMUL_WORD, team_kernel, MATMUL_BODY);
  atlas_load_program(ATLAS_IMEM + MATMUL_WORD + MATMUL_BODY, jump_to_gelu, 3);
  atlas_load_program(ATLAS_IMEM + GELU_WORD, gelu_kernel, sizeof(gelu_kernel) / 4);
  (void)ATLAS_IMEM[0];
  atlas_fence();
}

static void data_setup(void) {
  mm_setup();
  volatile uint32_t *in = (volatile uint32_t *)GELU_IN;
  volatile uint32_t *out = (volatile uint32_t *)GELU_OUT;
  for (int i = 0; i < 512; i++) {
    in[i] = gelu_in[i];
    out[i] = 0;
  }
  atlas_fence();
}

static int gelu_check(void) {
  volatile uint32_t *out = (volatile uint32_t *)GELU_OUT;
  atlas_fence();
  for (int i = 0; i < 512; i++) {
    if (out[i] != gelu_golden[i]) {
      return 0;
    }
  }
  return 1;
}

/* ---- the host's frame preparation ---- */

static uint16_t f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);   /* round to nearest even */
}

static float pixel_value(uint8_t p) {
  return (float)p / 255.0f * 2.0f - 1.0f;
}

static void frame_init(void) {
  for (int y = 0; y < FRAME_H; y++) {
    for (int x = 0; x < FRAME_W; x++) {
      for (int c = 0; c < 3; c++) {
        frame[(y * FRAME_W + x) * 3 + c] = (uint8_t)(x * 3 + y * 7 + c * 85 + ((x * y) >> 5));
      }
    }
  }
  for (int p = 0; p < 256; p++) {
    pixel_to_bf16[p] = f32_to_bf16(pixel_value((uint8_t)p));
  }
}

/* Patch p (row-major over the 32x32 patch grid) to its 768 BF16 values. Integer code only. */
static void prep_patch(int p) {
  int py = p / (FRAME_W / PATCH), px = p % (FRAME_W / PATCH);
  uint16_t *out = patches[p];
  for (int ky = 0; ky < PATCH; ky++) {
    const uint8_t *row = &frame[((py * PATCH + ky) * FRAME_W + px * PATCH) * 3];
    for (int kx = 0; kx < PATCH; kx++) {
      out[0 * 256 + ky * PATCH + kx] = pixel_to_bf16[row[kx * 3 + 0]];
      out[1 * 256 + ky * PATCH + kx] = pixel_to_bf16[row[kx * 3 + 1]];
      out[2 * 256 + ky * PATCH + kx] = pixel_to_bf16[row[kx * 3 + 2]];
    }
  }
}

static void prep_step(int k) {
  for (int i = 0; i < PATCHES_PER_STEP; i++) {
    prep_patch(k * PATCHES_PER_STEP + i);
  }
}

/* Every patch value against a direct computation (not the table). */
static int patches_ok(void) {
  for (int p = 0; p < PATCHES; p++) {
    int py = p / (FRAME_W / PATCH), px = p % (FRAME_W / PATCH);
    for (int c = 0; c < 3; c++) {
      for (int ky = 0; ky < PATCH; ky++) {
        for (int kx = 0; kx < PATCH; kx++) {
          uint8_t pix = frame[((py * PATCH + ky) * FRAME_W + px * PATCH + kx) * 3 + c];
          if (patches[p][c * 256 + ky * PATCH + kx] != f32_to_bf16(pixel_value(pix))) {
            return 0;
          }
        }
      }
    }
  }
  return 1;
}

/* ---- the scheduler in the doorbell hook ---- */

static volatile int next_step;             /* the Atlas step to start on the next doorbell */
static volatile int bad_status;            /* steps that did not end with STATUS 5 */
static uint64_t run_start;
static uint64_t step_launch[STEPS], step_doorbell[STEPS];
static uint32_t step_inst0[STEPS], step_inst1[STEPS];

static void launch(int k) {
  step_inst0[k] = ATLAS->INSTCNT;
  atlas_run_start();
  step_launch[k] = get_cycles();
}

static void on_doorbell(void *context) {
  (void)context;
  int done = next_step - 1;
  step_doorbell[done] = atlas_irq_timestamps.entry;
  step_inst1[done] = ATLAS->INSTCNT;
  uint32_t status = atlas_run_status();
  for (int spin = 0; spin < 1000 && status != ATLAS_STATUS_DONE; spin++) {
    status = atlas_run_status();          /* the ECALL follows the doorbell store by a few Atlas cycles */
  }
  if (status != ATLAS_STATUS_DONE) {
    bad_status = bad_status + 1;
    return;                               /* do not restart a busy or failed Atlas */
  }
  if (next_step < STEPS) {
    launch(next_step);
    next_step = next_step + 1;
  }
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
  uint64_t prep_start[STEPS], prep_end[STEPS];

  frame_init();
  load_jobs();
  data_setup();
  atlas_irq_init();

  /* each piece alone (polled; the doorbell's msip is cleared afterwards) */
  uint32_t i0 = ATLAS->INSTCNT;
  uint64_t t0 = get_cycles();
  atlas_run_start();
  uint32_t s1 = atlas_run_poll(NULL);
  uint64_t t_atlas = get_cycles() - t0;
  uint32_t inst = ATLAS->INSTCNT - i0;
  atlas_irq_ack();
  t0 = get_cycles();
  prep_step(0);
  uint64_t t_prep = get_cycles() - t0;
  int ok = (s1 == ATLAS_STATUS_DONE) && mm_check() && gelu_check();
  failures += !ok;
  printf("alone T_atlas_step=%lu atlas_instret=%lu T_prep_step=%lu patches_per_step=%d ok=%d\n",
         (unsigned long)t_atlas, (unsigned long)inst, (unsigned long)t_prep, PATCHES_PER_STEP, ok);

  for (int rep = 0; rep < REPEATS; rep++) {
    /* serial */
    data_setup();
    memset(patches, 0, sizeof(patches));
    int status_ok = 1;
    t0 = get_cycles();
    for (int k = 0; k < STEPS; k++) {
      atlas_run_start();
      status_ok &= (atlas_run_poll(NULL) == ATLAS_STATUS_DONE);
      atlas_irq_ack();
      prep_step(k);
    }
    uint64_t t_serial = get_cycles() - t0;
    ok = status_ok && mm_check() && gelu_check() && patches_ok();
    failures += !ok;
    printf("serial T_serial=%lu ok=%d\n", (unsigned long)t_serial, ok);

    /* pipeline */
    data_setup();
    memset(patches, 0, sizeof(patches));
    bad_status = 0;
    uint32_t msip_before = *(volatile uint32_t *)0x02000000UL;
    uint64_t count0 = atlas_irq_count();
    atlas_irq_set_handler(on_doorbell, NULL);
    watchdog_arm(WATCHDOG_TICKS);
    run_start = get_cycles();
    launch(0);
    next_step = 1;
    enable_global_interrupt();
    for (int k = 0; k < STEPS; k++) {
      prep_start[k] = get_cycles();
      prep_step(k);
      prep_end[k] = get_cycles();
    }
    uint64_t prep_done = get_cycles();
    disable_global_interrupt();
    while (atlas_irq_count() - count0 < STEPS && bad_status == 0) {
      asm volatile("wfi");               /* wakes on a pending interrupt even with interrupts off */
      enable_global_interrupt();
      disable_global_interrupt();
    }
    uint64_t end = get_cycles();
    watchdog_off();
    atlas_irq_set_handler(NULL, NULL);

    uint64_t doorbells = atlas_irq_count() - count0;
    ok = (bad_status == 0) && (doorbells == STEPS) && mm_check() && gelu_check() && patches_ok();
    failures += !ok;
    uint64_t total = end - run_start;
    printf("pipeline T_pipeline=%lu prep_done=%lu doorbells=%lu msip_before=%lu speedup_x100=%lu ok=%d\n",
           (unsigned long)total, (unsigned long)(prep_done - run_start), (unsigned long)doorbells,
           (unsigned long)msip_before, (unsigned long)(t_serial * 100 / total), ok);
    for (int k = 0; k < STEPS; k++) {
      printf("step k=%d launch=%lu doorbell=%lu atlas_instret=%lu\n", k, (unsigned long)(step_launch[k] - run_start),
             (unsigned long)(step_doorbell[k] - run_start), (unsigned long)(step_inst1[k] - step_inst0[k]));
    }
    for (int k = 0; k < STEPS; k++) {
      printf("prep k=%d start=%lu end=%lu\n", k, (unsigned long)(prep_start[k] - run_start),
             (unsigned long)(prep_end[k] - run_start));
    }
  }

  printf(failures ? "*** FAILED ***\n" : "*** PASSED ***\n");
  return failures ? 1 : 0;
}

int main(void) {
  app_init();
  return app_main();
}
