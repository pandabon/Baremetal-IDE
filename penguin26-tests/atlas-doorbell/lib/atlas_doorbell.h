/**
 * @file atlas_doorbell.h
 * @brief Interrupt-driven completion for Atlas on penguin26: the "doorbell".
 *
 * An Atlas program that ends with the doorbell epilogue writes 1 to the host's CLINT msip register
 * (CLINT_BASE + 0, hart 0) before it halts. That raises a machine software interrupt (mcause 3) on
 * the host, so the host can do other work instead of polling STATUS or DBG0.
 *
 * This layer is the only code that knows how the interrupt reaches us. Here it uses glossy's weak
 * machine_software_interrupt_callback() (glossy/src/trap/trap.c) and the CLINT helpers in
 * driver/rocket-chip/clint/clint.h. To move to another platform (atlas-mlir runtime/, Zephyr),
 * re-implement these functions; the tests only call what is declared here and in hal_atlas.h.
 */
#ifndef __ATLAS_DOORBELL_H
#define __ATLAS_DOORBELL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/** Called from the interrupt, after msip is cleared. Keep it short; it runs with interrupts off. */
typedef void (*atlas_doorbell_hook)(void *context);

/** Statistics the interrupt path updates (read them from the test, not from the hook). */
typedef struct {
  volatile uint64_t count;       /* doorbells taken since atlas_doorbell_init() */
  volatile uint64_t last_entry;  /* mcycle when the callback started (glossy's trap entry is before it) */
  volatile uint64_t last_acked;  /* mcycle right after the callback cleared msip */
  volatile uint64_t last_exit;   /* mcycle when the callback finished, hook included (glossy's restore and
                                    mret follow) */
} atlas_doorbell_stats;

extern atlas_doorbell_stats atlas_doorbell;

/** Clear a stale doorbell, reset the statistics and enable the machine software interrupt (mie.MSIE).
 *  Global interrupts (mstatus.MIE) stay as they are: the test decides when it can be interrupted. */
void atlas_doorbell_init(void);

/** Install the hook called on every doorbell (NULL: count only). */
void atlas_doorbell_set_hook(atlas_doorbell_hook hook, void *context);

/** Acknowledge: clear msip for this hart. The interrupt path does it before the hook. */
void atlas_doorbell_ack(void);

/** Ring the doorbell from the host itself (what the Atlas epilogue does), to measure the interrupt by itself. */
void atlas_doorbell_ring_self(void);

/** Disable the machine software interrupt again (mie.MSIE = 0). */
void atlas_doorbell_disable(void);

#ifdef __cplusplus
}
#endif

#endif /* __ATLAS_DOORBELL_H */
