/**
 * @file atlas_doorbell.c
 * @brief glossy + CLINT implementation of atlas_doorbell.h.
 */
#include "atlas_doorbell.h"

#include "chip_config.h"
#include "rocketcore.h"
#include "riscv_encoding.h"  /* IRQ_M_SOFT */

atlas_doorbell_stats atlas_doorbell;

static atlas_doorbell_hook doorbell_hook;
static void *doorbell_context;

void atlas_doorbell_ack(void) {
  clint_clear_software_interrupt(CLINT, (uint32_t)get_hart_id());
}

void atlas_doorbell_ring_self(void) {
  clint_trigger_software_interrupt(CLINT, (uint32_t)get_hart_id());
}

void atlas_doorbell_init(void) {
  atlas_doorbell_ack();
  atlas_doorbell.count = 0;
  atlas_doorbell.last_entry = 0;
  enable_irq(IRQ_M_SOFT);
}

void atlas_doorbell_disable(void) {
  disable_irq(IRQ_M_SOFT);
}

void atlas_doorbell_set_hook(atlas_doorbell_hook hook, void *context) {
  doorbell_context = context;
  doorbell_hook = hook;
}

/* Overrides the weak default in glossy/src/trap/trap.c; trap_handler() calls it for mcause 3.
 * msip is level-sensitive: clear it first, or mret takes the same interrupt again. */
void machine_software_interrupt_callback(void) {
  atlas_doorbell.last_entry = get_cycles();
  atlas_doorbell_ack();
  atlas_doorbell.last_acked = get_cycles();
  atlas_doorbell.count = atlas_doorbell.count + 1;
  if (doorbell_hook) {
    doorbell_hook(doorbell_context);
  }
  atlas_doorbell.last_exit = get_cycles();
}
