/**
 * @file atlas_watchdog.c
 * @brief A CLINT timer interrupt that catches a doorbell that never comes. See atlas_run.h.
 *
 * This is a separate file so that a test which needs a different reaction can call watchdog_set_hook()
 * instead of redefining machine_timer_interrupt_callback() (a second definition would not link).
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "atlas_run.h"
#include "chip_config.h"
#include "rocketcore.h"

static void (*watchdog_hook)(void);

void watchdog_arm(uint64_t ticks) {
  uint64_t now = clint_get_time(CLINT);
  clint_set_timer_interrupt_target(CLINT, (uint32_t)get_hart_id(), now + ticks);
  enable_timer_interrupt();
}

void watchdog_off(void) {
  clint_set_timer_interrupt_target(CLINT, (uint32_t)get_hart_id(), UINT64_MAX);
  disable_timer_interrupt();
}

void watchdog_set_hook(void (*hook)(void)) {
  watchdog_hook = hook;
}

/* Overrides the weak default in glossy/src/trap/trap.c (mcause 7). Disarm first, or the timer stays pending
 * and the same interrupt is taken again after mret. */
void machine_timer_interrupt_callback(void) {
  watchdog_off();
  if (watchdog_hook) {
    watchdog_hook();
    return;
  }
  printf("watchdog fired\n");
  exit(3);
}
