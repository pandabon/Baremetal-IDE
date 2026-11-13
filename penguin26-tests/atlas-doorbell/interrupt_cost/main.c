/*
 * main.c - atlas_doorbell_interrupt_cost: what does taking the doorbell interrupt cost through glossy, and
 * which clock runs while the host sleeps in wfi?
 *
 * The interrupt goes through glossy's trap_vector and trap_handler, so the numbers include glossy's
 * 32-register save and dispatch.
 * Start-up and UART set-up follow penguin26-tests/atlas_example/src/main.c.
 *
 *   (a) the cost of reading mcycle twice (the resolution of every timestamp below)
 *   (b) the host rings its own msip 10 times. Per ring:
 *         base_cold      the delay loop alone, right after the previous line's printf (code may be cold)
 *         base           the delay loop alone again, now warm
 *         with_irq       the delay loop with the ring at its start; the interrupt lands inside it
 *         irq_cost       with_irq - base: the whole interrupt (trap entry, glossy's save/restore,
 *                        callback, mret), with warm code on both sides
 *         ring_to_entry  cycles from the msip store to the start of the doorbell callback
 *         round_trip     a second method: ring, then spin until the doorbell count changes (a cached
 *                        load); includes the ring store, the interrupt and one more load
 *         irqs           doorbells counted in that round (must be 1)
 *       All three loops call the same non-inlined delay_loop(), so they are the same instructions at the
 *       same address. (A first version inlined two copies and timed the plain one right after printf; it
 *       reported the ring loop as ~2,270 cycles faster than the plain one, which this split explains or not.)
 *   (c) 3 sleeps in wfi, each woken by a CLINT timer interrupt 50 mtime ticks later: how far
 *       mcycle, mtime and the Atlas cycle counter (CSR 0x00) advanced.
 *
 * Atlas itself does not run here. Output: one "read_cost", ten "ring" and three "sleep" lines of key=value.
 * read_cost is taken twice; the second is warm.
 * Pass (return 0): every round of (b) took exactly two doorbells (one per method). glossy calls the doorbell callback only for
 * mcause 3 (machine software interrupt), so a counted doorbell is also the right cause.
 */
#include <stdio.h>
#include <stdint.h>

#include "chip_config.h"
#include "rocketcore.h"
#include "riscv_encoding.h"
#include "atlas_doorbell.h"

#define RINGS        10
#define DELAY_LOOPS  1000   /* long enough for the interrupt to arrive inside the loop */
#define SLEEPS       3
#define SLEEP_TICKS  50     /* mtime ticks per sleep */

/* A delay whose length does not depend on the compiler: DELAY_LOOPS iterations of addi + bnez. Not inlined,
 * so every measurement runs the very same instructions. */
static __attribute__((noinline)) void delay_loop(void) {
  uint64_t n = DELAY_LOOPS;
  asm volatile("1: addi %0, %0, -1\n\tbnez %0, 1b" : "+r"(n));
}

/* Time the delay loop alone; atlas_doorbell_ack() is the same kind of CLINT access as the ring. */
static uint64_t time_base(void) {
  uint64_t t0 = get_cycles();
  atlas_doorbell_ack();
  delay_loop();
  return get_cycles() - t0;
}

static volatile uint64_t timer_hits;

/* Overrides the weak default in glossy/src/trap/trap.c (mcause 7). One shot: push the compare
 * value to the far future and stop the timer interrupt, so it does not fire again. */
void machine_timer_interrupt_callback(void) {
  clint_set_timer_interrupt_target(CLINT, (uint32_t)get_hart_id(), UINT64_MAX);
  disable_timer_interrupt();
  timer_hits = timer_hits + 1;
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
  int fail = 0;

  /* (a) */
  uint64_t c0 = get_cycles();
  uint64_t c1 = get_cycles();
  uint64_t c2 = get_cycles();
  uint64_t c3 = get_cycles();
  printf("read_cost mcycle_pair=%lu warm=%lu\n", (unsigned long)(c1 - c0), (unsigned long)(c3 - c2));

  /* (b) */
  atlas_doorbell_init();
  for (int i = 0; i < RINGS; i++) {
    uint64_t base_cold = time_base();
    uint64_t base = time_base();

    uint64_t before = atlas_doorbell.count;
    enable_global_interrupt();
    uint64_t t0 = get_cycles();
    atlas_doorbell_ring_self();
    delay_loop();
    uint64_t with_irq = get_cycles() - t0;
    uint64_t ring_to_entry = atlas_doorbell.last_entry - t0;

    /* round trip: ring, then wait for the callback to count it */
    uint64_t mid = atlas_doorbell.count;
    uint64_t r0 = get_cycles();
    atlas_doorbell_ring_self();
    while (atlas_doorbell.count == mid) {
    }
    uint64_t round_trip = get_cycles() - r0;
    disable_global_interrupt();

    uint64_t irqs = atlas_doorbell.count - before;
    if (irqs != 2) {
      fail = 1;
    }
    printf("ring ring_to_entry=%lu base_cold=%lu base=%lu with_irq=%lu irq_cost=%ld round_trip=%lu irqs=%lu\n",
           (unsigned long)ring_to_entry, (unsigned long)base_cold, (unsigned long)base, (unsigned long)with_irq,
           (long)(with_irq - base), (unsigned long)round_trip, (unsigned long)irqs);
  }
  atlas_doorbell_disable();

  /* (c) */
  uint32_t hart = (uint32_t)get_hart_id();
  for (int i = 0; i < SLEEPS; i++) {
    ATLAS->CYCLE = 0;
    uint64_t hits = timer_hits;
    uint64_t m0 = get_cycles();
    uint64_t t0 = clint_get_time(CLINT);
    clint_set_timer_interrupt_target(CLINT, hart, t0 + SLEEP_TICKS);
    enable_timer_interrupt();
    while (timer_hits == hits) {
      asm volatile("wfi");          /* wakes when the timer is pending, even with mstatus.MIE = 0 */
      enable_global_interrupt();    /* take it (machine_timer_interrupt_callback above) */
      disable_global_interrupt();
    }
    uint64_t m1 = get_cycles();
    uint64_t t1 = clint_get_time(CLINT);
    printf("sleep mcycle_d=%lu mtime_d=%lu atlas_cycle_d=%lu\n", (unsigned long)(m1 - m0),
           (unsigned long)(t1 - t0), (unsigned long)ATLAS->CYCLE);
  }

  printf(fail ? "*** FAILED ***\n" : "*** PASSED ***\n");
  return fail;
}

int main(void) {
  app_init();
  return app_main();
}
