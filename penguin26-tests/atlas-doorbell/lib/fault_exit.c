/**
 * @file fault_exit.c
 * @brief Make an unexpected exception end the test instead of hanging it.
 *
 * glossy's default exception handlers (glossy/src/trap/trap.c) spin in `while (1) {}`. In an RTL
 * simulation a test that faults would then run until the simulator's own timeout, with no output.
 * These strong definitions replace the weak defaults: print the cause and where it happened, then
 * exit with code 9, which the simulator reports as a failure.
 *
 * This file is linked as an object into every doorbell test (see CMakeLists.txt), not through the
 * static library: the linker pulls a library member only to resolve an undefined symbol, and these
 * symbols are already defined (weakly) by glossy, so a library copy would never be used.
 */
#include <stdio.h>
#include <stdlib.h>

#include "riscv.h"

static void fault_exit(const char *what) {
  printf("fault %s mcause=%lu mepc=0x%lx mtval=0x%lx\n", what, READ_CSR("mcause"), READ_CSR("mepc"),
         READ_CSR("mtval"));
  exit(9);
}

void instruction_address_misaligned_exception_handler(void) { fault_exit("instruction_misaligned"); }
void instruction_access_fault_exception_handler(void)       { fault_exit("instruction_access"); }
void illegal_instruction_exception_handler(void)            { fault_exit("illegal_instruction"); }
void breakpoint_exception_handler(void)                     { fault_exit("breakpoint"); }
void load_address_misaligned_exception_handler(void)        { fault_exit("load_misaligned"); }
void load_access_fault_exception_handler(void)              { fault_exit("load_access"); }
void store_amo_address_misaligned_exception_handler(void)   { fault_exit("store_misaligned"); }
void store_amo_access_fault_exception_handler(void)         { fault_exit("store_access"); }
void environment_call_m_mode_exception_handler(void)        { fault_exit("ecall"); }
