/**
 * @file atlas_run.h
 * @brief What the matmul doorbell tests share: load and start an Atlas kernel, wait for it by polling,
 *        the matmul's input/output data, the host's own job, and a watchdog.
 *
 * The launch and poll rule follows the Atlas Zephyr driver (merlin, examples/atlas/.../atlas_demo/src/direct_launch.h,
 * branch feat/atlas-zephyr-driver): read the retired-instruction counter, fence, start; a halt counts only after
 * Atlas has been seen running or the counter has moved. The kernel and data come from matmul_data.h.
 */
#ifndef __ATLAS_RUN_H
#define __ATLAS_RUN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* STATUS value of a clean finish: halted (bit 0) with reason 2 = ECALL (bits 2:1). */
#define ATLAS_STATUS_DONE 5U

/* ---- Atlas: load, start, wait ---- */

/** Stop Atlas, copy a kernel into IMEM, then read one IMEM word back and fence so the writes are done.
 *  (A full atlas_verify_program() is slower; do not put it inside a timed region.) */
void atlas_run_load(const uint32_t *words, uint32_t n_words);

/** The Atlas team's matmul, unchanged. */
void atlas_run_load_team(void);

/** The same matmul with the doorbell epilogue (it writes 1 to the host's msip before halting). */
void atlas_run_load_doorbell(void);

/** Start the loaded kernel at PC 0: clear DBG0, note the retired-instruction count, fence, EXEC_CONTROL = 1. */
void atlas_run_start(void);

/** One read of the STATUS register. */
uint32_t atlas_run_status(void);

/** Poll STATUS until Atlas has halted after running. Returns STATUS; *polls (if not NULL) gets the number of reads. */
uint32_t atlas_run_poll(uint64_t *polls);

/* ---- the matmul's data in DRAM ---- */

/** Copy the inputs (A, B) to MM_IN, clear the 4 KiB output area at MM_OUT, fence. */
void mm_setup(void);

/** Fence, then compare MM_OUT with the golden result. Returns 1 if equal, 0 if not. */
int mm_check(void);

/* ---- the host's own work ---- */

/** Step counter of host_job_tracked(): the steps finished so far (read it from an interrupt hook). */
extern volatile uint64_t job_step;

/** A fixed number of steps of xorshift64 arithmetic. Returns a checksum, so an interrupted run can be compared
 *  with an uninterrupted one. It never looks at Atlas or at any shared flag. */
uint64_t host_job(uint64_t steps);

/** The same work, plus one store of the step counter to job_step per step. */
uint64_t host_job_tracked(uint64_t steps);

/* ---- watchdog: a timer interrupt if a doorbell does not come in time ---- */

/** Arm a CLINT timer interrupt `ticks` mtime ticks from now (1 tick = 1000 core cycles). Enables the machine
 *  timer interrupt; it is taken only while mstatus.MIE is set. */
void watchdog_arm(uint64_t ticks);

/** Disarm: push the compare value to the far future and disable the machine timer interrupt. */
void watchdog_off(void);

/** Replace what happens when the watchdog fires (default: print "watchdog fired" and exit with code 3).
 *  The watchdog is already disarmed when the hook runs. NULL restores the default. */
void watchdog_set_hook(void (*hook)(void));

#ifdef __cplusplus
}
#endif

#endif /* __ATLAS_RUN_H */
