########################################################################################################################
# Atlas doorbell tests for the penguin26 chip: Atlas signals completion with an interrupt instead of being polled.
#
#   lib/            atlas_doorbell.{h,c}: the interrupt side (glossy's machine_software_interrupt_callback + CLINT msip).
#                   The only file that knows how the interrupt reaches the host; port it to move the tests elsewhere.
#                   fault_exit.c: an unexpected exception prints its cause and exits 9 instead of hanging.
#   common/         what the matmul tests share: the Atlas team's matmul kernel and data (matmul_data.h), launch and
#                   poll helpers, the host's own job, a watchdog, and the same matmul on Saturn (fp8_matmul.S).
#   interrupt_cost/ atlas_doorbell_interrupt_cost: cost of one doorbell interrupt through glossy, and which clocks
#                   run while the host sleeps in wfi.
#   poll_baseline/  atlas_doorbell_poll_baseline: the matmul with the host polling STATUS, then the host job (serial).
#   overlap/        atlas_doorbell_overlap: the matmul and the host job at the same time; the doorbell only acknowledges.
#   check_in_handler/ atlas_doorbell_check_in_handler: the same, but the doorbell handler checks the result.
#   edge_cases/     atlas_doorbell_edge_cases: no doorbell, an Atlas error, a re-ring, a handler that starts the next job.
#
#   make build CHIP=penguin26 PLATFORM=SIMS TARGET=atlas_doorbell_interrupt_cost
########################################################################################################################

file(GLOB _common_srcs CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/common/*.c ${CMAKE_CURRENT_SOURCE_DIR}/common/*.S)
add_library(atlas-doorbell STATIC lib/atlas_doorbell.c ${_common_srcs})
target_include_directories(atlas-doorbell PUBLIC lib common)
target_link_libraries(atlas-doorbell PUBLIC chip-config rocketcore)

# One executable per test, linked the way penguin26-tests/atlas_example is.
# Each test is every .c and .S file in its folder.
function(atlas_doorbell_test name dir)
  file(GLOB _srcs CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.c ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.S)
  add_executable(${name} ${_srcs} ${CMAKE_CURRENT_SOURCE_DIR}/lib/fault_exit.c)
  target_link_libraries(${name} PRIVATE
    -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  )
  target_link_libraries(${name} PRIVATE atlas-doorbell chip-config rocketcore)
  if (PROF_COV)
    target_link_libraries(${name} PRIVATE gcov)
  endif()
endfunction()

atlas_doorbell_test(atlas_doorbell_interrupt_cost interrupt_cost)
foreach(_t poll_baseline overlap check_in_handler edge_cases)
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_t})
    atlas_doorbell_test(atlas_doorbell_${_t} ${_t})
  endif()
endforeach()
