target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/sm70_dequantize.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sm70_gemm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sm70_integer.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sm70_project.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sm70_small.cu"
)

# Native builds of the other tiers retain their existing routes and need no CUTLASS tree.
# The dedicated Volta objects carry their own sm_70 kernel registrations in mixed builds.
if(NINFER_ARCH_VOLTA)
  target_sources(ninfer_volta_kernels PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/v100_qpn.cu"
    "${CMAKE_CURRENT_LIST_DIR}/v100_w4a8.cu"
    "${CMAKE_CURRENT_LIST_DIR}/v100_cutlass.cu"
    "${CMAKE_CURRENT_LIST_DIR}/v100_drafter_gemv.cu"
    "${CMAKE_CURRENT_LIST_DIR}/v100_fp8.cu"
    "${CMAKE_CURRENT_LIST_DIR}/v100_fp8_cutlass.cu")
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/v100_cutlass.cu"
    "${CMAKE_CURRENT_LIST_DIR}/v100_fp8_cutlass.cu"
    PROPERTIES COMPILE_OPTIONS "--expt-relaxed-constexpr")
else()
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/v100_native_stubs.cpp")
endif()
