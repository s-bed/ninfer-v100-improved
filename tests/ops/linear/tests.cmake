add_library(ninfer_linear_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_test_common.cpp")
ninfer_test_includes(ninfer_linear_test_support)
ninfer_op_oracle_options(ninfer_linear_test_support)
target_link_libraries(ninfer_linear_test_support PUBLIC ninfer_ops ninfer_artifact)

ninfer_add_op_test(ninfer_linear_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q5_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q6_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q6_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

# A tensor-parallel share's projections: shapes outside the tuned tables, BF16 or FP32 out.
ninfer_add_op_test(ninfer_linear_rowsplit_runtime_shape_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rowsplit_runtime_shape.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_t2_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_t2_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_t2_a8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_t2_a8.cpp"
  LIBRARIES ninfer_linear_test_support)

# A wide projection first called inside a CUDA graph capture (sm_70: the cuBLAS route's handle).
ninfer_add_op_test(ninfer_linear_capture_first_use_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_capture_first_use.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_nvfp4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_nvfp4_a4_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a4.cpp"
  LIBRARIES ninfer_linear_test_support)

# Back-to-back TMA staging race probe for both the plain Linear and fused LinearSwiGLU routes:
# distinct weights fired without inter-launch sync, each checked against an isolated reference.
ninfer_add_op_test(ninfer_nvfp4_tma_staging_race_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_tma_staging_race.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_fp8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_fp8_a8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a8.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_bf16_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bf16_a16.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_gguf_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gguf.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_gguf_mmq_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gguf_mmq_store.cu"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_gguf_mmq_limits_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gguf_mmq_limits.cpp")

ninfer_add_op_test(ninfer_linear_gguf_cooperative_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gguf_cooperative.cu"
  LIBRARIES ninfer_ops)

target_include_directories(ninfer_linear_gguf_cooperative_test PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/ggml-quants/include"
  "${PROJECT_SOURCE_DIR}/third_party/ggml-quants/src")

# STQ products (the sushi DSpark drafter's matrices) against an FP64 oracle over an independent
# decode of the trellis.
ninfer_add_op_test(ninfer_linear_stq_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_stq.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_v100_port_test STANDALONE
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_v100_port.cpp"
  LIBRARIES ninfer_linear_test_support ninfer_linear_swiglu_test_support)

# Qualify packed-layout native fallbacks using the same independent FP64 oracles.
add_test(NAME ninfer_v100_native_packed_test
  COMMAND ninfer_v100_port_test --native-packed)
set_tests_properties(ninfer_v100_native_packed_test PROPERTIES SKIP_RETURN_CODE 77)

# Opt-in event timings: invoke a separate process for each fixed kernel policy.
ninfer_add_op_test(ninfer_v100_port_bench STANDALONE
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/v100_port_bench.cpp"
  LIBRARIES ninfer_ops)

# Qualified native RowScale FP8 donor routes; oracle and events remain separate.
ninfer_add_op_test(ninfer_v100_fp8_test STANDALONE
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_v100_fp8.cpp"
  LIBRARIES ninfer_linear_test_support)
ninfer_add_op_test(ninfer_v100_fp8_bench STANDALONE
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/v100_fp8_bench.cpp"
  LIBRARIES ninfer_ops)

# Private FP8 byte-layout experiment. It directly instantiates the donor leaf;
# it links no production Ops, materializer, artifact reader or layout validator.
if(NINFER_ARCH_VOLTA)
  ninfer_add_op_test(ninfer_v100_fp8_prepack_experiment STANDALONE
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/v100_fp8_prepack_experiment.cu"
    LIBRARIES ${NINFER_CUDART_TARGET})
  set_target_properties(ninfer_v100_fp8_prepack_experiment PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  ninfer_cuda_archive(ninfer_v100_fp8_prepack_experiment)
  target_compile_options(ninfer_v100_fp8_prepack_experiment PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ninfer_v100_fp8_prepack_experiment PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-fast-math>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
  endif()
  set_tests_properties(ninfer_v100_fp8_prepack_experiment PROPERTIES TIMEOUT 900)
endif()
