# Supply this as CMAKE_PROJECT_ninfer_INCLUDE to link the same probe against either checkout.
# Explicitly enable W4A8 only against an engine exposing the corresponding Engine option.
function(ninfer_add_v100_quality_probe)
  if(NOT NINFER_ARCH_VOLTA OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The frozen V100 quality probe needs a Linux build with Volta")
  endif()
  add_executable(ninfer-v100-quality-probe
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../ninfer_precision_probe.cpp")
  ninfer_internal_includes(ninfer-v100-quality-probe)
  target_link_libraries(ninfer-v100-quality-probe PRIVATE
    ninfer_engine ninfer_product_logging ninfer::json)
  target_compile_definitions(ninfer-v100-quality-probe PRIVATE NINFER_V100_FROZEN_QUALITY=1)
  if(NINFER_V100_QUALITY_W4A8)
    target_compile_definitions(ninfer-v100-quality-probe PRIVATE NINFER_V100_QUALITY_W4A8=1)
  endif()
endfunction()

cmake_language(DEFER CALL ninfer_add_v100_quality_probe)
