"""CPU-only build-boundary checks: CMake selection, C++ stubs and CUDA pass visibility.

No CUDA compiler, GPU, object files, linking or inference are used. The host compiler
only preprocesses CU text or performs syntax checking of the native C++ fallback.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def run(command: list[str]) -> str:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"{command!r}\n{result.stderr[-16000:]}")
    return result.stdout


def source_selection(source: Path, cmake: str) -> None:
    manifest = source / "src/ops/linear/sm70/sources.cmake"
    for volta in (False, True):
        script = f'''
set(NINFER_ARCH_VOLTA {"ON" if volta else "OFF"})
function(target_sources target)
  foreach(item IN LISTS ARGN)
    if(item MATCHES "\\\\.(cu|cpp)$")
      set_property(GLOBAL APPEND PROPERTY collected "${{item}}")
    endif()
  endforeach()
endfunction()
function(set_source_files_properties)
endfunction()
include("{manifest.as_posix()}")
get_property(collected GLOBAL PROPERTY collected)
foreach(item IN LISTS collected)
  message("selected=${{item}}")
endforeach()
'''
        with tempfile.TemporaryDirectory(prefix="ninfer-arch-guard-") as temporary:
            path = Path(temporary) / "select.cmake"
            path.write_text(script)
            result = subprocess.run([cmake, "-P", str(path)], text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode:
                raise RuntimeError(result.stderr)
            names = {Path(line.removeprefix("selected=")).name
                     for line in (result.stdout + result.stderr).splitlines()
                     if line.startswith("selected=")}
        assert {"sm70_project.cu", "sm70_small.cu", "sm70_gemm.cu"} <= names, names
        assert ("v100_native_stubs.cpp" in names) is not volta, names
        assert ("v100_qpn.cu" in names) is volta, names
        assert ("v100_cutlass.cu" in names) is volta, names
        assert ("v100_fp8.cu" in names) is volta, names
        assert ("v100_fp8_cutlass.cu" in names) is volta, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--cuda-include", type=Path, required=True)
    parser.add_argument("--cutlass", default="")
    parser.add_argument("--cmake", default="cmake")
    args = parser.parse_args()
    source = args.source.resolve()
    base = [args.cxx, "-std=c++20", "-x", "c++", "-I", str(source / "src"),
            "-I", str(source / "include"), "-I", str(args.cuda_include)]
    source_selection(source, args.cmake)
    stub = source / "src/ops/linear/sm70/v100_native_stubs.cpp"
    # This is the actual native fallback translation unit, not a substitute fixture.
    run(base + ["-DNINFER_TIER_AMPERE=1", "-fsyntax-only", str(stub)])
    for name, markers in (
        ("v100_qpn.cu", (r"bool\s+v100_qpn_project\s*\([^;{}]*\)\s*\{",)),
        ("v100_cutlass.cu", (r"V100 CUTLASS GEMM failed",)),
        ("v100_fp8.cu", (r"bool\s+v100_fp8_project\s*\([^;{}]*\)\s*\{",
                         r"void\s+fp8_volta_qpn_gemm_kernel\s*\(")),
        ("v100_fp8_cutlass.cu", (r"V100 CUTLASS cannot implement qualified FP8 shape",)),
    ):
        path = source / "src/ops/linear/sm70" / name
        # A 70;75 build uses one host launch implementation and separate device passes.
        # The unsupported pass must not see definitions referring to sm_70-only leaves.
        unsupported = run(base + ["-DNINFER_TIER_VOLTA=1", "-D__CUDA_ARCH__=750",
                                  "-E", "-P", str(path)])
        for marker in markers:
            assert re.search(marker, unsupported) is None, (name, marker)
        cutlass = Path(args.cutlass) if args.cutlass else None
        if cutlass and (cutlass / "include/cutlass/gemm/device/gemm.h").exists():
            for architecture in (None, 700):
                defines = ["-DNINFER_TIER_VOLTA=1"]
                if architecture is not None:
                    defines.append(f"-D__CUDA_ARCH__={architecture}")
                supported = run(base + ["-I", str(cutlass / "include")] + defines +
                                ["-E", "-P", str(path)])
                for marker in markers:
                    assert re.search(marker, supported) is not None, (name, architecture, marker)
    if not args.cutlass:
        print("NOTE: host/sm_70 expansion check skipped without a cached CUTLASS path")
    print("OK: native CMake/stub path and mixed-device pass guards")


if __name__ == "__main__":
    main()
