# llama.cpp Volta MMA flash-attention (vendored)

Vendored from MIT-licensed [llama.cpp](https://github.com/ggml-org/llama.cpp) at
[`62bf73d25`](https://github.com/ggml-org/llama.cpp/commit/62bf73d25c53b8161f8a22894d4f90c4aebbd7d0).
This snapshot follows the extraction from
[`JimmyMax/ninfer-v100@ec1d53a9`](https://github.com/JimmyMax/ninfer-v100/commit/ec1d53a9dabecf7d1f76e9381896391d82eca7fa).
The MIT notice is in `LICENSE`.

The Volta-only causal attention bridge uses these files for BF16 KV prefill.

## Contents

| file | origin | change |
|---|---|---|
| `mma.cuh` | `ggml/src/ggml-cuda/mma.cuh` | none (byte-for-byte) |
| `cp-async.cuh` | `ggml/src/ggml-cuda/cp-async.cuh` | none (byte-for-byte) |
| `fattn-mma-f16.cuh` | `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | truncated + one config case, see below |
| `fattn-stream-k.cuh` | `ggml/src/ggml-cuda/fattn-common.cuh` lines 721-971 | extracted verbatim |
| `common.cuh` | shim, not upstream | see below |

## Deviations from upstream

**`common.cuh` is ours.** Upstream's is 1,669 lines and pulls in `ggml.h`,
`ggml-impl.h`, `ggml-cuda.h` and `ggml-common.h`. The device side of the kernel
needs about a dozen symbols from it, none involving a ggml type. Providing them
under the upstream file name is what lets `mma.cuh`, `cp-async.cuh` and the
kernel body stay byte-for-byte. Definitions inside are copied verbatim from
upstream except where a comment marks otherwise.

**`fattn-mma-f16.cuh` is truncated at line 1893**, dropping the host-side
`ggml_cuda_flash_attn_ext_mma_f16_case` launcher and its `DECL_` instantiation
macros, which take `ggml_tensor` / `ggml_backend_cuda_context`. ninfer drives
the kernel from its own launcher. Its `#include "fattn-common.cuh"` becomes
`#include "fattn-stream-k.cuh"`; nothing in the device kernel referenced that
header. `fattn-common.cuh` itself is not vendored, so `vecdotq.cuh` and
`convert.cuh` (quantized-KV only) drop out of the closure entirely.

**One added config case.** `ggml_cuda_fattn_mma_get_config_volta` has no
256/256 entry upstream and falls through to the Ampere table's
`nbatch_combine = 128`. On Volta `get_cols_per_warp()` is 32 (16 on Turing+),
making the combine buffer 66 KB. The donor's 256/256 configuration uses the
64-value setting found in the Volta table to reduce shared memory.

## Updating

Re-vendor from a single upstream commit and re-run the correctness and throughput harness before
trusting the result. Keep the pin in this file in sync.
