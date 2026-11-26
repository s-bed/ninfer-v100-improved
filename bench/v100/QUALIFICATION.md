# Tesla V100 qualification

Qualify the candidate against the ported server from main at 42d8d03f. Keep native as the
production default while the INT8 and BF16 gates are open. Record the candidate
commit or tree, the build flags, and the exact V100 used. Use the same physical GPU, driver,
power limit, application clocks, ECC, CUDA toolkit, and artifact for every pair. Follow the
GPU host's operating procedure for securing an idle card. Use separate build trees, log
directories, and test persistence files.

## Start on the test host

Use Linux, CUDA **12.9**, a supported C++ compiler, Ninja, CMake, Python 3.11, the project’s
FFmpeg/curl development packages and CUTLASS **4.4.2** (pre-cached or staged with
`python3.11 .github/scripts/install-cutlass.py`). Reserve one idle V100. Keep the
baseline checkout clean at `42d8d03ffe7ac9e9c5cd1cb227a195c099a03f88`.
From a local clone of `s-bed/ninfer-v100-improved`, fetch the published candidate and create
two detached worktrees. Record the candidate SHA before starting the campaign.

~~~bash
git fetch origin refs/heads/codex/v100-decode-prefill:refs/remotes/origin/codex/v100-decode-prefill
export BASELINE="$PWD/build/v100-baseline" CANDIDATE="$PWD/build/v100-candidate"
git worktree add --detach "$BASELINE" 42d8d03ffe7ac9e9c5cd1cb227a195c099a03f88
git worktree add --detach "$CANDIDATE" refs/remotes/origin/codex/v100-decode-prefill
git -C "$CANDIDATE" rev-parse HEAD
~~~

Point `ARTIFACT` at the exact Qwen3.8-27B NVFP4 v3 artifact with the MTP component.
`FIXTURES` is the frozen precision campaign root described below. Place the archived warm
and pressure request manifest at `$FIXTURES/cache.json` and the per-KV admission and cache
profiles at `$FIXTURES/deployment.json`. Missing private inputs leave their gates open.
Choose a fresh results root for each candidate revision. The wrapper keeps
separate build directories, logs, JUnit records and numbered attempts.

~~~bash
export CUDA=/usr/local/cuda-12.9 CUTLASS=/path/to/cutlass-4.4.2
export ARTIFACT=/path/to/explicit-model.ninfer FIXTURES=/path/to/frozen-campaign
export RESULTS=/path/to/results/v100-decode-prefill
export PATH="$CUDA/bin:$PATH"
export BASELINE_BUILD="$RESULTS/build/baseline-70" CANDIDATE_BUILD="$RESULTS/build/candidate-70"
run_v100() {
  local stage="$1"
  shift
  python3.11 "$CANDIDATE/bench/v100/handoff.py" "$stage" \
    --baseline-source "$BASELINE" --candidate-source "$CANDIDATE" \
    --artifact "$ARTIFACT" --fixture-root "$FIXTURES" --results "$RESULTS" \
    --cuda "$CUDA" --cutlass "$CUTLASS" "$@"
}
run_v100 preflight
run_v100 build
run_v100 cpu
run_v100 correctness
run_v100 gdn
run_v100 quality
run_v100 performance
run_v100 report
~~~

`build` builds the complete candidate sm_70 Release tree and all baseline apps, tests and
benchmarks plus the quality probe. It omits the unrelated broken standalone multi-GPU probe
from the baseline build. The two timed sm_70 builds use the same Release defaults.
The runner then links the candidate server and engine for mixed `70;75` with speed-oriented
fatbinary compression. It checks that the Volta launch files emit sm_70 machine code and
that the shared dispatcher emits both sm_70 and sm_75. It checks the non-Volta `75`
compilation database for excluded Volta sources and compiles the native stubs. After a memory-pressure failure, rerun
`run_v100 build --jobs 4`; Ninja resumes the same build directories. `correctness`
includes Compute Sanitizer on the 200,000-key Flash Attention boundary. `gdn` times
16/32 and 16/48 across prompt widths from 15 to 2,048 under the profiled, recurrent,
forced two-stage and forced legacy routes. Select production GDN cutovers only where those
measurements and numerical evidence justify them. `quality` validates the frozen manifests
and runs both scoring and task grading; `performance` reads its paired observations.
Inspect failed and skipped rows before continuing dependent stages. Independent evidence
can still be collected after an open fixture gate. `report.json` lists every stage attempt,
unrun stages and the remaining release review. `workloads.csv` gives one row per measured request and configuration plus separate rows for failed, inconclusive and skipped gates. Keep each attempt directory with the report.

## Implementation-host evidence

The complete Linux sm_70 Release build linked its normal apps, tests, benchmarks and frozen
quality probe under CUDA 12.9.86, CUTLASS 4.4.2 and GCC 13.4. Thirteen focused CPU CTests
passed, covering admission of the V100 policy, cache identities, architecture guards, CLI
and Serve options, request logging, cache configuration/index and the native/ported loading
rules. The Python 3.11.9 handoff suite passed 25 tests. The Linux package suite ran 13
tests with two dependency-dependent skips. The mixed `70;75` build linked the production
server and its `--help` exposed both V100 options. On a configured sm_86 Release tree the handoff
source-exclusion check passed and the real native stub compiled.

The Ubuntu 26.04 WSL host needed six math exception declarations aligned in its isolated
CUDA 12.9 header copy, and its current system libstdc++ for FFmpeg's SRT dependency at
link time. A 24-job build exhausted host memory in existing Q8 device assembly; bounded
parallelism completed it. At eight jobs, the mixed build also exhausted memory assembling
the existing causal prompt and small-T attention kernels. Four jobs completed those objects.
The mixed link exposed missing host launch stubs in Flash staging, W4A8 quantization,
and the inherited QPN, CUTLASS and FP8 helpers. Exposing the helper signatures alone
allowed the server to link while leaving templated kernels without host registrations.
The eight Volta-only CUDA launch files now compile for `70-real` in a dedicated object
target inside the regular Ops archive. Inspection of the eight mixed-build objects found
all 118 host registrations from the sm_70-only build. The shared dispatcher contains both
sm_70 and sm_75 device images. The complete sm_70 build also exposed a missing source in
the multi-GPU probe target. That probe now links its existing peer-preflight implementation
and requires a CUDA driver to run.

All V100 numerical, inference, capacity, cache, quality and performance results are pending.
The two real DFlash2 regression tests also require the separate compatible artifact
documented below. No route has been promoted on the strength of a host build.

## Differences from the comparison fork

Rodrigo’s original Flash implementation branch is `v100-flash` at `f81ddc87`. Its deployed
measurements were recorded on the default branch `v100-qpn` at `52bc516a`. Their same-binary BF16 Flash comparison on
a 27B workload reported 674 → 709 tok/s at 11.6K and 703 → 817 tok/s at 33K; decode step
latency stayed near 44 ms. The 11K greedy continuations differed at the last word. These
are donor measurements. Their earlier three-prompt comparison reported 50.6/69.4/57.1 tok/s
for their `cd602003` and 47.5/62.9/52.9 tok/s for our ported PR revision `baa2c757`.
The former used BF16 KV and W4A8, and the latter used INT8 KV. Our refreshed baseline is
`42d8d03f`. The paired runs below resolve the changed arithmetic and deployment settings.
[Donor measurements](https://github.com/RodrigoR78/ninfer-all-v100/blob/52bc516a/v100-port/HANDOFF.md)

The default branch at `75bd78a6` reported that a newer integration branch, `v100-25.4`, reproduced the deployment's
timings and fixed a DFlash2/DSpine top-k consumer that required the original head layout.
Our pinned tree contains DFlash2. Its fused top-k requires the original Q4 or FP8 layout;
the loader excludes a physical head that it can consume. DSpine belongs to the newer donor
tree. The earlier MTP attribution was corrected in that report. These findings do not change
the paired baseline. [Donor update](https://github.com/RodrigoR78/ninfer-all-v100/blob/75bd78a6/v100-port/HANDOFF.md)

The subsequent deployment report at `ca53b46c` identifies `v100-25.4` at `0923957a` as
the deployed binary, with BF16 KV in production and a separate INT8 KV trial. Both forks
support INT8 KV. [Donor deployment and INT8 trial](https://github.com/RodrigoR78/ninfer-all-v100/blob/ca53b46c/v100-port/HANDOFF.md#update-2026-11-26-1710-utc-v100-254-deployed--int8-kv-trial-entry)

| Area | Our starting point at 42d8d03f | What this candidate tests |
|---|---|---|
| Row-scaled FP8 | Packed eligible body projections already stage FP16 and repair out-of-range BF16 columns. | Extend to the 248,320 × 5,120 head, vet physical-weight consumers and bound packing scratch. |
| MTP | Existing Q4/Q8 projections and optimized proposal head. | Streaming one to four token columns, including compatible panel consumers, with the previous five-column route. The Op selects by format, layout and width, so another Q4/Q8 projection in an explicitly ported model can also use it. |
| GDN prefill | The device profile already chooses two-stage or chunked execution. | Compare the donor recurrent choice at each measured geometry and width with both existing routes. |
| Causal attention | Paged Volta attention serves BF16 and INT8. | Correct the donor BF16 V conversion and 256-key scratch rounding, clamp per-1024-query-block causal key extents, and test single-sequence BF16 dense prefill with sequential visibility. INT8 keeps its profiled attention route. |
| NVFP4 | Our ported A16 handles both packed and stored weights and has its own wide crossover. | Explicit W4A8 uses per-token group-32 activation scales for eligible narrow (`T <= 4`, decode and MTP3 verify) projections and the single-pass Volta FP16 QPN/CUTLASS/cuBLAS routes at `T >= 5`. Measure quality independently in each KV format. |

At the 200,000-key boundary, the Flash launcher rounds to 200,192 keys. The calculated K/V,
mask and Q/output buffers total 1,221 MiB at 24/4 heads and 814 MiB at 16/2 heads, plus
stream partials and alignment. The Program reuses its workspace across Ops. Measure the
actual change in peak memory and automatic KV capacity on the card; the workspace sum alone
cannot establish that change. The FP8 head uses at most 32 MiB of extra device packing
scratch during load or reload.

## Op and model checks on the V100

Run the focused suites on the baseline and candidate and keep all failures. For a failing case
that reproduces on both, record its matching failure and establish whether it leaves the new
route uncovered. A route that skipped or never dispatched remains open.

~~~bash
NINFER_OP_REPORT_STATS=1 ctest --test-dir "$CANDIDATE_BUILD" --output-on-failure \
  -R '^ninfer_(v100_(port|native_packed|w4a8|fp8.*|drafter.*|gdn.*)|gated_delta_net.*|softmax_attention.*|linear_(q4|q8)_(a16|a8int)|linear_swiglu_(q4_a16|q8_a16|nvfp4))_test$'
~~~

Use the candidate's targeted cases and [Op qualification rules](../../docs/maintainer/op-development.md).
Check the packed FP8 head at real 27B width against independently decoded weights and the FP64
projection oracle. Include row scales, partial tiles and BF16 values outside normal FP16 range.
The loading test uses a sparse `248320 × 5120` parent to check exact device and artifact bytes
at separated panels across load and suspend/resume. It keeps the test's explicit Host payload
small, checks the full-geometry text/MTP head and tied embedding selection, and verifies Q8
`RowSplitPanel` MTP K/V projection against the FP64 oracle under `--gpu`.
Cover the drafter's Q4/Q8 projection and SwiGLU paths at T=1–4 and their boundary fallback. Cover
NVFP4 W4A8's activation quantization and observable rounding independently. Check exact packed
layout transforms and workspace bounds, plus ordinary execution and Graph capture/replay.

Qualify GDN using the independent recurrence for output and final state, with nonzero initial
state, grouped heads, full chunks and tails, FP16 and FP32 state where supported, and split and
snapshot/restore cases. Run `ninfer_v100_gdn_recurrent_test` and
`ninfer_v100_gdn_forced_two_stage_test`. Run `ninfer_softmax_attention_volta_flash_test` to qualify
Flash Attention against the paged-attention FP64 oracle using
fragmented BF16 KV, causal/residue boundaries, a partial block, and the dispatch boundaries.
Exercise decode and INT8 fallback and check that a supported large BF16 prefill hits the new
route. Use the route diagnostics or a focused Nsight Systems capture if the reported Op run
cannot establish dispatch. The Volta Flash Op test checks both BF16 geometries, workspace
guards around the key cap, and the native fallback for out-of-range BF16. It also runs an
append and a read on independent streams and arenas, then reads the data appended by the
first execution.

Run the standard real-artifact integration controls including cache and MTP3. The environment
below selects a specific scenario. Inspect the test's effective Engine policy before treating
its result as candidate-route coverage. Exercise the candidate through Serve for the same
scenarios if the test constructs a native-only Engine.

~~~bash
export NINFER_TEST_ARTIFACT="$ARTIFACT"
export NINFER_HYBRID_V100_POLICY=ported NINFER_HYBRID_KV_DTYPE=int8
for scenario in restore-exact-mtp interleaved full-pool persist vision; do
  NINFER_HYBRID_REAL_SCENARIO="$scenario" ctest --test-dir "$CANDIDATE_BUILD" --output-on-failure \
    -R '^ninfer_qwen3_5_hybrid_prefix_real_test$' || exit $?
done
ctest --test-dir "$CANDIDATE_BUILD" --output-on-failure \
  -R '^ninfer_qwen3_5_volta_flash_suspend_real_test$'
~~~

The Flash suspend scenario uses the BF16 KV format and verifies Flash selection after the
Engine remaps its workspace. Its continuation and cold generation must match the resident run.

The CPU FP8 loading suite also checks both DFlash2 bindings. The full fused selector shares the
target's FP8 head and keeps its stored row layout. The optimized fused selector keeps its indexed
Q4 proposal in RowSplit, and its separate target FP8 head can still be packed. For the end-to-end
check, use an explicit DFlash2-compatible artifact with these two head formats. Run both real
cases on the V100 when that artifact is available. These cases skip without this separate fixture.

~~~bash
NINFER_TEST_V100_DFLASH2_ARTIFACT=/path/to/dflash2-model.ninfer \
  ctest --test-dir "$CANDIDATE_BUILD" --output-on-failure \
    -R '^ninfer_qwen3_5_v100_dflash2_(full|optimized)_real_test$'
~~~

## Paired performance

Use the mixed NVFP4/row-scaled FP8 Qwen3.8-27B v3 artifact with its MTP weights and optimized
proposal head. Supply the exact paths and an empty output location. Standalone loopback
servers run one at a time. The runner uses private files, sends SIGINT to each child it
started, and has no deployment or service-manager integration.

~~~bash
python3.11 "$CANDIDATE/bench/v100/qualify.py" \
  --baseline "$BASELINE_BUILD/apps/ninfer-serve" \
  --candidate "$CANDIDATE_BUILD/apps/ninfer-serve" \
  --artifact "$ARTIFACT" \
  --output "$RESULTS/manual-performance" \
  --cache-fixture "$FIXTURES/cache.json" \
  --deployment-fixture "$FIXTURES/deployment.json" \
  --quality-evidence "$RESULTS/manual-quality/quality-evidence.json"
~~~

Omit a private input flag if its evidence is unavailable. The runner
still collects independent results. Run a subset with repeated `--phase` flags when necessary
and keep all phase reports. Exit status 0 means all of the runner's evidence gates passed, 1
means a failed gate, and 2 means open evidence. Op correctness, restart/rollback behavior,
and production promotion also require the checks below regardless of runner status.

The runner first compares old native against old ported, then candidate native against candidate
ported with the four new routes disabled. It isolates the FP8 head, drafter GEMV, recurrent GDN and
Flash routes against the candidate's ported control before comparing the combined candidate.
It records all trials and eight A/B pairs per latency and throughput comparison, alternating the
launch order. It uses the artifact tokenizer to freeze the two long prompt bodies once, and compares
expanded token counts and effective prefill chunk 2,048. Each fresh process warms the model before
cold timing with prefix reuse disabled. Paired arms use the same KV codec, speculation settings,
temperature, seed, Graph option, proposal head, context and chunk size. N-gram drafts are disabled.
The runner checks the named Engine shutdown counters for the routes each comparison exercises.
These count host selections, including Graph capture. `gpu-memory.json` records sampled load and
runtime peaks for the server's process ID when telemetry is available. Peaks between samples can
exceed the recorded values.

| Phase | Matched workloads | Pass evidence |
|---|---|---|
| Controls | Old native and old ported, then candidate native and candidate ported with new routes disabled. INT8 MTP3 | Record the reference differences and verify selected policies and effective settings. |
| Head, drafter and GDN | Each candidate route alone against the candidate's ported control. INT8 MTP3 | Named route exercised and individual workload regression gates. |
| Cold | INT8 and BF16, MTP off and MTP3, prose/code/list with 300 output tokens, prefills about 11.6K and 33K | MTP3 decode geometric mean at least +3% for each KV with paired 95% lower bound above zero. Apply individual regression gates to both speculation modes. |
| Flash | Candidate BF16 Flash route alone against the candidate's ported control | 33K prefill paired 95% lower bound above zero. Report 11.6K, decode, and effective capacity separately. |
| W4A8 | Candidate ported with the option on/off, INT8 and BF16, MTP3 | Matched decode improvement with paired 95% lower bound above zero and no individual regression. W4A8 also needs the quality gate. |
| Warm and pressure | INT8 MTP3, 20 GiB Host, four Device snapshots, separate fresh runs of the 90,787-token replay and the 47-request sequence | Both reproduce frozen output and cache expectations with no repeatable regression. |
| Capacity | INT8 long profile and BF16 shorter profile, auto KV and a short generation on both builds | Report effective capacity, memory and delta. |
| Deployment | Frozen INT8 and BF16 profiles, baseline followed by candidate | Admit a cold prompt near each declared context limit, show the exact warmed frontier, and prove reuse from a private same-build restart. Every server must meet its profile's physical capacity floor. |

For each cold/warm workload, a paired 95% interval showing more than a 2% slowdown fails the
gate. A point estimate worse than 2% with an interval overlapping the threshold stays open.
Report client-visible TTFT alongside server prefill/decode, elapsed time, actual MTP accepted
tokens and round counts, memory, and any failure or outlier. Decode rate uses the output token
intervals after the first token. Wall-time prefill is included for reference to prior published
runs but the gate uses computed tokens divided by server prefill time. Record client
warm-turn TTFT and summed per-request TTFT for pressure separately.

The default cold profile uses a 65,536-token context. The default INT8 cache profile uses
262,144 with auto KV. Override `--cold-context`, `--cache-context`, `--cache-kv-capacity` and
`--cache-prefill-chunk` to match the reference performance configuration. Use `--vision` or
`--cache-vision` for the matching performance phase if the reference includes a vision tower.
The deployment fixture provides this phase's settings separately for each KV codec.

With `--cache-kv-capacity auto`, the runner first measures the INT8 capacity of both builds in
the long profile. Warm comparisons then use the smaller observed physical capacity for both arms.
The cache fixture is an external JSON object containing `warm` (two identical request bodies)
and `pressure` (47 request bodies). Each entry carries `body`, the exact expanded `prompt_tokens`,
`minimum_cached_tokens`, and `expected_text`. Set the warm second request's minimum above zero to
establish reuse in-process and after a fresh same-build restart. Optional `media` holds a priming
request and its changed-media follow-up with a `maximum_cached_tokens` frontier and a changed
expected answer. Optional `near_limit` holds the actual deployment-limit admission requests with
their expected results. Missing optional fixtures leave their gates open.

The bodies use OpenAI chat format. The runner adds its own model ID and streams with usage.
Each trial's private persistence files are temporary. The first candidate trial gives private
copies to the old binary, to the candidate with the native policy, and to the W4A8 candidate
when that option is evaluated. Each must report a fingerprint rejection and compute the entire
prompt cold. Optional `expected_native_text` and `expected_w4a8_text` assert continuations
specific to the respective policy. Keep fixtures and expected outputs from the reference host
or approved archive. Use synthetic input for diagnostic runs.

`deployment.json` contains `int8` and `bf16` objects frozen independently from the qualified
reference profiles. Each present profile requires integer `context`, `minimum_capacity`, and
`prefill_chunk` (a multiple of 128). Set `kv_capacity` to a token count or `"auto"` (default).
Optional settings are `drafts` (default 3), `vision` (default false), `host_cache_mib`
(default 20480), and `device_snapshot_slots` (default 4). Requests have the cache fixture
fields above plus a nonempty `name` unique within their group.

- `near_limit` starts with a fully cold request whose expanded prompt is within one prefill
  chunk of `context`. Freeze its answer and set its reuse ceiling to zero.
- `frontier` contains a priming request followed by at least one known shared prefix with
  `minimum_cached_tokens` and `maximum_cached_tokens` set to the same positive value.
- `restart` contains `prime` and `after` arrays. The first `after` request must declare a
  positive reuse floor. The second server starts with the first server's saved file and must
  acknowledge restoration before this sequence can pass.
- Include `warm` (the exact two-request 90,787-token replay) and `pressure` (47 requests)
  for profiles with the corresponding workloads. Every declared prompt must fit that profile.

For each supplied profile, Serve runs baseline first and candidate second using separate
private caches. The runner records physical capacity from every server, restart file size,
each request's observed cache path and token frontier, and the independently frozen ranges.
A missing KV profile leaves only that deployment gate open. Every declared case in an existing
profile must pass on both binaries. The fixed eight-pair comparison supplies the performance
evidence separately. An incompatible W4A8 cache failure is recorded under that option's gate,
while the A16 warm and pressure results remain available for their own gates.

## Frozen quality

Measure the refreshed-main ported baseline, the candidate A16 route and the same candidate with
W4A8 for both INT8 and BF16 KV. The baseline and candidate probes use the same
[public Engine driver](../ninfer_precision_probe.cpp). Use the original campaign root identified
in the [precision lab notes](../NINFER_V100_LAB.txt). The runner verifies `fixtures/freeze.json`
and `fixtures/tokens-freeze.json` against the selected `corpus.json`, `quality.json`,
`quality-order.json`, `score-windows.json` and `tokens.jsonl` bytes.

Use the existing Linux Volta build options and include the probe target definition at configure
time. Configure the candidate additionally with `-DNINFER_V100_QUALITY_W4A8=ON`.

~~~bash
quality_include="$CANDIDATE/bench/v100/quality_probe.cmake"
cmake -S "$BASELINE" -B "$BASELINE_BUILD" \
  -DCMAKE_PROJECT_ninfer_INCLUDE="$quality_include"
cmake -S "$CANDIDATE" -B "$CANDIDATE_BUILD" \
  -DCMAKE_PROJECT_ninfer_INCLUDE="$quality_include" -DNINFER_V100_QUALITY_W4A8=ON
cmake --build "$BASELINE_BUILD" -j --target ninfer-v100-quality-probe
cmake --build "$CANDIDATE_BUILD" -j --target ninfer-v100-quality-probe
python3.11 "$CANDIDATE/bench/v100/quality.py" \
  --baseline-probe "$BASELINE_BUILD/ninfer-v100-quality-probe" \
  --candidate-probe "$CANDIDATE_BUILD/ninfer-v100-quality-probe" \
  --artifact "$ARTIFACT" --fixtures "$FIXTURES" \
  --output "$RESULTS/manual-quality"
~~~

The probe uses the same frozen prompt token IDs and task order in each arm, with greedy sampling,
seed 1729, no speculation or prefix reuse, and the established exact, JSON and code graders.
A16 scoring uses 16 streams with 8,192 targets each, a 4,096-token context and 2,048-token stride;
because W4A8 quantizes narrow (`T <= 4`) projections only and chunked `CausalScoring` executes zero
`w4a8_narrow` calls, W4A8 omits the vacuous scoring pass and `qualify.py` rejects `w4a8` score
records if supplied. All 384 tasks use their original cluster labels and output budgets. The W4A8
generation arm must record a narrow W4A8 selection. The run directory retains commands, route
counts, fixture and job hashes, probe hashes, raw log probabilities, answers, and grade reasons.
Missing or incomplete inputs leave `status.json` open and produce no quality evidence.

Pass `$RESULTS/manual-quality/quality-evidence.json` as `--quality-evidence`. It contains a `source`
link to the run manifest and four `comparisons` covering `comparison` values `a16` and `w4a8`
crossed with `kv_dtype` values `int8` and `bf16`. Each `a16` comparison contains `documents` with
`id`, `domain`, `tokens`, `control_mean_nll` and `candidate_mean_nll`, while `w4a8` carries an empty
`documents` list; all four carry `tasks` with `id`, `domain`, `cluster` and Boolean `control` and
`candidate` grades. Here the A16 control is refreshed main and the W4A8 control is candidate A16.
The gate bootstraps whole paired streams and fixture clusters using 20,000 replicates and seed
1729. All six two-sided 99.1667% intervals (two `a16` perplexity intervals and four `a16`/`w4a8`
accuracy intervals) must fit within ±0.5% relative perplexity and ±2 percentage points accuracy.
Review per-domain effects and discordant grades with the aggregate intervals.

## Production-equivalent cache and promotion checks

On a test instance using the same runtime policy as the intended deployment, exercise near-limit
admission, media or history changes, and same-build restart from a separately named test
`--prefix-cache-file`. Record the published frontiers and token reuse, repeat a known
continuation and compare committed token IDs, and verify the old binary cannot consume the
candidate's incompatible saved state. Keep independent baseline and candidate cache paths.
Use a private copy to probe an incompatible file. Preserve the original baseline cache and
service configuration throughout. Distinguish a cache rejection plus cold start from an error.
A missing locked fixture leaves its gate open even if the smoke check passes.

Only after reviewing all required Op, model, performance, capacity, cache and quality evidence
should the authorized release agent prepare promotion. Rehearse startup, native fallback,
the selected policy under the intended default or auto configuration, the app health check
and one cold and one warm request before changing the deployment. Rehearse rollback using
the old binary, original options, and preserved cache path. Check actual route/persistence
identity on both directions and verify that steady-state readiness and a correct response
return. Submit the raw reports and explicitly identify failed and open gates. An open or failed A16 gate keeps the native default. W4A8 has its own result in
`summary.json`; it is publishable only after its incremental gates and the underlying A16
profile pass. A W4A8-only regression prevents publishing that option.

For promotion, record the exact measured device profile, selected GDN width intervals and
passing A16 routes in the release change. Resolve the startup policy before packing weights,
computing the cache fingerprint or capturing graphs. Require every selected device to match
the qualified profile. Leave W4A8 behind its explicit option. Run these on the proposed
promotion build after both KV gates pass. Preserve the logs alongside Serve’s cold/warm
checks. Omit the policy flag on the first launch to exercise the real product default.

~~~bash
set -o pipefail
for kv in int8 bf16; do
  common=("$ARTIFACT" --kv-dtype "$kv" --prefill-chunk 2048 --spec mtp --draft-tokens 3
          --lm-head-draft --greedy --max-new 32 --prompt "List three uses of a cache.")
  "$CANDIDATE_BUILD/apps/ninfer" "${common[@]}" 2>&1 | tee "$RESULTS/default-$kv.log"
  "$CANDIDATE_BUILD/apps/ninfer" "${common[@]}" --v100-kernels native 2>&1 | tee "$RESULTS/native-$kv.log"
done
~~~

The current qualification build defaults to native. On the promotion build, check the
default log for the selected A16 profile and exercised routes, and the native log for
zero ported selections. Repeat this pair through Serve with separate test cache files.
