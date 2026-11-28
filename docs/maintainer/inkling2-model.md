# Inkling-2-Mini

The Inkling-2 loader, frontend, vision encoder, DSpark drafter and CUDA operations come from
`ashalliants/ninfer-3090` at `be23dfa`. The native s-bed Engine dispatches the artifact to
this model using its text configuration. The V100 configuration uses MXFP8 expert banks, global
rk4v4 KV, an 8192-token prefill chunk, vision and three draft tokens.

## Hybrid prefix cache

Enable the single-lane cache with `--use-alt-prefix-caching --host-cache-mib 20480`.
It starts empty on each process launch. The shared index owns the 64-token prefix tree, slab
reservations, snapshot selection and eviction decisions. A block holds one 64-row slice of each
global target and drafter KV plane. An image holds the live windowed target rows, the four
convolution histories for each layer visit, windowed drafter rows and its joint input.
Short frontiers reserve only the slabs needed for their live rows.
The image and all blocks on its path must finish copying before another request can resume there.
Media dimensions and normalized pixel data contribute to block identity.

Only completed prompt forwards produce snapshots. The first 64 tokens and subsequent absolute
prefill chunk boundaries are repeatable cut points for the tensor-core kernels. A request starts
with the selected snapshot and computes the remaining prompt, including its final logits.
Draft verification and rollback do not publish prefix state.

The server reports slab occupancy, device evictions, host block and image restores, and reused
tokens under `/stats`. The real-model qualification driver is
`bench/v100/inkling_hybrid.py`. Its Qwen restart phase uses a separate persistent cache file
and compares generated IDs before and after restart.

## Running the live checks

Start Inkling with the intended serving flags and a private request log. Point the probe at
the backend port. Keep the output directory across restarts so the next process can compare
its output with the first process.

```sh
python3 bench/v100/inkling_hybrid.py inkling --url http://127.0.0.1:8080 \
    --model inkling2-mini-v100 --request-log "$INKLING_REQUEST_LOG" --output "$CHECK_DIR"
```

Run `inkling-reload` with the same arguments after restarting Inkling. Run `inkling-short`
on a cold instance for the bounded release check. The full test displaces a 32,768-token
prefix, checks the restored token IDs and exercises shared branches, changed images and a
disconnected streaming client.

For Qwen, use the production settings with a fresh private `--prefix-cache-file`. Run `qwen`
with the Qwen backend URL, served model ID and output directory. Stop the server gracefully,
restart it with the same cache file and run `qwen-restart` against that process.
