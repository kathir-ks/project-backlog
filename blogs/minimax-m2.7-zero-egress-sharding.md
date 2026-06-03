---
title: "Running a 230B model on a pod that can't hold it"
subtitle: "Zero-egress, host-distributed checkpoint sharding for MiniMax-M2.7 on TPU"
date: 2026-05-31
tags: [tpu, jax, moe, inference, maxtext, distributed-systems]
status: validated   # decode + API server verified on real hardware
---

# Running a 230B model on a pod that can't hold it

MiniMax-M2.7 is a 230-billion-parameter mixture-of-experts model: 256 experts, top-8
routing, 62 layers, ~10B parameters active per token. In `float16` the weights are about
**460 GB**.

The target hardware was a **TPU v5e-64** — a pod of 16 hosts, each with 4 chips. The
catch: each v5e host has only **~188 GB of RAM**. The standard MaxText inference path
loads the *full* checkpoint into host memory on *every* host and lets the runtime shard it
onto the chips. On v5e that means 16 hosts each trying to hold 460 GB in 188 GB of RAM. It
doesn't fit. It isn't close.

The naive workaround — replicate the checkpoint to every host's local disk — needs
`460 GB × 16 = 7.4 TB` of storage moving across the pod, and on a RAM-backed scratch FS
(which is what you use on TPU hosts) that storage doesn't exist. There was also a harder
constraint behind all of this: a previous project had run up a **~$1,800 cross-region GCS
bill** by moving checkpoints between a US VM and a European bucket. The rule since then has
been absolute: **zero outbound bytes per host**, everything stays in-region, nothing gets
replicated that doesn't have to be.

So the requirement crystallized into something specific:

> Each host should hold **only the slices of the model its own 4 chips will actually own at
> decode time** — nothing more — and it should fetch those slices by streaming from
> HuggingFace once, never from another host or another region.

This post is how that works. The punchline up front: each host ends up storing **~27–30
GB** instead of 460 GB, the pod-wide footprint drops from 7.4 TB to **~432 GB**, and the
model decodes correctly and fast (**6,678 tok/s** on v6e-64).

---

## The key insight: the engine already knows the sharding

Here is the observation that makes the whole approach clean. When MaxText builds its
inference engine, it computes a **sharding for every parameter** — a
`jax.sharding.Sharding` that says exactly how each weight tensor is split across the device
mesh. That information exists *before* any weights are loaded. You can ask for it against an
*abstract* state (shapes and dtypes only, no actual arrays), so getting it costs nothing.

```python
def maxtext_pyconfig_and_shardings(model_size, extra_args):
    """Initialize MaxText config + engine, return the param sharding tree."""
    config = pyconfig.initialize(
        ["convert_minimax_m2_distributed"] + extra_args + [
            f"model_name={model_size}",
            "scan_layers=true",
            "weight_dtype=bfloat16",
            "attention=dot_product",
            "quantization=",
        ]
    )
    engine = maxengine.MaxEngine(config)
    init_state_fn = functools.partial(
        maxtext_utils.init_initial_state, engine.model, None, engine.config, False, rng)
    # get_abstract_state never materializes weights — just shapes + shardings.
    _, _, state_mesh_shardings = maxtext_utils.get_abstract_state(
        engine.config, engine._mesh, init_state_fn, False)
    return engine._mesh, state_mesh_shardings.params
```

Once that sharding tree is in hand, the rest is bookkeeping. For each parameter leaf, JAX
reports **which slice of the global tensor each device owns**:

```python
addr = sharding.addressable_devices_indices_map(global_shape)
# -> { <device>: (slice(0,3072), slice(0,8), slice(...)), ... }
```

The critical word is **addressable**. Run this inside a `jax.distributed` job and each
process only sees its *local* devices in that map. So running **one converter process per
host** means process `p` is told precisely the slices its 4 chips need — and nothing about
any other host's slices. That is the entire trick: the model's own sharding plan, queried
per-host, *is* the work allocation.

```python
jax.distributed.initialize()           # now the mesh spans all 64 chips
proc_idx   = jax.process_index()        # 0..15
local_devs = list(jax.local_devices())  # this host's 4 chips

mesh, param_shardings = maxtext_pyconfig_and_shardings(args.model_size, args.maxtext_args)

per_dev_slices = {}
for leaf, (global_shape, _dtype) in shapes.items():
    sharding = flat[leaf]
    addr = sharding.addressable_devices_indices_map(global_shape)
    for dev, idx in addr.items():
        per_dev_slices[(leaf, local_devs.index(dev))] = _normalize_idx(idx, global_shape)
```

One wrinkle is worth calling out, because it is easy to miss:
`addressable_devices_indices_map` returns `slice(None, None, None)` for axes that are
**fully replicated** across the mesh (e.g. the hidden dimension of a tensor that's only
sharded along experts). Downstream code needs concrete `[start, stop)` bounds to size the
output files, so every index gets normalized first:

```python
def _normalize_idx(idx, global_shape):
    out = []
    for s, dim in zip(idx, global_shape):
        if not isinstance(s, slice):           # scalar index on this axis
            out.append(slice(int(s), int(s) + 1))
            continue
        start, stop, step = s.indices(dim)     # resolves slice(None) -> (0, dim, 1)
        assert step == 1, f"non-unit step unsupported: {s}"
        out.append(slice(start, stop))
    return tuple(out)
```

## Downloading only what this host needs

From the per-device slices, this host's **expert range** follows directly from the MoE
shardings — the experts are sharded along the `expert` mesh axis, so each host owns a
contiguous band of the 256 experts:

```python
expert_ranges = []
for leaf, e_ax in _EXPERT_AXIS.items():         # leaves that have an expert axis
    for d in range(len(local_devs)):
        idx = per_dev_slices[(leaf, d)]
        expert_ranges.append((idx[e_ax].start, idx[e_ax].stop))
host_expert_lo = min(r[0] for r in expert_ranges)
host_expert_hi = max(r[1] for r in expert_ranges)
# e.g. "expert range [128, 144) (host owns 16 of 256)"
```

That range determines which HuggingFace safetensors files actually matter to this host. The
download loop is **streaming**: it fetches a file only right before the layer that needs it,
and `unlink()`s it the moment its last layer has been processed. The MoE weights arrive as
FP8 with block scales, so each tensor is dequantized to bf16 on CPU as it's read:

```python
def _block_dequant_to_bf16(x, s, block=128):
    # FP8 weight x with per-128x128-block scale s -> bf16
    M, N = x.shape
    x = x.to(torch.float32)
    y = torch.empty_like(x, dtype=torch.bfloat16)
    for i in range(0, M, block):
        for j in range(0, N, block):
            scale = s[i // block, j // block]
            y[i:i+block, j:j+block] = (x[i:i+block, j:j+block] * scale).to(torch.bfloat16)
    return y
```

Because files are fetched-then-dropped, the **peak FP8 footprint on disk is ~15 GB**, not
the ~215 GB you'd need to stage the whole quantized checkpoint at once. Streaming is the
difference between "fits comfortably" and "OOMs the scratch FS."

## Writing one `.npy` per (leaf, device)

For each parameter leaf and each of the host's 4 devices, a single `.npy` file is written,
sized exactly to that device's slice. Layered weights (the per-layer attention and MoE
tensors) are filled in layer-by-layer as each layer's HF file streams in, so the full global
tensor is never assembled in memory:

```python
def write_layer_slice(arrs, per_dev_slices, leaf, L, source, local_devs):
    """Write layer L's contribution to every local device that owns it."""
    layer_axis = _LAYER_AXIS[leaf]
    for d in range(len(local_devs)):
        idx = per_dev_slices[(leaf, d)]
        l_slc = idx[layer_axis]
        if not (l_slc.start <= L < l_slc.stop):
            continue                       # this device doesn't own layer L
        local_L = L - l_slc.start
        src_idx = _drop_axis(idx, layer_axis)
        tgt = arrs[(leaf, d)]
        tgt_idx = tuple(local_L if i == layer_axis else slice(None)
                        for i in range(tgt.ndim))
        tgt[tgt_idx] = source[src_idx]     # write just this layer's slab
```

Each host then writes a `manifest.p{proc_idx}.json` recording the global shapes, dtype, and
the list of shard files it produced. After a clean run on v5e-64, `du -sh` on a host's output
directory reads **~27–30 GB**. Sixteen of those is ~432 GB across the pod — versus 7.4 TB to
replicate. The same physical bytes that would have been copied 16 times are instead
partitioned once.

## The decode side: rebuilding a global array from local shards

At decode time, the loader detects which layout it's looking at — a single `manifest.json`
(replicated) or per-host `manifest.p*.json` (distributed) — and for the distributed case
rebuilds each parameter as a globally-sharded `jax.Array` from its local per-device `.npy`
files:

```python
def _detect_distributed(npy_dir):
    return any(npy_dir.glob("manifest.p*.json"))

# ... for each leaf this host owns:
local_arrays = []
for dev, shard_path in shards_for(leaf, proc_idx):
    mm  = np.load(str(shard_path), mmap_mode="r")     # mmap — no full read
    arr = jax.device_put(np.asarray(mm).astype(global_dtype), dev)
    local_arrays.append(arr)

built[leaf] = jax.make_array_from_single_device_arrays(
    global_shape, sharding, local_arrays)             # stitch into the global array
```

`make_array_from_single_device_arrays` is the mirror image of
`addressable_devices_indices_map`: the files were built *from* the sharding plan, and handing
the per-device buffers back *to* that same plan reconstitutes the distributed array. No
process ever materializes the full 460 GB tree. The whole thing is then patched into
MaxEngine's parameter-loading path, so the rest of the inference stack is none the wiser.

## Does it work?

Yes. On **v5e-64 (europe-west4-b)**, distributed decode produces coherent output that is
**byte-identical across all 16 hosts**:

```
"The capital of France is" ->
  " Paris. The capital of Germany is Berlin. The capital of Italy is Rome.
   The capital of Spain is Madrid. ..."
```

Per-host conversion takes ~40 minutes (dominated by the sequential HF download at
~35 s/layer); compile plus first token is ~5 minutes after launch. An **OpenAI- and
Anthropic-compatible FastAPI server** was also stood up over the `.npy`-backed engine, and
all three surfaces were verified through an IAP tunnel (`/v1/completions`,
`/v1/chat/completions`, `/v1/messages`) — 9.1 s cold, 2.5 s warm, with correct token
accounting and only **8.3 GB/chip** of HBM at batch=1, ctx=2048, int4 KV.

## Throughput, and a 902× cliff

On **v6e-64**, with proper sparse MoE routing, the model is fast. The peak from the sweep
(verified in the run's `bench_steps` JSON):

| batch | KV  | target | step p50 | **tok/s (pod)** |
|------:|-----|-------:|---------:|----------------:|
| 1     | bf16 | 256   |  74 ms   |   860 |
| 16    | int8 | 256   | 251 ms   | 4,087 |
| 32    | int4 | 128   | 451 ms   | 4,539 |
| 64    | int4 | 128   | 698 ms   | 5,866 |
| **96**| **int4** | **128** | **920 ms** | **6,678** |

That `b=96` row is `global_batch_size=6144` across 64 chips at `tok_per_s_chip=104.3`.

The interesting part is what happens on **v5e**. The same model, same code, tops out at about
**7.4 tok/s** pod-wide — roughly **902× slower** than v6e. That is not a memory problem (the
sharding already solved that) and it's not raw FLOPs. It's **one missing collective**.

MaxText's fast MoE path (`megablox` / `sparse_matmul`) ships each token only to its top-8
experts using a **ragged all-to-all** over the inter-chip interconnect. v6e supports that
collective; **v5e does not** — XLA bails with:

```
RET_CHECK failure ... !HasLimitedIciRouting() Ragged all-to-all is <...>
```

The only fallback on v5e is `megablox=false sparse_matmul=false capacity_factor=2.0`, which
routes *every* token to *every* expert in a dense matmul and masks the unused outputs. For a
256-expert model, that is a staggering amount of wasted compute — and it's the single line
responsible for the ~900× gap. The fix is a custom Pallas grouped-GEMM kernel that does the
gather/scatter on-chip and skips ragged-all-to-all entirely; that's scaffolded (with passing
correctness tests) but the on-device tuning is a separate story.

## What broke along the way

- **Manifest idempotency.** Conversion is long and TPUs get preempted. Each host's first act
  is to check whether its `manifest.p{idx}.json` and every expected `.npy` already exist, and
  skip the work if so. The streaming-convert logs show several FAIL→RUN cycles before the run
  that reached `layers: 100%` — the idempotency check is what made re-launching cheap instead
  of a 40-minute restart.
- **Handle cache vs. dtmpfs pages.** Holding open safetensors file handles kept unlinked
  files' pages pinned in the RAM-backed FS, quietly eating the headroom that streaming was
  supposed to buy. The fix was to always release the handle cache after each window so the
  `unlink()` actually frees memory.
- **Sharding match is load-bearing.** The converter must be launched with the *same*
  `ici_tensor_parallelism=8 ici_expert_parallelism=8` as decode. If they differ, the
  `addressable_devices_indices_map` slices won't line up with what the decode engine expects,
  and the result is a shape mismatch on load — after a 40-minute conversion. Passing the mesh
  args through to both sides is non-negotiable.

## Takeaways

The core idea generalizes well beyond this one model: **don't fight the framework's sharding
plan — query it, and let it tell you the work partition.**
`addressable_devices_indices_map` on the way out and `make_array_from_single_device_arrays` on
the way back are a clean, symmetric pair for turning "the runtime knows how to shard this"
into "every host persists only its share." Combined with fetch-then-drop streaming from the
source, a 230B model that doesn't fit on any single host becomes ~27 GB of local state per
host, with zero host-to-host or cross-region traffic.

And the v5e cliff is a good reminder that on TPU, the bottleneck is often not memory or FLOPs
but **which collectives a given hardware generation actually implements**. A model can *fit*
perfectly and still run 900× too slow because one all-to-all isn't there.

---

*Code: `convert_minimax_m2_distributed.py` and `decode_minimax_m2_npy.py` in the MaxText
`feature/minimax-m2.7` branch. Numbers from the v6e-64 sweep (2026-05-22) and the v5e-64
analysis (2026-05-23).*
