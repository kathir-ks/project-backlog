---
title: "A 21× bandwidth gap hiding in one dynamic_slice"
subtitle: "Chasing the real decode throughput ceiling of a 229B MoE on TPU v5e and v6e"
date: 2026-05-31
tags: [tpu, jax, xla, moe, inference, performance, maxtext]
status: validated   # measured on hardware; point-in-time figures
---

# A 21× bandwidth gap hiding in one dynamic_slice

Bringing up MiniMax-M2.5 — a 229B-parameter, 256-expert MoE — on TPU was only half the work.
Getting it *correct* meant verifying that routing, quantization, and RoPE all matched the reference.
Getting it *fast* meant a long argument with the XLA compiler about how it schedules a decode loop.
This post is about the second half, because the surprises there were more instructive than the
speedups themselves — they're a tour of the gap between "the model fits and runs" and "the model
runs as fast as the hardware can physically go."

The headline numbers, all measured on hardware (and worth treating as point-in-time — TPU software
moves, and these are notes from a particular week):

| TPU | layout | best config | tok/s (pod) |
|---|---|---|---:|
| v6e-64 | `scan=False` | batch 16, int4 KV | **6,792** |
| v6e-64 | `scan=True`  | batch 12, int4 KV | 4,701 |
| v5e-64 | `scan=False` | batch 7, target 96 | **2,556** |
| v5e-64 | `scan=True`  | batch 7, warm cache | 1,836 |

Two things jump out: `scan=False` beats `scan=True` everywhere (by +39% on v5e, +44% on v6e), and
the batch-size curve is bizarre. Both have the same root cause, and finding it is the story.

## Correctness first, so the perf numbers mean something

None of the throughput work matters if the model is wrong, so that got nailed down up front — and
MoE routing is exactly the place where a model can be subtly, plausibly incorrect and still produce
fluent-looking text. MiniMax routes with a **sigmoid in fp32** (not the more common softmax),
applies a learned correction bias *before* expert selection, and weights the chosen experts *before*
normalizing them to sum to one. Attention applies **QK-norm before RoPE** to match the HuggingFace
reference, with partial RoPE (rotary dim 64, i.e. only half the head dimension is rotated). Each of
those is a place where getting the order or the dtype wrong yields a model that's *almost* right —
right enough to fool a casual eye, wrong enough to be useless. Six routing unit tests pin all of it:
sigmoid dtype, pre-bias selection, pre-bias weighting, sum-to-one normalization, and the RoPE
timescale match. And the model answers the standard sanity prompts correctly — "capital of France"
→ Paris, `sqrt(144) = 12`, `12! = 479,001,600`, a multi-step word problem — which is the coarse
end-to-end check that the fine-grained unit tests are backed by reality.

A couple of memory-layout tricks were load-bearing just to make the thing *fit*, before any
throughput question was even on the table:

- **`param_scan_axis=0`** for the MoE weights gives a `wi_0` layout of `(62, 256, 3072, 1536)` and
  avoids a **2.18 GB XLA transpose buffer** that was OOMing v5e outright. The transpose was implicit —
  XLA inserted it to reconcile the checkpoint layout with the compute layout — and choosing the scan
  axis carefully makes it unnecessary. This single change is what made v5e eligible to run the model
  at all.
- The stacked-tensor builder **pre-allocates and fills `result[i, j]`** in place rather than
  concatenating per-expert slabs, avoiding a **284 GB peak** that the naive assembly produced during
  checkpoint load. 284 GB is not a typo; the naive concatenate-everything path briefly needs that
  much, and pre-allocation sidesteps it entirely.

With correctness pinned and the model fitting, the throughput investigation could trust its own
numbers.

## The 21× wall

A decode step on this model should be bound by HBM bandwidth, not compute. The arithmetic is simple:
you stream the ~10B active parameters per token from HBM and do comparatively little math on them, so
the chip should be limited by how fast it can read weights, which on v6e is ~1,800 GB/s peak. The
measured *effective* bandwidth during decode was about **80 GB/s** — a ~21× gap. The chip was
spending the overwhelming majority of each step *waiting on memory it should have been able to fetch
ahead of time.*

The culprit is a `dynamic_slice` inside the decode `while_loop`. To index "the current position" each
step, the loop performs a data-dependent slice — and a data-dependent index inside a `while_loop`
**prevents XLA from prefetching the next layer's weights**, because the compiler can't prove ahead of
time which memory the slice will touch. Without that proof, it can't issue the load early, so every
layer stalls on a cold fetch instead of overlapping the fetch with the previous layer's compute. The
weights are sitting right there in HBM; the scheduler simply isn't allowed to go get them before it's
certain it needs them.

That single fact explains a whole cluster of otherwise-baffling experiments, each of which looked like
it *should* help and didn't:

- **`scan_unroll=2`: −7.8%.** Unrolling the layer loop should give XLA more to work with. It hurt
  (1,708 vs 1,854 tok/s) — a bigger `while_loop` body is a harder scheduling problem, and with
  prefetch already blocked there's no overlap to win back, so all you've added is complexity.
- **LIBTPU async-collective flags: ~0%.** Flags that make the all-to-all / all-reduce collectives
  asynchronous did nothing, in both `scan=True` and `scan=False`. The MoE all-reduce sits on the
  critical path with no independent compute to hide behind, so making it async buys nothing — there's
  nothing to overlap it *with*.
- **`scan=False` beats `scan=True`.** Unrolling the layers statically (rather than running them as a
  scanned `while_loop`) exposes far more of the schedule to XLA and sidesteps some of the loop-carried
  dependency that blocks prefetch. That's *why* it's faster despite generating a much larger compiled
  program — and it's a direct consequence of the prefetch diagnosis, not a coincidence.

The lesson is a familiar one stated precisely: on an accelerator, *fits in memory* and *saturates
memory bandwidth* are completely different problems, and the gap between them can be a single
instruction the compiler can't see through. You will not find this by reading tok/s; you find it by
computing effective GB/s and noticing it's 20× below the spec sheet.

## The batch-size curve that isn't a curve

Throughput vs. batch size on v6e `scan=False` doesn't slope smoothly; it has cliffs:

- batch 2 → 336 tok/s, batch 6 → 480 tok/s — roughly **9× slower** than their neighbors.
- batch 4 → 2,868, batch 16 → **6,792** — fast.
- only batches that are **multiples of 4 with `batch × 4 ≥ 16`** are fast (4, 8, 12, 16, 20, 24).

And on `scan=True`, odd batches (9, 11) hit a **2× step-time cliff** while even batches scale roughly
linearly. This is XLA tiling: the per-expert work gets tiled into fixed-shape blocks, and batch sizes
that don't line up with the tile shape fall off the efficient path into a near-scalar fallback. For a
256-expert model the "outer" dimension that gets tiled is `batch × local_experts`, which is why the
fast batches are precisely the ones where that product is a healthy multiple of the tile size.

The practical upshot is that "pick the biggest batch that fits" is actively wrong advice here. You
pick the biggest batch that fits *and* lands on a good tile multiple, and you measure rather than
assume — because the penalty for guessing wrong isn't a few percent, it's 9×. This is the kind of
thing that's invisible in a framework's documentation and obvious the moment you sweep batch size and
plot it.

## A quantization bug worth the war story

Enabling KV-cache quantization with `scan=False` produced a shape mismatch in
`dynamic_update_index_in_dim` at batch ≥ 7. The cause: the quantization *scale* tensor's batch axis
wasn't being sharded by GSPMD the way the cache itself was, so the two disagreed on shape when the
cache update tried to write. The fix is one line — constrain the scale's sharding right after
quantizing so GSPMD treats it the same as the cache:

```python
scale = nn.with_logical_constraint(scale, ar_cache_scale_axis_names)  # after kv_quant.quantize()
```

The twist, and the reason it's a war story rather than a footnote: that same constraint **degraded
v6e batch-4 by 2.5×** (1,124 vs 2,868 tok/s) because it forced a layout the fast tiling path didn't
want. So the correct fix in one regime is a major regression in another. It ships off by default and
gets applied only where the shape mismatch actually bites. A correctness fix that's simultaneously a
2.5× perf regression elsewhere is a good, humbling reminder that on TPU there is rarely a free
`with_logical_constraint` — every sharding hint you add is also a sharding hint you've taken away from
the compiler's own search.

## Beyond raw throughput: the supporting wins

Two more results round out the picture. First, **int8 weight quantization** was wired through
correctly — the MoE path routes via `get_einsum → AqtEinsum` so that *dense* matmuls (not just the
sparse routing) get quantized, with the AQT path-matching handling the nested `layers_0` naming. The
expectation is that halving the bytes moved per token roughly doubles the bandwidth-bound ceiling,
pushing v6e past 8,000 tok/s — but it was blocked on producing a bf16 scan checkpoint first (and on a
stretch of genuinely brutal TPU capacity shortages, where v6e slices were cycling
`CREATING → GONE` every couple of minutes). Second, the **JAX compilation cache** matters enormously
operationally: a warm cache drops generate-step-0 from ~135 s cold to ~4 s, a 34× faster restart,
which on preemptible hardware is the difference between recovering from a preemption gracefully and
re-paying two minutes of compile every single time the slice comes back.

## Where it landed, and the transferable takeaways

Correctness fully verified; throughput characterized end to end. The v6e ceiling of **6,792 tok/s**
(`scan=False`, batch 16) is +44% over the best scanned config, and — more valuable than the number —
the perf model is now legible: bandwidth-bound, prefetch blocked by a loop-carried `dynamic_slice`,
gated by XLA tile multiples in the batch dimension. That legibility is what lets you reason about the
next model instead of re-discovering everything.

The transferable points:

1. **Bandwidth-bound ≠ bandwidth-saturated.** Measure effective GB/s, not just tok/s; a 21× gap is
   invisible until you compute it against the spec sheet.
2. **A data-dependent index in a `while_loop` can kill weight prefetch.** That's often the real reason
   a decode loop is slow, and `scan=False` can be faster precisely because it sidesteps the loop.
3. **TPU throughput is quantized in batch size.** Tile multiples beat "just go bigger" — and getting
   it wrong costs 9×, not 9%.
4. **`with_logical_constraint` is a sharp tool** — sometimes the fix and the regression at once. Add
   sharding hints surgically, not globally.

---

*Numbers measured on v5e-64 and v6e-64 (MaxText `feature/minimax-m2.5-support`); treat as
point-in-time. Routing correctness in `tests/unit/minimax_routing_test.py`.*
