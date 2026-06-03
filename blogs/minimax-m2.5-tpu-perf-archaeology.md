---
title: "A 21× bandwidth gap hiding in one dynamic_slice"
subtitle: "Chasing the real decode throughput ceiling of a 229B MoE on TPU v5e and v6e"
date: 2026-05-31
tags: [tpu, jax, xla, moe, inference, performance, maxtext]
status: validated   # measured on hardware; point-in-time figures
---

# A 21× bandwidth gap hiding in one dynamic_slice

Bringing up MiniMax-M2.5 — a 229B-parameter, 256-expert MoE — on TPU was only half the work.
Getting it *correct* meant verifying that routing, quantization, and RoPE all matched the
reference. Getting it *fast* meant a long argument with the XLA compiler about how it schedules
a decode loop. This post is about the second half, because the surprises there were more
instructive than the speedups.

The headline numbers, all measured on hardware (and worth treating as point-in-time — TPU
software moves):

| TPU | layout | best config | tok/s (pod) |
|---|---|---|---:|
| v6e-64 | `scan=False` | batch 16, int4 KV | **6,792** |
| v6e-64 | `scan=True`  | batch 12, int4 KV | 4,701 |
| v5e-64 | `scan=False` | batch 7, target 96 | **2,556** |
| v5e-64 | `scan=True`  | batch 7, warm cache | 1,836 |

Two things jump out: `scan=False` beats `scan=True` everywhere (by +39% on v5e, +44% on v6e),
and the batch-size curve is bizarre. Both have the same root cause.

## Correctness first, so the perf numbers mean something

None of the throughput work matters if the model is wrong, so that got nailed down up front. The
MoE router is the easy place to be subtly incorrect: MiniMax routes with a **sigmoid in fp32**,
applies a learned correction bias *before* expert selection, and weights *before* normalizing.
Attention applies **QK-norm before RoPE** to match the HuggingFace reference, with partial RoPE
(rotary dim 64). Six routing unit tests pin all of this — sigmoid dtype, pre-bias selection,
pre-bias weighting, sum-to-one normalization, RoPE timescale — and the model answers the usual
sanity prompts correctly ("capital of France" → Paris, `sqrt(144)=12`, `12! = 479,001,600`).

A couple of memory-layout tricks were load-bearing just to *fit*:

- **`param_scan_axis=0`** for the MoE weights gives a `wi_0` layout of
  `(62, 256, 3072, 1536)` and avoids a **2.18 GB XLA transpose buffer** that was OOMing v5e
  outright. This is what made v5e eligible at all.
- The stacked-tensor builder **pre-allocates and fills `result[i, j]`** in place, avoiding a
  **284 GB peak** that the naive concatenate-everything assembly produced during checkpoint
  load.

With correctness and fit settled, the throughput investigation could trust its own numbers.

## The 21× wall

A decode step on this model should be bound by HBM bandwidth: you stream ~10B active params per
token and do relatively little compute. On v6e the peak HBM bandwidth is ~1,800 GB/s. The
measured *effective* bandwidth during decode was about **80 GB/s** — a ~21× gap. The chip was
spending most of each step waiting on memory it should have been able to prefetch.

The culprit is a `dynamic_slice` inside the decode `while_loop`. To index "the current
position" each step, the loop does a data-dependent slice — and a data-dependent index inside a
`while_loop` **prevents XLA from prefetching the next layer's weights**, because it can't prove
ahead of time which memory it will touch. So every layer stalls on a cold fetch instead of
overlapping the fetch with the previous layer's compute. The weights are right there in HBM; the
scheduler just isn't allowed to go get them early.

That single fact explains a cluster of otherwise-baffling experiments:

- **`scan_unroll=2`: −7.8%.** Unrolling the layer loop should help. It hurt (1,708 vs 1,854
  tok/s) — a bigger `while_loop` body gives XLA a harder scheduling problem, and with prefetch
  already blocked there's no overlap to win back.
- **LIBTPU async-collective flags: ~0%.** Async all-to-all/all-reduce flags did nothing, in both
  `scan=True` and `scan=False`. The MoE all-reduce sits on the critical path with no independent
  compute to hide behind, so making it async buys nothing.
- **`scan=False` beats `scan=True`.** Unrolling the layers statically (rather than as a scanned
  `while_loop`) exposes more of the schedule to XLA and sidesteps some of the loop-carried
  dependency, which is exactly why it's faster despite generating a much larger program.

The lesson is a familiar one stated precisely: on an accelerator, *fits in memory* and
*saturates memory bandwidth* are completely different problems, and the gap between them can be
a single instruction the compiler can't see through.

## The batch-size curve that isn't a curve

Throughput vs. batch size on v6e `scan=False` doesn't slope; it has cliffs:

- batch 2 → 336 tok/s, batch 6 → 480 tok/s — roughly **9× slower** than neighbors.
- batch 4 → 2,868, batch 16 → **6,792** — fast.
- only batches that are **multiples of 4 with `batch × 4 ≥ 16`** are fast (4, 8, 12, 16, 20, 24).

And on `scan=True`, odd batches (9, 11) hit a **2× step-time cliff** while even batches scale
roughly linearly. This is XLA tiling: the per-expert work gets tiled, and batch sizes that don't
line up with the tile shape fall off the efficient path into a scalar-ish fallback. The
practical upshot is that "pick the biggest batch that fits" is wrong advice here — you pick the
biggest batch that fits *and* lands on a good tile multiple, and you measure rather than assume.

## A quantization bug worth the war story

Enabling KV-cache quantization with `scan=False` produced a shape mismatch in
`dynamic_update_index_in_dim` at batch ≥ 7. The cause: the quantization *scale* tensor's batch
axis wasn't being sharded by GSPMD the way the cache itself was. The fix is one line — constrain
the scale's sharding right after quantizing:

```python
scale = nn.with_logical_constraint(scale, ar_cache_scale_axis_names)  # after kv_quant.quantize()
```

The twist: that same constraint **degraded v6e batch-4 by 2.5×** (1,124 vs 2,868 tok/s) because
it forced a layout the fast tiling path didn't want. So it ships off by default and gets applied
only where it's actually needed. A correctness fix that's also a 2.5× perf regression elsewhere
is a good reminder that on TPU there is rarely a free `with_logical_constraint`.

## Where it landed

Correctness fully verified; throughput characterized end to end. The v6e ceiling of **6,792
tok/s** (`scan=False`, batch 16) is +44% over the best scanned config, and the perf model is now
legible: bandwidth-bound, prefetch blocked by a loop-carried `dynamic_slice`, gated by XLA tile
multiples. The obvious next lever — int8 weight quantization to roughly halve the bytes moved per
token, which should push v6e well past 8,000 tok/s — was scaffolded but blocked on producing a
bf16 checkpoint first, and on a stretch of genuinely brutal TPU capacity shortages.

The transferable takeaways:

1. **Bandwidth-bound ≠ bandwidth-saturated.** Measure effective GB/s, not just tok/s; a 21× gap
   is invisible until you compute it.
2. **A data-dependent index in a `while_loop` can kill weight prefetch.** That's often the real
   reason a decode loop is slow.
3. **TPU throughput is quantized in batch size.** Tile multiples beat "just go bigger."
4. **`with_logical_constraint` is a sharp tool** — sometimes the fix and the regression at once.

---

*Numbers measured on v5e-64 and v6e-64 (MaxText `feature/minimax-m2.5-support`); treat as
point-in-time. Routing correctness in `tests/unit/minimax_routing_test.py`.*
