---
title: "Porting an autonomous training-optimization loop to TPU"
subtitle: "How a not-fusing-two-jit-calls decision and a bigger batch moved val_bpb from 1.136 to 1.115"
date: 2026-05-31
tags: [jax, flax-nnx, tpu, training, performance, autonomous-research]
status: validated   # documented optimization with kept/rejected experiments
---

# Porting an autonomous training-optimization loop to TPU

The premise is Karpathy's autoresearch loop: an agent is handed a training script, allowed to
edit it, runs a fixed-budget experiment (here, a ~5-minute training run), reads a single metric
(`val_bpb`, bits-per-byte on a held-out set), and keeps or discards the change — with an explicit
*simplicity criterion*, so a tiny win that deletes code beats a tiny win that adds a hack. The
original is GPU/PyTorch. This project ports the whole thing to a single-host **TPU v4-8** (4 chips,
32 GB HBM each) with JAX and Flax NNX, data-parallel across the four chips. The interesting content
isn't the loop scaffolding — it's the TPU-specific optimization decisions the loop converged on, and
the one that's genuinely counterintuitive.

## The rules of the game

What makes this kind of loop trustworthy is that the *measurement* can't be gamed. The protocol
enforces that by splitting the code into mutable and immutable halves:

- `prepare_jax.py` owns the data loader, the tokenizer, the `evaluate_bpb` function, and the time
  budget. It is the ground-truth metric harness and is **never touched** — the agent literally
  cannot edit it. Whatever the agent does to go faster, it's still being graded by the same
  unmovable ruler.
- `train_jax.py` is the only mutable file: model definition, optimizer, training loop. (`train.py`,
  the original GPU path, is kept around untouched as a reference.)

Each experiment runs under a fixed wall-clock budget, logs to a results file, and lives on its own
branch so kept and rejected changes are both auditable after the fact. The simplicity criterion is
the soul of it: between two changes with equal `val_bpb`, the one that *removes* code wins. That
single rule is what keeps an optimization loop from degenerating into an ever-growing pile of
micro-hacks, and — as it happens — it's also what produced the biggest single speedup here.

Everything below is a change to `train_jax.py`, measured against that fixed harness, on the same
v4-8.

## The headline: a smaller knob beat a bigger model

The optimized configuration improved on the baseline across the board within the same ~5-minute
budget:

| metric | baseline | optimized | Δ |
|---|---:|---:|---:|
| val_bpb | 1.136 | **1.115** | −0.021 |
| MFU | 8.71% | **9.80%** | +12.5% |
| throughput | ~400k tok/s | **~440k tok/s** | +10% |

Lower bits-per-byte is better, so −0.021 is a real gain, and it came with higher utilization and
throughput rather than trading them away. But the more instructive result is the model-size sweep —
because the depth that wins is emphatically *not* the biggest:

| depth | batch | params | tokens seen | val_bpb | MFU |
|---:|---:|---:|---:|---:|---:|
| **8** | 64 | 50.3M | 141.0M | **1.115** | 9.8% |
| 10 | 64 | 85.9M | 101.2M | 1.143 | 12.4% |
| 12 | 32 | 135.3M | 73.9M | 1.232 | 14.3% |

Read the columns together and the trade-off is stark. As the model gets deeper, **MFU climbs**
(more FLOPs per step means the chip is busier) but **val_bpb gets worse**, because in a fixed time
budget the bigger model sees far fewer tokens — 73.9M at depth 12 versus 141.0M at depth 8 — and
simply doesn't converge as far. The 12-layer model is using the hardware most efficiently and
producing the worst language model. Around 50M parameters is the sweet spot for *this* budget: small
enough to grind through enough tokens to converge, big enough to have the capacity to. It's a clean,
concrete demonstration that **MFU is not the objective; it's a means**, and optimizing it directly
can walk you straight away from the thing you actually want.

## Three TPU-port decisions

**1. A bigger batch deleted gradient accumulation entirely.** Raising `DEVICE_BATCH_SIZE` from 32 to
64 let the full intended batch fit in a single step. That sounds minor; it's structural. With the
batch fitting in one step, the entire micro-step *gradient-accumulation loop* disappears — and with
it the mid-step `float(loss)` host synchronization (a device→host transfer that stalls the pipeline)
and the `tree_map` gradient averaging across micro-batches. The code got shorter *and* faster at the
same time. This is the simplicity criterion working exactly as designed: the biggest win was a
deletion, not an addition.

**2. To afford that batch, gradient-checkpoint the block.** A batch-64 forward pass produces
attention-score tensors of shape `f32[64, 4, 2048, 2048]` — roughly **4 GB each**. Several of those
live simultaneously across the layers and they do not fit alongside everything else in 32 GB. Flax
NNX's `@nnx.remat` solves it by *not storing* those activations and instead recomputing them during
the backward pass — trading a bit of redundant compute for a large memory saving:

```python
@nnx.remat
def __call__(self, x, ve, cos_sin, mask):
    x = x + self.attn(norm(x), ve, cos_sin, mask)
    x = x + self.mlp(norm(x))
    return x
```

Remat is what makes the batch-64 win in (1) actually fit; the two changes are a package.

**3. The counterintuitive one: do *not* fuse the optimizer into the training step.** The obvious
"optimization" is to compile forward + backward + the optimizer update as a single `@jax.jit`
function — fewer dispatch boundaries, more for XLA to fuse, surely faster. On TPU it OOM'd. The
reason is the genuinely useful insight, recorded in the project's optimization notes:

> Combining `compute_grads` and `apply_optimizer` into a single `@jax.jit` function caused OOM
> (31.78G program memory vs 30.75G available). XLA could not free intermediate activations when the
> entire forward+backward+optimizer was a single compiled program. The attention score matrices
> (`f32[64,4,2048,2048]` = 4GB each) dominated memory. Keeping separate JIT calls acts as a natural
> memory checkpoint boundary.

This is worth internalizing because it inverts a common intuition. A `@jax.jit` boundary isn't only
a dispatch boundary — it's also a *memory lifetime* boundary. When gradient computation and the
optimizer step are separate compiled programs, XLA can release the backward pass's intermediate
buffers before the optimizer allocates its state, because the two programs have disjoint liveness.
Fuse them into one program and XLA must keep every intermediate alive across the whole thing, because
from its point of view they might all be needed — and that overlap pushes peak program memory from
30.75 GB available to 31.78 GB needed. The fix is to keep three separate jitted functions, and the
"missing" fusion is load-bearing: it's the thing that makes the step fit.

A smaller related win rounds it out: data goes straight to the device with
`jax.device_put(x_np, sharding)`, skipping a redundant `jnp.array` host copy that would otherwise
materialize the batch twice.

```python
x_np, y_np, epoch = next(train_loader)
x = jax.device_put(x_np, data_sharding)   # numpy -> device, no extra host copy
y = jax.device_put(y_np, data_sharding)   # data_sharding = NamedSharding(mesh, P('data'))
```

## What the loop produced

The artifact of an autoresearch run is the *trajectory*, not a single number. Here it's seven
timestamped experiment logs with kept-and-rejected changes, the depth sweep above, and a
`progress.png` tracking `val_bpb` over time. The final logged run — a DEPTH=10 experiment — ends at
`val_bpb 1.143226`, `peak_hbm_mb 2978.5`, `mfu_percent 12.41`, which lines up exactly with the sweep
table's depth-10 row. That detail is itself a small lesson: the *best* run (depth 8, val_bpb 1.115)
is not the *last* run. An exploration loop deliberately probes past the optimum — it tried depth 10
and 12 to confirm they were worse — so you read the whole trajectory to find the winner, you don't
just grab the endpoint. A loop that always ended on its best configuration wouldn't be exploring.

## What the loop is and isn't good at

This loop is good at exactly the things a patient, tireless engineer is good at: sweeping a knob,
running the fixed protocol honestly, and reporting kept/rejected with the numbers attached. It found a
real, defensible local optimum (depth 8, batch 64, remat, unfused optimizer) and produced a clean
account of *why* each change helped or hurt. What it does *not* do is invent a new architecture,
question the metric, or notice that the whole search space might be wrong — it optimizes within the
design space it's handed. That's not a criticism; it's the right division of labor, and naming it
sharply is the point. The human picks the search space and the ruler (here, `prepare_jax.py`, frozen);
the loop searches that space exhaustively and honestly.

The reason that division works *here* and degrades in messier settings is the metric. `val_bpb` is a
single scalar, cheap to compute, hard to game, and tightly coupled to the thing you actually care
about (a better language model). When the objective is that clean, an autonomous optimization loop is
genuinely useful: it will out-patience a human at grid-searching the knobs and will faithfully record
the trade-offs. The failure modes of autonomous research tend to show up when the objective is *not*
clean — when "is this experiment novel?" or "did we learn something?" has no scalar answer, the loop
loses its grip and starts spinning. This project deliberately stays on the easy side of that line:
fixed protocol, frozen harness, one number. That constraint is what makes the results trustworthy, and
it's a useful template — if you want an autonomous loop you can believe, give it a metric it can't
argue with.

One more structural note: keeping the experiment branches and the kept/rejected log makes the run
*auditable* after the fact, which matters more than it sounds. You can go back and ask "why did we
reject the fused optimizer?" and find the OOM number, not just a vague memory. An optimization loop
that doesn't leave that trail is much harder to trust than one that does, because you can't tell a real
win from a measurement fluke without the receipts.

## Takeaways

- **MFU is a means, not the goal.** Within a fixed compute budget, the model that uses the chip
  hardest can be the worse model, because it sees fewer tokens. Optimize the metric you actually care
  about (val_bpb), and watch utilization climb on the *losing* configurations as a warning sign.
- **Sometimes the optimization is *not* fusing.** A `@jax.jit` boundary is also a memory-release
  boundary; fusing forward+backward+optimizer into one program can OOM where splitting it into three
  fits comfortably.
- **The simplicity criterion has teeth.** The single biggest win (batch 64) was a *deletion* — it
  removed the gradient-accumulation machinery, the host sync, and the grad averaging all at once.
  Smaller, simpler code that's also faster is the best kind of result, and a loop that rewards it
  finds it.

---

*Single-host TPU v4-8, JAX + Flax NNX. Numbers from `TPU_OPTIMIZATION_SUMMARY.md` and the run logs;
`prepare_jax.py` (the metric harness) held fixed throughout, `train_jax.py` the only mutable file.*
