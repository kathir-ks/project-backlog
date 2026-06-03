---
title: "Porting an autonomous training-optimization loop to TPU"
subtitle: "How a not-fusing-two-jit-calls decision and a bigger batch moved val_bpb from 1.136 to 1.115"
date: 2026-05-31
tags: [jax, flax-nnx, tpu, training, performance, autonomous-research]
status: validated   # documented optimization with kept/rejected experiments
---

# Porting an autonomous training-optimization loop to TPU

The premise is Karpathy's autoresearch loop: an agent is handed a training script, allowed to
edit it, runs a fixed-budget experiment (here, a 5-minute training run), reads a single metric
(`val_bpb`, bits-per-byte on a held-out set), and keeps or discards the change — with an explicit
*simplicity criterion*, so a tiny win that deletes code beats a tiny win that adds a hack. This
project ports that loop to a single-host **TPU v4-8** (4 chips, 32 GB HBM each) with JAX and Flax
NNX, and the interesting content is the TPU-specific optimization decisions the loop converged on.

The protocol keeps two files immutable so the experiment is honest: `prepare_jax.py` owns the
data, the tokenizer, and the `evaluate_bpb` / time-budget harness — it is the ground-truth metric
and is never touched. `train_jax.py` is the only mutable file. Everything below is a change to
that one file, measured against that fixed harness.

## The headline: a smaller knob beat a bigger model

The optimized configuration improved on the baseline across the board within the same 5-minute
budget:

| metric | baseline | optimized | Δ |
|---|---:|---:|---:|
| val_bpb | 1.136 | **1.115** | −0.021 |
| MFU | 8.71% | **9.80%** | +12.5% |
| throughput | ~400k tok/s | **~440k tok/s** | +10% |

And a model-size sweep (all within the fixed budget) showed the depth that wins isn't the biggest:

| depth | batch | params | val_bpb | MFU |
|---:|---:|---:|---:|---:|
| **8** | 64 | 50.3M | **1.115** | 9.8% |
| 10 | 64 | 85.9M | 1.143 | 12.4% |
| 12 | 32 | 135.3M | 1.232 | 14.3% |

Bigger models hit *higher* hardware utilization (more FLOPs to fill the chip) but *worse* bits-
per-byte, because in a fixed time budget they see fewer tokens and don't converge as far. Around
50M parameters is the sweet spot for this budget — a nice, concrete demonstration that MFU is not
the objective; it's a means.

## Three TPU-port decisions

**1. A bigger batch deleted gradient accumulation entirely.** Raising `DEVICE_BATCH_SIZE` from 32
to 64 let the full batch fit in one step, which removed the whole micro-step accumulation loop —
and with it the mid-step `float(loss)` host sync and the tree-map gradient averaging. This is the
simplicity criterion working as intended: the change made the code *smaller* and faster at once.

**2. To afford that batch, gradient-checkpoint the block.** A batch-64 forward pass produces
attention score tensors of shape `f32[64, 4, 2048, 2048]` — about 4 GB *each*. They don't all fit.
Flax NNX's `@nnx.remat` recomputes them in the backward pass instead of stashing them:

```python
@nnx.remat
def __call__(self, x, ve, cos_sin, mask):
    x = x + self.attn(norm(x), ve, cos_sin, mask)
    x = x + self.mlp(norm(x))
    return x
```

**3. The counterintuitive one: do *not* fuse the optimizer into the training step.** The obvious
"optimization" is to compile forward + backward + optimizer update as a single `@jax.jit`
function. On TPU it OOM'd — and the reason is the genuinely useful insight:

> Combining `compute_grads` and `apply_optimizer` into a single `@jax.jit` function caused OOM
> (31.78G program memory vs 30.75G available). XLA could not free intermediate activations when
> the entire forward+backward+optimizer was a single compiled program. Keeping separate JIT calls
> acts as a natural memory checkpoint boundary.

The JIT boundary between the gradient computation and the optimizer step lets XLA release the
backward-pass intermediates before the optimizer allocates its state. Fuse them and XLA has to
keep everything live simultaneously, blowing past the 30.75 GB ceiling. So the layout is three
separate jitted functions, and the "missing" fusion is load-bearing.

A smaller related win: data goes straight to the device with `jax.device_put(x_np, sharding)`,
skipping a redundant `jnp.array` host copy.

```python
x_np, y_np, epoch = next(train_loader)
x = jax.device_put(x_np, data_sharding)   # numpy -> device, no extra host copy
y = jax.device_put(y_np, data_sharding)
```

## What the loop produced

The run logs are the artifact: seven timestamped experiments with kept-and-rejected changes, the
depth sweep above, and a `progress.png` tracking val_bpb over time. The final logged run (a
DEPTH=10 experiment) ends at `val_bpb 1.143226`, `peak_hbm_mb 2978.5`, `mfu_percent 12.41` — fully
consistent with the sweep, and a good reminder that the *best* run (depth 8, 1.115) isn't always
the *last* run; the loop explores past the optimum and you read the trajectory, not the endpoint.

## Takeaways

- **MFU is a means, not the goal.** Within a fixed compute budget, the model that uses the chip
  hardest can still be the worse model. Optimize the metric you actually care about.
- **Sometimes the optimization is *not* fusing.** A `@jax.jit` boundary is also a memory-release
  boundary; fusing everything into one compiled program can OOM where splitting it fits.
- **The simplicity criterion has teeth.** The single biggest win (batch 64) was a *deletion* —
  it removed the gradient-accumulation machinery rather than adding anything.

---

*Single-host TPU v4-8, JAX + Flax NNX. Numbers from `TPU_OPTIMIZATION_SUMMARY.md` and the run
logs; `prepare_jax.py` (the metric harness) held fixed throughout.*
