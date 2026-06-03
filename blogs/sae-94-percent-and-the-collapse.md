---
title: "94.6% explained variance, then −288%: training SAEs that don't quietly die"
subtitle: "Sparse autoencoders on 2.8B tokens of TPU activations, and the one flag that decided success"
date: 2026-05-31
tags: [interpretability, sae, jax, tpu, mechanistic-interpretability]
status: shipped   # v2 validated; v3/v4 trained, eval pending
---

# 94.6% explained variance, then −288%: training SAEs that don't quietly die

Sparse autoencoders (SAEs) are the workhorse of modern mechanistic interpretability: train an
overcomplete dictionary to reconstruct a model's residual stream under a sparsity penalty, and —
if it works — the dictionary atoms turn out to be human-interpretable "features." The "if it
works" is doing a lot of load-bearing. This post is about a JAX/TPU SAE trainer, the 94.6%
explained-variance dictionary it produced on a 2.8-billion-token corpus, and the run that
collapsed to **−288%** explained variance and what fixed it.

## The trainer

The library implements four SAE variants behind one `BaseSAE`, which is just a tied
encode/decode with a learned decoder bias used to center the input:

```python
def __call__(self, x):
    x_centered = x - self.b_dec
    z_pre = x_centered @ self.W_enc + self.b_enc
    z, aux = self.apply_sparsity(z_pre, x_centered=x_centered)
    x_hat = self.decode(z)
    return x_hat, z, aux
```

The variants differ only in `apply_sparsity` and the loss: **Vanilla** (ReLU + L1), **TopK**,
**Gated** (DeepMind 2024), and **JumpReLU** (Anthropic 2024). The two interesting ones are TopK
and JumpReLU.

**TopK** keeps exactly `k` features active per example — sparsity by construction, no penalty to
tune:

```python
def apply_sparsity(self, z_pre, *, x_centered=None):
    k = self.config.k
    topk_values, topk_indices = jax.lax.top_k(z_pre, k)
    z = jnp.zeros_like(z_pre)
    batch_idx = jnp.arange(z_pre.shape[0])[:, None]
    z = z.at[batch_idx, topk_indices].set(nn.relu(topk_values))
    return z, {"z_pre": z_pre, "topk_indices": topk_indices}
```

TopK's failure mode is **dead features**: a feature that never makes the top-k stops getting
gradient and stays dead forever. The standard remedy is an auxiliary loss that reconstructs
through *all* ReLU'd activations (not just the top-k), feeding gradient back to the inactive
ones:

```python
def topk_auxiliary_loss(z_pre, x, W_dec, b_dec, k, coeff=1.0/32):
    z_aux = jax.nn.relu(z_pre)
    x_hat_aux = z_aux @ W_dec + b_dec
    return coeff * jnp.mean((x - x_hat_aux) ** 2)
```

**JumpReLU** is the more delicate one: a hard threshold per feature, `z * (z > θ)`. A hard
threshold has zero gradient almost everywhere, so the threshold can't learn by ordinary
backprop. The fix is a `custom_jvp` straight-through estimator that approximates the derivative
of the step with a rectangular kernel of width `bandwidth` at the boundary:

```python
@jax.custom_jvp
def jumprelu(z_pre, threshold, bandwidth=0.001):
    return jnp.where(z_pre > threshold, z_pre, 0.0)

@jumprelu.defjvp
def jumprelu_jvp(primals, tangents):
    z_pre, threshold, bandwidth = primals
    dz, dthreshold, _ = tangents
    active = (z_pre > threshold).astype(z_pre.dtype)
    primal_out = jumprelu(z_pre, threshold, bandwidth)
    # rectangular-kernel approximation of the Dirac delta at the boundary
    delta_approx = jnp.where(jnp.abs(z_pre - threshold) < bandwidth / 2, 1.0 / bandwidth, 0.0)
    tangent_out = active * dz - delta_approx * z_pre * dthreshold
    return primal_out, tangent_out
```

The whole thing trains data-parallel across a TPU pod, with GCS checkpointing for preemption
recovery and pluggable data sources (the gzip-pickle activation shards, numpy, safetensors,
HuggingFace datasets) feeding a shuffle buffer.

## The data

The training corpus is 2.8-billion-token's worth of residual-stream activations extracted from
a Qwen-2.5 model at layer 19 (a separate extraction pipeline produced them, sharded and merged to
896-dim). SAE work is bottlenecked by *data* volume, not model size — the encoder/decoder is
tiny — so a multi-billion-token corpus is the point.

## The collapse, and the one-line fix

The first serious run, **v1** (TopK, 8× overcomplete → 7,168 features, k=32), looked great and
then detonated:

- step 125K: MSE 0.2384, **91.6% explained variance** — a strong dictionary.
- step 200K: MSE 11.0354, **−288% explained variance** — worse than predicting the mean.

Negative explained variance means the reconstruction is actively *anti-correlated* with the
input. The cause was the dead-neuron resampler. Resampling re-initializes dead features from
high-loss examples — useful early, when there's lots to learn — but it keeps firing late in
training at a low learning rate, where each re-init is a large, unschooled perturbation injected
into an otherwise-converged model. Past a certain point the resampler is just noise, and enough
of it tips the whole dictionary over.

The fix is embarrassingly small: **stop resampling once the model has mostly converged.** That
became a first-class config flag rather than a launcher hack:

```python
if (
    cfg.dead_neuron_resample_steps > 0
    and step % cfg.dead_neuron_resample_steps == 0
    and step > 0
    and (cfg.dead_neuron_resample_until == 0 or step <= cfg.dead_neuron_resample_until)
):
    self._resample_dead_neurons(batch)
```

**v2** is v1 with `dead_neuron_resample_until = 75000` and nothing else changed. Full evaluation
at step 125K on 24,064,000 tokens:

- **MSE 0.1254, explained variance 94.6%, L0 = 32.0** (0.45% sparsity)
- **6,744 / 7,168 features alive** (5.9% dead)

Same architecture, same data, same everything — the difference between a 94.6% dictionary and a
−288% pile of rubble was *when to stop resampling*.

## What the features turned out to be

A dictionary is only interesting if its atoms mean something. Probing the v2 features against the
activation set produced a clean tiering:

- **20 universal features** that fire on 80%+ of samples — the top handful at 99.9%. These look
  like grid-syntax primitives (the input here is ARC-style grid token streams).
- **252 task-discriminative features** of medium selectivity.
- **69 highly selective features** firing on only a handful of samples each.
- **27 sample clusters** recovered from the discriminative features.

The shape of that distribution — a small set of near-universal syntactic features plus a long
tail of selective ones — is exactly what you hope to see, and it's the substrate for the
cross-architecture feature-matching that a larger interpretability study wants to do.

The trainer is backed by 147 tests across 8 files (losses, metrics, data, models, training,
configs, resume/streaming), because "the SAE silently learned garbage" is the default outcome and
tests are the only thing standing between you and believing a −288% run was fine.

## Takeaways

- **Negative explained variance is a real, reachable failure mode**, not a bug in your metric.
  Watch it across the *whole* run, not just at the checkpoint you happened to evaluate.
- **Dead-neuron resampling is a phase-dependent tool**: helpful early, destructive late. Gate it
  on a step cutoff.
- The interesting interpretability result (the universal vs. selective feature tiering) was only
  reachable *after* the boring training-stability result. Most of SAE work is the boring part.

---

*v2 SAE evaluated on 24.06M tokens, layer-19 / 896-dim Qwen-2.5 activations. v3 (16× dict) and v4
(JumpReLU) are trained, evaluation pending.*
