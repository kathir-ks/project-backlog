---
title: "94.6% explained variance, then −288%: training SAEs that don't quietly die"
subtitle: "Sparse autoencoders on 2.8B tokens of TPU activations, and the one flag that decided success"
date: 2026-05-31
tags: [interpretability, sae, jax, tpu, mechanistic-interpretability]
status: shipped   # v2 validated; v3/v4 trained, eval pending
---

# 94.6% explained variance, then −288%: training SAEs that don't quietly die

Sparse autoencoders (SAEs) are the workhorse of modern mechanistic interpretability: train an
overcomplete dictionary to reconstruct a model's residual stream under a sparsity constraint, and —
if it works — the dictionary atoms turn out to be human-interpretable "features." The "if it works"
is doing a lot of load-bearing. This post is about a JAX/TPU SAE trainer, the 94.6%
explained-variance dictionary it produced on a 2.8-billion-token corpus, and the run that collapsed
to **−288%** explained variance and what fixed it.

## Four architectures behind one base

The library implements four SAE variants behind a single `BaseSAE`. The base is just a tied
encode/decode with a learned decoder bias used to *center* the input before encoding — a small detail
that matters because SAE features are most interpretable when they explain deviations from the mean
activation, not the mean itself:

```python
def __call__(self, x):
    x_centered = x - self.b_dec
    z_pre = x_centered @ self.W_enc + self.b_enc
    z, aux = self.apply_sparsity(z_pre, x_centered=x_centered)
    x_hat = self.decode(z)
    return x_hat, z, aux
```

The variants differ only in `apply_sparsity` and the loss term:

- **Vanilla** — ReLU plus an L1 penalty on the codes. The original SAE recipe. Simple, but the L1
  coefficient is a finicky knob and L1 biases feature magnitudes downward.
- **TopK** — keep exactly `k` active features per example, sparsity by construction, no penalty to
  tune.
- **Gated** (DeepMind, 2024) — split the encoder into a *gate* path (which features fire) and a
  *magnitude* path (how much), decoupling the two so the L1 penalty doesn't shrink magnitudes.
- **JumpReLU** (Anthropic, 2024) — a learnable per-feature hard threshold, `z * (z > θ)`.

The two most interesting to implement are TopK and JumpReLU.

**TopK** is appealingly direct: take the top `k` pre-activations, ReLU them, zero everything else.

```python
def apply_sparsity(self, z_pre, *, x_centered=None):
    k = self.config.k
    topk_values, topk_indices = jax.lax.top_k(z_pre, k)
    z = jnp.zeros_like(z_pre)
    batch_idx = jnp.arange(z_pre.shape[0])[:, None]
    z = z.at[batch_idx, topk_indices].set(nn.relu(topk_values))
    return z, {"z_pre": z_pre, "topk_indices": topk_indices}
```

TopK's failure mode is **dead features**: a feature that never makes the top-k stops receiving
gradient and stays dead forever, wasting dictionary capacity. The standard remedy is an auxiliary
loss that reconstructs through *all* ReLU'd activations — not just the top-k — to feed gradient back
to the inactive features:

```python
def topk_auxiliary_loss(z_pre, x, W_dec, b_dec, k, coeff=1.0/32):
    z_aux = jax.nn.relu(z_pre)
    x_hat_aux = z_aux @ W_dec + b_dec
    return coeff * jnp.mean((x - x_hat_aux) ** 2)
```

**JumpReLU** is the delicate one. A hard threshold has zero gradient almost everywhere, so the
threshold itself can't learn by ordinary backprop — the gradient signal that should move θ doesn't
exist. The fix is a `custom_jvp` straight-through estimator that approximates the derivative of the
step function with a rectangular kernel of width `bandwidth` centered on the threshold:

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

That `delta_approx` is the whole trick: at the threshold boundary it injects a finite, large gradient
so the threshold can move, and away from the boundary it's zero. It's a soft handle on a hard switch.

## The data path is the actual scaling problem

SAE work is bottlenecked by *data* volume, not model size — the encoder/decoder is tiny (here, 896
input dims to a few thousand features), but it needs to see billions of activation vectors to learn a
good dictionary. So the data plumbing is where the engineering goes. Activation sources are pluggable
behind an `ActivationSource` interface: gzip-pickle shards (the format a separate extraction pipeline
emits, sharded across hosts), raw numpy, zero-copy safetensors, or a HuggingFace dataset. Whatever
the source, it feeds an in-memory shuffle buffer (256k vectors ≈ ~915 MB) that decorrelates
consecutive activations — important, because activations from the same document are highly correlated
and training on them in order biases the dictionary.

Training is data-parallel across a TPU pod: the SAE params (~50 MB) are replicated on every chip, the
activation batch is sharded along the data axis, and the global array is assembled with
`make_array_from_single_device_arrays`. The `train_step` is JIT'd with `donate_argnums` so the
optimizer can update parameters in place, and it normalizes the decoder columns each step (a standard
SAE trick so feature magnitudes are carried by the codes, not absorbed into the decoder). Learning
rate supports cosine/linear/constant schedules with warmup and gradient clipping, and the whole run
checkpoints to GCS so a preemption doesn't cost more than the last checkpoint interval. Evaluation
reports the metrics that actually tell you whether the dictionary is any good: reconstruction MSE,
explained variance, L0 (average active features), dead-neuron fraction, and feature density.

## The corpus

The training corpus is 2.8-billion-tokens' worth of residual-stream activations extracted from a
Qwen-2.5 model at layer 19 (a separate extraction pipeline produced them; raw FSDP halves were merged
to 896-dim across host shards). Layer 19 is deep enough to carry rich, composed features but not so
deep that everything has collapsed toward the output distribution — a reasonable place to look for
interpretable structure.

## The collapse, and the one-line fix

The first serious run, **v1** (TopK, 8× overcomplete → 7,168 features, k=32, resampling every 25k
steps with no cutoff), looked great and then detonated:

- step 125K: MSE 0.2384, **91.6% explained variance** — a strong dictionary.
- step 200K: MSE 11.0354, **−288% explained variance** — worse than predicting the mean.

Negative explained variance is not a metric bug; it means the reconstruction became *anti-correlated*
with the input. The cause was the dead-neuron resampler. Resampling re-initializes dead features from
high-loss examples, which is useful early in training when there's a lot left to learn. But it keeps
firing late in training, at a low learning rate, where each re-initialization is a large, unschooled
perturbation injected into an otherwise-converged model. Past a certain point the resampler is just
noise, and enough of it tips the whole dictionary over a cliff. The resampling body makes the
magnitude of the perturbation concrete — it overwrites dead directions with the *highest-loss*
examples in the batch:

```python
def _resample_dead_neurons(self, batch):
    window = self.train_config.dead_neuron_window
    dead_info = compute_dead_neurons(self.state.dead_neuron_steps, window)
    if dead_info["dead_count"] == 0:
        return
    dead_mask = self.state.dead_neuron_steps >= window
    x_hat, z, _ = self.model.apply({"params": self.state.params}, batch)
    per_sample_loss = jnp.mean((batch - x_hat) ** 2, axis=-1)
    num_dead = int(dead_mask.sum())
    top_indices = jnp.argsort(per_sample_loss)[-num_dead:]   # highest-loss examples
    new_directions = batch[top_indices]
    # ... overwrite the dead feature rows with these directions ...
```

The fix is embarrassingly small: **stop resampling once the model has mostly converged.** Rather than
a launcher hack, it became a first-class config flag with a clean default (0 = never stop, preserving
old behavior):

```python
# configs/training.py
dead_neuron_resample_steps: int = 25_000
dead_neuron_window: int = 10_000
dead_neuron_resample_until: int = 0   # stop resampling after this step (0 = never stop)

# trainer.py
if (
    cfg.dead_neuron_resample_steps > 0
    and step % cfg.dead_neuron_resample_steps == 0
    and step > 0
    and (cfg.dead_neuron_resample_until == 0 or step <= cfg.dead_neuron_resample_until)
):
    self._resample_dead_neurons(batch)
```

**v2** is v1 with `dead_neuron_resample_until = 75000` and nothing else changed. Full evaluation at
step 125K on **24,064,000 tokens**:

- **MSE 0.1254, explained variance 94.6%, L0 = 32.0** (0.45% sparsity)
- **6,744 / 7,168 features alive** (5.9% dead)

Same architecture, same data, same everything — the difference between a 94.6% dictionary and a −288%
pile of rubble was *when to stop resampling*. That's the kind of result that doesn't make it into
papers but eats weeks of everyone's time, which is exactly why it's worth writing down.

## What the features turned out to be

A dictionary is only interesting if its atoms mean something. Probing the v2 features against the
activation set — measuring, for each feature, how often it fires and on what — produced a clean
tiering:

- **20 universal features** that fire on 80%+ of samples, the top handful at 99.9%. Against this
  corpus (ARC-style grid token streams) these look like grid-*syntax* primitives — the structural
  scaffolding present in essentially every input.
- **252 task-discriminative features** of medium selectivity — the ones that distinguish one kind of
  problem from another.
- **69 highly selective features** firing on only a handful of samples each — candidate
  special-case detectors.
- **27 sample clusters** recovered from the discriminative features.

The shape of that distribution — a small set of near-universal syntactic features plus a long tail of
selective ones — is exactly what you hope to see in an interpretable dictionary, and it's the
substrate for the cross-architecture feature-matching a larger interpretability study wants to do:
if two different model architectures both learn the same "grid-syntax" features, that's evidence for
universal representations; if they don't, that's evidence against.

## Tested, because SAEs lie quietly

The trainer is backed by 147 tests across 8 files (losses, metrics, data, models, training, configs,
resume/streaming, code-fixes). That isn't test-coverage theater. The default outcome of an SAE run is
"it silently learned something useless," and the v1 collapse is the proof — a run that hit 91.6% and
then quietly became garbage would, evaluated at the wrong checkpoint, look like a success. Tests on
the loss math, the metrics, and the data pipeline are the difference between trusting a number and
being fooled by one. v3 (a 16× dictionary → 14,336 features) and v4 (JumpReLU) are trained and
awaiting evaluation; the harness is what makes comparing them meaningful.

## Takeaways

- **Negative explained variance is a real, reachable failure mode**, not a metric bug. Watch it
  across the *whole* run, not just at the checkpoint you happened to evaluate — a mid-run peak can
  hide a late-run collapse.
- **Dead-neuron resampling is a phase-dependent tool**: helpful early, destructive late. Gate it on a
  step cutoff rather than running it for the whole schedule.
- **The data path is the SAE scaling problem.** The model is tiny; the billions of decorrelated
  activation vectors are the work. Shuffle buffers, pluggable sources, and preemption-safe
  checkpointing are where the engineering actually lives.
- The interesting interpretability result (the universal-vs-selective feature tiering) was only
  reachable *after* the boring training-stability result. Most of SAE work is the boring part, and
  the boring part is where the failures hide.

---

*v2 SAE evaluated on 24.06M tokens, layer-19 / 896-dim Qwen-2.5 activations. Four architectures
(Vanilla/TopK/Gated/JumpReLU); 147 tests across 8 files. v3 (16× dict) and v4 (JumpReLU) trained,
evaluation pending.*
