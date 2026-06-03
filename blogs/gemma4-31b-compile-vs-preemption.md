---
title: "When a model runs fine but won't serve: compile time vs. the preemption window"
subtitle: "Gemma 4 31B decodes correctly on TPU in seconds, and never comes up as an API server on spot capacity"
date: 2026-05-31
tags: [tpu, jax, xla, inference, serving, maxtext]
status: partial   # decode validated; API server blocked
---

# When a model runs fine but won't serve: compile time vs. the preemption window

Some projects fail because the model is wrong. This one is the opposite, and more annoying: the
model is completely correct, decodes in seconds, sips HBM — and still cannot be stood up as an API
server, for a reason that has nothing to do with the model and everything to do with the
economics of the hardware it runs on.

## The part that works

**Gemma 4 31B** runs on MaxText on a TPU v6e-64. Direct decode through
`maxtext.inference.decode` produces correct output — "The capital of France is **Paris**." — with
a perfectly ordinary config: scanned checkpoint, `ici_fsdp_parallelism=64`, weighted sampling at
temperature 0.7, `max_target_length=256`. The model is almost embarrassingly light on the
hardware: **0.93 GB per chip**, about 3% of the 31.25 GB of HBM each v6e chip has. There is no
memory pressure, no correctness issue, no exotic kernel. As a decode workload it's a non-event.

## The part that doesn't

The goal was an OpenAI-compatible API server. The blocker shows up in one place:
`MaxTextGenerator.__init__()` takes **40+ minutes to compile**. XLA's ahead-of-time compilation of
the full serving graph is slow for this model, and 40 minutes is a very long time to ask a piece of
hardware to sit still.

Here's where the hardware economics bite. These TPUs are **preemptible** (spot) capacity — far
cheaper, and the only realistic way to run experiments like this, but reclaimable by the provider
at any time, often within minutes. So the failure mode isn't a crash; it's a race the server keeps
losing:

```
launch server -> begin 40-min compile -> TPU preempted at ~minute X (X < 40) -> start over
```

The compile never finishes inside a preemption window, so the server never reaches the point where
it can accept a request. The exact same model, invoked through the *decode* path, is fine — because
decode's compile is short enough to complete before the slice gets reclaimed. The serving graph's
40-minute compile is the entire difference between "works" and "doesn't."

This connects to a lever from a sibling project: being able to **compile once and serialize the
executable** (PJRT supports exactly this) would turn a 40-minute cold start into a near-instant
reload — which is precisely the kind of thing that makes the difference on preemptible hardware.

## The version trap, for good measure

There's a second, smaller snag that's worth recording because it's the kind of thing that costs an
afternoon. Gemma 4 needs **two different `transformers` versions** at two different stages:

- **`transformers >= 5.0`** to *convert* the checkpoint — older versions don't know the `gemma4`
  model type.
- **`transformers == 4.57.3`** to *serve* — version 5.x has a tokenizer bug around an
  `extra_special_tokens` field that breaks loading.

So conversion and serving can't share an environment, and the cached `tokenizer_config.json` has to
be patched to strip the offending field before serving. None of this is hard; all of it is the sort
of friction that turns "deploy a working model" into a multi-day exercise.

## The lesson

The transferable point is that **compile time is a first-class deployment constraint on spot
hardware**, not an implementation detail you can wave away as "it's slow the first time." When your
compute can vanish in minutes, a 40-minute startup is not slow — it's *fatal*, regardless of how
fast or correct the steady state is. The fixes are all about closing that gap:

- run the server on **on-demand (non-preemptible)** capacity, where a long compile can finish;
- **persist a compiled executable** and reload it, so the cold start happens once, ever;
- or move to a serving stack (e.g. JetStream) whose compile fits inside a preemption window.

It's a useful counterweight to the usual "does the model run?" framing. Here the model runs
beautifully. Whether you can *serve* it is a completely separate question, and the answer was set by
a compile-time-versus-preemption-window inequality that no amount of model-side work would change.

---

*Gemma 4 31B on MaxText (`feature/gemma4-31b-inference`). Decode validated on v6e-64; API server
blocked on a 40+ minute compile that outlasts the preemption window.*
