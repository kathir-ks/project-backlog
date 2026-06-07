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
economics of the hardware it runs on. It's a clean case study in a constraint that doesn't show up
in any benchmark table: **how long your accelerator will hold still.**

## The part that works

**Gemma 4 31B** runs on MaxText on a TPU v6e-64. Direct decode through
`maxtext.inference.decode` produces correct output — "The capital of France is **Paris**." — with
a perfectly ordinary config: a scanned checkpoint, `ici_fsdp_parallelism=64`, the HuggingFace
tokenizer with the chat template enabled, weighted sampling at temperature 0.7,
`max_target_length=256`. Nothing exotic. The model is almost embarrassingly light on the
hardware: **0.93 GB per chip**, about 3% of the 31.25 GB of HBM each v6e chip carries. There is no
memory pressure, no correctness issue, no custom kernel. As a decode workload it's a non-event,
which is exactly the point — the hard part is everywhere except the model.

Getting to that "non-event" did take some checkpoint plumbing worth recording, because it's
representative of what bringing up any new model architecture on MaxText actually involves. Gemma 4
is a new enough model type that the checkpoint had to be converted from the HuggingFace format into
MaxText's layout, and the conversion was done **two ways**: a *scanned* checkpoint (all layers
stacked along a leading axis, which keeps the compiled program small because the layer loop is a
single `scan`), and an *unscanned* one (each layer materialized separately, which compiles to a
larger program but can be friendlier for some serving paths). The scanned variant is what the
decode path above uses. Both produce identical output; they're just different trade-offs between
compile time and graph shape — a distinction that, as we'll see, turns out to be the whole story.

## The part that doesn't

The goal was an OpenAI-compatible API server, so the model could be queried like any hosted LLM.
The blocker shows up in exactly one place: `MaxTextGenerator.__init__()` takes **40+ minutes to
compile**. XLA's ahead-of-time compilation of the full serving graph — prefill plus the
autoregressive decode loop plus the sampling machinery, all fused into one lowering — is slow for
this model, and 40 minutes is a very long time to ask a piece of hardware to sit still.

Here's where the hardware economics bite, and bite hard. These TPUs are **preemptible** (spot)
capacity. That's not a detail; it's the entire operating environment. Preemptible TPU is multiples
cheaper than on-demand, and for a self-funded research fleet it is frequently the *only* way to get
a v6e-64 at all — on-demand capacity for a 64-chip slice is both expensive and, during the capacity
crunches that recur on these accelerators, simply unavailable. The catch is in the name: the
provider can reclaim the slice at any moment, often within minutes of granting it.

So the failure mode isn't a crash. It's a race the server keeps losing:

```
launch server -> begin 40-min compile -> TPU preempted at ~minute X (X < 40) -> start over
```

The compile never finishes inside a preemption window, so the server never reaches the point where
it can accept its first request. Run the loop a dozen times and you've burned hours of TPU time and
have nothing serving. The exact same model, invoked through the *decode* path, is fine — because
decode's compile is short enough to complete before the slice gets reclaimed. The serving graph's
40-minute compile is the entire difference between "works" and "doesn't," on identical hardware,
with an identical, correct model.

It's worth dwelling on *why* serving compiles so much more slowly than decode. The decode CLI
compiles a relatively tight graph: load params, run a fixed number of generate steps, done. The API
server's generator compiles the full continuous-batching serving surface — prefill at multiple
sequence lengths, the decode step, the KV-cache update, and the sampler — and it does so
ahead-of-time so that request latency is low once it's up. That AOT thoroughness is exactly what you
want in steady state and exactly what kills you on preemptible hardware: you pay the entire compile
cost up front, in one indivisible 40-minute block, with no partial progress that survives a
preemption.

## The deeper fix, and why it connects to bare-metal PJRT

The honest fixes all aim at the same target — close the gap between "compile started" and "ready to
serve" so it fits inside a preemption window:

- **Run the server on on-demand (non-preemptible) capacity.** A long compile can finish when nobody
  is going to yank the slice out from under it. This is the boring, correct answer, gated only on
  cost and availability.
- **Persist the compiled executable and reload it.** This is the interesting one. XLA can serialize
  a compiled program to disk; PJRT exposes exactly this (compile once, save, deserialize-and-run
  later). A sibling project drove the PJRT C API directly from C and demonstrated the
  serialize→disk→reload→rerun path end to end. Applied here, it would turn a 40-minute *cold* start
  into a one-time cost paid once, ever — every subsequent launch reloads the cached executable in
  seconds, which is exactly what makes serving viable on ephemeral hardware. The compile stops being
  a per-launch tax and becomes a build artifact.
- **Move to a serving stack whose compile fits the window.** JetStream, for instance, is built for
  TPU serving and may lower a graph that compiles fast enough to come up before a preemption — or
  may be combinable with executable caching for the same effect.

The thread running through all three: a 40-minute compile is not a fixed law of nature, it's an
artifact of *when and how often* you pay it. Pay it once and cache it, or pay it somewhere it can
finish, and the model that already decodes correctly becomes a model you can serve.

## The version trap, for good measure

There's a second, smaller snag worth recording because it's the kind of thing that quietly costs an
afternoon. Gemma 4 needs **two different `transformers` versions** at two different stages of the
pipeline:

- **`transformers >= 5.0`** to *convert* the checkpoint — older releases don't know the `gemma4`
  model type at all, so conversion fails before it starts.
- **`transformers == 4.57.3`** to *serve* — version 5.x ships a tokenizer bug around an
  `extra_special_tokens` field that breaks loading at serve time.

So conversion and serving cannot share a Python environment, and on top of that the cached
`tokenizer_config.json` has to be patched to strip the offending `extra_special_tokens` field before
serving will load it. None of this is conceptually hard; all of it is the sort of friction that
turns "deploy a working model" into a multi-day exercise of pinning, patching, and re-pinning. It's
also a reminder that "the model works" and "the toolchain around the model is coherent" are
different claims — a fast-moving library ecosystem means the second one needs its own attention.

## Where compile time fits in a model bring-up

It's worth zooming out, because this project is a clean illustration of what "bring up a model on
TPU" actually decomposes into — and how lopsided the difficulty distribution is. A rough checklist for
getting a new architecture serving on MaxText looks like this:

1. **Convert the checkpoint** into MaxText's layout (and decide scanned vs. unscanned). Bounded work,
   but version-sensitive — as the `transformers` trap below shows.
2. **Get decode correct.** Match the reference output. For Gemma 4 this was clean: correct on the
   first serious try, with the chat template and sampling configured.
3. **Make it fit.** Often the hard part for big MoEs; here it was a non-issue at 0.93 GB/chip.
4. **Make it fast per token.** The throughput-ceiling work that dominates other bring-ups; not even
   reached here, because —
5. **Make it *start*.** Get the serving graph compiled and resident before you can field a request.

Gemma 4 31B sails through 1–4 and runs aground on 5. That's the unusual and instructive part: the
steps everyone budgets for (fit, speed, correctness) were free, and the step nobody puts on the chart
(time-to-first-request) was fatal. Most model bring-ups never *notice* compile time as a distinct
constraint because their compile happens to fit inside whatever window they have. Push the model
large enough, or the hardware ephemeral enough, and compile time graduates from "annoying first-run
latency" to "the thing that decides whether the project ships." A v6e-64 serving graph for a 31B model
is right at that boundary; a smaller model on the same slice would compile faster and never surface the
problem, while a larger one would make it worse. The lesson generalizes: as models and slices both
grow, expect compile-time-versus-availability to migrate up everyone's list of real constraints.

## The lesson

The transferable point is that **compile time is a first-class deployment constraint on spot
hardware**, not an implementation detail you can wave away as "it's slow the first time." When your
compute can vanish in minutes, a 40-minute startup is not slow — it's *fatal*, regardless of how
fast or correct the steady state is. The usual mental model of model deployment is "does it fit, is
it correct, is it fast per token." Gemma 4 31B passes all three with room to spare and still can't be
served, because there's a fourth axis nobody puts on the chart: **does the time-to-first-request fit
inside the lifetime of the hardware you can afford.**

That reframing is the actual deliverable here. The model is a solved problem. The interesting
engineering is entirely in the deployment envelope:

- run the server where a long compile can finish (on-demand), or
- make the long compile a one-time artifact (serialize the executable, reload it in seconds), or
- pick a serving stack whose compile already fits the window (JetStream).

It's a useful counterweight to the "does the model run?" framing that dominates model bring-up.
Here the model runs beautifully. Whether you can *serve* it was decided by a compile-time-versus-
preemption-window inequality that no amount of model-side work would ever change — and the path
forward is to attack the inequality, not the model.

---

*Gemma 4 31B on MaxText (`feature/gemma4-31b-inference`). Decode validated on v6e-64 (0.93 GB/chip,
correct output); API server blocked on a 40+ minute AOT compile that outlasts the preemption window.
Scanned and unscanned checkpoints both produced; serving fix paths: on-demand capacity, serialized
executables, or JetStream.*
