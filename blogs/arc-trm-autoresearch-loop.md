---
title: "37,292 experiments, 597 ideas: what an autonomous research loop actually does"
subtitle: "Probing a Tiny Recursive Model on ARC — and watching an LLM-driven research loop run out of questions"
date: 2026-05-31
tags: [interpretability, arc-agi, autonomous-research, llm-agents, trm]
status: real-result   # 37k deterministic runs; honest meta-finding included
---

# 37,292 experiments, 597 ideas: what an autonomous research loop actually does

This started as a mechanistic-interpretability study of a **Tiny Recursive Model (TRM)** —
a 7M-parameter network that solves ARC-AGI puzzles by applying one small transformer block
over and over, refining an answer-state and a latent "scratchpad" across ~16 recursive
iterations. The question: *what does TRM compute, iteration by iteration?*

To explore it, the project wired up an **autonomous research loop**: an LLM proposes
experiments, deterministic Python kernels run them against the real model, results accumulate,
and the LLM reads the recent results to propose the next batch. It ran **37,292 experiments**
across all 200 canonical ARC tasks. It found a genuine result. It also did something more
interesting and more cautionary, which is the real subject of this post.

## The setup

The model is a real checkpoint (`Sanjin2024/TinyRecursiveModels-ARC-AGI-1`, step 155718) run on
CPU — at 7M parameters, a TPU is pure overhead — over a concept-augmented ARC dataset (960 task
groups, ~876k puzzle identifiers). Three deterministic experiment kernels do the actual probing,
each emitting a small structured record:

- **convergence** — per iteration, how much the argmax answer changed (Hamming), softmax
  entropy, and whether/when it became correct. Emits `first_correct_step` and a `slow_convergence`
  flag.
- **task-ID ablation** — run the real `puzzle_id` vs. a blanked one, measure per-step KL between
  the two logit streams. Emits `peak_kl_step` and `peak_kl_value`.
- **halting** — track `sigmoid(q_halt_logits)` per step. Emits `first_would_halt_step` (first
  step the model would choose to stop, or −1 if never).

```python
# task-ID ablation kernel — the core of it
kls = [float(kl_div_per_position(real[i]["logits"], blank[i]["logits"]).mean()) for i in range(n)]
record = {
    "puzzle_id": pid, "n_steps": n,
    "peak_kl_step":  int(max(range(n), key=lambda i: kls[i])),
    "peak_kl_value": round(max(kls), 6),
}
```

The proposer is Claude, called headless with a hard budget:

```python
["claude", "--model", "claude-sonnet-4-6", "--output-format", "json",
 "--max-budget-usd", "0.15", "-p", prompt]   # ~$0.02 warm, ~$0.10 cold
```

Crucially, the prompt only ever shows the proposer **the last 12 results**. Hold that thought.

## The real finding: task-ID matters, halting and correctness don't follow

Across all 200 tasks, a clean **three-way dissociation** falls out. The model is strongly
sensitive to the task-ID token — but that sensitivity is decoupled from both knowing when to
stop and getting the answer right. Task `2bcee788` is the archetype:

- **peak KL = 132.09 at step 0** — blanking the task-ID hugely changes the very first
  iteration's logits. The model reads the task identity immediately and hard.
- **`first_would_halt_step = -1`** — across all iterations, it never crosses the halt threshold.
- **`first_correct_step = -1`** — and it never produces the correct grid.

So: *immediate, strong task-ID routing* + *never decides to stop* + *never correct*. Whatever
the task-ID token is doing at step 0, it is not setting up a trajectory that converges to a
correct, confidently-halted answer. And this isn't one cherry-picked task: in aggregate,
**107 of 200 tasks are never solved at any iteration**. The task-ID signal is real and
early; the competence it's supposed to enable often simply isn't there.

That is a publishable mechanistic nugget — a concrete dissociation between a model's internal
"I know which task this is" signal and its actual problem-solving, surfaced by brute-force
deterministic probing over the full task set.

## The meta-finding: the loop ran out of questions

Here's where it gets uncomfortable, and more useful. The loop *accepted* over 14,000 experiment
proposals. But when you deduplicate by what each proposal actually *was* — the (experiment type,
task) pair — those 14,019 accepted proposals collapse to **597 unique cells**. The loop
re-proposed the same experiments **23.5× on average**; the single most-repeated cell
(convergence on task `2bcee788`) was proposed **243 times**.

It wasn't copy-pasting — verbatim-duplicate rationales were **0.0%**. Every proposal was freshly
*worded*. They were just conceptually the same handful of ideas dressed differently: **81%** of
rationales were about KL / task-id, **72%** about halting, **46%** about convergence. The loop
had, in effect, three ideas and an inexhaustible vocabulary for restating them.

The root cause is that 12-result memory window. A proposer that can only see the last 12
results has no way to know it already ran this exact experiment 200 batches ago. There's no
novelty signal, no memory of the full history, no "we've covered this — what's *missing*?" So it
does the locally-sensible thing every time (follow up on the most salient recent result) and the
globally-pathological thing in aggregate (re-ask the same questions forever). Hash-based dedup
existed but couldn't fire, because each proposal's surface form was unique.

## Why this is the actual lesson

It's tempting to file the loop under "failure" and move on. That's the wrong read. The loop
*did* produce a real result, and it did so by exhaustively covering 200 tasks with three solid
probes — which is genuinely more than a human would have patiently done by hand. The failure is
specific and fixable: **breadth without memory degrades into repetition.**

The fixes follow directly from the diagnosis:

- **Give the proposer the coverage map, not the recent tail.** "Here are the (experiment, task)
  cells already run" beats "here are the last 12 results." Reward *gaps*, not salience.
- **Dedup on intent, not on text.** Normalize a proposal to its (type, target) and reject the
  cell if it's been run — surface form is the wrong key.
- **Add experiments to the menu.** The loop literally could not propose latent-state probing, an
  SAE over the recursive block, or a cross-architecture comparison, because those kernels weren't
  in its `CORE_TYPES`. A loop can only explore the moves it's given; an empty part of the menu is
  invisible, not "unpromising."

This is, in miniature, the central open problem of automated research: not *can a model propose
experiments* (obviously yes), but *can it know what it has already learned and what it hasn't.*
Without a memory of coverage and a notion of novelty, a research agent is a very expensive way to
ask the same good question several hundred times.

## Status

Phase 2 (TRM) is the one part of a larger four-architecture interpretability study that has
produced real experimental output: the 37,292-run dataset, the KL/halt/correctness dissociation,
and the 107/200 never-solved aggregate. The autonomous-loop autopsy is written up in the repo's
`LOOP_ANALYSIS.md`. The genuinely novel experiments the menu was missing — latent probing, an SAE
on the recursive block, the cross-architecture CKA — are the next, deliberately human-directed,
step.

---

*Figures from the live experiment state (`done.jsonl`: 37,292 jobs; 200/200 tasks per kernel;
3,682 proposer calls). Dedup statistics from `LOOP_ANALYSIS.md`.*
