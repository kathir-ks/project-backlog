---
title: "37,292 experiments, 597 ideas: what an autonomous research loop actually does"
subtitle: "Probing a Tiny Recursive Model on ARC — and watching an LLM-driven research loop run out of questions"
date: 2026-05-31
tags: [interpretability, arc-agi, autonomous-research, llm-agents, trm]
status: real-result   # 37k deterministic runs; honest meta-finding included
---

# 37,292 experiments, 597 ideas: what an autonomous research loop actually does

This started as a mechanistic-interpretability study of a **Tiny Recursive Model (TRM)** — a
7M-parameter network that solves ARC-AGI puzzles by applying one small transformer block over and
over, refining an answer-state and a latent "scratchpad" across ~16 recursive iterations. The
question: *what does TRM compute, iteration by iteration?* TRM is interesting precisely because it's
tiny and recursive — there's no prior mechanistic-interpretability work on it, and "the same block,
applied N times" is a fundamentally different computational shape from a stack of distinct layers.

To explore it, the project wired up an **autonomous research loop**: an LLM proposes experiments,
deterministic Python kernels run them against the real model, results accumulate, and the LLM reads
the recent results to propose the next batch. It ran **37,292 experiments** across all 200 canonical
ARC tasks. It found a genuine result. It also did something more interesting and more cautionary,
which is the real subject of this post.

## The model and the data

The substrate is real, not a toy reimplementation. The loop loads an actual trained checkpoint
(`Sanjin2024/TinyRecursiveModels-ARC-AGI-1`, step 155718), reads the model dimensions straight from
the state dict, and runs it on CPU in bfloat16 — at 7M parameters a TPU would be pure overhead, and
CPU keeps the experiment loop simple and cheap. The dataset is a concept-augmented version of
ARC-AGI-1 (960 task groups expanded to ~876k puzzle identifiers via the standard ARC augmentations —
rotations, reflections, color permutations), and the canonical 200-task subset is fixed up front so
every experiment is measured against the same tasks.

TRM's forward pass exposes three things worth probing at each of its recursive iterations: the
current **answer-state** (its best guess at the output grid so far), a **latent state** (the
scratchpad it carries between iterations), and a **halt head** (`q_halt_logits`, the model's own
signal for "I'm done, stop iterating"). Those three outputs are the raw material for the whole study.

## The three probes

Three deterministic experiment kernels do the actual work, each emitting a small structured record so
results are uniform and aggregable:

- **convergence** (`exp_a_convergence`) — per iteration, how much the argmax answer changed (Hamming
  distance to the previous step), the softmax entropy, and whether/when the answer became correct.
  Emits `first_correct_step` and a `slow_convergence` flag (true when more than 40% of the total
  Hamming change happens in the *second* half of the iterations — i.e. the model is still thrashing
  late).
- **task-ID ablation** (`exp_c_ablate`) — run the model with the real `puzzle_id` token versus a
  blanked one (`puzzle_identifiers=0`), and measure per-step KL divergence between the two logit
  streams. Emits `peak_kl_step` and `peak_kl_value`. This is the causal-tracing probe: how much, and
  when, does the task identity actually steer the computation?
- **halting** (`exp_d_halt`) — track `sigmoid(q_halt_logits)` across iterations. Emits
  `first_would_halt_step`, the first step where the halt probability crosses 0.5 (or −1 if it never
  does).

The task-ID kernel is the cleanest to show:

```python
# task-ID ablation kernel — the core of it
kls = [float(kl_div_per_position(real[i]["logits"], blank[i]["logits"]).mean()) for i in range(n)]
record = {
    "puzzle_id": pid, "n_steps": n,
    "peak_kl_step":  int(max(range(n), key=lambda i: kls[i])),
    "peak_kl_value": round(max(kls), 6),
}
```

## The proposer, and the detail that matters

The proposer is Claude, called headless with a hard per-call budget so an autonomous loop can't run
away with the bill:

```python
["claude", "--model", "claude-sonnet-4-6", "--output-format", "json",
 "--max-budget-usd", "0.15", "-p", prompt]   # ~$0.02 warm, ~$0.10 cold
```

Every ~10 finished jobs it's asked to propose 2–5 follow-up experiments. The prompt is built from
the recent results — and here is the detail that turns out to govern everything: it only ever shows
the proposer **the last 12 results**.

```python
for r in recent_done[-12:]:   # the entire memory of the loop
    ...
```

Twelve results is the loop's whole working memory. Hold that thought; it's the hinge.

## The real finding: task-ID matters, halting and correctness don't follow

Across all 200 tasks, a clean **three-way dissociation** falls out. The model is strongly sensitive
to the task-ID token — but that sensitivity is decoupled from both knowing when to stop and getting
the answer right. Task `2bcee788` is the archetype:

- **peak KL = 132.09 at step 0** — blanking the task-ID token hugely changes the very first
  iteration's logits. The model reads the task identity immediately and hard, right at the start.
- **`first_would_halt_step = −1`** — across every iteration, the halt probability never crosses the
  threshold. The model never decides it's done.
- **`first_correct_step = −1`** — and it never produces the correct grid.

So the picture for this task is: *immediate, strong task-ID routing* + *never decides to stop* +
*never correct.* Whatever the task-ID token does at step 0, it is not setting up a trajectory that
converges to a confidently-halted, correct answer. And this isn't a cherry-picked anecdote: in
aggregate, **107 of 200 tasks are never solved at any iteration**. The task-ID signal is real and
early across the board; the competence it's presumably meant to enable frequently just isn't there.

That is a genuine, publishable mechanistic nugget — a concrete dissociation between a model's internal
"I know which task this is" signal and its actual problem-solving ability, surfaced by brute-force
deterministic probing over the full task set. A human poking at a handful of tasks by hand might have
missed it; the exhaustive 200-task sweep makes it undeniable.

## The meta-finding: the loop ran out of questions

Here's where it gets uncomfortable, and more useful. The loop *accepted* over 14,000 experiment
proposals (14,019, to be exact). But when you deduplicate by what each proposal actually *was* — the
(experiment type, task) pair — those 14,019 accepted proposals collapse to **597 unique cells**. The
loop re-proposed the same experiments **23.5× on average**; the single most-repeated cell
(convergence on task `2bcee788`) was proposed **243 times**.

It wasn't copy-pasting. Verbatim-duplicate rationales were **0.0%** — every proposal was freshly
*worded*, with its own justification. They were just conceptually the same handful of ideas dressed
in different prose: **81%** of rationales were about KL / task-id, **72%** about halting, **46%**
about convergence. The loop had, in effect, three ideas and an inexhaustible vocabulary for restating
them. A hash-based deduplication step existed but couldn't fire, precisely because each proposal's
surface form was unique — the dedup was keyed on text, and the text was always new.

The root cause is that 12-result memory window. A proposer that can only see the last 12 results has
no way to know it already ran this exact experiment 200 batches ago. There's no novelty signal, no
memory of the full history, no "we've already covered this — what's *missing*?" So it does the
locally-sensible thing every single time (follow up on the most salient recent result, which is
usually the dramatic peak-KL task) and the globally-pathological thing in aggregate (re-ask the same
question forever). Locally rational, globally a treadmill.

## Why this is the actual lesson

It's tempting to file the loop under "failure" and move on. That's the wrong read on two counts.
First, the loop *did* produce a real result, and it did so by exhaustively covering 200 tasks with
three solid probes — which is genuinely more patient and complete than a human would have been by
hand. Second, the failure is *specific and diagnosable*, which makes it worth far more than a vague
"the agent got confused." The failure is: **breadth without memory degrades into repetition.**

The fixes follow directly from the diagnosis:

- **Give the proposer the coverage map, not the recent tail.** "Here are the (experiment, task) cells
  already run, and here are the gaps" beats "here are the last 12 results." Reward filling *gaps*, not
  reacting to *salience*. The information the loop needed was its own history, and it was the one thing
  the prompt withheld.
- **Dedup on intent, not on text.** Normalize a proposal to its (type, target) and reject the cell if
  it's already been run. Surface form is the wrong key; the loop proved that conclusively by
  generating 14,019 textually-unique restatements of 597 ideas.
- **Expand the menu.** The loop literally could not propose latent-state probing, an SAE over the
  recursive block, or a cross-architecture comparison, because those kernels weren't in its
  `CORE_TYPES`. A loop can only explore the moves it's given; an empty part of the menu isn't judged
  "unpromising," it's *invisible*. The three ideas it fixated on were, not coincidentally, exactly the
  three kernels that existed.

This is, in miniature, the central open problem of automated research. The easy part — *can a model
propose plausible experiments* — is obviously solved; it proposed 14,019 of them, each individually
reasonable. The hard part is *can it know what it has already learned and what it hasn't*. Without a
memory of coverage and an explicit notion of novelty, a research agent is a very expensive way to ask
the same good question several hundred times. Scaling the *number* of experiments is trivial; scaling
the number of *distinct* questions is the whole game, and it doesn't come for free with a bigger
model or a longer run.

## Status

Phase 2 (TRM) is the one part of a larger four-architecture interpretability study that has produced
real experimental output: the 37,292-run dataset (a 24 MB `done.jsonl`), the KL/halt/correctness
dissociation, and the 107/200-never-solved aggregate. The autonomous-loop autopsy — including all the
dedup statistics above — is written up in the repo's `LOOP_ANALYSIS.md`. The genuinely novel
experiments the menu was missing — latent-state probing, an SAE on the recursive block, the
cross-architecture CKA against an autoregressive model — are the next step, and deliberately
human-directed this time, with the coverage-map and novelty fixes as the lessons carried forward.

---

*Figures from the live experiment state (`done.jsonl`: 37,292 jobs; 200/200 tasks per kernel; 3,682
proposer calls). Dedup statistics from `LOOP_ANALYSIS.md`. Model:
`Sanjin2024/TinyRecursiveModels-ARC-AGI-1`, step 155718.*
