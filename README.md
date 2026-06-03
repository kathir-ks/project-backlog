# project-overview

A living, detailed analysis of the ML/systems projects running on the dev VM —
generated from git state, Claude Code session history, persistent memory notes,
and each project's `CLAUDE.md`.

- **[PROJECTS.md](PROJECTS.md)** — portfolio index: every project, its state, and git/session signals.
- **[PROJECTS_DEEP_DIVE.md](PROJECTS_DEEP_DIVE.md)** — website source material: per project, the core idea, what was built, experiments run, and concrete on-VM outputs (each number tagged measured ✅ vs planned 🟡), plus blog-post and research ideas.
- **[blogs/](blogs/)** — detailed technical blog posts (neutral voice, ~2,100 words each, real code + verified numbers). Enumerable via **[blogs/index.json](blogs/index.json)** for site rendering.

## Blog posts

| Post | Theme | Status |
|---|---|---|
| [Running a 230B model on a pod that can't hold it](blogs/minimax-m2.7-zero-egress-sharding.md) | TPU inference | validated |
| [A 21× bandwidth gap hiding in one dynamic_slice](blogs/minimax-m2.5-tpu-perf-archaeology.md) | TPU inference | validated |
| [When a model runs fine but won't serve](blogs/gemma4-31b-compile-vs-preemption.md) | TPU inference | partial |
| [94.6% explained variance, then −288%](blogs/sae-94-percent-and-the-collapse.md) | Interpretability | shipped |
| [37,292 experiments, 597 ideas](blogs/arc-trm-autoresearch-loop.md) | Interpretability | real result |
| [dtmpfs: a distributed RAM filesystem in Rust](blogs/dtmpfs-distributed-ram-filesystem.md) | Systems | shipped |
| [The 40-byte struct: driving a TPU from pure C](blogs/tpu-bare-metal-pjrt.md) | Systems | validated |
| [Round-tripping an AST loses your parentheses](blogs/rusty-v8-ast-roundtrip-precedence.md) | Tools | partial PoC |
| [Porting an autonomous training-optimization loop to TPU](blogs/autoresearch-tpu-val-bpb.md) | Tools | validated |

Snapshot: 2026-05-31.
