# Project Portfolio — Detailed Analysis

> Cross-project analysis of the work running on this GCP Linux VM (multi-project ML/systems research workstation, TPU + CPU).
> Snapshot date: **2026-05-31**. Owner: kathir-ks (kathirksw@gmail.com).
>
> This document is generated from git state, Claude Code session history (`~/.claude/projects/`),
> persistent memory notes, and each project's `CLAUDE.md`. It is a living index — re-generate when project state moves.

## Environment

- **Host:** GCP Linux VM (`us-central` region), multi-project research workstation.
- **Compute:** TPU fleet (v4-8, v5litepod-64 / v5e-64, v6e-64) across `us-central2-b`, `europe-west4-a/b`, `us-east1-d`; plus local CPU.
- **Python envs:** `~/venv-maxtext-py312` (3.12, for MaxText / activation-extract / levanter), `uv`-managed (autoresearch-tpu), `/usr/bin/python3` (3.10, system).
- **Primary GCS buckets:** `gs://arc-tpu-europe-west4` (MiniMax, primary), `gs://arc-data-europe-west4` (Qwen3), `gs://arc-mi-research` (ARC interpretability study).

### Standing operational constraints (hard-won)

- **No cross-region data ops from the dev VM.** A us-central VM → europe-west4 GCS transfer once cost ~$1,800 (1.5 lakh INR). Checkpoint reads/writes, HF streaming, and conversion must run on a TPU **in the same region** as the bucket. Never copy GCS↔GCS across regions.
- **Conversion runs on the TPU, never locally.** The TPU VM streams HF weights from the internet and writes to the same-region bucket.
- **On TPU preemption: wait and poll** for the slice to return (5–10 min intervals; re-provisioning can take 10–60+ min). Do **not** unilaterally switch TPU type/zone. Remember dtmpfs is host-local RAM and is wiped on preemption — a returning slice is a fresh boot.
- **Multi-worker TPU launch:** direct SSH with `setsid nohup ... </dev/null &`; two passes (kill by process name, then launch); `cd ~/maxtext` first; `eval $(ssh-agent -s); ssh-add` in the same shell. Never `pkill -f` in the launch command.

---

## At a glance

| Project | Dir | Stack | Branch | State | Last active |
|---|---|---|---|---|---|
| **MaxText — Qwen3.6-27B** | `maxtext/` | JAX/Flax | `qwen3.6-impl` | 🔥 In progress (uncommitted) | today |
| **ARC-AGI Interpretability** | `arc-agi-research/` + 4 worktrees | JAX/Flax + PyTorch | per-phase | 🔥 Active (Phase 2 critical path) | today |
| **MaxText — MiniMax-M2.7** | `maxtext/` worktree | JAX/Flax | `feature/minimax-m2.7` | 🟢 Working E2E on TPU | today |
| **vllm-under-the-hood** | `vllm-under-the-hood/` | study notes | — | 🟡 Exploration | today |
| **dtmpfs** | `dtmpfs/` | Rust | `main` | 🟡 Phase 6 | May 21 |
| **claude-hub** | `claude-hub/` | Python/aiohttp/xterm.js | `feature/ui-stability-mobile-persistence` | 🟡 Feature branch | May 17 |
| **activation-extract** | `activation-extract/` | JAX/Flax | `fix/dynamic-seq-length` | 🟢 Shipped, 25 uncommitted | May 14 |
| **MaxText — MiniMax-M2.5** | `maxtext/` | JAX/Flax | `feature/minimax-m2.5-support` | 🟢 Perf-characterized, blocked on HF token | Apr 24 |
| **MaxText — Gemma 4 31B** | `maxtext/` | JAX/Flax | `feature/gemma4-31b-inference` | 🟡 Decode works, API server parked | Apr 8 |
| **autoresearch-tpu** | `autoresearch-tpu/` | JAX/Flax NNX, `uv` | `tpu-port` | ⚪ Dormant | Apr 20 |
| **rusty-v8** | `rusty-v8/` | Python/libclang | `main` | ⚪ Dormant | Apr 9 |
| **tpu-bare-metal** | `tpu-bare-metal/` | C/PJRT | `main` | ⚪ Dormant | Mar 23 |
| **levanter** | `levanter/` | JAX/Equinox | `main` | ⚪ Upstream mirror | Oct 2025 |
| **attenRes-ViT** | `attenRes-ViT/` | Python | `master` | ⚪ Complete | Mar 21 |

🔥 active focus · 🟢 working/validated · 🟡 in progress / parked · ⚪ dormant

---

## 1. 🔥 MaxText — Qwen3.6-27B bring-up *(current live work)*

**Dir:** `~/maxtext` · **Branch:** `qwen3.6-impl` · **Upstream:** `git@github.com:kathir-ks/maxtext.git`

Bringing up a **dense 27B Qwen3.6** model for text-only inference on Google's MaxText, built on the **Qwen3.5 hybrid chassis** (GatedDeltaNet + GatedAttention). The vision encoder and MTP (multi-token-prediction) head are deliberately **not** loaded — text inference only.

**Architecture (from `configs/models/qwen3.6-27b.yml`):**
- `decoder_block: qwen3_5`, `base_emb_dim: 5120`, `base_num_decoder_layers: 64`
- `base_num_query_heads: 24`, `base_num_kv_heads: 4`, `head_dim: 256`
- `vocab_size: 248320`, RMSNorm `epsilon 1e-6`
- Dense MLP path selected via `num_experts == 1` (reuses `qwen3_5.py`'s `MlpBlock`).

**Work in progress (uncommitted):**
- `src/maxtext/configs/models/qwen3.6-27b.yml` *(new)* — model config.
- `src/maxtext/checkpoint_conversion/standalone_scripts/convert_qwen3_6_unscanned.py` *(new)* — unscanned-layout HF→MaxText checkpoint converter.
- Edits to `src/maxtext/models/qwen3_5.py` and `src/maxtext/configs/types.py`.

**State:** mid-implementation, **nothing committed on this branch yet**. Next steps: commit the config + converter, run conversion on an in-region TPU (per the no-local-conversion rule), and smoke-test decode.

---

## 2. 🔥 ARC-AGI Mechanistic Interpretability Study *(current heavy focus)*

**Hub:** `~/arc-agi-research` (branch `main`, `git@github.com:kathir-ks/arc-agi-research.git`) + four sibling git worktrees, one per phase.

A coordinated, **TPU-only** research project comparing how three model architectures internally solve ARC-AGI tasks, written up as an arXiv paper. The hub `main` branch holds the research plan (Markdown design docs) and the cross-phase contract (`shared/`); the actual code lives in phase worktrees on their own branches.

**Cross-phase invariants (single source of truth in `shared/`):**
- `shared/task_ids/` — canonical **200-task subset** (first-200-alphabetical from ARC-AGI-1 training). Drift here silently breaks Phase 2 Exp E, Phase 3 attention comparison, and Paper Figure 10.
- `shared/tpu_schedule.md` — three v4-8s for per-phase hooking; one v5litepod-64 staggered for SAE training (Phase 1 → Phase 3 → Phase 2); one v6e-64 reserved for Phase 4 synthesis.
- `shared/sae_handoff.md` — common SAE shard shape + `metadata.json` schema all model phases must emit.
- `shared/gcs_paths.md` — canonical GCS layout under `gs://arc-mi-research/`.

### Phase 1 — Qwen 2.5-0.5B + LoRA (autoregressive) · `~/arc-phase1-qwen` · branch `phase1`
Mechanistic interp of the ARC-fine-tuned merged Qwen. Four experiments: (A) attention-head specialization heatmap, (B) per-layer linear probes for spatial concepts, (C) residual-stream SAE feature dictionary (≥50 labeled features), (D) what TTT changes in weights/activations. Reuses `~/activation-extract`'s JAX/Flax Qwen + hooks. **TransformerLens/sae-lens/nnsight are NOT used** (CUDA-only / no GPU) — static matplotlib/plotly renders instead. *Last session May 23.*

### Phase 2 — TRM-7M (recursive) · `~/arc-phase2-trm` · branch `phase2` — **CRITICAL PATH**
The longest, most novel phase (no prior MI work on TRM, `SamsungSAILMontreal/TinyRecursiveModels`, MIT). **Runs on CPU PyTorch** (7M params; torch_xla is pure overhead). Five experiments: (A) answer-state evolution across recursive iterations, (B) latent "scratchpad" per-iteration probes, (C) causal tracing of the `task_id_token` mechanism, (D) single-block SAE feature library, (E) **CKA/RSA vs Qwen** — the cross-architecture comparison, run last. **By far the most active project — ~3,700 Claude sessions, active today.** SAE (Exp D) is the largest job (~23M activation vectors), scheduled last in the staggered v5e-64 slot (days 9–11).

### Phase 3 — LLaDA-8B (masked diffusion) · `~/arc-phase3-llada` · branch `phase3`
Architectural contrast: masked-diffusion vs AR vs recursive. A **standalone JAX/Flax LLaDA port** (the activation-extract registry is causal-LM-only and doesn't fit). Three experiments: (A) unmasking order + spatial clustering, (B) bidirectional-attention head heatmap, (C) 2D-RoPE vs 1D-RoPE representation diff. **Forward-pass parity vs HF (atol 1e-4) is non-negotiable** — most silent-failure risk of all phases. Scopeable down to Exp A only if compute tightens. *Last session May 23.*

### Phase 4 — Synthesis & Paper · `~/arc-phase4-paper` · branch `phase4`
Write-as-you-go arXiv paper (LaTeX) + the cross-architecture synthesis analyses: CKA/RSA across Qwen/TRM/LLaDA, attention-head taxonomy (Figure 10), universal-feature matching across the 3 SAE dictionaries, and a commitment by Day 11 to one of three narratives (Convergence / Divergence / Mixed). **Reads inputs only from GCS, never from sibling worktree filesystems.** *Last session May 19.*

> ⚠️ **Risk note:** the phase worktrees (`arc-phase1/2/3/4`) carry untracked/uncommitted state locally, and this is the project where the docs explicitly warn that "silent drift breaks cross-architecture comparisons." This is the highest-value work-in-flight to keep committed and backed up.

---

## 3. 🟢 MaxText — MiniMax-M2.7 *(working end-to-end on TPU)*

**Branch:** `feature/minimax-m2.7` (`kathir-ks/maxtext`) · worktree active today.

A **230B / 10B-active MoE** running end-to-end on TPU pods with **strictly zero outbound bytes** from any host (validated 2026-05-19). Two working configs:

- **v5e-64 (europe-west4-b) — distributed storage (primary):** each host stores only its ~27 GB of shards on local dtmpfs; pod-wide total 432 GB (vs 7.4 TB replicated). v5e host RAM is only 188 GB, so this is the *only* config that fits. Uses a 16-process `jax.distributed` converter (`convert_minimax_m2_distributed.py`) that pulls the engine's actual shardings, writes one `.npy` per (leaf, local_device), and streams HF safetensors (peak FP8 ~15 GB). Decode via `decode_minimax_m2_npy.py` auto-detects distributed vs replicated manifests. **Verified coherent output** ("The capital of France is Paris…", identical on all 16 hosts).
- **v6e-64 (europe-west4-a) — replicated storage:** full 460 GB local copy per host (708 GB host RAM). Works, but the slot was preempted between runs.

**Critical v5e knobs:** `megablox=false sparse_matmul=false capacity_factor=2.0` (v5e can't do ragged-all-to-all → capacity-factor routing falls back to dense). Converter `--maxtext_args` must match decode's `ici_tensor_parallelism=8 ici_expert_parallelism=8`. Per-host conversion ~40 min (sequential `hf_hub_download` at ~35 s/layer); compile + first token ~5 min. Built on the **dtmpfs** project (below).

---

## 4. 🟢 MaxText — MiniMax-M2.5 *(perf-characterized; blocked on HF token)*

**Branch:** `feature/minimax-m2.5-support`. Earlier inference-perf project on the same 229B MoE. Implementation correctness verified (sigmoid-fp32 routing, partial RoPE factor 0.5, QK-norm before RoPE, AQT int8 MoE path). Real-weight correctness confirmed ("Paris" ✓, `sqrt(144)=12` ✓, `12!=479,001,600` ✓).

**Measured ceilings:**
- **v6e-64:** `scan=False` batch=16 → **6,792 tok/s** (CEILING, +44% over scan=True). Valid scan=False batches are multiples of 4 with b×4≥16 (4, 8, 12, 16, 20, 24). scan=True batch=12 + int4 → 4,701 tok/s.
- **v5e-64:** `scan=False` batch=7 target=96 → **2,556 tok/s** (CEILING, +39% over scan=True 1,836). batch=8 is bf16 OOM.
- Root cause of the bandwidth gap: `dynamic_slice` in the while_loop blocks weight prefetch (~80 GB/s effective vs 1,800 GB/s peak).

**Blocked:** no HF token provided → bf16 scan checkpoint not yet created → int8 conversion (expected 8,000+ tok/s on v6e) can't proceed. Largely superseded by M2.7.

---

## 5. 🟡 MaxText — Gemma 4 31B *(decode works; API server parked)*

**Branch:** `feature/gemma4-31b-inference`. Model produces correct output ("The capital of France is **Paris**.") via direct `maxtext.inference.decode` on v6e-64 (`ici_fsdp_parallelism=64`, scanned checkpoint, only 0.93 GB/chip). **Blocker:** the OpenAI-compatible API server takes **40+ min to compile** in `MaxTextGenerator.__init__()` — preemptible TPUs get reclaimed first. Also a `transformers` version split: ≥5.0 needed for conversion (gemma4 model type), ==4.57.3 for serving (tokenizer `extra_special_tokens` bug in 5.x). Fix options: non-preemptible TPU, reduce compile time, or switch to JetStream.

---

## 6. 🟢 activation-extract *(shipped; data backbone for ARC study)*

**Dir:** `~/activation-extract` · **Branch:** `fix/dynamic-seq-length` (25 uncommitted files) · `git@github.com:kathir-ks/activation-extract.git`

JAX/Flax activation-extraction module for SAE training. Full pipeline: JSONL → tokenize → MaxEngine prefill → sown intermediates → masked flatten → safetensors writer → manifest → `load_merged`. **39 tests pass** (34 unit + 5 multi-host equivalence/parity), E2E validated on v4-8 with gemma-2b random init. **Throughput:** ~53k tok/s peak (gemma-2b, seq_len=1024) → ~3 B tokens/day realistic end-to-end. Now the data-side dependency reused by ARC Phases 1–3 (Qwen model + hooks, storage layer, SAE shard format).

> **Known gotcha:** MaxText's pydantic `configs/types.py` `validate_and_set_hlo_dump_defaults()` sets `XLA_FLAGS=--xla_dump_to=/tmp/xla_dump --xla_dump_large_constants` unconditionally (the deprecated `pyconfig_deprecated.py` had a `if not dump_hlo: return` guard the new code lost), which dumps ~10 GB of random-init weights and fills the root partition. Workaround: set `XLA_FLAGS=--xla_dump_to=` (empty, must start with `--`) before importing jax/maxtext.

---

## 7. 🟡 dtmpfs *(distributed tmpfs, Phase 6)*

**Dir:** `~/dtmpfs` · **Branch:** `main` · `git@github.com:kathir-ks/dtmpfs.git`

Rust multi-crate workspace (`dtmpfs-{client,common,meta,proto,store}`) — a distributed tmpfs that powers the **zero-egress per-host sharding** used by MiniMax-M2.7. Latest work (Phase 6, May 11): HTTP debug, replica failover, stale-write rejection, CI. Host-local RAM-backed, so contents are lost on TPU preemption (see operational constraints).

---

## 8. 🟡 claude-hub *(Claude Code session manager)*

**Dir:** `~/claude-hub` · **Branch:** `feature/ui-stability-mobile-persistence` · `git@github.com:kathir-ks/claude-hub.git`

Web UI for managing Claude Code sessions over tmux (Python, aiohttp, xterm.js) — the primary interface for running multi-agent Claude Code work on this box (`./start.sh` prod / `python3 server.py` dev). Recent feature work: UI stability, mobile layout, conversation history, Docker packaging, session improvements. *Last session May 17.*

---

## 9. ⚪ Dormant / reference projects

- **autoresearch-tpu** (`~/autoresearch-tpu`, branch `tpu-port`, `uv`-managed) — TPU port of Karpathy's autoresearch loop; JAX/Flax NNX. Edit `train_jax.py` only. *Apr 20.*
- **rusty-v8** (`~/rusty-v8`, `main`) — C++→Rust transpiler for V8 via Clang AST (Python + libclang). Last fix: NOT-operator precedence. *Apr 9.*
- **tpu-bare-metal** (`~/tpu-bare-metal`, `main`) — direct TPU control via `libtpu.so` PJRT C API. *Mar 23.*
- **levanter** (`~/levanter`, `main`) — Stanford CRFM foundation-model training framework (JAX/Equinox); mostly an upstream mirror, lightly touched. *Oct 2025.*
- **attenRes-ViT** (`~/attenRes-ViT`, `master`) — full AttnRes-ViT implementation, complete with code-review fixes. *Mar 21.*

### Scratch / non-git working dirs
`golfchallenge`, `weight-tieing`, `tiny-chip-rl`, `mailclaw`, `local_deployment`, `personal-site-chat`, `vllm-under-the-hood` (study notes, active today), `local_repos`, `sampleProb`, `sae-training` / `sae-worktree` (SAE trainer consumed by the ARC study).

---

## Themes & observations

1. **Two parallel tracks.** (a) **TPU inference / model bring-up** in MaxText — a pipeline of large MoE and dense models (MiniMax M2.5 → M2.7, Gemma 4 31B, now Qwen3.6-27B), each pushing zero-egress + perf ceilings. (b) A **multi-architecture mechanistic-interpretability study** (ARC-AGI) spanning four coordinated worktrees toward an arXiv paper.
2. **Supporting infrastructure is shared.** `activation-extract` (hooks/storage) feeds the ARC study; `dtmpfs` (distributed RAM store) enables MiniMax zero-egress sharding; `claude-hub` orchestrates all the agent sessions.
3. **Cost & preemption discipline is codified** into hard rules (no cross-region ops, in-region conversion, poll-don't-switch on preemption) after a real ~$1,800 incident.
4. **Immediate risks:** (a) the live Qwen3.6 work is entirely uncommitted; (b) the ARC phase worktrees carry uncommitted state in exactly the project where drift is most damaging. Both are worth committing/backing up.
