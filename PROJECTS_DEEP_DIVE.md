# Deep Dive — Projects, Experiments & Results

> Website source material. For each project: the **core idea**, **what I built**, the **experiments I ran**, and the **concrete outputs/results** that exist on the VM today.
> Snapshot: **2026-05-31**. Every number below is tagged ✅ *measured/validated* or 🟡 *designed / not yet run* so nothing on the public site overstates.
>
> Honesty notes baked in from a source audit: a couple of figures that float around in my notes are **not** substantiated by anything in the repos (flagged inline). Trust the tagged ones.

---

## Who / what this is

Self-taught frontier-AI researcher; embedded-systems engineer by day (VxWorks / C++). The work clusters into four themes, all running on a personal TPU fleet (v4-8, v5e-64, v6e-64) via Google TRC:

1. **TPU inference & model bring-up** — getting large MoE / dense LLMs to run fast and correctly on TPU pods (MaxText).
2. **Mechanistic interpretability** — SAEs, activation extraction, and a multi-architecture ARC-AGI interpretability study.
3. **Systems & infra** — a distributed RAM filesystem, bare-metal TPU control, a Claude-Code orchestration hub.
4. **Tools & one-off experiments** — an autonomous research loop, a C++→Rust transpiler, small architecture/RL studies.

There's also a **Hindi / multilingual NLP** thread (datasets + small models) that lives partly off-VM (HuggingFace) — noted at the end.

---

# Theme 1 — TPU Inference & Model Bring-up (MaxText)

The throughline: large models on pods whose **host RAM can't hold a replica**, and chasing the real throughput ceiling through XLA's tiling quirks. Driven in part by a costly lesson — a single cross-region GCS transfer once cost **~$1,800** — which is why every pipeline is zero-egress and in-region.

## 1.1 MiniMax-M2.7 — 230B MoE with zero-egress host-distributed sharding ⭐
*Branch `feature/minimax-m2.7`. The flagship systems result.*

**Core idea.** Run **MiniMax-M2.7 (230B total, 10B active, 256-expert sigmoid-routed MoE, 62 layers)** end-to-end on a TPU pod with **strictly zero outbound bytes per host** — and make it fit on **v5e-64 whose hosts have only ~188 GB RAM** when a full replica is ~460 GB.

**What I built.**
- `convert_minimax_m2_distributed.py` — the headline trick. Runs as **16 `jax.distributed` processes (one per host)**. For each parameter leaf it reads the engine's actual `sharding.addressable_devices_indices_map(global_shape)` and writes **one `.npy` per (leaf, local device)** — each host persists only the slices its 4 chips will own. HF weights are **streamed**: only the safetensors files for this host's expert range are pulled, fetched right before the layer needs them and unlinked after. Planning uses `jax.eval_shape`, so it never touches HBM.
  - Result: **~27–30 GB/host (~432 GB pod-wide) vs ~7.4 TB replicated.** ✅
- `decode_minimax_m2_npy.py` — auto-detects distributed (`manifest.p*.json`) vs replicated (`manifest.json`); rebuilds the global `jax.Array` with `make_array_from_single_device_arrays`, mmap-loading only each host's slices.
- `serve_minimax_m2_npy.py` — OpenAI- *and* Anthropic-compatible FastAPI server over the `.npy`-backed engine.

**Experiments & results.**
- ✅ **v5e-64 distributed decode validated (2026-05-19):** `"The capital of France is"` → `" Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid…"` — coherent and **identical on all 16 hosts**. Per-host conversion ~40 min; compile + first token ~5 min.
- ✅ **v6e-64 throughput sweep (2026-05-22)**, pod-wide tok/s, scan_layers + megablox sparse routing, tensor=8 × expert=8:

  | batch | config | tok/s |
  |---:|---|---:|
  | 1 | bf16 | 860 |
  | 8 | int8-KV | 3,234 |
  | 32 | KVq | 4,539 |
  | 64 | KVq, t=128 | 5,866 |
  | **96** | **KVq, t=128** | **6,678 ← ceiling** |

- ✅ **Live API server on v6e-64** verified through an IAP tunnel: OpenAI `/v1/completions` (Paris, 9.1 s cold), `/v1/chat/completions` (2.5 s warm), Anthropic `/v1/messages` (correct stop_reason/token mapping). **8.3 GB/chip** at B=1 ctx=2048 int4-KV — lots of headroom.
- ✅ **The v5e wall, quantified:** v5e XLA **can't do `ragged-all-to-all`** (which megablox MoE routing needs), forcing dense capacity-factor routing (`megablox=false sparse_matmul=false capacity_factor=2.0`). Best v5e ≈ **7.4 tok/s** vs 6,678 on v6e → a **~902× gap**, entirely attributable to that one missing collective.
- 🟡 **Spin-off:** a custom **v5e Pallas grouped-GEMM** MoE kernel (branch `…-v5e-pallas`) to escape the wall — JAX-backend version + 4 green correctness tests committed; Pallas tuning blocked on v5e access.

**Status:** distributed decode + API server validated; Pallas kernel is the open frontier.

## 1.2 MiniMax-M2.5 — perf archaeology on a 229B MoE
*Branch `feature/minimax-m2.5-support`. The predecessor; where the TPU-perf lessons were mined.*

**Core idea.** Same model family, focused on **routing correctness, int8/int4 (AQT) quantization, and the true throughput ceiling** including XLA tiling cliffs.

**Engineering tricks (each a mini-finding).**
- `param_scan_axis=0` for MoE weights → avoids a **2.18 GB XLA transpose buffer** that was OOMing v5e (this is what made v5e viable at all). ✅
- `_build_multi_axis_stacked_tensor` pre-allocates and fills `result[i,j]`, avoiding a **284 GB peak** during checkpoint assembly. ✅
- int8 MoE quantization routed via `get_einsum → AqtEinsum` so *dense* matmuls get quantized, not just sparse.
- A KV-scale sharding bug in `kvcache.py` fixed with `nn.with_logical_constraint` on the quantized scale.
- Routing correctness: sigmoid in fp32, pre-bias selection, QK-norm **before** RoPE to match HF — 6 unit tests pass. ✅

**Results.** Real-weight correctness ✅ ("capital of France"→Paris, sqrt(144)=12, 12!=479,001,600). Throughput ceilings (✅ measured, point-in-time):
- **v6e-64:** scan=False **batch=16 → 6,792 tok/s** (the global ceiling); scan=True batch=12 int4 → 4,701.
- **v5e-64:** scan=False **batch=7 t96 → 2,556 tok/s** (+39% over scan=True 1,836).
- **The blog-worthy root cause:** a **~21× HBM-bandwidth gap** (~80 GB/s effective vs ~1,800 peak) traced to a `dynamic_slice` inside the decode `while_loop` blocking weight prefetch. Corollaries: `scan_unroll=2` was **−7.8%** (bigger loop body hurts XLA scheduling); async-collective flags gave **0%** (the MoE all-reduce sits on the critical path with nothing to overlap). And the **odd-batch 2× cliff** / multiples-of-4 fast path on v6e scan=False.

**Status:** correctness + perf fully characterized; int8 weight-quant tooling ready but was blocked on producing the bf16 checkpoint (no HF token) + TPU capacity shortages.

## 1.3 Qwen3.6-27B — dense bring-up on the hybrid chassis
*Branch `qwen3.6-impl`. Current live work.*

**Core idea.** Bring up Qwen's dense 27B for text-only inference on the **Qwen3.5 hybrid chassis** — alternating **Gated DeltaNet** (linear attention) and **Gated full Attention** layers (gated-attention every 4th layer), dense MLP. Vision tower and MTP head deliberately dropped.

**What I built (uncommitted on branch).**
- `configs/models/qwen3.6-27b.yml` — 64 layers, emb 5120, 24 Q / 4 KV heads, head_dim 256, vocab 248320; `num_experts: 1` selects the dense MLP path; partial rotary 0.25; GDN params (48 value / 16 key heads, conv kernel 4, chunk 64).
- `convert_qwen3_6_unscanned.py` (372 lines) — HF→unscanned-Orbax converter. The hard part is **re-interleaving GDN projections**: HF splits the fused Qwen3-Next projections into four tensors; `_fuse_qkv_z()` / `_fuse_b_a()` re-interleave them per-K-head-group into MaxText's fused layout (handling GQA). Gated-attention `q_proj` carries `[query, gate]` so its head-dim is doubled. Drops `vision_tower.*`/`mtp.*`, runs CPU-only.

**Results.** ✅ **Validated end-to-end on v4-8:** conversion parity **max KL = 0.020 across 3 prompts** vs HF (3/3 passed); coherent decode; ~54 GB bf16 fits v4-8 HBM. No throughput sweep yet (this was a correctness bring-up).

**Status:** functionally complete + parity-validated; needs commit + an in-region TPU conversion run + decode smoke test.

## 1.4 Gemma 4 31B — works in decode, blocked at serving
*Branch `feature/gemma4-31b-inference`. A clean "it runs but won't serve" story.*

**Core idea & result.** ✅ Direct decode produces correct output ("…**Paris**.") on v6e-64, scanned checkpoint, **0.93 GB/chip** (3% of HBM). **Blocker (the interesting part):** the OpenAI API server takes **40+ minutes to compile** in `__init__()` → **preemptible TPUs get reclaimed before it finishes**, so it never comes up on spot capacity. Plus a `transformers` version split (≥5.0 to convert, ==4.57.3 to serve). Fixes: on-demand TPU, faster compile, or JetStream.

---

# Theme 2 — Mechanistic Interpretability

A coherent pipeline: **extract** activations at scale → **train** SAEs on them → **interpret** across architectures.

## 2.1 activation-extract — residual-stream extraction at pod scale
*Branch `fix/dynamic-seq-length`.*

**Core idea.** Extract intermediate activations from arbitrary HF causal LMs on TPU pods, as the training corpus for SAEs. Key bet: **single forward passes, no autoregressive generation** (much faster, true batching, no KV cache).

**What I built.** A model-agnostic JAX/Flax layer with a registry keyed on HF `model_type` (Qwen2/3, Llama, Mistral, Gemma/2, Phi-3, plus MoE: Mixtral, Qwen2/3-MoE, OLMoE), template-driven HF→Flax conversion, three data pipelines (`text`, ARC `prompt`, ARC `grid_chunking`), gzip-pickle sharded storage + `metadata.json`, and **multi-host SPMD across 64 devices** with a socket-TCP startup **barrier** (`barrier_sync.py`) gating 6 phases to defeat staggered-SSH JAX init races. Preemption-resilient (nohup + 5-min health poll + resume-from-GCS + chunk cache).

**Experiments & results.**
- ✅ **49 test functions / 8 files** — JAX-vs-HF parity (float32 max_diff < 0.01; top-5 token agreement in bf16), per-family exact parity, multi-host equivalence (rel err < 5%), e2e roundtrip (< 1e-4).
- ✅ **Run 001:** Qwen2.5-0.5B layer 12, 50K ARC tasks × 8 = **400K sequences, 688 shards, ~530 GB**. Documented flaw that motivated the current branch: `max_seq_length=2048` right-truncated **94.6%** of prompts (median = 8,947 tokens) — the answer fell off the end.
- ✅ **Run 002:** layer 19, grid-chunking, 5120-token chunks → merged to 896-dim, **2,345 shards, ~2.82B tokens** — this is the SAE training corpus.
- 🟡 Run 003 (layers 15+22) scaffolded.
- ⚠️ **Honesty flag:** the "~53k tok/s / ~3B tokens/day" figure that appears in my older notes is **not** recorded anywhere in this repo — treat it as unverified.

## 2.2 SAE training library — JAX/TPU sparse autoencoders
*Worktree `sae-worktree`, branch `feat/sae-training`.*

**Core idea.** Train SAEs on the extracted shards, data-parallel on a pod, preemption-resilient. Four architectures behind one `BaseSAE`: **Vanilla** (ReLU+L1), **TopK** (+ aux dead-neuron loss), **Gated** (DeepMind 2024), **JumpReLU** (Anthropic 2024 — learnable per-feature thresholds via `custom_jvp` straight-through estimators).

**Experiments & results (real, on layer-19 / 896-dim / 2.82B tokens).**
- 🟥 **v1 TopK 8× (dict 7,168, k=32): diverged** — 91.6% EV at 125K steps, then collapsed to **−288% EV** by 200K. Root cause: dead-neuron resampling injecting noise at low LR.
- ✅ **v2 TopK 8× (stop resampling at 75K): the headline result** — at 125K steps on 24M tokens: **MSE 0.1254, explained variance 94.6%, L0 = 32.0** (0.45% sparsity), **6,744/7,168 features alive** (5.9% dead).
- ✅ **Feature analysis (v2):** **20 universal grid-syntax features** (fire on 80%+ samples), 252 task-discriminative, 69 highly selective, 27 sample clusters. Full report: `activation-extract/reports/sae_layer19_evaluation_report.md`.
- 🟡 v3 (TopK 16×) and v4 (JumpReLU) trained, eval pending.
- ✅ **147 test functions / 8 files.**
- *In flight:* promoting the v1→v2 "resampling cutoff" launcher-hack into a first-class `--dead_neuron_resample_until` config flag.

**Blog gold:** the v1→v2 divergence-and-fix is a perfect "interpretability training is finicky" post — a 91.6%→−288% collapse fixed by a single scheduling change.

## 2.3 ARC-AGI Mechanistic Interpretability Study — three architectures, one question
*Hub `arc-agi-research` + 4 worktrees. The most intellectually ambitious project.*

**Core question.** Do structurally different architectures that solve the same abstract visual-reasoning tasks (ARC-AGI-1) **converge on the same internal features, or invent incompatible ones?** Three deliberately maximal contrasts:
- **Autoregressive** (Qwen 2.5-0.5B + LoRA) — causal, left-to-right.
- **Recursive** (TRM, 7M) — one tiny block applied N times to a refining answer-state + latent scratchpad.
- **Masked diffusion** (LLaDA-8B + 2D-RoPE) — bidirectional, unmask-by-confidence.

The payload is a cross-architecture **CKA/RSA** comparison + **SAE feature dictionaries matched across models**, pre-committing to one of three narratives: **Convergence** (universal ARC features), **Divergence** (architecture-specific), or **Mixed** ("layered universality" — shared low-level primitives, architecture-specific high-order decomposition). Careful shared contract: a **canonical 200-task tripwire**, common SAE shard format, staggered TPU schedule, and a "no-fake-numbers" paper build (every quoted number must flow through `results.record(...)` or the PDF renders a loud red `??key??`).

**What actually ran (honest ledger):**
- **Phase 2 (TRM) — the only phase with real outputs.** ✅ Built on a real TRM checkpoint + the ARC1 concept-augmented dataset (960 puzzles, 876,406 puzzle IDs). An **autonomous research loop** (the ~3,700 Claude sessions) ran **37,292 deterministic experiments** covering **200/200 canonical tasks** across three kernels: per-iteration convergence, task-ID-token ablation (KL), and halting. **Real finding:** a clean **dissociation** — e.g. task `2bcee788` has peak KL **132.09 at step 0** (strong task-ID routing) yet **never halts and is never correct**; aggregate **107/200 tasks are never solved at any iteration**. *Meta-finding (also blog-worthy):* the loop's 14k+ proposals collapsed to **~597 unique (experiment, task) cells re-asked ~23.5× each** — an honest case study in autonomous-research-loop failure modes (goldfish memory, no novelty signal), documented in `LOOP_ANALYSIS.md`.
- **Phase 1 (Qwen):** 🟡 scaffold + one 5-sample smoke shard. Experiments A–D (head specialization, probing, SAE, TTT deltas) designed, not run.
- **Phase 3 (LLaDA):** 🟡 a from-scratch JAX/Flax port (~1,330 LOC) with parity tests *by design* — but `model/llada.py.__call__` still raises `NotImplementedError`. No outputs.
- **Phase 4 (paper):** 🟡 a complete, rigorous LaTeX + figure-generator + no-fake-numbers harness — but `results.json == {}`, all 10 figures are placeholders, narrative undecided.

**Status:** rich concept + infrastructure; execution front-loaded on Phase 2 (which produced a genuine result). The cross-architecture payoff (CKA, feature matching) is **not yet run**.

## 2.4 weight-tieing — a research idea (notes only)
🟡 Hypothesis doc, no code: use **weight tying as a cross-scale interpretability probe** — tie embedding, one or more middle layers, and unembedding between a deep and a shallow model to isolate the inner-layer mechanisms small models lack. Compelling, unstarted.

---

# Theme 3 — Systems & Infrastructure

## 3.1 dtmpfs — a distributed RAM filesystem in Rust ⭐
*Branch `main`, 5-crate workspace, ~2,450 LOC Rust.*

**Core idea.** A **distributed tmpfs**: mount the same namespace on several LAN hosts, file data lives in RAM **sharded across nodes**, write on host A and read on host B after `close()`. Target: ML training scratch / shared artifacts on a small trusted cluster. The systems counterpart to the MiniMax zero-egress sharding.

**What I built.** gRPC (tonic/prost) + FUSE (`fuser`): `metasrv` (single authoritative metadata server), `storesrv` (RAM block store with heartbeats + stale-write rejection), `dtmpfs-mount` (FUSE client w/ block cache + replica failover). Design highlights: **HRW/rendezvous hashing** on `(ino, block_idx)` for stable placement (test asserts <2/8 of 1024 blocks move when a node leaves), close-to-open consistency with a per-inode **generation epoch** for cache coherence, 1 MiB `Bytes` blocks fanned out `buffer_unordered(16)`, `flush()` waits primaries / `fsync()` waits all replicas.

**Experiments & results.**
- ✅ **36-case black-box acceptance harness** (real meta+2-store+mount cluster): 200 MiB md5-match read/write, append/truncate/RMW, **10k files in one dir**, depth-30 trees, bad-token→EIO, delete-recreate-new-inode, unicode names, hardlink→EPERM.
- ✅ Unit tests (HRW determinism/disruption) + CI (fmt + build + test + clippy `-D warnings`).
- ✅ Phases P1→P6 shipped (single-process → gRPC split → multi-store sharding → meta service + generation → replication R≥2 → heartbeats/failover/stale-write/GC). Only Raft (P7) unbuilt.
- ⚠️ **Honesty flags:** the `CLAUDE.md` status block is **stale** — it claims Phase 6 isn't implemented, but commit `9f988b1` shipped it (verified in source). The Rust `tests/integration/` dir is **empty** — coverage lives in the shell harness. Unusually thorough design docs (HLD 52 KB, LLD 88 KB, protocol 44 KB).

## 3.2 tpu-bare-metal — driving a TPU from pure C via PJRT ⭐
*Branch `main`. The best self-contained blog piece.*

**Core idea.** Control TPU hardware with **no Python/JAX/TF at runtime** — `dlopen("libtpu.so")`, grab the PJRT C API table, call in directly. The hook is reverse-engineering the undocumented in-memory ABI.

**The discovery.** ✅ `GetPjrtApi()` returns a struct whose header is **40 bytes, not the naive 24** — `struct_size(8) + extension_start(8) + a nested PJRT_Api_Version(24)`, because the version struct itself follows the PJRT convention. The 113 function pointers start at `api_ptr + 40`, accessed by index.

**What I built & verified (all 8 examples compile and pass on v4 hardware).** ✅ A reusable C framework (`tpu.h` public API hiding PJRT types, `tpu_strerror` error handling) demonstrating: device enumeration + host↔TPU roundtrip, HLO compile+execute (`x+x`→`[2,4,…,32] PASSED`, `B@B` matmul PASSED), **multi-device SPMD** across all 4 chips, **device-to-device copy** (no host roundtrip), **async execution** via `OnReady`, **serialize→disk→reload→rerun** executables, buffer sharding, and per-device HBM/topology stats. JAX is used *offline* only, to emit the HLO `.pb` files the C code consumes. Documents that `libtpu.so` exports 226 symbols across three API layers.

**Status:** ✅ working and polished — the most complete of the systems set.

## 3.3 claude-hub — a web control plane for Claude Code sessions
*Branch `feature/ui-stability-mobile-persistence`. In active daily use.*

**Core idea.** Self-hosted web UI to run many Claude Code CLI sessions at once — locally or over SSH, from a browser or **phone**. Each session is a real `tmux` window; the browser attaches over WebSocket to a forked PTY, so sessions survive restarts/reloads/network drops (the killer feature for babysitting long agent runs).

**What I built.** aiohttp + vanilla-JS/xterm.js (no build step). `pty_bridge.py` forks a PTY → `tmux attach` (or `ssh -tt … tmux attach` for remote), fanning output to a scroll buffer, SQLite persistence, a heuristic **harness parser** (`harness/claude_code.py` regexes Claude's `●`/`⎿`/`>` glyphs into structured events), and all WebSocket clients (replaying the last 100 KB on connect). JWT+bcrypt auth (header/query/cookie), an orchestrator for multi-agent task groups, and Docker + a **cloudflared quick-tunnel** sidecar for instant public HTTPS.

**Status:** ✅ working, used in anger — `claude_hub.db` holds **~30 MB** of real session/transcript history. No automated tests (noted gap).

## 3.4 levanter — preemptible-TPU automation + multilingual configs
**Reality:** the framework is an **upstream mirror** (kept current via merges). The custom work is a thin, real automation layer: `helpers/automate_run.py` (TPU lifecycle state machines + retry-until-allocated for preemptible nodes), region-specific systemd units, a latest-checkpoint bugfix, and **multilingual (South-Asian) training configs** (Tamil/Hindi/Bengali/Telugu/Marathi + English; CulturaX + Sangraha + FineWeb-Edu-Hindi; LLaMA-3.2 128K tokenizer). ✅ **Actively running today** — multi-region run logs (v5e EU/US, v6e) are being written with today's date.

---

# Theme 4 — Tools & Experiments

## 4.1 autoresearch-tpu — autonomous LLM-pretraining research loop
**Core idea.** Karpathy's autoresearch loop ported to **TPU v4-8 (JAX/Flax NNX)**: an agent edits the training script, runs a fixed 5-min experiment, reads `val_bpb`, keeps/discards, repeats — with a simplicity criterion (deleting code beats adding hacks).
**TPU-port engineering.** Batch 32→64 to drop gradient accumulation entirely; `@nnx.remat` checkpointing to fit it (attention scores are `f32[64,4,2048,2048]`≈4 GB); kept fwd/bwd and optimizer as **separate JIT calls** because fusing OOM'd (the JIT boundary acts as a memory checkpoint).
**Results.** ✅ Documented baseline→optimized: **val_bpb 1.136→1.115**, MFU 8.71%→9.80%, ~400k→440k tok/s; a depth sweep (8/10/12) showing ~50M params is near-optimal for the 300 s budget. 7 timestamped run logs + `progress.png`. Dormant since March.

## 4.2 rusty-v8 — Clang-AST C++→Rust transpiler for V8 ⭐ (blog-worthy)
**Core idea.** Mechanically translate V8 (~6,000 C++ files) into an idiomatic Rust workspace via a real **Clang AST → JSON IR → rule-based codegen** pipeline (not regex hacking), leaving explicit `todo!()` where the AST can't be safely lowered.
**What I built.** extractor (libclang traversal) → IR (modules/structs/funcs + global type registry + dependency graph) → mapper (classes→struct/trait, templates→generics, `std::`→Rust std, V8 `Handle`/`Tagged`/`Smi` patterns) → codegen, with an **iterative Tarjan SCC** pass to break inter-crate dependency cycles. Real Clang flags (`V8_COMPRESS_POINTERS`, `V8_ENABLE_SANDBOX`, …), 48-subdir→crate map.
**The standout story — AST-roundtrip precedence bugs:** because C++ parenthesization is implicit in the AST and lost on the way back to text, `!(x >= 1)` re-emitted as `!x >= 1` (applies `!` to `x`, wrong type) — fixed by re-wrapping. Plus shift-vs-generics ambiguity (`expr as Type <<`), comparison chaining, and an honest **revert** of an over-aggressive fix.
**Output.** ✅ A real **49-crate / 1,206-`.rs`-file** Cargo workspace — correct structure + signatures, but with `todo!()` gaps and post-processing artifacts (`!!!!IsSupported`, nested `debug_assert!`). Partial PoC; the precedence story is the post.

## 4.3 attenRes-ViT — selective residual stream over depth
**Core idea.** 🟡 A ViT variant replacing the additive residual with **"Attention Residuals"**: each layer computes a softmax-weighted aggregation over *all preceding layers'* outputs (`h_l = Σ_i α_{i→l}·RMSNorm(v_i)`), with zero-init per-layer pseudo-queries so it starts as uniform averaging — i.e. **depth-wise selective attention over the residual stream**.
**Status.** Code-complete + tested (baseline ViT-S/16 + AttnRes variant, CIFAR-100/ImageNet configs, attention-heatmap analysis) but **never trained — no results**. ⚠️ Cites a not-yet-existing arXiv ref; verify before publishing. Strong post *if run*.

## 4.4 tiny-chip-rl — RL chip placement in one file
**Core idea.** ✅ A single-file (12 KB), CPU-only **PPO agent that places macros on a grid to minimize HPWL** (a miniature of Google's RL-for-chip-placement), with legality masking, generalization to fresh netlists, and a random baseline. Complete, runs in minutes. Clean standalone demo.

## 4.5 mailclaw — Gmail triage agent
✅ A working OpenClaw skill: fetches recent Gmail, classifies (important/blog/promo/other) via Gemini, stores in SQLite, serves an important-first dashboard at `127.0.0.1:8082`, runs every 2h via cron. Operational personal tool.

---

# Theme 5 — Multilingual / Hindi NLP (largely off-VM)

Referenced in my bio + the levanter configs; the artifacts live on HuggingFace, so flag as ✅-shipped-elsewhere / not-fully-on-this-VM:
- **FineWeb-Edu-Hindi** — ~300B-token Hindi dataset (IndicTrans2 on TPU v4-256 via TRC), ~2K monthly HF users.
- **Gemma-200M-Hindi** — small Hindi model.
- The levanter multilingual configs above are the on-VM training side of this thread.

---

# Blog post ideas (ranked by "ready to write")

1. **"Running a 230B model on a pod that can't hold it"** — MiniMax-M2.7 zero-egress per-host `.npy` sharding (27 GB/host vs 7.4 TB). Concrete, validated, visual. ⭐
2. **"The 40-byte struct: driving a TPU from pure C"** — the PJRT ABI reverse-engineering. Self-contained, verified on hardware. ⭐
3. **"A 902× gap from one missing collective"** — why v5e can't do MoE decode (ragged-all-to-all) and the `dynamic_slice` prefetch wall (~21× bandwidth gap, odd-batch cliffs). ⭐
4. **"My SAE hit 94.6% EV, then collapsed to −288%"** — the dead-neuron-resampling divergence and the one-line fix.
5. **"When your autonomous research loop asks the same question 23 times"** — the ARC Phase-2 loop meta-finding (LOOP_ANALYSIS), plus the real KL/halt/correctness dissociation it did surface.
6. **"Round-tripping an AST loses your parentheses"** — rusty-v8 operator-precedence bugs.
7. **"Compile time vs preemption window"** — why Gemma-4-31B runs but won't serve on spot TPUs.

# Research ideas (open threads)

- **Cross-scale interpretability via weight tying** (weight-tieing) — tie middle layers between deep/shallow models to expose what small models lack.
- **The ARC three-narrative question** (Convergence/Divergence/Mixed) — still open; needs the CKA + SAE-feature-matching across Qwen/TRM/LLaDA to actually run.
- **AttnRes-ViT** — does depth-wise selective residual attention help, and what do the learned per-layer weights reveal?
- **v5e Pallas grouped-GEMM** — beating the ragged-all-to-all wall with a custom kernel.

---

# Status ledger (validated vs planned)

| Project | Built | Ran experiments | Concrete outputs | Verdict |
|---|:--:|:--:|:--:|---|
| MiniMax-M2.7 | ✅ | ✅ | decode + API + 6,678 tok/s sweep | **validated** |
| MiniMax-M2.5 | ✅ | ✅ | 6,792 tok/s ceiling + perf root-causes | **validated** (point-in-time) |
| Qwen3.6-27B | ✅ | ✅ | KL=0.02 parity on v4-8 | **validated, uncommitted** |
| Gemma-4-31B | ✅ | ✅ | correct decode; serving blocked | **partial** |
| activation-extract | ✅ | ✅ | 2.82B-token corpus, 49 tests | **shipped** |
| SAE training | ✅ | ✅ | 94.6% EV, feature dictionary | **shipped** |
| ARC study (Phase 2) | ✅ | ✅ | 37k runs, KL/halt dissociation | **real result** |
| ARC study (Ph 1/3/4) | 🟡 | ✗ | scaffold + smoke only | **planned** |
| dtmpfs | ✅ | ✅ | 36-case acceptance, P1–P6 | **shipped** (CLAUDE.md stale) |
| tpu-bare-metal | ✅ | ✅ | 8 verified PJRT examples | **shipped** |
| claude-hub | ✅ | ✅ | 30 MB live session DB | **in daily use** |
| levanter (custom) | ✅ | ✅ | live multi-region run logs | **active** |
| autoresearch-tpu | ✅ | ✅ | 1.136→1.115 val_bpb | **dormant** |
| rusty-v8 | ✅ | ✅ | 49-crate / 1,206-file output | **partial PoC** |
| attenRes-ViT | ✅ | ✗ | none (never trained) | **code-complete** |
| tiny-chip-rl | ✅ | 🟡 | runnable, no saved logs | **complete script** |
| mailclaw | ✅ | ✅ | operational tool | **shipped** |
| weight-tieing | ✗ | ✗ | idea doc | **idea** |

---

# Already have a website starting point: `kathir-os`

`~/personal-site-chat/site/kathir-os` is a **complete, deployable** layered terminal-over-GUI portfolio (GitHub Pages + Cloudflare Workers + R2 + Gemini 2.0 Flash). It includes:
- a draggable functional **terminal** (`ls/cd/cat` over a fake filesystem of my work, `neofetch`, themes, easter eggs) over an animated-canvas GUI;
- a Workers **API** (`/feed /status /logs /papers /ask`) with a Gemini system prompt that already encodes my bio + projects;
- **4 cron agents** (GitHub/HF sync, daily digest, arXiv paper monitor, hourly status);
- an **MCP server** so Claude Desktop can query my live state.

Never deployed (API URL is a placeholder, R2 not provisioned). Two paths for the new site: **deploy `kathir-os` as-is** (~30 min: Cloudflare account + Gemini key + fill the API URL), or **lift its frontend + bio/system-prompt** and pair it with the project write-ups in this document.
