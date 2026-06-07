---
title: "Round-tripping an AST loses your parentheses"
subtitle: "Transpiling V8 from C++ to Rust through Clang, and the precedence bugs that live in the gap between tree and text"
date: 2026-05-31
tags: [compilers, clang, rust, transpiler, ast, v8]
status: partial-poc   # full pipeline runs; output is a compiling skeleton with gaps
---

# Round-tripping an AST loses your parentheses

V8 is about 6,000 files of dense C++. The idea behind this project was to translate it,
mechanically, into an idiomatic multi-crate Rust workspace — not with regex hacks over raw source,
but through a real Clang AST front-end. Parse each translation unit with libclang, lower the AST to
a JSON intermediate representation, map it construct-by-construct (classes → struct + trait,
templates → generics, `std::` → Rust std, V8's `Handle`/`Tagged`/`Smi` machinery to their
analogues), then emit Rust. Where the AST can't be safely lowered, emit an explicit `todo!()` rather
than plausible-but-wrong code.

The pipeline runs end to end and produces a **49-crate, 1,206-file** Cargo workspace. It is not
semantically complete — there are roughly **22,000 `todo!()` stubs** and a fair amount of cleanup
left. But the most *interesting* part isn't the scale; it's a category of bug that turns out to be
fundamental to the whole approach, and is a clean little lesson about what an AST is and isn't.

## The pipeline, stage by stage

Four stages, each a small Python package, and the boundaries between them matter:

- **extractor** — `ast_parser.py` walks the libclang cursor tree; `type_collector.py`,
  `stmt_converter.py`, and `include_resolver.py` gather the rest. Clang is invoked with V8's real
  build flags so the AST matches an actual build configuration rather than some idealized one:
  `-DV8_COMPRESS_POINTERS`, `-DV8_31BIT_SMIS_ON_64BIT_ARCH`, `-DV8_ENABLE_SANDBOX`,
  `-DV8_TARGET_ARCH_X64`, `-DV8_OS_LINUX`, `-DV8_OS_POSIX`, `-DV8_ENABLE_WEBASSEMBLY`, and friends.
  This is non-negotiable: the same header parsed under different defines produces a *different* tree
  (different `#if` branches taken, different type sizes), so pinning the flags pins the meaning.
- **ir** — `nodes.py` defines the module/struct/function/enum node types; `type_registry.py` is a
  *global* registry spanning translation units (so a type defined in one file can be referenced from
  another); `dependency_graph.py` records who-includes-whom.
- **mapper** — the translation rules: `type_mapper` (C++ types → Rust types), `class_mapper` (a class
  hierarchy → struct + trait), `template_mapper` (templates → generics), `stmt_mapper`,
  `stdlib_mapper` (`std::vector`, `std::string`, … → Rust std), and a V8-specific `v8_mapper` for the
  engine's own idioms — `Handle<T>`, `Tagged<T>`, `Smi`, the garbage-collected pointer wrappers that
  are everywhere in V8 and mean nothing to a generic C++→Rust pass.
- **codegen** — `rust_emitter.py` produces the source text, `module_emitter.py` lays out modules,
  `cargo_emitter.py` writes the `Cargo.toml`s.

One systems detail is genuinely nice. V8's modules have **circular `#include` dependencies** — file A
includes B which (transitively) includes A — which is completely legal in C++ but illegal in a Cargo
workspace, where the crate dependency graph must be a DAG. `main.py` builds the dependency graph from
the includes and runs an **iterative Tarjan strongly-connected-components pass** to find the cycles,
then removes back-edges within each multi-node SCC so the emitted workspace is acyclic:

```python
cycles = _break_dependency_cycles(all_deps)   # _tarjan_sccs(graph) under the hood
if cycles:
    print(f"  Broke {len(cycles)} dependency cycle(s): {cycles}")
```

That's the kind of problem you don't anticipate until the Rust compiler refuses your workspace, and
it's a good example of an impedance mismatch between the two languages' module systems that has
nothing to do with any individual line of code.

## The bug that's actually about ASTs

Here's the thing nobody warns you about when you decide to go `source → AST → source`: **an abstract
syntax tree has already thrown away your parentheses.** Parentheses exist in source *text* to
disambiguate operator precedence. Once the parser has built the tree, the *structure* of the tree
encodes that precedence unambiguously, so the parentheses are redundant and the parser discards
them — they were scaffolding, and the building is up. The trouble is that when you walk back *down*
from tree to text, you have to *re-derive* where parentheses are needed, and the rules for that
depend on the precedence table of the **target** language, which is not the source language.

The cleanest example is logical NOT. In C++:

```cpp
if (!(x >= 1)) { ... }   // negate the comparison
```

The Clang AST records "unary-not applied to the comparison node `x >= 1`." There are no parentheses
in the tree; there's nesting, which means the same thing. A naive emitter walks the unary-not, prints
`!`, then walks its operand and prints `x >= 1`, yielding:

```rust
if !x >= 1 { ... }       // WRONG: this parses as (!x) >= 1
```

In Rust `!` binds tighter than `>=`, so `!x >= 1` applies `!` to `x` (a type error on most operands,
or worse, a silently-valid bitwise-not on integers) and *then* compares — a completely different
program from negating the whole comparison. C++ and Rust happen to agree on this precedence, but the
*nesting in the AST* still has to be turned back into *explicit parens in Rust text*, and the naive
walk doesn't do it. The fix re-inserts the parentheses the tree discarded, as a post-processing pass
over the emitted source:

```python
# Fix `!expr op val` -> `!(expr op val)` (NOT-operator precedence)
# In C++, `!(x >= 1)` sometimes loses its parens during AST conversion.
source = re.sub(r'\bif\s+!(\w+)\s*(>=|<=|>|<|==|!=)\s*(\S+)',  r'if !(\1 \2 \3)',  source)
source = re.sub(r'\bwhile\s+!(\w+)\s*(>=|<=|>|<|==|!=)\s*(\S+)', r'while !(\1 \2 \3)', source)
source = re.sub(r'([^a-zA-Z0-9_])!(\w+)\s*(>=|<=|>|<)\s*(\w+)',  r'\1!(\2 \3 \4)',  source)
```

The same family of bug recurs anywhere Rust and C++ disagree on precedence *or on tokenization*. The
tokenization case is even sneakier, because there the two languages don't just rank operators
differently — they lex the same characters into different tokens. The standout is the
shift-versus-generics ambiguity: `expr as Type <<` is, in C++, a cast followed by a left shift, but
to the Rust grammar `Type <` looks like the *start of a generic argument list* (`Type<...>`). The
emitter has to parenthesize the cast to force the shift reading:

```python
# `value as Type <<` parses as generics; wrap the cast
source = re.sub(r'(\w+)\s+as\s+(\w+)\s*<<', r'(\1 as \2) <<', source)
source = re.sub(r'(\w+)\s+as\s+(\w+)\s*>>', r'(\1 as \2) >>', source)
# C++ operator overloads have no Rust spelling; rename them to methods
source = re.sub(r'\boperator<<', 'shift_left', source)
source = re.sub(r'\boperator>>', 'shift_right', source)
source = re.sub(r'\boperator!',  'not_op',     source)
```

A telling artifact of iterating on these: one commit is an honest *revert* of an over-aggressive
comparison-chaining fix that had started mangling correct code. Precedence repair via regex is a game
of whack-a-mole, and sometimes the mole you whack is load-bearing. That tension — every new
post-processing rule fixes one pattern and risks breaking another — is itself the argument against
doing this purely with text rewriting, and for eventually moving the precedence logic *up* into the
emitter where it has the tree and can reason structurally.

## What the output actually looks like

It's worth being blunt about the state of the generated Rust, because "transpiled V8 to Rust" can
sound far more finished than it is. Here's a real emitted function with every characteristic artifact
in one place:

```rust
fn i8x16_splat_pre_avx2<Op>(dst: XMMRegister, src: Op, scratch: XMMRegister) {
    loop {
        if !!!!IsSupported(AVX2) {
            debug_assert!(debug_assert!(!CpuFeatures::IsSupported(AVX2)));
        }
        if !(debug_assert!(!cpufeatures::issupported(avx2))) {
            break;
        }
    }
    todo!("TODO: manually translate (CursorKind.OVERLOADED_DECL_REF): Movd");
    todo!("TODO: manually translate (CursorKind.OVERLOADED_DECL_REF): Xorps");
}
```

You can read the whole pipeline's current limits off that one snippet. The **stacked negations**
(`!!!!`) are a precedence/post-processing pass that ran more than once. The **nested macro emission**
(`debug_assert!(debug_assert!(...))`) is the mapper translating a C++ `DCHECK` macro whose expansion
already contained an assert. The **lowercased identifier** (`cpufeatures::issupported`) is an
over-eager identifier-normalization rule that turned a real type/function name into something that no
longer resolves. And the `todo!()` stubs are the honest part: where an `OVERLOADED_DECL_REF` — a
reference to an overloaded function that needs *semantic* overload resolution to disambiguate —
couldn't be lowered, the emitter refuses to guess. The crate *structure* and function *signatures*
are right; the bodies are a skeleton with marked gaps.

## Two pipelines, and the boundary between them

The repo also contains an earlier, alternative approach worth mentioning: an **LLM-assisted
converter** (a `pipeline/` with rotating free-tier Gemini/OpenRouter providers) that hands chunks of
C++ to a language model and asks for Rust. The two approaches map cleanly onto the two halves of the
problem. The rule-based Clang pipeline is *excellent* at the mechanical, structural work —
parsing under the right defines, building the global type registry, breaking the include cycles into
a valid crate DAG, getting signatures right. It is *bad* at the semantic work — overload resolution,
idiom translation, anything that needs understanding rather than rewriting. The `todo!()` markers are
quite literally a precise, machine-generated to-do list of exactly where mechanical translation ends
and judgment begins — which is the natural seam to hand to an LLM pass.

## Where it stands, and why it's worth writing about

This is a proof of concept, and an honest one: the transpiler runs over the full V8 tree and emits a
49-crate workspace that stands up as a skeleton, but the bodies are far from compile-and-run. The
value isn't a working V8-in-Rust; it's two things.

First, it's a concrete demonstration that **AST → source is harder than source → AST**, and
specifically that *re-deriving precedence and re-resolving tokenization* in the target language is a
named, recurring, costly problem. If you build any code-to-code translator, budget for a
precedence-repair pass, expect it to be whack-a-mole, and plan to push it into the emitter (where the
tree is) rather than leaving it as regex over text (where the tree is gone).

Second, it maps the boundary between what a rule-based transpiler handles cleanly — structure, types,
the dependency DAG — and what it fundamentally can't — overload resolution, idiom translation,
semantic judgment. That boundary is exactly where an LLM-assisted pass earns its keep, and the
`todo!()` stubs are a ready-made specification of where to point it. The most interesting transpiler
isn't pure rules or pure LLM; it's rules for the structure and a model for the seams, with the
machine-generated to-do list connecting them.

---

*Pipeline in `transpiler/` (extractor → ir → mapper → codegen). Output: `output/transpiled/`,
49 crates / 1,206 `.rs` files / ~22k `todo!()` stubs. Precedence and tokenization fixes in
`codegen/rust_emitter.py`; cycle-breaking in `main.py`.*
