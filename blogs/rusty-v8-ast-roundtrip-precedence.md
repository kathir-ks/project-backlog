---
title: "Round-tripping an AST loses your parentheses"
subtitle: "Transpiling V8 from C++ to Rust through Clang, and the precedence bugs that live in the gap between tree and text"
date: 2026-05-31
tags: [compilers, clang, rust, transpiler, ast, v8]
status: partial-poc   # full pipeline runs; output is a compiling skeleton with gaps
---

# Round-tripping an AST loses your parentheses

V8 is about 6,000 files of dense C++. The idea behind this project was to translate it,
mechanically, into an idiomatic multi-crate Rust workspace — not with regex hacks, but through a
real Clang AST front-end. Parse each translation unit with libclang, lower the AST to a JSON
intermediate representation, map construct-by-construct (classes → struct + trait, templates →
generics, `std::` → Rust std, V8's `Handle`/`Tagged`/`Smi` patterns to their analogues), then
emit Rust. Where the AST can't be safely lowered, emit an explicit `todo!()` rather than
plausible-but-wrong code.

The pipeline runs end to end and produces a **49-crate, 1,206-file** Cargo workspace. It is not
semantically complete — there are **~22,000 `todo!()` stubs** and a fair amount of cleanup left.
But the most *interesting* part isn't the scale; it's a category of bug that turns out to be
fundamental to the whole approach, and is a clean little lesson about what an AST is and isn't.

## The pipeline

Four stages, each a small package:

- **extractor** — `ast_parser.py` walks the libclang cursor tree; `type_collector.py`,
  `stmt_converter.py`, `include_resolver.py` gather the rest. Clang is invoked with V8's real
  build flags (`-DV8_COMPRESS_POINTERS`, `-DV8_ENABLE_SANDBOX`, `-DV8_TARGET_ARCH_X64`,
  `-DV8_ENABLE_WEBASSEMBLY`, …) so the AST matches a real build.
- **ir** — `nodes.py` (modules/structs/functions/enums), a global `type_registry.py` spanning
  translation units, and a `dependency_graph.py`.
- **mapper** — `type_mapper`, `class_mapper`, `template_mapper`, `stmt_mapper`, `stdlib_mapper`,
  and a V8-specific `v8_mapper`.
- **codegen** — `rust_emitter.py`, `module_emitter.py`, `cargo_emitter.py`.

One nice systems detail: V8's modules have circular `#include` dependencies, but a Cargo
workspace must be a DAG. `main.py` scans the includes to build a dependency graph and runs an
**iterative Tarjan SCC pass** to find and break the cycles before emitting crates:

```python
cycles = _break_dependency_cycles(all_deps)
if cycles:
    print(f"  Broke {len(cycles)} dependency cycle(s): {cycles}")
```

## The bug that's actually about ASTs

Here's the thing nobody warns you about when you decide to go `source → AST → source`: **an
abstract syntax tree has already thrown away your parentheses.** Parentheses exist in source text
to disambiguate precedence; once the parser has built the tree, the *structure* encodes the
precedence and the parens are redundant — so they're gone. When you walk back down from tree to
text, you have to *re-derive* where parentheses are needed. Miss one and you've silently changed
the meaning of the program.

The cleanest example is logical NOT. In C++:

```cpp
if (!(x >= 1)) { ... }   // negate the comparison
```

The AST records "unary-not applied to the comparison `x >= 1`." A naive emitter prints the
operands left to right and produces:

```rust
if !x >= 1 { ... }       // WRONG: this is (!x) >= 1
```

In Rust `!` binds tighter than `>=`, so `!x >= 1` applies `!` to `x` (a type error on most
operands) and then compares — completely different from negating the comparison. The fix is to
re-insert the parentheses the AST discarded, as a post-processing pass on the emitted text:

```python
# Fix `!expr op val` -> `!(expr op val)` (NOT-operator precedence)
# In C++, `!(x >= 1)` sometimes loses its parens during AST conversion.
source = re.sub(r'\bif\s+!(\w+)\s*(>=|<=|>|<|==|!=)\s*(\S+)',  r'if !(\1 \2 \3)',  source)
source = re.sub(r'\bwhile\s+!(\w+)\s*(>=|<=|>|<|==|!=)\s*(\S+)', r'while !(\1 \2 \3)', source)
source = re.sub(r'([^a-zA-Z0-9_])!(\w+)\s*(>=|<=|>|<)\s*(\w+)',  r'\1!(\2 \3 \4)',  source)
```

The same family of bug recurs anywhere Rust and C++ disagree on precedence or tokenization. The
other one worth showing is the shift-vs-generics ambiguity: `expr as Type <<` — a cast followed
by a left shift — looks to the Rust grammar like the start of a generic argument list (`Type<…>`),
so it has to be parenthesized:

```python
# `value as Type <<` parses as generics; wrap the cast
source = re.sub(r'(\w+)\s+as\s+(\w+)\s*<<', r'(\1 as \2) <<', source)
source = re.sub(r'(\w+)\s+as\s+(\w+)\s*>>', r'(\1 as \2) >>', source)
# C++ operator overloads have no Rust spelling; rename them
source = re.sub(r'\boperator<<', 'shift_left', source)
source = re.sub(r'\boperator>>', 'shift_right', source)
```

A telling artifact of iterating on these: one commit is an honest *revert* of an over-aggressive
comparison-chaining fix. Precedence repair via regex is a game of whack-a-mole, and sometimes the
mole you whack is load-bearing.

## What the output actually looks like

It's worth being honest about the state of the generated Rust, because "transpiled V8 to Rust" can
sound more finished than it is. Here's a real emitted function — every characteristic artifact in
one place:

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

You can read the whole pipeline's current limits off that one snippet: stacked negations
(`!!!!`), nested macro emission (`debug_assert!(debug_assert!(...))`), an identifier that got
lowercased into something that no longer resolves (`cpufeatures::issupported`), and the honest
`todo!()` stubs where an overloaded-declaration reference couldn't be lowered. The crate
*structure* and function *signatures* are right; the bodies are a skeleton with marked gaps.

## Where it stands, and why it's worth writing about

This is a proof of concept, and an honest one: the transpiler runs over the full V8 tree and
emits a 49-crate workspace that builds as a skeleton, but the bodies are far from
compile-and-run. The value isn't a working V8-in-Rust; it's two things.

First, it's a concrete demonstration that **AST → source is harder than source → AST**, and
specifically that *re-deriving precedence* is a named, recurring problem with a real cost. If
you're building any code-to-code translator, budget for a precedence-repair pass and expect it to
be whack-a-mole.

Second, it maps the boundary between what a rule-based transpiler handles cleanly (structure,
types, the dependency DAG) and what it can't (overload resolution, idiom translation, anything
needing semantic understanding rather than syntactic rewriting) — which is exactly the boundary
where an LLM-assisted pass earns its keep. The `todo!()` markers aren't failures; they're a
precise to-do list of where mechanical translation ends.

---

*Pipeline in `transpiler/` (extractor → ir → mapper → codegen). Output: `output/transpiled/`,
49 crates / 1,206 `.rs` files / ~22k `todo!()` stubs. Precedence fixes in
`codegen/rust_emitter.py`.*
