---
title: "The 40-byte struct: driving a TPU from pure C"
subtitle: "Reverse-engineering the PJRT C API to run kernels on a TPU with no Python, no JAX, no TensorFlow"
date: 2026-05-31
tags: [tpu, pjrt, c, systems, reverse-engineering, xla]
status: validated   # all 8 examples compile and verify on TPU v4
---

# The 40-byte struct: driving a TPU from pure C

Every normal way to run something on a Google TPU goes through a tall stack: Python → JAX or
TensorFlow → XLA → the PJRT plugin → `libtpu.so` → the hardware. That's a lot of layers to own a few
matmuls. The question this project set out to answer was simple: **can you delete all of them and call
the chip directly from C?**

The answer is yes — `dlopen("libtpu.so")`, grab the PJRT C API function table, and call into it. The
interesting part isn't that it's possible; it's the one piece of undocumented ABI you have to get
exactly right before anything works at all, and the way it crashes until you do.

## The entry point and the table that isn't where you think

`libtpu.so` exports a single bootstrap symbol, `GetPjrtApi`, which returns a pointer to a `PJRT_Api`
struct. That struct is, in effect, a giant vtable: 113 function pointers for everything from creating
a client to executing a compiled program. The entire C interface is "index into this table and call."

```c
void* handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);

typedef void* (*GetPjrtApiFn)(void);
GetPjrtApiFn get_api = (GetPjrtApiFn)dlsym(handle, "GetPjrtApi");

void* api_ptr = get_api();
void** fn     = (void**)((char*)api_ptr + 40);   // <-- the whole trick is this 40
```

That `+ 40` is the entire reverse-engineering story. The naive assumption is that the `PJRT_Api`
header is the usual PJRT preamble — `struct_size` (8 bytes) plus an `extension_start` pointer (8
bytes) — so the function pointers should begin at offset 24. They don't. They begin at **40**, and
until you figure out why, the program reads function pointers that are actually fields of the version
struct, calls them, and segfaults in ways that point nowhere near the real cause. That's the
signature of an ABI bug: the crash is downstream of the mistake by a wide margin.

The reason is that `PJRT_Api` embeds a version struct, and **the version struct itself follows the
PJRT convention** — it leads with its own `struct_size` and `extension_start` before its data:

```
+0:   size_t struct_size          (e.g. 944)
+8:   PJRT_Extension_Base* ext
+16:  PJRT_Api_Version {
        size_t struct_size (=24)
        void*  ext
        int    major (=0)
        int    minor (=69)
      }                           // 24 bytes
+40:  function pointers begin...  // 113 pointers
```

So the header is `8 + 8 + 24 = 40` bytes, and the function table is right after it. The PJRT
convention of "every struct starts with its size" is recursive, and the version struct nests one full
copy of that preamble inside the outer one — which is exactly the 16 extra bytes the naive count
misses. With that single constant correct, the rest of the API becomes a clean indexed dispatch. The
framework encodes the indices as an enum so call sites read normally:

```c
enum {
    FN_ERROR_DESTROY          = 0,
    FN_EVENT_ONREADY          = 9,    /* async completion callback   */
    FN_CLIENT_CREATE          = 10,
    FN_CLIENT_DEVICES         = 15,
    FN_CLIENT_COMPILE         = 20,   /* HLO/MLIR -> executable       */
    FN_DEV_MEMSTATS           = 34,   /* per-device HBM stats         */
    FN_EXEC_SERIALIZE         = 49,   /* save a compiled executable   */
    FN_LEXEC_EXECUTE          = 55,   /* run it                       */
    FN_EXEC_DESERIALIZE_LOAD  = 56,   /* load a saved executable      */
    FN_BUF_COPYTODEV          = 69,   /* device-to-device, no host    */
    FN_CLIENT_TOPODESC        = 95,
};
```

## The other PJRT convention you can't skip

One more thing matters everywhere: every PJRT call takes an `_Args` struct, and **you must set
`args.struct_size = sizeof(args)` before the call.** The plugin uses that field to detect which version
of each arg struct the caller compiled against — it's how PJRT stays ABI-stable as it adds fields — and
it writes its outputs *back into the same struct in place*. Forget to set `struct_size` and you get
silent garbage or a hard crash; expect a return value somewhere it isn't and you read uninitialized
memory. So the framework wraps the whole pattern in a macro, and every call site sets the size, makes
the call through the indexed function pointer, and checks the returned error object. It's tedious, and
hiding it behind one macro is most of what turns "I called a TPU function" into "I have a usable C
library."

## From "it links" to "it computes"

Getting the table offset right earns you device enumeration. The next milestone is actually running
math. PJRT executes compiled XLA programs, not arbitrary C, so the workflow is: produce an HLO
program, compile it with `fn[20]`, upload inputs, execute, download outputs.

Where does the HLO come from without a runtime dependency on JAX? You let JAX emit it *offline, once*.
A small `gen_hlo.py` lowers `x + x` and a matmul to HLO protobufs and dumps the `CompileOptions`. Those
`.pb` files are checked in, and the C program loads them at runtime. JAX is a build-time HLO compiler,
never a runtime dependency — the running binary is pure C against `libtpu.so`, with no Python
interpreter anywhere in the process. That separation is the whole point: you use the big stack to
*produce an artifact*, then run the artifact with nothing but the vendor blob.

The verified results, straight from the example output on **TPU v4**:

```
[4] Executing x + x on TPU...
    Result (x + x):
    [2, 4, 6, 8, 10, 12, 14, 16,
     18, 20, 22, 24, 26, 28, 30, 32]
    Verify: PASSED

[8] Executing B @ B on TPU...
    Result (B @ B):
    [     90    100    110    120  ]
    [    202    228    254    280  ]
    [    314    356    398    440  ]
    [    426    484    542    600  ]
    Verify: PASSED
```

## A real C framework, not just a proof of concept

The first cut was two standalone files that dlopen'd the library and ran an add. The project then grew
into a reusable C framework that hides the PJRT arg-struct ceremony behind a clean public API
(`tpu_init`, `tpu_compile_file`, `tpu_upload_f32`, `tpu_run2`, `tpu_download`), with real error
propagation through `tpu_strerror(ctx)` instead of `exit(1)` scattered everywhere. The split between
`tpu_pjrt.h` (the internal index/struct definitions) and `tpu.h` (the clean public surface that hides
the raw PJRT types) is what makes it something you'd actually build on rather than a one-off script.

On top of it sit eight example programs, each demonstrating one capability and each verifying its own
output on hardware:

- **roundtrip** — enumerate devices, host → TPU → host.
- **compute** — compile and execute HLO (`x+x` and the matmul above).
- **SPMD** (`fn[55]` with `num_devices > 1`) — one compiled kernel running across all 4 chips at once,
  the same single-program-multiple-data model JAX uses, but driven by hand.
- **device-to-device** (`fn[69]`) — copy a buffer chip-to-chip with no host roundtrip, which is the
  primitive collectives are built on.
- **async** (`fn[9]`) — fire execution and get an `OnReady` callback on completion, instead of
  blocking.
- **serialize / load** (`fn[49]` / `fn[56]`) — compile once, persist the executable to disk, reload
  and rerun without recompiling.
- **sharding** — split a buffer across N devices.
- **topology + memstats** (`fn[95]` / `fn[34]`) — read the topology string and per-device HBM usage.

That serialize/load path is quietly the most consequential one. XLA compilation is slow — slow enough
that, in a sibling project, a 40-minute serving compile made a model un-servable on preemptible
hardware because the compile never finished before the slice got reclaimed. Being able to compile
*once*, serialize the executable, and reload it in seconds turns a multi-minute (or multi-*ten*-minute)
cold start into a near-instant one. The bare-metal C path makes that mechanism explicit and reusable —
it's the same lever, exposed directly.

## What's actually down there

Poking at the library turned up more than the documented surface. `libtpu.so` exports **226 symbols
across three API layers**: the modern PJRT C API behind `GetPjrtApi()` (113 functions), a legacy
TensorFlow/TPU C API (the `Tpu*` symbols, roughly 100 functions), and an undocumented SDK API behind
`GetLibtpuSdkApi()`. Only the first is needed here, but knowing the other two exist is useful — they're
where you'd look if PJRT didn't expose something you needed. Tested against **PJRT API v0.69**
(`major=0, minor=69`) on a v4 with a 2×2 chip topology.

## Why bother

Two reasons, beyond the obvious "because it's there." First, it makes the cost of the normal stack
*legible*. Once you've run a matmul in ~200 lines of C, you understand exactly what JAX is doing for
you — and what it's charging in startup latency, dependency weight, and process footprint. The
abstraction stops being magic and becomes a set of choices you can second-guess.

Second, it's a real option for embedded-flavored deployments. A minimal C inference runtime with no
Python interpreter, no multi-gigabyte framework install, just the vendor blob and a thin shim, is an
attractive shape for constrained environments — and it's the natural meeting point of "ML on TPU" and
an embedded-systems background. The serialize-and-reload capability is what makes it practical: ship a
precompiled executable and a tiny C loader, and you have TPU inference with essentially no startup tax
and no Python.

The headline lesson, though, is almost comically small: **the header is 40 bytes, not 24, because the
version struct nests the PJRT preamble inside itself.** One wrong constant stood between "this is
impossible" and "all eight capabilities verify on hardware." Reverse-engineering an ABI is often
exactly this — not a grand insight, but one offset, found the hard way, after a few segfaults that
pointed everywhere except at the real problem.

---

*Code: the `framework/` C library (`tpu.c`, `tpu_pjrt.h`, `tpu_spmd.c`, `tpu_serial.c`) and 8 example
programs, verified on TPU v4, PJRT API v0.69. JAX used offline only, to emit the HLO protobufs the C
binaries consume.*
