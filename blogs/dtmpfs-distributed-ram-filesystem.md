---
title: "dtmpfs: a distributed RAM filesystem in Rust"
subtitle: "Mount the same namespace on every host, shard the bytes across the cluster's RAM"
date: 2026-05-31
tags: [rust, distributed-systems, filesystem, fuse, grpc, tpu]
status: shipped   # P1-P6 implemented; 35 automated acceptance cases pass
---

# dtmpfs: a distributed RAM filesystem in Rust

When you run a model across a pod of TPU hosts, you keep wanting a filesystem that several
machines can share, that's as fast as RAM, and that you don't have to care about losing. Local
`tmpfs` is fast but not shared. NFS is shared but slow and operationally heavy. A real
distributed store like Ceph is the right answer for durable data and badly over-built for
"scratch space that disappears when the job ends."

`dtmpfs` is the thing in the gap: a **distributed tmpfs**. Mount the same namespace on several
LAN hosts; file *data* lives in RAM, sharded across storage nodes; write a file on host A and,
once you `close()` it, read it on host B. The `t` is for *transient* — it is explicitly **not**
durable, not strongly consistent, and not a SAN replacement. It's RAM-backed scratch for a
small, trusted cluster (2–8 nodes) where RAM is cheap and disk is too slow.

## Shape of the system

Five Rust crates, gRPC over HTTP/2 (tonic/prost) for the wire, FUSE (`fuser`) for the mount:

| crate | binary | role |
|---|---|---|
| `dtmpfs-proto` | — | tonic-generated stubs from the `.proto` files |
| `dtmpfs-common` | — | shared types, **rendezvous hashing**, config, errors |
| `dtmpfs-meta` | `metasrv` | the single authoritative metadata server |
| `dtmpfs-store` | `storesrv` | a RAM block store; heartbeats, stale-write rejection |
| `dtmpfs-client` | `dtmpfs-mount` | the FUSE client; block cache, flush/fsync, failover |

Data is chunked into **1 MiB blocks**. Three design decisions carry most of the weight:
*where* a block lives, *when* a reader is allowed to see a write, and *what happens* when a
store node dies mid-read.

## Placement: rendezvous hashing, generation deliberately excluded

Every block has a logical key `(inode, block_idx)`. To decide which store node(s) own it,
dtmpfs uses **HRW (highest-random-weight / rendezvous) hashing**: score every node against the
key, rank, take the top `r`. The score is two xxh3 hashes (one per 64-bit half of the key)
XORed together:

```rust
pub fn pick_nodes(key: &BlockKey, nodes: &[NodeId], r: usize) -> Vec<NodeId> {
    if nodes.is_empty() { return Vec::new(); }
    let r = r.min(nodes.len());
    let pk = key.placement_key();    // u128: ino<<64 | block_idx
    let lo = pk as u64;
    let hi = (pk >> 64) as u64;

    let mut scored: Vec<(u64, &NodeId)> = nodes.iter()
        .map(|n| {
            let mut h = xxh3_64_with_seed(n.as_str().as_bytes(), lo);
            h ^= xxh3_64_with_seed(n.as_str().as_bytes(), hi);
            (h, n)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    scored.into_iter().take(r).map(|(_, n)| n.clone()).collect()
}
```

The subtle, important choice is that the block's **generation is deliberately left out of
`placement_key`**. If placement depended on generation, every rewrite would move the block to a
different node — pure overhead. Excluding it means a block stays put across rewrites, and
membership changes move only the minimum. There's a test that pins exactly this property: take
8 nodes, drop one, and assert that fewer than `2/8` of 1024 blocks change owner.

```rust
#[test]
fn minimal_disruption() {
    let n8 = nodes(8);
    let n7: Vec<_> = n8.iter().cloned().filter(|n| n.as_str() != "store-3").collect();
    let mut moved = 0usize;
    for i in 0..1024 {
        let k = BlockKey { ino: InodeId(1), block_idx: BlockIdx(i), generation: Generation(0) };
        if pick_nodes(&k, &n8, 1)[0] != pick_nodes(&k, &n7, 1)[0] { moved += 1; }
    }
    assert!(moved < 1024 * 2 / 8, "moved={moved}");
}
```

## Consistency: close-to-open, with a per-inode generation epoch

dtmpfs gives you **close-to-open** semantics — the same contract NFS made famous. Writes you
made before `close()` are guaranteed visible to anyone who `open()`s afterward; concurrent
writers to the same file are not coordinated. That's the honest, useful guarantee for scratch
space, and it's cheap to implement.

The mechanism is a per-inode **generation** counter that doubles as a cache-coherence epoch.
The metadata server bumps it on every `close()` that actually flushed dirty blocks, guarded by
the generation the client *thought* it had (optimistic concurrency):

```rust
if !r.written_block_idxs.is_empty() {
    let inode = s.inodes.get(&ino).ok_or_else(not_found)?;
    if inode.generation != Generation(r.expected_generation) {
        return Err(Status::failed_precondition("generation mismatch"));
    }
    // ... update size/mtime ...
    let inode = s.inodes.get_mut(&ino).unwrap();
    inode.generation = inode.generation.bump();
}
```

Clients key their block cache on `(ino, generation, block_idx)`, so a generation bump silently
invalidates every stale cached block for that file — no explicit invalidation messages needed.

The same generation rides along on writes to the store nodes, which use it to **reject stale
writes**: a node tracks a high-water generation per logical block and refuses anything older,
so a delayed write from a previous epoch can't clobber newer data.

```rust
let logical = (key.ino, key.block_idx);
if let Some(hw) = self.state.high_water.get(&logical) {
    if key.generation < *hw {
        return Err(Status::failed_precondition("stale generation"));
    }
}
// ... store the block ...
self.state.high_water.entry(logical)
    .and_modify(|hw| { if key.generation > *hw { *hw = key.generation; } })
    .or_insert(key.generation);
```

## Failure: read-failover across replicas

With replication factor `r ≥ 2`, a block lives on a primary plus replicas. The client reads from
the primary, and on *any* transport error evicts that cached channel and falls through to the
next replica — returning a zero block for a genuinely-absent (sparse) block rather than an
error:

```rust
for node in std::iter::once(&placement.primary).chain(placement.replicas.iter()) {
    let mut client = match self.stores.get(node).await {
        Ok(c) => c, Err(e) => { last_err = e; continue; }
    };
    match client.read_block(req).await {
        Ok(r) => { /* cache + return */ }
        Err(s) if s.code() == tonic::Code::NotFound => {
            return Ok(Bytes::from(vec![0u8; self.block_size]));   // sparse block
        }
        Err(s) => {
            self.stores.evict(node);   // reconnect next time
            tracing::warn!(node = node.as_str(), "read_block failed, trying next replica");
            continue;
        }
    }
}
Err(last_err)
```

## How it's tested

Because a filesystem is exactly the kind of thing that's "obviously correct" and then eats your
data, the test story leans hard on a black-box harness rather than just unit tests. There are 4
Rust unit tests (all on the placement hashing — determinism, capping, the minimal-disruption
property above), and the real coverage is a shell **acceptance harness that spins up a live
cluster** (one meta, two stores, a mount) and runs **35 automated cases** (the full spec
enumerates 79). A sampling:

- write a **200 MiB** file and assert the md5 round-trips byte-for-byte;
- create **10,000 files in one directory** and count them back;
- build a **depth-30 directory tree**;
- append / truncate / read-modify-write; chmod/chown/stat; delete-then-recreate yields a new
  inode; zero-byte files; unicode names;
- `ln` (hardlink) returns **EPERM** — dtmpfs doesn't pretend to support what it can't.

CI runs fmt + build + the full workspace test + clippy with `-D warnings` on every push.

## State and honesty

Phases P1 through P6 are implemented and in the tree: single-process FUSE → client/store split
over gRPC → multi-store sharding → metadata service with generation + attribute cache →
replication → heartbeats, stale-write rejection, read-failover, and a GC sweep. Only P7 (a Raft
consensus layer to remove the single-metadata-server SPOF) is unbuilt — and for a transient
scratch FS, that SPOF is an acceptable v1 trade.

Two caveats worth stating plainly, because the repo's own docs drifted: the `CLAUDE.md` status
block still claims Phase 6 isn't done — it is, and you can read the failover and stale-write code
above. And the Rust `tests/integration/` directory is empty; integration coverage genuinely lives
in the shell acceptance harness, not in `#[test]` functions. Good to know before you trust a green
`cargo test` to mean "the cluster works."

---

*Code: the `dtmpfs` workspace — `metasrv`, `storesrv`, `dtmpfs-mount`. Design docs (HLD/LLD/
protocol) and the acceptance harness live alongside the crates.*
