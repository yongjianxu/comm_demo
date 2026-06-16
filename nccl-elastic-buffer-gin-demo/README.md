# Mixed-Segment Elastic GIN Demo (NCCL 2.30.7)

A self-contained, single-process, multi-GPU program that builds NCCL symmetric
windows whose virtual address is stitched from **two physical segments — GPU
device memory + host-NUMA memory** — and runs a GPU-initiated (GIN) all-to-all
across them.

It demonstrates the NCCL 2.30.7 elastic-buffer feature
(`NCCL_ELASTIC_BUFFER_REGISTER`): user-controlled per-segment placement, the
multi-segment registration path, and how one verifies where data physically
lives.

Source: [`mixed_segment_gin_demo.cu`](mixed_segment_gin_demo.cu)

---

## What it proves

1. A window can span **device + host** physical memory in one contiguous VA
   (`numSegments == 2`).
2. The user fully controls placement (via the split offset `seg0Size`) — it is
   **not** transparent.
3. Placement is queryable at runtime via the CUDA driver.
4. A GIN `put` crossing the device↔host boundary transfers correctly (verified
   element-by-element).

The VA layout each window is built from:

```
VA:  [ va ............ va+seg0Size ............ va+total )
      '---- segment 0 ----''------- segment 1 -------'
             DEVICE                  HOST_NUMA
```

For any byte offset `o`: `o < seg0Size` ⇒ **device**, `o >= seg0Size` ⇒
**host-NUMA**. The user knows where a tensor lives because they chose
`seg0Size` and where to place each tensor — there is no NCCL API that reports
it.

> The demo intentionally does **not** use `ncclMemAlloc()`: that only ever
> produces single-segment device memory and would never take the elastic path.

---

## Program structure (top to bottom)

### 1. Boilerplate
- `CUDA_DRV_CHECK` / `CUDA_RT_CHECK` / `NCCL_CHECK` error macros.
- `setenv("NCCL_ELASTIC_BUFFER_REGISTER", "1", 0)` early in `main` — the feature
  gate; host segments are rejected with `ncclInvalidArgument` without it.
- `cuInit(0)`, query device count, pick `ndev = min(count, 2)` (overridable by
  argv).

### 2. The mixed-segment allocator — `allocMixed(usefulBytes, dev)`
Returns a `MixedBuffer { va, total, seg0Size, h0, h1, numaNode }`.
- Query `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` for the device's host NUMA node.
- Get allocation granularity for both a DEVICE prop and a HOST_NUMA prop; use
  `G = max(both)`.
- Split: `seg0Size = alignUp(usefulBytes/2, G)` (device),
  `seg1Size = alignUp(usefulBytes - seg0Size, G)` (host);
  `total = seg0Size + seg1Size`.
- `cuMemCreate` **two** handles: seg0 `CU_MEM_LOCATION_TYPE_DEVICE`, seg1
  `CU_MEM_LOCATION_TYPE_HOST_NUMA`.
- `cuMemAddressReserve(&va, total, G, 0, 0)` — **one** VA.
- `cuMemMap(va, seg0Size, …, h0)` then `cuMemMap(va + seg0Size, seg1Size, …, h1)`
  — contiguous, back-to-back.
- `cuMemSetAccess` over the **whole** `[va, va+total)` with **both** a DEVICE and
  a HOST_NUMA access descriptor.

### 3. Placement introspection — `describeAddress(buf, off, label)`
- Computes the **expected** segment from `off < seg0Size ? DEVICE : HOST_NUMA`
  (the user's design-time knowledge).
- Calls `cuMemRetainAllocationHandle` + `cuMemGetAllocationPropertiesFromHandle`
  on `va+off` for the **driver's ground truth** (`prop.location.type`,
  `prop.location.id`), then `cuMemRelease`.
- Logs both and flags agreement. This is the "how do you know where the tensor
  lives" answer made executable. (Same driver calls NCCL uses internally in
  `ncclDevrValidateHandleLocationType` / `ncclDevrBuildGinSegmentInfos`.)

### 4. Sizing for a meaningful boundary
- Each segment is forced to ≥ `G` (~2 MB) by granularity. The default
  `elemsPerPeer` (1,048,576 ints) makes `usefulBytes` several × `G`, so the
  per-peer chunks land on **both** sides of the boundary rather than all inside
  segment 0.
- After allocation the demo calls `describeAddress` on each per-peer send chunk,
  so the output explicitly shows chunks on device vs host.

### 5. Init & verification (no raw CPU access to the device half)
- The device segment is not CPU-addressable, so the demo cannot touch the buffer
  through a raw CPU pointer, and a single `cudaMemcpy` cannot span both segments
  (the driver picks one direction from the base pointer's memory type). The
  `copyMixed()` helper therefore **splits at `seg0Size`**: `cudaMemcpy` for the
  device half, plain `memcpy` for the CPU-mapped HOST_NUMA half.
- Pattern `send[peer*elems + i] = rank*1e6 + peer*1e4 + i` encodes src/dst/index
  for unambiguous checking. Readback after the collective is the reverse split.

### 6. NCCL setup
- `ncclCommInitAll`; per rank `ncclCommQueryProperties`, bailing cleanly if
  `!deviceApiSupport || ginType == NCCL_GIN_TYPE_NONE`.
- All ranks allocate the **same** `usefulBytes` ⇒ identical `seg0Size`/`seg1Size`
  — **required** by `ncclDevrVerifySegmentLayouts`, which rejects windows whose
  per-segment sizes differ across ranks.
- Register inside one `ncclGroupStart/End` with
  `ncclCommWindowRegister(..., NCCL_WIN_COLL_SYMMETRIC)`.
- Create the device comms for **all ranks inside one `ncclGroupStart/End`** with
  `ginConnectionType = NCCL_GIN_CONNECTION_FULL`. This is mandatory:
  `ncclDevCommCreate` with FULL connections is a cross-rank rendezvous, so in a
  single-process multi-GPU program issuing it sequentially per rank deadlocks
  (rank 0 blocks waiting for rank 1). See *Gotchas* below.

### NIC ↔ GPU ↔ host-segment affinity
GIN here runs over **IB GDAKI** (GPUDirect-async RDMA over InfiniBand/RoCE), so
NIC locality matters. `bindToDeviceNuma()` binds each rank's host thread (and its
default allocations) to the GPU's NUMA node, which co-locates three things on one
socket:
- the **GPU**,
- the **HOST_NUMA segment** (allocated on `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID`), and
- the **GIN NIC**, which NCCL selects per-comm from topology via
  `ncclTopoGetLocalGinDevs()`.

A NIC on the wrong NUMA node would make every GIN access to the host segment
cross the inter-socket link. The demo prints the affinity triple per rank;
confirm in `NCCL_DEBUG=INFO` `NET/IB` lines that the chosen device [0] is the
PIX-adjacent NIC for the GPU (e.g. GPU0 → `mlx5_3`).

> We deliberately do **not** set `NCCL_GIN_HCA` / `NCCL_IB_HCA`: in a
> single-process multi-GPU job those envs are process-global and would pin *all*
> ranks to one NIC, breaking the per-GPU affinity NCCL otherwise gets right.

### 7. Device kernel — `mixedGinAllToAllKernel`
- Standard GIN all-to-all: a `ncclGinBarrierSession` acquire-sync, each thread
  issues `gin.put(world, peer, recvWin, rank*bytes, sendWin, peer*bytes, bytes,
  WeakSignalInc{...}, …)`, then `waitSignal` for `nRanks` arrivals, `gin.flush`,
  release-sync.
- **Key line:** the segment tag is **`ncclGin_SegmentMixed{}`** (not
  `SegmentDevice`/`SegmentHostNuma`). The runtime's `findSegmentFromWindow` /
  `advanceSegmentCursor` handle boundary-crossing internally, so a single
  `bytes` transfer that straddles `seg0Size` needs no manual offset splitting.

### 8. Teardown
- Sync, verify, print `PASSED/FAILED`, then `ncclDevCommDestroy`, grouped
  window deregister, `freeMixed` (unmap whole range, free VA, release both
  handles), `ncclCommFinalize/Destroy`.

---

## Build & run

```sh
make                       # builds mixed_segment_gin_demo
make run                   # runs with NCCL_DEBUG=INFO NCCL_ELASTIC_BUFFER_REGISTER=1
# or directly:
./mixed_segment_gin_demo [num_devices] [elems_per_peer]
```

Gencode covers `sm_90` (Hopper, e.g. H20/H100) and `sm_100` (Blackwell, e.g.
B200); override `NVCC_GENCODE` if needed. Requires a CUDA runtime/driver with
host-NUMA VMM support (CUDA ≥ 12.x) and an NCCL build with Device API + GIN.

---

## Expected output (success criteria)

- Allocator logs two segments with sizes and the host NUMA node.
- `describeAddress` lines showing device vs host chunks, driver-confirmed, with
  `OK`.
- NCCL `INFO` shows **2-segment** registration with **no**
  "set NCCL_ELASTIC_BUFFER_REGISTER=1" warning (confirms the multi-segment /
  `numGinSegments > 1` path ran).
- Final line: `mixed-segment GIN all-to-all: PASSED`.

---

## Gotchas (learned by running it)

These are the three things that break a mixed device+host window and how the
demo handles each:

1. **`cuMemSetAccess` over the whole range fails** with `NOT_SUPPORTED` — you
   cannot grant a `HOST_NUMA` accessor on a device-backed segment. Set access
   **per segment**: device segment → DEVICE accessor only; host segment →
   DEVICE + HOST_NUMA accessors.
2. **A single `cudaMemcpy` cannot span both segments** — it segfaults walking
   off the end of one memory type. Split the copy at `seg0Size` (see
   `copyMixed`).
3. **Sequential `ncclDevCommCreate` deadlocks** with
   `NCCL_GIN_CONNECTION_FULL` in single-process multi-GPU — it is a cross-rank
   rendezvous. Issue it for all ranks inside one `ncclGroupStart/End`.

## Notes & knobs

- **Split ratio:** currently 50/50 device/host in `allocMixed`. For a more
  realistic "hot data on GPU, overflow on host" scenario, change `seg0Size` to a
  device-heavy split (e.g. 75/25). Keep the formula identical across ranks.
- **Single-node only:** this does not exercise the MNNVL multi-node branch
  (`CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED`), which adds extra
  constraints for host-backed buffers across nodes.
