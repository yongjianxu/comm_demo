/*************************************************************************
 * Host-only elastic GIN demo for NCCL 2.30.7.
 *
 * Each NCCL symmetric window is backed by a SINGLE contiguous virtual
 * address range stitched from TWO physical CUDA VMM segments that are
 * BOTH host-backed (CU_MEM_LOCATION_TYPE_HOST_NUMA):
 *
 *     VA:  [ va ............ va+seg0Size ............ va+total )
 *           '---- segment 0 ----''------- segment 1 -------'
 *               HOST_NUMA                HOST_NUMA
 *
 * This shows that the elastic-buffer feature works with NO device pages at
 * all: the window is still multi-segment (numSegments == 2), still gated by
 * NCCL_ELASTIC_BUFFER_REGISTER, and a GPU-initiated (GIN) all-to-all crosses
 * the segment boundary -- but every byte lives in CPU memory.
 *
 * Because the whole VA is CPU-addressable, init/verify use a plain memcpy
 * (no device/host split, unlike the mixed demo).  The device kernel uses
 * ncclGin_SegmentHostNuma since all segments are host-backed.
 *
 * Intentionally does NOT use ncclMemAlloc(): that only ever produces
 * single-segment device memory and would never take the elastic path.
 *************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <numa.h>     // host-thread / memory affinity to the GPU's NUMA node
#include <numaif.h>
#include <unistd.h>

#define CUDA_DRV_CHECK(cmd)                                                                  \
  do {                                                                                        \
    CUresult e = (cmd);                                                                       \
    if (e != CUDA_SUCCESS) {                                                                  \
      const char* name = nullptr;                                                             \
      const char* str = nullptr;                                                              \
      cuGetErrorName(e, &name);                                                               \
      cuGetErrorString(e, &str);                                                              \
      std::fprintf(stderr, "CUDA driver error %s:%d: %s (%s)\n", __FILE__, __LINE__,          \
                   name ? name : "unknown", str ? str : "no details");                       \
      std::exit(EXIT_FAILURE);                                                                \
    }                                                                                         \
  } while (0)

#define CUDA_RT_CHECK(cmd)                                                                    \
  do {                                                                                        \
    cudaError_t e = (cmd);                                                                    \
    if (e != cudaSuccess) {                                                                   \
      std::fprintf(stderr, "CUDA runtime error %s:%d: %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(e));                                                    \
      std::exit(EXIT_FAILURE);                                                                \
    }                                                                                         \
  } while (0)

#define NCCL_CHECK(cmd)                                                                       \
  do {                                                                                        \
    ncclResult_t ncclStatus__ = (cmd);                                                        \
    if (ncclStatus__ != ncclSuccess) {                                                        \
      std::fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__,                     \
                   ncclGetErrorString(ncclStatus__));                                         \
      std::exit(EXIT_FAILURE);                                                                \
    }                                                                                         \
  } while (0)

static constexpr int kCtas = 16;
static constexpr int kThreads = 256;

// A window VA built from two HOST_NUMA segments mapped back-to-back, each on a
// (potentially) different host NUMA node.
struct HostBuffer {
  void* va = nullptr;     // base of the contiguous VA spanning both segments
  size_t total = 0;       // seg0Size + seg1Size (granularity-padded)
  size_t seg0Size = 0;    // bytes of the first segment == the segment boundary
  CUmemGenericAllocationHandle h0 = 0;
  CUmemGenericAllocationHandle h1 = 0;
  int numaNode0 = 0;      // NUMA node backing segment 0
  int numaNode1 = 0;      // NUMA node backing segment 1
};

static size_t alignUp(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

static const char* locTypeName(int locType) {
  switch (locType) {
    case CU_MEM_LOCATION_TYPE_DEVICE: return "DEVICE";
    case CU_MEM_LOCATION_TYPE_HOST_NUMA: return "HOST_NUMA";
    default: return "OTHER";
  }
}

// Bind the calling thread (and its default allocations) to the NUMA node
// closest to the given CUDA device.  This co-locates the GPU, the HOST_NUMA
// segments we allocate, and the GIN NIC that NCCL's topology picks for this
// rank.  We do NOT set NCCL_GIN_HCA/NCCL_IB_HCA -- in a single-process
// multi-GPU job that env is process-global and would pin all ranks to one NIC.
static int bindToDeviceNuma(int cudaDev) {
  CUdevice dev;
  CUDA_DRV_CHECK(cuDeviceGet(&dev, cudaDev));
  int numaNode = -1;
  CUDA_DRV_CHECK(cuDeviceGetAttribute(&numaNode, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev));
  if (numaNode < 0) numaNode = 0;

  if (numa_available() != -1) {
    numa_run_on_node(numaNode);
    numa_set_preferred(numaNode);
  }
  return numaNode;
}

// Allocate one contiguous VA backed by [HOST_NUMA(node0) | HOST_NUMA(node1)].
// The two segments may live on different host NUMA nodes -- still a valid
// elastic buffer (both are CU_MEM_LOCATION_TYPE_HOST_NUMA; only location.id
// differs, which NCCL does not constrain).  node1 != node0 means seg1 is
// remote to the GPU/NIC -- correct but slower -- a capacity, not bandwidth, win.
static HostBuffer allocHostOnly(size_t usefulBytes, int cudaDev, int node0, int node1) {
  HostBuffer out;
  out.numaNode0 = node0;
  out.numaNode1 = node1;

  // Per-segment properties, differing only in the host NUMA node id.
  CUmemAllocationProp prop0 = {};
  prop0.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop0.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop0.location.id = node0;
  prop0.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  CUmemAllocationProp prop1 = prop0;
  prop1.location.id = node1;

  // Single granularity that satisfies both (host props share granularity).
  size_t g0 = 0, g1 = 0;
  CUDA_DRV_CHECK(cuMemGetAllocationGranularity(&g0, &prop0, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  CUDA_DRV_CHECK(cuMemGetAllocationGranularity(&g1, &prop1, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  const size_t G = g0 > g1 ? g0 : g1;

  // 50/50 split, each segment rounded up to granularity.  Both ranks run this
  // identical formula, which is REQUIRED: ncclDevrVerifySegmentLayouts rejects
  // windows whose per-segment sizes differ across ranks.
  out.seg0Size = alignUp(usefulBytes / 2, G);
  size_t seg1Size = alignUp(usefulBytes - out.seg0Size, G);
  if (seg1Size == 0) seg1Size = G;  // keep a genuine second segment
  out.total = out.seg0Size + seg1Size;

  // One physical handle per segment, each on its own NUMA node.
  CUDA_DRV_CHECK(cuMemCreate(&out.h0, out.seg0Size, &prop0, 0));
  CUDA_DRV_CHECK(cuMemCreate(&out.h1, seg1Size, &prop1, 0));

  // One contiguous VA; map the two handles back-to-back into it.
  CUdeviceptr va = 0;
  CUDA_DRV_CHECK(cuMemAddressReserve(&va, out.total, G, 0, 0));
  CUDA_DRV_CHECK(cuMemMap(va, out.seg0Size, 0, out.h0, 0));
  CUDA_DRV_CHECK(cuMemMap(va + out.seg0Size, seg1Size, 0, out.h1, 0));

  // Grant access from the GPU (for GIN) and from BOTH host NUMA nodes over the
  // whole range, so the CPU can reach either segment regardless of node.  All
  // segments are host-backed, so a single whole-range cuMemSetAccess is valid
  // (the per-segment split the mixed demo needs is only for device segments).
  CUmemAccessDesc access[3] = {};
  access[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access[0].location.id = cudaDev;
  access[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  access[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  access[1].location.id = node0;
  access[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  access[2].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  access[2].location.id = node1;
  access[2].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  int nAccess = (node1 == node0) ? 2 : 3;  // drop the duplicate when nodes match
  CUDA_DRV_CHECK(cuMemSetAccess(va, out.total, access, nAccess));

  out.va = reinterpret_cast<void*>(va);
  std::printf(
    "GPU %d host-only buffer %p: seg0=HOST_NUMA(node %d) %zu bytes | seg1=HOST_NUMA(node %d) %zu bytes | "
    "total=%zu (useful=%zu)\n",
    cudaDev, out.va, node0, out.seg0Size, node1, seg1Size, out.total, usefulBytes);
  return out;
}

static void freeHostOnly(HostBuffer* b) {
  if (b == nullptr || b->va == nullptr) return;
  CUdeviceptr va = reinterpret_cast<CUdeviceptr>(b->va);
  CUDA_DRV_CHECK(cuMemUnmap(va, b->total));
  CUDA_DRV_CHECK(cuMemAddressFree(va, b->total));
  CUDA_DRV_CHECK(cuMemRelease(b->h0));
  CUDA_DRV_CHECK(cuMemRelease(b->h1));
  *b = HostBuffer{};
}

// Confirm the driver's ground truth for a VA offset: which segment and which
// host NUMA node, vs the user's design-time expectation (offset + split).
static void describeAddress(const HostBuffer& b, size_t off, const char* label) {
  int expectedSeg = off < b.seg0Size ? 0 : 1;
  int expectedNode = expectedSeg == 0 ? b.numaNode0 : b.numaNode1;

  CUmemGenericAllocationHandle h;
  CUmemAllocationProp prop;
  void* addr = reinterpret_cast<void*>(reinterpret_cast<char*>(b.va) + off);
  CUDA_DRV_CHECK(cuMemRetainAllocationHandle(&h, addr));
  CUDA_DRV_CHECK(cuMemGetAllocationPropertiesFromHandle(&prop, h));
  CUDA_DRV_CHECK(cuMemRelease(h));

  bool isHost = prop.location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA;
  bool nodeOk = isHost && (int)prop.location.id == expectedNode;
  std::printf("  %s @ off 0x%zx: expected segment %d HOST_NUMA(node %d), driver reports %s(id %d) %s\n",
              label, off, expectedSeg, expectedNode, locTypeName(prop.location.type), prop.location.id,
              nodeOk ? "OK" : "MISMATCH!");
}

__global__ void hostOnlyGinAllToAllKernel(ncclWindow_t sendWin, ncclWindow_t recvWin,
                                          size_t elemsPerPeer, ncclDevComm devComm) {
  int ginContext = 0;
  unsigned int signalIndex = blockIdx.x;
  ncclGin gin { devComm, ginContext };
  uint64_t signalBase = gin.readSignal(signalIndex);

  ncclGinBarrierSession<ncclCoopCta> bar { ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x };
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::None);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;
  size_t bytes = elemsPerPeer * sizeof(int);

  for (int peer = tid; peer < devComm.nRanks; peer += nthreads) {
    gin.put(ncclTeamWorld(devComm), peer,
            recvWin, (size_t)devComm.rank * bytes,
            sendWin, (size_t)peer * bytes,
            bytes,
            ncclGin_WeakSignalInc{signalIndex},
            ncclGin_None{},
            ncclCoopThread{},
            ncclGin_None{},
            cuda::thread_scope_thread,
            cuda::thread_scope_device,
            ncclGinOptFlagsDefault,
            // All segments are CPU-backed; GIN still walks the segment boundary
            // internally when a single put straddles seg0/seg1.
            ncclGin_SegmentHostNuma{});
  }

  int receivingCta = (devComm.rank % nthreads) / blockDim.x;
  if (blockIdx.x == receivingCta) {
    gin.waitSignal(ncclCoopCta(), signalIndex, signalBase + devComm.nRanks);
  }

  gin.flush(ncclCoopCta());
  bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::None);
}

static void usage(const char* argv0) {
  std::fprintf(stderr, "Usage: %s [num_devices] [elems_per_peer]\n", argv0);
  std::fprintf(stderr, "  num_devices defaults to min(cudaGetDeviceCount(), 2)\n");
  std::fprintf(stderr, "  elems_per_peer defaults to 1048576 ints (sized so chunks span both segments)\n");
}

int main(int argc, char** argv) {
  if (argc > 3) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  // The feature gate: without this, registering a window that contains a
  // CPU-backed (HOST_NUMA) segment is rejected with ncclInvalidArgument.
  setenv("NCCL_ELASTIC_BUFFER_REGISTER", "1", 0);

  CUDA_DRV_CHECK(cuInit(0));

  int deviceCount = 0;
  CUDA_RT_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount <= 0) {
    std::fprintf(stderr, "No CUDA devices found\n");
    return EXIT_FAILURE;
  }

  int ndev = argc >= 2 ? std::atoi(argv[1]) : (deviceCount < 2 ? deviceCount : 2);
  // Default large enough that per-peer chunks land on both sides of the
  // granularity-forced segment boundary (not all inside segment 0).
  size_t elemsPerPeer = argc >= 3 ? static_cast<size_t>(std::strtoull(argv[2], nullptr, 10)) : 1048576;
  if (ndev <= 0 || ndev > deviceCount || elemsPerPeer == 0) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  std::vector<int> devices(ndev);
  for (int i = 0; i < ndev; ++i) devices[i] = i;

  std::vector<ncclComm_t> comms(ndev);
  NCCL_CHECK(ncclCommInitAll(comms.data(), ndev, devices.data()));

  for (int r = 0; r < ndev; ++r) {
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comms[r], &props));
    if (!props.deviceApiSupport || props.ginType == NCCL_GIN_TYPE_NONE) {
      std::fprintf(stderr,
                   "Rank %d: Device API or GIN is not available (deviceApi=%d, ginType=%d). "
                   "The demo compiled, but this runtime cannot execute it.\n",
                   r, props.deviceApiSupport, props.ginType);
      for (ncclComm_t c : comms) {
        ncclCommFinalize(c);
        ncclCommDestroy(c);
      }
      return EXIT_FAILURE;
    }
  }

  // Choose the two host NUMA nodes for the segments.  ALL ranks must use the
  // SAME pair (ncclDevrVerifySegmentLayouts requires identical layouts), so
  // these are fixed system-wide, not per-GPU: node 0 and node 1 when the box
  // has >=2 nodes, else both segments fall back to node 0.
  int segNode0 = 0;
  int segNode1 = (numa_available() != -1 && numa_max_node() >= 1) ? 1 : 0;
  std::printf("Segment NUMA nodes: seg0 -> node %d, seg1 -> node %d%s\n",
              segNode0, segNode1,
              segNode1 == segNode0 ? " (single-node fallback)" : " (seg1 is cross-node)");

  size_t usefulBytes = ndev * elemsPerPeer * sizeof(int);
  std::vector<HostBuffer> sendBuffers(ndev);
  std::vector<HostBuffer> recvBuffers(ndev);
  std::vector<ncclWindow_t> sendWins(ndev);
  std::vector<ncclWindow_t> recvWins(ndev);
  std::vector<ncclDevComm> devComms(ndev);
  std::vector<cudaStream_t> streams(ndev);
  std::vector<int> staging(usefulBytes / sizeof(int));

  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    int boundNode = bindToDeviceNuma(devices[r]);
    int cpu = sched_getcpu();
    std::printf("GPU %d affinity: host-NUMA node %d, running on CPU %d "
                "(GIN NIC chosen by NCCL topology -- see 'NET/IB ... GID' INFO lines)\n",
                devices[r], boundNode, cpu);
    sendBuffers[r] = allocHostOnly(usefulBytes, devices[r], segNode0, segNode1);
    recvBuffers[r] = allocHostOnly(usefulBytes, devices[r], segNode0, segNode1);

    // Confirm each per-peer send chunk is host-backed (driver ground truth).
    if (r == 0) {
      std::printf("GPU %d send-buffer per-peer chunk placement (boundary seg0Size=%zu):\n",
                  devices[r], sendBuffers[r].seg0Size);
      size_t bytes = elemsPerPeer * sizeof(int);
      for (int peer = 0; peer < ndev; ++peer) {
        char label[32];
        std::snprintf(label, sizeof(label), "peer %d", peer);
        describeAddress(sendBuffers[r], (size_t)peer * bytes, label);
      }
    }

    // The whole VA is CPU-addressable, so init/verify are plain memcpy -- no
    // device/host split is needed (contrast the mixed demo's copyMixed).
    size_t nElems = usefulBytes / sizeof(int);
    for (size_t i = 0; i < nElems; ++i) {
      int peer = static_cast<int>(i / elemsPerPeer);
      int idx = static_cast<int>(i % elemsPerPeer);
      staging[i] = r * 1000000 + peer * 10000 + idx;
    }
    std::memcpy(sendBuffers[r].va, staging.data(), usefulBytes);
    std::memset(recvBuffers[r].va, 0xff, usefulBytes);  // recv := -1 pattern
  }

  NCCL_CHECK(ncclGroupStart());
  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    NCCL_CHECK(ncclCommWindowRegister(comms[r], sendBuffers[r].va, usefulBytes, &sendWins[r],
                                      NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comms[r], recvBuffers[r].va, usefulBytes, &recvWins[r],
                                      NCCL_WIN_COLL_SYMMETRIC));
  }
  NCCL_CHECK(ncclGroupEnd());

  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    CUDA_RT_CHECK(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
  }
  // ncclDevCommCreate with NCCL_GIN_CONNECTION_FULL establishes cross-rank GIN
  // connections -- a rendezvous.  In single-process multi-GPU it must be issued
  // for all ranks inside one group, or rank 0 blocks waiting for rank 1.
  NCCL_CHECK(ncclGroupStart());
  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.worldGinBarrierCount = kCtas;
    reqs.ginSignalCount = kCtas;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    NCCL_CHECK(ncclDevCommCreate(comms[r], &reqs, &devComms[r]));
  }
  NCCL_CHECK(ncclGroupEnd());

  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    hostOnlyGinAllToAllKernel<<<kCtas, kThreads, 0, streams[r]>>>(
      sendWins[r], recvWins[r], elemsPerPeer, devComms[r]);
    CUDA_RT_CHECK(cudaGetLastError());
  }

  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    CUDA_RT_CHECK(cudaStreamSynchronize(streams[r]));
  }

  bool ok = true;
  for (int dst = 0; dst < ndev && ok; ++dst) {
    const int* recv = static_cast<const int*>(recvBuffers[dst].va);  // host-addressable
    for (int src = 0; src < ndev && ok; ++src) {
      for (size_t i = 0; i < elemsPerPeer; ++i) {
        int got = recv[src * elemsPerPeer + i];
        int expected = src * 1000000 + dst * 10000 + static_cast<int>(i);
        if (got != expected) {
          std::fprintf(stderr, "Mismatch dst=%d src=%d elem=%zu got=%d expected=%d\n",
                       dst, src, i, got, expected);
          ok = false;
          break;
        }
      }
    }
  }

  std::printf("host-only GIN all-to-all: %s\n", ok ? "PASSED" : "FAILED");

  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    NCCL_CHECK(ncclDevCommDestroy(comms[r], &devComms[r]));
    CUDA_RT_CHECK(cudaStreamDestroy(streams[r]));
  }

  NCCL_CHECK(ncclGroupStart());
  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    NCCL_CHECK(ncclCommWindowDeregister(comms[r], sendWins[r]));
    NCCL_CHECK(ncclCommWindowDeregister(comms[r], recvWins[r]));
  }
  NCCL_CHECK(ncclGroupEnd());

  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    freeHostOnly(&sendBuffers[r]);
    freeHostOnly(&recvBuffers[r]);
    NCCL_CHECK(ncclCommFinalize(comms[r]));
    NCCL_CHECK(ncclCommDestroy(comms[r]));
  }

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
