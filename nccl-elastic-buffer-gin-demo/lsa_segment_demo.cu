/*************************************************************************
 * Elastic-buffer LSA demo for NCCL 2.30.7.
 *
 * Unlike the GIN demos (which move data with gin.put over the NIC), this one
 * uses the LSA (Load/Store Accessible) path: each GPU directly load/stores into
 * a PEER GPU's symmetric-window memory over NVLink / C2C, via
 * ncclGetLsaPointer(window, offset, peer).  It demonstrates that an elastic
 * buffer whose segments include HOST_NUMA memory is reachable by peer GPUs
 * through the LSA window -- i.e. host pages ARE accessible over NVLink.
 *
 * Window layout is selectable:
 *   (default)  host-only : seg0 = HOST_NUMA, seg1 = HOST_NUMA
 *   --mixed              : seg0 = DEVICE,    seg1 = HOST_NUMA
 * Both are multi-segment (numSegments == 2) and gated by
 * NCCL_ELASTIC_BUFFER_REGISTER.
 *
 * Algorithm: an all-gather done with LSA writes.  Each rank writes its own
 * contribution into every peer's recv buffer at slot[rank], using a peer LSA
 * pointer.  An LSA barrier orders the writes.  Then every rank's recv buffer
 * holds all ranks' contributions, verified on the host.
 *
 * NOTE: LSA requires the peers to be in the same NVLink/P2P (LSA) domain.  On a
 * single NVLink-connected node all ranks share one LSA team (ncclTeamTagLsa).
 *
 * Intentionally does NOT use ncclMemAlloc(): single-segment device memory only,
 * never the elastic path.
 *************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <numa.h>
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

// A symmetric-window VA built from two segments.  seg0 is DEVICE (mixed mode) or
// HOST_NUMA (host-only mode); seg1 is always HOST_NUMA.
struct SegBuffer {
  void* va = nullptr;
  size_t total = 0;
  size_t seg0Size = 0;       // boundary between segment 0 and segment 1
  CUmemGenericAllocationHandle h0 = 0;
  CUmemGenericAllocationHandle h1 = 0;
  bool seg0IsDevice = false; // false => seg0 is HOST_NUMA (host-only mode)
  int numaNode = 0;
};

static size_t alignUp(size_t v, size_t a) { return ((v + a - 1) / a) * a; }

static const char* locTypeName(int t) {
  switch (t) {
    case CU_MEM_LOCATION_TYPE_DEVICE: return "DEVICE";
    case CU_MEM_LOCATION_TYPE_HOST_NUMA: return "HOST_NUMA";
    default: return "OTHER";
  }
}

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

// Allocate a 2-segment symmetric window.  seg0 = DEVICE if mixed, else HOST_NUMA;
// seg1 = HOST_NUMA always.
static SegBuffer allocSeg(size_t usefulBytes, int cudaDev, bool mixed) {
  SegBuffer out;
  out.seg0IsDevice = mixed;

  CUdevice dev;
  CUDA_DRV_CHECK(cuDeviceGet(&dev, cudaDev));
  int numaNode = -1;
  CUDA_DRV_CHECK(cuDeviceGetAttribute(&numaNode, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev));
  if (numaNode < 0) numaNode = 0;
  out.numaNode = numaNode;

  CUmemAllocationProp deviceProp = {};
  deviceProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  deviceProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  deviceProp.location.id = cudaDev;
  deviceProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  CUmemAllocationProp hostProp = {};
  hostProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  hostProp.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  hostProp.location.id = numaNode;
  hostProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  CUmemAllocationProp& prop0 = mixed ? deviceProp : hostProp;

  size_t gDev = 0, gHost = 0;
  CUDA_DRV_CHECK(cuMemGetAllocationGranularity(&gDev, &deviceProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  CUDA_DRV_CHECK(cuMemGetAllocationGranularity(&gHost, &hostProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  const size_t G = gDev > gHost ? gDev : gHost;

  out.seg0Size = alignUp(usefulBytes / 2, G);
  size_t seg1Size = alignUp(usefulBytes - out.seg0Size, G);
  if (seg1Size == 0) seg1Size = G;
  out.total = out.seg0Size + seg1Size;

  CUDA_DRV_CHECK(cuMemCreate(&out.h0, out.seg0Size, &prop0, 0));
  CUDA_DRV_CHECK(cuMemCreate(&out.h1, seg1Size, &hostProp, 0));

  CUdeviceptr va = 0;
  CUDA_DRV_CHECK(cuMemAddressReserve(&va, out.total, G, 0, 0));
  CUDA_DRV_CHECK(cuMemMap(va, out.seg0Size, 0, out.h0, 0));
  CUDA_DRV_CHECK(cuMemMap(va + out.seg0Size, seg1Size, 0, out.h1, 0));

  // Access: the GPU needs DEVICE access on every segment (for LSA load/store);
  // host segments additionally get a HOST_NUMA accessor for the CPU.  A device
  // segment cannot take a HOST_NUMA accessor, so set access per segment.
  CUmemAccessDesc devAccess = {};
  devAccess.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  devAccess.location.id = cudaDev;
  devAccess.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  CUmemAccessDesc hostAccess[2] = {};
  hostAccess[0] = devAccess;
  hostAccess[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  hostAccess[1].location.id = numaNode;
  hostAccess[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  if (mixed) {
    CUDA_DRV_CHECK(cuMemSetAccess(va, out.seg0Size, &devAccess, 1));            // device seg0
    CUDA_DRV_CHECK(cuMemSetAccess(va + out.seg0Size, seg1Size, hostAccess, 2)); // host seg1
  } else {
    CUDA_DRV_CHECK(cuMemSetAccess(va, out.total, hostAccess, 2));               // both host
  }

  out.va = reinterpret_cast<void*>(va);
  std::printf("GPU %d %s buffer %p: seg0=%s %zu bytes | seg1=HOST_NUMA(node %d) %zu bytes | total=%zu\n",
              cudaDev, mixed ? "mixed" : "host-only", out.va, mixed ? "DEVICE" : "HOST_NUMA", out.seg0Size,
              numaNode, seg1Size, out.total);
  return out;
}

static void freeSeg(SegBuffer* b) {
  if (b == nullptr || b->va == nullptr) return;
  CUdeviceptr va = reinterpret_cast<CUdeviceptr>(b->va);
  CUDA_DRV_CHECK(cuMemUnmap(va, b->total));
  CUDA_DRV_CHECK(cuMemAddressFree(va, b->total));
  CUDA_DRV_CHECK(cuMemRelease(b->h0));
  CUDA_DRV_CHECK(cuMemRelease(b->h1));
  *b = SegBuffer{};
}

static void describeAddress(const SegBuffer& b, size_t off, const char* label) {
  int expectedSeg = off < b.seg0Size ? 0 : 1;
  bool expectDevice = (expectedSeg == 0) && b.seg0IsDevice;

  CUmemGenericAllocationHandle h;
  CUmemAllocationProp prop;
  void* addr = reinterpret_cast<void*>(reinterpret_cast<char*>(b.va) + off);
  CUDA_DRV_CHECK(cuMemRetainAllocationHandle(&h, addr));
  CUDA_DRV_CHECK(cuMemGetAllocationPropertiesFromHandle(&prop, h));
  CUDA_DRV_CHECK(cuMemRelease(h));

  bool driverDevice = prop.location.type == CU_MEM_LOCATION_TYPE_DEVICE;
  bool ok = expectDevice == driverDevice;
  std::printf("  %s @ off 0x%zx (seg %d): expected %s, driver reports %s(id %d) %s\n",
              label, off, expectedSeg, expectDevice ? "DEVICE" : "HOST_NUMA",
              locTypeName(prop.location.type), prop.location.id, ok ? "OK" : "MISMATCH!");
}

// Whether the seg0/seg1 boundary is CPU-addressable straight through (host-only).
static bool wholeBufferIsHost(const SegBuffer& b) { return !b.seg0IsDevice; }

// Copy host<->window respecting the seg boundary (device seg0 needs cudaMemcpy).
enum CopyDir { HOST_TO_BUF, BUF_TO_HOST };
static void copySeg(const SegBuffer& b, int* host, size_t bytes, CopyDir dir) {
  if (wholeBufferIsHost(b)) {
    if (dir == HOST_TO_BUF) std::memcpy(b.va, host, bytes);
    else                    std::memcpy(host, b.va, bytes);
    return;
  }
  char* buf = reinterpret_cast<char*>(b.va);
  char* h = reinterpret_cast<char*>(host);
  size_t devBytes = bytes < b.seg0Size ? bytes : b.seg0Size;
  size_t hostPart = bytes - devBytes;
  if (dir == HOST_TO_BUF) {
    CUDA_RT_CHECK(cudaMemcpy(buf, h, devBytes, cudaMemcpyHostToDevice));
    if (hostPart) std::memcpy(buf + devBytes, h + devBytes, hostPart);
  } else {
    CUDA_RT_CHECK(cudaMemcpy(h, buf, devBytes, cudaMemcpyDeviceToHost));
    if (hostPart) std::memcpy(h + devBytes, buf + devBytes, hostPart);
  }
}

// LSA all-gather: each rank writes its slot into every peer's recv window via a
// direct peer pointer (NVLink/C2C load-store), then an LSA barrier orders it.
__global__ void lsaAllGatherKernel(ncclWindow_t recvWin, ncclWindow_t sendWin,
                                   size_t elemsPerRank, ncclDevComm devComm) {
  ncclCoopCta coop = ncclCoopCta();
  ncclLsaBarrierSession<ncclCoopCta> bar { coop, devComm, ncclTeamTagLsa(), blockIdx.x };
  bar.sync(coop, cuda::memory_order_acquire);

  const int rank = devComm.rank;
  const int nRanks = devComm.nRanks;
  const size_t bytesPerRank = elemsPerRank * sizeof(int);

  // My send slot lives at offset 0 in my send window; I write it into peer's
  // recv window at slot[rank].  Split the work across blocks and threads.
  int* myLocalSend = (int*)ncclGetLocalPointer(sendWin, 0);

  for (int peer = 0; peer < nRanks; peer++) {
    // Peer's recv window, my slot: direct LSA pointer into peer memory.
    int* peerRecvSlot = (int*)ncclGetLsaPointer(recvWin, (size_t)rank * bytesPerRank, peer);
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elemsPerRank;
         i += (size_t)gridDim.x * blockDim.x) {
      peerRecvSlot[i] = myLocalSend[i];   // store across NVLink / C2C
    }
  }

  // Ensure all peers' writes into my recv buffer are visible before host reads.
  bar.sync(coop, cuda::memory_order_release);
}

static void usage(const char* a0) {
  std::fprintf(stderr, "Usage: %s [--mixed] [num_devices] [elems_per_rank]\n", a0);
  std::fprintf(stderr, "  --mixed         seg0=DEVICE, seg1=HOST_NUMA (default: both HOST_NUMA)\n");
  std::fprintf(stderr, "  num_devices     defaults to min(cudaGetDeviceCount(), 2)\n");
  std::fprintf(stderr, "  elems_per_rank  defaults to 1048576 ints\n");
}

int main(int argc, char** argv) {
  bool mixed = false;
  std::vector<char*> pos;
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--mixed") == 0) mixed = true;
    else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) { usage(argv[0]); return 0; }
    else pos.push_back(argv[i]);
  }

  setenv("NCCL_ELASTIC_BUFFER_REGISTER", "1", 0);
  CUDA_DRV_CHECK(cuInit(0));

  int deviceCount = 0;
  CUDA_RT_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount <= 0) { std::fprintf(stderr, "No CUDA devices found\n"); return EXIT_FAILURE; }

  int ndev = pos.size() >= 1 ? std::atoi(pos[0]) : (deviceCount < 2 ? deviceCount : 2);
  size_t elemsPerRank = pos.size() >= 2 ? std::strtoull(pos[1], nullptr, 10) : 1048576;
  if (ndev <= 0 || ndev > deviceCount || elemsPerRank == 0) { usage(argv[0]); return EXIT_FAILURE; }

  std::printf("LSA %s demo: %d GPUs, %zu elems/rank\n", mixed ? "mixed-segment" : "host-only", ndev, elemsPerRank);

  std::vector<int> devices(ndev);
  for (int i = 0; i < ndev; ++i) devices[i] = i;

  std::vector<ncclComm_t> comms(ndev);
  NCCL_CHECK(ncclCommInitAll(comms.data(), ndev, devices.data()));

  for (int r = 0; r < ndev; ++r) {
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comms[r], &props));
    if (!props.deviceApiSupport) {
      std::fprintf(stderr, "Rank %d: Device API not available (deviceApi=%d). Cannot run.\n", r,
                   props.deviceApiSupport);
      for (ncclComm_t c : comms) { ncclCommFinalize(c); ncclCommDestroy(c); }
      return EXIT_FAILURE;
    }
  }

  // recv buffer holds nRanks slots (the gathered result); send buffer holds one.
  size_t recvUseful = (size_t)ndev * elemsPerRank * sizeof(int);
  size_t sendUseful = elemsPerRank * sizeof(int);

  std::vector<SegBuffer> recvBuf(ndev), sendBuf(ndev);
  std::vector<ncclWindow_t> recvWin(ndev), sendWin(ndev);
  std::vector<ncclDevComm> devComms(ndev);
  std::vector<cudaStream_t> streams(ndev);
  std::vector<int> staging(recvUseful / sizeof(int));

  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    int node = bindToDeviceNuma(devices[r]);
    std::printf("GPU %d affinity: host-NUMA node %d, CPU %d\n", devices[r], node, sched_getcpu());
    recvBuf[r] = allocSeg(recvUseful, devices[r], mixed);
    sendBuf[r] = allocSeg(sendUseful, devices[r], mixed);

    if (r == 0) {
      std::printf("GPU %d recv-buffer slot placement (boundary seg0Size=%zu):\n", devices[r], recvBuf[r].seg0Size);
      for (int slot = 0; slot < ndev; ++slot) {
        char label[32];
        std::snprintf(label, sizeof(label), "slot %d", slot);
        describeAddress(recvBuf[r], (size_t)slot * elemsPerRank * sizeof(int), label);
      }
    }

    // send[i] = rank-encoded payload; recv := -1.
    size_t sendElems = sendUseful / sizeof(int);
    for (size_t i = 0; i < sendElems; ++i) staging[i] = r * 1000000 + (int)i;
    copySeg(sendBuf[r], staging.data(), sendUseful, HOST_TO_BUF);
    for (size_t i = 0; i < recvUseful / sizeof(int); ++i) staging[i] = -1;
    copySeg(recvBuf[r], staging.data(), recvUseful, HOST_TO_BUF);
  }

  NCCL_CHECK(ncclGroupStart());
  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    NCCL_CHECK(ncclCommWindowRegister(comms[r], recvBuf[r].va, recvUseful, &recvWin[r], NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comms[r], sendBuf[r].va, sendUseful, &sendWin[r], NCCL_WIN_COLL_SYMMETRIC));
  }
  NCCL_CHECK(ncclGroupEnd());

  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    CUDA_RT_CHECK(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
  }
  // LSA uses peer (NVLink/C2C) connections; create all devcomms in one group.
  NCCL_CHECK(ncclGroupStart());
  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = kCtas;   // one LSA barrier per block
    NCCL_CHECK(ncclDevCommCreate(comms[r], &reqs, &devComms[r]));
  }
  NCCL_CHECK(ncclGroupEnd());

  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    lsaAllGatherKernel<<<kCtas, kThreads, 0, streams[r]>>>(recvWin[r], sendWin[r], elemsPerRank, devComms[r]);
    CUDA_RT_CHECK(cudaGetLastError());
  }
  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    CUDA_RT_CHECK(cudaStreamSynchronize(streams[r]));
  }

  bool ok = true;
  for (int dst = 0; dst < ndev && ok; ++dst) {
    CUDA_RT_CHECK(cudaSetDevice(devices[dst]));
    copySeg(recvBuf[dst], staging.data(), recvUseful, BUF_TO_HOST);
    for (int src = 0; src < ndev && ok; ++src) {
      for (size_t i = 0; i < elemsPerRank; ++i) {
        int got = staging[src * elemsPerRank + i];
        int expected = src * 1000000 + (int)i;
        if (got != expected) {
          std::fprintf(stderr, "Mismatch dst=%d src=%d elem=%zu got=%d expected=%d\n", dst, src, i, got, expected);
          ok = false; break;
        }
      }
    }
  }

  std::printf("LSA %s all-gather: %s\n", mixed ? "mixed-segment" : "host-only", ok ? "PASSED" : "FAILED");

  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    NCCL_CHECK(ncclDevCommDestroy(comms[r], &devComms[r]));
    CUDA_RT_CHECK(cudaStreamDestroy(streams[r]));
  }
  NCCL_CHECK(ncclGroupStart());
  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    NCCL_CHECK(ncclCommWindowDeregister(comms[r], recvWin[r]));
    NCCL_CHECK(ncclCommWindowDeregister(comms[r], sendWin[r]));
  }
  NCCL_CHECK(ncclGroupEnd());
  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    freeSeg(&recvBuf[r]);
    freeSeg(&sendBuf[r]);
    NCCL_CHECK(ncclCommFinalize(comms[r]));
    NCCL_CHECK(ncclCommDestroy(comms[r]));
  }
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
