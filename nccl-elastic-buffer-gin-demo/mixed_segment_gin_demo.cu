/*************************************************************************
 * Mixed-segment elastic GIN demo for NCCL 2.30.7.
 *
 * Each NCCL symmetric window is backed by a SINGLE contiguous virtual
 * address range that is stitched from TWO physical CUDA VMM segments:
 *
 *     VA:  [ va ............ va+seg0Size ............ va+total )
 *           '---- segment 0 ----''------- segment 1 -------'
 *                  DEVICE                  HOST_NUMA
 *
 * This drives NCCL's multi-segment registration path (numSegments == 2),
 * exercises the NCCL_ELASTIC_BUFFER_REGISTER feature gate, and runs a
 * GPU-initiated (GIN) all-to-all whose transfers cross the device<->host
 * segment boundary.  The device kernel uses ncclGin_SegmentMixed so GIN
 * walks the segment boundaries internally.
 *
 * Placement is user-controlled (the split offset seg0Size) and confirmed
 * at runtime via the CUDA driver (cuMemGetAllocationPropertiesFromHandle).
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

// A window VA built from one DEVICE segment followed by one HOST_NUMA segment.
struct MixedBuffer {
  void* va = nullptr;     // base of the contiguous VA spanning both segments
  size_t total = 0;       // seg0Size + seg1Size (granularity-padded)
  size_t seg0Size = 0;    // bytes of the leading DEVICE segment == device/host boundary
  CUmemGenericAllocationHandle h0 = 0;  // DEVICE handle
  CUmemGenericAllocationHandle h1 = 0;  // HOST_NUMA handle
  int numaNode = 0;
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

// Bind the calling thread (and its default memory allocations) to the NUMA
// node closest to the given CUDA device.  This keeps three things co-located on
// one socket: the GPU, the HOST_NUMA segment we allocate for it, and the GIN
// NIC that NCCL's topology picks for this rank (verified via NCCL INFO).  GIN
// over IB-GDAKI does GPUDirect-async RDMA; a NIC on the wrong NUMA node forces
// every host-segment access across the inter-socket link.
//
// Note: we do NOT set NCCL_GIN_HCA/NCCL_IB_HCA -- in a single-process
// multi-GPU job that env is process-global and would pin BOTH ranks to one
// NIC.  NCCL already selects the NUMA-local NIC per comm via
// ncclTopoGetLocalGinDevs(); binding the host side is what we add.
static int bindToDeviceNuma(int cudaDev) {
  CUdevice dev;
  CUDA_DRV_CHECK(cuDeviceGet(&dev, cudaDev));
  int numaNode = -1;
  CUDA_DRV_CHECK(cuDeviceGetAttribute(&numaNode, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev));
  if (numaNode < 0) numaNode = 0;

  if (numa_available() != -1) {
    numa_run_on_node(numaNode);             // schedule this thread on that node's CPUs
    numa_set_preferred(numaNode);           // prefer that node for subsequent allocations
  }
  return numaNode;
}

// Allocate one contiguous VA backed by [DEVICE segment | HOST_NUMA segment].
static MixedBuffer allocMixed(size_t usefulBytes, int cudaDev) {
  MixedBuffer out;

  CUdevice dev;
  CUDA_DRV_CHECK(cuDeviceGet(&dev, cudaDev));

  int numaNode = -1;
  CUDA_DRV_CHECK(cuDeviceGetAttribute(&numaNode, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev));
  if (numaNode < 0) numaNode = 0;
  out.numaNode = numaNode;

  // Allocation properties for each segment.
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

  // Use a single granularity that satisfies both segment types.
  size_t gDev = 0, gHost = 0;
  CUDA_DRV_CHECK(cuMemGetAllocationGranularity(&gDev, &deviceProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  CUDA_DRV_CHECK(cuMemGetAllocationGranularity(&gHost, &hostProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  const size_t G = gDev > gHost ? gDev : gHost;

  // 50/50 split, each segment rounded up to the common granularity.  Both ranks
  // run this identical formula, which is REQUIRED: ncclDevrVerifySegmentLayouts
  // rejects windows whose per-segment sizes differ across ranks.
  out.seg0Size = alignUp(usefulBytes / 2, G);
  size_t seg1Size = alignUp(usefulBytes - out.seg0Size, G);
  if (seg1Size == 0) seg1Size = G;  // keep a genuine second (host) segment
  out.total = out.seg0Size + seg1Size;

  // Two physical handles: segment 0 on the GPU, segment 1 on the host NUMA node.
  CUDA_DRV_CHECK(cuMemCreate(&out.h0, out.seg0Size, &deviceProp, 0));
  CUDA_DRV_CHECK(cuMemCreate(&out.h1, seg1Size, &hostProp, 0));

  // One contiguous VA; map the two handles back-to-back into it.
  CUdeviceptr va = 0;
  CUDA_DRV_CHECK(cuMemAddressReserve(&va, out.total, G, 0, 0));
  CUDA_DRV_CHECK(cuMemMap(va, out.seg0Size, 0, out.h0, 0));
  CUDA_DRV_CHECK(cuMemMap(va + out.seg0Size, seg1Size, 0, out.h1, 0));

  // Access must be set per segment: a HOST_NUMA accessor cannot be granted on a
  // device-backed segment (cuMemSetAccess returns NOT_SUPPORTED).  The GPU needs
  // DEVICE access on both segments; the host segment additionally gets a
  // HOST_NUMA accessor so the CPU side can reach it.
  CUmemAccessDesc devAccess = {};
  devAccess.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  devAccess.location.id = cudaDev;
  devAccess.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  CUmemAccessDesc hostAccess[2] = {};
  hostAccess[0] = devAccess;  // GPU can also reach the host segment
  hostAccess[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  hostAccess[1].location.id = numaNode;
  hostAccess[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  CUDA_DRV_CHECK(cuMemSetAccess(va, out.seg0Size, &devAccess, 1));
  CUDA_DRV_CHECK(cuMemSetAccess(va + out.seg0Size, seg1Size, hostAccess, 2));

  out.va = reinterpret_cast<void*>(va);
  std::printf(
    "GPU %d mixed buffer %p: seg0=DEVICE %zu bytes | seg1=HOST_NUMA(node %d) %zu bytes | total=%zu (useful=%zu)\n",
    cudaDev, out.va, out.seg0Size, numaNode, seg1Size, out.total, usefulBytes);
  return out;
}

static void freeMixed(MixedBuffer* b) {
  if (b == nullptr || b->va == nullptr) return;
  CUdeviceptr va = reinterpret_cast<CUdeviceptr>(b->va);
  CUDA_DRV_CHECK(cuMemUnmap(va, b->total));
  CUDA_DRV_CHECK(cuMemAddressFree(va, b->total));
  CUDA_DRV_CHECK(cuMemRelease(b->h0));
  CUDA_DRV_CHECK(cuMemRelease(b->h1));
  *b = MixedBuffer{};
}

// Print the segment a VA offset lives in, comparing the user's design-time
// expectation (offset vs seg0Size) against the driver's ground truth.
static void describeAddress(const MixedBuffer& b, size_t off, const char* label) {
  bool expectDevice = off < b.seg0Size;

  CUmemGenericAllocationHandle h;
  CUmemAllocationProp prop;
  void* addr = reinterpret_cast<void*>(reinterpret_cast<char*>(b.va) + off);
  CUDA_DRV_CHECK(cuMemRetainAllocationHandle(&h, addr));
  CUDA_DRV_CHECK(cuMemGetAllocationPropertiesFromHandle(&prop, h));
  CUDA_DRV_CHECK(cuMemRelease(h));

  bool driverDevice = prop.location.type == CU_MEM_LOCATION_TYPE_DEVICE;
  bool agree = expectDevice == driverDevice;
  std::printf("  %s @ off 0x%zx: expected %s, driver reports %s(id %d) %s\n",
              label, off, expectDevice ? "DEVICE" : "HOST_NUMA",
              locTypeName(prop.location.type), prop.location.id, agree ? "OK" : "MISMATCH!");
}

// Copy between a plain host array and the mixed VA, respecting the segment
// boundary.  A single cudaMemcpy cannot span the device+host segments (the
// driver picks one direction from the base pointer's type), so split at
// seg0Size: cudaMemcpy for the device half, plain memcpy for the CPU-mapped
// host half.
enum CopyDir { HOST_TO_BUF, BUF_TO_HOST };
static void copyMixed(const MixedBuffer& b, int* host, size_t bytes, CopyDir dir) {
  char* devPart = reinterpret_cast<char*>(b.va);
  char* hostBytes = reinterpret_cast<char*>(host);
  size_t devBytes = bytes < b.seg0Size ? bytes : b.seg0Size;
  size_t hostOff = devBytes;
  size_t hostPartBytes = bytes - devBytes;  // resides in the HOST_NUMA segment

  if (dir == HOST_TO_BUF) {
    CUDA_RT_CHECK(cudaMemcpy(devPart, hostBytes, devBytes, cudaMemcpyHostToDevice));
    if (hostPartBytes) std::memcpy(devPart + hostOff, hostBytes + hostOff, hostPartBytes);
  } else {
    CUDA_RT_CHECK(cudaMemcpy(hostBytes, devPart, devBytes, cudaMemcpyDeviceToHost));
    if (hostPartBytes) std::memcpy(hostBytes + hostOff, devPart + hostOff, hostPartBytes);
  }
}

__global__ void mixedGinAllToAllKernel(ncclWindow_t sendWin, ncclWindow_t recvWin,
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
            // VA mixes DEVICE and HOST_NUMA segments; GIN walks segment
            // boundaries internally even when a single put straddles them.
            ncclGin_SegmentMixed{});
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
  // granularity-forced device/host boundary (not all inside segment 0).
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

  size_t usefulBytes = ndev * elemsPerPeer * sizeof(int);
  std::vector<MixedBuffer> sendBuffers(ndev);
  std::vector<MixedBuffer> recvBuffers(ndev);
  std::vector<ncclWindow_t> sendWins(ndev);
  std::vector<ncclWindow_t> recvWins(ndev);
  std::vector<ncclDevComm> devComms(ndev);
  std::vector<cudaStream_t> streams(ndev);
  std::vector<int> staging(usefulBytes / sizeof(int));

  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    // Co-locate host work + HOST_NUMA segment with this GPU's NUMA node.
    int boundNode = bindToDeviceNuma(devices[r]);
    int cpu = sched_getcpu();
    std::printf("GPU %d affinity: host-NUMA node %d, running on CPU %d "
                "(GIN NIC chosen by NCCL topology -- see 'NET/IB ... GID' INFO lines)\n",
                devices[r], boundNode, cpu);
    sendBuffers[r] = allocMixed(usefulBytes, devices[r]);
    recvBuffers[r] = allocMixed(usefulBytes, devices[r]);

    // Show where each per-peer send chunk physically lives (user expectation
    // vs driver ground truth) -- this is "how do I know where it lives".
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

    // The DEVICE half is not CPU-addressable, so initialize via cudaMemcpy
    // (unified addressing routes each page to its real segment).
    size_t nElems = usefulBytes / sizeof(int);
    for (size_t i = 0; i < nElems; ++i) {
      int peer = static_cast<int>(i / elemsPerPeer);
      int idx = static_cast<int>(i % elemsPerPeer);
      staging[i] = r * 1000000 + peer * 10000 + idx;
    }
    copyMixed(sendBuffers[r], staging.data(), usefulBytes, HOST_TO_BUF);
    for (size_t i = 0; i < nElems; ++i) staging[i] = -1;
    copyMixed(recvBuffers[r], staging.data(), usefulBytes, HOST_TO_BUF);
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
    mixedGinAllToAllKernel<<<kCtas, kThreads, 0, streams[r]>>>(
      sendWins[r], recvWins[r], elemsPerPeer, devComms[r]);
    CUDA_RT_CHECK(cudaGetLastError());
  }

  for (int r = 0; r < ndev; ++r) {
    CUDA_RT_CHECK(cudaSetDevice(devices[r]));
    CUDA_RT_CHECK(cudaStreamSynchronize(streams[r]));
  }

  bool ok = true;
  for (int dst = 0; dst < ndev && ok; ++dst) {
    CUDA_RT_CHECK(cudaSetDevice(devices[dst]));
    copyMixed(recvBuffers[dst], staging.data(), usefulBytes, BUF_TO_HOST);
    for (int src = 0; src < ndev && ok; ++src) {
      for (size_t i = 0; i < elemsPerPeer; ++i) {
        int got = staging[src * elemsPerPeer + i];
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

  std::printf("mixed-segment GIN all-to-all: %s\n", ok ? "PASSED" : "FAILED");

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
    freeMixed(&sendBuffers[r]);
    freeMixed(&recvBuffers[r]);
    NCCL_CHECK(ncclCommFinalize(comms[r]));
    NCCL_CHECK(ncclCommDestroy(comms[r]));
  }

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
