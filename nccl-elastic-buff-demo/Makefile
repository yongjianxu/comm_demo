NCCL_BUILD ?= /root/nccl-2.30.7-1/build
NVCC ?= nvcc

TARGETS := mixed_segment_gin_demo host_only_gin_demo lsa_segment_demo

# Include Hopper (sm_90, e.g. H20/H100) and Blackwell (sm_100, e.g. B200).
# Override if needed, for example:
#   make NVCC_GENCODE="-gencode arch=compute_90,code=sm_90"
NVCC_GENCODE ?= -gencode arch=compute_90,code=sm_90 \
                -gencode arch=compute_100,code=sm_100 \
                -gencode arch=compute_100,code=compute_100

NVCCFLAGS ?= -O2 -std=c++17
INCLUDES := -I$(NCCL_BUILD)/include
LIBS := -L$(NCCL_BUILD)/lib -Xlinker -rpath -Xlinker $(NCCL_BUILD)/lib \
        -lnccl -lcuda -lcudart -lpthread -lnuma

.PHONY: all run run-host run-lsa run-lsa-mixed clean

all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_GENCODE) $(INCLUDES) $< -o $@ $(LIBS)

run: mixed_segment_gin_demo
	NCCL_DEBUG=INFO NCCL_ELASTIC_BUFFER_REGISTER=1 ./mixed_segment_gin_demo

run-host: host_only_gin_demo
	NCCL_DEBUG=INFO NCCL_ELASTIC_BUFFER_REGISTER=1 ./host_only_gin_demo

run-lsa: lsa_segment_demo
	NCCL_DEBUG=INFO NCCL_ELASTIC_BUFFER_REGISTER=1 ./lsa_segment_demo

run-lsa-mixed: lsa_segment_demo
	NCCL_DEBUG=INFO NCCL_ELASTIC_BUFFER_REGISTER=1 ./lsa_segment_demo --mixed

clean:
	rm -f $(TARGETS)
