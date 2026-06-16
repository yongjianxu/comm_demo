NCCL_BUILD ?= /root/nccl-2.30.7-1/build
NVCC ?= nvcc

TARGET := mixed_segment_gin_demo

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

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(TARGET).cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_GENCODE) $(INCLUDES) $< -o $@ $(LIBS)

run: $(TARGET)
	NCCL_DEBUG=INFO NCCL_ELASTIC_BUFFER_REGISTER=1 ./$(TARGET)

clean:
	rm -f $(TARGET)
