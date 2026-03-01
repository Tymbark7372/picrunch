NVCC      := nvcc
CXX       := g++
TARGET    := pi

# CUDA source
CUDA_SRC  := pi.cu
CUDA_OBJ  := pi.o

# C++ sources (OpenCL + CPU backends)
CL_SRC    := pi_opencl.cpp
CL_OBJ    := pi_opencl.o
CPU_SRC   := pi_cpu.cpp
CPU_OBJ   := pi_cpu.o

# CUDA architectures — build for all supported SMs.
# For faster builds, override:  make ARCH_FLAGS="-gencode arch=compute_89,code=sm_89"
ARCH_FLAGS := \
	-gencode arch=compute_50,code=sm_50 \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_90,code=sm_90

# Flags
NVCCFLAGS := -O3 $(ARCH_FLAGS) --std c++17 -DHAS_OPENCL -DHAS_OPENMP
CXXFLAGS  := -O3 -std=c++17 -fopenmp -DHAS_OPENCL -DHAS_OPENMP
LDFLAGS   := -lnvidia-ml -lOpenCL -lgomp -lpthread -lgmp

.PHONY: all clean

all: $(TARGET)

$(CUDA_OBJ): $(CUDA_SRC) backends.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(CL_OBJ): $(CL_SRC) backends.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(CPU_OBJ): $(CPU_SRC) backends.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGET): $(CUDA_OBJ) $(CL_OBJ) $(CPU_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) $(CUDA_OBJ) $(CL_OBJ) $(CPU_OBJ)
