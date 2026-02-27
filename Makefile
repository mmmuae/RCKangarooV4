CXX ?= g++
CUDA_PATH ?=
CUDA_HOME ?=
NVCC ?=

ifeq ($(strip $(NVCC)),)
  ifneq ($(strip $(CUDA_PATH)),)
    NVCC := $(CUDA_PATH)/bin/nvcc
  else ifneq ($(strip $(CUDA_HOME)),)
    NVCC := $(CUDA_HOME)/bin/nvcc
  else
    NVCC := $(shell command -v nvcc 2>/dev/null)
  endif
endif

ifeq ($(strip $(NVCC)),)
  NVCC := /usr/local/cuda/bin/nvcc
endif

ifeq ($(strip $(CUDA_PATH)),)
  ifneq ($(strip $(CUDA_HOME)),)
    CUDA_PATH := $(CUDA_HOME)
  else
    CUDA_PATH := $(abspath $(dir $(NVCC))/..)
  endif
endif

ifeq ($(wildcard $(NVCC)),)
  $(error "nvcc not found. Set CUDA_PATH or NVCC to a valid CUDA installation")
endif

ifeq ($(wildcard $(CUDA_PATH)/include/cuda_runtime.h),)
  ifneq ($(wildcard /usr/local/cuda/include/cuda_runtime.h),)
    CUDA_PATH := /usr/local/cuda
  endif
endif

ifeq ($(wildcard $(CUDA_PATH)/include/cuda_runtime.h),)
  $(error "cuda_runtime.h not found. Set CUDA_PATH/CUDA_HOME to your CUDA root")
endif

CUDA_LIB_DIRS := $(CUDA_PATH)/lib64 $(CUDA_PATH)/targets/x86_64-linux/lib
CUDA_LIB_FLAGS := $(foreach d,$(CUDA_LIB_DIRS),-L$(d))
CUDA_RPATH_FLAGS := $(foreach d,$(CUDA_LIB_DIRS),-Wl,-rpath,$(d))

CCFLAGS := -O3 -I$(CUDA_PATH)/include
CUDA_EXTRA_CCFLAGS ?=

# Prefer native GPU architectures, then fall back to nvcc supported list.
NVCC_SUPPORTED_ARCH_LIST := $(strip $(shell $(NVCC) --list-gpu-arch 2>/dev/null | sed -n 's/^compute_//p' | tr '\n' ' '))
GPU_DETECTED_ARCH_LIST := $(strip $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ' | sed -n 's/^\([0-9]\+\)\.\([0-9]\+\)$$/\1\2/p' | sort -u | tr '\n' ' '))
ifeq ($(origin CUDA_ARCH_LIST), undefined)
  CUDA_ARCH_LIST := $(GPU_DETECTED_ARCH_LIST)
  ifneq ($(strip $(NVCC_SUPPORTED_ARCH_LIST)),)
    ifneq ($(strip $(CUDA_ARCH_LIST)),)
      CUDA_ARCH_LIST := $(filter $(NVCC_SUPPORTED_ARCH_LIST),$(CUDA_ARCH_LIST))
    else
      CUDA_ARCH_LIST := $(NVCC_SUPPORTED_ARCH_LIST)
    endif
  endif
endif
ifeq ($(strip $(CUDA_ARCH_LIST)),)
  CUDA_ARCH_LIST := 89 86 80 75
endif
CUDA_ARCH_LIST := $(strip $(CUDA_ARCH_LIST))

CUDA_ARCH_FLAGS := $(foreach arch,$(CUDA_ARCH_LIST),-gencode=arch=compute_$(arch),code=sm_$(arch))
CUDA_PTX_ARCH := $(lastword $(CUDA_ARCH_LIST))
CUDA_ARCH_FLAGS += -gencode=arch=compute_$(CUDA_PTX_ARCH),code=compute_$(CUDA_PTX_ARCH)
NVCCFLAGS := -O3 $(CUDA_ARCH_FLAGS)
CUDA_EXTRA_NVCCFLAGS ?=

# SM120 tuning defaults for single-arch cloud builds (can be disabled/overridden).
SM120_TUNE_ENABLE ?= 1
SM120_TUNE_DEFS ?= -DPNT_GROUP_NEW_GPU=16 -DBLOCK_SIZE_NEW_GPU=256
ifeq ($(strip $(CUDA_ARCH_LIST)),120)
  ifneq ($(strip $(SM120_TUNE_ENABLE)),0)
    CUDA_EXTRA_CCFLAGS += $(SM120_TUNE_DEFS)
    CUDA_EXTRA_NVCCFLAGS += $(SM120_TUNE_DEFS)
  endif
endif

CUDA_MAXRREGCOUNT ?=
ifneq ($(strip $(CUDA_MAXRREGCOUNT)),)
  CUDA_EXTRA_NVCCFLAGS += -Xptxas -maxrregcount=$(CUDA_MAXRREGCOUNT)
endif

LDFLAGS := $(CUDA_LIB_FLAGS) $(CUDA_RPATH_FLAGS) -lcudart -lcuda -pthread

CPU_SRC := RCKangaroo.cpp GpuKang.cpp Ec.cpp utils.cpp WildSpoolWriter.cpp
GPU_SRC := RCGpuCore.cu

CPP_OBJECTS := $(CPU_SRC:.cpp=.o)
CU_OBJECTS := $(GPU_SRC:.cu=.o)

TARGET := rckangaroo

all: $(TARGET)

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CXX) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CCFLAGS) $(CUDA_EXTRA_CCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_EXTRA_NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS) $(TARGET)
