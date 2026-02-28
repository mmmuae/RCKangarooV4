CXX ?= g++
CUDA_PATH ?=
CUDA_HOME ?=
RCK_DEFS ?=

ifeq ($(strip $(CUDA_PATH)),)
  ifneq ($(strip $(CUDA_HOME)),)
    CUDA_PATH := $(CUDA_HOME)
  else ifneq ($(wildcard /usr/local/cuda/include/cuda_runtime.h),)
    CUDA_PATH := /usr/local/cuda
  else
    CUDA_PATH := /usr
  endif
endif

ifeq ($(wildcard $(CUDA_PATH)/include/cuda_runtime.h),)
  $(error "cuda_runtime.h not found. Set CUDA_PATH/CUDA_HOME to your CUDA root")
endif

CUDA_LIB_DIRS := $(CUDA_PATH)/lib64 $(CUDA_PATH)/targets/x86_64-linux/lib
CUDA_LIB_FLAGS := $(foreach d,$(CUDA_LIB_DIRS),-L$(d))
CUDA_RPATH_FLAGS := $(foreach d,$(CUDA_LIB_DIRS),-Wl,-rpath,$(d))

CCFLAGS := -O3 -std=c++17 -I$(CUDA_PATH)/include $(RCK_DEFS)
LDFLAGS := $(CUDA_LIB_FLAGS) $(CUDA_RPATH_FLAGS) -lcudart -lcuda -pthread

CPU_SRC := RCKangaroo.cpp GpuKang.cpp Ec.cpp utils.cpp WildSpoolWriter.cpp
CPP_OBJECTS := $(CPU_SRC:.cpp=.o)

TARGET := rckangaroo

all: $(TARGET)

$(TARGET): $(CPP_OBJECTS)
	$(CXX) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $< -o $@

clean:
	rm -f $(CPP_OBJECTS) $(TARGET)
