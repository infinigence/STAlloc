CXX = g++
# python3-config --includes   python3-config --ldflags

PYTHON_VERSION := $(shell python3 -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
PYTHON_INCLUDES := $(shell python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
PYTHON_LDFLAGS := $(shell python3-config --ldflags)
PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)

CUDA_PATH ?= /usr/local/cuda
CUDA_INCLUDES = -I$(CUDA_PATH)/include
CUDA_LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart

CXXFLAGS = -Wall -fPIC -O3 -std=c++17 $(CUDA_INCLUDES) -I$(PYTHON_INCLUDES) $(PYBIND11_INCLUDES)
LDFLAGS = $(CUDA_LDFLAGS) $(PYTHON_LDFLAGS)

SRCS = main.cpp CUDACachingAllocator.cpp
HDRS = Hardware_Allocator.h Allocator.hpp BestFitAllocator.hpp StaticAllocator.hpp VarAllocator.hpp OnlineDynamicManager.hpp CUDACachingAllocator.h util.h

# Target shared libraries
TARGET1 = alloc.so
TARGET2 = trace.so

# Default target
all: $(TARGET1) $(TARGET2)

# Rule to compile the first shared library
$(TARGET1): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ $(LDFLAGS)

# Rule to compile the second shared library
$(TARGET2): trace.cpp CUDACachingAllocator.cpp Hardware_Allocator.h util.h Allocator.hpp OnlineDynamicManager.hpp CUDACachingAllocator.h
	$(CXX) $(CXXFLAGS) -shared -o $@ $^

clean:
	rm -f $(TARGET1) $(TARGET2) *.o

.PHONY: all clean