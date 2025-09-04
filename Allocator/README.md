# STAlloc 
## Compilation

### Nvidia
```makefile
CXXFLAGS = \
-Wall -fPIC -O3 -std=c++17 \
-I/usr/local/cuda/include \ # CUDA_HOME/include
-I/usr/include/python3.10 -I/usr/local/lib/python3.10/dist-packages/pybind11/include \ # pybind11-config --includes

LDFLAGS = \
-L/usr/local/cuda/lib64 -lcudart \ # CUDA_HOME/lib64 && cudart
-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -lpython3.10 -lcrypt -ldl -lm -lm \ # python3-config --ldflags
```

### AMD
* `CUDA_HOME` => `ROCM_HOME`
* `cudart` => `amdhip64`
* add `-D__HIP_PLATFORM_AMD__` in `CXXFLAGS`

### Muxi
* add `-D__MUXI_PLATFORM__` in `CXXFLAGS`
* remove `-lcudart`