/**
 * Copyright 2025 Infinigence AI.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HARDWARE_ALLOCATOR_H
#define HARDWARE_ALLOCATOR_H

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime_api.h>
typedef hipStream_t cudaStream_t;
typedef hipError_t cudaError_t;
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaSuccess hipSuccess
#define cudaErrorMemoryAllocation hipErrorMemoryAllocation
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMemGetInfo hipMemGetInfo
#elif defined(__MUXI_PLATFORM__)
#include <mcr/mc_runtime_api.h>
typedef mcStream_t cudaStream_t;
typedef mcError_t cudaError_t;
#define cudaMalloc mcMalloc
#define cudaFree mcFree
#define cudaSuccess mcSuccess
#define cudaErrorMemoryAllocation mcErrorMemoryAllocation
#define cudaGetErrorString mcGetErrorString
#define cudaGetLastError mcGetLastError
#define cudaMemGetInfo mcMemGetInfo
#else
#include <cuda_runtime_api.h>
#endif

#include <stdlib.h>
#include <sys/types.h>

#endif // HARDWARE_ALLOCATOR_H
