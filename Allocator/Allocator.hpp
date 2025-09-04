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

#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include "CUDACachingAllocator.h"
#include "Hardware_Allocator.h"
#include "util.h"
#include <unordered_set>

#include <assert.h>
#include <iostream>
#include <mutex>
#include <stack>
#include <stdexcept>
#include <unordered_map>
#include <vector>

static constexpr ssize_t VAR_BLOCK_SIZE = 512;

struct Allocator
{
    ssize_t max_reserved;
    ssize_t max_allocated;
    void *global_ptr;
    void *edge_ptr;
    const ssize_t device;
    Allocator();
    Allocator(ssize_t device, ssize_t max_reserved, ssize_t max_allocated = 0)
        : device(device), max_reserved(max_reserved), max_allocated(max_allocated)
    {
        C10_CUDA_CHECK(cudaMalloc(&(this->global_ptr), max_reserved));
        this->edge_ptr = this->global_ptr + max_reserved;
    }
    ~Allocator()
    {
        if (this->global_ptr)
        {
            C10_CUDA_CHECK(cudaFree(this->global_ptr));
        }
    }

    virtual void *alloc_fn_(ssize_t size) = 0;
    virtual void free_fn_(void *ptr, ssize_t size) = 0;

    // TODO:: process cudaStream_t
    void *malloc(ssize_t size, int device, cudaStream_t stream)
    {
        void *ptr = alloc_fn_(size);
        return ptr;
    }

    void free(void *ptr, ssize_t size, int device, cudaStream_t stream)
    {
        free_fn_(ptr, size);
    }

    inline bool is_allcoated(void *ptr)
    {
        return ptr >= this->global_ptr && ptr < this->edge_ptr;
    }

    virtual ssize_t getMaxAllocated()
    {
        return this->max_allocated;
    }

    std::pair<ssize_t, ssize_t> getStat()
    {
        return std::make_pair(this->getMaxAllocated(), this->max_reserved);
    }
};

// -----------------------------------------------------------CUDACachingAllocator----------------------------------------------------------

auto deviceCachingAllocator = DeviceCachingAllocator::getInstance();
std::unordered_map<void *, Block *> Block_map;

void *deviceCachingAllocator_malloc(ssize_t size, int device, cudaStream_t stream)
{
    if(size == 0){
        return nullptr;
    }
    Block *block = deviceCachingAllocator->malloc(device, size, stream);
    Block_map.emplace(block->ptr, block);
    return block->ptr;
}

void deviceCachingAllocator_free(void *ptr, ssize_t size, int device, cudaStream_t stream)
{
    if(size == 0){
        return;
    }
    Block *block = Block_map[ptr];
    Block_map.erase(ptr);
    return deviceCachingAllocator->free(block);
}
#endif // ALLOCATOR_H
