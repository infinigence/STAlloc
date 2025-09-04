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

#ifndef VAR_ALLOCATOR_H
#define VAR_ALLOCATOR_H

#include "Allocator.hpp"

struct VarAllocator : public Allocator
{
    std::stack<void *> avaliable_table;
    std::vector<void *> blocks;

    VarAllocator(ssize_t device, ssize_t max_reserved) : Allocator(device, max_reserved)
    {
        this->blocks.emplace_back(this->global_ptr);
        for (int i = 0; i < max_reserved; i += VAR_BLOCK_SIZE)
        {
            this->avaliable_table.push(this->global_ptr + i);
        }
    }

    void *alloc_fn_(ssize_t size) override
    {
        if (this->avaliable_table.empty())
        {
            this->realloc();
            LOG_PRINT(INFO, "VarAllocator realloc(). current VarAllocator size is", this->max_reserved);
        }
        assert(!this->avaliable_table.empty());
        void *ptr = this->avaliable_table.top();
        this->avaliable_table.pop();

        ssize_t cur_usage = this->max_reserved - this->avaliable_table.size() * VAR_BLOCK_SIZE;
        this->max_allocated = std::max(this->max_allocated, cur_usage);
        return ptr;
    }
    void free_fn_(void *ptr, ssize_t size) override
    {
        this->avaliable_table.push(ptr);
    }

    void realloc()
    {
        ssize_t realloc_size = this->max_reserved;
        void *ptr;
        C10_CUDA_CHECK(cudaMalloc(&ptr, realloc_size));
        this->blocks.emplace_back(ptr);
        for (int i = 0; i < realloc_size; i += VAR_BLOCK_SIZE)
        {
            this->avaliable_table.push(ptr + i);
        }
        this->max_reserved += realloc_size;
    }

    ssize_t getMaxAllocated() override
    {
        ssize_t res = this->max_allocated;
        this->max_allocated = 0;
        return res;
    }

    VarAllocator(const VarAllocator &) = delete;
    VarAllocator &operator=(const VarAllocator &) = delete;
};

#endif // VAR_ALLOCATOR_H
