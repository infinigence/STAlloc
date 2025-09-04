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

#ifndef BESTFIT_ALLOCATOR_H
#define BESTFIT_ALLOCATOR_H

#include "Allocator.hpp"
#include <algorithm>

#define alignment512(a) (((a) + 511) & (~511))
constexpr ssize_t MAX_BYTES = ssize_t(100) * 1024 * 1024 * 1024;

struct BestFitAllocator
{
    struct Block
    {
        void *ptr;
        ssize_t size;

        Block(void *ptr, ssize_t size) : ptr(ptr), size(size)
        {
        }
        bool operator<(const Block &other) const
        {
            return ptr < other.ptr;
        }
    };

    const ssize_t device;
    ssize_t max_reserved;
    const void *global_ptr;
    ssize_t max_allocated;

    std::set<Block> BlockPool;
    ssize_t allocated_bytes;

    ssize_t id;
    const std::unordered_set<ssize_t> skip_ids;

    BestFitAllocator(ssize_t device, ssize_t max_reserved, void *global_ptr)
        : device(device), max_reserved(max_reserved), global_ptr(global_ptr), max_allocated(0), allocated_bytes(0)
    {
        BlockPool.emplace(global_ptr, max_reserved);
    }

    BestFitAllocator(ssize_t device, ssize_t max_reserved, void *global_ptr, std::unordered_set<ssize_t> skip_ids)
        : device(device), max_reserved(max_reserved), global_ptr(global_ptr), max_allocated(0), allocated_bytes(0),
          skip_ids(skip_ids), id(-1)
    {
        BlockPool.emplace(global_ptr, max_reserved);
    }

    bool init()
    {
        bool is_released = (this->allocated_bytes == 0) && (BlockPool.size() == 1) && (BlockPool.begin()->ptr == global_ptr);
        this->id = -1;
        return is_released;
    }

    void *alloc_fn_(ssize_t size)
    {
        if (this->skip_ids.count(++(this->id)))
        {
            // LOG_PRINT(WARNING, "BestFitAllocator skip id =", this->id, "size =", size);
            return nullptr;
        }

        size = alignment512(size);
        auto best_it = BlockPool.end();
        ssize_t min_size = MAX_BYTES;
        for (auto it = BlockPool.begin(); it != BlockPool.end(); ++it)
        {
            if (it->size >= size && (it->size < min_size || (it->size == min_size && it->ptr < best_it->ptr)))
            {
                min_size = it->size;
                best_it = it;
            }
        }
        if (best_it == BlockPool.end())
        {
            // if(device == 0){
            //     LOG_PRINT(WARNING, "BestFitAllocator can't find available block");
            // }
            return nullptr;
        }

        Block allocated = *best_it;
        void *ptr = allocated.ptr;

        BlockPool.erase(best_it);
        if (allocated.size > size)
        {
            BlockPool.emplace(allocated.ptr + size, allocated.size - size);
        }

        this->allocated_bytes += size;
        this->max_allocated = std::max(this->max_allocated, this->allocated_bytes);

        return ptr;
    }

    void free_fn_(void *ptr, ssize_t size)
    {
        size = alignment512(size);
        this->allocated_bytes -= size;

        Block new_block(ptr, size);
        auto ret = BlockPool.insert(new_block);
        assert(ret.second);

        auto it = ret.first;
        while (it != BlockPool.begin())
        {
            auto prev = std::prev(it);
            if (prev->ptr + prev->size == it->ptr)
            {
                Block merged(prev->ptr, prev->size + it->size);
                BlockPool.erase(prev);
                BlockPool.erase(it);
                it = BlockPool.insert(merged).first;
            }
            else
            {
                break;
            }
        }
        while (true)
        {
            auto next_it = std::next(it);
            if (next_it != BlockPool.end() && it->ptr + it->size == next_it->ptr)
            {
                Block merged(it->ptr, it->size + next_it->size);
                BlockPool.erase(it);
                BlockPool.erase(next_it);
                it = BlockPool.insert(merged).first;
            }
            else
            {
                break;
            }
        }
    }

    std::pair<ssize_t, ssize_t> getStat()
    {
        ssize_t _max_allocated = this->max_allocated;
        this->max_allocated = 0;
        return std::make_pair(_max_allocated, this->allocated_bytes); // max_allocated, cur_allocated
    }

    BestFitAllocator(const BestFitAllocator &) = delete;
    BestFitAllocator &operator=(const BestFitAllocator &) = delete;
};

#endif // BESTFIT_ALLOCATOR_H