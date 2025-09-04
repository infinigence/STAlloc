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

#ifndef SEMI_ALLOCATOR_H
#define SEMI_ALLOCATOR_H

#include "Allocator.hpp"

#define alignment512(a) (((a) + 511) & (~511))

struct SemiAllocator : public Allocator
{
    const ssize_t alloc_event_num;
    const std::unordered_set<ssize_t> target_event_idx;
    ssize_t event_id;
    ssize_t used_size;
    ssize_t avaliable_size;
    bool in_exception;

    SemiAllocator(ssize_t device, ssize_t max_reserved, ssize_t alloc_event_num,
                  std::unordered_set<ssize_t> target_event_idx)
        : Allocator(device, max_reserved), alloc_event_num(alloc_event_num), target_event_idx(target_event_idx),
          event_id(-1), used_size(0), avaliable_size(max_reserved), in_exception(false)
    {
    }

    inline bool is_tail_memory_enough(ssize_t size)
    {
        return this->avaliable_size >= size;
    }

    inline void update_event_id()
    {
        ++this->event_id;
    }

    void *alloc_fn_(ssize_t size)
    {
        this->update_event_id();
        if (this->in_exception)
        {
            LOG_PRINT(WARNING, "SemiDynamicAllocator in exception, event_id =", this->event_id);
            return deviceCachingAllocator_malloc(size, device, nullptr);
        }
        if (this->target_event_idx.count(this->event_id))
        {
            if (!this->is_tail_memory_enough(size))
            {
                LOG_PRINT(WARNING, "SemiDynamicAllocator OOM, event_id =", this->event_id, ",size =", size);
                return deviceCachingAllocator_malloc(size, device, nullptr);
            }
            void *ptr = this->global_ptr + this->used_size;
            ssize_t size_aligned = alignment512(size);
            this->used_size += size_aligned;
            this->avaliable_size -= size_aligned;

            this->max_allocated = std::max(this->max_allocated, this->used_size);
            return ptr;
        }
        return deviceCachingAllocator_malloc(size, device, nullptr);
    }

    // Assuming between two consecutive calls to reset_status(), two groups of tensors are continuously allocated/freed, 
    // with no temporal overlap between tensor groups
    void free_fn_(void *ptr, ssize_t size)
    {
        this->avaliable_size += alignment512(size);
        if (this->in_exception && this->avaliable_size == this->max_reserved)
        {
            this->in_exception = false;
        }
    }

    inline bool try_reset_status(int device)
    {
        bool is_available = (this->avaliable_size == this->max_reserved);
        if (is_available)
        {
            this->reset_status(device);
        }
        return is_available;
    }

    inline void reset_status(int device)
    {
        this->event_id = -1;
        this->used_size = 0;
        if (this->avaliable_size != this->max_reserved)
        {
            this->in_exception = true;
        }
        this->in_exception = false;
    }

    ssize_t getMaxAllocated() override
    {
        ssize_t res = this->max_allocated;
        this->max_allocated = 0;
        return res;
    }

    SemiAllocator(const SemiAllocator &) = delete;
    SemiAllocator &operator=(const SemiAllocator &) = delete;
};

#endif // SEMI_ALLOCATOR_H