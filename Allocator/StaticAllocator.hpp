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

#ifndef STATIC_ALLOCATOR_H
#define STATIC_ALLOCATOR_H

#include "Allocator.hpp"
struct StaticAllocator : public Allocator
{

    std::vector<std::vector<ssize_t>> plans;
    ssize_t iter0_event_num;
    ssize_t iter1_event_num;
    ssize_t alloc_event_num;
    ssize_t event_id;
    ssize_t cur_iteraion;
    std::set<std::pair<ssize_t, ssize_t>> active_blocks;
    ssize_t FALLBACK_LEVEL;

    StaticAllocator(ssize_t device, ssize_t max_reserved, ssize_t max_allocated, ssize_t iter0_event_num,
                    ssize_t iter1_event_num, std::vector<std::vector<ssize_t>> plans)
        : Allocator(device, max_reserved, max_allocated), iter0_event_num(iter0_event_num),
          iter1_event_num(iter1_event_num), plans(plans)
    {
        this->alloc_event_num = iter0_event_num + iter1_event_num;
        this->event_id = -1;
        this->cur_iteraion = 0;
        FALLBACK_LEVEL = getenv_int("STALLOC_STATIC_FALLBACK", 0);
    }

    void update_iteraion()
    {
        this->event_id = this->iter0_event_num - 1;
    }

    void *alloc_fn_(ssize_t size) override
    {
        ++this->event_id;
        if (this->event_id == this->alloc_event_num)
        {
            this->event_id = this->iter0_event_num;
        }
        if (this->event_id == this->iter0_event_num)
        {
            ++cur_iteraion;
        }
        if (this->event_id >= alloc_event_num)
        {
            LOG_PRINT(WARNING, "StaticAllocator MISMATCH(Out of Plan), device =", this->device, "size =", size);
            return deviceCachingAllocator_malloc(size, device, nullptr);
        }

        if (this->plans[this->event_id][1] != size || this->plans[this->event_id][0] == -1)
        {
            LOG_PRINT(WARNING, "StaticAllocator MISMATCH, device =", this->device, ",event_id =", this->event_id,
                      ",offset =", this->plans[this->event_id][0], ", plan_size =", this->plans[this->event_id][1],
                      ",real_size =", size);

            return deviceCachingAllocator_malloc(size, device, nullptr);
        }

        void *ptr = this->global_ptr + this->plans[this->event_id][0];

        if (FALLBACK_LEVEL > 0)
        {
            if (is_overlap(ptr, size))
            {
                // LOG_PRINT(WARNING, "StaticAllocator OVERLAP, device =", this->device, ",event_id =", this->event_id,
                //           ",offset =", this->plans[this->event_id][0], ", plan_size =", this->plans[this->event_id][1],
                //           ",real_size =", size);
                return deviceCachingAllocator_malloc(size, device, nullptr);
            }
            this->active_blocks.emplace((ssize_t)ptr, size);
        }

        return ptr;
    }

    void free_fn_(void *ptr, ssize_t size) override
    {
        if (FALLBACK_LEVEL > 0)
        {
            assert(this->active_blocks.count({(ssize_t)ptr, size}) == 1);
            this->active_blocks.erase({(ssize_t)ptr, size});
        }
    }

    bool is_overlap(void *ptr, ssize_t size)
    {
        std::pair<ssize_t, ssize_t> key = std::make_pair((ssize_t)ptr, 0);
        auto lower = active_blocks.lower_bound(key);
        auto upper = active_blocks.upper_bound(key);
        if (lower != active_blocks.begin())
        {
            --lower;
            ssize_t prev_block_end = (*lower).first + (*lower).second;
            if ((ssize_t)ptr < prev_block_end)
            {
                return true;
            }
        }
        if (upper != active_blocks.end())
        {
            ssize_t next_block_start = (*upper).first;
            if ((ssize_t(ptr) + size) > next_block_start)
            {
                return true;
            }
        }
        return false;
    }

    StaticAllocator(const StaticAllocator &) = delete;
    StaticAllocator &operator=(const StaticAllocator &) = delete;
};
#endif // STATIC_ALLOCATOR_H