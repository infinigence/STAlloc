/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * This file is derived from PyTorch's CUDACachingAllocator implementation
 * and includes local modifications.
 *
 * Original work:
 *   Copyright (c) 2016-present, Facebook, Inc. and its affiliates.
 *   Licensed under the BSD 3-Clause License.
 *   See THIRD_PARTY_LICENSES/PYTORCH-BSD-3-CLAUSE.txt for the full license text.
 *
 * Modifications:
 *   Copyright (c) 2025, Infinigence AI.
 */
#include "CUDACachingAllocator.h"
#include "llvmMathExtras.h"
#include "util.h"

#define alignment512(a) (((a) + 511) & (~511))

using namespace c10;

static std::string format_size(uint64_t size)
{
    std::ostringstream os;
    os.precision(2);
    os << std::fixed;
    if (size <= 1024)
    {
        os << size << " bytes";
    }
    else if (size <= 1048576)
    {
        os << (size / 1024.0);
        os << " KiB";
    }
    else if (size <= 1073741824ULL)
    {
        os << size / 1048576.0;
        os << " MiB";
    }
    else
    {
        os << size / 1073741824.0;
        os << " GiB";
    }
    return os.str();
}

//...device stat opts  begin...//
void update_stat(Stat &stat, int64_t amount)
{
    stat.current += amount;
    stat.peak = std::max(stat.current, stat.peak);
    if (amount > 0)
    {
        stat.allocated += amount;
    }
    if (amount < 0)
    {
        stat.freed += -amount;
    }
}

void reset_accumulated_stat(Stat &stat)
{
    stat.allocated = 0;
    stat.freed = 0;
}

void reset_peak_stat(Stat &stat)
{
    stat.peak = stat.current;
}

template <typename Func> void for_each_selected_stat_type(const StatTypes &stat_types, Func f)
{
    // original style: for (const auto stat_type : c10::irange(stat_types.size())) {
    for (int i = 0; i < stat_types.size(); ++i)
    {
        if (stat_types[i])
        {
            f(i);
        }
    }
}

void update_stat_array(StatArray &stat_array, int64_t amount, const StatTypes &stat_types)
{
    for_each_selected_stat_type(
        stat_types, [&stat_array, amount](size_t stat_type) { update_stat(stat_array[stat_type], amount); });
}
//...device stat opts  end...//

DeviceCachingAllocator::DeviceCachingAllocator()
    : large_blocks(BlockComparator, /*is_small=*/false), small_blocks(BlockComparator, /*is_small=*/true)
{
    stats.max_split_size = CachingAllocatorConfig::max_split_size();
}

void DeviceCachingAllocator::init_empty_config(int utilization_threshold, int diff_threshold)
{
    this->Dynamic_Utilization_Threshold = getenv_int("STALLOC_DYNAMIC_UTILIZATION", utilization_threshold);
    this->Dynamic_Diff_Threshold = getenv_int("STALLOC_DYNAMIC_DIFF", diff_threshold);

    assert(Dynamic_Utilization_Threshold >= 0 && Dynamic_Utilization_Threshold <= 100 &&
           "[STAlloc-Error] Dynamic_Utilization_Threshold(%) should be in range(0, 100)");
    assert(Dynamic_Diff_Threshold >= 0 && "[STAlloc-Error] Dynamic_Diff_Threshold(GB) <= 0");
    LOG_PRINT(INFO, "STALLOC_DYNAMIC_UTILIZATION =", Dynamic_Utilization_Threshold,
              "%, STALLOC_DYNAMIC_DIFF =", Dynamic_Diff_Threshold, "MB");
    Dynamic_Diff_Threshold *= (ssize_t)1024 * 1024;
}

// This function takes the size and number of divisions argument and rounds
// up the size argument for the nearest power-of-2 division.
// For example, if we need to round-up 1200 and number of divisions is 4,
// the size 1200 lies between 1024 and 2048 and if we do 4 divisions between
// them, the values are 1024, 1280, 1536, and 1792. So the function will
// return 1280 as the nearest ceiling of power-2 divison.

BlockPool &DeviceCachingAllocator::get_pool(size_t size, cudaStream_t stream)
{
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    // captures_underway is a conservative guess that the current stream may be
    // capturing. It's only > 0 if some thread has begun and not yet ended a
    // capture, so it's usually 0, and we can short-circuit
    // cudaStreamCaptureStatus (which does a TLS lookup).
    if (C10_UNLIKELY(captures_underway))
    {
        CaptureId_t id;
        cudaStreamCaptureStatus status;
        C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &id));
        if (status != cudaStreamCaptureStatus::cudaStreamCaptureStatusNone)
        {
            TORCH_INTERNAL_ASSERT(status != cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated);
            // Retrieves the private pool assigned to this capture.
            auto it0 = capture_to_pool_map.find(id);
            TORCH_INTERNAL_ASSERT(it0 != capture_to_pool_map.end());
            auto it1 = graph_pools.find(it0->second);
            TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
            if (size <= kSmallSize)
            {
                return it1->second->small_blocks;
            }
            else
            {
                return it1->second->large_blocks;
            }
        }
    }
#endif
    if (size <= kSmallSize)
    {
        return small_blocks;
    }
    else
    {
        return large_blocks;
    }
}

StatType DeviceCachingAllocator::get_stat_type_for_pool(const BlockPool &pool)
{
    return pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL;
}

// search block in pools -> found best block if it has  -> create a new block if it hasn't.
Block *DeviceCachingAllocator::malloc(int device, size_t orig_size, cudaStream_t stream)
{

    // mutex create:
    // std::unique_lock<std::recursive_mutex> lock(mutex);
    // block info create：
    // size_t size = round_size(orig_size);                 //  rounded to times of 512；
    size_t size = alignment512(orig_size);
    auto &pool = get_pool(size, stream);                 //
    const size_t alloc_size = get_allocation_size(size); // alloc size suggestion。
    AllocParams params(device, size, stream, &pool, alloc_size, stats);
    // change stat_types
    params.stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    params.stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;

    // First, try to get a block from the existing pool.
    bool block_found =
        // Search pool
        get_free_block(params)
        // Trigger callbacks and retry search
        || (trigger_free_memory_callbacks(params) && get_free_block(params));

    if (!block_found)
    {
        // Do garbage collection if the flag is set.
        if (C10_UNLIKELY(set_fraction && CachingAllocatorConfig::garbage_collection_threshold() > 0.0))
        {
            garbage_collect_cached_blocks();
        }
        // Attempt allocate
        block_found = alloc_block(params, false)
                      // Free enough available cached blocks to satisfy alloc and retry
                      // alloc.
                      || (release_available_cached_blocks(params) && alloc_block(params, false))
                      // Free all non-split cached blocks and retry alloc.
                      || (C10_LIKELY(captures_underway == 0) && release_cached_blocks() && alloc_block(params, true));

        if (!block_found)
        {
            // For any error code other than cudaErrorMemoryAllocation,
            // alloc_block should have thrown an exception already.
            TORCH_INTERNAL_ASSERT(params.err == cudaErrorMemoryAllocation);

            size_t device_free;
            size_t device_total;
            C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
            std::string allowed_info;

            if (set_fraction)
            {
                allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
            }

            stats.num_ooms += 1;

            // "total capacity": total global memory on GPU
            // "allowed": memory is allowed to use, which set by fraction.
            // "already allocated": memory allocated by the program using the
            //                      caching allocator
            // "free": free memory as reported by the CUDA API
            // "cached": memory held by the allocator but not used by the program
            //
            // The "allocated" amount  does not include memory allocated outside
            // of the caching allocator, such as memory allocated by other programs
            // or memory held by the driver.
            //
            // The sum of "allocated" + "free" + "cached" may be less than the
            // total capacity due to memory held by the driver and usage by other
            // programs.
            //
            // Note that at this point free_cached_blocks has already returned all
            // possible "cached" memory to the driver. The only remaining "cached"
            // memory is split from a larger block that is partially in-use.
            TORCH_CHECK_WITH(false, "STAlloc : CUDA out of memory. Tried to allocate ", format_size(alloc_size),
                             " (GPU ", device, "; ", format_size(device_total), " total capacity; ",
                             format_size(stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
                             " already allocated; ", format_size(device_free), " free; ", allowed_info,
                             format_size(stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current),
                             " reserved in total by PyTorch)",
                             " If reserved memory is >> allocated memory try setting max_split_size_mb to avoid"
                             " fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF",
                             "");
        }
    }

    TORCH_INTERNAL_ASSERT(params.err == cudaSuccess && params.block != nullptr && params.block->ptr != nullptr);
    Block *block = params.block;
    Block *remaining = nullptr;

    const bool already_split = block->is_split();
    if (should_split(block, size))
    {
        remaining = block;

        block = new Block(device, stream, size, &pool, block->ptr);
        block->prev = remaining->prev;
        if (block->prev)
        {
            block->prev->next = block;
        }
        block->next = remaining;

        remaining->prev = block;
        remaining->ptr = static_cast<char *>(remaining->ptr) + size;
        remaining->size -= size;
        bool inserted = pool.blocks.insert(remaining).second;
        // TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

        // if (context) {
        //   trimHistoryBefore(remaining, (char*)block->ptr + size);
        // }

        if (already_split)
        {
            // An already-split inactive block is being shrunk by size bytes.
            update_stat_array(stats.inactive_split_bytes, -block->size, params.stat_types);
        }
        else
        {
            // A new split inactive block is being created from a previously unsplit
            // block, size remaining->size bytes.
            for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
                update_stat(stats.inactive_split_bytes[stat_type], remaining->size);
                update_stat(stats.inactive_split[stat_type], 1);
            });
        }
    }
    else if (already_split)
    {
        // An already-split block is becoming active
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
            update_stat(stats.inactive_split_bytes[stat_type], -block->size);
            update_stat(stats.inactive_split[stat_type], -1);
        });
    }

    block->allocated = true;
    // if (context) {
    //   trimHistoryBefore(block, (char*)block->ptr + size);
    //   block->history = std::make_unique<History>(History{
    //       block->ptr,
    //       orig_size,
    //       std::move(context),
    //       std::move(block->history)});
    //   if (!block->history_last) {
    //     block->history_last = block->history.get();
    //   }
    // }

    bool inserted = active_blocks.insert(block).second;
    // TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        update_stat(stats.allocation[stat_type], 1);
        update_stat(stats.allocated_bytes[stat_type], block->size);
        update_stat(stats.active[stat_type], 1);
        update_stat(stats.active_bytes[stat_type], block->size);
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
        update_stat(stats.oversize_allocations, 1);

    // c10::reportMemoryUsageToProfiler(
    //     block->ptr,
    //     block->size,
    //     stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
    //     stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
    //     c10::Device(c10::DeviceType::CUDA, device));

    return block;
}

// Dose not invoke cudaFree。
void DeviceCachingAllocator::free(Block *block)
{
    // std::lock_guard<std::recursive_mutex> lock(mutex);

    block->allocated = false;

    // following logic might modifying underlaying Block, causing the size
    // changed. We store ahead for reporting
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
        update_stat(stats.allocation[stat_type], -1);
        update_stat(stats.allocated_bytes[stat_type], -block->size);
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
        update_stat(stats.oversize_allocations, -1);

    if (!block->stream_uses.empty())
    {
        if (C10_UNLIKELY(captures_underway))
        {
            // It's forbidden to cudaEventQuery an event recorded during CUDA graph
            // capture. We conservatively defer recording end-of-life events until
            // the next call to process_events() (which won't happen until no
            // captures are underway)

            // needs_events_deferred_until_no_capture.push_back(block);
        }
        else
        {
            // insert_events(block);
        }
    }
    else
    {
        free_block(block);
    }

    // c10::reportMemoryUsageToProfiler(
    //     orig_block_ptr,
    //     -orig_block_size,
    //     stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
    //     stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
    //     c10::Device(c10::DeviceType::CUDA, block->device));
}

void DeviceCachingAllocator::free_block(Block *block)
{
    TORCH_INTERNAL_ASSERT(!block->allocated && block->event_count == 0 && block->stream_uses.empty());

    size_t original_block_size = block->size;

    auto &pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Block *, 2> merge_candidates = {block->prev, block->next};
    for (Block *merge_candidate : merge_candidates)
    {
        const int64_t subsumed_size = try_merge_blocks(block, merge_candidate, pool);
        if (subsumed_size > 0)
        {
            net_change_inactive_split_blocks -= 1;
            net_change_inactive_split_size -= subsumed_size;
        }
    }

    active_blocks.erase(block);
    // Makes sure the Block* isn't already present in the pool we're freeing it
    // back into.
    bool inserted = pool.blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT(inserted);

    if (block->is_split())
    {
        net_change_inactive_split_blocks += 1;
        net_change_inactive_split_size += block->size;
    }

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
        update_stat(stats.inactive_split[stat_type], net_change_inactive_split_blocks);
        update_stat(stats.inactive_split_bytes[stat_type], net_change_inactive_split_size);
        update_stat(stats.active[stat_type], -1);
        update_stat(stats.active_bytes[stat_type], -original_block_size);
    });
}

/** combine previously split blocks. returns the size of the subsumed block,
 * or 0 on failure. */
size_t DeviceCachingAllocator::try_merge_blocks(Block *dst, Block *src, BlockPool &pool)
{
    if (!src || src->allocated || src->event_count > 0 || !src->stream_uses.empty())
    {
        return 0;
    }

    TORCH_CHECK(dst->is_split() && src->is_split());

    if (dst->prev == src)
    { // [src dst]
        dst->ptr = src->ptr;
        dst->prev = src->prev;
        if (dst->prev)
        {
            dst->prev->next = dst;
        }
        if (!dst->history)
        {
            dst->history = std::move(src->history);
            dst->history_last = src->history_last;
        }
        else if (src->history)
        {
            src->history_last->next = std::move(dst->history);
            dst->history = std::move(src->history);
        }
        src->history_last = nullptr;
    }
    else
    { // [dest src]
        dst->next = src->next;
        if (dst->next)
        {
            dst->next->prev = dst;
        }

        if (!dst->history)
        {
            dst->history = std::move(src->history);
            dst->history_last = src->history_last;
        }
        else if (src->history)
        {
            dst->history_last->next = std::move(src->history);
            dst->history_last = src->history_last;
        }
        src->history_last = nullptr;
    }
    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    auto erased = pool.blocks.erase(src);
    // TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    delete src;

    return subsumed_size;
}

bool DeviceCachingAllocator::get_free_block(AllocParams &p)
{
    BlockPool &pool = *p.pool;

    auto it = pool.blocks.lower_bound(&p.search_key); // set-container search, return minium satisfied value.
    if (it == pool.blocks.end() || (*it)->stream != p.stream())
        return false;

    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size()))
        return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) && ((*it)->size >= p.size() + kLargeBuffer))
        return false;
    p.block = *it;
    (*it)->gc_count = 0; // Denote this block has been used
    pool.blocks.erase(it);
    return true;
}

// only one invoking
bool DeviceCachingAllocator::trigger_free_memory_callbacks(AllocParams &p)
{
    bool freed_memory = false;
    // code commented
    // for (const auto& name : FreeCudaMemoryCallbacksRegistry()->Keys()) {
    //   freed_memory |=
    //       FreeCudaMemoryCallbacksRegistry()->Create(name)->Execute();
    // }
    return freed_memory;
}

void DeviceCachingAllocator::garbage_collect_cached_blocks()
{
    // Free unused cached blocks to reclaim GPU memory.
    // Unlike release_cached_blocks(), this does not enforce synchronization and
    // therefore should be of less overheads.

    size_t gc_threshold =
        static_cast<size_t>(CachingAllocatorConfig::garbage_collection_threshold() * allowed_memory_maximum);
    // No need to trigger GC yet
    if (total_allocated_memory <= gc_threshold)
    {
        return;
    }
    const auto target_size = total_allocated_memory - gc_threshold;
    size_t gc_reclaimed = 0;

    // Calculate the total age of the free-able blocks. We'll use it later to
    // get "avg age" threshold.
    double total_age = 0.0;
    int freeable_block_count = 0;
    for (auto &b : large_blocks.blocks)
    {
        if (!b->is_split())
        {
            total_age += b->gc_count;
            ++freeable_block_count;
        }
    }
    // No free-able blocks?
    if (freeable_block_count == 0)
    {
        return;
    }

    // Repeat GC until we reach reclaim > target size.
    bool block_freed = true;
    while (gc_reclaimed < target_size && block_freed == true && freeable_block_count > 0)
    {
        // Free blocks exceeding this age threshold first.
        double age_threshold = total_age / freeable_block_count;
        // Stop iteration if we can no longer free a block.
        block_freed = false;

        // Free blocks of > avg age. Don't stop upon reaching the target_size,
        // we don't want this GC to be triggered frequently.
        auto it = large_blocks.blocks.begin();
        while (it != large_blocks.blocks.end())
        {
            Block *block = *it;
            ++it;
            if (!block->is_split() && block->gc_count >= age_threshold)
            {
                block_freed = true;
                gc_reclaimed += block->size;
                total_age -= block->gc_count; // Decrement the age
                freeable_block_count--;       // One less block that can be freed
                release_block(block);
            }
        }
    }
}

bool DeviceCachingAllocator::alloc_block(AllocParams &p, bool isRetry)
{
    // Defensively checks for preexisting CUDA error state.
    C10_CUDA_CHECK(cudaGetLastError());

    size_t size = p.alloc_size;
    void *ptr;

    if (isRetry)
    {
        stats.num_alloc_retries += 1;
    }

    if (set_fraction && total_allocated_memory + size > allowed_memory_maximum)
    {
        p.err = cudaErrorMemoryAllocation;
        return false;
    }
    else
    {
        // origin： p.err = cudaMallocMaybeCapturing(&ptr, size);
        p.err = cudaMalloc(&ptr, size); // TODO: modify
        if (p.err != cudaSuccess)
        {
            if (p.err == cudaErrorMemoryAllocation)
            {
                // If this is the first attempt (!isRetry), we can forgive and clear
                // CUDA's
                //   internal error state.
                // If this is the second attempt (isRetry), malloc's TORCH_CHECK_WITH
                // will take
                //   over to throw a helpful exception. The user can choose to catch
                //   the exception, free some stuff in their script, and attempt their
                //   allocation again. In this case, we can also forgive and clear
                //   CUDA's internal error state.
                cudaGetLastError();
            }
            else
            {
                // If the error's unrelated to memory allocation, we should throw
                // immediately.
                C10_CUDA_CHECK(p.err);
            }
            // std::cout << "call cudaMalloc() Error. total_allocated_memory =  " << total_allocated_memory << std::endl;
            return false;
        }
    }

    // if (p.pool->owner_PrivatePool) {
    //   // The block is for a CUDA graph's PrivatePool.
    //   p.pool->owner_PrivatePool->cudaMalloc_count++;
    // }

    total_allocated_memory += size;
    p.block = new Block(p.device(), p.stream(), size, p.pool, (char *)ptr);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
        update_stat(stats.segment[stat_type], 1);
        update_stat(stats.reserved_bytes[stat_type], size);
    });
    if (size >= CachingAllocatorConfig::max_split_size())
        update_stat(stats.oversize_segments, 1);

    try_empty_cache();

    // p.block came from new, not cudaMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    return true;
}

void DeviceCachingAllocator::try_empty_cache()
{
    if (Dynamic_Utilization_Threshold == 0 || Dynamic_Diff_Threshold == 0)
    {
        return;
    }

    const ssize_t cur_reserved = stats.reserved_bytes[0].current;
    const ssize_t max_allocated = stats.allocated_bytes[0].peak;

    if (cur_reserved - max_allocated >= Dynamic_Diff_Threshold)
    {
        const ssize_t utilization = ssize_t(100.0 * max_allocated / cur_reserved);
        if (utilization <= Dynamic_Utilization_Threshold)
        {
            this->emptyCache();
        }
    }
}

bool DeviceCachingAllocator::should_split(const Block *block, size_t size)
{
    size_t remaining = block->size - size;
    if (block->pool->is_small)
    {
        return remaining >= kMinBlockSize;
    }
    else
    {
        return (size < CachingAllocatorConfig::max_split_size()) && (remaining > kSmallSize);
    }
}

/** Free one or more oversize blocks to the system allocator.  But only enough
 * **/
/** to satisfy the target size **/
// for  alloc()  emptyCache();
bool DeviceCachingAllocator::release_available_cached_blocks(const AllocParams &p)
{
    if (CachingAllocatorConfig::max_split_size() == std::numeric_limits<size_t>::max())
        return false;
    BlockPool &pool = *p.pool;

    // because of std::unique_ptr, block cannot be trivially copied
    Block key(p.search_key.device, p.search_key.stream, p.search_key.size, p.search_key.pool, p.search_key.ptr);
    key.size =
        (key.size < CachingAllocatorConfig::max_split_size()) ? CachingAllocatorConfig::max_split_size() : key.size;
    auto it = pool.blocks.lower_bound(&key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream())
    {
        // No single block is large enough; free multiple oversize blocks,
        // starting with the largest
        if (it == pool.blocks.begin())
            return false;
        size_t totalReleased = 0;
        --it; // Back up one item.  Now on the largest block for the correct
              // stream
        while ((totalReleased < key.size) && ((*it)->size >= CachingAllocatorConfig::max_split_size()) &&
               ((*it)->stream == p.stream()))
        {
            auto cur = it;
            totalReleased += (*it)->size;
            if (it != pool.blocks.begin())
            {
                --it;
                release_block(*cur);
            }
            else
            {
                release_block(*cur);
                break;
            }
        }
        if (totalReleased < key.size)
            return false;
    }
    else
    {
        release_block(*it);
    }
    return true;
}

bool DeviceCachingAllocator::release_cached_blocks()
{
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    // synchronize_and_free_events();

    // Free all non-split cached blocks to system allocator
    release_blocks(large_blocks);
    release_blocks(small_blocks);

    // for (auto it = graph_pools_freeable.begin();
    //      it != graph_pools_freeable.end();) {
    //   // See notifyCaptureDestroy for the strategy here.
    //   TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
    //   release_blocks(it->second->small_blocks);
    //   release_blocks(it->second->large_blocks);
    //   if (it->second->cudaMalloc_count == 0) {
    //     auto erase_count = graph_pools.erase(it->first);
    //     TORCH_INTERNAL_ASSERT(erase_count == 1);
    //     it = graph_pools_freeable.erase(it);
    //   } else {
    //     ++it;
    //   }
    // }

    return true;
}

/*
 * Do not invoke release_block() without if(!block->prev && !block->next).
 * It will raise a segment error, if release parts of segment.
 */
void DeviceCachingAllocator::release_block(Block *block)
{
    C10_CUDA_CHECK(cudaFree((void *)block->ptr));
    total_allocated_memory -= block->size;
    auto *pool = block->pool;

    // if (pool->owner_PrivatePool) {
    //   // The cudaFreed block belonged to a CUDA graph's PrivatePool.
    //   TORCH_INTERNAL_ASSERT(pool->owner_PrivatePool->cudaMalloc_count > 0);
    //   pool->owner_PrivatePool->cudaMalloc_count--;
    // }

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*pool))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
        update_stat(stats.segment[stat_type], -1);
        update_stat(stats.reserved_bytes[stat_type], -block->size);
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
        update_stat(stats.oversize_segments, -1);

    pool->blocks.erase(block);
    delete block;
}

void DeviceCachingAllocator::release_blocks(BlockPool &pool)
{
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end())
    {
        Block *block = *it;
        ++it;
        if (!block->prev && !block->next)
        {
            release_block(block);
        }
    }
}

/** returns cached blocks to the system allocator **/
void DeviceCachingAllocator::emptyCache()
{
    // std::lock_guard<std::recursive_mutex> lock(mutex);
    release_cached_blocks();
}

std::pair<ssize_t, ssize_t> DeviceCachingAllocator::getStats()
{
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto res = stats;
    reset_peak_stat(stats.reserved_bytes[0]);
    reset_peak_stat(stats.allocated_bytes[0]);
    // std::cout << "current max_split_size = " << (CachingAllocatorConfig::max_split_size() / (1024 * 1024)) << "MB" << std::endl;
    return std::make_pair(res.allocated_bytes[0].peak, res.reserved_bytes[0].peak);
}