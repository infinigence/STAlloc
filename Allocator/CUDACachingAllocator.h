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
#ifndef THC_DEVICE_ALLOCATOR_INC
#define THC_DEVICE_ALLOCATOR_INC

#include "Hardware_Allocator.h"

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cstdio>
#include <deque>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <vector>

#include "flat_hash_map.h"
#include "llvmMathExtras.h"

using namespace std;
using namespace c10;
// original code: using stream_set = ska::flat_hash_set<cuda::CUDAStream>;
// flat_hash_set -> set:
// using stream_set = set<cudaStream_t>;
using stream_set = ska::flat_hash_set<cudaStream_t>;

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define C10_LIKELY(expr) (expr)
#define C10_UNLIKELY(expr) (expr)
#endif

template <typename T> void toStringStream(std::stringstream &ss, T value)
{
    ss << value;
}

template <typename T, typename... Args> void toStringStream(std::stringstream &ss, T first, Args... args)
{
    ss << first;
    toStringStream(ss, args...);
}

template <typename... Args> std::string concatenate(Args... args)
{
    std::stringstream ss;
    toStringStream(ss, args...);
    return ss.str();
}

#define C10_CUDA_CHECK(EXPR)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t __err = EXPR;                                                                                      \
        if (__err != cudaSuccess)                                                                                      \
        {                                                                                                              \
            fprintf(stderr, "CUDA ERROR: (error code %s)!\n", cudaGetErrorString(__err));                              \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// Simplified torch_check：
#define TORCH_CHECK(cond, ...)                                                                                         \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
        printf("error info:%s", ##__VA_ARGS__);                                                                        \
        exit(EXIT_FAILURE);                                                                                            \
    }

#define TORCH_INTERNAL_ASSERT(...) TORCH_CHECK(__VA_ARGS__)

#define TORCH_CHECK_WITH(cond, ...)                                                                                    \
    if (!(cond))                                                                                                       \
    {                                                                                                                  \
        cout << concatenate(__VA_ARGS__) << endl;                                                                      \
        exit(EXIT_FAILURE);                                                                                            \
    }

constexpr size_t kMinBlockSize = 512;       // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576;      // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer = 2097152;    // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer = 20971520;   // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc = 10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
// constexpr size_t kRoundLarge = 2097152;     // round up large allocations to 2 MiB
constexpr size_t kRoundLarge = 512;     // round up large allocations to 2 MiB

struct Stat
{
    int64_t current = 0;
    int64_t peak = 0;
    int64_t allocated = 0;
    int64_t freed = 0;
};

enum struct StatType : uint64_t
{
    AGGREGATE = 0,
    SMALL_POOL = 1,
    LARGE_POOL = 2,
    NUM_TYPES = 3 // remember to update this whenever a new stat type is added
};

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

typedef std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)> StatArray;

// Struct containing memory allocator summary statistics for a device.
struct DeviceStats
{
    // COUNT: allocations requested by client code
    StatArray allocation;
    // COUNT: number of allocated segments from cudaMalloc().
    StatArray segment;
    // COUNT: number of active memory blocks (allocated or used by stream)
    StatArray active;
    // COUNT: number of inactive, split memory blocks (unallocated but can't be
    // released via cudaFree)
    StatArray inactive_split;

    // SUM: bytes requested by client code
    StatArray allocated_bytes;
    // SUM: bytes reserved by this memory allocator (both free and used)
    StatArray reserved_bytes;
    // SUM: bytes within active memory blocks
    StatArray active_bytes;
    // SUM: bytes within inactive, split memory blocks
    StatArray inactive_split_bytes;

    // COUNT: total number of failed calls to CUDA malloc necessitating cache
    // flushes.
    int64_t num_alloc_retries = 0;

    // COUNT: total number of OOMs (i.e. failed calls to CUDA after cache flush)
    int64_t num_ooms = 0;

    // COUNT: total number of oversize blocks allocated from pool
    Stat oversize_allocations;

    // COUNT: total number of oversize blocks requiring malloc
    Stat oversize_segments;

    // SIZE: maximum block size that is allowed to be split.
    int64_t max_split_size = 0;
};

struct Context
{
    virtual ~Context()
    {
    }
};

typedef std::unique_ptr<Context> (*CreateContextFn)(void);

struct History
{
    void *addr;
    size_t real_size;                 // unrounded, actually requested size
    std::unique_ptr<Context> context; // per-watcher context
    std::unique_ptr<History> next;    // when blocks are merged we keep records of
                                      // what used to be in the block
};

// Struct containing info of an allocation block (i.e. a fractional part of a
// cudaMalloc)..
struct BlockInfo
{
    int64_t size = 0;
    int32_t gc_counter = 0;
    bool allocated = false;
    bool active = false;
    History *history = nullptr; // borrowed reference because it is owned by the allocator
};

// Struct containing info of a memory segment (i.e. one contiguous cudaMalloc).
struct SegmentInfo
{
    int64_t device = 0;
    int64_t address = 0;
    int64_t total_size = 0;
    int64_t allocated_size = 0;
    int64_t active_size = 0;
    cudaStream_t stream = 0;
    bool is_large = false;
    std::vector<BlockInfo> blocks;
};

struct Block;
struct PrivatePool; // CUDA graphs helper
typedef bool (*Comparison)(const Block *, const Block *);

struct BlockPool
{
    BlockPool(Comparison comparator, bool small, PrivatePool *private_pool = nullptr)
        : blocks(comparator), is_small(small), owner_PrivatePool(private_pool)
    {
    }
    std::set<Block *, Comparison> blocks;
    const bool is_small;
    PrivatePool *owner_PrivatePool;
};

struct Block
{
    int device;             // gpu
    cudaStream_t stream;    // allocation stream
    stream_set stream_uses; // streams on which the block was used
    size_t size;            // block size in bytes
    BlockPool *pool;        // owning memory pool
    void *ptr;              // memory address
    bool allocated;         // in-use flag
    Block *prev;            // prev block if split from a larger allocation
    Block *next;            // next block if split from a larger allocation
    int event_count;        // number of outstanding CUDA events
    int gc_count;           // counter for prioritizing older / less useful blocks for
                            // garbage collection
    std::unique_ptr<History> history;
    History *history_last;

    Block(int device, cudaStream_t stream, size_t size, BlockPool *pool, void *ptr)
        : device(device), stream(stream), stream_uses(), size(size), pool(pool), ptr(ptr), allocated(0), prev(nullptr),
          next(nullptr), event_count(0), gc_count(0)
    {
    }

    // constructor for search key
    Block(int device, cudaStream_t stream, size_t size)
        : device(device), stream(stream), stream_uses(), size(size), pool(nullptr), ptr(nullptr), allocated(0),
          prev(nullptr), next(nullptr), event_count(0), gc_count(0)
    {
    }

    bool is_split() const
    {
        return (prev != nullptr) || (next != nullptr);
    }
};

static bool BlockComparator(const Block *a, const Block *b)
{
    if (a->stream != b->stream)
    {
        return (uintptr_t)a->stream < (uintptr_t)b->stream;
    }
    if (a->size != b->size)
    {
        return a->size < b->size;
    }
    return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

struct AllocParams
{
    // for set range.
    Block search_key;
    BlockPool *pool;
    size_t alloc_size;
    Block *block;
    StatTypes stat_types = {false};
    cudaError_t err;

    AllocParams(int device, size_t size, cudaStream_t stream, BlockPool *pool, size_t alloc_size, DeviceStats &stats)
        : search_key(device, stream, size), pool(pool), alloc_size(alloc_size), block(nullptr), err(cudaSuccess)
    {
    }

    int device() const
    {
        return search_key.device;
    }
    cudaStream_t stream() const
    {
        return search_key.stream;
    }
    size_t size() const
    {
        return search_key.size;
    }
};

static std::string format_size(uint64_t size);

/* Add some tests */
void testDeviceCachingAllocator();
void testDeviceCachingAllocatorE2E();
void testDeviceCachingAllocatorSmallManagement();
void testDeviceCachingAllocatorFragment();

class CachingAllocatorConfig
{
    // private -> public for data observing.
  public:
    static size_t max_split_size()
    {
        return instance().m_max_split_size;
    }
    static double garbage_collection_threshold()
    {
        return instance().m_garbage_collection_threshold;
    }

    // This is used to round-up allocation size to nearest power of 2 divisions.
    // More description below in function roundup_power2_next_division
    // As ane example, if we want 4 divisions between 2's power, this can be done
    // using env variable: PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4
    static size_t roundup_power2_divisions()
    {
        return instance().m_roundup_power2_divisions;
    }
    static size_t roundup_bypass_threshold()
    {
        return instance().m_roundup_bypass_threshold;
    }

    static CachingAllocatorConfig &instance()
    {
        static CachingAllocatorConfig *s_instance = ([]() {
            auto inst = new CachingAllocatorConfig();
            const char *env = getenv("PYTORCH_CUDA_ALLOC_CONF");
            inst->parseArgs(env);
            return inst;
        })();
        return *s_instance;
    }

    void parseArgs(const char *env)
    {
        // If empty, set the default values
        // m_max_split_size = std::numeric_limits<size_t>::max();
        m_max_split_size = 128 * 1024 * 1024;
        m_roundup_power2_divisions = 0;
        m_roundup_bypass_threshold = std::numeric_limits<size_t>::max();
        m_garbage_collection_threshold = 0;

        if (env == nullptr)
        {
            return;
        }

        const std::string config(env);

        std::regex exp("[\\s,]+");
        std::sregex_token_iterator it(config.begin(), config.end(), exp, -1);
        std::sregex_token_iterator end;
        std::vector<std::string> options(it, end);

        for (auto option : options)
        {
            std::regex exp2("[:]+");
            std::sregex_token_iterator it2(option.begin(), option.end(), exp2, -1);
            std::sregex_token_iterator end2;
            std::vector<std::string> kv(it2, end2);
            if (kv.size() >= 2)
            {
                /* Maximum split size in MB.  Limited to large size blocks */
                if (kv[0].compare("max_split_size_mb") == 0)
                {
                    size_t val2 = stoi(kv[1]);
                    TORCH_CHECK(val2 > kLargeBuffer / (1024 * 1024),
                                "CachingAllocator option max_split_size_mb too small, must be > ",
                                kLargeBuffer / (1024 * 1024), "");
                    val2 = std::max(val2, kLargeBuffer / (1024 * 1024));
                    val2 = std::min(val2, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
                    m_max_split_size = val2 * 1024 * 1024;
                }
                else if (kv[0].compare("roundup_power2_divisions") == 0)
                {
                    size_t val2 = stoi(kv[1]);
                    TORCH_CHECK(llvm::isPowerOf2_64(val2), "For roundups, the divisons has to be power of 2 ", "");
                    m_roundup_power2_divisions = val2;
                }
                else if (kv[0].compare("roundup_bypass_threshold_mb") == 0)
                {
                    size_t val2 = stoi(kv[1]);
                    m_roundup_bypass_threshold = val2 * 1024 * 1024;
                }
                else if (kv[0].compare("garbage_collection_threshold") == 0)
                {
                    /*
                     * Perform garbage collection of GPU memory blocks to avoid
                     * triggering expensive sync-and-reclaim-all operation. Upon setting
                     * the threshold (e.g., 0.8), the allocator will start reclaiming
                     * blocks if GPU memory capacity usage exceeds the threshold (i.e.,
                     * 80% of total memory).
                     * Values 0.0 and 1.0 are not allowed as they are less meaningful.
                     */
                    double val2 = stod(kv[1]);
                    TORCH_CHECK(val2 > 0, "garbage_collect_threshold too small, set it 0.0~1.0", "");
                    TORCH_CHECK(val2 < 1.0, "garbage_collect_threshold too big, set it 0.0~1.0", "");
                    m_garbage_collection_threshold = val2;
                }
                else
                {
                    TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", kv[0]);
                }
            }
        }
    }

  private:
    CachingAllocatorConfig()
        : m_max_split_size(std::numeric_limits<size_t>::max()), m_roundup_power2_divisions(0),
          m_garbage_collection_threshold(0)
    {
    }
    std::atomic<size_t> m_max_split_size;
    std::atomic<size_t> m_roundup_power2_divisions;
    std::atomic<size_t> m_roundup_bypass_threshold;
    std::atomic<double> m_garbage_collection_threshold;
};

class DeviceCachingAllocator
{
  private:
    DeviceCachingAllocator();
    DeviceCachingAllocator(const DeviceCachingAllocator &) = delete;
    DeviceCachingAllocator &operator=(const DeviceCachingAllocator &) = delete;
    static std::shared_ptr<DeviceCachingAllocator> instance;
    // static std::once_flag deviceCachingAllocatorSingletonFlag;

    // private -> public for data observing.
  public:
    // lock around all operations
    mutable std::recursive_mutex mutex;

    // device statistics
    DeviceStats stats;

    // pool for unused block。
    // unallocated cached blocks larger than 1 MB
    BlockPool large_blocks;

    // unallocated cached blocks 1 MB or smaller
    BlockPool small_blocks;

    // allocated or in use by a stream. Holds all active allocations,
    // whether they came from graph_pools or one of the BlockPools above.
    // set<Block *> active_blocks;
    ska::flat_hash_set<Block *> active_blocks;

    // captures_underway tracks if a capture might be underway on any stream.
    // Most of the time it's zero, in which case malloc can avoid calling
    // cudaStreamGetCaptureInfo in the hot path.
    int captures_underway = 0;

    // record used memory.
    size_t total_allocated_memory = 0;

    size_t allowed_memory_maximum = 0;

    bool set_fraction = false;

  public:
    // This function takes the size and number of divisions argument and rounds
    // up the size argument for the nearest power-of-2 division.
    // For example, if we need to round-up 1200 and number of divisions is 4,
    // the size 1200 lies between 1024 and 2048 and if we do 4 divisions between
    // them, the values are 1024, 1280, 1536, and 1792. So the function will
    // return 1280 as the nearest ceiling of power-2 divison.

    static size_t roundup_power2_next_division(size_t size, size_t divisions)
    {
        // C10_UNLIKELY(size <= 4 || divisions <= 1)
        if (size <= 4 || divisions <= 1)
        {
            return size;
        }
        if (llvm::isPowerOf2_64(size))
        {
            return size;
        }

        // divide the space between these 2's power into equal divisions
        // If division is zero, return the power-of-2 ceiling.
        size_t power2_floor = llvm::PowerOf2Floor(size);
        size_t power2_divison = power2_floor >> (63 - llvm::countLeadingZeros(divisions));
        if (power2_divison == 0)
        {
            return (power2_floor << 1);
        }
        size_t round_size_floor = size & (~(power2_divison - 1));
        return (round_size_floor == size) ? size : round_size_floor + power2_divison;
    }

    static size_t round_size(size_t size)
    {
        if (size < kMinBlockSize)
        {
            return kMinBlockSize;
        }
        else if (size > CachingAllocatorConfig::roundup_bypass_threshold())
        {
            return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
        }
        else
        {
            auto divisions = CachingAllocatorConfig::roundup_power2_divisions();
            if (divisions > 0 && size > (kMinBlockSize * divisions))
            {
                return roundup_power2_next_division(size, divisions);
            }
            else
            {
                // 512 * (( size + 511 ) / 512)
                return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
            }
        }
    }

    BlockPool &get_pool(size_t size, cudaStream_t stream);

    static size_t get_allocation_size(size_t size)
    {
        if (size <= kSmallSize)
        {
            return kSmallBuffer;
        }
        else if (size < kMinLargeAlloc)
        {
            return kLargeBuffer;
        }
        else
        {
            return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
        }
    }

    StatType get_stat_type_for_pool(const BlockPool &pool);

    // search block in pools -> found best block if it has  -> create a new block if it hasn't.
    Block *malloc(int device, size_t orig_size, cudaStream_t stream);
    // Dose not invoke cudaFree。
    void free(Block *block);

    void free_block(Block *block);

    /** combine previously split blocks. returns the size of the subsumed block,
     * or 0 on failure. */
    size_t try_merge_blocks(Block *dst, Block *src, BlockPool &pool);

    bool get_free_block(AllocParams &p);

    // only one invoking
    bool trigger_free_memory_callbacks(AllocParams &p);

    void garbage_collect_cached_blocks();

    bool alloc_block(AllocParams &p, bool isRetry);

    bool should_split(const Block *block, size_t size);
    /** Free one or more oversize blocks to the system allocator.  But only enough
     * **/
    /** to satisfy the target size **/
    // for  alloc()  emptyCache();
    bool release_available_cached_blocks(const AllocParams &p);
    bool release_cached_blocks();

    /*
     * Do not invoke release_block() without if(!block->prev && !block->next).
     * It will raise a segment error, if release parts of segment.
     */
    void release_block(Block *block);

    void release_blocks(BlockPool &pool);

    /** returns cached blocks to the system allocator **/
    void emptyCache();

    std::pair<ssize_t, ssize_t> getStats();

    static std::shared_ptr<DeviceCachingAllocator> getInstance()
    {
        static std::once_flag deviceCachingAllocatorSingletonFlag;
        static std::shared_ptr<DeviceCachingAllocator> instance = nullptr;
        std::call_once(deviceCachingAllocatorSingletonFlag,
                       [&] { instance = std::shared_ptr<DeviceCachingAllocator>(new DeviceCachingAllocator()); });
        return instance;
    }

    ssize_t Dynamic_Utilization_Threshold;
    ssize_t Dynamic_Diff_Threshold;
    void try_empty_cache();
    void init_empty_config(int utilization_threshold = 0, int diff_threshold = 1);
};

// static std::shared_ptr<DeviceCachingAllocator> deviceCachingAllocator = nullptr;
// static std::once_flag deviceCachingAllocatorSingletonFlag;

#endif
