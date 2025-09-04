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

#include "Allocator.hpp"
#include "CUDACachingAllocator.h"
#include "Hardware_Allocator.h"
#include "util.h"

#include <cassert>
#include <fstream>
#include <string>

std::fstream file;
bool initialized = false;
LogLevel LOG_LEVEL;
int FAST_MODE;
bool init(int device)
{
    LOG_LEVEL = LogLevel(getenv_int("STALLOC_LOG_LEVEL", 3));
    FAST_MODE = getenv_int("STALLOC_TRACE_FAST_MODE", 0);
    auto STALLOC_MODEL_INFO_PATH = getenv_string("STALLOC_MODEL_INFO_PATH");
    std::string mem_log_path = STALLOC_MODEL_INFO_PATH + "/trace/mem_log_dev" + std::to_string(device) + ".txt";
    file.open(mem_log_path.c_str(), std::ios::app);
    if (!file.good())
    {
        LOG_PRINT(ERROR, "can't open memlog file, device =", device);
        std::abort();
    }
    if (FAST_MODE)
    {   
        // In Fast Trace mode, execute empty_cache when fragmentation exceeds 1024MB and utilization is below 90%
        DeviceCachingAllocator::getInstance()->init_empty_config(90, 1024);
    }
    return true;
}

extern "C"
{
    void *my_malloc(ssize_t size, int device, cudaStream_t stream)
    {
        if (!initialized)
        {
            initialized = init(device);
        }
        void *ptr;
        if (FAST_MODE)
        {
            ptr = deviceCachingAllocator_malloc(size, device, stream);
        }
        else
        {
            C10_CUDA_CHECK(cudaMalloc(&ptr, size));
        }
        file << "1 " << ptr << " " << size << " " << stream << std::endl;
        return ptr;
    }

    void my_free(void *ptr, ssize_t size, int device, cudaStream_t stream)
    {
        file << "0 " << ptr << " " << size << " " << stream << std::endl;
        if (FAST_MODE)
        {
            return deviceCachingAllocator_free(ptr, size, device, stream);
        }
        C10_CUDA_CHECK(cudaFree(ptr));
    }
}
