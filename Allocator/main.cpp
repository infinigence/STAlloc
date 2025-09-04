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
#include "BestFitAllocator.hpp"
#include "CUDACachingAllocator.h"
#include "StaticAllocator.hpp"
#include "VarAllocator.hpp"
#include "OnlineDynamicManager.hpp"

#include <assert.h>
#include <fstream>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#define alignment512(a) (((a) + 511) & (~511))

namespace py = pybind11;

struct AllocationInfo {
    ssize_t size;
    std::string layer_name;
    int device;
    
    AllocationInfo() : size(0), device(-1) {}
    
    AllocationInfo(ssize_t s, const std::string& l, int d) 
        : size(s), layer_name(l), device(d) {}
};

// global
bool initialized = false;
bool semi_initialized = false;
int eval_state = 0;
int DYNAMIC = 0;
int REUSE_STATIC = 0;
int num_experts = 0;
int expert_cnt = 0;
int RECORD_ALLOC = 0;
std::fstream file;

LogLevel LOG_LEVEL;
std::string PLAN_PATH;
std::string INTERVAL_PATH;
std::unordered_map<std::string, Allocator *> Allocator_map;
std::stack<std::string> model_layers;

std::vector<BestFitAllocator *> BestFitAllocator_vec;
ssize_t moe_layer_id = -1;
ssize_t mlp_layer_id = 0;
ssize_t moe_layer_num = 0;
ssize_t dynamic_layer_id = -1;
ssize_t dynamic_event_cnt = 0;
ssize_t iter0_dynamic_layer_num = 0;
ssize_t iter1_dynamic_layer_num = 0;
bool is_row_layer = false;
static MemoryIntervalManager* interval_manager = nullptr;
OnlineAllocator online_allocator;

std::unordered_set<ssize_t> unreleased_ids;

std::unordered_map<std::string, std::vector<Interval>> layer_intervals;
std::unordered_map<std::string, std::vector<Interval>> intra_layer_intervals;
std::unordered_map<void*, AllocationInfo> allocated_memory;  // Record allocated memory information

void init_reuse_intervals(int device, void *static_global_ptr, ssize_t static_size)
{
    interval_manager = new MemoryIntervalManager(static_global_ptr, static_size);
    std::string plan_file = INTERVAL_PATH + "/dev" + std::to_string(device) + "_cross_interval_spare_addr.txt";
    std::string intra_plan_file = INTERVAL_PATH + "/dev" + std::to_string(device) + "_intra_interval_spare_addr.txt";
    
    std::cout << "device = " << device << " Reading from file: " << plan_file << std::endl;
    std::ifstream file(plan_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + plan_file);
    }
    
    std::string line;
    
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        iss >> iter0_dynamic_layer_num >> iter1_dynamic_layer_num;
        std::cout << "device = " << device << " iter0_dynamic_layer_num = " << iter0_dynamic_layer_num 
                  << ", iter1_dynamic_layer_num = " << iter1_dynamic_layer_num << std::endl;
    } else {
        throw std::runtime_error("Failed to read first line from " + plan_file);
    }
    
    // Read subsequent lines, parse layer names and intervals
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string layer_name;
        iss >> layer_name;
        
        std::vector<Interval> intervals;
        std::string interval_str;
        while (iss >> interval_str) {
            if (interval_str[0] == '(') {
                interval_str = interval_str.substr(1, interval_str.length() - 2);
                std::istringstream interval_iss(interval_str);
                std::string start_str, end_str;
                std::getline(interval_iss, start_str, ',');
                std::getline(interval_iss, end_str);
                
                uint64_t start_offset = std::stoull(start_str);
                uint64_t end_offset = std::stoull(end_str);
                
                // convert relative offset to absolute address
                void* start_ptr = static_cast<char*>(static_global_ptr) + start_offset;
                void* end_ptr = static_cast<char*>(static_global_ptr) + end_offset;
                
                intervals.emplace_back(start_ptr, end_ptr);
            }
        }
        
        layer_intervals[layer_name] = intervals;
    }
    file.close();

    // read intra_layer_intervals
    std::cout << "device = " << device << " Reading from file: " << intra_plan_file << std::endl;
    std::ifstream intra_file(intra_plan_file);
    if (!intra_file.is_open()) {
        throw std::runtime_error("Failed to open file: " + intra_plan_file);
    }
    
    while (std::getline(intra_file, line)) {
        std::istringstream iss(line);
        std::string layer_name;
        iss >> layer_name;
        
        std::vector<Interval> intervals;
        std::string interval_str;
        while (iss >> interval_str) {
            if (interval_str[0] == '(') {
                interval_str = interval_str.substr(1, interval_str.length() - 2);
                std::istringstream interval_iss(interval_str);
                std::string start_str, end_str;
                std::getline(interval_iss, start_str, ',');
                std::getline(interval_iss, end_str);
                
                uint64_t start_offset = std::stoull(start_str);
                uint64_t end_offset = std::stoull(end_str);
                
                // convert relative offset to address
                void* start_ptr = static_cast<char*>(static_global_ptr) + start_offset;
                void* end_ptr = static_cast<char*>(static_global_ptr) + end_offset;
                
                intervals.emplace_back(start_ptr, end_ptr);
            }
        }
        
        intra_layer_intervals[layer_name] = intervals;
    }
    intra_file.close();
    
    std::cout << "device = " << device << " interval_manager init done" << std::endl;
    std::cout << "layer_intervals.size() = " << layer_intervals.size() << std::endl;
    std::cout << "intra_layer_intervals.size() = " << intra_layer_intervals.size() << std::endl;
    interval_manager->print_free_regions();
}

void *init_static_allocator(int device)
{
    std::string plan_file = PLAN_PATH + "/dev_" + std::to_string(device) + ".txt";
    std::fstream fin;
    fin.open(plan_file, std::ios::in);

    if (!fin.is_open()) {
        std::cerr << "Failed to open plan file for device " << device << ": " << plan_file << std::endl;
        return nullptr;
    }

    std::string line;
    std::getline(fin, line);
    std::istringstream iss(line);
    ssize_t unreleased_id;
    while (iss >> unreleased_id)
    {
        unreleased_ids.emplace(unreleased_id);
    }

    ssize_t static_memory_max_allocated;
    ssize_t static_memory_size;
    ssize_t static_iter0_event_num, static_iter1_event_num;
    ssize_t var_memory_size;

    fin >> static_memory_max_allocated >> static_memory_size >> static_iter0_event_num >> static_iter1_event_num >>
        var_memory_size;

    ssize_t total_static_event_num = static_iter0_event_num + static_iter1_event_num;
    std::vector<std::vector<ssize_t>> static_plans(total_static_event_num, std::vector<ssize_t>(2));

    ssize_t offset, size;
    for (int i = 0; i < total_static_event_num; ++i)
    {
        fin >> offset >> size;
        static_plans[i][0] = offset;
        static_plans[i][1] = size;
    }
    fin.close();
    try {
        Allocator_map.emplace("static", new StaticAllocator(device, static_memory_size, static_memory_max_allocated,
                                                        static_iter0_event_num, static_iter1_event_num, static_plans));
    } catch (const std::bad_alloc& e) {
        std::cerr << "static Memory allocation failed for device " << device << ": " << e.what() << std::endl;
        return nullptr;
    }
    Allocator_map.emplace("var", new VarAllocator(device, var_memory_size * 2));
    init_reuse_intervals(device, Allocator_map["static"]->global_ptr, static_memory_size);
    return Allocator_map["static"]->global_ptr;
}

bool init_bestfit(int device, void *static_global_ptr)
{
    std::string plan_file = PLAN_PATH + "/dev_" + std::to_string(device) + "_dynamic.txt";
    std::ifstream file(plan_file);
    if (!file.is_open())
    {
        return false;
    }
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);

    ssize_t static_offset;
    iss >> static_offset >> moe_layer_num;

    static_global_ptr += static_offset;

    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        ssize_t offset, size;
        if (offset == -1)
        {
            return false;
        }
        std::unordered_set<ssize_t> skip_ids;

        iss >> offset >> size;
        ssize_t num;
        while (iss >> num)
        {
            skip_ids.emplace(num);
        }
        BestFitAllocator_vec.emplace_back(new BestFitAllocator(device, size, static_global_ptr + offset, skip_ids));
    }
    return true;
}

bool init(int device)
{
    LOG_LEVEL = LogLevel(getenv_int("STALLOC_LOG_LEVEL", 3));

    auto STALLOC_MODEL_INFO_PATH = getenv_string("STALLOC_MODEL_INFO_PATH");
    PLAN_PATH = STALLOC_MODEL_INFO_PATH + "/output/plan";
    INTERVAL_PATH = STALLOC_MODEL_INFO_PATH + "/output/activations_memlog/dynamic_intervals";

    DYNAMIC = getenv_int("STALLOC_DYNAMIC", 0);
    REUSE_STATIC = getenv_int("STALLOC_REUSE_STATIC", 0);
    num_experts = getenv_int("EXPERT_PER_GPU", 0);
    RECORD_ALLOC = getenv_int("STALLOC_RECORD_ALLOC", 0);

    void *static_global_ptr = init_static_allocator(device);

    if (DYNAMIC)
    {
        semi_initialized = false;
        DeviceCachingAllocator::getInstance()->init_empty_config();
    }
    LOG_PRINT(INFO, "Init done, device =", device, ",Dynamic mode =", DYNAMIC,
              ",semiDynamicAllocator mode =", semi_initialized);
    if (RECORD_ALLOC) {
        auto STALLOC_MODEL_INFO_PATH = getenv_string("STALLOC_MODEL_INFO_PATH");
        std::string mem_log_path = STALLOC_MODEL_INFO_PATH + "/alloc_records/mem_log_dev" + std::to_string(device) + ".txt";
        file.open(mem_log_path.c_str(), std::ios::app);
        if (!file.is_open()) {
            std::cerr << "Failed to open record file for device " << device << ": " << mem_log_path << std::endl;
        }
    }
    return true;
}

const std::string MoE_layer_str = "moe";
const std::string ExpertMLP_layer_str = "mlp";
const std::string SharedMLP_layer_str = "shared";
const std::string Router_layer_str = "router";
const std::string Row_layer_str = "row";
const std::vector<std::string> checkpoint_op_types = {"", MoE_layer_str, ExpertMLP_layer_str, SharedMLP_layer_str, 
                                                      Router_layer_str, Row_layer_str};


std::unordered_map<void *, ssize_t> bestfit_allocated_ptr;

ssize_t global_malloc_idx = -1;

std::string get_string_prefix(const std::string &layer_info)
{
    std::string prefix = layer_info;
    std::string::size_type pos = prefix.find("-");
    if (pos != std::string::npos)
    {
        prefix = prefix.substr(0, pos);
    }
    return prefix;
}

std::string get_phase(const std::string &layer_info)
{
    std::string::size_type first_dash = layer_info.find('-');
    if (first_dash == std::string::npos) {
        return "";
    }
    
    std::string::size_type second_dash = layer_info.find('-', first_dash + 1);
    if (second_dash == std::string::npos) {
        return "";
    }
    
    return layer_info.substr(first_dash + 1, second_dash - first_dash - 1);
}

inline bool is_in_dynamic_layer()
{
    return !model_layers.empty() && get_string_prefix(model_layers.top()) != SharedMLP_layer_str && get_string_prefix(model_layers.top()) != Router_layer_str;
}

extern "C"
{
    ssize_t global_allocated = 0;
    ssize_t global_max_allocated = 0;
    void checkpoint(const char *layer_info, int op_type, bool forward, bool start, int device)
    {
        if (op_type == 0)
        {
            eval_state ^= 1;
            return;
        }
        std::string layer_info_str = std::string(layer_info);

        std::string F_B_W = forward ? "F" : "B";
        if (!model_layers.empty() && get_string_prefix(model_layers.top()) == checkpoint_op_types[op_type] && !start)
        {
            model_layers.pop();
            if (is_row_layer && !start)
            {
                expert_cnt++;
                if (expert_cnt > num_experts)
                {
                    expert_cnt = 1;
                }
            }
            is_row_layer = false;
        }
        else
        {
            if (!start)
            {
                std::cout << "checkpoint error, start = " << start << ", op_type = " << op_type << std::endl;
                throw std::runtime_error("checkpoint error");
            }
            if (op_type == 1)
            {
                if (dynamic_layer_id == iter0_dynamic_layer_num + iter1_dynamic_layer_num - 1)
                {
                    dynamic_layer_id = iter0_dynamic_layer_num;
                }
                else
                {
                    dynamic_layer_id++;
                }
            }
            if (op_type == 5)
            {
                is_row_layer = true;
            }
            model_layers.push(checkpoint_op_types[op_type] + "-" + F_B_W + "-" + std::to_string(dynamic_layer_id));
            dynamic_event_cnt = 0;
        }
    }

    void *my_malloc(ssize_t size, int device, cudaStream_t stream)
    {
        global_allocated += size;
        global_max_allocated = std::max(global_max_allocated, global_allocated);
        void* return_ptr = nullptr;

        if (!initialized)
        {
            initialized = init(device);
            if (!initialized) {
                std::cerr << "Initialization failed for device " << device << std::endl;
                return nullptr;
            }
        }

        if (size <= VAR_BLOCK_SIZE)
        {
            return_ptr = Allocator_map["var"]->malloc(size, device, stream);
        }
        else if (unreleased_ids.count(++global_malloc_idx))
        {
            return_ptr = deviceCachingAllocator_malloc(size, device, stream);
        }
        else if (eval_state)
        {
            return_ptr = online_allocator.allocate_from(interval_manager->get_free_intervals(), size);
            if (return_ptr == nullptr) {
                return_ptr = deviceCachingAllocator_malloc(size, device, stream);
            }
            else {
                bool success = interval_manager->allocate(return_ptr, size);
                if (success) {
                    allocated_memory[return_ptr] = AllocationInfo(size, "eval", device);
                }
                else {
                    throw std::runtime_error("interval_manager allocate error");
                }
            }
        }
        else if (is_in_dynamic_layer())
        {
                            
            if ((get_string_prefix(model_layers.top()) == ExpertMLP_layer_str || is_row_layer) && REUSE_STATIC && (expert_cnt < num_experts))
            {
                if (is_row_layer)
                {
                    // alloc in current layer
                    return_ptr = online_allocator.allocate_from(interval_manager->get_free_intervals(), size);
                }
                else
                {
                    // check reuse
                    // check if the layer is in the layer_intervals
                    if (layer_intervals.count(model_layers.top()) > 0) 
                    {
                        std::vector<Interval> intervals = interval_manager->get_intersection(layer_intervals[model_layers.top()]);
                        return_ptr = online_allocator.allocate_from(intervals, size);
                    } else 
                    {
                        return_ptr = online_allocator.allocate_from(interval_manager->get_free_intervals(), size);
                    }
                }
                if (return_ptr == nullptr)
                {
                    return_ptr = deviceCachingAllocator_malloc(size, device, stream);
                }
                else
                {
                    bool success = interval_manager->allocate(return_ptr, size);
                    if (success) 
                    {
                        allocated_memory[return_ptr] = AllocationInfo(size, model_layers.top(), device);
                    } 
                    else 
                    {
                        // check conflict
                        std::cout << "Memory allocation conflict detected:" << std::endl;
                        std::cout << "Failed to allocate " << size << " bytes at " << return_ptr << std::endl;
                        std::cout << "Current layer: " << model_layers.top() << std::endl;
                        std::cout << "Device: " << device << std::endl;
                        
                        // check overlap in allocation
                        for (const auto& [addr, info] : allocated_memory) {
                            if (addr <= return_ptr && 
                                static_cast<char*>(addr) + info.size > return_ptr) {
                                std::cout << "Conflict with existing allocation:" << std::endl;
                                std::cout << "  Address: " << addr << std::endl;
                                std::cout << "  Size: " << info.size << std::endl;
                                std::cout << "  Layer: " << info.layer_name << std::endl;
                                std::cout << "  Device: " << info.device << std::endl;
                            }
                        }
                        
                        std::cerr << "device = " << device << " reuse static error" << std::endl;
                        throw std::runtime_error("reuse static error");
                    }
                }
            }
            else if (get_string_prefix(model_layers.top()) == MoE_layer_str && get_phase(model_layers.top()) == "B" && REUSE_STATIC)
            {
                // check intra_layer_intervals
                if (intra_layer_intervals.count(model_layers.top()) > 0)
                {
                    std::vector<Interval> intervals = interval_manager->get_intersection(intra_layer_intervals[model_layers.top()]);
                    return_ptr = online_allocator.allocate_from(intervals, size);

                    if (return_ptr == nullptr)
                    {
                        return_ptr = deviceCachingAllocator_malloc(size, device, stream);
                    }
                    else
                    {
                        bool success = interval_manager->allocate(return_ptr, size);
                        if (success)
                        {
                            allocated_memory[return_ptr] = AllocationInfo(size, model_layers.top(), device);
                        }
                        else
                        {
                            // check for conflict
                            for (const auto& [addr, info] : allocated_memory) {
                                if (addr <= return_ptr && 
                                    static_cast<char*>(addr) + info.size > return_ptr) {
                                    std::cout << "Conflict with existing allocation:" << std::endl;
                                    std::cout << "  Address: " << addr << std::endl;
                                    std::cout << "  Size: " << info.size << std::endl;
                                    std::cout << "  Layer: " << info.layer_name << std::endl;
                                    std::cout << "  Device: " << info.device << std::endl;
                                }
                            }

                            throw std::runtime_error("intra_layer_intervals allocate error, current layer = " + model_layers.top());
                        }
                    }
                }
                else
                {
                    throw std::runtime_error("intra_layer_intervals not found, current layer = " + model_layers.top());
                }
            }
            else if (get_string_prefix(model_layers.top()) == MoE_layer_str && false) // semi_initialized
            {
                void *ptr = BestFitAllocator_vec[moe_layer_id]->alloc_fn_(size);
                if (ptr != nullptr)
                {
                    if (bestfit_allocated_ptr.count(ptr) != 0)
                    {
                        LOG_PRINT(ERROR, "bestfit-alloc double alloc: bestfit-id =", moe_layer_id, ", ptr = ", ptr,
                                ",device =", device);
                        throw std::runtime_error("double allocated ptr found");
                    }
                    bestfit_allocated_ptr[ptr] = moe_layer_id;
                    return_ptr = ptr;
                }
                else
                {
                    return_ptr = deviceCachingAllocator_malloc(size, device, stream);
                }
            }
            else
            {
                return_ptr = deviceCachingAllocator_malloc(size, device, stream);
            }
        }
        else
        {
            return_ptr = Allocator_map["static"]->malloc(size, device, stream);

            if (Allocator_map["static"]->is_allcoated(return_ptr))
            {
                bool success = interval_manager->allocate(return_ptr, size);
                if (!success) 
                {
                    return_ptr = deviceCachingAllocator_malloc(size, device, stream);
                }
            }
        }

        if (RECORD_ALLOC) {
            file << "1 " << return_ptr << " " << size << " " << is_in_dynamic_layer() << std::endl;
        }
        return return_ptr;
    }

    void my_free(void *ptr, ssize_t size, int device, cudaStream_t stream)
    {
        global_allocated -= size;
        if (RECORD_ALLOC) {
            file << "0 " << ptr << " " << size << " 0" << std::endl;
        }
        if (size <= VAR_BLOCK_SIZE)
        {
            return Allocator_map["var"]->free(ptr, size, device, stream);
        }

        if (semi_initialized && bestfit_allocated_ptr.count(ptr))
        {
            auto free_moe_id = bestfit_allocated_ptr[ptr];
            bestfit_allocated_ptr.erase(ptr);
            return BestFitAllocator_vec[free_moe_id]->free_fn_(ptr, size);
        }

        if (Allocator_map["static"]->is_allcoated(ptr))
        {
            interval_manager->free(ptr, size);
            if (allocated_memory.count(ptr) > 0) 
            {
                allocated_memory.erase(ptr);
                return;
            }
            return Allocator_map["static"]->free(ptr, size, device, stream);
        }

        return deviceCachingAllocator_free(ptr, size, device, stream);
    }
    
    py::dict MemoryStat()
    {
        py::dict result;

        py::dict static_stat;
        std::pair<int64_t, int64_t> pair_stat = Allocator_map["static"]->getStat();
        static_stat["max_allocated"] = pair_stat.first;
        static_stat["max_reserved"] = pair_stat.second;
        result["static"] = static_stat;
        int64_t max_reserved = pair_stat.second;

        pair_stat = Allocator_map["var"]->getStat();
        max_reserved += pair_stat.second;

        pair_stat = DeviceCachingAllocator::getInstance()->getStats();
        py::dict dynamic_stat;
        dynamic_stat["max_allocated"] = pair_stat.first;
        dynamic_stat["max_reserved"] = pair_stat.second;

        result["dynamic"] = dynamic_stat;
        max_reserved += pair_stat.second;

        py::dict global_stat;
        global_stat["max_allocated"] = global_max_allocated;
        global_stat["max_reserved"] = max_reserved;
        result["all"] = global_stat;

        global_max_allocated = 0;
        return result;
    }
}

PYBIND11_MODULE(alloc, m)
{
    m.doc() = "allocator tools";
    m.def("checkpoint", &checkpoint, "plan checkpoint");
    m.def("MemoryStat", &MemoryStat, "memory usage statitics");
}