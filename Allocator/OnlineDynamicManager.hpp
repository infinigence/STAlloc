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

#ifndef ONLINE_DYNAMIC_MANAGER_HPP
#define ONLINE_DYNAMIC_MANAGER_HPP

#include <iostream>
#include <set>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cstddef>
#include <limits>
#define alignment512(a) (((a) + 511) & (~511))

using Address = void*;

struct Interval {
    Address start;
    Address end;
    Interval(Address s, Address e) : start(s), end(e) {
        assert(s <= e);
    }

    bool operator<(const Interval& other) const {
        return end <= other.start;
    }

    bool overlaps(const Interval& other) const {
        return start < other.end && end > other.start;
    }

    Interval intersect(const Interval& other) const {
        return {
            std::max(start, other.start),
            std::min(end, other.end)
        };
    }

    ssize_t size() const {
        return static_cast<char*>(end) - static_cast<char*>(start);
    }
};

class MemoryIntervalManager {
public:
    
    // Init address range
    MemoryIntervalManager(Address start, ssize_t size)
        : managed_start(start),
          managed_end(static_cast<char*>(start) + alignment512(size))
    {
        assert(size > 0);
        free_intervals.insert({managed_start, managed_end});
    }

    bool check_overlap(void* addr, size_t size) {
        Interval requested(addr, static_cast<char*>(addr) + size);
        
        // Find the first interval with start address greater than requested.start
        auto it = free_intervals.upper_bound(requested);
        
        // If an interval is found, check if the previous interval completely contains requested
        if (it != free_intervals.begin()) {
            auto prev_it = std::prev(it);
            // If the previous interval's start address is less than or equal to requested.start, 
            // and end address is greater than or equal to requested.end
            if (prev_it->start <= requested.start && prev_it->end >= requested.end) {
                return false;  // Completely contained, return false
            }
        }
        
        // If no suitable interval completely contains requested, return true
        return true;
    }

    // try to allocate memory in the given range
    bool allocate(Address start, ssize_t size) {
        size = alignment512(size);
        if (!within_bounds(start, size)) {
            std::cout << "allocate out of bounds, start = " << start << " size = " << size << std::endl;
            return false;
        }

        Interval requested{start, static_cast<char*>(start) + size};
        
        // Allocate while searching for suitable intervals
        for (auto it = free_intervals.begin(); it != free_intervals.end(); ++it) {
            // Check if completely contained within free interval
            if (it->start <= requested.start && it->end >= requested.end) {
                // Calculate split intervals
                Interval left{it->start, requested.start};
                Interval right{requested.end, it->end};
                
                // Remove current interval
                it = free_intervals.erase(it);
                
                // Add split intervals (if valid)
                if (left.start < left.end) {
                    it = free_intervals.insert(it, left);
                    ++it;
                }
                if (right.start < right.end) {
                    free_intervals.insert(it, right);
                }
                return true;
            }
        }

        // std::cout << "Error: interval manager allocate failed, start = " << start 
        //           << " size = " << size << std::endl;
        // print_free_regions();
        return false;
    }

    // free memory in the given range and merge adjacent free intervals
    void free(Address start, ssize_t size) {
        size = alignment512(size);
        if (!within_bounds(start, size)) {
            std::cout << "free out of bounds, start = " << start << " size = " << size << std::endl;
            return;
        }

        Interval new_interval{start, static_cast<char*>(start) + size};
        
        // Use lower_bound to find the first interval that might overlap
        auto it = free_intervals.lower_bound(new_interval);
        
        // Check if the previous interval overlaps
        if (it != free_intervals.begin()) {
            auto prev = std::prev(it);
            if (prev->end > new_interval.start) {
                std::cout << "Error: trying to free an interval that overlaps with free intervals" << std::endl;
                std::cout << "To free: [" << new_interval.start << ", " << new_interval.end << ")" << std::endl;
                std::cout << "Overlaps with: [" << prev->start << ", " << prev->end << ")" << std::endl;
                return;
            }
        }

        // Check if subsequent intervals overlap
        if (it != free_intervals.end() && it->start < new_interval.end) {
            std::cout << "Error: trying to free an interval that overlaps with free intervals" << std::endl;
            std::cout << "To free: [" << new_interval.start << ", " << new_interval.end << ")" << std::endl;
            std::cout << "Overlaps with: [" << it->start << ", " << it->end << ")" << std::endl;
            return;
        }

        // Perform merge operation
        // 1. Merge subsequent intervals
        while (it != free_intervals.end() && new_interval.end >= it->start) {
            new_interval.end = std::max(new_interval.end, it->end);
            it = free_intervals.erase(it);
        }

        // 2. Merge previous intervals
        if (it != free_intervals.begin()) {
            auto prev = std::prev(it);
            if (prev->end >= new_interval.start) {
                new_interval.start = std::min(new_interval.start, prev->start);
                free_intervals.erase(prev);
            }
        }

        // 3. Insert merged interval
        free_intervals.insert(it, new_interval);
    }

    // get the free intervals that intersect with the given intervals
    std::vector<Interval> get_intersection(const std::vector<Interval>& intervals) const {
        std::vector<Interval> result;
        for (const auto& query : intervals) {
            for (const auto& free : free_intervals) {
                if (free.overlaps(query)) {
                    auto isec = free.intersect(query);
                    if (isec.start < isec.end) {
                        result.push_back(isec);
                    }
                }
            }
        }
        return result;
    }

    std::vector<Interval> get_free_intervals() const {
        return std::vector<Interval>(free_intervals.begin(), free_intervals.end());
    }

    void print_free_regions() const {
        std::cout << "Free intervals:\n";
        for (const auto& interval : free_intervals) {
            std::cout << "[" << interval.start << ", " << interval.end << ")" << " size = " << interval.size() << "\n";
        }
    }

private:
    Address managed_start;
    Address managed_end;
    std::set<Interval> free_intervals;

    bool within_bounds(Address addr, ssize_t size) const {
        char* a = static_cast<char*>(addr);
        char* b = a + size;
        return a >= managed_start && b <= managed_end && a < b;
    }
};

class OnlineAllocator {
public:
    enum class Strategy {
        BestFit,
        WorstFit
    };

    static void* allocate_from(
        const std::vector<Interval>& available_intervals,
        ssize_t size,
        Strategy strategy = Strategy::BestFit)
    {
        void* result = nullptr;
        size = alignment512(size);
        
        switch (strategy) {
            case Strategy::BestFit: {
                ssize_t best_size = std::numeric_limits<ssize_t>::max();
                for (const auto& interval : available_intervals) {
                    ssize_t free_size = interval.size();
                    if (free_size >= size && (result == nullptr || free_size < best_size)) {
                        best_size = free_size;
                        result = interval.start;
                    }
                }
                break;
            }

            case Strategy::WorstFit: {
                ssize_t worst_size = 0;
                for (const auto& interval : available_intervals) {
                    ssize_t free_size = interval.size();
                    if (free_size >= size && (result == nullptr || free_size > worst_size)) {
                        worst_size = free_size;
                        result = interval.start;
                    }
                }
                break;
            }
        }

        return result;
    }
};

#endif // ONLINE_DYNAMIC_MANAGER_HPP