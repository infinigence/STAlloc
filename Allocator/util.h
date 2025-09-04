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

#ifndef __UTIL_H__
#define __UTIL_H__

#include <iostream>
#include <stdio.h>
#include <string>

enum LogLevel
{
    DEBUG,   // device-0 debug msg
    WARNING, //
    INFO,    // init msg
    ERROR,   // error msg
};

extern LogLevel LOG_LEVEL;

template <typename FirstType, typename... Args> void log_print(FirstType first, Args... args)
{
    std::cout << first;
    auto func = [](const auto arg) { std::cout << " " << arg; };
    (..., func(args));
}

#define LOG_PRINT(level, ...)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (level >= LOG_LEVEL)                                                                                        \
        {                                                                                                              \
            std::cout << "[STAlloc-" #level "] ";                                                                      \
            log_print(__VA_ARGS__);                                                                                    \
            std::cout << std::endl;                                                                                    \
        }                                                                                                              \
    } while (0)

inline std::string getenv_string(const char *envVar, std::string defaultValue = std::string())
{
    const char *envValue = std::getenv(envVar);
    return envValue ? std::string(envValue) : defaultValue;
}

inline int getenv_int(const char *envVar, int defaultValue = 0)
{
    const char *envValue = std::getenv(envVar);
    int res = defaultValue;
    if (envValue)
    {
        try
        {
            res = std::stoi(envValue);
        }
        catch (...)
        {
            std::string msg =
                "Error converting env var: " + std::string(envVar) + " to int. Value = " + std::string(envValue);
            throw std::runtime_error(msg);
        }
    }
    return res;
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

#endif //__UTIL_H__