# Copyright 2025 Infinigence AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import torch
import threading
from enum import Enum


class STALLOC_OP_TYPE(Enum):
    Eval = 0
    MoE = 1
    MLP = 2
    SharedMLP = 3
    Router = 4
    Row = 5

class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class _STAllocConfig(metaclass=SingletonMeta):
    def __init__(self):
        self.mode = os.getenv("STALLOC_MODE", "Torch")
        self.dynamic = True if os.getenv("STALLOC_DYNAMIC", False) == "1" else False

        STALLOC_LIB_PATH = os.getenv("STALLOC_LIB_PATH", "")
        STALLOC_MODEL_INFO_PATH = os.getenv("STALLOC_MODEL_INFO_PATH", None)

        if self.mode == "Torch":
            print("[STAlloc-INFO] STAllocator is not uesd")
            return
        elif self.mode == "Trace":
            print("[STAlloc-INFO] STAllocator in trace mode")
            self.alloc_path = os.path.join(STALLOC_LIB_PATH, "trace.so")
        elif self.mode == "Alloc":
            print("[STAlloc-INFO] STAllocator in alloc mode")
            self.alloc_path = os.path.join(STALLOC_LIB_PATH, "alloc.so")
        else:
            raise ValueError(f"[STAlloc-ERROR] Unknown STALLOC_MODE={self.mode}")

        assert os.path.exists(
            self.alloc_path
        ), f"[STAlloc-ERROR] {self.alloc_path} not found"

        new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
            self.alloc_path, "my_malloc", "my_free"
        )
        torch.cuda.memory.change_current_allocator(new_alloc)

        assert os.path.exists(
            STALLOC_MODEL_INFO_PATH
        ), f"[STAlloc-ERROR] STALLOC_MODEL_INFO_PATH={STALLOC_MODEL_INFO_PATH} not found"

        trace_file_dir = os.path.join(STALLOC_MODEL_INFO_PATH, "trace")

        def _trace_line(line):
            trace_file = os.path.join(
                trace_file_dir, f"mem_log_dev{torch.cuda.current_device()}.txt"
            )
            with open(trace_file, "a") as f:
                f.write(line)

        self.trace_line = _trace_line

        def foo(*args, **kwargs):
            pass

        self.checkpoint = foo
        self.get_memory_stat = foo

        if self.mode == "Alloc":
            sys.path.append(STALLOC_LIB_PATH)
            import alloc

            def memory_stat(curr_iter: int):
                stat = alloc.MemoryStat()
                _GB = 1024**3
                memory_info = (
                    f"[STAlloc-Memory Info]: device = {torch.cuda.current_device()}\n"
                )

                for key, val in stat.items():
                    if key == "all":
                        continue
                    max_reserved = val["max_reserved"]
                    max_allocated = val["max_allocated"]
                    memory_info += f"{key} : max_reserved:{max_reserved/_GB:.2f}, max_allocated:{max_allocated/_GB:.2f}, utilization:{(max_allocated/max_reserved):.2%}\n"

                max_reserved = stat["all"]["max_reserved"]
                max_allocated = stat["all"]["max_allocated"]
                memory_info += f"dev{torch.cuda.current_device()}, iter{curr_iter} : max_reserved:{max_reserved/_GB:.2f}, max_allocated:{max_allocated/_GB:.2f}, utilization:{(max_allocated/max_reserved):.2%}\n"
                print(memory_info, flush=True)

            self.get_memory_stat = memory_stat

            def _checkpoint(
                name: str, op_type: STALLOC_OP_TYPE, forward: bool, start: bool
            ) -> None:
                alloc.checkpoint(
                    name, op_type.value, forward, start, torch.cuda.current_device()
                )

            self.checkpoint = _checkpoint

STAllocConfig = _STAllocConfig()

from .memory_patcher import apply_memory_patch
apply_memory_patch()
