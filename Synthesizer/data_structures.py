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

from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple
from math import ceil
import os


_KB: int = 1024
_MB: int = 1024**2
_GB: int = 1024**3
VAR_SIZE: int = 512
ALIGN: int = 512


def format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    elif size < _MB:
        return f"{int(size / 1024):.2f}KB"
    elif size < 0.2 * _GB:
        return f"{(size / _MB):.2f}MB"
    return f"{(size / _GB):.2f}GB"


def roundup(size: int, align: int = ALIGN) -> int:
    return ceil(size / align) * align


class MemType(Enum):
    WEIGHT_GRAD = 0
    OPTIMIZER_STATES = 1
    INIT_PHASE = 2
    CROSS_ITER = 3
    ACTIVATIONS_GROUP = 4
    ACTIVATIONS_TEMPORARY = 5
    ACTIVATIONS = 6
    UNRELEASED = 7
    DYNAMIC = 8


@dataclass
class MemLog:
    status: int
    size: int
    time: int
    iteration: str
    phase: str
    malloc_id: int = -1
    layer: str = ""

    def set_malloc_id(self, malloc_id):
        self.malloc_id = malloc_id

    def set_layer(self, layer):
        self.layer = layer

    def print_log(self):
        return f"status:{self.status}, size:{self.size}, time:{self.time}, iter:{self.iteration}, phase:{self.phase}"


@dataclass
class Tensor:
    size: int
    start_time: int
    end_time: int
    malloc_id: int
    start_iteration: str
    end_iteration: str
    start_phase: str
    end_phase: str
    start_layer: str
    end_layer: str

    mem_type: MemType = None
    offset: int = -1

    def __str__(self):
        return f"size:{format_size(self.size)}, start:[{self.start_time}->{self.start_iteration}->{self.start_phase}->{self.start_layer}], end:[{self.end_time}->{self.end_iteration}->{self.end_phase}->{self.end_layer}], malloc_id:{self.malloc_id}"

    def is_cross_iteration(self):
        return self.start_iteration != self.end_iteration

    def is_cross_phase(self):
        return self.start_phase != self.end_phase

    def is_var(self):
        return self.size <= VAR_SIZE

    def is_dynamic_moe(self):
        return "moe" in self.start_layer

    def is_dynamic_mlp(self):
        return "mlp" in self.start_layer

    def is_dynamic(self):
        return self.is_dynamic_moe() or self.is_dynamic_mlp()

    def set_malloc_id(self, malloc_id):
        self.malloc_id = malloc_id

    def set_memtype(self, mem_type: MemType):
        self.mem_type = mem_type

    def set_offset(self, offset):
        self.offset = offset


class GroupTensor(Tensor):

    def __init__(self, tensors):
        self.tensors = tensors
        self.tensors.sort(key=lambda x: x.start_time)
        self.group_total_size = get_tensors_align_size(tensors)
        self.group_max_usage = max(get_memory_usage(tensors))
        self.group_tensors_num = len(tensors)

        self.start_time = tensors[0].start_time
        self.start_iteraion = tensors[0].start_iteration

        self.end_iteration = self.start_iteraion
        self.end_time = self.start_time
        for tensor in tensors:
            if tensor.end_time > self.end_time:
                self.end_time = tensor.end_time
                self.end_iteration = tensor.end_iteration
        super().__init__(
            self.group_max_usage,
            self.start_time,
            self.end_time,
            tensors[0].malloc_id,
            self.start_iteraion,
            self.end_iteration,
            tensors[0].start_phase,
            "",
            tensors[0].start_layer,
            "",
        )

    def __str__(self):
        return f"start:[{self.start_iteraion}->{self.start_phase}], end:[{self.end_iteration}->{self.end_phase}], group_max_suage:{format_size(self.group_max_usage)}, group_total_size:{format_size(self.group_total_size)}, group_tensors_num:{len(self.tensors)}"


def get_tensors_align_size(tensors: List[Tensor]) -> int:
    usage = 0
    for tensor in tensors:
        usage = roundup(usage + tensor.size)
    return usage


def get_memory_usage(tensors: List[Tensor]) -> List[int]:
    min_time = tensors[0].start_time
    max_time = max([tensor.end_time for tensor in tensors]) + 1
    lens = max_time - min_time + 1
    memory_usage = [0 for i in range(lens)]
    for tensor in tensors:
        tensor_size = roundup(tensor.size)
        memory_usage[tensor.start_time - min_time] += tensor_size
        memory_usage[tensor.end_time - min_time] -= tensor_size

    for i in range(1, lens):
        memory_usage[i] += memory_usage[i - 1]
    return memory_usage


def get_tensors_memory_info(
    tensors: List[Tensor],
    msg: str,
    draw_memory_usage: bool = False,
    draw_save_dir: str = None,
    device: int = 0,
) -> Tuple[int, int, str]:

    num = len(tensors)
    if num == 0:
        return 0, 0, ""

    memory_usage = get_memory_usage(tensors)
    max_usage = max(memory_usage)
    if draw_memory_usage:
        assert os.path.exists(draw_save_dir)
        draw_memory(memory_usage, msg, draw_save_dir, device)

    info = (
        f"{f'{msg:<40}'}: {f'max_usage = {format_size(max_usage)},':<25} num = {num}\n"
    )

    return max_usage, num, info


class STAllocConfig:
    def __init__(self, args):
        self.model_memory_dir = args.model_memory_dir
        self.dynamic_model = args.dynamic_model
        self.autotune = args.autotune
        self.static_algo = args.static_algo if not self.autotune else "autotune"
        self.dynamic_algo = args.dynamic_algo if not self.autotune else "autotune"
        self.validation = args.validation
        self.debug = args.debug
        self.draw = args.draw
        self.allocator_exec_path = args.allocator_exec_path
        self.assume_dynamic_allocator_utilization = (
            args.assume_dynamic_allocator_utilization
        )
        self.assume_bestfit_allocator_utilization = (
            args.assume_bestfit_allocator_utilization
        )


def draw_memory(memory_usage: List[int], msg: str, save_dir: str, device: int):
    import matplotlib.pyplot as plt

    max_time = len(memory_usage)
    min_size = min(memory_usage)
    max_size = max(memory_usage)

    plt.figure(figsize=(12, 6))

    times = [i for i in range(0, max_time)]
    plt.plot(times, memory_usage, label="Memory Usage", color="blue", linewidth=1.5)

    plt.xlim(0, max_time)
    plt.ylim(min_size, max_size)

    plt.xlabel("Time", fontsize=32)
    plt.ylabel("Memory Usage", fontsize=32)

    plt.title("Memory Usage over Time", fontsize=32)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"dev{device}_{msg}.png"))
