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


from .patcher import Patcher

from .memory_config import STAllocConfig, STALLOC_OP_TYPE
import torch
from megatron.training.global_vars import get_args
from functools import wraps

def STAlloc_train_step_wrapper(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        global_args = get_args()
        if STAllocConfig.mode == "Trace":
            STAllocConfig.trace_line(
                f"Iteration Index : {global_args.curr_iteration}\n"
            )
        result = func(self, *args, **kwargs)
        if (
            STAllocConfig.mode == "Trace"
            and global_args.curr_iteration == global_args.train_iters - 1
        ):
            STAllocConfig.trace_line("Iteration Index : end\n")
        elif STAllocConfig.mode == "Torch":
            _GB = 1024**3
            max_reserved = torch.cuda.max_memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
            print(
                f"dev{torch.cuda.current_device()}, iter{global_args.curr_iteration} : max_reserved:{max_reserved/_GB:.2f}, max_allocated:{max_allocated/_GB:.2f}, utilization:{(max_allocated/max_reserved):.2%}",
                flush=True,
            )
            torch.cuda.reset_peak_memory_stats()
        elif STAllocConfig.mode == "Alloc":
            STAllocConfig.get_memory_stat(global_args.curr_iteration)
        return result

    return wrapper


def STAlloc_evaluate_wrapper(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if STAllocConfig.mode != "Alloc":
            return func(self, *args, **kwargs)
        STAllocConfig.checkpoint("Eval", STALLOC_OP_TYPE.Eval, True, True)
        result = func(self, *args, **kwargs)
        STAllocConfig.checkpoint("Eval", STALLOC_OP_TYPE.Eval, True, False)
        return result
    return wrapper

def STAlloc_forward_step_wrapper(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if STAllocConfig.mode != "Trace":
            return func(self, *args, **kwargs)
        STAllocConfig.trace_line("micro-batch : F\n")
        return func(self, *args, **kwargs)

    return wrapper

def STAlloc_backward_step_wrapper(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if STAllocConfig.mode != "Trace":
            return func(self, *args, **kwargs)
        STAllocConfig.trace_line("micro-batch : B\n")
        return func(self, *args, **kwargs)

    return wrapper

def STAlloc_report_memory_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return
    return wrapper

class STAllocMemoryBuffer:
    """Diabled"""
    def __init__(self):
        print("STAllocMemoryBuffer disabled GlobalMemoryBuffer")

    def get_tensor(self, tensor_shape, dtype, name):
        return torch.empty(tensor_shape, dtype=dtype, device=torch.cuda.current_device())

def token_permutation_wrapper(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if STAllocConfig.mode == "Trace":
            STAllocConfig.trace_line("token_permutation : F-start\n")
        result = func(self, *args, **kwargs)
        if STAllocConfig.mode == "Trace":
            STAllocConfig.trace_line("token_permutation : F-end\n")
        return result
    return wrapper

def token_dispatcher_preprocess_wrapper(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if STAllocConfig.mode == "Trace":
            STAllocConfig.trace_line("token_dispatcher preprocess: F-start\n")
        result = func(self, *args, **kwargs)
        if STAllocConfig.mode == "Trace":
            STAllocConfig.trace_line("token_dispatcher preprocess: F-end\n")
        return result
    return wrapper

def token_unpermutation_wrapper(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if STAllocConfig.mode == "Trace":
            STAllocConfig.trace_line("token_unpermutation : F-start\n")
        result = func(self, *args, **kwargs)
        if STAllocConfig.mode == "Trace":
            STAllocConfig.trace_line("token_unpermutation : F-end\n")
        return result
    return wrapper

def add_STAlloc_patch():
    Patcher.add_wrapper('megatron.training.training.train_step', STAlloc_train_step_wrapper)

    Patcher.add_wrapper('megatron.training.training.evaluate', STAlloc_evaluate_wrapper)

    Patcher.add_wrapper('megatron.training.utils.report_memory', STAlloc_report_memory_wrapper)

    Patcher.add_wrapper(
        'megatron.core.pipeline_parallel.schedules.forward_step', STAlloc_forward_step_wrapper
    )
    Patcher.add_wrapper(
        'megatron.core.pipeline_parallel.schedules.backward_step', STAlloc_backward_step_wrapper
    )
    if STAllocConfig.mode == "Torch":
        return
    
    Patcher.add_patch(
        'megatron.core.utils.GlobalMemoryBuffer', 'stalloc.utils.memory_patcher.STAllocMemoryBuffer'
    )

def apply_memory_patch():
    add_STAlloc_patch()
    Patcher.apply()
