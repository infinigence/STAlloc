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

from functools import partial
from .memory_config import STAllocConfig, STALLOC_OP_TYPE


def hook_nothing(model, args):
    return model


def hook_trace_model(model, args):
    def memlog_forward_pre_hook(name, module, input_tensor):
        STAllocConfig.trace_line(
            f"{name}->{module.__class__.__name__} Layer : F-start\n"
        )

    def memlog_forward_post_hook(name, module, input_tensor, output_tensor):
        STAllocConfig.trace_line(f"{name}->{module.__class__.__name__} Layer : F-end\n")

    def memlog_backward_pre_hook(name, module, grad_output):
        STAllocConfig.trace_line(
            f"{name}->{module.__class__.__name__} Layer : B-start\n"
        )

    def memlog_backward_post_hook(name, module, grad_input, grad_output):
        STAllocConfig.trace_line(f"{name}->{module.__class__.__name__} Layer : B-end\n")

    def gen_trace_hooks(name):
        forward_pre_hook = partial(memlog_forward_pre_hook, name)
        forward_post_hook = partial(memlog_forward_post_hook, name)
        backward_pre_hook = partial(memlog_backward_pre_hook, name)
        backward_post_hook = partial(memlog_backward_post_hook, name)
        return (
            forward_pre_hook,
            forward_post_hook,
            backward_pre_hook,
            backward_post_hook,
        )

    for name, module in model.named_modules():
        module_name = str(module.__class__.__name__).strip()
        
        if 'Model' in module_name:
            continue

        is_moe_layer = module_name in ("SwitchMLP", "MoELayer")
        is_dynamic_mlp_layer = module_name in ("ParallelMLP", "SequentialMLP")
        is_shared_mlp_layer = 'shared_expert' in name.split(".")[-1]
        is_router = 'router' in name
        is_row = "RowParallelLinear" in module_name and 'local_experts' in name

        if is_moe_layer or is_shared_mlp_layer or is_dynamic_mlp_layer or is_router or is_row:
            (
                forward_pre_hook,
                forward_post_hook,
                backward_pre_hook,
                backward_post_hook,
            ) = gen_trace_hooks(name)
            module.register_forward_pre_hook(forward_pre_hook)
            module.register_forward_hook(forward_post_hook)
            module.register_full_backward_pre_hook(backward_pre_hook)
            module.register_full_backward_hook(backward_post_hook)
    return model


def hook_alloc_model(model, args):
    def checkpoint_forward_pre_hook(name, op_type, module, input_tensor):
        STAllocConfig.checkpoint(name, op_type, True, True)

    def checkpoint_forward_post_hook(
        name, op_type, module, input_tensor, output_tensor
    ):
        STAllocConfig.checkpoint(name, op_type, True, False)

    def checkpoint_backward_pre_hook(name, op_type, module, grad_output):
        STAllocConfig.checkpoint(name, op_type, False, True)

    def checkpoint_backward_post_hook(name, op_type, module, grad_input, grad_output):
        STAllocConfig.checkpoint(name, op_type, False, False)

    # expert_layer_pattern = re.compile(r".*?layers\.(\d+).*?experts\.(\d+)$")
    # last_expert_id = int(args.num_experts / args.expert_model_parallel_size) - 1

    for name, module in model.named_modules():
        if args.num_experts is None:
            return model
        module_name = str(module.__class__.__name__).strip()
        if 'Model' in module_name:
            continue

        is_moe_layer = module_name in ("SwitchMLP", "MoELayer")
        is_dynamic_mlp_layer = module_name in ("ParallelMLP", "SequentialMLP")
        is_shared_mlp_layer = 'shared_expert' in name.split(".")[-1]
        is_router = 'router' in name
        is_row = "RowParallelLinear" in module_name and 'local_experts' in name

        STALLOC_OP = None
        if is_moe_layer:
            STALLOC_OP = STALLOC_OP_TYPE.MoE
        elif is_row:
            STALLOC_OP = STALLOC_OP_TYPE.Row
        elif is_dynamic_mlp_layer:
            STALLOC_OP = STALLOC_OP_TYPE.MLP
        elif is_shared_mlp_layer:
            STALLOC_OP = STALLOC_OP_TYPE.SharedMLP
        elif is_router:
            STALLOC_OP = STALLOC_OP_TYPE.Router

        if STALLOC_OP is not None:
            forward_pre_hook = partial(checkpoint_forward_pre_hook, name, STALLOC_OP)
            forward_post_hook = partial(checkpoint_forward_post_hook, name, STALLOC_OP)
            backward_pre_hook = partial(checkpoint_backward_pre_hook, name, STALLOC_OP)
            backward_post_hook = partial(checkpoint_backward_post_hook, name, STALLOC_OP)
            module.register_forward_pre_hook(forward_pre_hook)
            module.register_forward_hook(forward_post_hook)
            module.register_full_backward_pre_hook(backward_pre_hook)
            module.register_full_backward_hook(backward_post_hook)
    return model


def hook_memory_model(model, args):
    if STAllocConfig.dynamic:
        if STAllocConfig.mode == "Trace":
            return hook_trace_model(model, args)
        elif STAllocConfig.mode == "Alloc":
            return hook_alloc_model(model, args)
    return hook_nothing(model, args)
