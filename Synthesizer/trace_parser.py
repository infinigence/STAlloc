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

from data_structures import *
from typing import List, Dict, Tuple
import os
import re


class _GlobalTensors:
    config: STAllocConfig
    iterations: int = 3

    def __init__(self, device, config):
        self.device = device
        self.config = config
        # Trace
        self.static_tensors: List[Tensor] = []
        self.dynamic_moe_tensors: List[Tensor] = []
        self.dynamic_mlp_tensors: List[Tensor] = []
        self.unreleased_tensors: List[Tensor] = []
        self.var_tensors: List[Tensor] = []

        self.memory_info: Dict[str, Tuple[int, int]] = {}

        # Plan
        self.weight_grad_tensors: List[Tensor] = []
        self.optimizer_tensors: List[Tensor] = []
        self.init_phase_tensors: List[Tensor] = []
        self.cross_iteration_tensors_map: Dict[str, List[Tensor]] = {}
        self.static_activations_tensors: List[Tensor] = []
        self.static_activations_group_tensors: List[Tensor] = []
        self.static_activations_temporary_tensors: List[Tensor] = []
        self.fused_mlp_tensors: List[GroupTensor] = []

    def sort(self):
        self.static_tensors.sort(key=lambda x: x.start_time)
        self.dynamic_moe_tensors.sort(key=lambda x: x.start_time)
        self.dynamic_mlp_tensors.sort(key=lambda x: x.start_time)
        self.unreleased_tensors.sort(key=lambda x: x.start_time)
        self.var_tensors.sort(key=lambda x: x.start_time)

    def fuse_mlp_tensors(self):
        dynamic_fused_tensors_map = {}
        for tensor in self.dynamic_mlp_tensors:
            dynamic_fused_tensors_map.setdefault(tensor.start_layer, []).append(tensor)

        for layer, tensors in dynamic_fused_tensors_map.items():
            group = GroupTensor(tensors)
            group.set_memtype(MemType.DYNAMIC)

            iter = int(group.tensors[0].start_iteration)
            phase = group.tensors[0].start_phase
            layer = group.tensors[0].start_layer

            assert (
                group.start_iteraion == group.end_iteration
            ), f"group.start_iteraion {group.start_iteraion} != group.end_iteration {group.end_iteration}"

            self.fused_mlp_tensors.append(group)

            fused_mlp_tensor = Tensor(
                group.group_max_usage,
                group.start_time,
                group.end_time,
                group.tensors[0].malloc_id,
                iter,
                iter,
                phase,
                phase,
                layer,
                layer,
            )
            assert (
                group.size == fused_mlp_tensor.size
                and group.start_time == fused_mlp_tensor.start_time
                and group.end_time == fused_mlp_tensor.end_time
                and group.malloc_id == fused_mlp_tensor.malloc_id
            )

        self.fused_mlp_tensors.sort(key=lambda x: x.start_time)

    def update_memory_info(self, msg: str, max_suage: int, num: int):
        self.memory_info[msg] = (max_suage, num)


def load_trace(device: int, args) -> _GlobalTensors:
    GlobalTensors = _GlobalTensors(device, STAllocConfig(args))

    trace_dir = os.path.join(GlobalTensors.config.model_memory_dir, "trace")

    trace_file = os.path.join(trace_dir, f"mem_log_dev{device}.txt")
    with open(trace_file, "r") as f:
        lines = [line.strip() for line in f]

    global_time = 0
    tail_time = 0
    global_malloc_id = 0
    global_ptr_map = {}

    cur_iteration = "head"
    iterations = 0

    cur_microbatch_id = {"F": 0, "B": 0, "W": 0}  # F/B/W
    max_microbatch_id = {"F": 0, "B": 0, "W": 0}
    microbatch_pahses = []
    cur_phase = "init"

    cur_layer_stack = []
    dynamic_layer_id = 0

    dynamic_layer_pattern = re.compile(r"(\S+)->(\S+)\s*Layer\s*:\s*(\S+)-(\S+)")
    expert_layer_pattern = re.compile(r".*?layers\.(\d+).*?experts\.(\d+)$")
    for i, line in enumerate(lines):
        if "Iteration Index" in line:
            cur_iteration = line.replace("Iteration Index :", "").strip()
            if cur_iteration.isdigit():
                iterations = max(iterations, int(cur_iteration) + 1)
            if cur_iteration == "end":
                cur_iteration = "tail"
                tail_time = global_time
            for key, val in max_microbatch_id.items():
                max_microbatch_id[key] = max(val, cur_microbatch_id[key])
            cur_microbatch_id = {"F": 0, "B": 0, "W": 0}
            cur_phase = "init"
            continue
        if cur_iteration == "tail" and not line[0].isdigit():
            continue

        if "micro-batch" in line:
            F_B_W = line.split(":")[1].strip()
            cur_phase = str(cur_microbatch_id[F_B_W]) + F_B_W
            cur_microbatch_id[F_B_W] += 1
            if cur_iteration == "0":
                microbatch_pahses.append(cur_phase)
            continue

        if "Layer" in line:
            match = dynamic_layer_pattern.match(line)
            layer_name = match.group(1)
            layer_module_name = match.group(2)
            F_B_W = match.group(3)
            F_B_W_status = match.group(4) # start / end
            
            # Determine if entering MoE layer : ("SwitchMLP", "MoELayer")
            if layer_name.endswith(".mlp"):
                if F_B_W_status == "start":
                    cur_layer_stack.append(f"moe-{F_B_W}-{dynamic_layer_id}")
                    dynamic_layer_id += 1
                else:
                    assert cur_layer_stack[-1] == f"moe-{F_B_W}-{dynamic_layer_id-1}"
                    cur_layer_stack = cur_layer_stack[:-1]

            # shared_experts are inside MoE layer
            if layer_name.endswith(".shared_experts") or layer_name.endswith(".shared_expert"):
                if F_B_W_status == "start":
                    cur_layer_stack.append("shared")
                else:
                    assert cur_layer_stack[-1] == "shared"
                    cur_layer_stack = cur_layer_stack[:-1]
                continue
            
            # Process MLP layers in MoE
            if layer_module_name in ("ParallelMLP", "SequentialMLP"):
                if F_B_W_status == "start":
                    cur_layer_stack.append(f"mlp-{F_B_W}-{dynamic_layer_id - 1}")
                else:
                    assert cur_layer_stack[-1] == f"mlp-{F_B_W}-{dynamic_layer_id-1}"
                    cur_layer_stack = cur_layer_stack[:-1]
            
            # Process TopKRouter
            # TODO::deepseek-v2's TopKRouter and other MoE models' top/router layer layer_module_name may be inconsistent
            if "Router" in layer_module_name:
                if F_B_W_status == "start":
                    cur_layer_stack.append("topk_router")
                else:
                    assert cur_layer_stack[-1] == "topk_router"
                    cur_layer_stack = cur_layer_stack[:-1]
            continue

        status, ptr, size, stream = line.split(" ")  # 1 0x7f1e41329c00 1 0
        memlog = MemLog(int(status), int(size), global_time, cur_iteration, cur_phase)
        cur_layer = cur_layer_stack[-1] if len(cur_layer_stack) > 0 else ""
        memlog.set_layer(cur_layer)

        if int(status) == 1 and memlog.size > VAR_SIZE and cur_iteration != "tail":
            memlog.set_malloc_id(global_malloc_id)
            global_malloc_id += 1

        global_ptr_map.setdefault(ptr, []).append(memlog)

        global_time += 1

    GlobalTensors.iterations = iterations

    assert max_microbatch_id["F"] == max_microbatch_id["B"], "micro batch F/B mismatch"

    unreleased_tensors_total_size = 0
    for ptr, ptr_mem_logs in global_ptr_map.items():
        if ptr == "0":
            continue

        lens = len(ptr_mem_logs)
        lens = lens - 1 if lens % 2 != 0 else lens
        for i in range(0, lens, 2):
            alloc_memlog = ptr_mem_logs[i]
            free_memlog = ptr_mem_logs[i + 1]
            assert (
                alloc_memlog.status == 1
                and free_memlog.status == 0
                and alloc_memlog.size == free_memlog.size
            ), f"memlog is mismatch:{alloc_memlog.print_log()} - {free_memlog.print_log()}"

            if alloc_memlog.iteration == "tail":
                continue

            tensor = Tensor(
                alloc_memlog.size,
                alloc_memlog.time,
                free_memlog.time,
                alloc_memlog.malloc_id,
                alloc_memlog.iteration,
                free_memlog.iteration,
                alloc_memlog.phase,
                free_memlog.phase,
                alloc_memlog.layer,
                free_memlog.layer,
            )

            if tensor.is_var():
                GlobalTensors.var_tensors.append(tensor)
            elif tensor.is_dynamic_mlp():
                GlobalTensors.dynamic_mlp_tensors.append(tensor)
                assert not tensor.is_cross_iteration()
            elif tensor.is_dynamic_moe():
                # # TODO:: deepseek-v3 has a group of such dynamic-tensors
                # if tensor.start_iteration == "0" and tensor.end_iteration == "tail":
                #     print("dynamic error tensor : 0->tail", end=" : ")
                #     tensor.print_tensor()
                #     unreleased_tensors.append(tensor)
                # else:
                GlobalTensors.dynamic_moe_tensors.append(tensor)
            else:
                # For tensors allocated in initial stage && not satisfying head/tail stage release, consider as unreleased_tensors
                if tensor.start_iteration == "head" and tensor.end_iteration not in ['head', 'tail']:
                    GlobalTensors.unreleased_tensors.append(tensor)
                else:
                    GlobalTensors.static_tensors.append(tensor)

        if len(ptr_mem_logs) % 2 != 0 and ptr_mem_logs[-1].iteration != "tail":
            assert ptr_mem_logs[-1].status == 1, "memory log is not complete"
            if ptr_mem_logs[-1].size <= VAR_SIZE:
                continue

            unreleased_tensor = Tensor(
                ptr_mem_logs[-1].size,
                ptr_mem_logs[-1].time,
                tail_time,
                ptr_mem_logs[-1].malloc_id,
                ptr_mem_logs[-1].iteration,
                "tail",
                ptr_mem_logs[-1].phase,
                "init",
                ptr_mem_logs[-1].layer,
                "",
            )
            GlobalTensors.unreleased_tensors.append(unreleased_tensor)
            unreleased_tensors_total_size += unreleased_tensor.size
            assert unreleased_tensor.start_iteration == "0"

    assert unreleased_tensors_total_size <= 128 * (
        1024**2
    ), f"too much unreleased tensors. num={len(GlobalTensors.unreleased_tensors)}, total_size={format_size(unreleased_tensors_total_size)}"
    GlobalTensors.sort()

    # if args.fuse_mlp:
    if GlobalTensors.config.dynamic_algo != "dynamic-only":
        GlobalTensors.fuse_mlp_tensors()
    return GlobalTensors


def overview_analysis(
    GlobalTensors: _GlobalTensors,
    draw_memory_usage: bool = False,
    draw_save_dir: str = None,
    device: int = 0,
) -> str:
    """
    return the overview of the trace analysis result : Static Tensors, Dynamic Tensors, Fused MLP Tensors, Unreleased Tensors
    """

    overview_info = "[STAlloc-Trace] Trace Analysis Overview:\n"

    dynamic_tensors = (
        GlobalTensors.dynamic_moe_tensors + GlobalTensors.dynamic_mlp_tensors
    )
    dynamic_tensors.sort(key=lambda x: x.start_time)

    all_tensors = GlobalTensors.static_tensors + dynamic_tensors
    all_tensors.sort(key=lambda x: x.start_time)

    for tensors, msg in zip(
        [
            all_tensors,
            GlobalTensors.static_tensors,
            dynamic_tensors,
            GlobalTensors.unreleased_tensors,
        ],
        ["All", "Static_Tensors", "Dynamic_Tensors", "Unreleased_Tensors"],
    ):
        if msg == "Unreleased_Tensors":
            draw_memory_usage = False
        max_usage, num, info = get_tensors_memory_info(
            tensors, msg, draw_memory_usage, draw_save_dir, device
        )
        overview_info += info
        GlobalTensors.update_memory_info(msg, max_usage, num)
    return overview_info

def plot_allocated_memory_over_time(memory_activities_list, labels, colors):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.lines as mlines
    import numpy as np
    plt.rc('font', family='arial')
    
    def calculate_allocated_memory_over_time(memory_activities):
        # Find maximum time point
        max_time = max([activity['end_time'] for activity in memory_activities])

        # Initialize allocated memory list
        allocated_memory = np.zeros(max_time + 1)

        # Iterate through memory activities, update allocated memory
        for activity in memory_activities:
            start_time = activity['start_time']
            end_time = activity['end_time']
            size = activity['size']

            allocated_memory[start_time] += size  # Allocate memory
            if end_time < max_time:
                allocated_memory[end_time] -= size  # Free memory

        # Calculate cumulative memory
        allocated_memory = np.cumsum(allocated_memory)

        return allocated_memory

    # Create independent figure
    plt.figure(figsize=(10, 7))

    # Plot allocated memory over time for each file
    for memory_activities, label, color in zip(memory_activities_list, labels, colors):
        allocated_memory = calculate_allocated_memory_over_time(memory_activities)
        time_points = np.arange(len(allocated_memory))
        allocated_memory = np.array(allocated_memory) / (1024 ** 3)
        # Plot line chart

        plt.plot(time_points, allocated_memory, color=color, label=label)

    plt.xlabel('Memory Operation Order', fontsize=32)
    plt.ylabel('Accumulated Size (GB)', fontsize=32)
    # plt.xlim(0, 1.7e5)
    plt.ylim(0, 20)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    def format_func(value, tick_number):
        return f'{value / 1e3:.0f}K'
    xticks = plt.gca().get_xticks()
    yticks = plt.gca().get_yticks()
    
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_func))    
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(4e4))

    plt.gca().set_xticks([0] + plt.gca().get_xticks().tolist()[1:])
    plt.gca().set_yticks([0] + plt.gca().get_yticks().tolist()[1:])
    plt.gca().set_xticklabels([''] + plt.gca().get_xticklabels()[1:])
    plt.gca().set_yticklabels([''] + plt.gca().get_yticklabels()[1:])
    plt.xlim(0, 0.8e5)


    # plt.title('Allocated Memory Over Time')
    plt.grid(True)

    legend_elements_up = [
        mlines.Line2D([], [], color=colors[i], label=f'{case}', linewidth=2) for i, case in enumerate(labels)
    ]


    ax = plt.gca()
    legend = plt.legend(handles=legend_elements_up, loc='upper center', edgecolor=(0.5, 0.5, 0.5), borderpad=0, borderaxespad=0.5,
                            ncol=3, fontsize=22, bbox_to_anchor=(0.45, 1.15), handletextpad=0.1, handlelength=1.5,columnspacing=1, frameon=False)

    plt.tight_layout()
    plt.savefig('temporal_allocated_memory_over_time.pdf')
    
def lifecycle_analysis(
    GlobalTensors: _GlobalTensors,
    draw_memory_usage: bool = False,
    draw_save_dir: str = None,
    device: int = 0,
) -> str:
    """
    return the lifecycle analysis result
    Static Tensors => three parts
    1. [Weights & Gradients, Optimizer States, Init Phase Tensor]
    2. [Remaining Cross Iteration Tensor(Static)]
    3. [Activations::Temporary Tensor, Activations::Group Tensor, Dynamic Tensors]
    """

    lifecycle_analysis_info = "\n[STAlloc-Trace] LifeCycle Analysis:\n"

    # Step-1 : Resident Memory -> [Weights & Gradients, Optimizer States, Init Phase Tensors]

    def analysis_resident_memory():
        resident_analysis_info = "Resident Memory:\n"

        for tensor in GlobalTensors.static_tensors:
            if tensor.start_iteration == "1":
                break

            if tensor.start_iteration == "head":
                if tensor.end_iteration == "tail":
                    tensor.set_memtype(MemType.WEIGHT_GRAD)
                    GlobalTensors.weight_grad_tensors.append(tensor)
                elif tensor.end_iteration == "head":
                    tensor.set_memtype(MemType.INIT_PHASE)
                    GlobalTensors.init_phase_tensors.append(tensor)
                else:
                    raise RuntimeError(f"tensor {tensor.start_iteration}->{tensor.end_iteration} is not valid")
            elif tensor.start_iteration == "0" and tensor.end_iteration == "tail":
                tensor.set_memtype(MemType.OPTIMIZER_STATES)
                GlobalTensors.optimizer_tensors.append(tensor)

        for tensor, msg in zip(
            [
                GlobalTensors.weight_grad_tensors,
                GlobalTensors.optimizer_tensors,
                GlobalTensors.init_phase_tensors,
            ],
            ["Weight_Gradient", "Optimizer_States", "Init_Phase"],
        ):
            max_usage, num, info = get_tensors_memory_info(tensor, msg)
            resident_analysis_info += info
            GlobalTensors.update_memory_info(msg, max_usage, num)

        return resident_analysis_info

    resident_analysis_info = analysis_resident_memory()
    lifecycle_analysis_info += resident_analysis_info

    # Step-2 : [Cross Iteration Tensor]
    def analysis_cross_iteration_tensors(need_set_memtype: bool = False):
        cross_iter_info = "\nCross Iteration Tensors(Static):\n"
        for tensor in GlobalTensors.static_tensors:
            if tensor.mem_type is not None:
                continue
            if not tensor.is_cross_iteration():
                continue
            
            if need_set_memtype:
                tensor.set_memtype(MemType.CROSS_ITER)
            GlobalTensors.cross_iteration_tensors_map.setdefault(
                f"{tensor.start_iteration}->{tensor.end_iteration}", []
            ).append(tensor)

        for key, tensors in GlobalTensors.cross_iteration_tensors_map.items():
            max_usage, num, info = get_tensors_memory_info(
                tensors, f"Cross_Iteration_{key}"
            )
            cross_iter_info += info
            GlobalTensors.update_memory_info(f"Cross_Iteration_{key}", max_usage, num)

        return cross_iter_info

    # Only output cross_iter_info, don't set memtype, treat as activation tensors during allocation phase
    cross_iter_tensors_analysis_only = True
    cross_iter_info = analysis_cross_iteration_tensors(need_set_memtype=not cross_iter_tensors_analysis_only)
    lifecycle_analysis_info += cross_iter_info
    if cross_iter_tensors_analysis_only:
        GlobalTensors.cross_iteration_tensors_map = {}

    # Step-3 : [Activations::Temporary Tensor, Activations::Group Tensor
    def analysis_activations():
        activations_info = "\nActivations:\n"

        for tensor in GlobalTensors.static_tensors:
            if tensor.mem_type is not None:
                continue
            # assert not tensor.is_cross_iteration()
            if tensor.is_cross_phase():
                tensor.set_memtype(MemType.ACTIVATIONS_GROUP)
                GlobalTensors.static_activations_group_tensors.append(tensor)
            else:
                tensor.set_memtype(MemType.ACTIVATIONS_TEMPORARY)
                GlobalTensors.static_activations_temporary_tensors.append(tensor)

        GlobalTensors.static_activations_group_tensors.sort(key=lambda x: x.start_time)
        GlobalTensors.static_activations_temporary_tensors.sort(
            key=lambda x: x.start_time
        )
        GlobalTensors.static_activations_tensors = (
            GlobalTensors.static_activations_group_tensors
            + GlobalTensors.static_activations_temporary_tensors
        )
        GlobalTensors.static_activations_tensors.sort(key=lambda x: x.start_time)

        for tensors, msg in zip(
            [
                GlobalTensors.static_activations_tensors,
                GlobalTensors.static_activations_group_tensors,
                GlobalTensors.static_activations_temporary_tensors,
            ],
            [
                "Static_Activations",
                "Static_Activations_Group",
                "Static_Activations_Temporary",
            ],
        ):
            max_usage, num, info = get_tensors_memory_info(
                tensors,
                msg,
                draw_memory_usage,
                draw_save_dir,
                device,
            )
            activations_info += info
            GlobalTensors.update_memory_info(msg, max_usage, num)

        if True or GlobalTensors.config.dynamic_model:
            for tensors, msg in zip(
                [
                    GlobalTensors.dynamic_moe_tensors,
                    GlobalTensors.dynamic_mlp_tensors,
                    GlobalTensors.fused_mlp_tensors,
                ],
                [
                    "Dynamic_MoELayer_Tensors",
                    "Dynamic_MLPLayer_Tensors",
                    "Dynamic_MLPFused_Tensors",
                ],
            ):
                max_usage, num, info = get_tensors_memory_info(
                    tensors,
                    msg,
                    draw_memory_usage,
                    draw_save_dir,
                    device,
                )
                activations_info += info
                GlobalTensors.update_memory_info(msg, max_usage, num)

        return activations_info

    activations_info = analysis_activations()
    lifecycle_analysis_info += activations_info

    # Step-4 : check if there is any remaining static tensors
    error_tensors = [
        tensor for tensor in GlobalTensors.static_tensors if tensor.mem_type is None
    ]
    if error_tensors:
        _, _, info = get_tensors_memory_info(error_tensors, "Unassigned_Tensors")
        raise RuntimeError(f"There are some error tensors: {info}")

    return lifecycle_analysis_info


def parse_trace(
    GlobalTensors: _GlobalTensors,
    draw_memory_usage: bool = False,
    draw_save_dir: str = None,
    device: int = 0,
) -> str:

    return overview_analysis(
        GlobalTensors, draw_memory_usage, draw_save_dir, device
    ) + lifecycle_analysis(GlobalTensors, draw_memory_usage, draw_save_dir, device)
