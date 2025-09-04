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
from trace_parser import _GlobalTensors
from copy import deepcopy
from typing import List, Dict


class _MemoryPlan:
    device: int
    config: STAllocConfig
    model_memory_info_dir: str = ""

    unreleased_tensor_ids: List[int] = []
    var_tensors_max_usage: int = 0

    memory_addrs: List[int] = [
        0,  # weight_grad_tensors_addr
        0,  # optimizer_state_tensors_addr
        0,  # cross_iteration_tensors_addr
        0,  # activation_tensors_addr
        0,  # static tensors total reserved
    ]

    memory_addrs_map: Dict[MemType, int] = {
        MemType.WEIGHT_GRAD: 0,
        MemType.OPTIMIZER_STATES: 1,
        MemType.INIT_PHASE: 1,
        MemType.CROSS_ITER: 2,
        MemType.ACTIVATIONS: 3,
        MemType.ACTIVATIONS_GROUP: 3,
        MemType.ACTIVATIONS_TEMPORARY: 3,
    }

    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.model_memory_info_dir = config.model_memory_dir

    def update_offset(self, tensors: List[Tensor], mem_type: MemType):
        memory_area_id = self.memory_addrs_map[mem_type]
        start_addr = self.memory_addrs[memory_area_id]

        if mem_type != MemType.WEIGHT_GRAD:
            assert start_addr != 0, f"{mem_type} : {memory_area_id} {start_addr}"

        offset = start_addr
        for tensor in tensors:
            assert tensor.mem_type == mem_type
            assert tensor.offset == -1
            tensor.offset = offset
            offset += roundup(tensor.size)

        if mem_type != MemType.INIT_PHASE:
            self.memory_addrs[memory_area_id + 1] = offset


def generate_memory_plan(GlobalTensors: _GlobalTensors) -> Tuple[int, int]:
    MemoryPlan = _MemoryPlan(GlobalTensors.device, GlobalTensors.config)

    var_tensors_max_usage, _, _ = get_tensors_memory_info(
        GlobalTensors.var_tensors, "Var_Tensors"
    )
    MemoryPlan.var_tensors_max_usage = var_tensors_max_usage

    unreleased_tensor_ids = [
        tensor.malloc_id for tensor in GlobalTensors.unreleased_tensors
    ]
    MemoryPlan.unreleased_tensor_ids = unreleased_tensor_ids

    make_plan_for_residents_memory(MemoryPlan, GlobalTensors)

    check_activation_tensor_consistency(GlobalTensors)

    make_plan_for_cross_iteration_tensors(MemoryPlan, GlobalTensors)

    static_algos = [GlobalTensors.config.static_algo]
    if static_algos[0] == "autotune":
        static_algos = ["group", "split-and-fuse"]
    dynamic_algos = [GlobalTensors.config.dynamic_algo]
    if dynamic_algos[0] == "autotune" and GlobalTensors.config.dynamic_model:
        dynamic_algos = ["fuse", "dynamic-only", "static-reuse"]

    min_all_activations_tensor_reserved = 1024**4
    dynamic_allocator_utilization_assumed = (
        MemoryPlan.config.assume_dynamic_allocator_utilization / 100
    )
    bestfit_allocator_utilization_assumed = (
        MemoryPlan.config.assume_bestfit_allocator_utilization / 100
    )

    tune_num = len(static_algos) * len(dynamic_algos)

    GlobalTensors_tuned = [deepcopy(GlobalTensors) for i in range(tune_num)]
    tune_id = -1
    best_tune_id = -1
    best_tune_plan_reserved = -1

    for static_algo in static_algos:

        for dynamic_algo in dynamic_algos:
            tune_id += 1

            using_group_algo = static_algo == "group"
            using_fuse_mlp_algo = dynamic_algo == "fuse"
            using_static_reuse_algo = dynamic_algo == "static-reuse"

            activations_plan_allocated, activations_plan_reserved = (
                make_plan_for_activations_tensors(
                    MemoryPlan,
                    GlobalTensors_tuned[tune_id],
                    using_group_algo,
                    using_fuse_mlp_algo,
                    using_static_reuse_algo,
                )
            )

            if using_fuse_mlp_algo:
                dynamic_allocator_allocated = int(
                    GlobalTensors.memory_info["Dynamic_MoELayer_Tensors"][0]
                    + (1 - bestfit_allocator_utilization_assumed)
                    * GlobalTensors.memory_info["Dynamic_MLPLayer_Tensors"][0]
                )
            else:
                dynamic_allocator_allocated = GlobalTensors.memory_info[
                    "Dynamic_Tensors"
                ][0]

            dynamic_allocator_reserved = int(
                dynamic_allocator_allocated / dynamic_allocator_utilization_assumed
            )
            cur_total_activations_reserved = (
                dynamic_allocator_reserved + activations_plan_reserved
            )

            print(
                f"tune_id={tune_id}, algo1:{static_algo}, algo2:{dynamic_algo}  cur_total_activations_reserved = {format_size(cur_total_activations_reserved)}"
            )

            if cur_total_activations_reserved < min_all_activations_tensor_reserved:
                min_all_activations_tensor_reserved = cur_total_activations_reserved
                best_tune_id = tune_id
                best_tune_plan_reserved = activations_plan_reserved

            for tensor in GlobalTensors_tuned[tune_id].static_tensors:
                if tensor.start_iteration not in ("head", "0", "1"):
                    continue
                if tensor.mem_type == MemType.UNRELEASED:
                    assert tensor.is_cross_iteration()
                    print(tensor)
                    continue
                assert tensor.offset != -1, f"{tensor} {tensor.mem_type}"

            if using_fuse_mlp_algo:
                for tensor in GlobalTensors_tuned[tune_id].fused_mlp_tensors:
                    if tensor.start_iteration not in ("head", "0", "1"):
                        continue
                    assert tensor.offset != -1, f"{tensor}"

    print("best_tune_id = ", best_tune_id)
    if len(static_algos) > 1:
        use_group = best_tune_id // 2 == 0
    else:
        use_group = static_algos[0] == "group"

    memory_area_id = MemoryPlan.memory_addrs_map[MemType.ACTIVATIONS]
    MemoryPlan.memory_addrs[memory_area_id + 1] = (
        MemoryPlan.memory_addrs[memory_area_id] + best_tune_plan_reserved
    )
    save_memory_plan(MemoryPlan, GlobalTensors_tuned[best_tune_id], use_group)

    if MemoryPlan.config.validation:
        static_plan_validation(GlobalTensors_tuned[best_tune_id])

    return (
        GlobalTensors.memory_info["All"][0],
        MemoryPlan.memory_addrs[memory_area_id] + min_all_activations_tensor_reserved,
    )


def save_memory_plan(MemoryPlan: _MemoryPlan, GlobalTensors: _GlobalTensors, use_group: bool):
    unreleased_msg = ""
    for id in MemoryPlan.unreleased_tensor_ids:
        unreleased_msg += str(id) + " "

    static_memory_max_allocated = GlobalTensors.memory_info["Static_Tensors"][0]
    static_memory_max_reserved = MemoryPlan.memory_addrs[-1]

    static_malloc_event_num_before_iter1 = 0
    static_malloc_event_num_in_iter1 = 0
    for tensor in GlobalTensors.static_tensors:
        if tensor.mem_type == MemType.UNRELEASED:
            continue
        iter = tensor.start_iteration
        if iter in ("head", "0"):
            static_malloc_event_num_before_iter1 += 1
        elif iter == "1":
            static_malloc_event_num_in_iter1 += 1
        else:
            break

    title_msg = f"{static_memory_max_allocated} {static_memory_max_reserved} {(static_malloc_event_num_before_iter1)} {(static_malloc_event_num_in_iter1)} {MemoryPlan.var_tensors_max_usage}\n"

    plan_dir = os.path.join(MemoryPlan.model_memory_info_dir, "output", "plan")
    os.makedirs(plan_dir, exist_ok=True)
    static_plan_file = os.path.join(plan_dir, f"dev_{GlobalTensors.device}.txt")
    with open(static_plan_file, "w") as f:
        f.write(unreleased_msg + "\n")
        f.write(title_msg)
        for tensor in GlobalTensors.static_tensors:
            if tensor.mem_type == MemType.UNRELEASED:
                continue
            f.write(f"{tensor.offset} {tensor.size}\n")

    if use_group:
        origin_interval_file = os.path.join(MemoryPlan.model_memory_info_dir, "output", "activations_memlog", "activations_tensors", f"dev{GlobalTensors.device}_group_allocated_plan_spare_addr.txt")
    else:
        origin_interval_file = os.path.join(MemoryPlan.model_memory_info_dir, "output", "activations_memlog", "activations_tensors", f"dev{GlobalTensors.device}_allocated_plan_spare_addr.txt")

    dynamic_algo = GlobalTensors.config.dynamic_algo
    dynamic_plan_file = os.path.join(
        plan_dir, f"dev_{GlobalTensors.device}_dynamic.txt"
    )
    if dynamic_algo == "fuse":
        moe_layer_num_per_iter = int(
            len(GlobalTensors.fused_mlp_tensors) / GlobalTensors.iterations
        )
        with open(dynamic_plan_file, "w") as f:
            f.write(f"{0} {moe_layer_num_per_iter}\n")
            for i in range(moe_layer_num_per_iter * 2):
                tensor = GlobalTensors.fused_mlp_tensors[i]
                f.write(f"{tensor.offset} {tensor.size}\n")
    # elif dynamic_algo == "static-reuse":
    #     # copy from origin_interval_file
    #     with open(origin_interval_file, "r") as f:
    #         lines = f.readlines()
    #     with open(dynamic_plan_file, "w") as f:
    #         for line in lines:
    #             f.write(line)

def make_plan_by_allocator(
    MemoryPlan: _MemoryPlan,
    mem_type: MemType,
    tensors: List[Tensor],
    group_tensors: List[GroupTensor] = [],
    static_reuse_path: str = None,
    iter1_start: int = None,
    iter1_end: int = None,
) -> Tuple[int, int]:
    start_addr = MemoryPlan.memory_addrs[MemoryPlan.memory_addrs_map[mem_type]]
    # assert start_addr != 0

    from allocator_caller import allocator_call

    memlog_type = (
        "init_phase_tensors"
        if mem_type == MemType.INIT_PHASE
        else "activations_tensors"
    )
    memlog_dir = os.path.join(
        MemoryPlan.model_memory_info_dir, "output", "activations_memlog", memlog_type
    )

    max_allocated, max_reserved = allocator_call(
        MemoryPlan.config.allocator_exec_path,
        memlog_dir,
        MemoryPlan.device,
        start_addr,
        tensors,
        group_tensors,
        static_reuse_path,
        iter1_start,
        iter1_end,
    )

    print(
        f"call allocator : {mem_type}, max_reserved = {format_size(max_reserved)}, max_allocated = {format_size(max_allocated)}, utilization = {(max_allocated / max_reserved):.2%}"
    )
    return max_allocated, max_reserved


def make_plan_for_residents_memory(
    MemoryPlan: _MemoryPlan, GlobalTensors: _GlobalTensors
):
    # Weights and gradients
    MemoryPlan.update_offset(GlobalTensors.weight_grad_tensors, MemType.WEIGHT_GRAD)

    # Optimizer states
    MemoryPlan.update_offset(GlobalTensors.optimizer_tensors, MemType.OPTIMIZER_STATES)

    # Init phase tensors
    init_phase_tensors_total_size = get_tensors_align_size(
        GlobalTensors.init_phase_tensors
    )
    if init_phase_tensors_total_size < (
        GlobalTensors.memory_info["Static_Tensors"][0] - MemoryPlan.memory_addrs[1]
    ):
        MemoryPlan.update_offset(GlobalTensors.init_phase_tensors, MemType.INIT_PHASE)
    else:
        init_phase_tensors_max_allocated, init_phase_tensors_max_reserved = (
            make_plan_by_allocator(
                MemoryPlan, MemType.INIT_PHASE, GlobalTensors.init_phase_tensors
            )
        )

def get_dynamic_intervals(GlobalTensors: _GlobalTensors, path, device):
    dynamic_intervals = {}
    moe_dynamic_intervals = {}
    dynamic_tensors = sorted([t for t in (GlobalTensors.dynamic_mlp_tensors) if int(t.start_iteration) < 2], key=lambda x: x.start_time)
    moe_dynamic_tensors = sorted([t for t in (GlobalTensors.dynamic_moe_tensors) if int(t.start_iteration) < 2], key=lambda x: x.start_time)
    layer_event_count = {}
    cross_layer_id = {}
    layer_in_iter0 = set()
    layer_in_iter1 = set()

    for tensor in moe_dynamic_tensors:
        if tensor.start_phase not in moe_dynamic_intervals:
            moe_dynamic_intervals[tensor.start_phase] = {}
        if tensor.end_phase not in moe_dynamic_intervals:
            moe_dynamic_intervals[tensor.end_phase] = {}
        if tensor.start_layer not in moe_dynamic_intervals[tensor.start_phase]:
            moe_dynamic_intervals[tensor.start_phase][tensor.start_layer] = {
                'start_time': tensor.start_time,
                'end_time': 0,
            }
        else:
            moe_dynamic_intervals[tensor.start_phase][tensor.start_layer]['end_time'] = max(tensor.end_time, moe_dynamic_intervals[tensor.start_phase][tensor.start_layer]['end_time'])
        if tensor.end_layer not in moe_dynamic_intervals[tensor.end_phase]:
            moe_dynamic_intervals[tensor.end_phase][tensor.end_layer] = {
                'start_time': tensor.start_time,
                'end_time': 0,
            }
        else:
            moe_dynamic_intervals[tensor.end_phase][tensor.end_layer]['end_time'] = max(tensor.end_time, moe_dynamic_intervals[tensor.end_phase][tensor.end_layer]['end_time'])

    for tensor in dynamic_tensors:
        if tensor.start_phase not in dynamic_intervals:
            dynamic_intervals[tensor.start_phase] = {}
        if tensor.end_phase not in dynamic_intervals:
            dynamic_intervals[tensor.end_phase] = {}

        start_layer_type = tensor.start_layer.split('-')[0]
        if start_layer_type == 'mlp':
            if tensor.start_iteration == "0":
                layer_in_iter0.add(tensor.start_layer)
            else:
                layer_in_iter1.add(tensor.start_layer)

            if tensor.start_layer not in dynamic_intervals[tensor.start_phase]:
                dynamic_intervals[tensor.start_phase][tensor.start_layer] = {
                    'start_time': tensor.start_time,
                    'end_time': 0,
                }
            else:
                dynamic_intervals[tensor.start_phase][tensor.start_layer]['start_time'] = min(tensor.start_time, dynamic_intervals[tensor.start_phase][tensor.start_layer]['start_time'])
            
            if tensor.start_layer not in layer_event_count:
                layer_event_count[tensor.start_layer] = 0
                cross_layer_id[tensor.start_layer] = []
            
            if tensor.start_layer != tensor.end_layer:
                cross_layer_id[tensor.start_layer].append(layer_event_count[tensor.start_layer])
            layer_event_count[tensor.start_layer] += 1

        end_layer_type = tensor.end_layer.split('-')[0]
        if end_layer_type == 'mlp':
            if tensor.end_layer not in dynamic_intervals[tensor.end_phase]:
                dynamic_intervals[tensor.end_phase][tensor.end_layer] = {
                    'start_time': float('inf'),
                    'end_time': tensor.end_time,
                }
            else:
                dynamic_intervals[tensor.end_phase][tensor.end_layer]['end_time'] = max(tensor.end_time, dynamic_intervals[tensor.end_phase][tensor.end_layer]['end_time'])

    # sort by phase idx
    phases = sorted(list(dynamic_intervals.keys()), key=lambda x: int(x[:-1]))
    forward_phases = [p for p in phases if p.endswith('F')]
    backward_phases = [p for p in phases if p.endswith('B')]
    
    # Analyze each phase combination
    cross_layer_intervals = []
    
    for f_phase, b_phase in zip(forward_phases, backward_phases):
        # Get layer information for current phase combination
        f_layers = sorted(dynamic_intervals[f_phase].keys(), key=lambda x: int(x.split('-')[-1]))
        b_layers = sorted(dynamic_intervals[b_phase].keys(), key=lambda x: int(x.split('-')[-1]))
        
        # Get minimum layer number
        min_f_layer = int(f_layers[0].split('-')[-1]) if f_layers else 0
        iter_layers = int(f_layers[len(f_layers) // 2].split('-')[-1]) - min_f_layer
        min_b_layer = int(b_layers[0].split('-')[-1]) if b_layers else 0
        
        # Calculate number of layers in current phase combination
        num_layers = len(f_layers) // 2  # Each phase contains layers from two iterations
        num_recompute_layers = len(b_layers) // 2 - num_layers
        
        # Process each iteration
        for iter_idx in range(2):
            # Process layers that don't need recompute
            for layer_idx in range(num_recompute_layers, num_layers):
                # F phase layer numbers start from 0 and increase
                f_layer = f'mlp-F-{min_f_layer + layer_idx + iter_idx * iter_layers}'
                # B phase layer numbers start from num_layers-1 and decrease
                b_layer = f'mlp-B-{min_b_layer + num_layers - 1 - layer_idx + iter_idx * iter_layers}'
                
                if f_layer in dynamic_intervals[f_phase] and b_layer in dynamic_intervals[b_phase]:
                    cross_layer_intervals.append({
                        'layer': f'layer-{layer_idx}',
                        'iter': iter_idx,
                        'type': 'no_recompute',
                        'f_layer': f_layer,
                        'b_layer': b_layer,
                        'start_time': dynamic_intervals[f_phase][f_layer]['start_time'],
                        'end_time': dynamic_intervals[b_phase][b_layer]['end_time']
                    })
            
            # Process layers that need recompute
            for layer_idx in range(num_recompute_layers):
                f_layer = f'mlp-F-{min_b_layer + num_layers + num_recompute_layers - 2 * layer_idx - 2 + iter_idx * iter_layers}'
                b_layer = f'mlp-B-{min_b_layer + num_layers + num_recompute_layers - 2 * layer_idx - 1 + iter_idx * iter_layers}'
                if f_layer in dynamic_intervals[b_phase] and b_layer in dynamic_intervals[b_phase]:
                    cross_layer_intervals.append({
                        'layer': f'layer-{layer_idx}',
                        'iter': iter_idx,
                        'type': 'recompute',
                        'f_layer': f_layer,
                        'b_layer': b_layer,
                        'start_time': dynamic_intervals[b_phase][f_layer]['start_time'],
                        'end_time': dynamic_intervals[b_phase][b_layer]['end_time']
                    })
        
    # Write to file
    intra_interval_path = os.path.join(path, f"dev{device}_intra_interval.txt")
    cross_interval_path = os.path.join(path, f"dev{device}_cross_interval.txt")

    with open(intra_interval_path, 'w') as f:
        # Write original phase information
        for phase, layers in moe_dynamic_intervals.items():
            # f.write(f"Phase: {phase}\n")
            # sort layers by start time
            layers = dict(sorted(layers.items(), key=lambda x: x[1]['start_time']))
            for layer, interval in layers.items():
                f.write(f"{layer} {interval['start_time']} {interval['end_time']}\n")
        
        # Write cross-layer time interval information
    with open(cross_interval_path, 'w') as f:
        f.write(f"{len(layer_in_iter0)} {len(layer_in_iter1)}\n")
        for interval in sorted(cross_layer_intervals, key=lambda x: (x['start_time'], x['iter'])):
            f.write(f"{interval['f_layer']} {interval['b_layer']} {interval['start_time']} {interval['end_time']}\n")

    # with open(cross_idx_path, 'w') as f:
    #     for layer, idx in cross_layer_id.items():
    #         f.write(f"{layer}")
    #         for i in idx:
    #             f.write(f" {i}")
    #         f.write("\n")

def make_plan_for_activations_tensors(
    MemoryPlan: _MemoryPlan,
    GlobalTensors: _GlobalTensors,
    using_group_algo: bool,
    using_fuse_mlp_algo: bool,
    using_static_reuse_algo: str = None,
):
    dynamic_fused_mlp_tensors_before_iter2 = []
    static_reuse_path = None

    if using_static_reuse_algo:
        static_reuse_dir = os.path.join(MemoryPlan.model_memory_info_dir, "output", "activations_memlog", "dynamic_intervals")
        os.makedirs(static_reuse_dir, exist_ok=True)
        get_dynamic_intervals(GlobalTensors, static_reuse_dir, MemoryPlan.device)
        static_reuse_path = static_reuse_dir

    if using_group_algo:
        iter0_static_activations_groups = get_group_tensors(
            GlobalTensors.static_activations_group_tensors, 0
        )
        iter1_static_activations_groups = get_group_tensors(
            GlobalTensors.static_activations_group_tensors, 1
        )
        static_activations_groups_before_iter2 = (
            iter0_static_activations_groups + iter1_static_activations_groups
        )

        static_activations_tensors_before_iter2 = [
            tensor
            for tensor in GlobalTensors.static_activations_temporary_tensors
            if int(tensor.start_iteration) < 2
        ]
    else:
        static_activations_groups_before_iter2 = []

        static_activations_tensors_before_iter2 = [
            tensor
            for tensor in GlobalTensors.static_activations_tensors
            if int(tensor.start_iteration) < 2
        ]
    
    iter1_start = min([int(t.start_time) for t in static_activations_tensors_before_iter2 if t.start_iteration == "1"])
    iter1_end = max([int(t.end_time) for t in static_activations_tensors_before_iter2 if t.end_iteration == "1"])

    if using_fuse_mlp_algo:
        dynamic_fused_mlp_tensors_before_iter2 = [
            tensor
            for tensor in GlobalTensors.fused_mlp_tensors
            if int(tensor.start_iteration) < 2
        ]
        static_activations_tensors_before_iter2 += (
            dynamic_fused_mlp_tensors_before_iter2
        )
        static_activations_tensors_before_iter2.sort(key=lambda x: x.start_time)

    max_allocated, max_reserved = make_plan_by_allocator(
        MemoryPlan,
        MemType.ACTIVATIONS,
        static_activations_tensors_before_iter2,
        static_activations_groups_before_iter2,
        static_reuse_path,
        iter1_start,
        iter1_end,
    )

    return max_allocated, max_reserved


def make_plan_for_cross_iteration_tensors(
    MemoryPlan: _MemoryPlan, GlobalTensors: _GlobalTensors
):
    invalid_corss_iteration_tensors_map = {}

    # count for each type of cross iteration tensors
    cross_iter_cnt = {}
    for key in list(GlobalTensors.cross_iteration_tensors_map.keys()):
        cross_iter_cnt[key] = 0
        start_iter, end_iter = key.split("->")
        if start_iter == str(GlobalTensors.iterations - 1) and end_iter == "tail":
            continue
        if (
            start_iter.isdigit()
            and end_iter.isdigit()
            and int(start_iter) + 1 == int(end_iter)
        ):
            cross_iter_cnt[key] = len(GlobalTensors.cross_iteration_tensors_map[key])
            continue

        invalid_tensors = GlobalTensors.cross_iteration_tensors_map.pop(key)
        invalid_corss_iteration_tensors_map[key] = invalid_tensors
        for tensor in invalid_tensors:
            tensor.set_memtype(MemType.UNRELEASED)
            MemoryPlan.unreleased_tensor_ids.append(tensor.malloc_id)
        _, _, info = get_tensors_memory_info(
            invalid_tensors, f"Invalid Cross Iteration Tensors of {key}"
        )
        print(info)

    if not GlobalTensors.cross_iteration_tensors_map:
        MemoryPlan.update_offset([], MemType.CROSS_ITER)
        return

    for iter in range(GlobalTensors.iterations):
        key = f"{iter}->{iter+1}"
        if iter == GlobalTensors.iterations - 1:
            key = f"{iter}->tail"
        if key not in GlobalTensors.cross_iteration_tensors_map:
            continue
        tensors = GlobalTensors.cross_iteration_tensors_map[key]
        MemoryPlan.update_offset(tensors, MemType.CROSS_ITER)


def get_group_tensors(
    static_activations_group_tensors: List[Tensor], iter: int
) -> List[GroupTensor]:
    cross_map = {}
    for tensor in static_activations_group_tensors:
        if int(tensor.start_iteration) != iter:
            continue
        key = f"{tensor.start_phase}->{tensor.end_phase}"
        cross_map.setdefault(key, []).append(tensor)

    static_activations_groups = []

    for key, tensors in cross_map.items():
        group = GroupTensor(tensors)
        assert (
            group.group_total_size == group.group_max_usage
        ), f"cross phase tensor total_size != max_usage.\n{group}"
        group.end_phase = key.split("->")[1]
        static_activations_groups.append(group)

    return static_activations_groups


def check_activation_tensor_consistency(GlobalTensors: _GlobalTensors):
    static_activations_tensors = deepcopy(GlobalTensors.static_activations_tensors)
    for _, tensors in GlobalTensors.cross_iteration_tensors_map.items():
        static_activations_tensors += tensors
    static_activations_tensors.sort(key=lambda x: x.start_time)

    if (
        GlobalTensors.config.dynamic_model
        and GlobalTensors.config.dynamic_algo == "dynamic-only"
    ):
        fused_mlp_tensors = []
    else:
        fused_mlp_tensors = deepcopy(GlobalTensors.fused_mlp_tensors)

    iter0_static_tensors = [
        tensor for tensor in static_activations_tensors if tensor.start_iteration == "0"
    ]
    iter0_static_tensors_size = [tensor.size for tensor in iter0_static_tensors]

    iter1_static_tensors = [
        tensor for tensor in static_activations_tensors if tensor.start_iteration == "1"
    ]
    iter1_static_tensors_size = [tensor.size for tensor in iter1_static_tensors]

    if iter0_static_tensors_size != iter1_static_tensors_size:
        print("Warning : static activation tensors mismatch. iter0 != iter1")

    iter1_static_and_fused_mlp_tensors = deepcopy(iter1_static_tensors)
    iter1_static_and_fused_mlp_tensors += [
        tensor for tensor in fused_mlp_tensors if tensor.start_iteration == "1"
    ]
    iter1_static_and_fused_mlp_tensors.sort(key=lambda x: x.start_time)

    for iter in range(2, GlobalTensors.iterations):
        cur_iter_static_tensors = [
            tensor
            for tensor in static_activations_tensors
            if tensor.start_iteration == str(iter)
        ]

        cur_iter_static_and_fused_mlp_tensors = deepcopy(cur_iter_static_tensors)
        cur_iter_static_and_fused_mlp_tensors += [
            tensor
            for tensor in fused_mlp_tensors
            if tensor.start_iteration == str(iter)
        ]
        cur_iter_static_and_fused_mlp_tensors.sort(key=lambda x: x.start_time)

        lens = len(cur_iter_static_and_fused_mlp_tensors)
        if iter != GlobalTensors.iterations - 1:
            assert lens == len(iter1_static_and_fused_mlp_tensors)
        else:
            assert lens <= len(iter1_static_and_fused_mlp_tensors)

        for i in range(lens):
            cur_tensor = cur_iter_static_and_fused_mlp_tensors[i]
            iter1_tensor = iter1_static_and_fused_mlp_tensors[i]
            assert cur_tensor.mem_type == iter1_tensor.mem_type
            if cur_tensor.mem_type == MemType.DYNAMIC:
                continue
            assert cur_tensor.size == iter1_tensor.size

    print("check_activation_tensor_consistency passed!")


def static_plan_validation(GlobalTensors: _GlobalTensors):
    rectangles = []
    for tensor in GlobalTensors.static_tensors:
        assert tensor.offset != -1
        rectangles.append(
            [
                tensor.start_time,
                tensor.end_time,
                tensor.offset,
                tensor.offset + roundup(tensor.size),
            ]
        )  # x1,x2,y1,y2

    def is_overlap(rect1, rect2):
        x_overlap = max(rect1[0], rect2[0]) < min(rect1[1], rect2[1])
        y_overlap = max(rect1[2], rect2[2]) < min(rect1[3], rect2[3])
        return x_overlap and y_overlap

    lens = len(rectangles)
    for i in range(lens):
        for j in range(i + 1, lens):
            if is_overlap(rectangles[i], rectangles[j]):
                raise Exception("Error : StaticAllocator Validation failed")
