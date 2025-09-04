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
import os
import subprocess


def allocator_call(
    allocator_exec_path: str,
    memlog_dir: str,
    device: int,
    start_addr: int,
    tensors: List[Tensor], # tensor list
    group_tensors: List[GroupTensor] = [], # static group
    static_reuse_path: str = None,
    iter1_start: int = None,
    iter1_end: int = None,
) -> Tuple[int, int]:
    os.makedirs(memlog_dir, exist_ok=True)

    # Step-1 : save memlog
    _input = os.path.join(memlog_dir, f"dev{device}.txt")
    save_tensors_memlog(tensors, _input)

    if group_tensors:
        _unallocated_tensors_path = _input
        _input = os.path.join(memlog_dir, f"dev{device}_group.txt")
        save_group_tensors_memlog(group_tensors, _input)

        command = f"python {allocator_exec_path} --group-fuse --input={_input} --unallocated-tensors-path={_unallocated_tensors_path}"
    else:
        command = f"python {allocator_exec_path}  --split-and-fuse --input={_input}"
    
    if static_reuse_path is not None:
        command += f" --static-reuse-path {static_reuse_path}/dev{device}_cross_interval.txt {static_reuse_path}/dev{device}_intra_interval.txt --reuse-offset {start_addr}"
    if iter1_start is not None and iter1_end is not None:
        command += f" --iter1-start {iter1_start} --iter1-end {iter1_end}"
    # Step-2 : call allocator
    print("Shell CMD is : \n",command)
    result = subprocess.run(
        command, shell=True, check=True, text=True, capture_output=True
    )
    print(f"Return code: {result.returncode}")
    # print("Standard Output:")
    # print(result.stdout)
    # print("Standard Error:")
    # print(result.stderr)

    # Step-3 : load plan
    plan_file_path = os.path.join(memlog_dir, f"dev{device}_allocated_plan.txt")
    if group_tensors:
        plan_file_path = os.path.join(
            memlog_dir, f"dev{device}_group_allocated_plan.txt"
        )

    return load_tensors_plan(tensors, plan_file_path, start_addr, group_tensors)


def save_memlog(
    memlog_dir: str,
    device: int,
    tensors: List[Tensor],
    group_tensors: List[GroupTensor] = [],
):
    os.makedirs(memlog_dir, exist_ok=True)
    save_tensors_memlog(tensors, os.path.join(dir, f"dev{device}.txt"))
    if group_tensors:
        save_group_tensors_memlog(
            group_tensors, os.path.join(dir, f"dev{device}_group.txt")
        )


def save_tensors_memlog(tensors: List[Tensor], file_path: str):
    content = ""
    for tensor in tensors:
        content += f"{tensor.start_time} {tensor.end_time} {tensor.size}\n"
    with open(file_path, "w") as f:
        f.write(content)


def save_group_tensors_memlog(group_tensors: List[GroupTensor], file_path: str):
    group_tensors.sort(key=lambda x: x.start_time)
    content = ""
    for group in group_tensors:
        content += (
            f"group : {group.start_time} {group.end_time} {group.group_max_usage}\n"
        )
        for tensor in group.tensors:
            content += f"{tensor.start_time} {tensor.end_time} {tensor.size}\n"

    with open(file_path, "w") as f:
        f.write(content)


def load_tensors_plan(
    tensors: List[Tensor],
    file_path: str,
    start_addr: int,
    group_tensors: List[GroupTensor] = [],
) -> Tuple[int, int]:
    for group in group_tensors:
        tensors += group.tensors
    tensors.sort(key=lambda x: x.start_time)

    with open(file_path, "r") as f:
        plans = [line.strip() for line in f]

    max_reserved = int(plans[0])

    plans = plans[1:]
    lens = len(plans)
    assert (
        len(tensors) == lens
    ), f"load tensors plan error.  tensors_num:{len(tensors)}, plans_num:{lens}"

    for i in range(lens):
        tensor = tensors[i]
        start_time, end_time, size, offset = map(
            int, plans[i].split(" ")
        )  # 1 0x7f1e41329c00 1 0

        assert offset % ALIGN == 0, f"tensors plan misalign, offset=={offset}"
        assert (
            tensor.size == size
            and tensor.start_time == start_time
            and tensor.end_time == end_time
        ), f"tensors plan mismatch. {tensor}, plan.size:{plans[i]}"

        tensor.offset = offset + start_addr

    max_allocated, _, _ = get_tensors_memory_info(tensors, "")

    for tensor in tensors:
        assert tensor.offset != -1
        # assert tensor.mem_type == MemType.ACTIVATIONS_GROUP
    return max_allocated, max_reserved
