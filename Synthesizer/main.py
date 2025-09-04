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

from arguments import parse_args
import multiprocessing
from trace_parser import load_trace, parse_trace
from plan_generator import generate_memory_plan
import os
from data_structures import format_size
import time

def process_dev(device, args):
    print("start to process dev : ", device)

    GlobalTensors = load_trace(device, args)
    draw_dir = os.path.join(args.model_memory_dir, "output", "image")
    os.makedirs(draw_dir, exist_ok=True)

    parser_result = parse_trace(GlobalTensors, args.draw, draw_dir, device)

    if not args.trace_only:
        max_allocated, max_reserved = generate_memory_plan(GlobalTensors)
        plan_msg = "\n[STAlloc-Plan]:\n"
        if args.dynamic_model:
            plan_msg += f"Assume : Dynamic-Allocator Utilization = {args.assume_dynamic_allocator_utilization}%, BestFit-Allocator Utilization = {args.assume_bestfit_allocator_utilization}%\n"
        plan_msg += f"max_allocated={format_size(max_allocated)}, max_reserved={format_size(max_reserved)}, utilization={(max_allocated/max_reserved):.2%}\n"
        parser_result += plan_msg

    trace_analysis_result_dir = os.path.join(args.model_memory_dir, "output", "trace_analysis")
    os.makedirs(trace_analysis_result_dir, exist_ok=True)
    with open(os.path.join(trace_analysis_result_dir, f"dev_{device}.log"), 'w') as f:
        f.write(parser_result)
    
    print(parser_result)


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()

    devices_to_process = [i for i in range(args.devices)]
    if args.device is not None:
        devices_to_process = [int(args.device)]

    processes = []
    for device in devices_to_process:
        p = multiprocessing.Process(target=process_dev, args=(device, args))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")