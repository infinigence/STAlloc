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

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # General configuration
    parser.add_argument(
        "--model-memory-dir",
        type=str,
        required=True,
        help="Equivalent to STALLOC_MODEL_INFO_PATH in your trace script",
    )

    parser.add_argument(
        "--dynamic-model",
        action="store_true",
        help="Set when model is MoE(w/o batchedgemm). Equivalent to STALLOC_DYNAMIC in your trace script",
    )

    parser.add_argument("--trace-only", action="store_true")

    parser.add_argument("--draw", action="store_true", help="Draw memory")

    parser.add_argument(
        "--devices", type=int, default=8, help="Device used num per node"
    )

    # Plan-generator configuration
    parser.add_argument(
        "--allocator-exec-path",
        type=str,
        default="./allocator.py",
        help="path to allocator.py",
    )

    parser.add_argument(
        "--static-algo",
        type=str,
        choices=["group", "split-and-fuse", "autotune"],
        default="split-and-fuse",
        help="Choose algorithm for `allocator.py`. When set to autotune, both algorithms will be run once to get the best results.",
    )

    parser.add_argument(
        "--dynamic-algo",
        type=str,
        choices=["fuse", "dynamic-only", "static-reuse", "autotune"],
        default="static-reuse",
        help="When algorithm for dynamic model. When set to autotune, both algorithms will be run once to get the best results.",
    )

    parser.add_argument(
        "--autotune",
        action="store_true",
        help="Equivalent to setting both static-algo and dyamic-algo to autotune",
    )

    parser.add_argument(
        "--validation", action="store_true", help="Validate plan of Static Memory"
    )

    # Debug configuration
    parser.add_argument("--debug", action="store_true", help="Print debug info")

    parser.add_argument(
        "--device", type=int, help="Only one of the device memory info is parsed."
    )

    # Autotune Related
    parser.add_argument(
        "--assume-dynamic-allocator-utilization", type=int, default="70"
    )
    parser.add_argument(
        "--assume-bestfit-allocator-utilization", type=int, default="90"
    )
    return parser.parse_args()
