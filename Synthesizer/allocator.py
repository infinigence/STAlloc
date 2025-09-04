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

import numpy as np
import pickle
import argparse
import heapq
import re
from copy import deepcopy
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple
from bisect import bisect_right, bisect_left
from collections import defaultdict

global_max_time = 0
iter1_start = None
iter1_end = None
iter_length = None

class NaiveSegmentTree:

    def __init__(self, start, end, max_size):
        self.start = start
        self.end = end
        self.max_size = max_size
        self.data = np.zeros(end - start + 1)

    def update(self, start, end, value):
        assert start >= self.start and end <= self.end, f"Invalid range: [{start}, {end}], valid range: [{self.start}, {self.end}]"
        assert value <= self.max_size, f"Invalid value: {value}, max_size: {self.max_size}"
        self.data[start - self.start:end - self.start + 1] = value

    def query(self, start, end):
        assert start >= self.start and end <= self.end, f"Invalid range: [{start}, {end}], valid range: [{self.start}, {self.end}]"
        return np.max(self.data[start - self.start:end - self.start + 1])


class MemoryBlock:

    def __init__(self,
                 start_time,
                 end_time,
                 size,
                 block_id,
                 original_size=None,
                 offset=None):
        self.start_time = start_time
        self.end_time = end_time
        self.offset = offset
        self.size = size
        self.original_size = original_size  # Original size of the memory block
        self.block_id = block_id  # Unique ID for each memory block
        self.cluster_id = None  # Cluster label for clustering

    def __lt__(self, other):
        return self.start_time < other.start_time

    @property
    def range(self):
        return (self.offset, self.offset + self.size)

    def __repr__(self):
        return f"MemoryBlock(id={self.block_id}, start={self.start_time}, end={self.end_time}, offset={self.offset}, size={self.size})"


class MergedMemoryBlock(MemoryBlock):

    def __init__(self,
                 start_time,
                 end_time,
                 size,
                 block_id,
                 block_list: List[MemoryBlock] = None,
                 original_size=None,
                 offset=None):
        super().__init__(start_time, end_time, size, block_id, original_size,
                         offset)
        self.block_list = block_list if block_list is not None else []
        self.located_inside = False
        self.tree = None
        self.used = False

    def __repr__(self):
        return f"MergedMemoryBlock(id={self.block_id}, start={self.start_time}, end={self.end_time}, offset={self.offset}, size={self.size}, layer_id={self.block_id})"

    def add_block(self, block):
        self.block_list.append(block)
        self.size = max(self.size, block.size)
        self.end_time = max(self.end_time, block.end_time)
        self.start_time = min(self.start_time, block.start_time)
        if self.tree is not None:
            raise ValueError("Segment tree is already initialized")

    def add_and_locate(self, block):
        if self.tree is None:
            self.tree = NaiveSegmentTree(self.start_time, self.end_time,
                                         self.size)
        current_offset = self.tree.query(block.start_time, block.end_time)
        if current_offset + block.size > self.size:
            return None
        self.block_list.append(block)
        block.offset = current_offset
        self.tree.update(block.start_time, block.end_time,
                         current_offset + block.size)
        # print(f"Allocated {block}, offset: {current_offset}, size: {block.size}, next offset: {current_offset + block.size}")
        return current_offset

    def get_segment_tree(self):
        assert self.tree is not None, "Segment tree is not initialized"
        return self.tree

    def locate(self):
        # sort by lifetime of blocks
        # relative offset to merged_block
        self.block_list.sort(key=lambda x: x.end_time - x.start_time,
                             reverse=True)
        if self.tree is None:
            self.tree = NaiveSegmentTree(self.start_time, self.end_time,
                                         self.size)
        for block in self.block_list:
            current_offset = self.tree.query(block.start_time, block.end_time)
            block.offset = current_offset
            self.tree.update(block.start_time, block.end_time,
                             current_offset + block.size)

    def update_lifetime(self, start_time, end_time):
        # print('call update lifetime')
        # print(f"old lifetime: [{self.start_time}, {self.end_time}], new lifetime: [{start_time}, {end_time}]")
        self.start_time = start_time
        self.end_time = end_time
        if self.tree is not None:
            if start_time < self.start_time:
                # expand the segment tree
                new_data = np.concatenate(
                    (np.zeros(self.start_time - start_time), self.tree.data))
            else:
                new_data = self.tree.data[self.start_time - start_time:]
            if end_time > self.end_time:
                new_data = np.concatenate(
                    (new_data, np.zeros(end_time - self.end_time)))
            else:
                new_data = new_data[:end_time - start_time + 1]
            self.tree.data = new_data
            # print("new data len:", len(self.tree.data))

    def plot(self, path="merged_block.pdf"):
        max_time = 0
        min_time = float('inf')
        max_offset = 0
        fig, ax = plt.subplots(figsize=(20, 10))
        for block in self.block_list:
            # Add a rectangle for each allocated memory block
            rect = patches.Rectangle(
                (block.start_time,
                 block.offset),  # Bottom-left coordinates (x, y)
                block.end_time - block.start_time,  # Width (duration)
                block.size,  # Height (memory size)
                linewidth=0.1,
                edgecolor='black',
                facecolor='skyblue',
                alpha=0.6)
            max_time = max(max_time, block.end_time)
            min_time = min(min_time, block.start_time)
            max_offset = max(max_offset, block.offset + block.size)
            ax.add_patch(rect)
        # Set axis labels and title
        ax.set_xlabel("Time")
        ax.set_ylabel("Memory Offset")
        ax.set_title("Memory Allocation Over Time")
        ax.set_xlim(min_time, max_time)
        ax.set_ylim(0, max_offset)
        plt.savefig(path)


class MemoryLayer:

    def __init__(self, size, layer_id=None, offset=None):
        self.size = size
        self.layer_id = layer_id
        self.offset = offset
        self.blocks = []  # list of MemoryBlock
        self.intervals = []

    def add_block(self, block):
        global iter1_start, iter1_end, iter_length, global_max_time
        # check if the block can be added to the layer
        if len(self.intervals) > 0 and check_overlap(self.intervals, [(block.start_time, block.end_time)]):
            return False
        self.blocks.append(block)
        self.intervals.append((block.start_time, block.end_time))
        if block.start_time >= iter1_start and block.start_time + iter_length <= global_max_time:
            self.intervals.append((block.start_time + iter_length, min(block.end_time + iter_length, global_max_time)))
        self.size = max(self.size, block.size)
        return True

    def locate(self, offset):
        self.offset = offset
        for block in self.blocks:
            block.offset = offset

    def form_segment_tree(self):
        global global_max_time
        data = np.zeros(global_max_time + 1)
        tree = NaiveSegmentTree(0, global_max_time, self.size)
        for block in self.blocks:
            if isinstance(block, MergedMemoryBlock):
                sub_tree = block.get_segment_tree()
                # print(len(sub_tree.data), block.end_time - block.start_time)
                data[block.start_time - 1:block.end_time] = np.maximum(
                    data[block.start_time - 1:block.end_time], sub_tree.data)
            else:
                data[block.start_time - 1:block.end_time] = np.maximum(
                    data[block.start_time - 1:block.end_time], block.size)
        tree.data = data
        return tree

    def __repr__(self):
        return f"MemoryLayer(id={self.layer_id}, size={self.size}, offset={self.offset}, intervals={self.intervals})"


class MergedMemoryLayer:

    def __init__(self, size, layer_id=None, offset=None, layers=None):
        self.size = size
        self.layer_id = layer_id
        self.offset = offset
        self.layers = [] if layers == None else layers  # list of MemoryLayer
        self.relative_offset = [] if layers == None else [
            0
        ]  # list of relative offset

    def add_layer(self, layer, relative_offset):
        self.layers.append(layer)
        self.relative_offset.append(relative_offset)

    def locate(self, offset):
        self.offset = offset
        for i, layer in enumerate(self.layers):
            layer.locate(offset + self.relative_offset[i])

    def get_active_intervals(self, offset_start, size):
        if offset_start + size > self.size:
            return None
        candidate_intervals = []
        for i, layer in enumerate(self.layers):
            if self.relative_offset[
                    i] >= offset_start + size or self.relative_offset[
                        i] + layer.size <= offset_start:
                continue
            candidate_intervals.extend(layer.intervals)
        
        # merge intervals
        candidate_intervals.sort(key=lambda x: x[0])
        merged_intervals = []
        for interval in candidate_intervals:
            if len(merged_intervals
                   ) == 0 or interval[0] > merged_intervals[-1][1]:
                merged_intervals.append(interval)
            else:
                merged_intervals[-1] = (merged_intervals[-1][0],
                                        max(merged_intervals[-1][1],
                                            interval[1]))
        return merged_intervals

    def form_segment_tree(self):
        global global_max_time
        self.tree = NaiveSegmentTree(0, global_max_time, self.size)
        for i, layer in enumerate(self.layers):
            subtree = layer.form_segment_tree()
            self.tree.data = np.maximum(self.tree.data,
                                        subtree.data + self.relative_offset[i])

    def __repr__(self):
        return f"MergedMemoryLayer(id={self.layer_id}, size={self.size}, offset={self.offset}, layers={self.layers})"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process log file to generate memory trace")
    parser.add_argument("--input",
                        type=str,
                        default='local/mem_logs/mem_log_dev0.pkl',
                        help="Path to the log file")
    parser.add_argument("--unallocated-tensors-path",
                        type=str,
                        required=False,
                        default=None,
                        help="Path to the unallocated tensors file")
    parser.add_argument("--output",
                        type=str,
                        required=False,
                        help="Path to the output file")
    parser.add_argument("--plot-footprint",
                        action="store_true",
                        help="Plot the memory footprint")
    parser.add_argument("--split-and-fuse",
                        action="store_true",
                        help="Split and fuse the memory blocks")
    parser.add_argument("--group-fuse",
                        action="store_true",
                        help="Group and fuse the memory blocks")
    parser.add_argument("--fused-dynamic-path",
                        type=str,
                        required=False,
                        default=None,
                        help="Path to the fused dynamic file")
    parser.add_argument("--static-reuse-path",
                        type=str,
                        required=False,
                        default=None,
                        nargs='+',
                        help="Path to the static reuse file")
    parser.add_argument("--reuse-offset",
                        type=int,
                        required=False,
                        default=0,
                        help="Reuse offset")
    parser.add_argument("--iter1-start",
                        type=int,
                        required=False,
                        default=None,
                        help="Start iteration of iter1")
    parser.add_argument("--iter1-end",
                        type=int,
                        required=False,
                        default=None,
                        help="End iteration of iter1")
    return parser.parse_args()


def load_from_file(path, plot_footprint=False):
    global global_max_time
    # load from .pkl
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    # load from .txt
    elif path.endswith(".txt"):
        blocks = []
        events = []
        with open(path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # read to 'int'
                start_time, end_time, size = line.split()
                start_time, end_time, size = int(start_time), int(
                    end_time), int(size)
                global_max_time = max(global_max_time, end_time)
                aligned_size = (size + 511) // 512 * 512
                block = MemoryBlock(start_time, end_time, aligned_size, i,
                                    size)
                blocks.append(block)
                if plot_footprint:
                    events.append((start_time, size))
                    events.append((end_time, -size))
        if plot_footprint:
            max_mem = 0
            max_mem_time = []
            events.sort(key=lambda x: x[0])
            time = [x[0] for x in events]
            time.insert(0, 0)
            footprint = [0]
            for event in events:
                footprint.append(footprint[-1] + event[1])
                if footprint[-1] > max_mem:
                    max_mem = footprint[-1]
                    max_mem_time = [event[0]]
                elif footprint[-1] == max_mem:
                    max_mem_time.append(event[0])
            plt.figure(figsize=(12, 10))
            plt.step(time, footprint, where='post')
            plt.title('Memory Footprint')
            plt.xlabel('Time')
            plt.ylabel('Memory Footprint')
            plt.savefig("memory_footprint.pdf")
            print("Max memory footprint:", max_mem / (1024**3))
            print("Max memory footprint time:", max_mem_time)
        return blocks
    else:
        raise ValueError("Invalid file format")


def plot_memory_footprint(blocks, path="memory_footprint.pdf"):
    max_mem = 0
    max_mem_time = []
    events = []
    for block in blocks:
        events.append((block.start_time, block.size))
        events.append((block.end_time, -block.size))
    events.sort(key=lambda x: x[0])
    time = [x[0] for x in events]
    time.insert(0, 0)
    footprint = [0]
    for event in events:
        footprint.append(footprint[-1] + event[1])
        if footprint[-1] > max_mem:
            max_mem = footprint[-1]
            max_mem_time = [event[0]]
        elif footprint[-1] == max_mem:
            max_mem_time.append(event[0])
    plt.figure(figsize=(12, 10))
    plt.step(time, footprint, where='post')
    plt.title('Memory Footprint')
    plt.xlabel('Time')
    plt.ylabel('Memory Footprint')
    plt.savefig(path)
    print("Max memory footprint:", max_mem / (1024**3))
    print("Max memory footprint time:", max_mem_time)
    return max_mem_time


def kmeans_cluster(blocks):
    X = np.array([[block.size] for block in blocks])
    # set the n_cluster to be the number of distinct size
    n_clusters = min(len(set([block.size for block in blocks])), 100)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    for i, block in enumerate(blocks):
        block.cluster_id = kmeans.labels_[i]
    plot_cluster(blocks, path="kmeans_cluster.pdf")
    return kmeans.labels_

def cluster_blocks(blocks):
    # cluster the blocks by size
    clusters = defaultdict(list)
    for block in blocks:
        clusters[block.size].append(block)
        block.cluster_id = block.size

def relative_distance(a, b):
    return abs(a - b) / (a + b)


def dbscan_cluster(blocks, eps=0.01, min_samples=1):
    X = np.array([[block.size] for block in blocks])
    db = DBSCAN(eps=eps, min_samples=min_samples,
                metric=relative_distance).fit(X)
    for i, block in enumerate(blocks):
        block.cluster_id = db.labels_[i]
    plot_cluster(blocks, path="dbscan_cluster.pdf")
    return db.labels_


def plot_cluster(blocks, path="cluster.pdf"):
    tensor_sizes = np.array([tensor.size for tensor in blocks])
    labels = np.array([tensor.cluster_id
                       for tensor in blocks])  # cluster label
    plt.figure(figsize=(12, 10))

    for label in np.unique(labels):
        cluster_points = tensor_sizes[labels == label]
        indices = np.where(labels == label)[0]
        plt.scatter(indices, cluster_points, label=f'Cluster {label}', s=50)

    plt.title('Tensor Size vs Cluster Labels')
    plt.xlabel('Index')
    plt.ylabel('Tensor Size')
    # y-axis in log level
    plt.yscale('log')
    plt.legend()
    plt.savefig(path)


def form_layer(cluster, layer_id):
    # the blocks in the cluster are sorted by start_time
    layers = []
    tail = []
    for block in cluster:
        placed = False
        # find the layer with maximum tail idx and smaller than the block.start_time
        candidates = [(i, x) for i, x in enumerate(tail)
                      if x <= block.start_time]
        if len(candidates) > 0:
            # get sorted index of tail
            idx_list = [x[0] for x in sorted(candidates, key=lambda x: x[1], reverse=True)]
            for idx in idx_list:
                placed = layers[idx].add_block(block)
                if placed:
                    tail[idx] = block.end_time
                    break
            if placed:
                continue

        new_layer = MemoryLayer(size=block.size, layer_id=layer_id)
        new_layer.add_block(block)
        layers.append(new_layer)
        tail.append(block.end_time)
    return layers


def check_overlap(interval_list1, interval_list2):
    i, j = 0, 0
    while i < len(interval_list1) and j < len(interval_list2):
        if interval_list1[i][1] <= interval_list2[j][0]:
            i += 1
        elif interval_list2[j][1] <= interval_list1[i][0]:
            j += 1
        else:
            return True
    return False


def find_conflicting_indices_sorted(list1, list2):
    '''
    list1 is the intervals of base layer, find the conflict intervals in list2
    '''
    conflicting_indices = []
    i, j = 0, 0  # double pointers

    while i < len(list1) and j < len(list2):
        start1, end1 = list1[i]
        start2, end2 = list2[j]

        if max(start1, start2) < min(end1, end2):
            conflicting_indices.append(j)
            j += 1
        else:
            # move the pointer with smaller end time
            if end1 < end2:
                i += 1
            else:
                j += 1

    return conflicting_indices


def merge_interval_list(interval_list1, interval_list2):
    i, j = 0, 0
    merged = []
    while i < len(interval_list1) and j < len(interval_list2):
        if interval_list1[i][1] <= interval_list2[j][0]:
            merged.append(interval_list1[i])
            i += 1
        elif interval_list2[j][1] <= interval_list1[i][0]:
            merged.append(interval_list2[j])
            j += 1
        else:
            merged.append((min(interval_list1[i][0], interval_list2[j][0]),
                           max(interval_list1[i][1], interval_list2[j][1])))
            i += 1
            j += 1
    while i < len(interval_list1):
        merged.append(interval_list1[i])
        i += 1
    while j < len(interval_list2):
        merged.append(interval_list2[j])
        j += 1
    return merged


def split_layer(layer, new_idx: List):
    new_layer = MemoryLayer(size=layer.size, layer_id=layer.layer_id + 100)
    new_layer.blocks = [layer.blocks[i] for i in new_idx]
    new_layer.intervals = [layer.intervals[i] for i in new_idx]
    # delete the blocks from the original layer
    for i in sorted(new_idx, reverse=True):
        del layer.blocks[i]
        del layer.intervals[i]
    return new_layer


def allocate(blocks, cluster_method="kmeans", **kwargs):
    '''
    Deprecated: use allocate_group instead
    '''
    # Step 1: Clustering
    cluster_blocks(blocks)
    clusters = {}
    for i, block in enumerate(blocks):
        if block.cluster_id not in clusters:
            clusters[block.cluster_id] = []
        heapq.heappush(clusters[block.cluster_id], block)

    # Step 2: Form layers
    layers = []
    for cluster_id, cluster in clusters.items():
        layers.extend(form_layer(cluster, cluster_id))
    print(len(layers))
    # sort layers in descending order of size
    layers.sort(key=lambda x: x.size, reverse=True)

    offset = 0
    for layer in layers:
        layer.locate(offset)
        offset += layer.size

    # Step 3: Merge layers
    min_size = layers[-1].size

    def merge(merged_layer,
              base_layer,
              intervals,
              layers,
              relative_offset,
              start_idx=0,
              **kwargs):
        split = kwargs.get("split", False)
        available_size = base_layer.size
        i = start_idx
        while True:
            if i >= len(layers):
                break
            layer = layers[i]
            if layer is None:
                i += 1
                continue
            if layer.size > available_size:
                # print("No enough space")
                i += 1
                continue
            if layer.layer_id == base_layer.layer_id:
                # print("Same layer id")
                i += 1
                continue
            conflicts = find_conflicting_indices_sorted(
                intervals, layer.intervals)
            if len(conflicts) == 0:
                # print("Merge all")
                merged_intervals = merge_interval_list(intervals,
                                                       layer.intervals)
                merged_layer.layers.append(layer)
                merged_layer.relative_offset.append(relative_offset)
                layers[i] = None
                merge(merged_layer, layer, merged_intervals, layers,
                      relative_offset, i + 1, **kwargs)
                relative_offset += layer.size
                available_size -= layer.size
                i += 1
            elif len(conflicts) == len(layer.blocks):
                # no need to merge
                i += 1
                continue
            else:
                # split then merge
                if split:
                    new_layer = split_layer(
                        layer, conflicts)  # layer with conflict intervals
                    merged_intervals = merge_interval_list(
                        intervals, layer.intervals)
                    merged_layer.layers.append(layer)
                    merged_layer.relative_offset.append(relative_offset)
                    # set layer[i]=None and add new_layer to layers
                    layers[i] = None
                    layers.insert(i + 1, new_layer)
                    merge(merged_layer, layer, merged_intervals, layers,
                          relative_offset, i + 1, **kwargs)
                    relative_offset += layer.size
                    available_size -= layer.size
                    i += 2
                else:
                    i += 1
                    continue

    merged_layers = []
    for i, layer in enumerate(layers):
        if layer is None:
            continue
        # init merged layer
        merged_layer = MergedMemoryLayer(size=layer.size)
        merged_layer.layers.append(layer)
        merged_layer.relative_offset.append(0)
        layers[i] = None
        merge(merged_layer,
              base_layer=layer,
              intervals=layer.intervals,
              layers=layers,
              relative_offset=0,
              start_idx=i + 1,
              **kwargs)
        # print(merged_layer.layers)
        merged_layers.append(merged_layer)
    

    offset = 0
    result = []
    for merged_layer in merged_layers:
        merged_layer.offset = offset
        offset += max(merged_layer.size, 512)
        for i, layer in enumerate(merged_layer.layers):
            layer_offset = merged_layer.offset + merged_layer.relative_offset[i]
            assert layer_offset % 512 == 0, f"layer_offset={layer_offset} should be aligned to 512"
            layer.locate(layer_offset)
            result.extend(layer.blocks)
    plot_memory_allocation_after_merge(merged_layers)
    print('peak memory usage:', offset / (1024**3))
    result.sort(key=lambda x: x.start_time)
    return result, offset


# tmp function
def read_block_group_from_file(path):
    global global_max_time
    block_group = []
    merge_group = [[], []]
    pattern = re.compile(
        r"(group.*:|merged group\[\d+\]:)\s+(\d+)\s+(\d+)\s+(\d+)")
    i = -1
    with open(path, "r") as f:
        for line in f:
            line_match = pattern.match(line)
            if line_match:
                i += 1
                group_type = line_match.group(1).strip()
                group_start, group_end, group_size = map(
                    int,
                    line_match.groups()[1:])
                global_max_time = max(global_max_time, group_end)
                current_group = MergedMemoryBlock(start_time=group_start,
                                                  end_time=group_end,
                                                  size=0,
                                                  block_id=i)

                if group_type == "group :":
                    block_group.append(current_group)
                elif '0' in group_type:
                    merge_group[0].append(current_group)
                elif '1' in group_type:
                    merge_group[1].append(current_group)
                else:
                    raise ValueError(f"Invalid group type: {group_type}")
            else:
                block_start, block_end, block_size = map(
                    int,
                    line.strip().split())
                aligned_size = (block_size + 511) // 512 * 512
                block = MemoryBlock(start_time=block_start,
                                    end_time=block_end,
                                    size=aligned_size,
                                    block_id=i,
                                    original_size=block_size)
                assert block.start_time >= current_group.start_time and block.end_time <= current_group.end_time, f"block {block} should be in the group {current_group}"
                assert block.block_id == current_group.block_id, f"block_id={block.block_id} should be equal to current_group.block_id={current_group.block_id}"
                current_group.size += block.size
                current_group.add_block(block)

    return block_group, merge_group


def merge_block_group(group1: MergedMemoryBlock, group2: MergedMemoryBlock):
    # check two groups is overlaped but not completely overlap, and find which one is the former or latter
    if group1.end_time <= group2.start_time or group2.end_time <= group1.start_time:
        return None
    elif group1.start_time <= group2.start_time:
        if group1.end_time <= group2.end_time:
            group1_former = True
        else:
            return None
    else:
        if group2.end_time <= group1.end_time:
            group1_former = False
        else:
            return None
    
    # find the base group based on size, if size equal, choose the one with longer lifetime
    if group1.size > group2.size:
        base_group = group1
        merge_group = group2
        base_size = group1.size
        merge_size = group2.size
        base_former = group1_former
    elif group1.size < group2.size:
        base_group = group2
        merge_group = group1
        base_size = group2.size
        merge_size = group1.size
        base_former = not group1_former
    else:
        base_size = group1.size
        merge_size = group2.size
        if group1.end_time - group1.start_time > group2.end_time - group2.start_time:
            base_group = group1
            merge_group = group2
            base_former = group1_former
        else:
            base_group = group2
            merge_group = group1
            base_former = not group1_former
    
    start_time = min(group1.start_time, group2.start_time)
    end_time = max(group1.end_time, group2.end_time)
    usage_base = sum([
        block.size * (block.end_time - block.start_time)
        for block in base_group.block_list
    ]) / (base_size * (base_group.end_time - base_group.start_time))
    usage_merge = sum([
        block.size * (block.end_time - block.start_time)
        for block in merge_group.block_list
    ]) / (merge_size * (merge_group.end_time - merge_group.start_time))
    merged_group = MergedMemoryBlock(start_time=start_time,
                                     end_time=end_time,
                                     size=base_size,
                                     block_id=base_group.block_id)
    if base_former:
        base_blocks = sorted(base_group.block_list,
                             key=lambda x: x.end_time,
                             reverse=True)
        base_blocks_offset = []
        for block in base_blocks:
            base_blocks_offset.append(merged_group.add_and_locate(block))
        # add block in the merge group
        merge_blocks = sorted(merge_group.block_list,
                              key=lambda x: x.start_time,
                              reverse=False)
        current_offset = 0
        while len(merge_blocks) > 0:
            # find the idx of base blocks with maximum offset lower than current_offset
            base_block_idx = bisect_right(base_blocks_offset,
                                          current_offset) - 1
            assert base_block_idx >= 0, f"wrong offset: {current_offset}"
            current_end = base_blocks[base_block_idx].end_time
            # find the one merge block with minimum start time and lager than current_end
            merge = False
            for block in merge_blocks:
                if block.start_time >= current_end and block.size + current_offset <= base_size:
                    offset = merged_group.add_and_locate(block)
                    if offset is None:
                        continue
                    current_offset = offset + block.size
                    assert current_offset <= base_size, f"current_offset={current_offset} should be less than size={base_size}"
                    merge_blocks.remove(block)
                    merge = True
                    break
            if not merge:
                if base_block_idx == len(base_blocks_offset) - 1:
                    res_size = sum([block.size for block in merge_blocks])
                    res_ratio = res_size / merge_size
                    usage_res = sum([
                        block.size * (block.end_time - block.start_time)
                        for block in merge_blocks
                    ]) / (res_size *
                          (max([block.end_time for block in merge_blocks]) -
                           min([block.start_time for block in merge_blocks])))
                    # merged_group.end_time = max([block.end_time for block in merged_group.block_list])
                    merged_group.update_lifetime(
                        merged_group.start_time,
                        max([
                            block.end_time for block in merged_group.block_list
                        ]))
                    usage_total = sum([
                        block.size * (block.end_time - block.start_time)
                        for block in merged_group.block_list
                    ]) / (merged_group.size *
                          (merged_group.end_time - merged_group.start_time))

                    if usage_total * usage_res > usage_base * usage_merge:
                        # reserved the merged block
                        res_block = MergedMemoryBlock(
                            start_time=min(
                                [block.start_time for block in merge_blocks]),
                            end_time=max(
                                [block.end_time for block in merge_blocks]),
                            size=res_size,
                            block_id=merge_group.block_id,
                            block_list=merge_blocks)
                        res_block.locate()
                        return merged_group, res_block
                    else:
                        return None
                current_offset = base_blocks_offset[base_block_idx + 1]
    else:
        base_blocks = sorted(base_group.block_list,
                             key=lambda x: x.start_time,
                             reverse=False)

        base_blocks_offset = []
        for block in base_blocks:
            base_blocks_offset.append(merged_group.add_and_locate(block))
        
        # add block in the merge group
        merge_blocks = sorted(merge_group.block_list,
                              key=lambda x: x.end_time,
                              reverse=True)
        current_offset = 0
        while len(merge_blocks) > 0:
            # find the idx of base blocks with maximum offset lower than current_offset
            base_block_idx = bisect_right(base_blocks_offset,
                                          current_offset) - 1
            assert base_block_idx >= 0, f"wrong offset: {current_offset}"
            current_start = base_blocks[base_block_idx].start_time

            # find the one merge block with maximum end time and smaller than current_start
            merge = False
            for block in merge_blocks:
                if block.end_time <= current_start:
                    offset = merged_group.add_and_locate(block)
                    if offset is None:
                        continue
                    current_offset = offset + block.size
                    assert current_offset <= base_size, f"current_offset={current_offset} should be less than size={base_size}"
                    merge_blocks.remove(block)
                    merge = True
                    break

            if not merge:
                if base_block_idx == len(base_blocks_offset) - 1:
                    res_size = sum([block.size for block in merge_blocks])
                    res_ratio = res_size / merge_size
                    usage_res = sum([
                        block.size * (block.end_time - block.start_time)
                        for block in merge_blocks
                    ]) / (res_size *
                          (max([block.end_time for block in merge_blocks]) -
                           min([block.start_time for block in merge_blocks])))
                    # merged_group.start_time = min([block.start_time for block in merged_group.block_list])
                    merged_group.update_lifetime(
                        min([
                            block.start_time
                            for block in merged_group.block_list
                        ]), merged_group.end_time)
                    usage_total = sum([
                        block.size * (block.end_time - block.start_time)
                        for block in merged_group.block_list
                    ]) / (merged_group.size *
                          (merged_group.end_time - merged_group.start_time))

                    if usage_total * usage_res > usage_base * usage_merge:
                        # reserved the merged block
                        res_block = MergedMemoryBlock(
                            start_time=min(
                                [block.start_time for block in merge_blocks]),
                            end_time=max(
                                [block.end_time for block in merge_blocks]),
                            size=res_size,
                            block_id=merge_group.block_id,
                            block_list=merge_blocks)
                        res_block.locate()
                        return merged_group, res_block
                    else:
                        return None
                current_offset = base_blocks_offset[base_block_idx + 1]

    usage_total = sum([
        block.size * (block.end_time - block.start_time)
        for block in merged_group.block_list
    ]) / (base_size * (end_time - start_time))
    if usage_total * usage_total > usage_base * usage_merge:
        return merged_group
    else:
        return None


def merge_candidates(merged_layer, blocks):
    if len(blocks) == 0:
        return merged_layer, blocks
    block_size = max([block.size for block in blocks])
    block_id = blocks[0].block_id
    blocks = sorted(blocks, key=lambda x: x.start_time)
    new_relative_offset = 0
    for i, layer in enumerate(merged_layer.layers):
        if layer.size >= block_size:
            current_offset = merged_layer.relative_offset[i]
            if merged_layer.relative_offset[
                    i] + layer.size < merged_layer.size - block_size:
                new_relative_offset = max(
                    new_relative_offset,
                    merged_layer.relative_offset[i] + layer.size)
            while current_offset + block_size <= layer.size:
                active_intervals = merged_layer.get_active_intervals(
                    current_offset, block_size)
                if active_intervals is not None:
                    # get candidate blocks that is not overlapped with active_intervals
                    merge_block = []
                    for j, block in enumerate(blocks):
                        if block is not None and (not check_overlap(
                                active_intervals,
                            [(block.start_time, block.end_time)])):
                            merge_block.append(block)
                            blocks[j] = None
                            active_intervals = merge_interval_list(
                                active_intervals,
                                [(block.start_time, block.end_time)])
                    if len(merge_block) > 0:
                        # print('form new layer and merge, current offset:', current_offset)
                        new_layer = MemoryLayer(size=block_size,
                                                layer_id=block_id)
                        for block in merge_block:
                            new_layer.add_block(block)
                        merged_layer.add_layer(new_layer, current_offset)
                        current_offset += block_size
                    else:
                        break
    if new_relative_offset + block_size <= merged_layer.size:
        current_offset = new_relative_offset
        while current_offset + block_size <= merged_layer.size:
            active_intervals = merged_layer.get_active_intervals(
                current_offset, block_size)

            if active_intervals is not None:
                # get candidate blocks that is not overlapped with active_intervals
                merge_block = []
                for j, block in enumerate(blocks):
                    if block is not None and (not check_overlap(
                            active_intervals,
                        [(block.start_time, block.end_time)])):
                        merge_block.append(block)
                        blocks[j] = None
                        active_intervals = merge_interval_list(
                            active_intervals,
                            [(block.start_time, block.end_time)])
                if len(merge_block) > 0:
                    new_layer = MemoryLayer(size=block_size, layer_id=block_id)
                    for block in merge_block:
                        new_layer.add_block(block)
                    merged_layer.add_layer(new_layer, current_offset)
                    current_offset += block_size
                else:
                    break
    return merged_layer, blocks

def allocate_group(block_info, unallocated_blocks, cluster_method="kmeans", group=True):
    if group:
        block_group, _ = block_info
        max_group_size = max([block.size for block in block_group])
        for i, group in enumerate(block_group):
            if group.size < max_group_size * 0.1 and len(group.block_list) < 5:
                for block in group.block_list:
                    unallocated_blocks.append(block)
                block_group[i] = None
        block_group = [block for block in block_group if block is not None]
        min_group_size = min([block.size for block in block_group])

        # move the block bigger than min_group_size into block_group
        for block in unallocated_blocks:
            if block.size >= min_group_size:
                merged_block = MergedMemoryBlock(start_time=block.start_time,
                                                end_time=block.end_time,
                                                size=block.size,
                                                block_id=block.block_id)
                merged_block.add_block(block)
                block_group.append(merged_block)
                unallocated_blocks.remove(block)

        # check for block groups
        for i, group in enumerate(block_group):
            usage = sum([
                block.size * (block.end_time - block.start_time)
                for block in group.block_list
            ]) / (group.size * (group.end_time - group.start_time))
            if usage < 0.7:
                # split into sinple blocks
                unallocated_blocks.extend(group.block_list)
                block_group[i] = None
        block_group = [group for group in block_group if group is not None]
        if len(block_group) == 0:
            kwargs = {}
            kwargs['split'] = True
            print('Groups are not good enough, call split and fuse')
            return allocate_group(None, unallocated_blocks, group=False)

        # cluster block_groups
        cluster_blocks(block_group)
        clusters = {}
        for i, block in enumerate(block_group):
            if block.cluster_id not in clusters:
                clusters[block.cluster_id] = []
            heapq.heappush(clusters[block.cluster_id], block)
        
        # sort clusters by average size in descending order
        cluster_size = {}
        for cluster_id, cluster in clusters.items():
            cluster_size[cluster_id] = np.mean([block.size for block in cluster])
        sorted_clusters = sorted(cluster_size.items(),
                                key=lambda x: x[1],
                                reverse=True)

        merged_group = []
        for i in range(0, len(sorted_clusters)):
            merge_group_id = sorted_clusters[i][0]
            for merge_block in clusters[merge_group_id]:
                if merge_block.used:
                    continue
                for j in range(0, i):
                    base_group_id = sorted_clusters[j][0]
                    for base_block in clusters[base_group_id]:
                        if base_block.used:
                            continue
                        result_block = merge_block_group(base_block, merge_block)
                        if result_block is not None:
                            base_block.used = True
                            merge_block.used = True
                            if isinstance(result_block, tuple):
                                merged_group.extend(result_block)
                            else:
                                merged_group.append(result_block)

        for cluster_id, cluster in clusters.items():
            for block in cluster:
                if not block.used:
                    block.locate()
                    merged_group.append(block)

        # Step 2: cluster block_groups
        # cluster block_groups
        # merged_group.extend(unallocated_blocks)
        cluster_blocks(merged_group)
        clusters = {}
        for i, block in enumerate(merged_group):
            if block.cluster_id not in clusters:
                clusters[block.cluster_id] = []
            heapq.heappush(clusters[block.cluster_id], block)

        # sort clusters by average size in descending order
        cluster_size = {}
        for cluster_id, cluster in clusters.items():
            cluster_size[cluster_id] = np.mean([block.size for block in cluster])
        sorted_clusters = sorted(cluster_size.items(),
                                key=lambda x: x[1],
                                reverse=True)

        # Step 3: Form layers
        layers = []
        for i in range(len(sorted_clusters)):
            cluster_id = sorted_clusters[i][0]
            cluster = clusters[cluster_id]

            # Step 3.1: Insertion in foermer layers
            for i, layer in enumerate(layers):
                layers[i], cluster = merge_candidates(layer, cluster)
                # delete the None in cluster
                cluster = [block for block in cluster if block is not None]
                if len(cluster) == 0:
                    break

            # Step 3.2: Form new layers
            new_layers = form_layer(cluster, cluster_id)
            # convert to MergedLayer
            merged_layers = [
                MergedMemoryLayer(size=layer.size,
                                layer_id=layer.layer_id,
                                layers=[layer]) for layer in new_layers
            ]
            layers.extend(merged_layers)

        # sort layers in descending order of size
        layers.sort(key=lambda x: x.size, reverse=True)

        offset = 0
        current_blocks = 0
        for layer in layers:
            layer.locate(offset)
            offset += layer.size
            layer.form_segment_tree()
            for single_layer in layer.layers:
                for block in single_layer.blocks:
                    if isinstance(block, MergedMemoryBlock):
                        current_blocks += len(block.block_list)
                    else:
                        current_blocks += 1
    else:
        layers = []

    # Step 4: allocate rest tensors
    cluster_blocks(unallocated_blocks)
    clusters = {}
    for i, block in enumerate(unallocated_blocks):
        if block.cluster_id not in clusters:
            clusters[block.cluster_id] = []
        heapq.heappush(clusters[block.cluster_id], block)

    # sort clusters by average size in descending order
    cluster_size = {}
    for cluster_id, cluster in clusters.items():
        cluster_size[cluster_id] = np.mean([block.size for block in cluster])
    sorted_clusters = sorted(cluster_size.items(),
                             key=lambda x: x[1],
                             reverse=True)
    located_blocks = []
    for i in range(0, len(sorted_clusters)):
        cluster_id = sorted_clusters[i][0]
        blocks = clusters[cluster_id]
        # sort block in layer schedule
        origin_layers = form_layer(blocks, cluster_id)
        blocks = []
        for layer in origin_layers:
            blocks.extend(layer.blocks)
        block_allocate_order = list(range(len(blocks)))
        min_extra_size = float('inf')
        repeat_cnt = 0

        # pre-allocation
        while True:
            test_blocks = deepcopy(blocks)
            order_copy = deepcopy(block_allocate_order)
            for layer in layers:
                layer_copy = deepcopy(layer)
                for j, idx in enumerate(block_allocate_order):
                    block = test_blocks[idx]
                    if block is not None and layer.size >= block.size:
                        current_offset = layer_copy.tree.query(
                            block.start_time, block.end_time)
                        if current_offset + block.size <= layer.size:
                            layer_copy.tree.update(block.start_time,
                                                   block.end_time,
                                                   current_offset + block.size)
                            order_copy[j] = None

            # form new layers
            # delete None in blocks
            indices = list(
                filter(lambda i: order_copy[i] is not None,
                       range(len(order_copy))))
            test_blocks = sorted([test_blocks[i] for i in indices],
                                 key=lambda x: x.start_time)
            new_layers = form_layer(blocks, cluster_id)
            extra_size = sum([layer.size for layer in new_layers])

            if extra_size <= min_extra_size:
                # reorder the blocks
                block_allocate_order = indices + [
                    i for i in block_allocate_order if i not in indices
                ]
                repeat_cnt = repeat_cnt + 1 if extra_size == min_extra_size else 0
                min_extra_size = extra_size

                if extra_size == 0 or repeat_cnt >= 5:
                    break
            else:
                break

        # real allocation
        for layer in layers:
            for j, idx in enumerate(block_allocate_order):
                block = blocks[idx]
                if block is not None and layer.size >= block.size:
                    current_offset = layer.tree.query(block.start_time,
                                                      block.end_time)
                    if current_offset + block.size <= layer.size:
                        block.offset = current_offset + layer.offset
                        located_blocks.append(block)
                        layer.tree.update(block.start_time, block.end_time,
                                          current_offset + block.size)
                        blocks[idx] = None

        # form new layers
        # delete None in blocks
        indices = list(
            filter(lambda i: blocks[i] is not None, range(len(blocks))))
        blocks = sorted([blocks[i] for i in indices],
                        key=lambda x: x.start_time)
        new_layers = form_layer(blocks, cluster_id)
        merged_layers = [
            MergedMemoryLayer(size=layer.size,
                              layer_id=layer.layer_id,
                              layers=[layer]) for layer in new_layers
        ]
        for layer in merged_layers:
            if len(layers) == 0:
                offset = 0
            else:
                offset = layers[-1].offset + layers[-1].size
            layer.locate(offset)
            layer.form_segment_tree()
            layers.append(layer)

    peak_memory = int(layers[-1].offset + layers[-1].size)
    print('peak memory usage:', peak_memory / (1024**3))

    # write result
    result = []
    for merged_layer in layers:
        for layer in merged_layer.layers:
            for block in layer.blocks:
                if isinstance(block, MergedMemoryBlock):
                    for base_block in block.block_list:
                        base_block.offset += block.offset
                    result.extend(block.block_list)
                else:
                    # block.offset += layer.offset
                    result.append(block)
    result.extend(located_blocks)
    result.sort(key=lambda x: x.start_time)
    return result, peak_memory


def plot_memory_allocation(layers, path="memory_allocation.pdf"):
    max_time = 0
    max_offset = 0
    fig, ax = plt.subplots(figsize=(20, 10))
    for layer in layers:
        for block in layer.blocks:
            for base_block in block.block_list:
                # Add a rectangle for each allocated memory block
                rect = patches.Rectangle(
                    (base_block.start_time, block.offset +
                     base_block.offset),  # Bottom-left coordinates (x, y)
                    base_block.end_time -
                    base_block.start_time,  # Width (duration)
                    base_block.size,  # Height (memory size)
                    linewidth=0.1,
                    edgecolor='black',
                    facecolor='skyblue',
                    alpha=0.6)
                max_time = max(max_time, block.end_time)
                ax.add_patch(rect)
            max_offset = max(max_offset, block.offset + block.size)
    # Set axis labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Memory Offset")
    ax.set_title("Memory Allocation Over Time")
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, max_offset + 1000)
    plt.savefig(path)

def plot_memory_allocation_after_merge(merged_layers,
                                       located_blocks=None,
                                       path=None):
    args = parse_args()
    global global_max_time, iter1_start, iter1_end, iter_length

    if path is None:
        if args.input.endswith(".pkl"):
            path = args.input.replace(".pkl", "_allocated_plan.pdf")
        elif args.input.endswith(".txt"):
            path = args.input.replace(".txt", "_allocated_plan.pdf")
        else:
            path = "memory_allocation_after_merge.pdf"
    if args.split_and_fuse:
        path = path.replace(".pdf", "_sf.pdf")
    max_time = 0
    max_offset = 0
    num_blocks = 0
    fig, ax = plt.subplots(figsize=(20, 10))
    color_list = [
        'skyblue', 'lightcoral', 'lightgreen', 'lightpink', 'red',
        'lightyellow', 'lightgrey', 'lightseagreen', 'orange', 'lightcyan',
        'lightgoldenrodyellow', 'lightgreen', 'lightgrey', 'lightpink',
        'lightsalmon', 'lightseagreen', 'purple', 'lightslategray',
        'lightsteelblue', 'lightyellow'
    ]
    for merged_layer in merged_layers:
        for i, layer in enumerate(merged_layer.layers):
            # determine the color based on layer_id
            color = color_list[layer.layer_id % len(color_list)]
            for block in layer.blocks:
                if isinstance(block, MergedMemoryBlock):
                    for base_block in block.block_list:
                        # Add a rectangle for each allocated memory block
                        rect = patches.Rectangle(
                            (base_block.start_time,
                             block.offset + base_block.offset
                             ),  # Bottom-left coordinates (x, y)
                            base_block.end_time -
                            base_block.start_time,  # Width (duration)
                            base_block.size,  # Height (memory size)
                            linewidth=0.1,
                            edgecolor='black',
                            facecolor=color,
                            alpha=0.6)
                        max_time = max(max_time, block.end_time)
                        max_offset = max(
                            max_offset,
                            block.offset + base_block.offset + base_block.size)
                        ax.add_patch(rect)
                        num_blocks += 1
                        if base_block.start_time >= iter1_start and base_block.start_time + iter_length <= global_max_time:
                            rect = patches.Rectangle(
                                (base_block.start_time + iter_length,
                                 base_block.offset + base_block.offset
                                 ),  # Bottom-left coordinates (x, y)
                                base_block.end_time - base_block.start_time,  # Width (duration)
                                base_block.size,  # Height (memory size)
                                linewidth=0.1,
                                linestyle='--',
                                alpha=0.8
                            )
                            ax.add_patch(rect)
                else:
                    # Add a rectangle for each allocated memory block
                    rect = patches.Rectangle(
                        (block.start_time,
                         merged_layer.offset + merged_layer.relative_offset[i]
                         ),  # Bottom-left coordinates (x, y)
                        block.end_time - block.start_time,  # Width (duration)
                        block.size,  # Height (memory size)
                        linewidth=0.1,
                        edgecolor='black',
                        facecolor=color,
                        alpha=0.6)
                    max_time = max(max_time, block.end_time)
                    max_offset = max(
                        max_offset, merged_layer.offset +
                        merged_layer.relative_offset[i] + block.size)
                    ax.add_patch(rect)
                    num_blocks += 1
                    if block.start_time >= iter1_start and block.start_time + iter_length <= global_max_time:
                        rect = patches.Rectangle(
                            (block.start_time + iter_length,
                             block.offset),  # Bottom-left coordinates (x, y)
                            block.end_time - block.start_time,  # Width (duration)
                            block.size,  # Height (memory size)
                            linewidth=0.1,
                            linestyle='--',
                            alpha=0.8
                        )
                        ax.add_patch(rect)
    if located_blocks is not None:
        for block in located_blocks:
            rect = patches.Rectangle(
                (block.start_time,
                 block.offset),  # Bottom-left coordinates (x, y)
                block.end_time - block.start_time,  # Width (duration)
                block.size,  # Height (memory size)
                linewidth=0.1,
                edgecolor='black',
                facecolor='red',
                alpha=0.6)
            max_time = max(max_time, block.end_time)
            max_offset = max(max_offset, block.offset + block.size)
            ax.add_patch(rect)
            num_blocks += 1
            if block.start_time >= iter1_start and block.start_time + iter_length <= global_max_time:
                rect = patches.Rectangle(
                    (block.start_time + iter_length,
                     block.offset),  # Bottom-left coordinates (x, y)
                    block.end_time - block.start_time,  # Width (duration)
                    block.size,  # Height (memory size)
                    linewidth=0.1,
                    linestyle='--',
                    alpha=0.8
                )
                ax.add_patch(rect)

    # draw iter1_start and iter1_end
    ax.axvline(x=iter1_start, color='red', linestyle='--', linewidth=1)
    ax.axvline(x=iter1_end, color='red', linestyle='--', linewidth=1)

    # Set axis labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Memory Offset")
    ax.set_title("Memory Allocation Over Time after merge")
    ax.set_xlim(0, global_max_time)
    ax.set_ylim(0, max_offset + 100000000)
    plt.savefig(path, dpi=600)
    print(f"Total number of blocks: {num_blocks}")
    print("Total offset in plot: ", max_offset / (1024**3))


def plot_time_distribution(blocks, path="time_distribution.pdf"):
    start_time = [block.start_time for block in blocks]
    end_time = [block.end_time for block in blocks]
    plt.figure(figsize=(12, 10))
    # plot scatter (start_time, end_time)
    plt.scatter(start_time, end_time, label='Start Time vs End Time', s=3)
    plt.title('Time Distribution')
    plt.xlabel('start_time')
    plt.ylabel('end_time')
    # set xlim and ylim
    plt.xlim(0, 50000)
    plt.ylim(0, 50000)
    plt.legend()
    plt.savefig(path)

def get_lifespan_list(file) -> List[Tuple[int, int]]:
    lifespan_list = []
    pattern = re.compile(r'^fused_group\s*:\s*(\d+)\s+(\d+)\s+\d+')
    
    with open(file, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                start = int(match.group(1))
                end = int(match.group(2))
                lifespan_list.append((start, end))
                
    return lifespan_list

def get_interval_lifespan_list(file) -> Tuple[List[Tuple[int, int]], List[str]]:
    layer_name_list = []
    interval_lifespan_list = []
    num_layers_iter_list = None

    with open(file, 'r') as f:
        for line in f:
            # split line by space
            parts = line.split()
            if len(parts) == 4:
                start = int(parts[-2])
                end = int(parts[-1])
                interval_lifespan_list.append((start, end))
                layer_name_list.append(parts[0])
            elif len(parts) == 3:
                start = int(parts[-2])
                end = int(parts[-1])
                interval_lifespan_list.append((start, end))
                layer_name_list.append(parts[0])
            elif len(parts) == 2:
                num_layers_iter_list = [int(part) for part in parts]
            else:
                raise ValueError(f"line={line} should have 2 or 4 parts, but got {len(parts)}")
    return interval_lifespan_list, layer_name_list, num_layers_iter_list

def write_spare_addr(spare_addr_list: List[List[Tuple[int, int]]], write_path: str, layer_name_list: List[str], num_layers_iter_list: List[int], reuse_offset: int):
    assert len(spare_addr_list) == len(layer_name_list), f"spare_addr_list and layer_name_list should have the same length, but got {len(spare_addr_list)} and {len(layer_name_list)}"
    with open(write_path, "w") as f:
        if num_layers_iter_list is not None:
            f.write(" ".join([str(num) for num in num_layers_iter_list]) + "\n")
        for i, intervals in enumerate(spare_addr_list):
            f.write(layer_name_list[i] + " ")
            f.write(" ".join([f"({s+reuse_offset},{e+reuse_offset})" for s, e in intervals]) + "\n")

def get_spare_addr(blocks: List[MemoryBlock], max_size: int, lifespan_list: List[Tuple[int, int]]) -> List[List[Tuple]]:
    '''
    Get the spare address intervals for each lifespan in lifespan_list
    '''
    assert max_size > 0, f"max_size={max_size} should be greater than 0"

    # Sort blocks by start_time
    sorted_blocks = sorted(blocks, key=lambda b: b.start_time)
    start_times = [b.start_time for b in sorted_blocks]
    
    result = []
    for lifespan in lifespan_list:
        l_start, l_end = lifespan
        
        # Step1: Get active blocks
        right = bisect_left(start_times, l_end)
        candidate_blocks = sorted_blocks[:right]
        
        # Filter out blocks that end before the lifespan starts
        active_blocks = [b for b in candidate_blocks if b.end_time > l_start]
        
        # Step2: Get occupied intervals
        occupied = []
        for block in active_blocks:
            start = int(block.offset)
            end = int(min(block.offset + block.size, max_size))
            if start < end:
                occupied.append((start, end))
        
        # Merge intervals
        merged = []
        if occupied:
            occupied.sort()
            merged = [occupied[0]]
            for s, e in occupied[1:]:
                last_e = merged[-1][1]
                if s <= last_e:
                    merged[-1] = (merged[-1][0], max(last_e, e))
                else:
                    merged.append((s, e))
        
        # Step3: Get free intervals
        free = []
        prev = 0
        for s, e in merged:
            if s > prev:
                free.append((prev, s))
            prev = max(prev, e)
        if prev < max_size:
            free.append((prev, max_size))
        
        result.append(free)
    
    return result

if __name__ == "__main__":
    args = parse_args()
    if args.iter1_start is not None and args.iter1_end is not None:
        iter1_start = args.iter1_start
        iter1_end = args.iter1_end
        iter_length = iter1_end - iter1_start

    if args.group_fuse:
        assert not args.split_and_fuse, "Cannot set both group_fuse and split_and_fuse"
        block_info = read_block_group_from_file(args.input)
        if args.unallocated_tensors_path is not None:
            unallocated_blocks = load_from_file(args.unallocated_tensors_path,
                                                args.plot_footprint)
        else:
            unallocated_blocks = []
        result, max_allocated = allocate_group(block_info, unallocated_blocks)

    else:
        kwargs = {}
        kwargs['split'] = args.split_and_fuse
        blocks = load_from_file(args.input, args.plot_footprint)
        result, max_allocated = allocate_group(None, blocks, group=False)

    if args.output is None:
        args.output = args.input.replace(".pkl", ".txt")
        args.output = args.output.replace(".txt", "_allocated_plan.txt")

    if args.fused_dynamic_path is not None:
        lifespan_list = get_lifespan_list(args.fused_dynamic_path)
        addr_intervals = get_spare_addr(result, max_allocated, lifespan_list)
        # output addr_intervals to files
        with open(args.input.replace(".txt", "_spare_addr.txt"), "w") as f:
            # write all the intervals of one element in one line
            for intervals in addr_intervals:
                f.write(" ".join([f"({s},{e})" for s, e in intervals]) + "\n")

    if args.static_reuse_path is not None:
        for path in args.static_reuse_path:
            interval_lifespan_list, layer_name_list, num_layers_iter_list = get_interval_lifespan_list(path)
            addr_intervals = get_spare_addr(result, max_allocated, interval_lifespan_list)
            write_spare_addr(addr_intervals, path.replace(".txt", "_spare_addr.txt"), layer_name_list, num_layers_iter_list, args.reuse_offset)

    with open(args.output, "w") as f:
        f.write(f"{max_allocated}\n")
        for block in result:
            f.write(
                f"{block.start_time} {block.end_time} {block.original_size if block.original_size is not None else block.size} {int(block.offset)}\n"
            )
