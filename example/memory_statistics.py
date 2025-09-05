#!/usr/bin/env python3
"""
Memory Statistics Analysis Script

This script analyzes CSV files in specified directories to calculate the average reduction 
percentage of wasted memory for alloc.log compared to other log files.

For each CSV file:
1. Find all log entries and alloc.log entry
2. For each log entry, determine which group (group1 or group2) has higher allocated memory
3. Compare alloc.log's wasted memory reduction against other logs in that group
4. Calculate the reduction percentage: 1 - (alloc_wasted_memory / other_wasted_memory)
5. Average across CSV files for the same log comparison

Usage:
    python memory_statistics.py [directories...]
    
Examples:
    python memory_statistics.py                                    # Use default directories
    python memory_statistics.py example/analyze/llama             # Analyze single directory
    python memory_statistics.py example/analyze/llama example/analyze/gpt2  # Analyze multiple directories
"""

import os
import csv
import glob
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Note: we provide the original experiment data in ZOR, since offload is not support in Megatron-LM

def parse_memory_value(value_str: str) -> float:
    """
    Parse memory value string (e.g., "3.95GB") to float in GB.
    
    Args:
        value_str: Memory value string like "3.95GB"
        
    Returns:
        Memory value in GB as float
    """
    if not value_str or value_str.strip() == '':
        return 0.0
    
    value_str = value_str.strip().upper()
    if value_str.endswith('GB'):
        return float(value_str[:-2])
    elif value_str.endswith('MB'):
        return float(value_str[:-2]) / 1024
    elif value_str.endswith('KB'):
        return float(value_str[:-2]) / (1024 * 1024)
    else:
        # Assume it's already in GB
        return float(value_str)


def parse_csv_file(csv_path: str) -> Optional[Dict]:
    """
    Parse a single CSV file and extract memory statistics.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Dictionary containing parsed data or None if parsing failed
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if not rows:
            print(f"Warning: Empty CSV file {csv_path}")
            return None
            
        # Find all entries
        entries = {}
        alloc_entry = None
        
        for row in rows:
            log_file = row.get('log_file', '').strip()
            if log_file == 'alloc.log':
                alloc_entry = row
            elif log_file:  # Store all non-empty log files
                entries[log_file] = row
                
        if not entries:
            print(f"Warning: No valid log entries found in {csv_path}")
            return None
            
        if not alloc_entry:
            print(f"Warning: No alloc.log entry found in {csv_path}")
            return None
            
        # For each log entry, determine which group has higher allocated memory
        log_comparisons = {}
        
        for log_file, log_entry in entries.items():
            group1_allocated = parse_memory_value(log_entry.get('group1_allocated', '0'))
            group2_allocated = parse_memory_value(log_entry.get('group2_allocated', '0'))
            
            if group1_allocated >= group2_allocated:
                selected_group = 'group1'
            else:
                selected_group = 'group2'
                
            log_wasted_memory = parse_memory_value(log_entry.get(f'{selected_group}_wasted_memory', '0'))
            alloc_wasted_memory = parse_memory_value(alloc_entry.get(f'{selected_group}_wasted_memory', '0'))
            
            log_comparisons[log_file] = {
                'log_entry': log_entry,
                'selected_group': selected_group,
                'log_wasted_memory': log_wasted_memory,
                'alloc_wasted_memory': alloc_wasted_memory,
                'group1_allocated': group1_allocated,
                'group2_allocated': group2_allocated
            }
            
        result = {
            'csv_file': os.path.basename(csv_path),
            'alloc_entry': alloc_entry,
            'log_comparisons': log_comparisons
        }
            
        return result
        
    except Exception as e:
        print(f"Error parsing {csv_path}: {e}")
        return None


def find_csv_files(directories: List[str] = None) -> List[str]:
    """
    Find all CSV files in the specified directories.
    
    Args:
        directories: List of directory paths to search. If None, use default directories.
        
    Returns:
        List of CSV file paths
    """
    csv_files = []
    
    # Use default directories if none specified
    if not directories:
        directories = [
            "example/analyze/llama",
            "example/analyze/gpt2", 
            "example/analyze/test_data"
        ]
        print("No directories specified, using default directories:")
        for dir_path in directories:
            print(f"  - {dir_path}")
        print()
    
    # Search each specified directory
    for directory in directories:
        if os.path.exists(directory):
            dir_files = glob.glob(os.path.join(directory, "*.csv"))
            csv_files.extend(dir_files)
            print(f"Found {len(dir_files)} CSV files in '{directory}' directory")
        else:
            print(f"Warning: Directory '{directory}' does not exist")
    
    return csv_files


def calculate_reduction_statistics(parsed_data: List[Dict]) -> Dict:
    """
    Calculate wasted memory reduction statistics.
    
    Args:
        parsed_data: List of parsed CSV data
        
    Returns:
        Dictionary containing reduction statistics
    """
    # Group by log file type for averaging across CSV files
    log_reductions = defaultdict(list)  # {log_file: [reduction_percentages]}
    
    for data in parsed_data:
        csv_file = data['csv_file']
        print(f"\nAnalyzing {csv_file}:")
        
        for log_file, comparison in data['log_comparisons'].items():
            log_wasted = comparison['log_wasted_memory']
            alloc_wasted = comparison['alloc_wasted_memory']
            selected_group = comparison['selected_group']
            
            if log_wasted <= 0:
                print(f"  Skipping {log_file}: Invalid wasted memory value ({log_wasted})")
                continue
                
            # Calculate reduction percentage: 1 - (alloc_wasted / log_wasted)
            reduction_percentage = 1 - (alloc_wasted / log_wasted)
            log_reductions[log_file].append(reduction_percentage)
            
            print(f"  {log_file}:")
            print(f"    Selected group: {selected_group}")
            print(f"    Group1 allocated: {comparison['group1_allocated']:.2f} GB")
            print(f"    Group2 allocated: {comparison['group2_allocated']:.2f} GB")
            print(f"    {log_file} wasted memory: {log_wasted:.2f} GB")
            print(f"    Alloc.log wasted memory: {alloc_wasted:.2f} GB")
            print(f"    Reduction percentage: {reduction_percentage*100:.2f}%")
        
    # Calculate average reductions for each log file type
    results = {}
    for log_file, reductions in log_reductions.items():
        if reductions:
            avg_reduction = sum(reductions) / len(reductions)
            results[log_file] = {
                'individual_reductions': reductions,
                'average_reduction': avg_reduction,
                'valid_comparisons': len(reductions)
            }
        else:
            results[log_file] = {
                'individual_reductions': [],
                'average_reduction': 0.0,
                'valid_comparisons': 0
            }
            
    return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze CSV files to calculate wasted memory reduction statistics for alloc.log vs other log files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                           # Use default directories
  %(prog)s example/analyze/llama                     # Analyze single directory  
  %(prog)s example/analyze/llama example/analyze/gpt2  # Analyze multiple directories
  %(prog)s /path/to/custom/dir                       # Analyze custom directory
        """
    )
    
    parser.add_argument(
        'directories',
        nargs='*',
        help='Directories to search for CSV files. If not specified, uses default directories.'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the memory statistics analysis."""
    # Parse command line arguments
    args = parse_arguments()
    
    print("Memory Statistics Analysis")
    print("=" * 50)
    
    if args.verbose:
        print(f"Verbose mode enabled")
        if args.directories:
            print(f"Specified directories: {args.directories}")
        print()
    
    # Find all CSV files
    csv_files = find_csv_files(args.directories if args.directories else None)
    
    if not csv_files:
        print("No CSV files found in target directories!")
        return
        
    print(f"Total CSV files found: {len(csv_files)}")
    print()
    
    # Parse all CSV files
    parsed_data = []
    for csv_file in csv_files:
        if args.verbose:
            print(f"Parsing {csv_file}...")
        else:
            print(f"Parsing {os.path.basename(csv_file)}...")
        data = parse_csv_file(csv_file)
        if data:
            parsed_data.append(data)
            
    if not parsed_data:
        print("No valid data found in CSV files!")
        return
        
    print(f"Successfully parsed {len(parsed_data)} CSV files")
    print()
    
    # Calculate reduction statistics
    print("Calculating reduction statistics...")
    print("-" * 40)
    
    stats = calculate_reduction_statistics(parsed_data)
    
    # Print summary results
    print("\nSummary Results:")
    print("=" * 50)
    
    if not stats:
        print("No valid comparisons found!")
        return
        
    for log_file, log_stats in stats.items():
        if log_stats['valid_comparisons'] > 0:
            print(f"\nAlloc.log vs {log_file}:")
            print(f"  Valid comparisons: {log_stats['valid_comparisons']}")
            print(f"  Average wasted memory reduction: {log_stats['average_reduction']*100:.2f}%")
            print(f"  Individual reductions: {[f'{r*100:.2f}%' for r in log_stats['individual_reductions']]}")
        else:
            print(f"\nAlloc.log vs {log_file}: No valid comparisons found")
            
    print(f"\nAnalysis completed! Analyzed {len([s for s in stats.values() if s['valid_comparisons'] > 0])} log file types.")


if __name__ == "__main__":
    main()
