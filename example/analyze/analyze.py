import re
import csv
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_memory_stats(log_file):
    """
    Extract memory statistics from log file
    """
    # Read log file
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Use regex to extract memory statistics
    pattern = r'dev(\d+), iter(\d+) : max_reserved:(\d+\.\d+), max_allocated:(\d+\.\d+), utilization:(\d+\.\d+)%'
    matches = re.findall(pattern, content)
    
    # Organize data structure
    stats = {}
    for match in matches:
        dev_id = int(match[0])
        iter_num = int(match[1])
        
        if 5 <= iter_num <= 19:  # Only extract iterations 5-19
            if iter_num not in stats:
                stats[iter_num] = {}
                
            stats[iter_num][dev_id] = {
                'max_reserved': float(match[2]),
                'max_allocated': float(match[3]),
                'utilization': float(match[4]),
                'wasted_memory': float(match[2]) - float(match[3])
            }
    
    return stats

def calculate_group_stats(stats):
    """
    Calculate average values and other metrics for each group
    """
    group_stats = {}
    
    for iter_num, iter_stats in stats.items():
        group_stats[iter_num] = {
            'group1': {'devs': [], 'max_reserved': [], 'max_allocated': [], 'utilization': [], 'wasted_memory': []},
            'group2': {'devs': [], 'max_reserved': [], 'max_allocated': [], 'utilization': [], 'wasted_memory': []}
        }
        
        # Group devices and collect data
        for dev_id, dev_stats in iter_stats.items():
            group_key = 'group1' if 0 <= dev_id <= 3 else 'group2'
            group_stats[iter_num][group_key]['devs'].append(dev_id)
            group_stats[iter_num][group_key]['max_reserved'].append(dev_stats['max_reserved'])
            group_stats[iter_num][group_key]['max_allocated'].append(dev_stats['max_allocated'])
            group_stats[iter_num][group_key]['utilization'].append(dev_stats['utilization'])
            group_stats[iter_num][group_key]['wasted_memory'].append(dev_stats['wasted_memory'])
    
    # Calculate average values and other metrics for each group
    for iter_num, groups in group_stats.items():
        for group_key, group_data in groups.items():
            if group_data['devs']:  # Ensure data exists
                group_data['avg_max_reserved'] = np.mean(group_data['max_reserved'])
                group_data['avg_max_allocated'] = np.mean(group_data['max_allocated'])
                group_data['avg_utilization'] = np.mean(group_data['utilization'])
                group_data['avg_1_minus_utilization'] = 100 - group_data['avg_utilization']
                group_data['avg_wasted_memory'] = np.mean(group_data['wasted_memory'])
    
    return group_stats

def calculate_summary(group_stats):
    """
    Calculate summary statistics from group_stats
    """
    # Calculate averages across all iterations
    iter_nums = sorted(group_stats.keys())
    
    all_g1_util = [group_stats[i]['group1']['avg_utilization'] for i in iter_nums]
    all_g1_wasted = [group_stats[i]['group1']['avg_wasted_memory'] for i in iter_nums]
    all_g1_reserved = [group_stats[i]['group1']['avg_max_reserved'] for i in iter_nums]
    all_g1_allocated = [group_stats[i]['group1']['avg_max_allocated'] for i in iter_nums]
    
    all_g2_util = [group_stats[i]['group2']['avg_utilization'] for i in iter_nums]
    all_g2_wasted = [group_stats[i]['group2']['avg_wasted_memory'] for i in iter_nums]
    all_g2_reserved = [group_stats[i]['group2']['avg_max_reserved'] for i in iter_nums]
    all_g2_allocated = [group_stats[i]['group2']['avg_max_allocated'] for i in iter_nums]
    
    # Create summary data
    summary = {
        "g1_util": np.mean(all_g1_util),
        "g1_waste": np.mean(all_g1_wasted),
        "g1_reserved": np.mean(all_g1_reserved),
        "g1_allocated": np.mean(all_g1_allocated),
        "g2_util": np.mean(all_g2_util),
        "g2_waste": np.mean(all_g2_wasted),
        "g2_reserved": np.mean(all_g2_reserved),
        "g2_allocated": np.mean(all_g2_allocated),
    }
    
    return summary

def determine_higher_memory_group(summary_data):
    """Determine which group has higher memory allocation"""
    g1_allocated = summary_data["g1_allocated"]
    g2_allocated = summary_data["g2_allocated"]
    
    if g1_allocated >= g2_allocated:
        return "group1", "Group 1 (Devices 0-3)"
    else:
        return "group2", "Group 2 (Devices 4-7)"


def plot_specific_comparison(log_summaries, baseline_log, improved_log, output_dir):
    """
    Plot detailed comparison between two specific log files (usually baseline vs. improved)
    """
    if baseline_log not in log_summaries or improved_log not in log_summaries:
        print(f"Warning: Cannot compare {baseline_log} and {improved_log}. One or both logs missing.")
        return
    
    baseline_data = log_summaries[baseline_log]
    improved_data = log_summaries[improved_log]
    
    # Determine which group has higher allocation in baseline
    baseline_g1_allocated = baseline_data["g1_allocated"]
    baseline_g2_allocated = baseline_data["g2_allocated"]
    
    if baseline_g1_allocated >= baseline_g2_allocated:
        primary_group = "group1"
        group_label = "Group 1 (Devices 0-3)"
        baseline_util = baseline_data["g1_util"]
        baseline_waste = baseline_data["g1_waste"]
        improved_util = improved_data["g1_util"]
        improved_waste = improved_data["g1_waste"]
    else:
        primary_group = "group2"
        group_label = "Group 2 (Devices 4-7)"
        baseline_util = baseline_data["g2_util"]
        baseline_waste = baseline_data["g2_waste"]
        improved_util = improved_data["g2_util"]
        improved_waste = improved_data["g2_waste"]
    
    # Calculate improvement metrics
    util_improvement = improved_util - baseline_util
    waste_reduction = baseline_waste - improved_waste
    waste_reduction_pct = (waste_reduction / baseline_waste) * 100 if baseline_waste > 0 else 0
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prepare data for the selected group
    labels = [os.path.basename(baseline_log).split('.')[0], os.path.basename(improved_log).split('.')[0]]
    util_values = [baseline_util, improved_util]
    waste_values = [baseline_waste, improved_waste]
    
    # Plot 1: Memory Utilization
    x = np.arange(len(labels))
    width = 0.6
    
    rects1 = ax1.bar(x, util_values, width, color=['#ff9999', '#66b3ff'])
    
    ax1.set_ylabel('Memory Utilization (%)', fontsize=12)
    ax1.set_title(f'Memory Utilization Comparison ({group_label})', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylim(80, 101)
    
    # Add value labels to bars
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Wasted Memory
    rects2 = ax2.bar(x, waste_values, width, color=['#ff9999', '#66b3ff'])
    
    ax2.set_ylabel('Wasted Memory (GB)', fontsize=12)
    ax2.set_title(f'Wasted Memory Comparison ({group_label})', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    
    # Add value labels to bars
    for i, rect in enumerate(rects2):
        height = rect.get_height()
        if i == 0:
            label = f'{height:.2f} GB'
        else:
            label = f'{height:.2f} GB\n(-{waste_reduction_pct:.1f}%)'
        ax2.annotate(label,
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{primary_group}.png'), dpi=300)
    print(f"Comparison plot saved to {os.path.join(output_dir, f'comparison_{primary_group}.png')}")
    plt.close()
    
    return {
        "primary_group": primary_group,
        "group_label": group_label,
        "baseline_util": baseline_util,
        "baseline_waste": baseline_waste,
        "improved_util": improved_util,
        "improved_waste": improved_waste,
        "util_improvement": util_improvement,
        "waste_reduction": waste_reduction,
        "waste_reduction_pct": waste_reduction_pct
    }
def plot_multiple_logs_comparison(log_summaries, output_dir):
    """
    Plot comparison of multiple log files focusing only on the group with higher allocated memory,
    with torch logs first and alloc.log last
    """
    # Sort log files: torch logs first, alloc.log last, others in between
    log_files = []
    torch_logs = []
    other_logs = []
    alloc_log = None
    
    for log_file in log_summaries.keys():
        base_name = os.path.basename(log_file).lower()
        if base_name.startswith('torch'):
            torch_logs.append(log_file)
        elif base_name == 'alloc.log':
            alloc_log = log_file
        else:
            other_logs.append(log_file)
    
    # Sort each category
    torch_logs.sort()
    other_logs.sort()
    
    # Combine logs in the specified order
    log_files = torch_logs + other_logs
    if alloc_log:
        log_files.append(alloc_log)
    
    if not log_files:
        print("No log files to compare")
        return
    
    # Determine which group has higher allocated memory (use first torch log as reference)
    reference_log = torch_logs[0] if torch_logs else log_files[0]
    reference_data = log_summaries[reference_log]
    
    if reference_data["g1_allocated"] >= reference_data["g2_allocated"]:
        primary_group = "group1"
        group_label = "Group 1 (Devices 0-3)"
        get_util = lambda data: data["g1_util"]
        get_waste = lambda data: data["g1_waste"]
        get_allocated = lambda data: data["g1_allocated"]
    else:
        primary_group = "group2"
        group_label = "Group 2 (Devices 4-7)"
        get_util = lambda data: data["g2_util"]
        get_waste = lambda data: data["g2_waste"]
        get_allocated = lambda data: data["g2_allocated"]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data for plots
    file_names = [os.path.basename(log_file).split('.')[0] for log_file in log_files]
    utils = [get_util(log_summaries[log_file]) for log_file in log_files]
    wastes = [get_waste(log_summaries[log_file]) for log_file in log_files]
    
    # Plot 1: Memory Utilization
    x = np.arange(len(file_names))
    width = 0.618
    
    # Use gradient colors with torch logs in red shades and others in blue
    colors = []
    for log_file in log_files:
        base_name = os.path.basename(log_file).lower()
        if base_name.startswith('torch_'):
            colors.append('#ff9999')  # Light red for torch logs
        elif base_name.startswith('torch'):
            colors.append('#FFDEAD')  # Strong red for torch.log
        elif base_name == 'alloc.log':
            colors.append('#3366ff')  # Strong blue for alloc.log
        else:
            colors.append('#99ccff')  # Light blue for other logs
    
    rects1 = ax1.bar(x, utils, width, color=colors)
    
    ax1.set_ylabel('Memory Utilization (%)', fontsize=12)
    ax1.set_title(f'Memory Utilization Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(file_names, fontsize=12, ha='center')
    ax1.set_ylim(80, 101)
    
    # Add value labels to utilization bars
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)
    
    # Plot 2: Wasted Memory (Fragmentation)
    rects2 = ax2.bar(x, wastes, width, color=colors)
    
    ax2.set_ylabel('Wasted Memory (GB)', fontsize=12)
    ax2.set_title(f'Memory Fragmentation Comparison', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(file_names, fontsize=12, ha='center')
    
    # Add value labels to fragmentation bars
    for i, rect in enumerate(rects2):
        height = rect.get_height()
        ax2.annotate(f'{height:.2f} GB',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)
    
    # If alloc.log exists, add reduction percentage annotations compared to each torch log
    if alloc_log in log_files:
        alloc_idx = log_files.index(alloc_log)
        alloc_waste = wastes[alloc_idx]
        
        # Create secondary y-axis for percentage
        ax3 = ax2.twinx()
        ax3.set_ylabel('Fragmentation Reduction (%)', color='green', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='green')
        ax3.set_ylim([0, 105])  # Set limit to 0-105% for better visibility
        
        # Calculate reduction percentages and positions for line chart
        reduction_percentages = []
        x_positions = []

        # For each torch log, calculate reduction compared to alloc.log
        for i, log_file in enumerate(log_files):
            if not os.path.basename(log_file).lower().startswith('alloc'):
                baseline_waste = wastes[i]
                waste_reduction = baseline_waste - alloc_waste
                if baseline_waste > 0:
                    waste_reduction_pct = (waste_reduction / baseline_waste) * 100
                    reduction_percentages.append(waste_reduction_pct)
                    x_positions.append(i)
                    
                    # Add percentage labels to line points
                    ax3.annotate(f'{waste_reduction_pct:.1f}%',
                                xy=(i, waste_reduction_pct),
                                xytext=(0, 5),
                                textcoords="offset points",
                                ha='left', va='bottom',
                                fontsize=12,
                                color='green')

        
        # Plot the reduction line
        line = ax3.plot(x_positions, reduction_percentages, 'go-', linewidth=2, markersize=6, label='Reduction %')
        
        # Add legend for the line
        ax3.legend(loc='upper right')
        
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, ), dpi=300)
    print(f"Comparison plot saved to {output_dir}")
    plt.close()
    
    return primary_group, group_label

def save_summary_csv(log_summaries, output_file):
    """
    Save summary data from all log files to a CSV
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Prepare CSV headers
        headers = [
            'log_file',
            'group1_utilization', 'group1_wasted_memory', 'group1_reserved', 'group1_allocated',
            'group2_utilization', 'group2_wasted_memory', 'group2_reserved', 'group2_allocated'
        ]
        writer.writerow(headers)
        
        # Add data for each log file
        for log_file, summary in log_summaries.items():
            row = [
                os.path.basename(log_file),
                f"{summary['g1_util']:.2f}%", f"{summary['g1_waste']:.2f}GB", 
                f"{summary['g1_reserved']:.2f}GB", f"{summary['g1_allocated']:.2f}GB",
                f"{summary['g2_util']:.2f}%", f"{summary['g2_waste']:.2f}GB",
                f"{summary['g2_reserved']:.2f}GB", f"{summary['g2_allocated']:.2f}GB"
            ]
            writer.writerow(row)

def main():
    import argparse
    
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Analyze memory usage from log files')
    parser.add_argument('--log_name', type=str, required=True,
                        help='Directory containing log files to analyze')
    # Parse arguments
    args = parser.parse_args()
    # Set input directory and output paths
    model_name = args.log_name.split('-')[0]
    base_path = os.environ.get('STALLOC_DIR')
    log_dir = base_path+'/STAlloc/log/'+model_name+'/'+ args.log_name
    output_dir = base_path + '/STAlloc/example/analyze/'+model_name
    output_summary_file = os.path.join(output_dir, args.log_name+'.csv')
    output_comparison_file = os.path.join(output_dir, args.log_name+'.png')
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all log files in the directory
    log_files = glob.glob(os.path.join(log_dir, '*.log'))
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        return
    
    # Process each log file
    log_summaries = {}
    for log_file in log_files:
        try:
            print(f"Processing {os.path.basename(log_file)}...")
            stats = extract_memory_stats(log_file)
            if not stats:
                print(f"  - No valid memory stats found in {os.path.basename(log_file)}")
                continue
                
            group_stats = calculate_group_stats(stats)
            summary = calculate_summary(group_stats)
            log_summaries[log_file] = summary
            print(f"  - Found data for {len(stats)} iterations")
        except Exception as e:
            print(f"  - Error processing {os.path.basename(log_file)}: {str(e)}")
    
    if not log_summaries:
        print("No valid log data found. Exiting.")
        return
    
    # Save summary to CSV
    save_summary_csv(log_summaries, output_summary_file)
    print(f"Summary data saved to {output_summary_file}")
    
    # Plot comparison focusing only on the group with higher allocated memory
    primary_group, group_label = plot_multiple_logs_comparison(log_summaries, output_comparison_file)
    
    # Compare torch logs with alloc.log if both exist
    torch_logs = [log for log in log_summaries if os.path.basename(log).lower().startswith('torch')]
    alloc_log = os.path.join(log_dir, 'alloc.log')
    
    if torch_logs and alloc_log in log_summaries:
        # Compare the first torch log with alloc.log
        torch_log = torch_logs[0]
        torch_data = log_summaries[torch_log]
        alloc_data = log_summaries[alloc_log]
        
        # Get data for the primary group (higher allocated memory)
        if primary_group == "group1":
            torch_util = torch_data["g1_util"]
            torch_waste = torch_data["g1_waste"]
            alloc_util = alloc_data["g1_util"]
            alloc_waste = alloc_data["g1_waste"]
        else:
            torch_util = torch_data["g2_util"]
            torch_waste = torch_data["g2_waste"]
            alloc_util = alloc_data["g2_util"]
            alloc_waste = alloc_data["g2_waste"]
        
        # Calculate improvements
        util_improvement = alloc_util - torch_util
        waste_reduction = torch_waste - alloc_waste
        waste_reduction_pct = (waste_reduction / torch_waste) * 100 if torch_waste > 0 else 0
        
        # Print summary
        print("\nImprovement Summary:")
        print("======================================")
        print(f"  - PyTorch Default: Utilization={torch_util:.2f}%, Memory Fragmentation={torch_waste:.2f}GB")
        print(f"  - STAlloc Improved: Utilization={alloc_util:.2f}%, Memory Fragmentation={alloc_waste:.2f}GB")
        print(f"  - Improvement: Utilization +{util_improvement:.2f}%, Memory Fragmentation -{waste_reduction:.2f}GB ({waste_reduction_pct:.2f}%)")
        print("======================================")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()