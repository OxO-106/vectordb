import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def plot_dimension_results(csv_file=None, results_dict=None, dimensions=None, 
                           output_file='dimension_plot.png', show_plot=True,
                           x_as_power_of_two=True):
    """
    Plot the results of dimension benchmarking
    
    Parameters:
    -----------
    csv_file : str, optional
        Path to CSV file containing benchmark results
    results_dict : dict, optional
        Dictionary mapping dimensions to query times
    dimensions : list, optional
        List of dimensions (only needed if results_dict is provided)
    output_file : str
        Filename to save the plot
    show_plot : bool
        Whether to display the plot
    x_as_power_of_two : bool
        Whether to display x-axis as powers of 2 (2^n)
    """
    # Load data from CSV if provided
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        dimensions = df['dimension'].values
        query_times = df['query_time'].values
    # Otherwise use provided dictionary
    elif results_dict is not None and dimensions is not None:
        query_times = [results_dict[dim] for dim in dimensions]
    else:
        raise ValueError("Either csv_file or (results_dict and dimensions) must be provided")
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    # Calculate log2 values for x-ticks if needed
    if x_as_power_of_two:
        # Create x-axis labels
        x_labels = [f'2^{int(math.log2(dim))}' for dim in dimensions]
        
        # Plot using original indices but with custom labels
        plt.plot(range(len(dimensions)), query_times, 'b-o', linewidth=2, markersize=8)
        plt.xticks(range(len(dimensions)), x_labels)
        
        # Add vertical gridlines at each tick
        plt.grid(True, axis='both', linestyle='--', alpha=0.7)
    else:
        # Standard plot with dimensions as x values
        plt.plot(dimensions, query_times, 'b-o', linewidth=2, markersize=8)
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format y-axis to be more readable (milliseconds or microseconds if times are small)
    if max(query_times) < 0.001:
        # Convert to microseconds
        plt.plot(range(len(dimensions)), [t * 1000000 for t in query_times], 'b-o', linewidth=2, markersize=8)
        plt.ylabel('Query Time (microseconds)')
    elif max(query_times) < 1:
        # Convert to milliseconds
        plt.plot(range(len(dimensions)), [t * 1000 for t in query_times], 'b-o', linewidth=2, markersize=8)
        plt.ylabel('Query Time (milliseconds)')
    else:
        plt.ylabel('Query Time (seconds)')
    
    # Add labels and title
    plt.xlabel('Dimension')
    plt.title('Dimension vs. Query Time', fontsize=14, fontweight='bold')
    
    # Adjust y-axis limits to show differences more clearly
    y_min = min(query_times) * 0.9  # 10% below minimum
    y_max = max(query_times) * 1.1  # 10% above maximum
    plt.ylim(y_min, y_max)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300)
    print(f"Results plot saved as '{output_file}'")
    
    # Show the plot if requested
    if show_plot:
        plt.show()


if __name__ == "__main__":
    # Example usage with CSV file
    plot_dimension_results(csv_file="4.csv", 
                           output_file="dimension_plot_with_powers.png",
                           x_as_power_of_two=True)