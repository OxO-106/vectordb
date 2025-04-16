import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# Set high-quality plot parameters without LaTeX
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.titlesize": 12
})

# Set figure DPI for high-quality output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600

def create_caption_histogram_with_fit_line(csv_file, output_dir='data'):
    """
    Create a histogram showing only caption data points with a logarithmic fit line
    for specific dimensions: 4, 8, 16, 32, 64, 128, 256, 512
    Using linear x-axis and logarithmic fit function
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all dimensions and caption entropy values for fit line calculation
    all_dimensions = df['dimension'].tolist()
    all_caption_entropies = df['avg_caption_entropy'].tolist()
    
    # Filter for specific dimensions
    target_dimensions = [4, 8, 16, 32, 64, 128, 256, 512]
    
    # Create filtered lists
    filtered_indices = [i for i, dim in enumerate(all_dimensions) if dim in target_dimensions]
    filtered_dimensions = [all_dimensions[i] for i in filtered_indices]
    filtered_caption_entropies = [all_caption_entropies[i] for i in filtered_indices]
    
    # Define the logarithmic fitting function
    def log_func(x, a, b):
        return a * np.log(x) + b
    
    # Fit the caption entropy data to the logarithmic function
    caption_params, _ = curve_fit(log_func, all_dimensions, all_caption_entropies)
    
    # Generate smooth fit line
    x_smooth = np.linspace(min(all_dimensions), max(all_dimensions), 100)
    caption_fit = log_func(x_smooth, *caption_params)
    
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram of caption entropy values for filtered dimensions
    # Use fixed width bars for better appearance on linear scale
    bar_width = 15  # Fixed width for all bars
    bars = ax.bar(filtered_dimensions, filtered_caption_entropies, 
                 alpha=0.7, color='skyblue', width=bar_width, 
                 align='center', label='Caption Entropy')
    
    # Plot the logarithmic fit line
    ax.plot(x_smooth, caption_fit, color='black', linewidth=2.5, 
            linestyle='-', label='Log Fit: y = a·ln(x) + b')
    
    # Configure axis with linear scale
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Entropy')
    ax.set_title('Caption Embedding Entropy with Logarithmic Fit')
    
    # Set x-ticks to exactly the filtered dimensions
    ax.set_xticks(filtered_dimensions)
    ax.set_xticklabels([str(dim) for dim in filtered_dimensions])
    
    # Add grid and make it behind the bars
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, axis='y', zorder=0)
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Set y-axis to start from 0 with some padding
    ax.set_ylim(0, max(all_caption_entropies) * 1.05)
    
    # # Add value labels on top of bars
    # for bar in bars:
    #     height = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(all_caption_entropies),
    #             f'{height:.2f}', ha='center', va='bottom', fontsize=8)
                
    # # Display the fit parameters
    # a, b = caption_params
    # equation_text = f'y = {a:.4f}·ln(x) + {b:.4f}'
    # ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, 
    #         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure in high resolution
    plt.savefig(os.path.join(output_dir, 'entropy_histogram_with_fit_line.png'), 
                dpi=600, bbox_inches='tight')
    
    print(f"Caption histogram with fit line saved to {output_dir}/caption_entropy_histogram_with_fit_line.png")
    plt.close(fig)

if __name__ == "__main__":
    csv_file = "embed_entropy_summary.csv"  # Use the provided data file
    create_caption_histogram_with_fit_line(csv_file)
    
    # Print information about the filtering
    print("Histogram created showing only the specified dimensions: 4, 8, 16, 32, 64, 128, 256, 512")
    print("Fit line is calculated using all available data points for accuracy")