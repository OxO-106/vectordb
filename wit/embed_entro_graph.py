import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# Set high-quality plot parameters without LaTeX, matching the example graph
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

def create_combined_bar_chart_with_fit_line(csv_file, output_dir='data'):
    """
    Create a combined bar chart with a single fit line based on the average of
    image and caption embedding entropies with SIGMOD paper styling
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dimensions and entropy values directly from the columns
    dimensions = df['dimension'].tolist()
    image_entropies = df['avg_image_entropy'].tolist()
    caption_entropies = df['avg_caption_entropy'].tolist()
    
    # Calculate average entropy at each dimension
    average_entropies = [(img + cap) / 2 for img, cap in zip(image_entropies, caption_entropies)]
    
    # Log transform dimensions for fitting
    log_dimensions = np.log2(dimensions)
    
    # Define the fitting function (linear in log space)
    def fit_func(x, a, b):
        return a * x + b
    
    # Fit the average entropy data to the log-transformed dimensions
    avg_params, _ = curve_fit(fit_func, log_dimensions, average_entropies)
    
    # Generate smooth fit line
    x_smooth = np.linspace(min(log_dimensions), max(log_dimensions), 100)
    avg_fit = fit_func(x_smooth, *avg_params)
    
    # Convert back to original dimension scale
    x_smooth_original = 2**x_smooth
    
    # For bar chart, select a subset of dimensions to avoid overcrowding
    powers_of_two = [4, 8, 16, 32, 64, 128, 256, 512]
    available_powers = [dim for dim in powers_of_two if dim in dimensions]
    
    selected_indices = [dimensions.index(dim) for dim in available_powers]
    selected_dimensions = [dimensions[i] for i in selected_indices]
    selected_image_entropies = [image_entropies[i] for i in selected_indices]
    selected_caption_entropies = [caption_entropies[i] for i in selected_indices]
    
    # Create figure and axis objects
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot bar chart on the primary axis
    x = np.arange(len(selected_dimensions))
    width = 0.35  # Width of the bars
    
    image_bars = ax1.bar(x - width/2, selected_image_entropies, width, label='Image', color='blue', alpha=0.6)
    caption_bars = ax1.bar(x + width/2, selected_caption_entropies, width, label='Caption', color='#4CAF50', alpha=0.6)
    
    # Configure primary axis (bar chart)
    ax1.set_xlabel('Dimension (2^x)')
    ax1.set_ylabel('Entropy')
    ax1.set_title('Combined Embedding Entropy Comparison')
    ax1.set_xticks(x)
    # Format labels as 2^x
    ax1.set_xticklabels([f'2^{int(np.log2(dim))}' if np.log2(dim).is_integer() else f'{dim}' for dim in selected_dimensions])
    
    # Add a secondary axis that shares the x-axis for the fit line
    ax2 = ax1.twiny()
    
    # Set up the secondary x-axis to use the log scale
    ax2.set_xscale('log', base=2)
    
    # Set the limits to match the original data range
    ax2.set_xlim(min(dimensions), max(dimensions))
    
    # Make the secondary axis ticks invisible
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
    
    # Plot the average fit line on the secondary axis
    fit_line = ax2.plot(x_smooth_original, avg_fit, color='red', linewidth=2.5, linestyle='-', label='Fit Line')
    
    # Add grid and make it behind the bars
    ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, axis='y', zorder=0)
    
    # Create a combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    # Combined legend with only the specified labels
    ax1.legend(handles1 + handles2, 
              ['Image Bars', 'Caption Bars', 'Fit Line'], 
              loc='upper left')
    
    # Set y-axis to start from 0 with some padding
    max_entropy = max(max(image_entropies), max(caption_entropies))
    ax1.set_ylim(0, max_entropy * 1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure in high resolution
    plt.savefig(os.path.join(output_dir, 'combined_embedding_entropy_bar_with_fit_line.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'combined_embedding_entropy_bar_with_fit_line.pdf'), format='pdf', bbox_inches='tight')
    
    print(f"Bar chart with fit line saved to {output_dir}/combined_embedding_entropy_bar_with_fit_line.png")
    plt.close(fig)

if __name__ == "__main__":
    csv_file = "embed_entropy_summary.csv"  # Use the provided data file
    create_combined_bar_chart_with_fit_line(csv_file)

if __name__ == "__main__":
    csv_file = "embed_entropy_summary.csv"  # Use the provided data file
    create_combined_bar_chart_with_fit_line(csv_file)