import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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

def analyze_embedding_entropies(csv_file, output_dir='data'):
    """
    Analyze and visualize embedding entropies for different dimensions
    with SIGMOD paper styling
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dimensions from column names dynamically
    dimensions = []
    image_means = []
    caption_means = []
    
    # Find all dimensions present in the column names
    for col in df.columns:
        if col.startswith('image_embedding_entropy_'):
            dim = int(col.split('_')[-1])
            dimensions.append(dim)
    
    # Sort dimensions
    dimensions.sort()
    
    # Calculate means for each dimension
    for dim in dimensions:
        image_col = f'image_embedding_entropy_{dim}'
        caption_col = f'caption_embedding_entropy_{dim}'
        
        image_means.append(df[image_col].mean())
        caption_means.append(df[caption_col].mean())
    
    # Create Image Embedding Entropy plot
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Plot the data points
    ax.scatter(dimensions, image_means, color='black', marker='o', s=25, alpha=0.8)
    
    # Get min/max values for bounds
    x_min = min(dimensions) - 2
    x_max = max(dimensions) + 2
    
    # Create smooth line using piecewise linear segments instead of interpolation
    # This creates a straight line appearance rather than curved
    ax.plot(dimensions, image_means, color='blue', linewidth=2.0)
    
    # Set axis labels
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Entropy')
    
    # Set title
    ax.set_title('Image Embedding Entropy')
    
    # No EC50 mark or reference lines needed
    
    # Set y-axis to start from 0
    ax.set_ylim(0, max(image_means) * 1.05)
    
    # Set x-axis to log scale with base 2
    ax.set_xscale('log', base=2)
    
    # Add grid with appropriate styling
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Add minor ticks
    ax.minorticks_on()
    
    # Adjust layout
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    
    # Save the figure in high resolution
    plt.savefig(os.path.join(output_dir, 'image_embedding_entropy_sigmod.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'image_embedding_entropy_sigmod.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    # Create Caption Embedding Entropy plot with the same styling
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Plot the data points
    ax.scatter(dimensions, caption_means, color='black', marker='o', s=25, alpha=0.8)
    
    # Create straight line for caption means
    ax.plot(dimensions, caption_means, color='#4CAF50', linewidth=2.0)
    
    # Set axis labels
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Entropy')
    
    # Set title
    ax.set_title('Caption Embedding Entropy')
    
    # No EC50 mark or reference lines needed
    
    # Set y-axis to start from 0
    ax.set_ylim(0, max(caption_means) * 1.05)
    
    # Set x-axis to log scale with base 2
    ax.set_xscale('log', base=2)
    
    # Add grid with appropriate styling
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Add minor ticks
    ax.minorticks_on()
    
    # Adjust layout
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    
    # Save the figure in high resolution
    plt.savefig(os.path.join(output_dir, 'caption_embedding_entropy_sigmod.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'caption_embedding_entropy_sigmod.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    # Create a combined visualization
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # For combined view, just show data points with a single blue line (like in example)
    # Plot data points
    ax.scatter(dimensions, image_means, color='black', marker='o', s=25, alpha=0.8)
    
    # For combined view, use straight line instead of curved
    ax.plot(dimensions, image_means, color='blue', linewidth=2.0)
    
    # Set axis labels
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Entropy')
    
    # Set title
    ax.set_title('Embedding Entropy Comparison')
    
    # Set y-axis to start from 0
    ax.set_ylim(0, max(max(image_means), max(caption_means)) * 1.05)
    
    # Set x-axis to log scale with base 2
    ax.set_xscale('log', base=2)
    
    # Add grid with appropriate styling
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Add minor ticks
    ax.minorticks_on()
    
    # Adjust layout
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    
    # Save the figure in high resolution
    plt.savefig(os.path.join(output_dir, 'combined_embedding_entropy.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'combined_embedding_entropy.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    return dimensions, image_means, caption_means

if __name__ == "__main__":
    csv_file = os.path.join('data', 'embedding_entropy.csv')
    analyze_embedding_entropies(csv_file)