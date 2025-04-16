import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set high-quality plot parameters
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

# Load the data from precision16.csv
data = pd.read_csv('precision16.csv')

# Define metrics to plot
labels = ['precision_at_5', 'precision_at_10', 'precision_at_15']
titles = ['Precision@5', 'Precision@10', 'Precision@15']

# Filter dimensions between 300 and 512
data = data[(data['dimension'] >= 256) & (data['dimension'] <= 512)]

# Create histograms with full y-axis range (0 to 1.05)
for i, (label, title) in enumerate(zip(labels, titles)):
    # Create a new figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Extract x and y data
    x_data = data['dimension'].values
    y_data = data[label].values
    
    # Set x-axis limits
    x_min = min(x_data) - 10
    x_max = max(x_data) + 10
    
    # Calculate the width for the bars - wider bars for 16-dimension intervals
    width = 14  # Slightly narrower than the interval to have small gaps
    
    # Plot the histogram bars
    bars = ax.bar(x_data, y_data, width=width, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Set axis labels
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Precision')
    
    # Set title
    k_value = label.split('_')[-1]
    ax.set_title(f'Precision@{k_value} Histogram')
    
    # Set y-axis to start from 0 and end at 1.05
    ax.set_ylim(0.95, 1.01)
    
    # Set the x-axis limits
    ax.set_xlim(x_min, x_max)
    
    # Add minor ticks for y-axis only (x-axis would be too crowded)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    
    # Set major x-ticks at intervals to avoid overcrowding
    x_tick_interval = 32  # Adjust based on your dimension intervals
    x_ticks = np.arange(256, max(data['dimension']) + 1, x_tick_interval)
    ax.set_xticks(x_ticks)
    
    # Add grid for y-axis only
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(f'flickr_histogram_precision_{k_value}.png', dpi=600, bbox_inches='tight')
    
    plt.close(fig)

print("All 6 histogram plots have been generated successfully.")