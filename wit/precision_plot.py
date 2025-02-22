import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_precision_curves(csv_file='data/precision.csv', output_dir='data'):
    """
    Create three separate plots for precision@k values (k=5,10,15)
    against reduced dimensions
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Set the style to a built-in style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot settings
    plot_settings = {
        5: {'ax': ax1, 'color': '#1f77b4', 'title': 'Precision@5'},
        10: {'ax': ax2, 'color': '#2ca02c', 'title': 'Precision@10'},
        15: {'ax': ax3, 'color': '#d62728', 'title': 'Precision@15'}
    }
    
    # Create plots
    for k, settings in plot_settings.items():
        ax = settings['ax']
        column = f'precision_at_{k}'
        
        # Create the line plot
        ax.plot(df['reduced_dimension'], df[column], 
                color=settings['color'], 
                marker='o', 
                linewidth=2, 
                markersize=8)
        
        # Add scatter points
        ax.scatter(df['reduced_dimension'], df[column], 
                  color=settings['color'], 
                  s=100, 
                  alpha=0.6)
        
        # Customize the plot
        ax.set_title(settings['title'], fontsize=14, pad=15)
        ax.set_xlabel('Reduced Dimension', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis limits slightly above and below the data range
        y_min = df[column].min() * 0.95
        y_max = df[column].max() * 1.05
        ax.set_ylim(y_min, y_max)
        
        # Add value labels on the points
        for x, y in zip(df['reduced_dimension'], df[column]):
            ax.annotate(f'{y:.3f}', 
                       (x, y), 
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center',
                       fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure in the data folder
    output_path = os.path.join(output_dir, 'precision_plots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plots have been saved as '{output_path}'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_precision_curves()