import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def analyze_embedding_entropies(csv_file):
    """
    Analyze and visualize embedding entropies for different dimensions
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Print available columns
    print("Available columns in CSV:")
    print(df.columns.tolist())

    # Get dimensions from column names dynamically
    dimensions = []
    image_means = []
    description_means = []
    # Find all dimensions present in the column names
    for col in df.columns:
        if col.startswith('image_embedding_entropy_'):
            dim = int(col.split('_')[-1])
            dimensions.append(dim)

    # Sort dimensions in descending order
    dimensions.sort(reverse=True)
    print(f"\nFound dimensions: {dimensions}")

    # Calculate means for each dimension
    for dim in dimensions:
        image_col = f'image_embedding_entropy_{dim}'
        description_col = f'description_embedding_entropy_{dim}'

        image_means.append(df[image_col].mean())
        description_means.append(df[description_col].mean())
        print(f"\nDimension {dim}:")
        print(f"Image mean: {df[image_col].mean():.4f}")
        print(f"Description mean: {df[description_col].mean():.4f}")

    # Set the style to a clean, modern look
    plt.style.use('classic')

    # Plot 1: Image Embedding Entropies
    plt.figure(figsize=(12, 7))
    plt.plot(dimensions, image_means, marker='o', linewidth=2, markersize=8, color='#2196F3')
    plt.xscale('log', base=2)
    plt.xlabel('Embedding Dimension', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Entropy', fontsize=12, fontweight='bold')
    plt.title('Mean Image Embedding Entropy vs. Dimension', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add value annotations
    for x, y in zip(dimensions, image_means):
        plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=10)

    # Improve the layout
    plt.tight_layout()
    plt.savefig(os.path.join('data', 'image_embedding_entropy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Description Embedding Entropies
    plt.figure(figsize=(12, 7))
    plt.plot(dimensions, description_means, marker='o', linewidth=2, markersize=8, color='#4CAF50')
    plt.xscale('log', base=2)
    plt.xlabel('Embedding Dimension', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Entropy', fontsize=12, fontweight='bold')
    plt.title('Mean Description Embedding Entropy vs. Dimension', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add value annotations
    for x, y in zip(dimensions, description_means):
        plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=10)

    # Improve the layout
    plt.tight_layout()
    plt.savefig(os.path.join('data', 'description_embedding_entropy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Analysis
    print("\nAnalysis of Embedding Entropies:")
    print("-" * 50)

    # Calculate entropy per dimension
    print("\nEntropy per dimension:")
    for dim, img_mean, desc_mean in zip(dimensions, image_means, description_means):
        entropy_per_dim_img = img_mean / dim
        entropy_per_dim_desc = desc_mean / dim
        print(f"\nDimension {dim}:")
        print(f"Image: {entropy_per_dim_img:.4f} bits/dimension")
        print(f"Description: {entropy_per_dim_desc:.4f} bits/dimension")

    # Calculate reduction ratios between consecutive dimensions
    print("\nEntropy reduction ratios between consecutive dimensions:")
    for i in range(len(dimensions)-1):
        img_ratio = image_means[i+1] / image_means[i]
        desc_ratio = description_means[i+1] / description_means[i]
        print(f"\nFrom {dimensions[i]} to {dimensions[i+1]}:")
        print(f"Image: {img_ratio:.4f}")
        print(f"Description: {desc_ratio:.4f}")

if __name__ == "__main__":
    csv_file = os.path.join('data', 'embedding_entropy.csv')
    analyze_embedding_entropies(csv_file)