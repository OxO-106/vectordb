import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import json
import matplotlib.pyplot as plt
import os

def parse_vector(vec_str):
    """Parse vector string into numpy array"""
    try:
        if isinstance(vec_str, str):
            return np.array(json.loads(vec_str))
        else:
            return np.array(vec_str)
    except:
        return None

def combine_embeddings(image_vectors, caption_vectors):
    """Concatenate image and caption embeddings"""
    return np.concatenate((image_vectors, caption_vectors), axis=1)

def precision_at_k(original_neighbors, reduced_neighbors, k):
    """Calculate Precision@K by comparing neighbor sets"""
    original_set = set(original_neighbors[:k])
    reduced_set = set(reduced_neighbors[:k])
    intersection = original_set.intersection(reduced_set)
    return len(intersection) / k

def apply_pca(embeddings, n_components):
    """Apply PCA to reduce dimensionality of embeddings"""
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    return reduced_embeddings, pca, explained_variance

def evaluate_precision_for_dimension(original_image_vecs, original_caption_vecs, reduced_caption_vecs, k=10):
    """Evaluate precision@k for a specific reduced dimension"""
    # Combine embeddings
    original_combined = combine_embeddings(original_image_vecs, original_caption_vecs)
    reduced_combined = combine_embeddings(original_image_vecs, reduced_caption_vecs)
    
    # Calculate distance matrices
    original_distances = euclidean_distances(original_combined)
    reduced_distances = euclidean_distances(reduced_combined)
    
    # Get nearest neighbors (excluding self)
    n_samples = len(original_combined)
    original_neighbors = np.argsort(original_distances, axis=1)[:, 1:k+1]
    reduced_neighbors = np.argsort(reduced_distances, axis=1)[:, 1:k+1]
    
    # Calculate Precision@K
    precisions = []
    for i in range(n_samples):
        p_at_k = precision_at_k(original_neighbors[i], reduced_neighbors[i], k)
        precisions.append(p_at_k)
    
    avg_precision = np.mean(precisions)
    std_precision = np.std(precisions)
    
    return {
        'mean': avg_precision,
        'std': std_precision,
        'dimensionality': reduced_caption_vecs.shape[1]
    }

def calculate_rate_of_change(precision_results):
    """Calculate rate of change of precision between consecutive dimensions"""
    dims = sorted([dim for dim in precision_results.keys()])
    rate_of_change = []
    
    for i in range(1, len(dims)):
        curr_dim = dims[i]
        prev_dim = dims[i-1]
        curr_precision = precision_results[curr_dim]['mean']
        prev_precision = precision_results[prev_dim]['mean']
        
        # Calculate absolute difference
        abs_diff = curr_precision - prev_precision
        
        # Calculate rate of change (normalized by dimension difference)
        dim_diff = curr_dim - prev_dim
        rate = abs_diff / dim_diff
        
        rate_of_change.append({
            'dimension_range': f"{prev_dim}-{curr_dim}",
            'prev_dimension': prev_dim,
            'curr_dimension': curr_dim,
            'prev_precision': prev_precision,
            'curr_precision': curr_precision,
            'abs_difference': abs_diff,
            'rate_of_change': rate
        })
    
    return rate_of_change

def main():
    # Set up data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Input file path
    input_file = os.path.join(data_dir, "embedding.csv")
    
    # Read the original embeddings
    print("Loading embeddings...")
    df_original = pd.read_csv(input_file)
    
    # Parse original vectors
    print("Parsing original vectors...")
    original_image_vecs = np.array([parse_vector(vec) for vec in df_original['image_embedding']])
    original_caption_vecs = np.array([parse_vector(vec) for vec in df_original['caption_embedding']])
    
    # Remove any None values
    print("Filtering valid vectors...")
    valid_indices = [i for i, (img, cap) in enumerate(zip(original_image_vecs, original_caption_vecs)) 
                    if img is not None and cap is not None]
    
    original_image_vecs = np.array([original_image_vecs[i] for i in valid_indices])
    original_caption_vecs = np.array([original_caption_vecs[i] for i in valid_indices])
    
    # Make sure all vectors have the same dimensions
    print(f"Original image vectors shape: {original_image_vecs.shape}")
    print(f"Original caption vectors shape: {original_caption_vecs.shape}")
    
    # Set dimensions to analyze (multiples of 4 up to 320)
    dimensions = list(range(4, 321, 4))
    
    # Store precision results
    precision_results = {}
    precision_data = []
    
    # Process each dimension
    print("\nAnalyzing dimensions...")
    for dim in dimensions:
        print(f"Processing dimension {dim}...")
        
        # Apply PCA to caption embeddings
        reduced_caption_vecs, _, variance = apply_pca(original_caption_vecs, dim)
        
        # Evaluate precision
        result = evaluate_precision_for_dimension(
            original_image_vecs, 
            original_caption_vecs, 
            reduced_caption_vecs,
            k=10
        )
        
        # Store results
        precision_results[dim] = result
        precision_data.append({
            'dimension': dim,
            'precision_at_10': result['mean'],
            'explained_variance': variance
        })
    
    # Calculate rate of change
    print("\nCalculating rate of change...")
    rate_of_change = calculate_rate_of_change(precision_results)
    
    # Sort rate of change from highest to lowest
    sorted_rate_of_change = sorted(rate_of_change, key=lambda x: x['rate_of_change'], reverse=True)
    
    # Save precision results to CSV
    print("Saving precision results...")
    precision_df = pd.DataFrame(precision_data)
    precision_df.to_csv(os.path.join(data_dir, "precision4.csv"), index=False)
    
    # Save rate of change to CSV
    print("Saving rate of change results...")
    rate_df = pd.DataFrame(sorted_rate_of_change)
    rate_df.to_csv(os.path.join(data_dir, "rateofchange.csv"), index=False)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Create directory for plots
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot precision vs. dimension
    plt.figure(figsize=(12, 6))
    plt.plot([d['dimension'] for d in precision_data], 
             [d['precision_at_10'] for d in precision_data], 
             'o-', markersize=4)
    plt.xlabel('Dimension')
    plt.ylabel('Precision@10')
    plt.title('Precision@10 vs. Dimension')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, "precision_vs_dimension.png"), dpi=300, bbox_inches='tight')
    
    # Plot rate of change vs. dimension
    plt.figure(figsize=(12, 6))
    plt.bar([f"{r['prev_dimension']}-{r['curr_dimension']}" for r in rate_of_change[::8]],  # Use every 8th label to avoid crowding
            [r['rate_of_change'] for r in rate_of_change[::8]],
            width=0.7)
    plt.xlabel('Dimension Range')
    plt.ylabel('Rate of Change')
    plt.title('Rate of Change in Precision@10 vs. Dimension Range')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "rate_of_change.png"), dpi=300, bbox_inches='tight')
    
    # Plot top 10 highest rate of change
    top_10_roc = sorted_rate_of_change[:10]
    plt.figure(figsize=(12, 6))
    plt.bar([f"{r['prev_dimension']}-{r['curr_dimension']}" for r in top_10_roc],
            [r['rate_of_change'] for r in top_10_roc],
            width=0.7)
    plt.xlabel('Dimension Range')
    plt.ylabel('Rate of Change')
    plt.title('Top 10 Dimension Ranges with Highest Rate of Change')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "top10_rate_of_change.png"), dpi=300, bbox_inches='tight')
    
    print("\nAnalysis complete!")
    print(f"Precision results saved to: {os.path.join(data_dir, 'precision4.csv')}")
    print(f"Rate of change results saved to: {os.path.join(data_dir, 'rateofchange.csv')}")
    print(f"Plots saved to: {plots_dir}")
    
    # Print top 5 dimensions with highest rate of change
    print("\nTop 5 dimension ranges with highest rate of change:")
    for i, r in enumerate(sorted_rate_of_change[:10], 1):
        print(f"{i}. {r['dimension_range']}: {r['rate_of_change']:.6f}")

if __name__ == "__main__":
    main()