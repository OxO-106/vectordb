import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import json

def parse_vector(vec_str):
    """Parse vector string into numpy array"""
    try:
        return np.array(json.loads(vec_str))
    except:
        return None

def combine_embeddings(image_vectors, description_vectors):
    """Concatenate image and description embeddings"""
    return np.concatenate((image_vectors, description_vectors), axis=1)

def precision_at_k(original_neighbors, reduced_neighbors, k):
    """Calculate Precision@K by comparing neighbor sets"""
    original_set = set(original_neighbors[:k])
    reduced_set = set(reduced_neighbors[:k])
    intersection = original_set.intersection(reduced_set)
    return len(intersection) / k

def evaluate_combined_embeddings(original_file, reduced_file, k_values=[5, 10, 15]):
    # Read CSV files
    df_original = pd.read_csv(f'data/{original_file}')  # 320.csv with 512D embeddings
    df_reduced = pd.read_csv(f'data/{reduced_file}')    # 50.csv with reduced embeddings

    # Parse vectors
    original_image_vecs = np.array([parse_vector(vec) for vec in df_original['image_embedding']])
    original_description_vecs = np.array([parse_vector(vec) for vec in df_original['description_embedding']])
    reduced_description_vecs = np.array([parse_vector(vec) for vec in df_reduced['description_embedding']])

    # Remove any None values
    valid_indices = [i for i, (img, cap_orig, cap_red) in
                    enumerate(zip(original_image_vecs, original_description_vecs, reduced_description_vecs))
                    if img is not None and cap_orig is not None and cap_red is not None]

    original_image_vecs = original_image_vecs[valid_indices]
    original_description_vecs = original_description_vecs[valid_indices]
    reduced_description_vecs = reduced_description_vecs[valid_indices]

    # Combine embeddings
    original_combined = combine_embeddings(original_image_vecs, original_description_vecs)
    reduced_combined = combine_embeddings(original_image_vecs, reduced_description_vecs)

    # Calculate distance matrices
    original_distances = euclidean_distances(original_combined)
    reduced_distances = euclidean_distances(reduced_combined)

    # Get nearest neighbors (excluding self)
    n_samples = len(original_combined)
    max_k = max(k_values)

    original_neighbors = np.argsort(original_distances, axis=1)[:, 1:max_k+1]
    reduced_neighbors = np.argsort(reduced_distances, axis=1)[:, 1:max_k+1]

    # Calculate Precision@K for each k
    results = {}
    for k in k_values:
        precisions = []
        for i in range(n_samples):
            p_at_k = precision_at_k(original_neighbors[i], reduced_neighbors[i], k)
            precisions.append(p_at_k)

        avg_precision = np.mean(precisions)
        std_precision = np.std(precisions)
        results[k] = {
            'mean': avg_precision,
            'std': std_precision,
            'dimensionality': {
                'original': original_combined.shape[1],
                'reduced': reduced_combined.shape[1]
            }
        }

    return results

def save_precision_results(results, reduced_dim, output_file='data/precision.csv'):
    """Save precision results to CSV file, appending if file exists"""
    # Create a dictionary with the required format
    data = {
        'reduced_dimension': [reduced_dim],
        'precision_at_5': [results[5]['mean']],
        'precision_at_10': [results[10]['mean']],
        'precision_at_15': [results[15]['mean']]
    }

    # Convert new data to DataFrame
    new_df = pd.DataFrame(data)

    try:
        # Try to read existing CSV file
        existing_df = pd.read_csv(output_file)
        # Append new data to existing data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Sort by reduced_dimension for better organization
        combined_df = combined_df.sort_values('reduced_dimension').reset_index(drop=True)
    except FileNotFoundError:
        # If file doesn't exist, use only new data
        combined_df = new_df

    # Save to CSV
    combined_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Evaluate combined embeddings
    results = evaluate_combined_embeddings('embeddings.csv', '16.csv')
    reduced_dim = 16

    # Print results
    print("\nCombined Embeddings Precision@K Results")
    print("-" * 50)
    for k, metrics in results.items():
        print(f"K={k}:")
        print(f"  Mean Precision: {metrics['mean']:.4f}")
        print(f"  Std Deviation: {metrics['std']:.4f}")
        print(f"  Dimension: {metrics['dimensionality']}")

    # Save results to CSV
    save_precision_results(results, reduced_dim)
    print(f"\nResults have been saved to precision.csv")