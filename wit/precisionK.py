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

def combine_embeddings(image_vectors, caption_vectors):
    """Concatenate image and caption embeddings"""
    return np.concatenate((image_vectors, caption_vectors), axis=1)

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
    original_caption_vecs = np.array([parse_vector(vec) for vec in df_original['caption_embedding']])
    reduced_caption_vecs = np.array([parse_vector(vec) for vec in df_reduced['caption_embedding']])

    # Remove any None values
    valid_indices = [i for i, (img, cap_orig, cap_red) in
                    enumerate(zip(original_image_vecs, original_caption_vecs, reduced_caption_vecs)) 
                    if img is not None and cap_orig is not None and cap_red is not None]

    original_image_vecs = original_image_vecs[valid_indices]
    original_caption_vecs = original_caption_vecs[valid_indices]
    reduced_caption_vecs = reduced_caption_vecs[valid_indices]

    # Combine embeddings
    # Original: 512D image + 512D caption = 1024D
    original_combined = combine_embeddings(original_image_vecs, original_caption_vecs)
    # Reduced: 512D image + 50D caption = 562D
    reduced_combined = combine_embeddings(original_image_vecs, reduced_caption_vecs)

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
                'original': original_combined.shape[1],  # should be 1024
                'reduced': reduced_combined.shape[1]     # should be 562
            }
        }

    return results

if __name__ == "__main__":
    # Evaluate combined embeddings
    results = evaluate_combined_embeddings('embedding.csv', '16.csv')

    print("\nCombined Embeddings Precision@K Results")
    print(f"(Original: 1024D to Reduced: 562D)")
    print("-" * 50)
    for k, metrics in results.items():
        print(f"K={k}:")
        print(f"  Mean Precision: {metrics['mean']:.4f}")
        print(f"  Std Deviation: {metrics['std']:.4f}")
        print(f"  Dimension: {metrics['dimensionality']}")