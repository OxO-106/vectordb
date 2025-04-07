import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import json
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
    return reduced_embeddings, pca

def evaluate_precision_for_dimensions(original_image_vecs, original_caption_vecs, reduced_caption_vecs, k_values=[5, 10, 15]):
    """Evaluate precision@k for a specific reduced dimension at multiple k values"""
    # Get the maximum k from k_values
    max_k = max(k_values)

    # Combine embeddings
    original_combined = combine_embeddings(original_image_vecs, original_caption_vecs)
    reduced_combined = combine_embeddings(original_image_vecs, reduced_caption_vecs)

    # Calculate distance matrices
    original_distances = euclidean_distances(original_combined)
    reduced_distances = euclidean_distances(reduced_combined)

    # Get nearest neighbors (excluding self) for max_k
    n_samples = len(original_combined)
    original_neighbors = np.argsort(original_distances, axis=1)[:, 1:max_k+1]
    reduced_neighbors = np.argsort(reduced_distances, axis=1)[:, 1:max_k+1]

    # Results dictionary for different k values
    results = {}

    # Calculate Precision@K for each k value
    for k in k_values:
        precisions = []
        for i in range(n_samples):
            p_at_k = precision_at_k(original_neighbors[i][:k], reduced_neighbors[i][:k], k)
            precisions.append(p_at_k)

        avg_precision = np.mean(precisions)
        std_precision = np.std(precisions)

        results[k] = {
            'mean': avg_precision,
            'std': std_precision
        }

    results['dimensionality'] = reduced_caption_vecs.shape[1]

    return results

def main():
    # Set up data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Input file path
    input_file = os.path.join(data_dir, "embeddings.csv")

    # Read the original embeddings
    print("Loading embeddings...")
    df_original = pd.read_csv(input_file)

    # Parse original vectors
    print("Parsing original vectors...")
    original_image_vecs = np.array([parse_vector(vec) for vec in df_original['image_embedding']])
    original_caption_vecs = np.array([parse_vector(vec) for vec in df_original['reference_embedding']])

    # Remove any None values
    print("Filtering valid vectors...")
    valid_indices = [i for i, (img, cap) in enumerate(zip(original_image_vecs, original_caption_vecs)) 
                    if img is not None and cap is not None]

    original_image_vecs = np.array([original_image_vecs[i] for i in valid_indices])
    original_caption_vecs = np.array([original_caption_vecs[i] for i in valid_indices])

    # Make sure all vectors have the same dimensions
    print(f"Original image vectors shape: {original_image_vecs.shape}")
    print(f"Original caption vectors shape: {original_caption_vecs.shape}")

    # Set dimensions to analyze (1, then multiples of 2 from 2 to 320)
    dimensions = [1] + list(range(2, 321, 2))

    # Store precision results
    precision_data = []

    # Process each dimension
    print("Analyzing dimensions...")
    for dim in dimensions:
        # Apply PCA to caption embeddings
        reduced_caption_vecs, _ = apply_pca(original_caption_vecs, dim)

        # Evaluate precision at multiple k values
        result = evaluate_precision_for_dimensions(
            original_image_vecs,
            original_caption_vecs,
            reduced_caption_vecs,
            k_values=[5, 10, 15]
        )

        # Store results
        precision_data.append({
            'dimension': dim,
            'precision_at_5': result[5]['mean'],
            'precision_at_10': result[10]['mean'],
            'precision_at_15': result[15]['mean']
        })

    # Save precision results to CSV
    print("Saving precision results...")
    precision_df = pd.DataFrame(precision_data)
    precision_df.to_csv(os.path.join(data_dir, "precision4.csv"), index=False)

    print("\nAnalysis complete!")
    print(f"Precision results saved to: {os.path.join(data_dir, 'precision4.csv')}")

if __name__ == "__main__":
    main()