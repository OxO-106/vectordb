import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

def load_embeddings(embedding_path: str, reduced_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings from CSV files.

    Args:
        embedding_path: Path to CSV containing 512D embeddings
        reduced_path: Path to CSV containing 50D text embeddings

    Returns:
        Tuple of (image_embeddings, text_embeddings_512d, text_embeddings_50d)
    """
    # Load original embeddings
    df_original = pd.read_csv(embedding_path)

    # Convert string embeddings to numpy arrays
    image_embeddings = np.array([
        list(map(float, vec.strip('[]').split(',')))
        for vec in df_original['image_embedding']
    ])

    text_embeddings_512d = np.array([
        list(map(float, vec.strip('[]').split(',')))
        for vec in df_original['caption_embedding']
    ])

    # Load reduced text embeddings
    df_reduced = pd.read_csv(reduced_path)
    text_embeddings_50d = np.array([
        list(map(float, vec.strip('[]').split(',')))
        for vec in df_reduced.iloc[:, 0]  # Assuming the embeddings are in the first column
    ])

    return image_embeddings, text_embeddings_512d, text_embeddings_50d

def normalize_and_combine_embeddings(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    weights: Optional[Tuple[float, float]] = (0.5, 0.5)
) -> np.ndarray:
    """
    Normalize and combine image and text embeddings with optional weighting.
    """
    # Normalize each embedding type separately
    image_norm = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_norm = np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    norm_image_embeddings = image_embeddings / image_norm
    norm_text_embeddings = text_embeddings / text_norm

    # Apply weights and concatenate
    image_weight, text_weight = weights
    combined = np.concatenate([
        norm_image_embeddings * image_weight,
        norm_text_embeddings * text_weight
    ], axis=1)

    return combined

def get_k_nearest_neighbors(similarity_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Get indices of k-nearest neighbors for each vector based on similarity matrix.
    """
    neighbor_indices = np.argsort(-similarity_matrix, axis=1)
    return neighbor_indices[:, 1:k+1]

def evaluate_combined_embeddings(
    image_embeddings: np.ndarray,
    original_text_embeddings: np.ndarray,
    reduced_text_embeddings: np.ndarray,
    k: int = 5,
    weights: Optional[Tuple[float, float]] = (0.5, 0.5)
) -> Tuple[float, float, List[float], List[float]]:
    """
    Evaluate precision@k for combined embeddings using original and reduced text dimensions.
    """
    # Combine embeddings for both spaces
    original_combined = normalize_and_combine_embeddings(
        image_embeddings, original_text_embeddings, weights
    )
    reduced_combined = normalize_and_combine_embeddings(
        image_embeddings, reduced_text_embeddings, weights
    )

    # Calculate similarities in both spaces
    original_similarities = cosine_similarity(original_combined)
    reduced_similarities = cosine_similarity(reduced_combined)

    # Get k-nearest neighbors
    original_neighbors = get_k_nearest_neighbors(original_similarities, k)
    reduced_neighbors = get_k_nearest_neighbors(reduced_similarities, k)

    # Calculate precision@k for each sample
    original_precisions = []
    reduced_precisions = []

    for i in range(len(image_embeddings)):
        original_set = set(original_neighbors[i])
        reduced_set = set(reduced_neighbors[i])

        common_neighbors = original_set.intersection(reduced_set)
        precision = len(common_neighbors) / k

        original_precisions.append(1.0)  # Original space is reference
        reduced_precisions.append(precision)

    avg_original = np.mean(original_precisions)
    avg_reduced = np.mean(reduced_precisions)

    return avg_original, avg_reduced, original_precisions, reduced_precisions

def plot_results(reduced_precisions: List[float], weights: Tuple[float, float]):
    """
    Plot histogram of precision scores.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(reduced_precisions, bins=30)
    plt.title(f'Distribution of Precision@K Scores\nImage Weight: {weights[0]}, Text Weight: {weights[1]}')
    plt.xlabel('Precision@K')
    plt.ylabel('Count')
    plt.show()

if __name__ == "__main__":
    # Load embeddings from files
    image_emb, text_emb_512d, text_emb_50d = load_embeddings(
        'data/embedding.csv',
        'data/256.csv'
    )

    # Equal weights for image and text embeddings
    # Other possible weight combinations:
    # weight_combinations = [
    #     (0.7, 0.3),  # More weight to image
    #     (0.3, 0.7)   # More weight to text
    # ]

    img_w, txt_w = 0.5, 0.5  # Equal weights

    # Create results string
    results = []
    results.append("Evaluating with equal weights (0.5 each)")

    print("\nEvaluating with equal weights (0.5 each)")
    avg_orig, avg_red, orig_prec, red_prec = evaluate_combined_embeddings(
        image_emb,
        text_emb_512d,
        text_emb_50d,
        k=5,
        weights=(img_w, txt_w)
    )

    # Format results
    results.append(f"Original Space Average P@5: {avg_orig:.3f}")
    results.append(f"Reduced Space Average P@5: {avg_red:.3f}")
    results.append(f"Relative Performance: {(avg_red/avg_orig)*100:.1f}%")

    # Print results to console
    for line in results:
        print(line)

    # Save results to file (append mode)
    with open('data/precision.txt', 'a') as f:
        f.write('\n' + '\n'.join(results) + '\n')

    # Plot distribution of precision scores
    plot_results(red_prec, (img_w, txt_w))