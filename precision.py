import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from typing import List, Tuple

def load_embeddings(original_file: str, reduced_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load original image embeddings and both original and reduced caption embeddings.
    
    Args:
        original_file: Path to the file containing original embeddings
        reduced_file: Path to the file containing reduced embeddings
        
    Returns:
        Tuple of (image_embeddings, original_caption_embeddings, reduced_caption_embeddings)
    """
    # Load original embeddings
    original_df = pd.read_csv(original_file)
    image_embeddings = np.array([eval(emb) for emb in original_df['image_embedding']])
    original_caption_embeddings = np.array([eval(emb) for emb in original_df['caption_embedding']])
    
    # Load reduced embeddings
    reduced_df = pd.read_csv(reduced_file)
    reduced_caption_embeddings = np.array([eval(emb) for emb in reduced_df['caption_embedding']])
    
    return image_embeddings, original_caption_embeddings, reduced_caption_embeddings

def compute_similarity_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings
        
    Returns:
        Similarity matrix
    """
    return cosine_similarity(embeddings1, embeddings2)

def precision_at_k(similarity_matrix: np.ndarray, k: int) -> float:
    """
    Calculate precision@k for the similarity matrix.
    
    Args:
        similarity_matrix: Matrix of similarity scores
        k: Number of top predictions to consider
        
    Returns:
        Precision@k score
    """
    num_samples = similarity_matrix.shape[0]
    correct_matches = 0
    
    for i in range(num_samples):
        # Get top k predictions
        top_k_indices = np.argsort(similarity_matrix[i])[-k:]
        # Check if the correct match (index i) is in top k
        if i in top_k_indices:
            correct_matches += 1
    
    return correct_matches / num_samples

def evaluate_embeddings(image_embeddings: np.ndarray, 
                       caption_embeddings: np.ndarray,
                       k_values: List[int]) -> dict:
    """
    Evaluate embeddings using precision@k for multiple k values.
    
    Args:
        image_embeddings: Image embeddings
        caption_embeddings: Caption embeddings
        k_values: List of k values to evaluate
        
    Returns:
        Dictionary of precision@k scores for each k
    """
    similarity_matrix = compute_similarity_matrix(image_embeddings, caption_embeddings)
    results = {}
    
    for k in k_values:
        precision = precision_at_k(similarity_matrix, k)
        results[f'precision@{k}'] = precision
    
    return results

def main():
    # File paths
    original_file = 'data/embedding.csv'
    reduced_file = 'data/160.csv'
    
    # Load embeddings
    image_embeddings, original_caption_embeddings, reduced_caption_embeddings = load_embeddings(
        original_file, reduced_file
    )
    
    # K values to evaluate
    k_values = [1, 5, 10]
    
    # Evaluate original embeddings
    print("Evaluating original embeddings...")
    original_results = evaluate_embeddings(
        image_embeddings, original_caption_embeddings, k_values
    )
    
    # Reduce image embeddings to match the dimension of reduced caption embeddings
    print("\nReducing image embeddings dimension...")
    pca = PCA(n_components=160)
    reduced_image_embeddings = pca.fit_transform(image_embeddings)
    
    # Evaluate reduced embeddings
    print("Evaluating reduced embeddings...")
    reduced_results = evaluate_embeddings(
        reduced_image_embeddings, reduced_caption_embeddings, k_values
    )
    
    # Print results
    print("\nResults for original embeddings:")
    for k, score in original_results.items():
        print(f"{k}: {score:.4f}")
        
    print("\nResults for reduced embeddings:")
    for k, score in reduced_results.items():
        print(f"{k}: {score:.4f}")

if __name__ == "__main__":
    main()