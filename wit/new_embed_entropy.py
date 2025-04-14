import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
import math
from typing import Union, List, Tuple

def calculate_embedding_entropy(embedding: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the entropy of an embedding vector.

    Args:
        embedding: Embedding vector

    Returns:
        float: Entropy value of the embedding
    """
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)

    shifted = embedding - np.min(embedding)
    if np.sum(shifted) != 0:
        probabilities = shifted / np.sum(shifted)
    else:
        return 0.0

    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy

def parse_embedding(embedding_str: str) -> np.ndarray:
    """Convert string representation of embedding to numpy array"""
    if isinstance(embedding_str, str):
        values = embedding_str.strip('[]').split(',')
        return np.array([float(x) for x in values])
    else:
        return np.array(embedding_str)

def apply_pca(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """
    Apply PCA to reduce dimensionality of embeddings.

    Args:
        embeddings: numpy array of embeddings
        n_components: target number of dimensions

    Returns:
        np.ndarray: reduced embeddings
    """
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def process_embeddings(input_file: str, output_file: str) -> None:
    """
    Process embeddings by applying PCA for various dimensions and calculating entropy.
    
    Args:
        input_file: Path to CSV file containing original embeddings
        output_file: Path to CSV file where results will be saved
    """
    # Read the embeddings
    df = pd.read_csv(input_file)
    
    # Convert embeddings to numpy arrays
    image_embeddings = np.vstack([parse_embedding(e) for e in df['image_embedding']])
    caption_embeddings = np.vstack([parse_embedding(e) for e in df['caption_embedding']])
    
    # Determine the original embedding dimension
    original_dim = image_embeddings.shape[1]
    
    # Create results DataFrame
    results = pd.DataFrame()
    
    # Process dimensions from 1 to original in steps of 2, starting from 2
    dimensions = [1] + list(range(2, min(original_dim + 1, 513), 2))
    dimensions.append(original_dim)  # Ensure original dimension is included
    dimensions = sorted(list(set(dimensions)))  # Remove duplicates
    
    for dim in dimensions:
        if dim < original_dim:
            # Apply PCA reduction
            image_reduced = apply_pca(image_embeddings, dim)
            caption_reduced = apply_pca(caption_embeddings, dim)
        else:
            # Use original embeddings if dimension matches
            image_reduced = image_embeddings
            caption_reduced = caption_embeddings
        
        # Calculate entropy for all embeddings at this dimension
        image_entropies = [calculate_embedding_entropy(emb) for emb in image_reduced]
        caption_entropies = [calculate_embedding_entropy(emb) for emb in caption_reduced]
        
        # Prepare results for this dimension
        dim_results = pd.DataFrame({
            f'image_embedding_entropy_{dim}': image_entropies,
            f'caption_embedding_entropy_{dim}': caption_entropies
        })
        
        # Append to overall results
        if results.empty:
            results = dim_results
        else:
            results = pd.concat([results, dim_results], axis=1)
    
    # Save the results
    results.to_csv(output_file, index=False)
    
    # Generate summary statistics
    summary_data = []
    for dim in dimensions:
        img_mean = results[f'image_embedding_entropy_{dim}'].mean()
        cap_mean = results[f'caption_embedding_entropy_{dim}'].mean()
        
        summary_data.append({
            'dimension': dim,
            'avg_image_entropy': img_mean,
            'avg_caption_entropy': cap_mean
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.splitext(output_file)[0] + "_summary.csv"
    summary_df.to_csv(summary_file, index=False)

if __name__ == "__main__":
    # Set up paths
    DATA_FOLDER = "data"
    INPUT_FILE = os.path.join("embedding.csv")
    OUTPUT_FILE = os.path.join("embed_entropy.csv")
    
    # Process embeddings
    process_embeddings(INPUT_FILE, OUTPUT_FILE)