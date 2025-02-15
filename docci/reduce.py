import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os

def parse_embedding(embedding_str: str) -> np.ndarray:
    """Convert string representation of embedding to numpy array"""
    values = embedding_str.strip('[]').split(',')
    return np.array([float(x) for x in values])

def apply_pca(embeddings: np.ndarray, n_components: int) -> tuple:
    """
    Apply PCA to reduce dimensionality of embeddings.

    Args:
        embeddings: numpy array of embeddings
        n_components: target number of dimensions

    Returns:
        tuple: (reduced embeddings, PCA model, explained variance ratio)
    """
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    explained_variance = np.sum(pca.explained_variance_ratio_)

    return reduced_embeddings, pca, explained_variance

def process_embeddings(input_file: str, n_components: int) -> pd.DataFrame:
    """Process embeddings and reduce their dimensionality using PCA"""

    # Read the embeddings
    print("Loading embeddings...")
    df = pd.read_csv(input_file)

    # Convert embeddings to numpy arrays
    print("Processing image embeddings...")
    image_embeddings = np.vstack(df['image_embedding'].apply(parse_embedding))
    print("Processing description embeddings...")
    description_embeddings = np.vstack(df['description_embedding'].apply(parse_embedding))

    # Apply PCA to both types of embeddings
    print(f"\nReducing dimensionality to {n_components} components...")

    image_reduced, image_pca, image_variance = apply_pca(image_embeddings, n_components)
    print(f"Image embeddings explained variance: {image_variance:.4f}")

    description_reduced, description_pca, description_variance = apply_pca(description_embeddings, n_components)
    print(f"Description embeddings explained variance: {description_variance:.4f}")

    # Create output DataFrame
    results_df = pd.DataFrame({
        'image_embedding': [embedding.tolist() for embedding in image_reduced],
        'description_embedding': [embedding.tolist() for embedding in description_reduced]
    })

    return results_df

if __name__ == "__main__":
    # Set up paths
    DATA_FOLDER = "data"
    INPUT_FILE = os.path.join(DATA_FOLDER, "embeddings.csv")
    OUTPUT_FILE = os.path.join(DATA_FOLDER, "16.csv")

    # Set the target dimensionality
    N_COMPONENTS = 16

    # Process embeddings
    print("Starting PCA dimension reduction...")
    results = process_embeddings(INPUT_FILE, N_COMPONENTS)

    # Save results
    print("\nSaving reduced embeddings...")
    results.to_csv(OUTPUT_FILE, index=False)

    # Verify results
    print("\nVerifying output dimensions:")
    sample_image = np.array(results['image_embedding'].iloc[0])
    sample_description = np.array(results['description_embedding'].iloc[0])
    print(f"Reduced image embedding shape: {sample_image.shape}")
    print(f"Reduced description embedding shape: {sample_description.shape}")

    print(f"\nResults saved to {OUTPUT_FILE}")