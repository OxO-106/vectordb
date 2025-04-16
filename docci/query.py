import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict


def parse_embedding(embedding_str: str) -> np.ndarray:
    """Convert string representation of embedding to numpy array"""
    values = embedding_str.strip('[]').split(',')
    return np.array([float(x) for x in values])


def load_embeddings(filepath: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load embeddings from CSV and convert to numpy arrays"""
    print(f"Loading embeddings from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Parse string representations to numpy arrays
    print("Processing image embeddings...")
    image_embeddings = np.vstack([parse_embedding(emb) for emb in df['image_embedding']])
    
    print("Processing caption embeddings...")
    description_embedding = np.vstack([parse_embedding(emb) for emb in df['description_embedding']])
    
    return image_embeddings, description_embedding, df


def reduce_dimensions(description_embedding: np.ndarray, dimensions: List[int]) -> Dict[int, np.ndarray]:
    """Apply PCA to reduce caption embeddings to various dimensions"""
    reduced_embeddings = {}
    
    for dim in dimensions:
        print(f"Reducing caption embeddings to {dim} dimensions...")
        # For 512, no reduction needed
        if dim == 512:
            reduced_embeddings[dim] = description_embedding
            continue
            
        # Apply PCA
        pca = PCA(n_components=dim)
        reduced = pca.fit_transform(description_embedding)
        reduced_embeddings[dim] = reduced
        
    return reduced_embeddings


def create_composite_vectors(image_embeddings: np.ndarray, description_embedding: np.ndarray) -> np.ndarray:
    """Create composite vectors by concatenating image and caption embeddings"""
    return np.hstack((image_embeddings, description_embedding))


def measure_cosine_query_time(image_embeddings: np.ndarray, 
                             reduced_caption_embeddings: np.ndarray,
                             query_indices: List[int],
                             n_iterations: int = 3) -> float:
    """
    Measure average time to compute cosine similarity between
    query vectors and all others in the dataset.
    """
    # Create all composite vectors once
    all_composite_vectors = create_composite_vectors(image_embeddings, reduced_caption_embeddings)
    
    times = []
    for iteration in range(n_iterations):
        for idx in query_indices:
            # Get the query vector
            query_vector = all_composite_vectors[idx].reshape(1, -1)
            
            # Start timer
            start_time = time.time()
            
            # Calculate cosine similarity with all vectors
            similarities = cosine_similarity(query_vector, all_composite_vectors)[0]
            
            # End timer
            end_time = time.time()
            times.append(end_time - start_time)
            
    return np.mean(times)


def benchmark_dimensions(image_embeddings: np.ndarray, 
                        caption_embeddings: np.ndarray,
                        dimensions: List[int],
                        n_queries: int = 10,
                        n_iterations: int = 3) -> Dict[int, float]:
    """Run benchmarks for all dimension sizes"""
    
    # Apply dimension reduction
    reduced_embeddings = reduce_dimensions(caption_embeddings, dimensions)
    
    # Random indices for queries (same across all tests)
    query_indices = random.sample(range(len(image_embeddings)), n_queries)
    print(f"Selected {n_queries} random indices for queries: {query_indices}")
    
    # Measure query time for each dimension
    results = {}
    for dim in dimensions:
        print(f"Measuring query time for {dim} dimensions...")
        avg_time = measure_cosine_query_time(
            image_embeddings, 
            reduced_embeddings[dim],
            query_indices,
            n_iterations
        )
        results[dim] = avg_time
        print(f"Dimension: {dim}, Avg Query Time: {avg_time:.6f} sec")
    
    return results


def plot_results(results: Dict[int, float], dimensions: List[int]) -> None:
    """Plot the results of the benchmark"""
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot query time
    query_times = [results[dim] for dim in dimensions]
    plt.plot(dimensions, query_times, 'b-o', label='Query Time')
    
    # Add labels and title
    plt.xlabel('Dimension')
    plt.ylabel('Query Time (seconds)')
    plt.title('Dimension vs. Query Time')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig('dimension_reduction_results.png')
    print("Results plot saved as 'dimension_reduction_results.png'")
    
    # Show the plot
    plt.show()


def save_results_to_csv(results: Dict[int, float], dimensions: List[int]) -> None:
    """Save benchmark results to a CSV file"""
    df = pd.DataFrame({
        'dimension': dimensions,
        'query_time': [results[dim] for dim in dimensions]
    })
    
    filename = 'dimension_reduction_results.csv'
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def main():
    # Configuration
    input_file = "embedding.csv"
    dimensions = [4, 8, 16, 32, 64, 128, 256, 512]
    n_queries = 10  # Number of random query vectors
    n_iterations = 20  # Number of iterations to run for each query
    
    # Load data
    image_embeddings, description_embedding, _ = load_embeddings(input_file)
    
    # Run benchmarks
    results = benchmark_dimensions(
        image_embeddings, 
        description_embedding, 
        dimensions,
        n_queries,
        n_iterations
    )
    
    # Save results
    save_results_to_csv(results, dimensions)
    
    # Visualize results
    plot_results(results, dimensions)


if __name__ == "__main__":
    # Start timer for total execution
    total_start_time = time.time()
    
    # Run the main function
    main()
    
    # Calculate and print total execution time
    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.2f} seconds")