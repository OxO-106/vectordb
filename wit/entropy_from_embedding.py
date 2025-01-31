import pandas as pd
import numpy as np
from typing import Union, List
import math
import os

def calculate_embedding_entropy(embedding: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the entropy of an embedding vector.

    Args:
        embedding: CLIP embedding vector

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

def process_embeddings(data_file: str, output_file: str, column_suffix: str = '') -> None:
    """
    Process CLIP embeddings and append their entropies to the output file.

    Args:
        data_file: Path to CSV file containing embeddings
        output_file: Path to CSV file where entropies will be saved/appended
        column_suffix: Suffix to add to column names to distinguish different embedding sources
    """
    # Read the embeddings
    df = pd.read_csv(data_file)

    def parse_embedding(embedding_str: str) -> np.ndarray:
        values = embedding_str.strip('[]').split(',')
        return np.array([float(x) for x in values])

    results = []
    for idx, row in df.iterrows():
        try:
            image_embedding = parse_embedding(row['image_embedding'])
            image_embedding_entropy = calculate_embedding_entropy(image_embedding)

            caption_embedding = parse_embedding(row['caption_embedding'])
            caption_embedding_entropy = calculate_embedding_entropy(caption_embedding)

            result_row = {
                f'image_embedding_entropy{column_suffix}': image_embedding_entropy,
                f'caption_embedding_entropy{column_suffix}': caption_embedding_entropy
            }
            results.append(result_row)

        except Exception as e:
            print(f"Error processing entry {idx}: {str(e)}")
            continue

    results_df = pd.DataFrame(results)

    # If output file exists, read it and append new columns
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        combined_df = pd.concat([existing_df, results_df], axis=1)
    else:
        combined_df = results_df

    # Save results
    combined_df.to_csv(output_file, index=False)

    # Print summary statistics
    print(f"\nSummary Statistics for {os.path.basename(data_file)}:")
    print(f"\nImage Embedding Entropy{column_suffix}:")
    print(results_df[f'image_embedding_entropy{column_suffix}'].describe())
    print(f"\nCaption Embedding Entropy{column_suffix}:")
    print(results_df[f'caption_embedding_entropy{column_suffix}'].describe())

if __name__ == "__main__":
    # Set up paths
    DATA_FOLDER = "data"

    # Change this line to process different embedding files
    INPUT_FILE = os.path.join(DATA_FOLDER, "160.csv")

    # Output file remains constant
    OUTPUT_FILE = os.path.join(DATA_FOLDER, "embedding_entropy.csv")

    # Add a suffix to distinguish the source of embeddings
    # For example: "" for original embeddings, "_reduced" for reduced embeddings
    COLUMN_SUFFIX = "_160"  # Change this based on the input file type

    # Process embeddings and append results
    print(f"Processing embeddings from {os.path.basename(INPUT_FILE)}...")
    process_embeddings(INPUT_FILE, OUTPUT_FILE, COLUMN_SUFFIX)

    print(f"\nResults appended to {OUTPUT_FILE}")