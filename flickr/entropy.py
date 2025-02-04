import os
import cv2
import pandas as pd
import numpy as np
from collections import Counter
import math
from typing import Tuple, Union

def calculate_image_entropy(image_path: str, use_color: bool = True) -> Union[float, Tuple[float, float, float, float]]:
    """
    Calculate the entropy of an image, supporting both color and grayscale analysis.
    """
    # Read image
    if use_color:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

    def channel_entropy(channel: np.ndarray) -> float:
        """Calculate entropy for a single channel"""
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        total_pixels = channel.size
        probabilities = hist / total_pixels
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    if use_color:
        # Calculate entropy for each channel
        red_entropy = channel_entropy(img[:,:,0])
        green_entropy = channel_entropy(img[:,:,1])
        blue_entropy = channel_entropy(img[:,:,2])

        # Calculate average entropy across channels
        total_entropy = (red_entropy + green_entropy + blue_entropy) / 3

        return total_entropy, red_entropy, green_entropy, blue_entropy
    else:
        return channel_entropy(img)

def calculate_text_entropy(text: str) -> float:
    """
    Calculate the entropy of a text string.
    """
    if not isinstance(text, str) or not text:
        return 0.0

    # Split references into individual descriptions and calculate entropy for each
    descriptions = [desc.strip() for desc in text.split('.') if desc.strip()]

    entropies = []
    for description in descriptions:
        # Count character frequencies
        counter = Counter(description.lower())
        length = len(description)

        # Calculate entropy for this description
        entropy = 0
        for count in counter.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        entropies.append(entropy)

    # Return average entropy across all descriptions
    return np.mean(entropies) if entropies else 0.0

def process_dataset(image_folder: str, reference_file: str, color_entropy: bool = True) -> pd.DataFrame:
    """
    Process all images and references, calculating entropy for each.
    """
    df = pd.read_csv(reference_file)
    results = []

    for idx, row in df.iterrows():
        try:
            # Use actual image filename from CSV
            image_path = os.path.join(image_folder, row['image_filename'])

            # Calculate image entropy
            if color_entropy:
                total_entropy, r_entropy, g_entropy, b_entropy = calculate_image_entropy(
                    image_path, use_color=True
                )
            else:
                total_entropy = calculate_image_entropy(image_path, use_color=False)
                r_entropy = g_entropy = b_entropy = None

            # Calculate text entropy
            text_entropy = calculate_text_entropy(row['reference'])

            results.append({
                'image_filename': row['image_filename'],
                'reference': row['reference'],
                'image_entropy_total': total_entropy,
                'image_entropy_r': r_entropy,
                'image_entropy_g': g_entropy,
                'image_entropy_b': b_entropy,
                'text_entropy': text_entropy
            })

            if idx % 50 == 0:  # Print progress every 50 images
                print(f"Processed {idx} images...")

        except Exception as e:
            print(f"Error processing entry {idx} ({row['image_filename']}): {str(e)}")
            continue

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Set up paths
    DATA_FOLDER = "data"
    IMAGE_FOLDER = "img"
    REFERENCE_FILE = os.path.join(DATA_FOLDER, "reference500.csv")
    OUTPUT_FILE = os.path.join(DATA_FOLDER, "entropy500.csv")

    print("Starting entropy calculations...")

    # Process the dataset with color entropy
    results = process_dataset(IMAGE_FOLDER, REFERENCE_FILE, color_entropy=True)

    # Save results to CSV in the data folder
    results.to_csv(OUTPUT_FILE, index=False)

    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nImage Total Entropy:")
    print(results['image_entropy_total'].describe())
    print("\nText Entropy:")
    print(results['text_entropy'].describe())

    print(f"\nResults saved to {OUTPUT_FILE}")