import torch
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from PIL import Image
import os
import numpy as np

def setup_clip():
    """Initialize CLIP model and processor"""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def generate_reference_embedding(reference, model, processor):
    """Generate embedding for a single reference text"""
    # Split the reference into individual descriptions
    descriptions = [desc.strip() for desc in reference.split('.') if desc.strip()]

    # Generate embeddings for each description
    embeddings = []
    for desc in descriptions:
        inputs = processor(text=[desc], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            embeddings.append(text_features.squeeze().numpy())

    # Average the embeddings
    return np.mean(embeddings, axis=0)

def generate_image_embedding(image_path, model, processor):
    """Generate embedding for a single image"""
    try:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        return image_features.squeeze().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def process_all_data(csv_path, img_folder):
    """Process all images and references, returning a DataFrame with embeddings"""
    # Load data
    data = pd.read_csv(csv_path)

    # Setup CLIP
    model, processor = setup_clip()

    # Generate reference embeddings
    print("Generating reference embeddings...")
    data['reference_embedding'] = data['reference'].apply(
        lambda x: generate_reference_embedding(x, model, processor)
    )

    # Generate image embeddings
    print("Generating image embeddings...")
    image_embeddings = []
    for filename in data['image_filename']:
        img_path = os.path.join(img_folder, filename)
        if os.path.exists(img_path):
            embedding = generate_image_embedding(img_path, model, processor)
            image_embeddings.append(embedding)
        else:
            print(f"Warning: Image not found at {img_path}")
            image_embeddings.append(None)

    data['image_embedding'] = image_embeddings

    # Convert embeddings to lists for CSV storage
    data['reference_embedding'] = data['reference_embedding'].apply(lambda x: x.tolist())
    data['image_embedding'] = data['image_embedding'].apply(lambda x: x.tolist() if x is not None else None)

    return data

def main():
    # Set paths
    data_folder = 'data'
    csv_path = os.path.join(data_folder, 'reference500.csv')
    img_folder = 'img'
    output_path = os.path.join(data_folder, 'embeddings.csv')

    # Process data
    print("Starting processing...")
    data = process_all_data(csv_path, img_folder)

    # Save results
    data.to_csv(output_path, index=False)
    print(f"Processing complete. Results saved to {output_path}")

    # Verify embeddings shape
    sample_reference_embedding = np.array(data['reference_embedding'].iloc[0])
    sample_image_embedding = np.array(data['image_embedding'].iloc[0])
    print(f"\nEmbedding shapes:")
    print(f"Reference embedding: {sample_reference_embedding.shape}")
    print(f"Image embedding: {sample_image_embedding.shape}")

if __name__ == "__main__":
    main()