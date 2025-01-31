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

def generate_caption_embedding(caption, model, processor):
    """Generate embedding for a single caption"""
    inputs = processor(text=[caption], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.squeeze().numpy()

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
    """Process all images and captions, returning a DataFrame with embeddings"""
    # Load data
    data = pd.read_csv(csv_path)

    # Setup CLIP
    model, processor = setup_clip()

    # Generate caption embeddings
    print("Generating caption embeddings...")
    data['caption_embedding'] = data['caption'].apply(
        lambda x: generate_caption_embedding(x, model, processor)
    )

    # Generate image embeddings
    print("Generating image embeddings...")
    # Using the correct image filename format: image_0.jpg
    image_embeddings = []
    for i in range(len(data)):
        img_path = os.path.join(img_folder, f"image_{i}.jpg")
        if os.path.exists(img_path):
            embedding = generate_image_embedding(img_path, model, processor)
            image_embeddings.append(embedding)
        else:
            print(f"Warning: Image not found at {img_path}")
            image_embeddings.append(None)

    data['image_embedding'] = image_embeddings

    # Convert embeddings to lists for CSV storage
    data['caption_embedding'] = data['caption_embedding'].apply(lambda x: x.tolist())
    data['image_embedding'] = data['image_embedding'].apply(lambda x: x.tolist() if x is not None else None)

    # Select only the required columns
    data = data[['image_filename', 'caption', 'image_embedding', 'caption_embedding']]

    return data

def main():
    # Set paths with data folder
    data_folder = 'data'
    csv_path = os.path.join(data_folder, 'data256.csv')
    img_folder = 'img'
    output_path = os.path.join(data_folder, 'embedding.csv')

    # Process data
    print("Starting processing...")
    data = process_all_data(csv_path, img_folder)

    # Save results
    data.to_csv(output_path, index=False)
    print(f"Processing complete. Results saved to {output_path}")

    # Verify embeddings shape
    sample_caption_embedding = np.array(data['caption_embedding'].iloc[0])
    sample_image_embedding = np.array(data['image_embedding'].iloc[0])
    print(f"\nEmbedding shapes:")
    print(f"Caption embedding: {sample_caption_embedding.shape}")
    print(f"Image embedding: {sample_image_embedding.shape}")

if __name__ == "__main__":
    main()