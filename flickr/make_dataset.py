import pandas as pd
import shutil
import os

def create_dataset(source_img_dir, dest_img_dir, csv_path, output_dir, n_samples=500):
    """
    Extract a subset of images and their references to create a smaller dataset.
    
    Parameters:
    source_img_dir (str): Directory containing source images
    dest_img_dir (str): Directory to store selected images
    csv_path (str): Path to the original CSV file
    output_dir (str): Directory to store the output CSV file
    n_samples (int): Number of samples to extract (default: 500)
    """
    # Create output directories if they don't exist
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Take the first n_samples rows
    df_subset = df.head(n_samples)
    
    # Copy images
    for image_filename in df_subset['image_filename']:
        source_path = os.path.join(source_img_dir, image_filename)
        dest_path = os.path.join(dest_img_dir, image_filename)
        
        try:
            shutil.copy2(source_path, dest_path)
            print(f"Copied {image_filename}")
        except FileNotFoundError:
            print(f"Warning: Could not find {image_filename}")
        except Exception as e:
            print(f"Error copying {image_filename}: {str(e)}")
    
    # Save the subset of references to a new CSV file
    output_csv_path = os.path.join(output_dir, 'reference500.csv')
    df_subset.to_csv(output_csv_path, index=False)
    print(f"\nProcess completed:")
    print(f"- {len(df_subset)} references saved to {output_csv_path}")
    print(f"- Images copied to {dest_img_dir}")

# Define paths
source_img_dir = 'flickr30k-images'
dest_img_dir = 'img'
csv_path = 'image_references.csv'
output_dir = 'data'

# Run the extraction
create_dataset(source_img_dir, dest_img_dir, csv_path, output_dir)