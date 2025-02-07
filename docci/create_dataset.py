import json
import os
import shutil
import pandas as pd
from pathlib import Path

def create_dataset(jsonl_path, image_folder, n_samples=500):
    # Create directories if they don't exist
    data_dir = Path('data')
    img_dir = Path('img')
    data_dir.mkdir(exist_ok=True)
    img_dir.mkdir(exist_ok=True)

    # Read jsonl file
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Create DataFrame
    df = pd.DataFrame(data)

    # Initialize list to store valid entries
    valid_entries = []

    # Process each entry
    for _, row in df.iterrows():
        image_path = Path(image_folder) / row['image_file']

        # Check if image exists
        if image_path.exists():
            valid_entries.append({
                'image_file': row['image_file'],
                'description': row['description']
            })

            # Copy image to img directory
            shutil.copy2(image_path, img_dir / row['image_file'])

        # Break if we have enough samples
        if len(valid_entries) >= n_samples:
            break

    # Create final DataFrame and save to CSV
    final_df = pd.DataFrame(valid_entries)
    if len(final_df) < n_samples:
        print(f"Warning: Only found {len(final_df)} valid images out of requested {n_samples}")

    csv_path = data_dir / 'data500.csv'
    final_df.to_csv(csv_path, index=False)
    print(f"Created dataset with {len(final_df)} entries")
    print(f"CSV file saved at: {csv_path}")
    print(f"Images copied to: {img_dir}")

if __name__ == "__main__":
    jsonl_path = "docci_descriptions.jsonlines"
    image_folder = r"docci_images/docci_images/images"

    create_dataset(jsonl_path, image_folder)