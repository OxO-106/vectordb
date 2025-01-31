from datasets import load_dataset
import sys
import pandas as pd
import os
import requests

# Set console encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Load dataset in streaming mode
ds = load_dataset("wikimedia/wit_base", split="train", streaming=True)

# Create the "img" folder if it doesn't exist
os.makedirs("img", exist_ok=True)

# Ensure the data folder exists (though you mentioned it already exists)
data_folder = "data"
if not os.path.exists(data_folder):
    print(f"Warning: '{data_folder}' folder not found. Creating it...")
    os.makedirs(data_folder)

# Set a custom User-Agent header
user_agent = "MyImageDownloader/1.0"

# Initialize an empty list to store the cleaned data
cleaned_data = []

# Iterate over the dataset and collect the desired data points
count = 0
for example in ds:
    try:
        # Check if all required columns are present and have non-empty values
        if all(example.get(col) for col in ['image', 'image_url', 'embedding', 'caption_attribution_description']):
            # Extract the required columns
            image_url = example['image_url']
            caption = example['caption_attribution_description']

            # Skip if the image is a GIF
            if image_url.endswith('.gif'):
                print(f"Skipping GIF image: {image_url}")
                continue

            # Generate a unique filename for the image
            image_filename = f"image_{count}.jpg"
            image_path = os.path.join("img", image_filename)

            try:
                # Download the image using requests with the custom User-Agent header
                response = requests.get(image_url, headers={"User-Agent": user_agent}, timeout=10)
                response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
                with open(image_path, "wb") as file:
                    file.write(response.content)
            except (requests.exceptions.RequestException, IOError) as e:
                print(f"Error downloading image: {str(e)}")
                continue

            # Append the cleaned data point to the list
            cleaned_data.append({
                'image_filename': image_filename,
                'image_url': image_url,
                'caption': caption
            })

            count += 1
            if count >= 256:
                break
    except Exception as e:
        # Skip entries that encounter any errors
        print(f"Skipped entry due to error: {str(e)}")

# Create a DataFrame from the cleaned data
data256 = pd.DataFrame(cleaned_data)

# Save the DataFrame as a CSV file in the data folder
csv_path = os.path.join(data_folder, 'data256.csv')
data256.to_csv(csv_path, index=False)
print(f"Dataset saved to {csv_path}")