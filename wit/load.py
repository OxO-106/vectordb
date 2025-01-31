from datasets import load_dataset
import sys

# Set console encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Load dataset in streaming mode
ds = load_dataset("wikimedia/wit_base", split="train", streaming=True)

# View first few examples
count = 0
for example in ds:
    try:
        # Handle potential encoding issues safely
        caption = example['caption_attribution_description'] or "No caption available"
        url = example['image_url'] or "No URL available"

        print(f"Caption: {caption}")
        print(f"URL: {url}")
        print("---")

    except UnicodeEncodeError:
        # Skip entries that can't be encoded
        print("Skipped entry due to encoding issues")
        print("---")

    count += 1
    if count >= 5:
        break