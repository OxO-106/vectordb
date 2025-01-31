from bs4 import BeautifulSoup
import csv
import requests

def process_webpage(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    data = []

    # Find all image filenames (they're in <a> tags ending with .jpg)
    current_filename = None
    current_references = []

    # Iterate through all elements
    for element in soup.find_all(['a', 'li']):
        if element.name == 'a' and element.text.strip().endswith('.jpg'):
            # If we have a previous image's data, save it
            if current_filename and current_references:
                reference = ' '.join(current_references)
                data.append([current_filename, reference])

            # Start new image data
            current_filename = element.text.strip()
            current_references = []

        elif element.name == 'li' and current_filename:
            current_references.append(element.text.strip())

    # Don't forget to add the last image's data
    if current_filename and current_references:
        reference = ' '.join(current_references)
        data.append([current_filename, reference])

    return data

def save_to_csv(data, filename='image_references.csv'):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_filename', 'reference'])
        writer.writerows(data)

# Usage
url = "https://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.html"
response = requests.get(url)
data = process_webpage(response.text)
save_to_csv(data)