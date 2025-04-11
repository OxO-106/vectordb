import pandas as pd
import os

# Define the path to the CSV file
file_path = os.path.join('data', 'precision.csv')

# Read the CSV file
df = pd.read_csv(file_path)

# Calculate min and max for each precision metric
min_precision_at_5 = df['precision_at_5'].min()
max_precision_at_5 = df['precision_at_5'].max()

min_precision_at_10 = df['precision_at_10'].min()
max_precision_at_10 = df['precision_at_10'].max()

min_precision_at_15 = df['precision_at_15'].min()
max_precision_at_15 = df['precision_at_15'].max()

# Calculate midpoints
midpoint_precision_at_5 = (min_precision_at_5 + max_precision_at_5) / 2
midpoint_precision_at_10 = (min_precision_at_10 + max_precision_at_10) / 2
midpoint_precision_at_15 = (min_precision_at_15 + max_precision_at_15) / 2

# Find the dimensions closest to each midpoint
closest_dim_precision_at_5 = df.iloc[(df['precision_at_5'] - midpoint_precision_at_5).abs().argsort()[0]]['dimension']
closest_dim_precision_at_10 = df.iloc[(df['precision_at_10'] - midpoint_precision_at_10).abs().argsort()[0]]['dimension']
closest_dim_precision_at_15 = df.iloc[(df['precision_at_15'] - midpoint_precision_at_15).abs().argsort()[0]]['dimension']

# Print the results
print("Precision at 5:")
print(f"Min: {min_precision_at_5}, Max: {max_precision_at_5}")
print(f"Midpoint: {midpoint_precision_at_5}")
print(f"Closest dimension to midpoint: {closest_dim_precision_at_5}")
print()

print("Precision at 10:")
print(f"Min: {min_precision_at_10}, Max: {max_precision_at_10}")
print(f"Midpoint: {midpoint_precision_at_10}")
print(f"Closest dimension to midpoint: {closest_dim_precision_at_10}")
print()

print("Precision at 15:")
print(f"Min: {min_precision_at_15}, Max: {max_precision_at_15}")
print(f"Midpoint: {midpoint_precision_at_15}")
print(f"Closest dimension to midpoint: {closest_dim_precision_at_15}")