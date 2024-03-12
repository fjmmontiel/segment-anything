import json
import glob
import os
import pandas as pd

# Path to the directory containing json files
directory_path = 'test_flies_bk'

# Function to check if an object is likely a fly based on its area
def is_fly(ann, min_area=0, max_area=300):
    return min_area < ann['area'] < max_area

# Initialize a list to store flies count for each json
flies_data = []

# List all json files in the directory
json_files = glob.glob(os.path.join(directory_path, '*.json'))

# Process each file
for json_file in json_files:
    with open(json_file, 'r') as f:
        annotations = json.load(f)

    # Extract flies from annotations
    flies = [ann for ann in annotations if is_fly(ann)]
    flies_data.append({
        'file_name': os.path.basename(json_file),
        'flies_count': len(flies)
    })

# Create a DataFrame
df_flies = pd.DataFrame(flies_data)

# Calculate statistics
min_flies = df_flies['flies_count'].min()
max_flies = df_flies['flies_count'].max()
median_flies = df_flies['flies_count'].median()
avg_flies = df_flies['flies_count'].mean()

# Find the file with minimum and maximum number of flies
file_min_flies = df_flies[df_flies['flies_count'] == min_flies]['file_name'].values[0]
file_max_flies = df_flies[df_flies['flies_count'] == max_flies]['file_name'].values[0]

# Display the table
print(df_flies)

# Display the statistics
print(f"\nFile with minimum flies ({min_flies}): {file_min_flies}")
print(f"File with maximum flies ({max_flies}): {file_max_flies}")
print(f"Median flies count across all files: {median_flies}")
print(f"Average flies count across all files: {avg_flies}")
print(f"Total number of flies across all files: {df_flies['flies_count'].sum()}")
