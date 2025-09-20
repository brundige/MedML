import os
import shutil
import pandas as pd

# Paths
metadata_path = os.path.join('C:/Users/chrisb/Desktop/EMTS/MedML/dataset', 'HAM10000_metadata.csv')
images_dir = os.path.join('C:/Users/chrisb/Desktop/EMTS/MedML/dataset', 'images')

# Read metadata
metadata = pd.read_csv(metadata_path)

# Create subfolders for each lesion type (dx)
lesion_types = metadata['dx'].unique()
for lesion in lesion_types:
    lesion_folder = os.path.join(images_dir, lesion)
    os.makedirs(lesion_folder, exist_ok=True)

# Move images to their respective lesion subfolders
for idx, row in metadata.iterrows():
    image_id = row['image_id']
    lesion = row['dx']
    src = os.path.join(images_dir, f'{image_id}.jpg')
    dst = os.path.join(images_dir, lesion, f'{image_id}.jpg')
    if os.path.exists(src):
        shutil.move(src, dst)
    else:
        print(f'Warning: {src} does not exist.')

print('Image organization complete.')

