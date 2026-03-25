import kaggle
import pandas as pd
import zipfile
import os

# Download the Netflix dataset from Kaggle
# Dataset: https://www.kaggle.com/shivamb/netflix-shows
dataset_name = 'shivamb/netflix-shows'
download_path = './netflix-shows'

# Download the dataset
kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)

# Load the dataset into a pandas DataFrame
csv_file = os.path.join(download_path, 'netflix_titles.csv')
df = pd.read_csv(csv_file)

# Display the first few rows
print(df.head())

