import os
import pandas as pd

# Change this path to your local clone of the project
repo_path = '../tennis_MatchChartingProject'

for root, dirs, files in os.walk(repo_path):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            try:
                cols = pd.read_csv(file_path, nrows=0).columns.tolist()
                rel_path = os.path.relpath(file_path, repo_path)
                print(f'File: {rel_path}')
                print(f'Columns ({len(cols)}): {cols}\n')
            except Exception as e:
                print(f'Failed to read {file_path}:', e)
