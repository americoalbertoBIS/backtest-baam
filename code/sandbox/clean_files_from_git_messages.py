# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 00:26:44 2025

@author: al005366
"""

import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

root_dir = r"C:\git\backtest-baam\data\US\factors"

def clean_csv(file_path):
    try:
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
        mask = df.apply(lambda row: row.astype(str).str.contains("<<<<<<<|>>>>>>>|=======").any(), axis=1)
        cleaned_df = df[~mask]
        cleaned_df.to_csv(file_path, index=False)
        return f"Cleaned: {file_path}"
    except Exception as e:
        return f"Error cleaning {file_path}: {e}"

csv_files = []
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.lower().endswith(".csv") and "bootstrap_indices" not in filename:
            csv_files.append(os.path.join(dirpath, filename))

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(clean_csv, file_path) for file_path in csv_files]
    for result in tqdm(as_completed(futures), total=len(futures), desc="Cleaning CSV files"):
        print(result.result())
        
        
    