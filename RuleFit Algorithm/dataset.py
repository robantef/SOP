import pandas as pd
import numpy as np

# Load the dataset sheets
file_path = 'Brain-Mets-Lung-MRI-Path-Segs_Clinical_Data_2025_11_13.xlsx'
clinical_df = pd.read_excel(file_path, sheet_name='Clinical_Data')
radiomics_df = pd.read_excel(file_path, sheet_name='Radiomics')

# Normalize merge key casing
clinical_df = clinical_df.rename(columns={'Accession': 'accession'})

# Merge on accession
df = pd.merge(clinical_df, radiomics_df, on='accession', how='inner')

# Binary target: SCLC = 1, NSCLC variants = 0
df['target'] = (df['GPA Histology Class'] == 'SCLC').astype(int)
y = df['target'].values

# Define feature matrix (X) — radiomic features contain '_original_'
X_df = df.filter(regex='_original_')
X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)
X = X_df.values
feature_names = X_df.columns.tolist()