import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. DATA LOADING & PREPARATION
# ==========================================
print("Loading Yale-Brain-Mets-Longitudinal dataset...")
try:
    # Attempt to load your exact Excel file
    df = pd.read_excel('Yale-Brain-Mets-Longitudinal_ClinicalData_20250605.xlsx')
    # Assuming 'target' is the clinical diagnosis: 0 = Recurrence, 1 = Radiation Necrosis
    print("Columns found:", list(df.columns))
    X = df.filter(regex='radiomic_|original_|diagnostics_').values
    y = df['target'].values
except FileNotFoundError:
    print("Excel file not found. Simulating the Yale longitudinal radiomic distribution...")
    np.random.seed(42)
    X = np.random.rand(1000, 200)
    y = np.random.choice([0, 1], size=1000, p=[0.90, 0.10])
except KeyError as e:
    print(f"Column {e} not found in Excel file. Available columns printed above. Falling back to simulation...")
    print("Hint: Replace 'target' in the try block with the correct column name from your dataset.")
    np.random.seed(42)
    X = np.random.rand(1000, 200)
    y = np.random.choice([0, 1], size=1000, p=[0.90, 0.10])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ==========================================
# FIGURE 1.3: HISTOGRAM DISCRETIZATION & QUANTIZATION ERROR
# ==========================================
print("Simulating Problem 3: Histogram Binning Precision Loss...")

# Take a single continuous radiomic feature (e.g., GLCM Contrast)
continuous_feature = X_train[:, 0]
# Sort by feature value and sample evenly across the full range
sort_idx = np.argsort(continuous_feature)
sorted_all = continuous_feature[sort_idx]
# Take 150 evenly spaced samples across the full sorted range
sample_idx = np.linspace(0, len(sorted_all) - 1, 150, dtype=int)
sorted_feature = sorted_all[sample_idx]

# LightGBM default binning reduces continuous arrays into discrete buckets (use 8 bins for clearer steps)
bins = np.linspace(sorted_feature.min(), sorted_feature.max(), 8)
digitized_sorted = bins[np.clip(np.digitize(sorted_feature, bins) - 1, 0, len(bins) - 1)]

fig, ax = plt.subplots(figsize=(9, 6))
# Plot original continuous data as smooth ascending line
ax.plot(range(len(sorted_feature)), sorted_feature, color='blue', linewidth=1.5, alpha=0.8, label='True Continuous Radiomic Values')
# Plot how LightGBM sees the data (quantized into rigid bins)
ax.step(range(len(sorted_feature)), digitized_sorted, where='mid', color='red', linewidth=2, label='LightGBM Discretized Bins')
# Shade the quantization error between the two
ax.fill_between(range(len(sorted_feature)), sorted_feature, digitized_sorted, alpha=0.2, color='orange', label='Quantization Error')

ax.set_xlabel('Patient Samples (sorted by feature value)', fontsize=12)
ax.set_ylabel('Radiomic Feature Intensity', fontsize=12)
ax.set_title('Figure 1.3: Loss of Threshold Precision via Histogram Discretization', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.annotate('Quantization Error:\nSubtle variations between patients\nare erased when forced into the same bin',
            xy=(75, digitized_sorted[75]), xytext=(90, digitized_sorted[75] - 0.15),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11, bbox=dict(fc="white", ec="black"))
plt.tight_layout()
plt.savefig('Figure_1_3_Quantization_Error.png', dpi=300)
plt.show()

print("Figure 1.3 saved.")
