import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from dataset import X_df, X, y, feature_names

# --- Figure 1.1a: Correlation Heatmap ---
# Select a subset of features (e.g., first 20) for visual clarity
X_subset = X_df.iloc[:, :20]
plt.figure(figsize=(10, 8))
corr_matrix = X_subset.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Figure 1.1a: Severe Multicollinearity in Radiomic Features')
plt.tight_layout()
plt.savefig('figure_1_1a.png', dpi=150)
plt.close()
print('Saved figure_1_1a.png')

# --- Figure 1.1b: L1 Regularization Shrinkage ---
# Standardize features before applying Lasso
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)

# Apply strict L1 penalty (Lasso)
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X_scaled, y)

plt.figure(figsize=(12, 6))
plt.bar(X_subset.columns, lasso.coef_, color='darkred')
plt.xticks(rotation=45, ha='right')
plt.title('Figure 1.1b: L1 Penalty Arbitrarily Shrinking Correlated Features to Zero')
plt.ylabel('Lasso Coefficient Value')
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('figure_1_1b.png', dpi=150)
plt.close()
print('Saved figure_1_1b.png')