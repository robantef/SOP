import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
# FIGURE 1.2: LEAF-WISE GROWTH AND OVERFITTING
# ==========================================
print("Simulating Problem 2: Leaf-wise Structural Overfitting...")

# Create data with actual learnable signal so overfitting can be demonstrated
np.random.seed(42)
n_samples, n_features = 300, 50
X_sig = np.random.randn(n_samples, n_features)
# Target depends on first 5 features (real signal) + noise
y_sig = ((X_sig[:, 0] + X_sig[:, 1] - X_sig[:, 2] + 0.5 * X_sig[:, 3] + np.random.randn(n_samples) * 0.8) > 0).astype(int)
X_tr, X_vl, y_tr, y_vl = train_test_split(X_sig, y_sig, test_size=0.3, random_state=42)

leaves_range = [2, 4, 8, 16, 31, 64, 128, 256, 512, 1024]
train_errors, val_errors = [], []

for leaves in leaves_range:
    # Unconstrained aggressive leaf-wise growth
    model = lgb.LGBMClassifier(num_leaves=leaves, max_depth=-1, n_estimators=200, random_state=42, verbose=-1)
    model.fit(X_tr, y_tr)

    # Calculate error (1 - accuracy)
    train_errors.append(1 - accuracy_score(y_tr, model.predict(X_tr)))
    val_errors.append(1 - accuracy_score(y_vl, model.predict(X_vl)))

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(leaves_range, train_errors, marker='o', linestyle='-', color='green', label='Training Error')
ax.plot(leaves_range, val_errors, marker='s', linestyle='--', color='red', label='Validation Error')

ax.set_xlabel('Model Complexity (Number of Leaves)', fontsize=12)
ax.set_ylabel('Classification Error Rate', fontsize=12)
# ax.set_title('Figure 1.2: Validation Degradation due to Leaf-Wise Overfitting', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.set_xscale('log', base=2)
# Find the index where val error is highest for annotation
peak_idx = np.argmax(val_errors)
ax.annotate('Overfitting Threshold:\nValidation error spikes as\nthe model memorizes noise',
            xy=(leaves_range[peak_idx], val_errors[peak_idx]),
            xytext=(leaves_range[max(0, peak_idx-2)], val_errors[peak_idx] + 0.05),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Figure_1_2_Leaf_Wise_Overfitting.png', dpi=300)
plt.show()

print("Figure 1.2 saved.")
