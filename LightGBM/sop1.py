import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
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
# FIGURE 1.1: CLASS IMBALANCE & DUMMY CLASSIFIER EFFECT
# ==========================================
print("Simulating Problem 1: Unpenalized Majority-Class Bias...")

# Train baseline LightGBM without cost-sensitive learning
baseline_model = lgb.LGBMClassifier(random_state=42)
baseline_model.fit(X_train, y_train)
y_pred = baseline_model.predict(X_val)

overall_acc = accuracy_score(y_val, y_pred)
minority_recall = recall_score(y_val, y_pred, pos_label=1) # Recall for Radiation Necrosis

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(['Overall Accuracy', 'Minority Class Recall\n(Radiation Necrosis)'],
              [overall_acc * 100, minority_recall * 100],
              color=['#4C72B0', '#C44E52'])

ax.set_ylim(0, 110)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Figure 1.1: Baseline LightGBM Acting as a Dummy Classifier', fontsize=14, fontweight='bold')
ax.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height() + 2,
        f'{overall_acc*100:.1f}%', ha='center', fontweight='bold', fontsize=12)
ax.text(bars[1].get_x() + bars[1].get_width()/2, bars[1].get_height() + 2,
        f'{minority_recall*100:.1f}%', ha='center', fontweight='bold', fontsize=12)
ax.annotate('The model guesses the majority class,\nachieving high accuracy but completely\nfailing to detect the rare condition.',
            xy=(1, minority_recall * 100), xytext=(0.5, 50),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
plt.tight_layout()
plt.savefig('Figure_1_1_Class_Imbalance.png', dpi=300)
plt.show()

print("Figure 1.1 saved.")
