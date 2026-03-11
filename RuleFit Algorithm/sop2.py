from sklearn.model_selection import KFold
from rulefit import RuleFit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from dataset import X_df, X, y, feature_names


kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rule_sets = []

print("Simulating structural instability across 5 folds...")
for train_idx, test_idx in kf.split(X):
    X_train, y_train = X[train_idx], y[train_idx]
    
    # Train RuleFit
    rf = RuleFit(max_rules=50, max_iter=1000, random_state=42)
    rf.fit(X_train, y_train, feature_names=feature_names)
    
    # Extract active rules (coefficient != 0)
    n_features = X_train.shape[1]
    rule_coefs = rf.coef_[n_features:]
    active_rules = set()
    for rule, coef in zip(rf.rule_ensemble.rules, rule_coefs):
        if coef != 0:
            active_rules.add(str(rule))
    fold_rule_sets.append(active_rules)

# Calculate pairwise Dice-Sorensen Coefficient (DSC)
def calculate_dsc(set1, set2):
    if not set1 and not set2: return 0
    return 2 * len(set1.intersection(set2)) / (len(set1) + len(set2))

dsc_scores = []
for i in range(len(fold_rule_sets)):
    for j in range(i + 1, len(fold_rule_sets)):
        dsc_scores.append(calculate_dsc(fold_rule_sets[i], fold_rule_sets[j]))

# --- Figure 1.2: Structural Instability Plot ---
plt.figure(figsize=(6, 8))
sns.boxplot(y=dsc_scores, color='lightblue', width=0.4)
sns.stripplot(y=dsc_scores, color='navy', size=8, jitter=True)
plt.title('Figure 1.2: Structural Instability of Rule Sets (Pairwise DSC)')
plt.ylabel('Dice-Sorensen Coefficient (DSC)')
plt.ylim(-0.05, 1.05)
plt.axhline(1.0, color='green', linestyle='--', label='Perfect Stability (1.0)')
plt.legend()
plt.tight_layout()
plt.savefig('figure_1_2.png', dpi=150)
plt.close()
print('Saved figure_1_2.png')