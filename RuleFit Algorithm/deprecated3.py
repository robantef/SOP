from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from rulefit import RuleFit
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import X_df, X, y, feature_names

active_rule_counts = []
accuracies = []

# Train RuleFit with increasing max_rules, measure the number of active rules
# needed to reach acceptable accuracy via StratifiedKFold
max_rule_settings = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]

print("Simulating Rule Explosion Pareto Frontier...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for max_r in max_rule_settings:
    fold_accs = []
    fold_rule_counts = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        rf = RuleFit(max_rules=max_r, tree_size=4, max_iter=1000, random_state=42)
        rf.fit(X_train, y_train, feature_names=feature_names)

        preds = rf.predict(X_test)
        preds_class = (preds > 0.5).astype(int)
        acc = balanced_accuracy_score(y_test, preds_class)

        # Count all active terms (linear features + rules)
        active_count = int(np.sum(rf.coef_ != 0))
        fold_accs.append(acc)
        fold_rule_counts.append(active_count)

    mean_acc = np.mean(fold_accs)
    mean_rules = np.mean(fold_rule_counts)
    active_rule_counts.append(mean_rules)
    accuracies.append(mean_acc)
    print(f'  max_rules={max_r:4d} -> mean_active_terms={mean_rules:.1f}, mean_balanced_acc={mean_acc:.4f}')

# --- Figure 1.3: Pareto Frontier Plot ---
plt.figure(figsize=(10, 6))
plt.plot(active_rule_counts, accuracies, marker='o', linestyle='-', color='purple', linewidth=2, markersize=8)

# Annotate points with max_rules setting
for i, max_r in enumerate(max_rule_settings):
    plt.annotate(f'{max_r}', (active_rule_counts[i], accuracies[i]),
                 textcoords="offset points", xytext=(5, 5), fontsize=7, color='gray')

# Mark the cognitive limit
plt.axvline(x=15, color='red', linestyle='--', linewidth=2, label='Human Cognitive Limit (~15 rules)')

plt.title('Figure 1.3: The Accuracy-Interpretability Trade-off & Rule Explosion')
plt.xlabel('Mean Number of Active Terms Retained (Model Complexity)')
plt.ylabel('Mean Cross-Validated Balanced Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure_1_3.png', dpi=150)
plt.close()
print('Saved figure_1_3.png')