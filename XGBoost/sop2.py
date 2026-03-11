import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from dataset import df, features

# --- Temporal Split (Correct) ---
train_t = df[df['cohort_month'] <= 6]
test_t = df[df['cohort_month'] > 6]
model_t = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
model_t.fit(train_t[features], train_t['churned'])
pr_auc_temporal = average_precision_score(
    test_t['churned'], model_t.predict_proba(test_t[features])[:, 1]
)

# --- Random Split (Incorrect — causes temporal leakage) ---
X_all, y_all = df[features], df['churned']
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.5, random_state=42)
model_r = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
model_r.fit(X_tr, y_tr)
pr_auc_random = average_precision_score(y_te, model_r.predict_proba(X_te)[:, 1])

# --- Plot ---
labels = ['Temporal Split\n(Correct)', 'Random Split\n(Causes Leakage)']
scores = [pr_auc_temporal, pr_auc_random]
colors = ['tab:blue', 'tab:red']

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, scores, color=colors, width=0.4)
for bar, val in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_ylabel('PR-AUC Score')
ax.set_title('Figure 1.2: Temporal Leakage Inflates Model Performance')
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figure1_2.png', dpi=150, bbox_inches='tight')
plt.show()
