import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from dataset import df, features

# Train baseline model on months 1-3
train_df = df[df['cohort_month'] <= 3]
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
model.fit(train_df[features], train_df['churned'])

# Evaluate PR-AUC on each subsequent month (4-12)
test_months = list(range(4, 13))
monthly_pr_auc = []
for month in test_months:
    month_df = df[df['cohort_month'] == month]
    y_prob = model.predict_proba(month_df[features])[:, 1]
    monthly_pr_auc.append(average_precision_score(month_df['churned'], y_prob))

results_df = pd.DataFrame({'Month': test_months, 'Monthly PR-AUC': monthly_pr_auc})
results_df['3-month moving average'] = results_df['Monthly PR-AUC'].rolling(window=3, min_periods=1).mean()

plt.figure(figsize=(8, 5))
plt.plot(results_df['Month'], results_df['Monthly PR-AUC'], marker='o', label='Monthly PR-AUC')
plt.plot(results_df['Month'], results_df['3-month moving average'], marker='o', linestyle='--', label='3-month moving average')
plt.title('Figure 1.1: Model Performance Decline Across Monthly Cohorts (Concept Drift)')
plt.xlabel('Month (test period cohorts)')
plt.ylabel('PR-AUC')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('figure1_1.png', dpi=150, bbox_inches='tight')
plt.show()
