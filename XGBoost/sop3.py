import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from dataset import df, features

# Temporal Train/Test Split (months 1-8 train, 9-12 test)
train_df = df[df['cohort_month'] <= 8]
test_df = df[df['cohort_month'] > 8]
X_train, y_train = train_df[features], train_df['churned']
X_test, y_test = test_df[features], test_df['churned']

# Train Baseline XGBoost (no imbalance handling — demonstrates the issue)
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predict and Plot Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Churned (0)', 'Churned (1)'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title('Figure 1.3: Confusion Matrix (Class Imbalance Impact)')
plt.savefig('figure1_3.png', dpi=150, bbox_inches='tight')
plt.show()