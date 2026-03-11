import pandas as pd
import numpy as np

np.random.seed(42)
n_per_month = 1000

dfs = []
for month in range(1, 13):
    n = n_per_month
    age = np.random.randint(18, 70, n)
    inquiries = np.random.poisson(1, n)
    charges = np.random.uniform(9.99, 49.99, n)

    # Concept drift: early pattern (older + high charges -> churn) gradually reverses
    drift = (month - 1) / 11.0
    logit = (
        -2.2
        + (1 - 2 * drift) * 0.05 * (age - 44)
        + 0.5 * inquiries
        + (1 - 2 * drift) * 0.05 * (charges - 30)
    )
    prob = 1 / (1 + np.exp(-logit))
    churned = (np.random.rand(n) < prob).astype(int)

    dfs.append(pd.DataFrame({
        'age': age,
        'customer_service_inquiries': inquiries,
        'monthly_charges': charges,
        'churned': churned,
        'cohort_month': month
    }))

df = pd.concat(dfs, ignore_index=True)
df.insert(0, 'customer_id', range(1, len(df) + 1))

features = ['age', 'customer_service_inquiries', 'monthly_charges']
