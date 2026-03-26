import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

output_base = "output/data"
os.makedirs("output/features", exist_ok=True)

# Load data
df_ML_train = pickle.load(open(os.path.join(output_base, "df_ML_train.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(output_base, "feature_names.pkl"), "rb"))

X = df_ML_train[feature_names].values
y = df_ML_train["category"].cat.codes.values   # 0 = FR1, 1 = EXT

# Logistic regression (L1, C=0.90)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        penalty="l1",
        C=0.90,
        solver="liblinear",
        max_iter=10000,
        random_state=42
    ))
])

model.fit(X, y)

# Extract coefficients
lr = model["lr"]
coef = lr.coef_.ravel()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coef,
    "Importance": np.abs(coef),
}).sort_values("Importance", ascending=False)

importance_df["Rank"] = range(1, len(importance_df) + 1)

print(importance_df)

importance_df.to_csv(
    "output/features/logistic_feature_importance_ranked.csv",
    index=False
)

plt.figure(figsize=(8, 6))
plt.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance"][::-1]
)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance (Trained on Logistic Regression Model)")
plt.tight_layout()
plt.savefig(
    "output/features/logistic_feature_importances_ranked.png",
    dpi=300
)
plt.show()

