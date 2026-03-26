import os, pickle
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# Load saved data
output_base = "output/data"

with open(os.path.join(output_base, "df_ML_train.pkl"), "rb") as f:
    df_ML_train = pickle.load(f)

with open(os.path.join(output_base, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

# Prepare data
X = df_ML_train[feature_names].values
y = df_ML_train["label"].values   


# Logistic regression (L1, Best Parameter: C = 0.90)
model = Pipeline([("lr", LogisticRegression(penalty="l1",
                  C=0.90, solver="liblinear", 
                  max_iter=10000, random_state=42))])
model.fit(X, y)


# Extract coefficients
lr = model["lr"]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": lr.coef_.ravel(),
    "odds_ratio": np.exp(lr.coef_.ravel())
}).sort_values("coef", ascending=False)

print("Intercept:", float(lr.intercept_[0]))
print(coef_df.to_string(index=False))


# Save coefficients 
output_eval_dir = "output/evaluation"
os.makedirs(output_eval_dir, exist_ok=True)
coef_df.to_csv( os.path.join(output_eval_dir, "logistic_coef.csv"), index=False)
