import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

output_base = "output/data"
os.makedirs("output/features", exist_ok=True)

df_ML_train = pickle.load(open(os.path.join(output_base, "df_ML_train.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(output_base, "feature_names.pkl"), "rb"))

X = df_ML_train.loc[:, feature_names].values
y = df_ML_train.loc[:, 'category'].values

#Train the GB model with the best hyperparameters found from previous sweeps
model = GradientBoostingClassifier(
    n_estimators=100,   
    max_depth=3,
    subsample=0.8,
    min_samples_split=2,
    min_samples_leaf=3,
    random_state=42
)
model.fit(X, y)

#Feature Importances
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)


importance_df["Rank"] = range(1, len(importance_df) + 1)
print(importance_df)
importance_df.to_csv("output/features/feature_importance_ranked.csv", index=False)

#Bar Plot of Feature Importances
plt.figure(figsize=(8, 6))
plt.barh(importance_df["Feature"] [::-1], importance_df["Importance"][::-1], color="cornflowerblue")
plt.title("Feature Importances Ranked (Trained on Gradient Boosting Model)")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("output/features/feature_importances.png", dpi=300)
plt.show()

#Bar Plot of Top 10 Feature Importances
top_k=10
top_features = importance_df.head(top_k)
plt.figure(figsize=(8, 6))
plt.barh(top_features["Feature"][::-1], top_features["Importance"][::-1], color="blue")
plt.title(f"Top 10 Feature Importances (Gradient Boosting)")
plt.xlabel("Feature Importance") 
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("output/features/top_10_feature_importances.png", dpi=300)
plt.show()

