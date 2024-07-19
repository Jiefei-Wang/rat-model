from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from main import df4

X = df4.drop(columns=['frustration', 'id'], axis = 1)
y = df4['frustration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def logistic_model(train_target, train_features, test_features): 
    model = LogisticRegression(max_iter=1000)

    model.fit(train_features, train_target)

    probabilities = model.predict_proba(test_features)[:,1]

    return probabilities

predictions = logistic_model(y_train, X_train, X_test)
roc_auc = roc_auc_score(y_test.tolist(), predictions)

