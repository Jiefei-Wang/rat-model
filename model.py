import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from main import df4

X = df4.drop(columns=['frustration', 'id'], axis = 1)
y = df4['frustration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def logistic_model(train_target, train_features, test_features): 
    model = LogisticRegression(max_iter=1000)

    model.fit(train_features, train_target)

    probabilities = model.predict_proba(test_features)[:,1]

    return probabilities


def random_forest_model(train_target, train_features, test_features): 
    model = DecisionTreeClassifier() 

    model.fit(train_features, train_target)

    probabilities = model.predict_proba(test_features)[:,1]
    
    return probabilities 

def gradient_boosting_model(train_target, train_features, test_features): 
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=42)

    model.fit(train_features, train_target)

    probabilities = model.predict_proba(test_features)[:,1]

    return probabilities 

def cross_validate_auc(data, target_column, model_function, id_column='id', n_splits=10):
    X = data.drop(columns=[target_column, id_column], axis=1)
    y = data[target_column]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if len(np.unique(y_test)) < 2:
            print(f"Skipping fold with indices {test_index} due to only one class present in y_test")
            continue
        predictions = model_function(y_train, X_train, X_test)
        roc_auc = roc_auc_score(y_test, predictions)
        
        auc_list.append(roc_auc)
    
    average_auc = np.mean(auc_list)
    return average_auc, auc_list

average_auc_log, auc_list_log = cross_validate_auc(df4, 'frustration', logistic_model)
average_auc_tree, auc_list_tree = cross_validate_auc(df4, 'frustration', random_forest_model)
average_auc_gb, auc_list_gb = cross_validate_auc(df4, 'frustration', gradient_boosting_model)





