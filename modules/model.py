import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def build_logistic_model(train_target, train_features, config):
    key_list = ['penalty', 'C']
    model_config = {key: config[key] for key in key_list if key in config}
    model = LogisticRegression(max_iter=1000, **model_config)
    model.fit(train_features, train_target)
    return model

def build_random_forest_model(train_target, train_features, config):
    key_list = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
    model_config = {key: config[key] for key in key_list if key in config}
    model = RandomForestClassifier(random_state=42, **model_config) 
    model.fit(train_features, train_target)
    return model

def build_gradient_boosting_model(train_target, train_features, config):
    key_list = ['n_estimators', 'learning_rate', 'max_depth']
    model_config = {key: config[key] for key in key_list if key in config}
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=42, **model_config)
    model.fit(train_features, train_target)
    return model


def logistic_model(train_target, train_features, test_features, config={}): 
    model = build_logistic_model(train_target, train_features, config)
    probabilities = model.predict_proba(test_features)[:,1]
    return probabilities


def random_forest_model(train_target, train_features, test_features, config={}): 
    model = build_random_forest_model(train_target, train_features, config)
    probabilities = model.predict_proba(test_features)[:,1]
    
    return probabilities 

def gradient_boosting_model(train_target, train_features, test_features, config={}): 
    model= build_gradient_boosting_model(train_target, train_features, config)
    probabilities = model.predict_proba(test_features)[:,1]
    return probabilities 







def cross_validate_auc(model_function, X, y, n_splits=10):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred_all = []
    y_true_all = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        predictions = model_function(y_train, X_train, X_test)
        
        y_pred_all.extend(predictions)
        y_true_all.extend(y_test)
        
    
    roc_auc = roc_auc_score(y_true_all, y_pred_all)
    return roc_auc





