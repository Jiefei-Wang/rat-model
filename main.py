from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from read_data import read_data
from data_management import chunk_data
from feature_extraction import convert_to_features
from model import logistic_model, random_forest_model, gradient_boosting_model, cross_validate_auc

df = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

chunk_size = 1
df2 = chunk_data(df, chunk_size) 
X,y = convert_to_features(df2, chunk_size)



## Simple and naive train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
predictions_log = logistic_model(y_train, X_train, X_test)
roc_auc_log = roc_auc_score(y_test.tolist(), predictions_log)

predictions_tree = random_forest_model(y_train, X_train, X_test)
roc_auc_tree = roc_auc_score(y_test.tolist(), predictions_tree)

predictions_gb = gradient_boosting_model(y_train, X_train, X_test)
roc_auc_gb = roc_auc_score(y_test.tolist(), predictions_gb)



## K-fold cross validation
n_splits = 10
average_auc_log = cross_validate_auc(logistic_model, X, y, n_splits)
average_auc_tree = cross_validate_auc(random_forest_model, X, y, n_splits)
average_auc_gb = cross_validate_auc(gradient_boosting_model, X, y, n_splits)

