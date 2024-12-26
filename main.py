from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from read_data import read_data
from data_management import manage_data
from feature_extraction import convert_to_features
from model import logistic_model, random_forest_model, gradient_boosting_model, cross_validate_auc
from visualization import make_bar_press_plot

df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

##remove first 3 barpress data and save training data as .csv
df_train['data'] = df_train['data'].apply(lambda x: x[3:] if isinstance(x, list) else x)
## save the file to the output folder
output_path = 'output\\df_train.csv'
df_train.to_csv(output_path, index=False)


## remove the first 10 elements of the data
truncate_size = 1
chunk_size = 1
max_press = 80
standardize = True
df2 = manage_data(df_train, truncate_size, chunk_size, max_press, standardize)
X,y = convert_to_features(df2, chunk_size)




## Simple and naive train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
predictions_log = logistic_model(y_train, X_train, X_test)
roc_auc_log = roc_auc_score(y_test.tolist(), predictions_log)

predictions_tree = random_forest_model(y_train, X_train, X_test)
roc_auc_tree = roc_auc_score(y_test.tolist(), predictions_tree)

predictions_gb = gradient_boosting_model(y_train, X_train, X_test)
roc_auc_gb = roc_auc_score(y_test.tolist(), predictions_gb)

roc_auc_log
roc_auc_tree
roc_auc_gb

## K-fold cross validation
n_splits = 10
average_auc_log = cross_validate_auc(logistic_model, X, y, n_splits)
average_auc_tree = cross_validate_auc(random_forest_model, X, y, n_splits)
average_auc_gb = cross_validate_auc(gradient_boosting_model, X, y, n_splits)


average_auc_log
average_auc_tree
average_auc_gb





output_folder = 'output/barpress_raw'
make_bar_press_plot(output_folder, df_train)

## visualize barpress
output_folder = 'output/barpress_processed'
make_bar_press_plot(output_folder, df2)

