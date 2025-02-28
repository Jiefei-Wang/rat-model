from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from modules.read_data import read_data
from modules.data_management import manage_data
from modules.feature_extraction import convert_to_features
from modules.model import logistic_model, random_forest_model, gradient_boosting_model, cross_validate_auc
from modules.visualization import make_bar_press_plot

df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')


## save to output as excel file
# output_path = 'output\\df_train.xlsx'
# df_train.to_xlsx(output_path, index=False)



## remove the first 3 bar presses of the data
truncate_size = 3
chunk_size = 1
max_press = 1000
standardize = False
df2 = manage_data(df_train, 
                  truncate_size=truncate_size,
                  chunk_size=chunk_size, 
                  max_press=max_press,
                  standardize=standardize)
X,y = convert_to_features(df2) ## TODO:slow



## Simple and naive train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

## For each ML model: 
## logistic regression, random forest, gradient boosting
## 1. Train the model on the training data
## 2. Fit the model on the test data
## 3. get AUROC score
predictions_log = logistic_model(y_train, X_train, X_test)
roc_auc_log = roc_auc_score(y_test.tolist(), predictions_log)

predictions_tree = random_forest_model(y_train, X_train, X_test)
roc_auc_tree = roc_auc_score(y_test.tolist(), predictions_tree)

predictions_gb = gradient_boosting_model(y_train, X_train, X_test)
roc_auc_gb = roc_auc_score(y_test.tolist(), predictions_gb)

roc_auc_log
roc_auc_tree
roc_auc_gb

## K-fold cross validation to get AUROC score
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

