from sklearn.metrics import roc_auc_score
from modules.Data import data_from_pickle_train_valid_test
from modules.model import logistic_model, random_forest_model, gradient_boosting_model
from modules.visualization import make_bar_press_plot

df_train, df_test, x_train, x_test, y_train, y_test = data_from_pickle_train_valid_test()



## For each ML model: 
## logistic regression, random forest, gradient boosting
## 1. Train the model on the training data
## 2. Fit the model on the test data
## 3. get AUROC score
predictions_log = logistic_model(y_train, x_train, x_test)
roc_auc_log = roc_auc_score(y_test.tolist(), predictions_log)

predictions_tree = random_forest_model(y_train, x_train, x_test)
roc_auc_tree = roc_auc_score(y_test.tolist(), predictions_tree)

predictions_gb = gradient_boosting_model(y_train, x_train, x_test)
roc_auc_gb = roc_auc_score(y_test.tolist(), predictions_gb)

roc_auc_log
roc_auc_tree
roc_auc_gb






output_folder = 'output/barpress_raw'
make_bar_press_plot(output_folder, df_train)

## visualize barpress
output_folder = 'output/barpress_processed'
make_bar_press_plot(output_folder, df2)

