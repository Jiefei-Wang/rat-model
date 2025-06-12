from sklearn.metrics import roc_auc_score
from modules.Data import data_from_pickle
from modules.model_tradition import logistic_model, random_forest_model, gradient_boosting_model
from modules.visualization import make_bar_press_plot

df_raw, df_ML, row_train, row_valid, row_test,feature_names = data_from_pickle()


x_train = df_ML.loc[row_train, feature_names].values
y_train = df_ML.loc[row_train, 'category'].values

x_valid = df_ML.loc[row_valid, feature_names].values
y_valid = df_ML.loc[row_valid, 'category'].values

## For each ML model: 
## logistic regression, random forest, gradient boosting
## 1. Train the model on the training data
## 2. Fit the model on the test data
## 3. get AUROC score
predictions_log = logistic_model(y_train, x_train, x_valid)
roc_auc_log = roc_auc_score(y_valid.tolist(), predictions_log)

predictions_tree = random_forest_model(y_train, x_train, x_valid)
roc_auc_tree = roc_auc_score(y_valid.tolist(), predictions_tree)

predictions_gb = gradient_boosting_model(y_train, x_train, x_valid)
roc_auc_gb = roc_auc_score(y_valid.tolist(), predictions_gb)

roc_auc_log
roc_auc_tree
roc_auc_gb






# output_folder = 'output/barpress_raw'
# make_bar_press_plot(output_folder, df_train)

# ## visualize barpress
# output_folder = 'output/barpress_processed'
# make_bar_press_plot(output_folder, df2)

