from read_data import read_data,read_category_data
from data_management import chunk_data
from feature_extraction import convert_to_features
from model import build_logistic_model, build_random_forest_model, build_gradient_boosting_model
import torch
from RNN_model import Rat_base, concate_data, truncate_or_padding
from data_management import manage_data
from evaluation import append_prob, refactor_group, make_boxplot, make_single_boxplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')
df_test = read_category_data('data/application_Gen 2 Sucrose PR')

category_mapping = {"FR1": 1, "EXT": 0}
category_mapping_back = {1: "FR1", 0: "EXT"}


category_mapping = {"FR1": 1, "EXT": 0}
truncate_size = 1
chunk_size = 1
max_press = 80
standardize = True



df_train2 = manage_data(df_train, truncate_size, chunk_size, max_press, False)
df_train2["category"] = df_train2["category"].map(category_mapping)
df_ML_train = manage_data(df_train, truncate_size, chunk_size, max_press, standardize)
df_ML_train["category"] = df_ML_train["category"].map(category_mapping)
df_ML_train['data'] = df_ML_train['data'].apply(truncate_or_padding)
X_train,y_train = convert_to_features(df_train2, chunk_size)


df_test2 = manage_data(df_test, truncate_size, chunk_size, max_press, False)
df_ML_test = manage_data(df_test, truncate_size, chunk_size, max_press, standardize)
df_ML_test['data'] = df_ML_test['data'].apply(truncate_or_padding)
X_test,y_test = convert_to_features(df_test2, chunk_size)


logistic = build_logistic_model(y_train, X_train)
random_forest = build_random_forest_model(y_train, X_train)
gradient_boosting = build_gradient_boosting_model(y_train, X_train)


rnn = Rat_base.load('output/RNN/model_2024_10_01_11_28_02_9000.pth')
rnn = rnn.eval()


## calculate probability for each bar press in training
prob_log_train = logistic.predict_proba(X_train)[:,1]
prob_tree_train = random_forest.predict_proba(X_train)[:,1]
prob_gb_train = gradient_boosting.predict_proba(X_train)[:,1]
data = torch.tensor(df_ML_train['data'])
prob_rnn_train = rnn(data).tolist()


## calculate probability for each bar press in testing
prob_log_test = logistic.predict_proba(X_test)[:,1]
prob_tree_test = random_forest.predict_proba(X_test)[:,1]
prob_gb_test = gradient_boosting.predict_proba(X_test)[:,1]
data = torch.tensor(df_ML_test['data'])
prob_rnn_test = rnn(data).tolist()



df_train3 = df_train2.copy()
df_train3 = append_prob(df_train3, prob_log_train, prob_tree_train, prob_gb_train, prob_rnn_train)
df_train3 = refactor_group(df_train3, category_mapping_back)


df_test3 = df_test2.copy()
df_test3 = append_prob(df_test3, prob_log_test, prob_tree_test, prob_gb_test, prob_rnn_test)
df_test3 = refactor_group(df_test3, category_mapping_back)


########################## 
## Box plot visualization
##########################
prob_cols = ['prob_log', 'prob_tree', 'prob_gb', 'prob_rnn']
model_titles = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'RNN']
by_col = 'category'
## overall boxplot
make_single_boxplot(df_train3, prob_cols, model_titles, by_col)
# plt.show()
## save
plt.savefig('output/overall_boxplot_training.png')

output_folder = 'output/img_train'
## Generate boxplot for each rat
make_boxplot(output_folder, df_train3, prob_cols, model_titles, by_col)



## plot prob
def ave_rat_prob_by_time(df, col):
    ids = df['id'].unique()
    all_dt = []
    n_obs = []
    for rat in ids:
        rat_data = df[df['id'] == rat].reset_index(drop=True)
        all_dt.append(rat_data)
        n_obs.append(len(rat_data))
    min_obs = min(n_obs)
    all_dt = [dt[:min_obs] for dt in all_dt]
    cb = [dt[col].tolist() for dt in all_dt]
    cb_array = np.array(cb)
    average_values = np.mean(cb_array, axis=0)
    return average_values.tolist()
    

prob_logistic_cb = ave_rat_prob_by_time(df_test3, 'prob_log')
prob_tree_cb = ave_rat_prob_by_time(df_test3, 'prob_tree')
prob_gb_cb = ave_rat_prob_by_time(df_test3, 'prob_gb')
prob_rnn_cb = ave_rat_prob_by_time(df_test3, 'prob_rnn')


## plot
plt.plot(prob_logistic_cb)
plt.show()

plt.plot(prob_tree_cb)
plt.show()

plt.plot(prob_gb_cb)
plt.show()

plt.plot(prob_rnn_cb)
plt.show()




rat = df_test3[df_test3['id'] == 13]
plt.plot(rat['prob_rnn'])
plt.show()


## rolling average for prob
windows_size = 10
rolling_avg = rat['prob_rnn'].rolling(window=windows_size).mean()
plt.plot(rolling_avg)
plt.show()


## AUC for each model: logistic, random forest, gradient boosting, RNN
## Frustration level figure for 2 rats
## add box plot for 2 rats
## Rolling average for frustration level


