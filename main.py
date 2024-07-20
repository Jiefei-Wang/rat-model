from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from read_data import read_data
from data_management import chunk_data
from feature_extraction import calculate_duration, calculate_max_force, calculate_peaks, calculate_max_duration, add_percentile_columns
from model import logistic_model, random_forest_model, gradient_boosting_model, cross_validate_auc
df = read_data('data')
df2 = chunk_data(df, 5) 

## feature extraction
df3=df2.copy()
df3['duration'] = calculate_duration(df3['data'])
df3['max_force'] = calculate_max_force(df3['data'])
df3['num_of_peaks'] = calculate_peaks(df3['data'])
df3['max_duration'] = calculate_max_duration(df3['data'])
df3 = df3.drop('data', axis = 1)
add_percentile_columns(df3, 'duration', [25, 50, 75])
add_percentile_columns(df3, 'max_force', [25, 50, 75])
add_percentile_columns(df3, 'num_of_peaks', [25, 50, 75])
add_percentile_columns(df3, 'max_duration', [25, 50, 75])
df3['range_duration'] = abs(df3['duration_25'] - df3['duration_75'])
df3['range_force'] = abs(df3['max_force_25'] - df3['max_force_75'])
df3['range_peaks'] = abs(df3['num_of_peaks_25'] - df3['num_of_peaks_75'])
df3['range_max_duration'] = abs(df3['max_duration_25'] - df3['max_duration_75'])
df3 = df3.drop(columns = ['duration' , 'max_force', 'num_of_peaks', 'max_duration'], axis = 1)

##model 
df4 = df3.copy() 

X = df4.drop(columns=['frustration', 'id'], axis = 1)
y = df4['frustration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

predictions_log = logistic_model(y_train, X_train, X_test)
roc_auc_log = roc_auc_score(y_test.tolist(), predictions_log)

predictions_tree = random_forest_model(y_train, X_train, X_test)
roc_auc_tree = roc_auc_score(y_test.tolist(), predictions_tree)

predictions_gb = gradient_boosting_model(y_train, X_train, X_test)
roc_auc_gb = roc_auc_score(y_test.tolist(), predictions_gb)


average_auc_log, auc_list_log = cross_validate_auc(df4, 'frustration', logistic_model)
average_auc_tree, auc_list_tree = cross_validate_auc(df4, 'frustration', random_forest_model)
average_auc_gb, auc_list_gb = cross_validate_auc(df4, 'frustration', gradient_boosting_model)

