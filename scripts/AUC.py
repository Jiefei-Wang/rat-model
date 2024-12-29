## This script is used to calculate the AUC for different models and chunk sizes.

from sklearn.model_selection import train_test_split
import pandas as pd

from read_data import read_data
from modules.data_management import chunk_data, truncate_data
from modules.feature_extraction import convert_to_features
from model import logistic_model, random_forest_model, gradient_boosting_model, cross_validate_auc


df_train = read_data('data/01 Sucrose FR1 vs EXT 8_2024')

truncate_size = 3
df2 = truncate_data(df_train, truncate_size)

n_splits = 10
auc_df = []
chunk_size_list = range(1, 21)

for chunk_size in chunk_size_list:
    print(f'Chunk size: {chunk_size}')
    df3 = chunk_data(df2, chunk_size) 
    X,y = convert_to_features(df3, chunk_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    average_auc_log = cross_validate_auc(logistic_model, X, y, n_splits)
    average_auc_tree = cross_validate_auc(random_forest_model, X, y, n_splits)
    average_auc_gb = cross_validate_auc(gradient_boosting_model, X, y, n_splits)
    
    auc_df.append({'model': 'logistic', 'chunk_size': chunk_size, 'roc_auc': average_auc_log})
    auc_df.append({'model': 'random_forest', 'chunk_size': chunk_size, 'roc_auc': average_auc_tree})
    auc_df.append({'model': 'gradient_boosting', 'chunk_size': chunk_size, 'roc_auc': average_auc_gb})

auc_df = pd.DataFrame(auc_df)


## line plot
import seaborn as sns
import matplotlib.pyplot as plt

## rename logistic to Logistic Regression
## rename random_forest to Random Forest
## rename gradient_boosting to Gradient Boosting
auc_df['model'] = auc_df['model'].replace({'logistic': 'Logistic Regression', 'random_forest': 'Random Forest', 'gradient_boosting': 'Gradient Boosting'})

sns.lineplot(data=auc_df, x='chunk_size', y='roc_auc', hue='model')
plt.xticks(ticks=range(1, 21))
## change y to ROAUC
plt.ylabel('ROC AUC')
plt.xlabel('Chunk Size')
plt.show()
## save figure
plt.savefig('output/ROAUC.png')