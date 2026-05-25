import pandas as pd
import pickle


output_base = 'output/data' 

df_ML = pickle.load(open(f'{output_base}/df_ML.pkl', "rb"))


df_ML.sex

# for the df, keep one unique id per row
df_ML2 = df_ML.drop_duplicates(subset=['id'])


df_ML.sex.unique()
df_ML2.sex.value_counts(dropna=False)
