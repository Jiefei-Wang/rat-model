
import matplotlib.pyplot as plt
import os

def append_prob(df, prob_log, prob_tree, prob_gb, prob_rnn):
    df['prob_log'] = prob_log
    df['prob_tree'] = prob_tree
    df['prob_gb'] = prob_gb
    df['prob_rnn'] = prob_rnn
    return df

def refactor_group(df, category_mapping_back):
    df['category'] = df['category'].map(category_mapping_back)
    ## replace NAN with "default"
    df['category'] = df['category'].fillna('default')
    rats = df['id'].unique()
    categories = df['category'].unique()
    file_mapping = {}
    for rat in rats:
        for cat in categories:
            rat_data = df[(df['id'] == rat) & (df['category'] == cat)]
            files = rat_data['file'].unique()
            file_mapping.update({file: i for i, file in enumerate(files)})

    df['file_index'] = df['file'].map(file_mapping)
    df['group'] = df.apply(lambda row: f"{row['category']}_{row['file_index']}", axis=1)
    return df

