import wandb
import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


api = wandb.Api()


def get_run_history(project_name):
    runs = api.runs(f"{api.default_entity}/{project_name}")
    results = {}
    for run in runs:
        run_name = run.name
        history = run.history()
        results[run_name] = history
    return results


project_names = ['rat-frustration-RNN', 'rat-frustration-LSTM', 'rat-frustration-GRU']
project_runs = {}
for project_name in project_names:
    project_runs[project_name] = get_run_history(project_name)



project_dt = project_runs['rat-frustration-RNN']
run = project_dt['1_64_4']
run_name = "1_64_4"
def get_model_params(run_name):
    parts = run_name.split('_')
    input_size = int(parts[0])
    hidden_size = int(parts[1])
    num_layers = int(parts[2])
    return {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers
    }


records = []
for project, runs in project_runs.items():
    for run_name, run in runs.items():
        param = get_model_params(run_name)
        best_idx = run['valid_auc'].idxmax()
        best_epoch = best_idx*10
        best_auc = run['valid_auc'][best_idx]
        records.append({
            'project_name': project,
            "best_epoch": best_epoch,
            "best_auc": best_auc
        } | param)

records_df = pd.DataFrame(records)

records_df.columns


RNN = records_df[records_df['project_name'] == 'rat-frustration-RNN']
LSTM = records_df[records_df['project_name'] == 'rat-frustration-LSTM']
GRU = records_df[records_df['project_name'] == 'rat-frustration-GRU']



##heatmap for hidden size and num layers with values as best_auc
def make_heatmap(df, title, x, y, z):
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    # Pivot the data to create a proper heatmap format
    heatmap_data = df.pivot_table(values=z, index=y, columns=x, aggfunc='mean')
    sns.heatmap(data=heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    # plt.show(block =False)
    ## save the figure
    plt.savefig(f"output/nn_model_training/{title}.png")

x = 'hidden_size'
y = 'num_layers'
z = 'best_auc'
make_heatmap(RNN, "RNN Best AUC by Hidden Size and Num Layers", x, y, z)
make_heatmap(LSTM, "LSTM Best AUC by Hidden Size and Num Layers", x, y, z)
make_heatmap(GRU, "GRU Best AUC by Hidden Size and Num Layers", x, y, z)


x = 'hidden_size'
y = 'num_layers'
z = "best_epoch"
make_heatmap(RNN, "RNN Best Epoch by Hidden Size and Num Layers", x, y, z)
make_heatmap(LSTM, "LSTM Best Epoch by Hidden Size and Num Layers", x, y, z)
make_heatmap(GRU, "GRU Best Epoch by Hidden Size and Num Layers", x, y, z)

