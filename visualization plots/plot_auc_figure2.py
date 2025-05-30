import matplotlib.pyplot as plt
import seaborn as sns

# Data for the test AUC scores
test_models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'RNN', 'LSTM', 'GRU']
test_auc_scores = [0.579, 0.698, 0.668, 0.6442, 0.6602, 0.6828]

# Data for the train AUC scores
train_models = ['Logistic Regression', 'Gradient Boosting', 'Random Forest', 'RNN', 'LSTM', 'GRU']
train_auc_scores = [0.606, 0.671, 0.677, 0.679, 0.681, 0.7046]

# Set a nice color palette from Seaborn for better aesthetics
sns.set_palette("muted")  

# Create a grid of subplots (2 rows, 1 column)
fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # Larger figure for presentation

# Bar plot for Train AUC (now on top)
axs[0].bar(train_models, train_auc_scores, color=sns.color_palette("coolwarm", len(train_models)))
axs[0].set_ylabel('Train AUC', fontsize=14, fontweight='bold')
axs[0].set_title('Train AUC Scores for Different Models', fontsize=16, fontweight='bold')
axs[0].set_ylim(0.55, 0.75)  # Adjusting y-axis range to show more differences
axs[0].tick_params(axis='x', rotation=45, labelsize=12)
axs[0].tick_params(axis='y', labelsize=12)

# Adding values on top of the bars for Train AUC
for bar, score in zip(axs[0].patches, train_auc_scores):
    axs[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{score:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

# Bar plot for Test AUC (now at the bottom)
axs[1].bar(test_models, test_auc_scores, color=sns.color_palette("coolwarm", len(test_models)))
axs[1].set_xlabel('Models', fontsize=14, fontweight='bold')
axs[1].set_ylabel('Test AUC', fontsize=14, fontweight='bold')
axs[1].set_title('Test AUC Scores for Different Models', fontsize=16, fontweight='bold')
axs[1].set_ylim(0.55, 0.75)  # Adjusting y-axis range to show more differences
axs[1].tick_params(axis='x', rotation=45, labelsize=12)
axs[1].tick_params(axis='y', labelsize=12)

# Adding values on top of the bars for Test AUC
for bar, score in zip(axs[1].patches, test_auc_scores):
    axs[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{score:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
