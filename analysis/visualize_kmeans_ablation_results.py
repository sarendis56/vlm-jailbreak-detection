import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to the CSV file
csv_path = os.path.join(os.path.dirname(__file__), '../results/kmeans_ablation_results.csv')

# Load data
with open(csv_path, 'r') as f:
    header = f.readline().strip().split(',')

df = pd.read_csv(csv_path, skiprows=1, names=header)
df['Layer'] = pd.to_numeric(df['Layer'], errors='coerce')

# Filter for COMBINED dataset and even layers 12-24
layers = list(range(12, 25, 2))
df = df[(df['Dataset'] == 'COMBINED') & (df['Layer'].isin(layers))]

for col in ['F1_Mean', 'F1_Std', 'Accuracy_Mean', 'Accuracy_Std', 'AUROC_Mean', 'AUROC_Std']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Main Figure (all layers, no legend) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
metrics = ['AUROC_Mean', 'Accuracy_Mean']
titles = ['AUROC vs Layer', 'Accuracy vs Layer']

for i, metric in enumerate(metrics):
    for approach in df['Approach'].unique():
        sub = df[df['Approach'] == approach]
        axes[i].plot(sub['Layer'], sub[metric], marker='o', label=approach)
    axes[i].set_title(titles[i])
    axes[i].set_xlabel('Layer')
    axes[i].set_ylabel(metric.replace('_', ' '))
    axes[i].set_xticks(layers)
    # axes[i].legend()  # Remove legend as requested
    axes[i].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig.suptitle('KMeans Ablation: AUROC and Accuracy by Layer and Approach', fontsize=16, y=1.05)
plt.subplots_adjust(top=0.85)

out_path = os.path.join(os.path.dirname(__file__), 'figures/kmeans_ablation_auroc_accuracy.png')
plt.savefig(out_path, bbox_inches='tight')
print(f"Figure saved to {out_path}")

# --- New Figure: Focus on Layer 16 with Error Bars ---
layer16 = df[df['Layer'] == 16].copy()

# Shorten approach labels for x-axis and prepare for sorting
short_labels = []
order_keys = []
for approach in layer16['Approach']:
    if approach.startswith('kmeans_k'):
        parts = approach.replace('kmeans_k', '').split('_')
        if len(parts) == 2:
            short_labels.append(f"{parts[0]}/{parts[1]}")
            # Use tuple for sorting: (1, int, int)
            order_keys.append((1, int(parts[0]), int(parts[1])))
        else:
            short_labels.append(approach)
            order_keys.append((2, 0, 0))
    elif approach == 'dataset_based':
        short_labels.append('dataset')
        order_keys.append((0, 0, 0))
    else:
        short_labels.append(approach)
        order_keys.append((2, 0, 0))

# Add short_labels and order_keys to DataFrame for sorting
layer16['short_label'] = short_labels
layer16['order_key'] = order_keys
layer16 = layer16.sort_values('order_key').reset_index(drop=True)

# Print the data used for visualization
print("\nData used for Layer 16 visualization:")
print(layer16[['short_label', 'F1_Mean', 'F1_Std', 'Accuracy_Mean', 'Accuracy_Std', 'AUROC_Mean', 'AUROC_Std']])

fig2, ax2 = plt.subplots(figsize=(12, 6))  # Further increased width for 3 bars
bar_width = 0.25
index = range(len(layer16))

# Set font sizes
title_fontsize = 20
label_fontsize = 12
tick_fontsize = 10
legend_fontsize = 12

ax2.bar([i - bar_width for i in index], layer16['F1_Mean'],
        yerr=layer16['F1_Std'], width=bar_width, label='F1', capsize=5, color='#1f77b4')
ax2.bar(index, layer16['Accuracy_Mean'],
        yerr=layer16['Accuracy_Std'], width=bar_width, label='Accuracy', capsize=5, color='#ff7f0e')
ax2.bar([i + bar_width for i in index], layer16['AUROC_Mean'],
        yerr=layer16['AUROC_Std'], width=bar_width, label='AUROC', capsize=5, color='#2ca02c')

ax2.set_xticks(index)
ax2.set_xticklabels(layer16['short_label'], rotation=30, ha='right', fontsize=tick_fontsize)
ax2.set_ylabel('Score', fontsize=label_fontsize)
ax2.set_title('Layer 16: F1, Accuracy, and AUROC by Clustering Strategy (with Std)', fontsize=title_fontsize)
ax2.legend(fontsize=legend_fontsize)
ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
ax2.set_ylim([0.8, 1.0])
ax2.tick_params(axis='y', labelsize=tick_fontsize)
ax2.tick_params(axis='x', labelsize=tick_fontsize)

plt.tight_layout()
layer16_out_path_png = os.path.join(os.path.dirname(__file__), 'figures/layer16_f1_accuracy_auroc.png')
layer16_out_path_pdf = os.path.join(os.path.dirname(__file__), 'figures/layer16_f1_accuracy_auroc.pdf')
plt.savefig(layer16_out_path_png, bbox_inches='tight')
plt.savefig(layer16_out_path_pdf, bbox_inches='tight')
print(f"Layer 16 figure saved to {layer16_out_path_png} and {layer16_out_path_pdf}")
