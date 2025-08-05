import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_visualize_k_comparison():
    """Load and visualize k-value comparison results"""

    # Try to load the new layer-specific aggregated results first
    layer_csv_path = os.path.join(os.path.dirname(__file__), '../results/k_experiments/aggregated_k_layer_comparison.csv')
    old_csv_path = os.path.join(os.path.dirname(__file__), '../results/k_experiments/k_value_comparison.csv')

    figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    if os.path.exists(layer_csv_path):
        print("Found layer-specific aggregated results, creating enhanced visualizations...")
        create_layer_specific_visualizations(layer_csv_path, figures_dir)
    elif os.path.exists(old_csv_path):
        print("Found old format results, creating basic visualizations...")
        create_basic_visualizations(old_csv_path, figures_dir)
    else:
        print("No results files found. Please run the k-value experiments first.")
        return

def create_layer_specific_visualizations(csv_path, figures_dir):
    """Create visualizations for layer-specific aggregated results"""

    # Load layer-specific data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows of layer-specific data")
    print(f"Available columns: {list(df.columns)}")
    print(f"K values: {sorted(df['k_value'].unique())}")
    print(f"Layers: {sorted(df['layer'].unique())}")

    # 1. Layer 14 and 16 separate visualizations with all three metrics
    create_layer_metrics_visualization(df, figures_dir, layer=14)
    create_layer_metrics_visualization(df, figures_dir, layer=16)

    # 2. All layers heatmap (using AUROC)
    create_layers_heatmap(df, figures_dir)

def create_layer_metrics_visualization(df, figures_dir, layer):
    """Create visualization for a specific layer with Accuracy, AUROC, and F1"""

    # Filter for the specified layer
    layer_df = df[df['layer'] == layer].copy()

    if layer_df.empty:
        print(f"Warning: No data found for layer {layer}")
        return

    # Sort by k_value
    layer_df = layer_df.sort_values('k_value')

    # Create the plot
    _, ax = plt.subplots(figsize=(10, 6))

    k_values = layer_df['k_value']

    # Plot Accuracy
    if 'avg_accuracy_mean' in layer_df.columns:
        ax.errorbar(k_values, layer_df['avg_accuracy_mean'],
                   yerr=layer_df.get('avg_accuracy_std', 0),
                   marker='o', linewidth=2, markersize=8, capsize=5, capthick=2,
                   label='Accuracy', color='blue')

    # Plot AUROC
    if 'avg_auroc_mean' in layer_df.columns:
        ax.errorbar(k_values, layer_df['avg_auroc_mean'],
                   yerr=layer_df.get('avg_auroc_std', 0),
                   marker='s', linewidth=2, markersize=8, capsize=5, capthick=2,
                   label='AUROC', color='orange')

    # Plot F1
    if 'avg_f1_mean' in layer_df.columns:
        ax.errorbar(k_values, layer_df['avg_f1_mean'],
                   yerr=layer_df.get('avg_f1_std', 0),
                   marker='^', linewidth=2, markersize=8, capsize=5, capthick=2,
                   label='F1', color='green')

    ax.set_title(f'Layer {layer}: Performance Metrics vs K Value', fontsize=14, fontweight='bold')
    ax.set_xlabel('K Value', fontsize=12)
    ax.set_ylabel('Score (Mean ± Std)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.0)

    # Add some styling
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()

    # Save both PNG and PDF versions
    png_path = os.path.join(figures_dir, f'k_value_layer{layer}_metrics.png')
    pdf_path = os.path.join(figures_dir, f'k_value_layer{layer}_metrics.pdf')

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')

    print(f"Layer {layer} metrics figure saved to {png_path}")
    print(f"Layer {layer} metrics figure saved to {pdf_path}")
    plt.close()

def create_layers_heatmap(df, figures_dir):
    """Create heatmap showing AUROC performance across all layers and k values"""

    if 'avg_auroc_mean' not in df.columns:
        print("Warning: No avg_auroc_mean found for heatmap")
        return

    # Create pivot table for heatmap
    heatmap_data = df.pivot(index='layer', columns='k_value', values='avg_auroc_mean')

    # Create the heatmap
    _, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heatmap_data.values, cmap='RdYlBu_r', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AUROC (Mean)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            if not np.isnan(value):
                ax.text(j, i, f'{value:.3f}', ha="center", va="center",
                       color="white" if value < heatmap_data.values.mean() else "black",
                       fontsize=8)

    ax.set_title('AUROC Heatmap: All Layers vs K Values', fontweight='bold', pad=20)
    ax.set_xlabel('K Value')
    ax.set_ylabel('Layer')

    plt.tight_layout()
    out_path = os.path.join(figures_dir, 'k_value_layers_auroc_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Layers AUROC heatmap saved to {out_path}")
    plt.close()



def create_basic_visualizations(csv_path, figures_dir):
    """Create basic visualizations for old format results"""

    # Load data
    df = pd.read_csv(csv_path)

    # Group by k_value and compute mean and std for accuracy and f1
    agg = df.groupby('k_value').agg(
        accuracy_mean=('avg_accuracy', 'mean'),
        accuracy_std=('avg_accuracy', 'std'),
        f1_mean=('avg_f1', 'mean'),
        f1_std=('avg_f1', 'std'),
        count=('avg_accuracy', 'count')
    ).reset_index()

    # Plotting
    _, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(agg))

    ax.bar([i - bar_width/2 for i in index], agg['accuracy_mean'],
           yerr=agg['accuracy_std'], width=bar_width, label='Accuracy', capsize=5)
    ax.bar([i + bar_width/2 for i in index], agg['f1_mean'],
           yerr=agg['f1_std'], width=bar_width, label='F1 Score', capsize=5)

    ax.set_xticks(index)
    ax.set_xticklabels(agg['k_value'])
    ax.set_xlabel('k Value')
    ax.set_ylabel('Score')
    ax.set_title('Mean and Std of Accuracy and F1 Score by k Value')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.set_ylim(0.8, 1.0)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, 'k_value_comparison_accuracy_f1.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Basic comparison figure saved to {out_path}")
    plt.close()

if __name__ == "__main__":
    load_and_visualize_k_comparison()
