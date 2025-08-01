#!/usr/bin/env python3
"""
Simple script to combine text-only and multimodal layer selection results.
Just averages the scores and creates a final ranking.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def combine_results():
    """Combine the two CSV files and create final ranking."""
    
    # Load both result files
    text_results = pd.read_csv("results/principled_layer_selection_results.csv")
    multimodal_results = pd.read_csv("results/principled_layer_selection_results_multimodal.csv")
    
    # Merge on Layer column
    merged = pd.merge(text_results, multimodal_results, on='Layer', suffixes=('_text', '_multimodal'))
    
    # Score columns to average
    score_columns = [
        'Overall_Score', 'Distributional_Score', 'Geometric_Score', 'Information_Score',
        'MMD', 'Wasserstein', 'KL_Divergence', 'SVM_Margin', 
        'Silhouette', 'Distance_Ratio', 'Mutual_Info', 'Entropy_Reduction'
    ]
    
    # Create combined results
    combined = pd.DataFrame()
    combined['Layer'] = merged['Layer']
    
    # Average scores across both analyses
    for col in score_columns:
        text_col = f"{col}_text"
        multimodal_col = f"{col}_multimodal"
        combined[col] = (merged[text_col] + merged[multimodal_col]) / 2
    
    # Sort by Overall_Score and assign final ranks
    combined = combined.sort_values('Overall_Score', ascending=False).reset_index(drop=True)
    combined['Rank'] = range(1, len(combined) + 1)
    
    # Reorder columns
    column_order = [
        'Rank', 'Layer', 'Overall_Score', 
        'Distributional_Score', 'Geometric_Score', 'Information_Score',
        'MMD', 'Wasserstein', 'KL_Divergence', 
        'SVM_Margin', 'Silhouette', 'Distance_Ratio', 
        'Mutual_Info', 'Entropy_Reduction'
    ]
    
    combined = combined[column_order]
    
    # Save combined results
    combined.to_csv("results/principled_layer_selection_results_combined.csv", index=False, float_format='%.4f')
    print(f"Combined results saved to: results/principled_layer_selection_results_combined.csv")

    # Create combined statistics
    create_combined_stats(text_results, multimodal_results, combined)

    # Create simple visualization
    create_simple_visualization(combined)

    # Print top 10
    print("\nTop 10 Layers:")
    print(combined.head(10)[['Rank', 'Layer', 'Overall_Score']].to_string(index=False))

    return combined

def create_combined_stats(text_results, multimodal_results, combined):
    """Create combined statistics CSV."""

    # Calculate statistics for each analysis
    stats_data = []

    # Text-only stats
    stats_data.append({
        'Analysis': 'Text-Only',
        'Mean_Overall_Score': text_results['Overall_Score'].mean(),
        'Std_Overall_Score': text_results['Overall_Score'].std(),
        'Max_Overall_Score': text_results['Overall_Score'].max(),
        'Min_Overall_Score': text_results['Overall_Score'].min(),
        'Best_Layer': text_results.loc[text_results['Overall_Score'].idxmax(), 'Layer'],
        'Worst_Layer': text_results.loc[text_results['Overall_Score'].idxmin(), 'Layer']
    })

    # Multimodal stats
    stats_data.append({
        'Analysis': 'Multimodal',
        'Mean_Overall_Score': multimodal_results['Overall_Score'].mean(),
        'Std_Overall_Score': multimodal_results['Overall_Score'].std(),
        'Max_Overall_Score': multimodal_results['Overall_Score'].max(),
        'Min_Overall_Score': multimodal_results['Overall_Score'].min(),
        'Best_Layer': multimodal_results.loc[multimodal_results['Overall_Score'].idxmax(), 'Layer'],
        'Worst_Layer': multimodal_results.loc[multimodal_results['Overall_Score'].idxmin(), 'Layer']
    })

    # Combined stats
    stats_data.append({
        'Analysis': 'Combined',
        'Mean_Overall_Score': combined['Overall_Score'].mean(),
        'Std_Overall_Score': combined['Overall_Score'].std(),
        'Max_Overall_Score': combined['Overall_Score'].max(),
        'Min_Overall_Score': combined['Overall_Score'].min(),
        'Best_Layer': combined.loc[combined['Overall_Score'].idxmax(), 'Layer'],
        'Worst_Layer': combined.loc[combined['Overall_Score'].idxmin(), 'Layer']
    })

    # Create DataFrame and save
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv("results/combined_layer_selection_stats.csv", index=False, float_format='%.4f')
    print(f"Combined statistics saved to: results/combined_layer_selection_stats.csv")

def create_simple_visualization(combined):
    """Create simple visualization of combined results."""

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Top 15 layers overall scores
    top_15 = combined.head(15)
    bars = ax1.bar(range(len(top_15)), top_15['Overall_Score'],
                   color='steelblue', alpha=0.7)
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Combined Overall Score')
    ax1.set_title('Top 15 Layers - Combined Analysis')
    ax1.set_xticks(range(len(top_15)))
    ax1.set_xticklabels([f'L{int(layer)}' for layer in top_15['Layer']], rotation=45)
    ax1.grid(True, alpha=0.3)

    # Highlight top 3
    for i in range(min(3, len(top_15))):
        bars[i].set_color('red')
        bars[i].set_alpha(0.8)

    # Plot 2: Category scores for top 10 layers
    top_10 = combined.head(10)
    x = np.arange(len(top_10))
    width = 0.25

    ax2.bar(x - width, top_10['Distributional_Score'], width,
            label='Distributional', alpha=0.8, color='lightcoral')
    ax2.bar(x, top_10['Geometric_Score'], width,
            label='Geometric', alpha=0.8, color='lightgreen')
    ax2.bar(x + width, top_10['Information_Score'], width,
            label='Information', alpha=0.8, color='lightblue')

    ax2.set_xlabel('Top 10 Layers')
    ax2.set_ylabel('Category Score')
    ax2.set_title('Category Performance - Top 10 Layers')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'L{int(layer)}' for layer in top_10['Layer']], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Distribution of overall scores across all layers in order
    # Sort by layer number for ordered display
    combined_sorted = combined.sort_values('Layer').reset_index(drop=True)

    ax3.plot(combined_sorted['Layer'], combined_sorted['Overall_Score'],
             'o-', linewidth=2, markersize=6, color='darkgreen', alpha=0.7)
    ax3.set_xlabel('Layer Number')
    ax3.set_ylabel('Combined Overall Score')
    ax3.set_title('Score Distribution Across All Layers')
    ax3.grid(True, alpha=0.3)

    # Highlight the best performing layer
    best_idx = combined_sorted['Overall_Score'].idxmax()
    best_layer = combined_sorted.loc[best_idx, 'Layer']
    best_score = combined_sorted.loc[best_idx, 'Overall_Score']
    ax3.scatter(best_layer, best_score, color='red', s=100, zorder=5)
    ax3.annotate(f'Best: L{int(best_layer)}\n({best_score:.3f})',
                xy=(best_layer, best_score), xytext=(10, 10),
                textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig("results/combined_layer_selection_visualization.png", dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: results/combined_layer_selection_visualization.png")
    plt.close()

if __name__ == "__main__":
    combine_results()
