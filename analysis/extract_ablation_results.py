import pandas as pd

# Define file paths
base = "/home/server2/pchua/vlm-jailbreak-detection/ablation_results"
files = {
    "kcd": {
        "perf": f"{base}/ablation_kcd_20runs_20250804_144523/aggregated_kcd_ablation_performance.csv",
        "classif": f"{base}/ablation_kcd_20runs_20250804_144523/aggregated_kcd_ablation_classification.csv"
    },
    "mcd": {
        "perf": f"{base}/ablation_mcd_20runs_20250804_145127/aggregated_mcd_ablation_performance.csv",
        "classif": f"{base}/ablation_mcd_20runs_20250804_145127/aggregated_mcd_ablation_classification.csv"
    },
    "kcd_pca": {
        "perf": f"{base}/ablation_kcd_pca_20runs_20250804_160211/aggregated_kcd_pca_ablation_performance.csv",
        "classif": f"{base}/ablation_kcd_pca_20runs_20250804_160211/aggregated_kcd_pca_ablation_classification.csv"
    },
    "mcd_pca": {
        "perf": f"{base}/ablation_mcd_pca_20runs_20250804_162408/aggregated_mcd_pca_ablation_performance.csv",
        "classif": f"{base}/ablation_mcd_pca_20runs_20250804_162408/aggregated_mcd_pca_ablation_classification.csv"
    }
}

def extract_metrics(perf_csv, classif_csv, layer=16, pca_components=None):
    perf = pd.read_csv(perf_csv)
    classif = pd.read_csv(classif_csv)
    # For PCA, filter by PCA_Components as well
    if pca_components is not None:
        perf = perf[(perf['Layer'] == layer) & (perf['Dataset'] == 'COMBINED') & (perf['PCA_Components'] == pca_components)]
        classif = classif[(classif['Layer'] == layer) & (classif['Dataset'] == 'COMBINED') & (classif['PCA_Components'] == pca_components)]
    else:
        perf = perf[(perf['Layer'] == layer) & (perf['Dataset'] == 'COMBINED')]
        classif = classif[(classif['Layer'] == layer) & (classif['Dataset'] == 'COMBINED')]
    if perf.empty or classif.empty:
        return None
    return {
        "Accuracy_Mean": perf.iloc[0]['Accuracy_Mean'],
        "Accuracy_Std": perf.iloc[0]['Accuracy_Std'],
        "AUROC_Mean": perf.iloc[0]['AUROC_Mean'],
        "AUROC_Std": perf.iloc[0]['AUROC_Std'],
        "AUPRC_Mean": perf.iloc[0]['AUPRC_Mean'],
        "AUPRC_Std": perf.iloc[0]['AUPRC_Std'],
        "F1_Mean": classif.iloc[0]['F1_Mean'],
        "F1_Std": classif.iloc[0]['F1_Std'],
    }

# No-projection ablation
print("KCD (no projection):", extract_metrics(files['kcd']['perf'], files['kcd']['classif']))
print("MCD (no projection):", extract_metrics(files['mcd']['perf'], files['mcd']['classif']))

# PCA ablation (for each dimension)
for dim in [32, 64, 128, 256]:
    print(f"KCD PCA (dim={dim}):", extract_metrics(files['kcd_pca']['perf'], files['kcd_pca']['classif'], pca_components=dim))
    print(f"MCD PCA (dim={dim}):", extract_metrics(files['mcd_pca']['perf'], files['mcd_pca']['classif'], pca_components=dim))
