import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import seaborn as sns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="NPZ with tangent-space features X,y,subject")
    ap.add_argument("--report-dir", default="reports", help="Directory to save reports and visualizations")
    args = ap.parse_args()

    # Create report directory
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.features, allow_pickle=True)
    X, y = data['X'], data['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), SVC(kernel='rbf', C=1, probability=True))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Metrics
    balacc = balanced_accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Balanced Acc: {balacc:.3f}  AUROC: {auroc:.3f}")
    
    # Save metrics to JSON
    metrics = {
        "balanced_accuracy": float(balacc),
        "auroc": float(auroc),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "confusion_matrix": cm.tolist()
    }
    
    with open(report_dir / "svm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save classification report
    report = classification_report(y_test, y_pred, target_names=["Baseline", "Pain"], output_dict=True)
    with open(report_dir / "svm_classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Create visualizations
    plt.style.use('default')
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=['Baseline', 'Pain'], yticklabels=['Baseline', 'Pain'])
    axes[0].set_title(f'Confusion Matrix\nBalanced Acc: {balacc:.3f}')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, label=f'ROC (AUC = {auroc:.3f})', linewidth=2)
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Prediction Distribution
    axes[2].hist(y_proba[y_test == 0], bins=20, alpha=0.7, label='Baseline', color='blue')
    axes[2].hist(y_proba[y_test == 1], bins=20, alpha=0.7, label='Pain', color='red')
    axes[2].axvline(0.5, color='black', linestyle='--', label='Decision Threshold')
    axes[2].set_xlabel('Predicted Probability (Pain)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Prediction Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(report_dir / "svm_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[INFO] Results saved to {report_dir}/")
    print(f"  - svm_metrics.json")
    print(f"  - svm_classification_report.json")
    print(f"  - svm_results.png")


if __name__ == "__main__":
    main()
