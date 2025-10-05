import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def load_features_and_labels(features_path, labels_path=None):
    """Load features and optionally labels."""
    features_df = pd.read_csv(features_path)
    
    if labels_path and os.path.exists(labels_path):
        labels_df = pd.read_csv(labels_path)
        merged = features_df.merge(labels_df, on="filename", how="inner")
        return merged, True
    else:
        return features_df, False

def plot_feature_distributions(df, has_labels=False, output_dir="outputs"):
    """Plot distributions of features by class."""
    if not has_labels:
        print("No labels available for distribution plots")
        return
    
    # Get feature columns (exclude filename and label)
    feature_cols = [col for col in df.columns if col not in ['filename', 'label']]
    
    # Create subplots for different frequency bands
    bands = ['theta', 'alpha', 'beta', 'gamma']
    q_values = [2, 3, 4]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, band in enumerate(bands):
        ax = axes[i]
        
        # Get features for this band
        band_features = [col for col in feature_cols if band in col]
        
        # Plot mean features for this band
        mean_features = [col for col in band_features if 'mean' in col]
        
        if mean_features:
            # Create a melted dataframe for seaborn
            plot_data = []
            for feature in mean_features:
                for _, row in df.iterrows():
                    plot_data.append({
                        'feature': feature,
                        'value': row[feature],
                        'label': row['label']
                    })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Create box plot
            sns.boxplot(data=plot_df, x='feature', y='value', hue='label', ax=ax)
            ax.set_title(f'{band.capitalize()} Band - Mean TsEn Features')
            ax.set_xlabel('Feature')
            ax.set_ylabel('TsEn Value')
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_correlation(df, output_dir="outputs"):
    """Plot correlation matrix of features."""
    feature_cols = [col for col in df.columns if col not in ['filename', 'label']]
    
    # Sample features if too many (for visualization)
    if len(feature_cols) > 50:
        # Select representative features
        sample_features = []
        for band in ['theta', 'alpha', 'beta', 'gamma']:
            for q in [2, 3, 4]:
                for stat in ['mean', 'var']:
                    band_features = [col for col in feature_cols if band in col and f'q{q}' in col and stat in col]
                    if band_features:
                        sample_features.append(band_features[0])  # Take first channel
        
        feature_cols = sample_features[:50]  # Limit to 50 features
    
    corr_matrix = df[feature_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_pca_analysis(df, has_labels=False, output_dir="outputs"):
    """Plot PCA analysis of features."""
    feature_cols = [col for col in df.columns if col not in ['filename', 'label']]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot explained variance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, min(21, len(pca.explained_variance_ratio_) + 1)), 
             pca.explained_variance_ratio_[:20], 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    
    # Plot cumulative explained variance
    plt.subplot(1, 2, 2)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, min(21, len(cumsum) + 1)), cumsum[:20], 'ro-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot first two principal components
    plt.figure(figsize=(10, 8))
    
    if has_labels:
        for label in df['label'].unique():
            mask = df['label'] == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.7)
        plt.legend()
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA: First Two Principal Components')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_2d.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_tsne_analysis(df, has_labels=False, output_dir="outputs"):
    """Plot t-SNE analysis of features."""
    feature_cols = [col for col in df.columns if col not in ['filename', 'label']]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)-1))
    X_tsne = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    
    if has_labels:
        for label in df['label'].unique():
            mask = df['label'] == label
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=label, alpha=0.7)
        plt.legend()
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization of Features')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_2d.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(df, has_labels=False, output_dir="outputs"):
    """Plot feature importance based on variance."""
    feature_cols = [col for col in df.columns if col not in ['filename', 'label']]
    
    # Calculate variance for each feature
    feature_vars = df[feature_cols].var().sort_values(ascending=False)
    
    # Plot top 20 most variable features
    top_features = feature_vars.head(20)
    
    plt.figure(figsize=(12, 8))
    top_features.plot(kind='bar')
    plt.title('Top 20 Most Variable Features')
    plt.xlabel('Feature')
    plt.ylabel('Variance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_summary_stats(df, has_labels=False, output_dir="outputs"):
    """Plot summary statistics."""
    feature_cols = [col for col in df.columns if col not in ['filename', 'label']]
    
    # Basic statistics
    stats = df[feature_cols].describe()
    
    # Plot mean vs std for all features
    plt.figure(figsize=(10, 6))
    plt.scatter(stats.loc['mean'], stats.loc['std'], alpha=0.7)
    plt.xlabel('Mean')
    plt.ylabel('Standard Deviation')
    plt.title('Feature Mean vs Standard Deviation')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_stats.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\nDataset Summary:")
    print(f"Number of files: {len(df)}")
    print(f"Number of features: {len(feature_cols)}")
    if has_labels:
        print(f"Classes: {df['label'].unique()}")
        print(f"Class distribution:")
        print(df['label'].value_counts())

def main():
    parser = argparse.ArgumentParser(description="Visualize EEG TsEn features")
    parser.add_argument("--features", type=str, default="outputs/features.csv", 
                       help="Path to features CSV file")
    parser.add_argument("--labels", type=str, default="labels.csv", 
                       help="Path to labels CSV file (optional)")
    parser.add_argument("--out", type=str, default="outputs", 
                       help="Output directory for plots")
    parser.add_argument("--all", action="store_true", 
                       help="Generate all visualizations")
    parser.add_argument("--distributions", action="store_true", 
                       help="Plot feature distributions")
    parser.add_argument("--correlation", action="store_true", 
                       help="Plot correlation matrix")
    parser.add_argument("--pca", action="store_true", 
                       help="Plot PCA analysis")
    parser.add_argument("--tsne", action="store_true", 
                       help="Plot t-SNE analysis")
    parser.add_argument("--importance", action="store_true", 
                       help="Plot feature importance")
    parser.add_argument("--summary", action="store_true", 
                       help="Plot summary statistics")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Load data
    df, has_labels = load_features_and_labels(args.features, args.labels)
    
    print(f"Loaded {len(df)} samples with {len([col for col in df.columns if col not in ['filename', 'label']])} features")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Generate visualizations
    if args.all or args.summary:
        plot_summary_stats(df, has_labels, args.out)
    
    if args.all or args.distributions:
        plot_feature_distributions(df, has_labels, args.out)
    
    if args.all or args.correlation:
        plot_feature_correlation(df, args.out)
    
    if args.all or args.pca:
        plot_pca_analysis(df, has_labels, args.out)
    
    if args.all or args.tsne:
        plot_tsne_analysis(df, has_labels, args.out)
    
    if args.all or args.importance:
        plot_feature_importance(df, has_labels, args.out)
    
    # If no specific plots requested, show summary
    if not any([args.all, args.distributions, args.correlation, args.pca, args.tsne, args.importance, args.summary]):
        plot_summary_stats(df, has_labels, args.out)
        plot_feature_distributions(df, has_labels, args.out)
        plot_pca_analysis(df, has_labels, args.out)

if __name__ == "__main__":
    main()
