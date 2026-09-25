# EDA Tools - Deterministic EDA visualization functions
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from pathlib import Path


def set_style():
    """Set consistent plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")


def save_plot(fig, output_dir: str, filename: str) -> str:
    """Save plot to file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(filepath)


def plot_target_distribution(y: pd.Series, output_dir: str, problem_type: str) -> str:
    """Plot target variable distribution."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if problem_type == "classification":
        counts = y.value_counts()
        colors = sns.color_palette("husl", len(counts))
        bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor='white', linewidth=1.5)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right')
        ax.set_ylabel("Count")
        ax.set_title("Target Class Distribution")
        
        # Add percentage labels
        for i, (bar, count) in enumerate(zip(bars, counts.values)):
            pct = count / len(y) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(y)*0.01,
                   f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    else:
        ax.hist(y.dropna(), bins=30, edgecolor='white', linewidth=0.5, alpha=0.8)
        ax.set_xlabel("Target Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Target Distribution")
        
        # Add statistics
        mean_val = y.mean()
        median_val = y.median()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='blue', linestyle='--', label=f'Median: {median_val:.2f}')
        ax.legend()
    
    plt.tight_layout()
    return save_plot(fig, output_dir, "target_distribution.png")


def plot_numerical_distributions(df: pd.DataFrame, numerical_columns: List[str], output_dir: str) -> List[str]:
    """Plot distributions of numerical features."""
    if not numerical_columns:
        return []
    
    set_style()
    n_cols = min(3, len(numerical_columns))
    n_rows = (len(numerical_columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for i, col in enumerate(numerical_columns):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=30, edgecolor='white', linewidth=0.5, alpha=0.8)
        ax.set_title(f"{col} (skew={data.skew():.2f})")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
    
    # Hide unused subplots
    for i in range(len(numerical_columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return [save_plot(fig, output_dir, "numerical_distributions.png")]


def plot_correlation_matrix(df: pd.DataFrame, numerical_columns: List[str], output_dir: str) -> str:
    """Plot correlation heatmap for numerical features."""
    if len(numerical_columns) < 2:
        return ""
    
    set_style()
    corr = df[numerical_columns].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f',
        cmap='RdBu_r', center=0, square=True,
        cbar_kws={"shrink": 0.8}, ax=ax
    )
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    return save_plot(fig, output_dir, "correlation_matrix.png")


def plot_categorical_distributions(df: pd.DataFrame, categorical_columns: List[str], output_dir: str, max_categories: int = 20) -> List[str]:
    """Plot distributions of categorical features."""
    if not categorical_columns:
        return []
    
    set_style()
    plots = []
    
    for col in categorical_columns:
        unique_count = df[col].nunique()
        if unique_count > max_categories:
            continue  # Skip high cardinality
        
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = df[col].value_counts().head(max_categories)
        colors = sns.color_palette("husl", len(counts))
        bars = ax.barh(range(len(counts)), counts.values, color=colors, edgecolor='white', linewidth=1)
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(counts.index)
        ax.set_xlabel("Count")
        ax.set_title(f"{col} Distribution")
        ax.invert_yaxis()
        
        plt.tight_layout()
        plots.append(save_plot(fig, output_dir, f"categorical_{col}.png"))
    
    return plots


def plot_feature_target_relationship(
    df: pd.DataFrame,
    feature: str,
    target: str,
    output_dir: str,
    problem_type: str
) -> str:
    """Plot relationship between feature and target."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if problem_type == "classification":
        # Box plot for numerical features
        if df[feature].dtype in [np.number]:
            df.boxplot(column=feature, by=target, ax=ax)
            ax.set_title(f"{feature} by {target}")
            ax.set_xlabel(target)
        else:
            # Stacked bar for categorical
            crosstab = pd.crosstab(df[feature], df[target], normalize='index')
            crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
            ax.set_title(f"{target} distribution by {feature}")
            ax.legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Scatter plot for regression
        ax.scatter(df[feature], df[target], alpha=0.5, s=20)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.set_title(f"{feature} vs {target}")
        
        # Add trend line
        from scipy import stats
        clean = df[[feature, target]].dropna()
        if len(clean) > 1:
            slope, intercept, r_value, _, _ = stats.linregress(clean[feature], clean[target])
            x_line = np.array([clean[feature].min(), clean[feature].max()])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r--', label=f'R²={r_value**2:.3f}')
            ax.legend()
    
    plt.tight_layout()
    return save_plot(fig, output_dir, f"feature_target_{feature}.png")


def plot_outliers(df: pd.DataFrame, numerical_columns: List[str], output_dir: str) -> List[str]:
    """Plot boxplots for outlier visualization."""
    if not numerical_columns:
        return []
    
    set_style()
    n_cols = min(3, len(numerical_columns))
    n_rows = (len(numerical_columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for i, col in enumerate(numerical_columns):
        ax = axes[i]
        data = df[col].dropna()
        ax.boxplot(data, vert=True, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                  medianprops=dict(color='red', linewidth=2))
        ax.set_title(f"{col} (Outliers)")
        ax.set_ylabel(col)
    
    for i in range(len(numerical_columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return [save_plot(fig, output_dir, "outliers.png")]


def generate_eda_report(df: pd.DataFrame, target_column: str, output_dir: str, problem_type: str) -> Dict[str, Any]:
    """Generate complete EDA with all plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)
    
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    plots = {}
    
    # Target distribution
    plots["target_distribution"] = plot_target_distribution(df[target_column], output_dir, problem_type)
    
    # Numerical distributions
    plots["numerical_distributions"] = plot_numerical_distributions(df, numerical_columns, output_dir)
    
    # Correlation matrix
    plots["correlation_matrix"] = plot_correlation_matrix(df, numerical_columns, output_dir)
    
    # Categorical distributions
    plots["categorical_distributions"] = plot_categorical_distributions(df, categorical_columns, output_dir)
    
    # Feature-target relationships (top 5 numerical)
    for feat in numerical_columns[:5]:
        plots[f"feature_target_{feat}"] = plot_feature_target_relationship(
            df, feat, target_column, output_dir, problem_type
        )
    
    # Outliers
    plots["outliers"] = plot_outliers(df, numerical_columns, output_dir)
    
    return plots