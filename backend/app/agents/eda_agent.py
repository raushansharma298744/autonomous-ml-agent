# EDA Agent - Generates exploratory data analysis
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from app.tools.eda_tools import generate_eda_report, set_style
from app.tools.dataset_tools import load_dataset

logger = logging.getLogger(__name__)


def analyze_target_distribution(target_series: pd.Series) -> Dict[str, Any]:
    """Analyze target variable distribution."""
    analysis = {
        "dtype": str(target_series.dtype),
        "missing": int(target_series.isnull().sum()),
        "unique_values": int(target_series.nunique()),
    }
    
    if target_series.dtype in ["object", "category"] or target_series.nunique() <= 20:
        # Classification
        analysis["type"] = "classification"
        vc = target_series.value_counts()
        analysis["class_distribution"] = vc.to_dict()
        analysis["class_balance"] = target_series.value_counts(normalize=True).to_dict()
        analysis["is_imbalanced"] = vc.min() / vc.max() < 0.1
    else:
        # Regression
        analysis["type"] = "regression"
        analysis["target_stats"] = {
            "mean": float(target_series.mean()),
            "std": float(target_series.std()),
            "min": float(target_series.min()),
            "max": float(target_series.max()),
            "median": float(target_series.median()),
            "skewness": float(target_series.skew()),
            "kurtosis": float(target_series.kurtosis()),
        }
    
    return analysis


def analyze_correlations(df: pd.DataFrame, target_column: str, top_k: int = 10) -> Dict[str, Any]:
    """Analyze feature correlations with target."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    
    if len(numerical_cols) < 1:
        return {}
    
    correlations = {}
    
    # Safely convert target for correlation if it's categorical/string
    target_series = df[target_column]
    if not pd.api.types.is_numeric_dtype(target_series):
        try:
            target_series = pd.Series(pd.factorize(target_series)[0])
        except Exception:
            return {} # Skip correlation if we can't factorize

    for col in numerical_cols:
        try:
            corr = df[col].corr(target_series)
            if not np.isnan(corr):
                correlations[col] = float(corr)
        except Exception:
            continue
    
    # Sort by absolute correlation
    sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
    
    return {
        "top_positive": dict(list(sorted_corr.items())[:top_k]),
        "top_negative": dict(list(sorted_corr.items())[-top_k:]) if len(sorted_corr) > top_k else {},
        "all_correlations": sorted_corr,
    }


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze missing value patterns."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    return {
        "total_missing": int(missing.sum()),
        "columns_with_missing": missing[missing > 0].to_dict(),
        "missing_percentage": missing_pct[missing_pct > 0].to_dict(),
        "high_missing_cols": missing_pct[missing_pct > 50].index.tolist(),
    }


def analyze_outliers(df: pd.DataFrame, numerical_columns: List[str]) -> Dict[str, Any]:
    """Analyze outliers in numerical columns."""
    outliers = {}
    for col in numerical_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        outliers[col] = {
            "count": int(outlier_mask.sum()),
            "percentage": float(outlier_mask.mean() * 100),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
        }
    return outliers


def generate_eda_findings(
    df: pd.DataFrame,
    target_column: str,
    dataset_profile: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate structured EDA findings."""
    logger.info(f"[EDAAgent] Generating EDA for {len(df)} rows")
    
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)
    
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    findings = {
        "target_analysis": analyze_target_distribution(df[target_column]),
        "correlations": analyze_correlations(df, target_column),
        "missing_values": analyze_missing_values(df),
        "outliers": analyze_outliers(df, numerical_columns),
        "feature_types": {
            "numerical": numerical_columns,
            "categorical": categorical_columns,
        },
        "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
    }
    
    # Key insights
    insights = []
    
    # Target insights
    target_analysis = findings["target_analysis"]
    if target_analysis.get("type") == "classification":
        dist = target_analysis.get("class_distribution", {})
        if dist:
            total = sum(dist.values())
            for cls, count in dist.items():
                pct = count / total * 100
                insights.append(f"Target class '{cls}': {count} samples ({pct:.1f}%)")
        if target_analysis.get("is_imbalanced"):
            insights.append("⚠️ Class imbalance detected - minority class underrepresented")
    else:
        stats = target_analysis.get("target_stats", {})
        insights.append(f"Target range: {stats.get('min', 0):.2f} - {stats.get('max', 0):.2f}")
        if abs(stats.get("skewness", 0)) > 1:
            insights.append(f"Target is skewed (skewness={stats.get('skewness', 0):.2f}) - consider log transform")
    
    # Correlation insights
    corr = findings["correlations"]
    top_pos = corr.get("top_positive", {})
    if top_pos:
        top_feat = list(top_pos.keys())[0]
        top_val = list(top_pos.values())[0]
        insights.append(f"Strongest correlation with target: {top_feat} (r={top_val:.3f})")
    
    # Missing value insights
    missing = findings["missing_values"]
    if missing["total_missing"] > 0:
        insights.append(f"Missing values in {len(missing['columns_with_missing'])} columns "
                       f"(total: {missing['total_missing']})")
    
    # Outlier insights
    outliers = findings["outliers"]
    high_outlier_cols = [c for c, v in outliers.items() if v["percentage"] > 5]
    if high_outlier_cols:
        insights.append(f"High outlier percentage (>5%) in: {', '.join(high_outlier_cols[:3])}")
    
    findings["insights"] = insights
    findings["eda_summary"] = "; ".join(insights[:5])  # Top 5 insights
    
    logger.info(f"[EDAAgent] EDA complete: {len(insights)} insights generated")
    
    return findings


def create_eda_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for EDA."""
    dataset_path = state["dataset_path"]
    target_column = state["target_column"]
    dataset_profile = state.get("dataset_profile", {})
    
    # Load dataset
    df = load_dataset(dataset_path)
    
    # Generate EDA findings
    eda_results = generate_eda_findings(df, target_column, dataset_profile)
    
    # Generate plots (optional, for report)
    output_dir = f"./reports/eda_{state.get('job_id', 'temp')}"
    try:
        plots = generate_eda_report(df, target_column, output_dir, 
                                   dataset_profile.get("target_analysis", {}).get("problem_type", "classification"))
        eda_results["plots"] = plots
    except Exception as e:
        logger.warning(f"EDA plot generation failed: {e}")
        eda_results["plots"] = {}
    
    return {
        "eda_results": eda_results,
        "eda_summary": eda_results.get("eda_summary", ""),
        "current_step": "eda",
    }