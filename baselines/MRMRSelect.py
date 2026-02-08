import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.sparse import issparse
import warnings

import random
RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def feautre_importance_mrmr(X, y, k=None, random_state=None):
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0].values
    elif y.ndim == 2:
        y = y.ravel()
    
    n_features = X.shape[1]
    k = min(k, n_features) if k is not None else min(10, n_features)
    
    if issparse(X):
        scaler = StandardScaler(with_mean=False)
    else:
        scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if issparse(X_scaled):
        X_scaled = X_scaled.toarray()

    n_features = X.shape[1]
    relevance = mutual_info_regression(X_scaled, y, discrete_features="auto", random_state=RANDOM_SEED)
    redundancy_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i+1, n_features):
            redundancy_matrix[i, j] = mutual_info_regression(
                X_scaled[:,[i]], X_scaled[:,j], discrete_features="auto", random_state=RANDOM_SEED
            )[0]
            redundancy_matrix[j, i] = redundancy_matrix[i, j]

    mrmr_scores = []
    for i in range(n_features):
        redundancy = redundancy_matrix[i, :].mean()
        mrmr_scores.append(relevance[i] - redundancy)

    return pd.DataFrame({
        "feature": [f"feature_{i+1}" for i in range(n_features)],
        "feature_importance": mrmr_scores
    }).sort_values(by="feature_importance", ascending=False).reset_index(drop=True)
    


def plot_mrmr_results(feature_importance_sorted, top_n=None):
    if top_n is not None:
        plot_data = feature_importance_sorted.head(top_n)
    else:
        plot_data = feature_importance_sorted
    
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.figure(figsize=(12, 8))
    
    colors = ["#2E86AB" if x else "#A23B72" for x in plot_data]
    bars = plt.barh(
        y=plot_data["feature"],
        width=plot_data["feature_importance"],
        color=colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5
    )
    
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{width:.4f}",
            va="center",
            ha="left",
            fontsize=9
        )
    
    plt.title(
        "MRMR Feature Selection Results",
        fontsize=14,
        fontweight="bold",
        pad=20
    )
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E86AB", label="Selected Feature"),
        Patch(facecolor="#A23B72", label="Not Selected")
    ]
    plt.legend(handles=legend_elements, loc="lower right")
    
    plt.tight_layout()
    plt.savefig("mrmr_feature_selection.png", dpi=300, bbox_inches="tight")
    plt.close()
    return plot_data['feature'].tolist()


def run_MRMR(X, y, k=None):
    if y.ndim == 2 and y.shape[1] == 1:
        y1D = y.values.ravel() 
    else:
        y1D = y
    feature_importance = feautre_importance_mrmr(
        X, y1D, k=k)
    
    print(feature_importance[["feature", "feature_importance"]])
    
    top_features = plot_mrmr_results(feature_importance, top_n=10)
    
    top_pos = [int(name.split('_')[1])-1 for name in top_features]
    return top_pos
    