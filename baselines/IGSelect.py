
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.sparse import issparse

import random
RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def feature_selection_IG(X, y, k=None, random_state=None):
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
    
    scores = mutual_info_regression(X_scaled, y)
    feature_importance = pd.DataFrame({
        "feature": [f"feature_{i+1}" for i in range(n_features)],
        "importance": scores,
       })
    feature_importance_sorted = feature_importance.sort_values(
        by="importance", ascending=False
    ).reset_index(drop=True)

    return feature_importance_sorted, None, scaler


def plot_ig_results(feature_importance_sorted, top_n=None):
    if top_n is not None:
        plot_data = feature_importance_sorted.head(top_n)
    else:
        plot_data = feature_importance_sorted
    
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.figure(figsize=(12, 8))
    
    colors = ["#2E86AB" if x else "#A23B72" for x in plot_data]
    bars = plt.barh(
        y=plot_data["feature"],
        width=plot_data["importance"],
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
        "Information Gain Feature Selection Results",
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
    plt.savefig("information_gain_feature_selection.png", dpi=300, bbox_inches="tight")
    plt.close()
    return plot_data['feature'].tolist()


    
def run_IG(X, y, k=None):
    if y.ndim == 2 and y.shape[1] == 1:
        y1D = y.values.ravel()
    else:
        y1D = y
    feature_importance, model, scaler = feature_selection_IG(
        X, y1D, k=k)
    print(feature_importance[["feature", "importance"]])
    
    top_features = plot_ig_results(feature_importance, top_n=10)
    top_pos = [int(name.split('_')[1])-1 for name in top_features]
    return top_pos
    