import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


import random
RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def lasso_feature_importance(X, y, alpha=0.1, test_size=0.2, random_state=RANDOM_SEED):
    feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)
    if isinstance(X_scaled, np.ndarray):
        X_scaled_dense = X_scaled
    else:
        X_scaled_dense = X_scaled.toarray()  
    X_scaled_df = pd.DataFrame(X_scaled_dense, columns=feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=test_size, random_state=random_state
    )
    
    lasso = Lasso(alpha=alpha, random_state=random_state, max_iter=10000)
    lasso.fit(X_train, y_train)
    
    y_pred_train = lasso.predict(X_train)
    y_pred_test = lasso.predict(X_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "coefficient": lasso.coef_,  
        "importance": np.abs(lasso.coef_)
    })
    
    feature_importance_sorted = feature_importance.sort_values(
        by="importance", ascending=False
    ).reset_index(drop=True)
    
    feature_importance_sorted["is_informative"] = feature_importance_sorted["coefficient"] != 0
    
    return feature_importance_sorted, lasso, scaler

def plot_feature_importance(feature_importance_sorted, top_n=None):
    if top_n is not None:
        plot_data = feature_importance_sorted.head(top_n)
    else:
        plot_data = feature_importance_sorted
    
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.figure(figsize=(12, 8))
    
    colors = ["#2E86AB" if x else "#A23B72" for x in plot_data["is_informative"]]
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
        "Lasso Regression Feature Importance Ranking",
        fontsize=14,
        fontweight="bold",
        pad=20
    )
    plt.xlabel("Importance (Absolute Value of Lasso Coefficient)", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2E86AB", label="Informative Feature (Non-zero Coefficient)"),
        Patch(facecolor="#A23B72", label="Redundant Feature (Zero Coefficient)")
    ]
    plt.legend(handles=legend_elements, loc="lower right")
    
    plt.tight_layout()
    plt.savefig("lasso_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    return plot_data['feature'].tolist()

def run_Lasso(X,y):
    feature_importance, lasso_model, scaler = lasso_feature_importance(
        X, y, alpha=0.05, test_size=0.2
    )
    
    print(feature_importance[["feature", "coefficient", "importance", "is_informative"]])
    
    top_features = plot_feature_importance(feature_importance, top_n=10)
    top_pos = [int(name.split('_')[1])-1 for name in top_features]
    return top_pos
    