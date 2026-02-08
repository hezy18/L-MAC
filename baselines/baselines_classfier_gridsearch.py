import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from config.config import DATA_PATH, FEATURES, DATA_ROOT, DATA_SETTING, DATA_DATE_VERSION, USE_IMAGE
from utils.metrics import evaluate_predictions
from utils.data_loader_writer import save_multiple_matrices_to_csv
import os
import json
import random
RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
from sklearn.model_selection import GridSearchCV


def run_classifier(X_train_top, X_test_top, X_dev_top, y_train_df, y_test_df, y_dev_df, 
                   model_names:str='all', label_type:str='select', attr_setting:str='all', save=1):
    supported_models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_SEED),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_SEED),
    }
    
    param_grids = {
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        },
        'DecisionTree': {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10]
        },
        'RandomForest': {
            'n_estimators': [100, 200, 500],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
    }

    if model_names=='all':
        model_names_list = list(supported_models.keys())
    elif ',' in model_names:
        model_names_list = [item.strip() for item in model_names.split(',')]
    else:
        model_names_list = [model_names]
    
    from config.config import PROMPT_SETTING
    
    results = {}
    cms={}
    for model_name in model_names_list:
        if model_name not in supported_models:
            continue
        
        base_model = supported_models[model_name]
        param_grid = param_grids.get(model_name, {})

        grid = GridSearchCV(
            base_model,
            param_grid,
            scoring="accuracy",  
            cv=3,
            n_jobs=-1
        )
        grid.fit(X_train_top, y_train_df.values.ravel())

        print(f"{model_name} best param: {grid.best_params_}")

        best_model = grid.best_estimator_

        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = best_model.predict_proba(X_test_top)
            y_pred_score = best_model.predict(X_test_top)

            if y_pred_proba.shape[1] == 2:
                y_pred_prob_auc = y_pred_proba[:, 1]
                
            
            elif y_pred_proba.shape[1] == 1:
                y_pred_prob_auc = y_pred_proba.ravel()

            else:
                raise ValueError(f"unknown shape: {y_pred_proba.shape}")
        else:
            y_pred_score = best_model.predict(X_test_top)
            y_pred_prob_auc = None

        metrics, cm = evaluate_predictions(y_test_df.values.ravel(), y_pred_score, y_pred_prob_auc, num_class=2)
        
        print(f"{model_name} test eval:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        
        metrics['best_params'] = grid.best_params_
        results[model_name] = metrics
        cms[model_name] = cm
    if save:
        output_path = DATA_PATH['baseline_metrics_path']
        with open( f'{output_path.split('.')[0]}_{label_type}_{attr_setting}_grid.{output_path.split('.')[1]}', 'w') as f:
            json.dump(results, f, indent=4)

        save_multiple_matrices_to_csv(list(cms.values()), list(cms.keys()), f'{output_path.split('.')[0]}_{label_type}_cm_{attr_setting}_grid.csv')
        
    else:
        return results
