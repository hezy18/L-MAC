import numpy as np
from sklearn.metrics import ndcg_score, roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, f1_score
from sklearn.preprocessing import label_binarize
from typing import Dict, Union, Callable, List, Optional

def reduce_classes_by_threshold(y_true, y_pred, threshold1, threshold2):
    if not (1 <= threshold1 <= 9 and threshold2 > threshold1 and threshold2 <= 10):
        raise ValueError(f"Invalid threshold settings: threshold1={threshold1}, threshold2={threshold2}")
    
    if not (np.all(y_true >= 1) and np.all(y_true <= 10)):
        raise ValueError("True labels contain values outside the range 1-10")
    if not (np.all(y_pred >= 1) and np.all(y_pred <= 10)):
        raise ValueError("Predicted labels contain values outside the range 1-10")
    
    def map_class(value):
        if value <= threshold1:
            return 1  
        elif value <= threshold2:
            return 2 
        else:
            return 3
    
    new_y_true = np.vectorize(map_class)(y_true)
    new_y_pred = np.vectorize(map_class)(y_pred)
    
    return new_y_true, new_y_pred

def calculate_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    if len(y_true) != len(y_pred):
        raise ValueError("True labels and predicted labels must have the same length")
    
    for i in range(len(y_true)):
        true_class = y_true[i]
        pred_class = y_pred[i]
        
        if true_class < 1 or true_class > num_classes:
            raise ValueError(f"True labels contains value outside the range: {true_class}")
        if pred_class < 1 or pred_class > num_classes:
            raise ValueError(f"Predicted labels contain values outside the range: {pred_class}")
        
        cm[true_class-1][pred_class-1] += 1
    print(cm)
    return cm


def weighted_error_rate(ground_truths, predictions, weight_func=None):
    if weight_func is None:
        weight_func = lambda x, y: abs(x - y)
    
    total_weighted_error = 0
    for gt, pred in zip(ground_truths, predictions):
        total_weighted_error += weight_func(gt, pred)
    
    return total_weighted_error / len(ground_truths)

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_pred_rounded = np.round(y_pred).astype(int)
    mse_rounded = mean_squared_error(y_true, y_pred_rounded)
    mae_rounded = mean_absolute_error(y_true, y_pred_rounded)
    
    # import pdb; pdb.set_trace()
    return {
        'MSE': round(mse_rounded, 4),
        'MAE': round(mae_rounded, 4),
    }

def calculate_regression_metrics_category(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    unique_categories = np.unique(y_true)
    
    metrics = {}
    
    for category in unique_categories:
        mask = (y_true == category)
        
        y_true_cat = y_true[mask]
        y_pred_cat = y_pred[mask]
        
        mse_raw = mean_squared_error(y_true_cat, y_pred_cat)
        mae_raw = mean_absolute_error(y_true_cat, y_pred_cat)
        
        y_pred_rounded = np.round(y_pred_cat).astype(int)
        mse_rounded = mean_squared_error(y_true_cat, y_pred_rounded)
        mae_rounded = mean_absolute_error(y_true_cat, y_pred_rounded)
        
        metrics[f'MSE_{category}'] = round(mse_rounded, 4)
        metrics[f'MAE_{category}'] = round(mae_raw, 4)
    
    return metrics

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    y_true_bin = label_binarize(y_true, classes=list(range(1, num_classes + 1)))
    y_pred_bin = label_binarize(y_pred, classes=list(range(1, num_classes + 1)))
    
    auc_scores = {}
    # for i in range(num_classes):
    #     try:
    #         auc_scores[f'AUC_class_{i+1}'] = round(roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i]), 4)
    #     except ValueError:
    #         auc_scores[f'AUC_class_{i+1}'] = float('nan')
    
    # import pdb; pdb.set_trace()
    try:
        new_true = np.where((y_true_bin[:, 2] == 1) | (y_true_bin[:, 3] == 1), 1, 0)
        
        new_score = y_pred_bin[:, 2] + y_pred_bin[:, 3]
        
        auc_scores['AUC_gt3'] = round(roc_auc_score(new_true, new_score), 4)
    
    except ValueError:
        auc_scores['AUC_gt3'] = float('nan')
    
    valid_auc = [score for score in auc_scores.values() if not np.isnan(score)]
    auc_macro = np.mean(valid_auc) if valid_auc else float('nan')
    
    metrics= {
        'ACC': round(acc, 4),
        'F1score': round(f1, 4),
        'AUCmacro': round(auc_macro, 4)
    }
    metrics.update(auc_scores)
    print(auc_scores)
    # import pdb; pdb.set_trace()
    return metrics

def calculate_recall_metrics(y_true: np.ndarray, y_pred_scores: np.ndarray, k_list: List[int] = None, threshold: int = 4) -> Dict[str, float]:
    if k_list is None:
        k_list = [10, 20, 50, 100]
    
    results = {}
    
    for k in k_list:
        sorted_indices = np.argsort(-y_pred_scores)[:k]
        top_k_labels = y_true[sorted_indices]
        
        relevant_total = np.sum(y_true >= threshold)
        
        relevant_hits = np.sum(top_k_labels >= threshold)
        
        recall = relevant_hits / relevant_total if relevant_total > 0 else 0
        results[f'Recall>={threshold}@{k}'] = round(recall, 4)
    
    return results

def calculate_ndcg_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_scores: np.ndarray, k_list: List[int] = None, is_trace: bool = False) -> Dict[str, float]:
    if k_list is None:
        k_list = [3, 5, 10]
    
    results = {}
    
    for k in k_list:
        ndcg_value = _calculate_ndcg(y_true, y_pred_scores, k, is_trace=is_trace)
        ndcg_value_raw = _calculate_ndcg(y_true, y_pred, k)
        # ndcg_value_binary = calculate_binary_ndcg(y_true, y_pred_scores, k, threshold=4)
        results[f'NDCG@{k}'] = round(ndcg_value, 4)
        results[f'NDCG@{k}_binary'] = round(ndcg_value_raw, 4)
        # results[f'NDCG@{k}_binary'] = round(ndcg_value_binary, 4)
        
        weighted_hit_rate = _calculate_weighted_hit_rate(y_true, y_pred_scores, k)
        results[f'Weighted_Hit_Rate@{k}'] = round(weighted_hit_rate, 4)
    print(results)
    # import pdb; pdb.set_trace()
    return results

def _calculate_ndcg(labels: np.ndarray, predictions: np.ndarray, k: int, is_trace: bool = False) -> float:
    if k is None:
        k = len(predictions)
    
    predictions = predictions - 1
    labels = labels - 1

    sorted_indices = np.argsort(predictions)[::-1]
    sorted_labels = labels[sorted_indices]
    dcg = 0.0
    for i in range(k):
        rel = sorted_labels[i] 
        numerator = 2 ** rel - 1 
        denominator = np.log2(i + 2) 
        dcg += numerator / denominator
    
    ideal_sorted_labels = np.sort(labels)[::-1]  
    idcg = 0.0
    for i in range(k):
        rel = ideal_sorted_labels[i]
        numerator = 2 ** rel - 1
        denominator = np.log2(i + 2)
        idcg += numerator / denominator
    
    return dcg / idcg if idcg != 0 else 0


def _calculate_weighted_hit_rate(labels: np.ndarray, predictions: np.ndarray, k: int = 5) -> float:
    sorted_indices = np.argsort(-predictions)[:k]
    top_k_labels = labels[sorted_indices] - 1  # 转换为0-2以便计算
    
    hit_score = np.sum(top_k_labels) / (2 * k)  # 最大可能分数为2*k
    return hit_score

def is_continuous_probability(prob_array):
    prob_array = np.asarray(prob_array)
    return not np.all((prob_array == 0) | (prob_array == 1))

def evaluate_classifier(y_true, y_pred, y_pred_proba=None):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
    cm = confusion_matrix(y_true, y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    auc = None
    
    if y_pred_proba is not None:
        if len(np.unique(y_true)) == 2:
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                y_pred_proba = y_pred_proba[:, 1]
            
            if not is_continuous_probability(y_pred_proba):
                print("Warning: Probability values contain only 0 and 1, AUC cannot be computed.")
                auc = 0.0
            else:
                try:
                    auc = roc_auc_score(y_true, y_pred_proba)
                except ValueError as e:
                    print(f"Computing AUC rror: {e}")
                    auc = 0.0
        else:
            auc = 0.0
    else:
        auc = 0.0
    
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    metrics = {
        'ACC': round(accuracy,4),
        'AUC_gt3': round(precision,4),
        'Recall>=4@10': round(recall,4),
        'F1score': round(f1,4),
        'AUCmacro': round(auc,4),  
    }
    print(cm)
    print(class_report)

    return metrics, cm


def evaluate_predictions(y_true, y_pred, y_pred_prob_auc=None, num_class=10):
    if num_class==2:
        metrics, cm = evaluate_classifier(y_true, y_pred, y_pred_prob_auc)
        return metrics, cm
    
    else:
        metrics = {}
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        # import pdb; pdb.set_trace()
        y_pred_round = np.round(y_pred).astype(int)

        metrics={}
        cm = calculate_confusion_matrix(y_true.astype(int), y_pred_round, num_class)

        regression_metrics = calculate_regression_metrics(y_true, y_pred)
        metrics.update(regression_metrics)

        # regression_category_metrics = calculate_regression_metrics_category(y_true, y_pred)
        # metrics.update(regression_category_metrics)
        
        classification_metrics = calculate_classification_metrics(y_true, y_pred, num_classes=num_class)
        metrics.update(classification_metrics)
        
        if y_pred_prob_auc is not None:
            # import pdb; pdb.set_trace()
            ndcg_metrics = calculate_ndcg_metrics(y_true, y_pred_prob_auc, y_pred_round, is_trace=True)
            metrics.update(ndcg_metrics)
        else:
            ndcg_metrics = calculate_ndcg_metrics(y_true, y_pred, y_pred_round)
            metrics.update(ndcg_metrics)

        # new_y_true, new_y_pred = reduce_classes_by_threshold(y_true, y_pred, 7, 9)
        
        return metrics, cm
