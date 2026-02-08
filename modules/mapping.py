import numpy as np
from utils.metrics import weighted_error_rate
from config.config import DATA_PATH, PRED_NUM
import json
import random
import math
from itertools import combinations,permutations

def strict_order_mapping_dp(ground_truths, predictions):
    n = 10 
    k = 4
    
    pred_dist = [[] for _ in range(n + 1)] 
    for gt, pred in zip(ground_truths, predictions):
        pred_dist[pred].append(gt)
    
    def calculate_error(i, j, c):
        error = 0
        count = 0
        for p in range(i, j + 1):
            for gt in pred_dist[p]:
                error += abs(gt - c)
                count += 1
        return error / max(1, count)
    
    dp = np.full((n + 1, k + 1), float('inf'))
    dp[0][0] = 0
    
    split_points = np.zeros((n + 1, k + 1), dtype=int)
    
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            for m in range(j - 1, i):
                current_error = dp[m][j - 1] + calculate_error(m + 1, i, j)
                if current_error < dp[i][j]:
                    dp[i][j] = current_error
                    split_points[i][j] = m
    
    thresholds = [0] * k
    current_i = n
    for j in range(k, 0, -1):
        thresholds[j - 1] = split_points[current_i][j]
        current_i = thresholds[j - 1]
    
    thresholds.sort()
    
    mapping = {}
    for x in range(1, PRED_NUM+1):
        if x <= thresholds[0]+1:
            mapping[x]=1
        elif x <= thresholds[1]+1:
            mapping[x]=2
        elif x <= thresholds[2]+1:
            mapping[x]=3
        else:
            mapping[x]=4
    
    return mapping, thresholds

def non_strict_mapping_at_least1(ground_truths, predictions, pred_num, true_num=4, 
                                focus_cate_weight={}, IsErrorLoss=1):
    n = pred_num  
    k = true_num 
    use_ranking_loss = len(focus_cate_weight) > 0 
    total_samples = len(predictions) 
    
    pred_dist = [[] for _ in range(n + 1)] 
    pred_counts = {c: np.zeros(n + 1) for c in range(1, k+1)}
    for gt, pred in zip(ground_truths, predictions):
        pred_dist[pred].append(gt)
        pred_counts[gt][pred] += 1
    
    prefix_sums = {c: np.zeros(n + 1) for c in range(1, k+1)}
    for c in range(1, k+1):
        for i in range(1, n+1):
            prefix_sums[c][i] = prefix_sums[c][i-1] + pred_counts[c][i]
    
    pred_total_counts = np.zeros(n + 1)  
    for pred in predictions:
        pred_total_counts[pred] += 1
    
    prefix_total = np.zeros(n + 1)
    for i in range(1, n + 1):
        prefix_total[i] = prefix_total[i-1] + pred_total_counts[i]

    def calculate_general_error(pred_set, c):
        total_error = 0
        count = 0
        for p in pred_set:
            for gt in pred_dist[p]:
                total_error += abs(gt - c)
                count += 1
        avg_error = total_error / max(1, count)  
        return avg_error, total_error, count 

    def calculate_single_ranking_loss(pred_set, c, focus_c):
        if not pred_set:
            return 0
        current_score = sum(pred_set) / len(pred_set)
        
        other_preds = set(range(1, n+1)) - pred_set
        prev_pos = sum(prefix_sums[focus_c][p] - prefix_sums[focus_c][p-1] for p in other_preds)
        prev_neg = len(other_preds) - prev_pos
        
        curr_pos = sum(prefix_sums[focus_c][p] - prefix_sums[focus_c][p-1] for p in pred_set)
        curr_neg = len(pred_set) - curr_pos
        
        if c == focus_c:
            loss = prev_pos * curr_neg
        else:
            loss = curr_pos * prev_neg
        
        inner_loss = curr_pos * curr_neg
        return loss + inner_loss
    
    def calculate_auc_related_loss(pred_set, c, focus_c):
        if not pred_set:
            return 0.0
            
        current_score = sum(pred_set) / len(pred_set)
        temperature = 0.1
        
        other_preds = set(range(1, n+1)) - pred_set
        
        prev_pos = sum(prefix_sums[focus_c][p] - prefix_sums[focus_c][p-1] for p in other_preds)
        prev_total = len(other_preds)
        prev_neg = prev_total - prev_pos
        
        curr_pos = sum(prefix_sums[focus_c][p] - prefix_sums[focus_c][p-1] for p in pred_set)
        curr_total = len(pred_set)
        curr_neg = curr_total - curr_pos
        
        prev_pos, prev_neg = max(0, prev_pos), max(0, prev_neg)
        curr_pos, curr_neg = max(0, curr_pos), max(0, curr_neg)
        
        loss = 0.0
        
        if c == focus_c:
            if prev_neg > 0 and curr_pos > 0:
                score_diff = current_score - (sum(other_preds)/len(other_preds) if other_preds else 0)
                loss += prev_neg * curr_pos * np.log(1 + np.exp(-score_diff / temperature))
        else:
            if prev_pos > 0 and curr_neg > 0:
                score_diff = (sum(other_preds)/len(other_preds) if other_preds else 0) - current_score
                loss += prev_pos * curr_neg * np.log(1 + np.exp(-score_diff / temperature))
        
        if curr_pos > 0 and curr_neg > 0:
            loss += curr_pos * curr_neg * np.log(1 + np.exp(-1 / temperature))
            
        return loss

    def calculate_auc_related_loss_01(pred_set, c, focus_c):
        if not pred_set:
            return 0.0
            
        current_score = sum(pred_set) / len(pred_set)
        temperature = 0.1
        
        other_preds = set(range(1, n+1)) - pred_set
        
        prev_pos = sum(prefix_sums[focus_c][p] - prefix_sums[focus_c][p-1] for p in other_preds)
        prev_total = len(other_preds) if other_preds else 0
        prev_neg = prev_total - prev_pos
        
        curr_pos = sum(prefix_sums[focus_c][p] - prefix_sums[focus_c][p-1] for p in pred_set)
        curr_total = len(pred_set)
        curr_neg = curr_total - curr_pos
        
        prev_pos, prev_neg = max(0, prev_pos), max(0, prev_neg)
        curr_pos, curr_neg = max(0, curr_pos), max(0, curr_neg)
        
        max_single_loss = np.log(1 + np.exp(1 / temperature))
        max_single_loss = max(max_single_loss, 1e-6)
        
        total_pos = prefix_sums[focus_c][n]
        total_neg = max(0, total_samples - total_pos)
        max_pair_count = max(1, total_pos * total_neg)
        
        loss_part1 = 0.0
        if c == focus_c and prev_neg > 0 and curr_pos > 0 and other_preds:
            other_score = sum(other_preds) / len(other_preds)
            score_diff = current_score - other_score
            single_loss = np.log(1 + np.exp(-score_diff / temperature))
            normalized_single = single_loss / max_single_loss
            normalized_pairs = (prev_neg * curr_pos) / max_pair_count
            loss_part1 = normalized_single * normalized_pairs
        
        loss_part2 = 0.0
        if c != focus_c and prev_pos > 0 and curr_neg > 0 and other_preds:
            other_score = sum(other_preds) / len(other_preds)
            score_diff = other_score - current_score
            single_loss = np.log(1 + np.exp(-score_diff / temperature))
            normalized_single = single_loss / max_single_loss
            normalized_pairs = (prev_pos * curr_neg) / max_pair_count
            loss_part2 = normalized_single * normalized_pairs
        
        loss_part3 = 0.0
        if curr_pos > 0 and curr_neg > 0:
            single_loss = np.log(1 + np.exp(-1 / temperature))
            normalized_single = single_loss / max_single_loss
            normalized_pairs = (curr_pos * curr_neg) / max_pair_count
            loss_part3 = normalized_single * normalized_pairs
        
        total_loss = (loss_part1 + loss_part2 + loss_part3) / 3.0
        return max(0.0, min(1.0, total_loss))
    
    min_thresholds = k - 1 
    max_thresholds = n - 1
    
    best_loss = float('inf')
    best_mapping_dict = None 
    best_thresholds = None
    best_mae = None
    
    for num_thresholds in range(min_thresholds, max_thresholds + 1):
        for thresholds in combinations(range(1, n), num_thresholds):
            thresholds = sorted(thresholds)
            intervals = []
            prev = 0
            for t in thresholds:
                intervals.append(set(range(prev + 1, t + 1)))
                prev = t
            intervals.append(set(range(prev + 1, n + 1)))
            
            if len(intervals) != k:
                continue
                
            for category_mapping in permutations(range(1, k + 1)):
                total_loss = 0.0
                total_error = 0
                total_count = 0
                
                if use_ranking_loss:
                    ranking_loss = 0.0
                    for focus_c, weight in focus_cate_weight.items():
                        for interval, c in zip(intervals, category_mapping):
                            loss = calculate_auc_related_loss_01(interval, c, focus_c)
                            ranking_loss += loss * weight
                    total_loss += ranking_loss
                
                if IsErrorLoss:
                    error_loss = 0.0
                    for interval, c in zip(intervals, category_mapping):
                        avg_err, total_err, count = calculate_general_error(interval, c)
                        error_loss += avg_err
                        total_error += total_err
                        total_count += count
                    error_loss /= len(intervals)
                    total_loss += error_loss
                
                mae = total_error / max(1, total_count) if total_count > 0 else 0
                
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_mapping_dict = {}
                    for interval, c in zip(intervals, category_mapping):
                        for p in interval:
                            best_mapping_dict[p] = c
                    best_thresholds = thresholds
                    best_mae = mae
    
    for p in range(1, n+1):
        if p not in best_mapping_dict:
            best_mapping_dict[p] = 1 
    print(best_mapping_dict)
    # import pdb; pdb.set_trace()
    return best_mapping_dict, best_thresholds, best_loss, best_mae


def strict_order_mapping_dp_atleast1(ground_truths, predictions, pred_num, true_num=4, 
                                       focus_cate_weight={}, IsErrorLoss=1):
    n = pred_num  
    k = true_num 
    use_ranking_loss = len(focus_cate_weight) > 0 
    
    pred_dist = [[] for _ in range(n + 1)] 
    pred_counts = {c: np.zeros(n + 1) for c in range(1, k+1)} 
    for gt, pred in zip(ground_truths, predictions):
        pred_dist[pred].append(gt)
        pred_counts[gt][pred] += 1
    
    prefix_sums = {c: np.zeros(n + 1) for c in range(1, k+1)}
    for c in range(1, k+1):
        for i in range(1, n+1):
            prefix_sums[c][i] = prefix_sums[c][i-1] + pred_counts[c][i]
    pred_total_counts = np.zeros(n + 1)  
    for pred in predictions:
        pred_total_counts[pred] += 1
    prefix_total = np.zeros(n + 1)
    for i in range(1, n + 1):
        prefix_total[i] = prefix_total[i-1] + pred_total_counts[i]

    def calculate_general_error(i, j, c):
        total_error = 0
        count = 0
        for p in range(i, j + 1):
            for gt in pred_dist[p]:
                total_error += abs(gt - c)
                count += 1
        avg_error = total_error / max(1, count) 
        return avg_error, total_error, count 

    def calculate_single_ranking_loss(m, i, c, focus_c):
        current_score = (m + 1 + i) / 2
        prev_pos = prefix_sums[focus_c][m]
        prev_neg = m - prev_pos
        curr_pos = prefix_sums[focus_c][i] - prefix_sums[focus_c][m]
        curr_neg = (i - m) - curr_pos
        
        if c == focus_c:
            loss = prev_pos * curr_neg
        else:
            loss = curr_pos * prev_neg
        
        inner_loss = curr_pos * curr_neg
        if loss<0 or inner_loss<0:
            print(f"prev_pos={prev_pos}, curr_neg={curr_neg}, prev_neg={prev_neg}, curr_neg={curr_neg}, loss={loss}, inner_loss={inner_loss}")
            import pdb; pdb.set_trace()
        return loss + inner_loss
    
    def calculate_auc_related_loss(m, i, c, focus_c):
        current_score = (m + 1 + i) / 2
        
        prev_pos = prefix_sums[focus_c][m] 
        prev_total = prefix_total[m] 
        prev_neg = prev_total - prev_pos 
        
        curr_pos = prefix_sums[focus_c][i] - prefix_sums[focus_c][m] 
        curr_total = prefix_total[i] - prefix_total[m]  
        curr_neg = curr_total - curr_pos
        
        prev_pos, prev_neg = max(0, prev_pos), max(0, prev_neg)
        curr_pos, curr_neg = max(0, curr_pos), max(0, curr_neg)
        
        loss = 0.0
        temperature = 0.1 
        
        if c == focus_c:
            if prev_neg > 0 and curr_pos > 0:
                score_diff = current_score - (m / 2) 
                loss += prev_neg * curr_pos * np.log(1 + np.exp(-score_diff / temperature))
        else:
            if prev_pos > 0 and curr_neg > 0:
                score_diff = (m / 2) - current_score 
                loss += prev_pos * curr_neg * np.log(1 + np.exp(-score_diff / temperature))
        
        if curr_pos > 0 and curr_neg > 0:
            loss += curr_pos * curr_neg * np.log(1 + np.exp(-1 / temperature))
        print('loss', loss)
        return loss

    def calculate_auc_related_loss_01(m, i, c, focus_c, total_samples, temperature=0.1):
        current_score = (m + 1 + i) / 2
        prev_pos = max(0, prefix_sums[focus_c][m])
        prev_total = max(0, prefix_total[m])
        prev_neg = max(0, prev_total - prev_pos)
        curr_pos = max(0, prefix_sums[focus_c][i] - prefix_sums[focus_c][m])
        curr_total = max(0, prefix_total[i] - prefix_total[m])
        curr_neg = max(0, curr_total - curr_pos)
        
        max_single_loss = np.log(1 + np.exp(1 / temperature))
        if max_single_loss < 1e-6: 
            max_single_loss = 1e-6
        
        total_pos = prefix_sums[focus_c][pred_num]
        total_neg = max(0, total_samples - total_pos)
        max_pair_count = max(1, total_pos * total_neg) 
        
        loss_part1 = 0.0 
        if c == focus_c and prev_neg > 0 and curr_pos > 0:
            score_diff = current_score - (m / 2)
            single_loss = np.log(1 + np.exp(-score_diff / temperature))
            normalized_single = single_loss / max_single_loss 
            normalized_pairs = (prev_neg * curr_pos) / max_pair_count 
            loss_part1 = normalized_single * normalized_pairs 
        
        loss_part2 = 0.0 
        if c != focus_c and prev_pos > 0 and curr_neg > 0:
            score_diff = (m / 2) - current_score
            single_loss = np.log(1 + np.exp(-score_diff / temperature))
            normalized_single = single_loss / max_single_loss
            normalized_pairs = (prev_pos * curr_neg) / max_pair_count
            loss_part2 = normalized_single * normalized_pairs  # 乘积≤1
        
        loss_part3 = 0.0
        if curr_pos > 0 and curr_neg > 0:
            single_loss = np.log(1 + np.exp(-1 / temperature))
            normalized_single = single_loss / max_single_loss
            normalized_pairs = (curr_pos * curr_neg) / max_pair_count
            loss_part3 = normalized_single * normalized_pairs  # 乘积≤1
        
        total_loss = (loss_part1 + loss_part2 + loss_part3) / 3.0
        total_loss = max(0.0, min(1.0, total_loss))
        print(total_loss)
        return total_loss
    
    def calculate_weighted_ranking_loss(m, i, c):
        total_loss = 0
        for focus_c, weight in focus_cate_weight.items():
            # total_loss += weight * calculate_single_ranking_loss(m, i, c, focus_c)
            total_loss += weight * calculate_auc_related_loss_01(m, i, c, focus_c, len(ground_truths))
        return total_loss

    def has_ground_truth(i, j, c):
        for p in range(i, j + 1):
            if c in pred_dist[p]:
                return True
        return False
    
    dp = np.full((n + 1, k + 1, 1 << k), float('inf'))
    dp[0][0][0] = 0 

    dp_total_error = np.full((n + 1, k + 1, 1 << k), 0)
    dp_sample_count = np.full((n + 1, k + 1, 1 << k), 0)

    split_points = np.zeros((n + 1, k + 1, 1 << k), dtype=int)
    class_assignments = np.zeros((n + 1, k + 1, 1 << k), dtype=int)

    for i in range(1, n + 1):
        for j in range(1, k + 1):
            for mask in range(1 << k):
                for m in range(j - 1, i):
                    for c in range(1, k + 1):
                        if not has_ground_truth(m + 1, i, c):
                            continue
                            
                        new_mask = mask | (1 << (c - 1))
                        
                        if use_ranking_loss:
                            ranking_loss = calculate_weighted_ranking_loss(m, i, c)
                            current_loss = ranking_loss
                            
                            if IsErrorLoss:
                                general_error, total_err, count = calculate_general_error(m + 1, i, c)
                                current_loss += general_error
                        else:
                            general_error, total_err, count = calculate_general_error(m + 1, i, c)
                            current_loss = general_error
                        print('current_loss: ', current_loss)
                        total_loss = dp[m][j - 1][mask] + current_loss
                        if total_loss < dp[i][j][new_mask]:
                            dp[i][j][new_mask] = total_loss
                            split_points[i][j][new_mask] = m
                            class_assignments[i][j][new_mask] = c
                            
                            if use_ranking_loss and IsErrorLoss:
                                dp_total_error[i][j][new_mask] = dp_total_error[m][j-1][mask] + total_err
                                dp_sample_count[i][j][new_mask] = dp_sample_count[m][j-1][mask] + count
                            elif not use_ranking_loss:
                                dp_total_error[i][j][new_mask] = dp_total_error[m][j-1][mask] + total_err
                                dp_sample_count[i][j][new_mask] = dp_sample_count[m][j-1][mask] + count

    min_loss = float('inf')
    best_mask = 0
    full_mask = (1 << k) - 1 
    
    if dp[n][k][full_mask] < min_loss:
        min_loss = dp[n][k][full_mask]
        best_mask = full_mask

    thresholds = []
    intervals = [] 
    current_i = n
    current_j = k
    current_mask = best_mask

    while current_j > 0:
        c = class_assignments[current_i][current_j][current_mask]
        m = split_points[current_i][current_j][current_mask]
        intervals.append( (m+1, current_i, c) )
        thresholds.append(m)
        current_mask ^= (1 << (c - 1))
        current_i = m
        current_j -= 1

    thresholds = thresholds[::-1]
    intervals = intervals[::-1]
    mapping = {}
    for x in range(1, n + 1):
        for (start, end, c) in intervals:
            if start <= x <= end:
                if isinstance(c, np.integer):
                    mapping[x] = c.item()
                else:
                    mapping[x] = c
                break
        else:
            mapping[x] = k
    
    total_actual_error = 0
    total_actual_count = 0
    
    for gt, pred in zip(ground_truths, predictions):
        mapped_pred = mapping.get(pred, k)
        total_actual_error += abs(gt - mapped_pred)
        total_actual_count += 1
    
    actual_mae = total_actual_error / max(1, total_actual_count)
    
    dp_calculated_mae = 0
    if not use_ranking_loss or (use_ranking_loss and IsErrorLoss):
        total_err = dp_total_error[n][k][full_mask]
        total_count = dp_sample_count[n][k][full_mask]
        dp_calculated_mae = total_err / max(1, total_count)
    
    print(f"DP final loss: {min_loss:.6f}")
    if use_ranking_loss:
        if IsErrorLoss:
            print(f"MAE in DP: {dp_calculated_mae:.6f}")
    else:
        print(f"MAE in DP: {dp_calculated_mae:.6f}")
    print(f"True MAE: {actual_mae:.6f}")
    
    if not use_ranking_loss or (use_ranking_loss and IsErrorLoss):
        mae_diff = abs(dp_calculated_mae - actual_mae)
        print(f"MAE differences: {mae_diff:.6f}")
        if mae_diff < 1e-6:
            print("same MAE, logic correct")
        else:
            print("warning: MAE mismatch, check logic")
    
    print("=========================\n")
    # import pdb; pdb.set_trace()
    return mapping, thresholds

def unordered_mapping_clustering(ground_truths, predictions, pred_num, true_num=4):
    ground_truths = [int(gt) for gt in ground_truths]
    
    pred_gt_distribution = {pred: [0] * (true_num + 1) for pred in range(1, pred_num + 1)}
    
    for gt, pred in zip(ground_truths, predictions):
        if 1 <= pred <= pred_num and 1 <= gt <= true_num:
            pred_gt_distribution[pred][gt] += 1
    
    def calculate_error(mapping):
        error = 0
        for pred, gt_dist in pred_gt_distribution.items():
            target_label = mapping[pred]
            error += sum(gt_dist[i] for i in range(1, true_num + 1) if i != target_label)
        return error / len(predictions) if predictions else 0
    
    initial_mapping = {}
    for pred in range(1, pred_num + 1):
        gt_counts = pred_gt_distribution[pred][1:]
        if sum(gt_counts) > 0:
            initial_mapping[pred] = gt_counts.index(max(gt_counts)) + 1
        else:
            initial_mapping[pred] = random.randint(1, true_num)
    
    current_mapping = initial_mapping.copy()
    best_mapping = current_mapping.copy()
    current_error = calculate_error(current_mapping)
    best_error = current_error
    
    temperature = 100.0
    cooling_rate = 0.95
    iterations_per_temp = 100
    min_temperature = 0.1
    
    tabu_list = set()
    tabu_tenure = 10
    tabu_counter = 0
    
    while temperature > min_temperature:
        for _ in range(iterations_per_temp):
            pred = random.randint(1, pred_num)
            current_target = current_mapping[pred]
            
            new_targets = [t for t in range(1, true_num + 1) if t != current_target]
            if not new_targets:
                continue
            target = random.choice(new_targets)
            
            move_key = (pred, target)
            
            if move_key in tabu_list:
                continue
            
            temp_mapping = current_mapping.copy()
            temp_mapping[pred] = target
            
            temp_error = calculate_error(temp_mapping)
            error_delta = temp_error - current_error
            
            if error_delta < 0 or random.random() < math.exp(-error_delta / temperature):
                current_mapping = temp_mapping
                current_error = temp_error
                
                tabu_list.add(move_key)
                tabu_counter += 1
                if tabu_counter > tabu_tenure:
                    tabu_list.clear()
                    tabu_counter = 0
                
                if current_error < best_error:
                    best_error = current_error
                    best_mapping = current_mapping.copy()
        
        temperature *= cooling_rate
    
    return best_mapping

def apply_mapping(scores, mapping):
    return [mapping[s] if isinstance(mapping, dict) else mapping(s) 
                         for s in scores]

def learn_mapping(ground_truths, predictions, mapping_focus_cate_weight, mapping_IsErrorLoss):
    if PRED_NUM==4:
        strict_mapping = {1:1,2:2,3:3,4:4}
    else:
        strict_mapping, thresholds = strict_order_mapping_dp_atleast1(ground_truths, predictions, int(PRED_NUM), true_num=4, 
                                        focus_cate_weight = mapping_focus_cate_weight, IsErrorLoss=mapping_IsErrorLoss)
    print("strict ordered mapping:", thresholds)
    strict_predictions = apply_mapping(predictions, strict_mapping)

    
    if PRED_NUM==4:
        unordered_mapping = {1:1,2:2,3:3,4:4}
    else:
        if int(PRED_NUM)>20:
            unordered_mapping = unordered_mapping_clustering(ground_truths, predictions, int(PRED_NUM))
        else:
            unordered_mapping, best_thresholds, best_loss, best_mae = non_strict_mapping_at_least1(ground_truths, predictions, int(PRED_NUM), true_num=4, 
                                            focus_cate_weight = mapping_focus_cate_weight, IsErrorLoss=mapping_IsErrorLoss)
            
    print("unordered mapping:", unordered_mapping)
    unordered_predictions = apply_mapping(predictions, unordered_mapping)

    return strict_mapping, strict_predictions, unordered_mapping, unordered_predictions

def loop_threshold(clf_label, y_pred, clf_type='select'):

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    unique_scores = np.unique(y_pred)
    thresholds = (unique_scores[:-1] + unique_scores[1:]) / 2
    if len(thresholds) == 0:
        thresholds = np.array([unique_scores[0]])
    
    metrics_list = []
    for threshold in thresholds:
        mapping_dict = {}
        for score in range(1,int(PRED_NUM)+1):
            if clf_type == 'select':
                mapping_dict[score] = 1 if score > threshold else 0
            else:
                mapping_dict[score] = 0 if score > threshold else 1
        
        pred = apply_mapping(y_pred, mapping_dict)
        accuracy = accuracy_score(clf_label, pred)
        precision = precision_score(clf_label, pred)  
        recall = recall_score(clf_label, pred)
        f1 = f1_score(clf_label, pred, pos_label=1) 

        
        metrics_list.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "precision":precision,
            "recall": recall,
            "f1": f1,
            "mean_metric": accuracy, 
            "mapping": mapping_dict,
            "predictions": pred
        })
    
    best_idx = np.argmax([m["mean_metric"] for m in metrics_list])
    
    best_result = metrics_list[best_idx]
    best_threshold = best_result["threshold"]
    best_mapping = best_result["mapping"]
    best_predictions = best_result["predictions"]
    best_metrics = {
        "accuracy": best_result["accuracy"],
        "precision": best_result["precision"],
        "recall": best_result["recall"],
        "f1": best_result["f1"]
    }
    
    print(f"nest threshold: {best_threshold:.4f}")
    print(f"best metrics - accuracy: {best_metrics['accuracy']:.4f}, precision: {best_metrics['precision']:.4f}, recall: {best_metrics['recall']:.4f}, f1 score: {best_metrics['f1']:.4f}")
    print(f"mapping rules: pred score > {best_threshold:.4f} → 1, else → 0")
    return best_mapping, best_predictions, best_threshold, best_metrics

def learn_mapping_clf(clfselect_label, clffilter_label, y_pred):
    clfselect_label = np.array(clfselect_label)
    clffilter_label = np.array(clffilter_label)
    y_pred = np.array(y_pred)

    clfselect_mapping, clfselect_predictions, clfselect_threshold, clfselect_metrics = loop_threshold(clfselect_label, y_pred, clf_type='select')

    clffilter_mapping, clffilter_predictions, clffilter_threshold, clffilter_metrics = loop_threshold(clffilter_label, y_pred, clf_type='filter')

    def convert_numpy_keys_to_int(mapping_dict):
        return {int(score): label for score, label in mapping_dict.items()}
    clfselect_mapping = convert_numpy_keys_to_int(clfselect_mapping)
    clffilter_mapping = convert_numpy_keys_to_int(clffilter_mapping)

    return clfselect_mapping, clfselect_predictions, clffilter_mapping, clffilter_predictions


