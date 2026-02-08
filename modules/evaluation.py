import json
import os
import numpy as np
from config.config import DATA_PATH, PRED_NUM
from utils.data_loader_writer import load_ground_truth, save_multiple_matrices_to_csv
from utils.metrics import evaluate_predictions
from modules.mapping import apply_mapping, learn_mapping, learn_mapping_clf
import glob

def process_files_predictions(file_pattern, ground_truths):
    file_paths = glob.glob(file_pattern)
    if not file_paths:
        print(f"No file in '{file_pattern}'")
        return None
    
    print(f"Found {len(file_paths)} files")
    
    predictions = {}
    results = {}
    for file_path in file_paths:
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        content = content[8:-3]
        try:
            result_list = json.loads(content)
        except:
            result_list = []
            for c in content.split('\n'):
                try:
                    result_list.append(json.loads(c))
                except:
                    print(c)
        
        if isinstance(result_list, list):
            for result in result_list:
                ad_id = result['ad_id']
                if str(ad_id) not in ground_truths:
                    continue
                prediction_level = result['Popularity level']
                
                try:
                    prediction = int(prediction_level)
                    if not (1 <= prediction <= int(PRED_NUM)):
                        raise ValueError
                except (ValueError, TypeError):
                    if isinstance(prediction_level, list):
                        for num in range(1, int(PRED_NUM)+1):
                            import pdb; pdb.set_trace()
                            if str(num) in prediction_level:
                                prediction = num
                                break
                    else:
                        continue
                
                reason = result['Reason']
                gt = ground_truths[str(ad_id)]
                results[ad_id] = {'label': gt, 'prediction': prediction, 'reason': reason}
                predictions[str(ad_id)] = prediction
        else:
            print(f"Warning: file {file_path} json is not list")
    
    
    return results, predictions

def generate_pred_mapping(gt_mapping, pred_num):
    cate_raw_values = {}
    for raw_val, cate in gt_mapping.items():
        if cate not in cate_raw_values:
            cate_raw_values[cate] = []
        cate_raw_values[cate].append(raw_val)
    
    cate_raw_info = {}
    total_raw_length = 0  
    for cate, raw_vals in sorted(cate_raw_values.items()):
        raw_min = min(raw_vals)
        raw_max = max(raw_vals)
        raw_length = raw_max - raw_min + 1  
        cate_raw_info[cate] = {
            "raw_min": raw_min,
            "raw_max": raw_max,
            "raw_length": raw_length
        }
        total_raw_length += raw_length
    
    cate_pred_length = {}
    remaining_length = pred_num 
    for cate in sorted(cate_raw_info.keys()):
        raw_length = cate_raw_info[cate]["raw_length"]
        base_length = (raw_length / total_raw_length) * pred_num
        cate_pred_length[cate] = round(base_length)  
        remaining_length -= cate_pred_length[cate]
    
    if remaining_length != 0:
        max_length_cate = max(cate_pred_length.keys(), key=lambda x: cate_pred_length[x])
        cate_pred_length[max_length_cate] += remaining_length
    
    interval_info = {}
    current_start = 1 
    for cate in sorted(cate_raw_info.keys()): 
        pred_length = cate_pred_length[cate]
        current_end = current_start + pred_length - 1 
        current_end = min(current_end, pred_num)
        interval_info[cate] = [current_start, current_end]
        current_start = current_end + 1
        if current_start > pred_num:
            break
    
    pred_mapping = {}
    for cate, [start, end] in interval_info.items():
        for pred_val in range(start, end + 1):
            pred_mapping[pred_val] = cate
    
    return pred_mapping, interval_info

def save_lists_to_json(y_true, strict_predictions, file_path):
    if isinstance(y_true, np.ndarray):
        y_true = y_true.tolist()
    if isinstance(strict_predictions, np.ndarray):
        strict_predictions = strict_predictions.tolist()
    
    data = {
        'y_true': y_true,
        'strict_predictions': strict_predictions
    }
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
def run_evaluation(phrase='dev',merge_type='all', mapping_focus_cate_weight={}, mapping_IsErrorLoss=0):
    ground_truths_org = load_ground_truth(phrase)
    if merge_type=='all':
        results, predictions = process_files_predictions(f"{DATA_PATH[f'{phrase}_result_dir']}/[0-9]*-[0-9]*.txt", ground_truths_org)
    elif merge_type=='by_region':
        results, predictions = process_files_predictions(f"{DATA_PATH[f'{phrase}_result_dir']}/*_[0-9]*-[0-9]*.txt", ground_truths_org)

    with open(f"{DATA_PATH[f'{phrase}_result_dir']}/all_results_{merge_type}.json", 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(f"{DATA_PATH[f'{phrase}_result_dir']}/predictions_{merge_type}.json", 'w', encoding='utf-8') as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)
    
    print(predictions)
    print(len(ground_truths_org), len(predictions))
    if not set(predictions.keys()).issubset(set(ground_truths_org.keys())):
        extra_keys = set(predictions.keys()) - set(ground_truths_org.keys())
        raise ValueError(f"predictions include the key that isn't exist in ground_truths: {extra_keys}")
    
    ad_ids = list(predictions.keys())
    y_true = [ground_truths_org[ad_id] for ad_id in ad_ids]
    y_pred = [predictions[ad_id] for ad_id in ad_ids]
    
    gt_mapping = {1.0: 1, 2.0: 1, 3.0: 1, 
                  4.0: 2, 5.0: 2, 
                  6.0: 3, 7.0: 3, 
                  8.0: 4, 9.0: 4, 10.0: 4}
    y_true_gtmap = apply_mapping(y_true, gt_mapping)


    gt_mapping_clfselect = {1.0: 0, 2.0: 0, 3.0: 0, 
                  4.0: 0, 5.0: 0, 
                  6.0: 0, 7.0: 0, 
                  8.0: 1, 9.0: 1, 10.0: 1}
    clfselect_label = apply_mapping(y_true, gt_mapping_clfselect)

    gt_mapping_clffilter = {1.0: 1, 2.0: 1, 3.0: 1, 
                  4.0: 0, 5.0: 0, 
                  6.0: 0, 7.0: 0, 
                  8.0: 0, 9.0: 0, 10.0: 0}
    clffilter_label = apply_mapping(y_true, gt_mapping_clffilter)


    pred_mapping, _ = generate_pred_mapping(gt_mapping, int(PRED_NUM))
    print('pred_mapping', pred_mapping)
    y_pred_gtmap = apply_mapping(y_pred, pred_mapping)
    metrics_gtmap, cm_gtmap = evaluate_predictions(np.array(y_true_gtmap), np.array(y_pred_gtmap),  np.array(y_pred), num_class=4) 
    print(metrics_gtmap)
    print(cm_gtmap)
    


    if phrase=='dev':
        strict_mapping, strict_predictions, _ , _ = learn_mapping(y_true_gtmap, y_pred, mapping_focus_cate_weight,mapping_IsErrorLoss)
        metrics_strict, cm_strict = evaluate_predictions(np.array(y_true_gtmap), np.array(strict_predictions), num_class=4)
        print(metrics_strict)
        print(cm_strict)

        clfselect_mapping, clfselect_predictions, clffilter_mapping, clffilter_predictions = learn_mapping_clf(clfselect_label, clffilter_label, y_pred)
        metrics_clfselect, cm_clfselect = evaluate_predictions(np.array(clfselect_label), np.array(clfselect_predictions), num_class=2)
        print(metrics_clfselect)
        print(cm_clfselect)
        metrics_clffilter, cm_clffilter = evaluate_predictions(np.array(clffilter_label), np.array(clffilter_predictions), num_class=2)
        print(metrics_clffilter)
        print(cm_clffilter)

        combined_data = {
            "strict_mapping": strict_mapping,
            "clfselect_mapping": clfselect_mapping,
            "clffilter_mapping": clffilter_mapping
        }
        print(combined_data)
        with open(f"{DATA_PATH[f'{phrase}_result_dir']}/learned_mapping_{merge_type}_{mapping_focus_cate_weight}_{mapping_IsErrorLoss}.json", 'w', encoding='utf-8') as file:

            json.dump(combined_data, file, ensure_ascii=False, indent=2)
    elif phrase=='test':
        with open(f"{DATA_PATH[f'dev_result_dir']}/learned_mapping_{merge_type}_{mapping_focus_cate_weight}_{mapping_IsErrorLoss}.json", 'r', encoding='utf-8') as file:
            combined_data = json.load(file)
        strict_mapping = combined_data['strict_mapping']
        clfselect_mapping = combined_data['clfselect_mapping']
        clffilter_mapping = combined_data['clffilter_mapping']
        strict_mapping = {int(k): v for k, v in strict_mapping.items()}
        clfselect_mapping = {int(k): v for k, v in clfselect_mapping.items()}
        clffilter_mapping = {int(k): v for k, v in clffilter_mapping.items()}

        strict_predictions = apply_mapping(y_pred, strict_mapping)
        clfselect_predictions = apply_mapping(y_pred, clfselect_mapping)
        clffilter_predictions = apply_mapping(y_pred, clffilter_mapping)

        metrics_strict, cm_strict = evaluate_predictions(np.array(y_true_gtmap), np.array(strict_predictions), num_class=4)
        print(metrics_strict)
        print(cm_strict)
        # import pdb; pdb.set_trace()
        save_lists_to_json(y_true_gtmap, strict_predictions, "ali_ours_tmp.json")

        metrics_clfselect, cm_clfselect = evaluate_predictions(np.array(clfselect_label), np.array(clfselect_predictions), num_class=2)
        print(metrics_clfselect)
        print(cm_clfselect)
        # save_lists_to_json(clfselect_label, clfselect_predictions, "ali_ours_tmp_select.json")
        # import pdb; pdb.set_trace()
        metrics_clffilter, cm_clffilter = evaluate_predictions(np.array(clffilter_label), np.array(clffilter_predictions), num_class=2)
        print(metrics_clffilter)
        print(cm_clffilter)

    metrics = {
        'gt_mapping': metrics_gtmap,
        'strict_mapping': metrics_strict,
        'clfselect_mapping': metrics_clfselect,
        'clffilter_mapping': metrics_clffilter
    }
    
    output_path = DATA_PATH[f'llm_metrics_path_{phrase}'].split('.')[0]+f'_+{merge_type}_{mapping_focus_cate_weight}{mapping_IsErrorLoss}.json'
    # matrices = [cm_org, cm_gtmap, cm_strict, cm_unordered]
    # names = ["cm_org", "cm_gtmap", "cm_strictmap", "cm_unoderedmap"]
    matrices = [cm_gtmap, cm_strict, cm_clfselect, cm_clffilter]
    names = ["cm_org", "cm_strictmap", "cm_clfselect", "cm_clffilter"]
    save_multiple_matrices_to_csv(matrices, names, f'{output_path.split('.')[0]}_cm.csv')
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    