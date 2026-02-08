import json
import os
import re
import glob
from tqdm import tqdm
from config.config import DATA_PATH, FEATURES
from utils.file_utils import check_file_exists
from collections import defaultdict
from llm.output_constrain import load_formatted_data

feature_name = FEATURES


def extract_json_between_markers(raw_str):
    json_start = raw_str.find("```json")
    if json_start == -1:  
        print("No ```json` found in the string.")
        return ""
    
    sub_str_after_json = raw_str[json_start:]  
    newline_pos_in_sub = sub_str_after_json.find("\n```")
    
    if newline_pos_in_sub == -1:
        extracted_content = sub_str_after_json
        print("Found '```json' but no subsequent '\\n```', returning all content after '```json'")
    else:
        extracted_content = sub_str_after_json[:newline_pos_in_sub]
        print("Sucessfully extract content between '```json' and the first '\\n```'")
    
    return extracted_content+'```'


def process_files_scoring(file_pattern, feature_name, output_path=None):
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        print(f"No files found matching the pattern '{file_pattern}'")
        return None
    
    all_ads = []
    
    # all_summary=[]
    for file_path in file_paths:
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        # content = content[8:-3]
        # parts = content.strip().split("\n\n")
        # if len(parts) != 2:
        #     raise ValueError("No expected two parts in the file content")
        # ads_part = parts[0]
        # ads_data = json.loads(ads_part)
        ads_data = load_formatted_data(content)
        # summary_part = parts[1]
        # summary = json.loads(summary_part)
        # all_summary.append(summary['summary'])
        
        # import pdb; pdb.set_trace()
        if isinstance(ads_data, list):
            all_ads.extend(ads_data)
            print(f"Successfully parsed file: {file_path}, added {len(ads_data)} ads data")
        elif isinstance(ads_data, dict):
            for key, value in ads_data.items():
                if isinstance(value, list) and 'ad_id' in value[0]:
                    all_ads.extend(value)
                    break
        else:
            print(f"Warning: JSON in file {file_path} is not in list format")
            continue
            # import pdb;pdb.set_trace()
    
    print(f"\n Collected a total of {len(all_ads)} ads data")
    
    if not all_ads:
        print("No valid ad data for analysis")
        return None
    
    score_sums = defaultdict(float)
    score_counts = defaultdict(int)
    for ad in all_ads:
        if 'importance_scores' in ad and isinstance(ad['importance_scores'], list):
            for key, value in zip(feature_name, ad['importance_scores']):
                if isinstance(value, (int, float)):
                    score_sums[key] += value
                    score_counts[key] += 1
    avg_importance_scores = {}
    for key in score_sums:
        if score_counts[key] > 0:
            avg_importance_scores[key] = round(score_sums[key] / score_counts[key])
    
    for key, value in sorted(avg_importance_scores.items()):
        print(f"{key}: {value:.6f}")
    
    result = {
        'avg_importance_scores': avg_importance_scores
    }
    str_avg_importance_scores = str(avg_importance_scores)
    str_avg_importance_scores = str_avg_importance_scores.replace("'", "")
    str_avg_importance_scores = str_avg_importance_scores.replace(" ", "")
    print(str_avg_importance_scores)
    if output_path:
        with open(output_path+'.json', 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        with open(output_path+'.txt', 'w', encoding='utf-8') as file:
            file.write(str_avg_importance_scores)  # 直接写入字符串
        
def fix_truncated_text(original_text):
    if original_text[8]=='[':
        return original_text.rsplit('}', 1)[0] + '}\n ]```'
    elif original_text[8]=='{':
        if "ad_id" in original_text[:20]:
            insert_original_text = original_text[:8] + '[\n' + original_text[8:]
            return insert_original_text.rsplit('}', 1)[0] + '}\n ]```'
        else:
            return original_text.rsplit('}', 1)[0] + '}\n }```'
    else:
        return extract_json_between_markers(original_text)
        # print(original_text[8])
        # raise TypeError("Unexpected type of the first character")

def process_files_ranking(file_pattern, feature_name, output_path=None):
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        print(f"No files found matching the pattern '{file_pattern}'")
        return None
    
    print(f"Found {len(file_paths)} files to process")
    
    
    # all_ads = []

    all_importance=[]
    
    
    # all_summary=[]
    for file_path in file_paths:
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        

        if content[-3:] == '```':
            if content[:3] == '```':
                cut_content = content[8:-3]
                # parts = content.strip().split("\n\n")
                try:
                    ads_data = json.loads(cut_content)
                except:
                    print(cut_content)
                    raise Exception(f"Warning: File {file_path} has JSON format error")
                    
            else:
                fixed_content = fix_truncated_text(content)
                try:
                    cut_fixed_content = fixed_content[8:-3]
                    # parts = content.strip().split("\n\n")
                    ads_data = json.loads(cut_fixed_content)
                except:
                    print(fixed_content)
                    raise Exception(f"Warning: File {file_path} has JSON format error")
        else:
            fixed_content = fix_truncated_text(content)
            print(fixed_content)
            try: 
                cut_fixed_content = fixed_content[8:-3]
            
                ads_data = json.loads(cut_fixed_content)
            except:
                print(cut_fixed_content)
                raise Exception(f"Warning: File {file_path} has JSON format error")
                

        if isinstance(ads_data, dict) and 'ad_id' in ads_data:
            ads_data = [ads_data]
        for data in ads_data:
            weights = {}
            if 'ad_id' not in data:
                ad_id = data
                data = ads_data[ad_id]
            else:
                ad_id = data['ad_id']
            if 'features_ranking_list' in data:
                features_ranking_list = data['features_ranking_list']
            elif 'features ranking list' in data:
                features_ranking_list = data['features ranking list']
            n = len(features_ranking_list)
            for i, feature in enumerate(features_ranking_list):
                weight = 1.0 - (i / (n - 1)) if n > 1 else 1.0
                weights[feature] = round(weight, 4) 
            importance_data = {'ad_id':ad_id, 'importance_scores':weights}
            all_importance.append(importance_data)

    score_sums = defaultdict(float)
    score_counts = defaultdict(int)
    for ad in all_importance:
        if 'importance_scores' in ad and isinstance(ad['importance_scores'], dict):
            for key in ad['importance_scores']:
                value = ad['importance_scores'][key]
                if isinstance(value, (int, float)):
                    score_sums[key] += value
                    score_counts[key] += 1

    avg_importance_scores = {}
    for key in score_sums:
        if score_counts[key] > 0:
            avg_importance_scores[key] = float(score_sums[key] / score_counts[key])
    avg_importance_scores_new = {}
    # import pdb; pdb.set_trace()
    for cur_feature in feature_name:
        if cur_feature not in avg_importance_scores:
            avg_importance_scores_new[cur_feature] = 0
        else:
            avg_importance_scores_new[cur_feature] = avg_importance_scores[cur_feature]
    
    result = {
        'avg_importance_scores': avg_importance_scores_new
    }
    if output_path:
        with open(output_path + '.json', 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
    sorted_features = sorted(avg_importance_scores.items(), key=lambda item: item[1], reverse=True)
    sorted_feature_nums = [item[0] for item in sorted_features]
    result_str = ",".join(sorted_feature_nums)
    print('ranking:',result_str)
    with open(output_path + '.txt', 'w', encoding='utf-8') as file:
        file.write(result_str)
    
def run_merge_training_result(setting: str): # setting: 'scoring' or 'ranking'
    file_pattern = os.path.join(DATA_PATH['train_result_dir'], '*-*.txt')
    output_path = os.path.join(DATA_PATH['train_result_dir'], 'summary')
    if setting == 'scoring':
        process_files_scoring(file_pattern, feature_name, output_path)
    elif setting == 'ranking':
        process_files_ranking(file_pattern, feature_name, output_path)
    else:
        raise ValueError("Invalid setting. Please use 'scoring', 'ranking' or 'summary.")

