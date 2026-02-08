import json
import os
import re
import glob
from tqdm import tqdm
from config.config import DATA_PATH, FEATURES
from utils.file_utils import check_file_exists
from collections import defaultdict
from llm.output_constrain import load_formatted_data
import math

feature_name = FEATURES
def process_files_scoring(file_pattern, id2region, feature_name, output_path=None):
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        return None
    
    all_ads = []
    
    for file_path in file_paths:
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        ads_data = load_formatted_data(content)
        if isinstance(ads_data, list):
            all_ads.extend(ads_data)
        elif isinstance(ads_data, dict):
            for key, value in ads_data.items():
                if isinstance(value, list) and 'ad_id' in value[0]:
                    all_ads.extend(value)
                    break
        else:
            continue
    
    if not all_ads:
        return None
    
    score_sums_region = dict()
    score_counts_region = dict()
    for ad in all_ads:
        if 'ad_id' in ad and int(ad['ad_id']) in id2region:
            region = id2region[int(ad['ad_id'])]
            score_sums_region[region] = defaultdict(float)
            score_counts_region[region] = defaultdict(int)
        if 'importance_scores' in ad and isinstance(ad['importance_scores'], list):
            for key, value in zip(feature_name, ad['importance_scores']):
                if isinstance(value, (int, float)):
                    score_sums_region[region][key] += value
                    score_counts_region[region][key]+= 1
    avg_importance_scores = dict()
    for region in score_sums_region:
        avg_importance_scores[region] = dict()
        for key in score_sums_region[region]:
            if score_counts_region[region][key] > 0:
                avg_importance_scores[region][key] = round(score_sums_region[region][key] / score_counts_region[region][key])
    
    for region in avg_importance_scores:
        for key, value in sorted(avg_importance_scores[region].items()):
            print(f"{key}: {value:.6f}")
    
    output_region2str = dict()
    for region in avg_importance_scores:
        str_avg_importance_scores = str(avg_importance_scores[region])
        str_avg_importance_scores = str_avg_importance_scores.replace("'", "")
        str_avg_importance_scores = str_avg_importance_scores.replace(" ", "")
        print(str_avg_importance_scores)
        output_region2str[region] = str_avg_importance_scores
    
    if output_path:
        with open(output_path+'_by_region.json', 'w', encoding='utf-8') as file:
            json.dump(avg_importance_scores, file, ensure_ascii=False, indent=2)
        with open(output_path+'_by_region_txt.json', 'w', encoding='utf-8') as file:
            json.dump(output_region2str, file, ensure_ascii=False, indent=2) 
        
def fix_truncated_text(original_text):
    if original_text[8]=='[':
        return original_text.rsplit('}', 1)[0] + '}\n ]```'
    elif original_text[8]=='{':
        if "ad_id" in original_text[:20]:
            insert_original_text = original_text[:8] + '[\n' + original_text[8:]
            return insert_original_text.rsplit('}', 1)[0] + '}\n ]```'
        else:
            return original_text.rsplit('}', 1)[0] + '}\n }```'
    
        return fixed_text
    else:
        print(original_text[8])
        raise TypeError("Unexpected type of the first character")


def process_files_ranking(file_pattern, id2region, feature_name, output_path=None):
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        return None
    
    all_importance=[]
    
    # all_summary=[]
    for file_path in file_paths:
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if content[-3:] == '```':

            try:
                cut_content = content[8:-3]
                # parts = content.strip().split("\n\n")
                ads_data = json.loads(cut_content)
            except:
                print(cut_content)
                raise Exception(f"warning: File {file_path} has JSON format error")
        else:
            fixed_content = fix_truncated_text(content)
            print(fixed_content)
            try: 
                cut_fixed_content = fixed_content[8:-3]
            
                ads_data = json.loads(cut_fixed_content)
            except:
                print(cut_fixed_content)
                raise Exception(f"warning: File {file_path} has JSON format error")
                
       

        for data in ads_data:
            weights = {}
            # import pdb; pdb.set_trace()
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

    score_sums_region = dict()
    score_counts_region = dict()
    for ad in all_importance:
        ad_id = int(ad['ad_id'])
        if ad_id in id2region:
            region = id2region[ad_id]
            if region not in score_sums_region:
                score_sums_region[region] = defaultdict(float)
                score_counts_region[region] = defaultdict(int)
            if 'importance_scores' in ad and isinstance(ad['importance_scores'], list):
                for key, value in zip(feature_name, ad['importance_scores']):
                    if isinstance(value, (int, float)):
                        score_sums_region[region][key] += value
                        score_counts_region[region][key] += 1
            elif 'importance_scores' in ad and isinstance(ad['importance_scores'], dict):
                for key in ad['importance_scores']:
                    value = ad['importance_scores'][key]
                    if isinstance(value, (int, float)):
                        score_sums_region[region][key] += value
                        score_counts_region[region][key] += 1
        else:
            raise ValueError(f"ad_id {ad_id} not in id2region")

    avg_importance_scores = dict()
    output_region2str = dict()
    for region in score_sums_region:
        avg_importance_scores[region] = dict()
        for key in score_sums_region[region]:
            if score_counts_region[region][key] > 0:
                avg_importance_scores[region][key] = float(score_sums_region[region][key] / score_counts_region[region][key])

        sorted_features = sorted(avg_importance_scores[region].items(), key=lambda item: item[1], reverse=True)
        sorted_feature_nums = [item[0] for item in sorted_features]
        result_str = ",".join(sorted_feature_nums)
        print(f'{region} ranking:{result_str}')
        output_region2str[region] = result_str

    # import pdb; pdb.set_trace()
    if output_path:
        with open(output_path + '_by_region.json', 'w', encoding='utf-8') as file:
            json.dump(avg_importance_scores, file, ensure_ascii=False, indent=2)
        
        with open(output_path + '_by_region_txt.json', 'w', encoding='utf-8') as file:
            json.dump(output_region2str, file, ensure_ascii=False, indent=2)
        


        
        
        
def run_merge_training_result_by_region(setting: str): # setting: 'scoring' or 'ranking'
    """总结scoring的training结果"""
    file_pattern = os.path.join(DATA_PATH['train_result_dir'], '*-*.txt')
    output_path = os.path.join(DATA_PATH['train_result_dir'], 'summary')
    
    import pandas as pd
    # input_df = pd.read_csv(DATA_PATH['train_data_path'].split('.')[0]+'.csv')
    input_df = pd.read_csv(DATA_PATH['train_data_path'].split('_')[0]+'_'+DATA_PATH['train_data_path'].split('_')[1]+'_df_train.csv')
    id2region = input_df.set_index('adID')['region'].to_dict()
    
    if setting == 'scoring':
        process_files_scoring(file_pattern, id2region, feature_name, output_path)
    elif setting == 'ranking':
        process_files_ranking(file_pattern, id2region, feature_name, output_path)
    else:
        raise ValueError("Invalid setting. Please use 'scoring', 'ranking' or 'summary.")

