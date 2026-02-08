# modules/summary.py
import os
import glob, re, math, json
from tqdm import tqdm
from config.config import DATA_PATH, LLM_MODEL, MAX_INPUT_TOKENS, FEATURES
from llm.llm_client import LLMClient
from llm.prompt_builder import build_summary_prompt
from llm.input_constrain import count_tokens, truncate_text_to_tokens
from utils.file_utils import read_file, check_file_exists, create_directory
import pandas as pd


def process_files(file_pattern):
    files = glob.glob(file_pattern)
    content_list=[]
    for file_path in tqdm(files, desc="Processing files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            content_list.append(content)
    return content_list
    
def truncate_files(max_tokens, reserve_prompt, content_list):
    available_tokens = max_tokens - count_tokens(reserve_prompt) - 1000
    if len(content_list):
        tokens_per_file = max(1, available_tokens // len(content_list))
    truncated_content_list = []
    for content in content_list:
        original_tokens = count_tokens(content)
        if original_tokens > tokens_per_file:
            truncated_content = truncate_text_to_tokens(content, tokens_per_file)
            num_truncated_token = count_tokens(truncated_content)
            truncated_content_list.append(truncated_content)
        else:
            truncated_content_list.append(content)
    return truncated_content_list

def run_merge_training_summary(llm_client: LLMClient):
    if os.path.exists(f'{DATA_PATH['train_result_dir']}/summary.txt'):
        print('summary.txt already exists')
        return
    file_pattern = os.path.join(DATA_PATH['train_result_dir'], '*-*.txt')
    content_list = process_files(file_pattern)
    # summary_content = '\n'.join(content_list)
    # prompt, reserve_prompt = build_summary_prompt(summary_content, limit_tokens=True)
    # num_token = count_tokens(prompt)
    # if num_token > MAX_INPUT_TOKENS[LLM_MODEL]:
    #     print(f"input text length {num_token} exceed max limit {MAX_INPUT_TOKENS[LLM_MODEL]}, truncate files")
        # truncated_content_list = truncate_files(MAX_INPUT_TOKENS[LLM_MODEL], reserve_prompt, content_list)
        # summary_content = '\n'.join(truncated_content_list)
        # prompt, reserve_prompt = build_summary_prompt(summary_content, limit_tokens=True)
    import random
    sampled_content_list = random.sample(content_list, 100)
    summary_content = '\n'.join(sampled_content_list)
    prompt, reserve_prompt = build_summary_prompt(summary_content, limit_tokens=True)
    prompt += "Any reference to individual cases will be considered an invalid output. "

    print(f'Try {LLM_MODEL}. Prompt: {prompt}')
    llm_client.timeout=50
    status, result = llm_client.chat(prompt)
    
    if status is not None and result is not None:
        print(f"state code: {status}")
        print(f"result: {result}")
        with open(f'{DATA_PATH['train_result_dir']}/summary.txt', 'w', encoding='utf-8') as file:
            file.write(result)
    # import pdb; pdb.set_trace()

def extract_ad_info(text):
    adid_match = re.search(r'AdID: (\d+)', text)
    # region_match = re.search(r'\(.*?(?:in\s+)?([A-Za-z\s]+)[,\)]', text)
    region_pattern = r'(?:in\s+([A-Za-z\s]+)\) | \(([A-Za-z\s]+),)'
    region_match = re.search(region_pattern, text, re.VERBOSE)
    
    if adid_match and region_match:
        ad_id = adid_match.group(1)
        region = region_match.group(1) or region_match.group(2)
        region = region.strip()  
        return ad_id, region
    elif not adid_match: 
        return None, None
    else:
        print(text)
        print(adid_match, region_match)
        ad_id = adid_match.group(1)
        return ad_id, None
    
def run_merge_training_summary_by_region(llm_client: LLMClient):
    if os.path.exists(f'{DATA_PATH['train_result_dir']}/summary_by_region_txt.json'):
        print('summary.txt already exists')
        return
    file_pattern = os.path.join(DATA_PATH['train_result_dir'], '*-*.txt')
    output_path = os.path.join(DATA_PATH['train_result_dir'], 'summary')
    feature_name = FEATURES
    # input_df = pd.read_csv(DATA_PATH['train_data_path'].split('.')[0]+'.csv')
    input_df = pd.read_csv(DATA_PATH['train_data_path'].split('_')[0]+'_'+DATA_PATH['train_data_path'].split('_')[1]+'_df_train.csv')
    id2region = input_df.set_index('adID')['region'].to_dict()
    
    
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        print(f"No files found matching the pattern '{file_pattern}'")
        return None
    
    summary_by_region={}
    for file_path in file_paths:
        print(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        content_piece= content.split('\n\n')
        for summary in content_piece:
            ad_id, region = extract_ad_info(summary)
            if ad_id == None:
                continue
            if int(ad_id) not in id2region:
                print(file_path, ad_id, region)
                continue
            region_check = id2region[int(ad_id)]
            # if region == None:
            #     pass
            # elif region == 'nan region':
            #     region = 'nan'
            # elif region == 'nan' and math.isnan(region_check):
            #     pass
            # elif region!=region_check:
            #     print(summary)
            #     raise ValueError(f"region unmatch, {region} != {region_check}")
            if region_check not in summary_by_region:
                summary_by_region[region_check] = []
            summary_by_region[region_check].append(summary)
    region_summary={}
    for region in summary_by_region:
        content_list = summary_by_region[region]
        summary_content = '\n'.join(content_list)
        prompt, reserve_prompt = build_summary_prompt(summary_content, limit_tokens=True)
        num_token = count_tokens(prompt)
        if num_token > MAX_INPUT_TOKENS[LLM_MODEL]:
            print(f"Input text length {num_token} exceed max limit {MAX_INPUT_TOKENS[LLM_MODEL]}, truncate files")
            # truncated_content_list = truncate_files(MAX_INPUT_TOKENS[LLM_MODEL], reserve_prompt, content_list)
            # summary_content = '\n'.join(truncated_content_list)
            # prompt, reserve_prompt = build_summary_prompt(summary_content, limit_tokens=True)
            import random
            sampled_content_list = random.sample(content_list, 100)
            summary_content = '\n'.join(sampled_content_list)
            prompt, reserve_prompt = build_summary_prompt(summary_content, limit_tokens=True)

        print(f'Try {LLM_MODEL}. Prompt: {prompt}')
        llm_client.timeout=50
        status, result = llm_client.chat(prompt)
        if status is not None and result is not None:
            print(f"state code: {status}")
            print(f"result: {result}")
            with open(f'{DATA_PATH['train_result_dir']}/summary_by_region.txt', 'w', encoding='utf-8') as file:
                file.write(result)
        region_summary[region] = result
    with open(f'{DATA_PATH['train_result_dir']}/summary_by_region_txt.json', 'w', encoding='utf-8') as file:
        json.dump(region_summary, file, ensure_ascii=False, indent=2)
    