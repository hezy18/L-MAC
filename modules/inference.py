# modules/inference.py
import os
from tqdm import tqdm
from config.config import DATA_PATH, LLM_MODEL, USE_IMAGE, DATA_DATE_VERSION, ONLY_IMAGE
from utils.data_loader_writer import load_test_input, save_test_result
from llm.llm_client import LLMClient
from llm.prompt_builder import build_inference_prompt
from utils.file_utils import check_file_exists, extract_ad_images
import re
import pandas as pd
import json
import math

def load_test_input_by_region(phrase):
    # input_df = pd.read_csv(DATA_PATH[f'{phrase}_data_path'].split('.')[0]+'.csv')
    input_df = pd.read_csv(DATA_PATH[f'{phrase}_data_path'].split('_')[0]+'_'+DATA_PATH[f'{phrase}_data_path'].split('_')[1]+f'_df_{phrase}.csv')
    
    id2region = input_df.set_index('adID')['region'].to_dict()
    infers_by_region = {}
    file_path =  DATA_PATH[f'{phrase}_data_path']
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    for line in lines:
        match = re.search(r'adID:\s*(\d+)', line)
        if match:
            ad_id = int(match.group(1))
        else:
            raise ValueError(f"Can't extract adID in '{line}'")
        region = id2region[ad_id]
        if region not in infers_by_region:
            infers_by_region[region] = []
        infers_by_region[region].append(line)
    infers = {}
    for region in infers_by_region:
        last=0
        print(region, len(infers_by_region[region]))
        if len(infers_by_region[region])>8 and len(infers_by_region[region])%8>0 and len(infers_by_region[region])%8<3:
            tmp_length = 6
            if len(infers_by_region[region])%6>0 and len(infers_by_region[region])%6<3:
                tmp_length = 7
        else:
            tmp_length = 8
        print(tmp_length)
        if not isinstance(region, str) and math.isnan(region):
            key_region = 'NaN'
        else:
            key_region = region
        tmp_input = ''
        i=0
        for line in infers_by_region[region]:
            tmp_input = tmp_input + '\n{' + line + '}'
            i += 1
            if i % tmp_length == 0:
                infers[f'{key_region}_{i-tmp_length}-{i}'] = tmp_input
                last = i
                tmp_input = ''
        if last!=i:
            infers[f'{key_region}_{last}-{i}'] = tmp_input
    return infers    

def run_inference(llm_client: LLMClient, phrase='dev' ,setting: str='scoring', merge_type = 'all'): # setting = 'scoring' or 'ranking' or 'summary'
    if merge_type == 'all':
        if setting=='zeroShot':
            summary=''
            setting = 'ranking'
        elif setting=='preDefine': 
            with open(f'baselines/{DATA_DATE_VERSION}_LassoSelect.txt', 'r', encoding='utf-8') as file:
                summary = file.read()
            setting = 'ranking'
        elif setting=='preMRMR': 
            with open(f'baselines/{DATA_DATE_VERSION}_MRMRSelect.txt', 'r', encoding='utf-8') as file:
                summary = file.read()
            setting = 'ranking'
        elif setting=='preIG': 
            with open(f'baselines/{DATA_DATE_VERSION}_IGSelect.txt', 'r', encoding='utf-8') as file:
                summary = file.read()
            setting = 'ranking'
        elif setting=='llmselect': 
            with open(f'baselines/{DATA_DATE_VERSION}_llmSelect.txt', 'r', encoding='utf-8') as file:
                summary = file.read()
            setting = 'ranking'
        elif setting=='Random': 
            with open(f'baselines/{DATA_DATE_VERSION}_RandomSelect.txt', 'r', encoding='utf-8') as file:
                summary = file.read()
            setting = 'ranking'
        elif setting in ["preScoring","preRanking","preSummary"]:
            if LLM_MODEL=='gpt4o':
                pre_LLM = 'gemini'
            else:
                pre_LLM = 'gpt4o'
            if setting=='preScoring':
                setting  = 'scoring'
            elif setting=='preRanking':
                setting  = 'ranking'
            elif setting=='preSummary':
                setting  = 'summary'
            if USE_IMAGE and setting=='summary':
                summarydata_file = DATA_PATH['train_result_dir'].split('train_')[0]+'train_'+pre_LLM+'_'+setting+'Image'
            else:
                summarydata_file = DATA_PATH['train_result_dir'].split('train_')[0]+'train_'+pre_LLM+'_'+setting
            with open(f'{summarydata_file}/summary.txt', 'r', encoding='utf-8') as file:
                summary = file.read()
        else:
            with open(f'{DATA_PATH['train_result_dir']}/summary.txt', 'r', encoding='utf-8') as file:
                summary = file.read()
        testing_data = load_test_input(phrase=phrase)

        for key in tqdm(testing_data):
            print(key)
            
            if check_file_exists(os.path.join(DATA_PATH[f'{phrase}_result_dir'], f"{key}.txt")):
                continue

            image_urls=None
            if USE_IMAGE:
                image_urls = extract_ad_images(mapping_path=DATA_PATH['image_mapping_path'], text=testing_data[key]) 
            
            if ONLY_IMAGE:
                adID_list = [adLine.split(',')[0] for adLine in testing_data[key].split('\n{adID: ')[1:]]
                dataInputs = '\nadID:'.join(adID_list)
                dataInputs = 'adID:' + dataInputs
            else:
                dataInputs = testing_data[key]
            prompt = build_inference_prompt(dataInputs, summary, setting=setting)
            
            print(f'Try {LLM_MODEL}. Prompt: {prompt}')
            
             
            status, result = llm_client.chat(prompt, image_urls=image_urls)
            
            if status is not None and result is not None: 
                print(f"state code: {status}")
                print(f"result: {result}")
                
                save_test_result(key, result, phrase=phrase)

    elif merge_type == 'by_region':
        with open(f'{DATA_PATH['train_result_dir']}/summary_by_region_txt.json', 'r', encoding='utf-8') as file:
            summary_by_region = json.load(file)
        infers = load_test_input_by_region(phrase=phrase)
        for key in tqdm(infers):
            print(key)
            region = key.split('_')[0]

            if check_file_exists(os.path.join(DATA_PATH[f'{phrase}_result_dir'], f"{key}.txt")):
                continue

            summary = f'{summary_by_region[region]} in {region}.'
            
            prompt = build_inference_prompt(infers[key], summary, setting=setting)

            # import pdb; pdb.set_trace()
            print(f'Try {LLM_MODEL}. Prompt: {prompt}')

            image_urls=None
            if USE_IMAGE:
                image_urls = extract_ad_images(mapping_path=DATA_PATH['image_mapping_path'], text=infers[key])    
            status, result = llm_client.chat(prompt, image_urls=image_urls)
            
            if status is not None and result is not None:
                print(f"state code: {status}")
                print(f"result: {result}")
                
                save_test_result(key, result, phrase=phrase)
