# modules/rule_generation.py
import os
from tqdm import tqdm
from config.config import DATA_PATH, LLM_MODEL, USE_IMAGE
from utils.data_loader_writer import load_train_input, save_train_result
from llm.llm_client import LLMClient
from llm.prompt_builder import build_training_prompt
from llm.output_constrain import check_data_format
from utils.file_utils import check_file_exists, extract_ad_images

def run_rule_generation(llm_client: LLMClient, setting: str='scoring'): # setting = 'scoring' or 'ranking' or 'summary'
    training_data = load_train_input()
    
    error_keys = []
    for key in tqdm(training_data):
        print(key)
        
        if check_file_exists(os.path.join(DATA_PATH['train_result_dir'], f"{key}.txt")):
            continue
        
        prompt = build_training_prompt(training_data[key], setting=setting)
        
        print(f'Try {LLM_MODEL}. Prompt: {prompt}')
        
        retries = 0
        success = False

        image_urls=None
        if USE_IMAGE:
            image_urls = extract_ad_images(mapping_path=DATA_PATH['image_mapping_path'], text=training_data[key])    

        while retries < 3 and not success:
            status, result = llm_client.chat(prompt, image_urls=image_urls)
            if setting=='summary':
                success = True
            else:
                success = check_data_format(result)
            retries += 1
        if not success:
            error_keys.append(key)

        if status is not None and result is not None:
            print(f"state code: {status}")
            print(f"result: {result}")

            save_train_result(key, result)

        print(f'Error keys: {error_keys}')