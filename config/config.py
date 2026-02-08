# config/config.py
import os

LLM_MODEL = os.getenv('LLM_MODEL', 'gpt4o') # gpt4o, gemini
# API配置
API_CONFIG = {
    'gpt4o':
    { 
        'headers':{
            'Token': '',
            'Content-Type': 'application/json'
            },
        'url':'',
        'model':'gpt-4o-2024-11-20',
        'temperature': 0.1,
        'max_output_tokens': 2048
    },
    'gemini':
    {
       'headers':{
            'Content-Type': 'application/json'
            },
        'url':'',
        'model':'gemini-1.5-pro-002',
        'temperature': 0.1,
        'max_output_tokens': 2048
    },
     'gpt5':
    {
       'headers':{
            'Content-Type': 'application/json'
            },
        'url':'',
        'model':'gpt-5-2025-08-07',
        'temperature': 0.1,
        'reasoning_effort':"none"
    },
    'seed':
    {
       'headers':{
            'Content-Type': 'application/json',
            'Authorization': '' 
            },
        'url':'',
        'model':'ep-20260113101129-kgppr',
        'temperature': 0.1,
    },
    'max_retries':10,
    'timeout':30
}

MAX_INPUT_TOKENS = {
    'gpt4o': 127000,
    'gemini': 2097152
}

USE_IMAGE = os.getenv('USE_IMAGE', 0) 
ONLY_IMAGE = os.getenv('ONLY_IMAGE', 0) 
isScore = os.getenv('isScore', 0) 

PROMPT_SETTING = os.getenv('PROMPT_SETTING', 'scoring') # scoring, ranking, summary

DATA_ROOT = os.getenv('DATA_ROOT', '')
DATA_DATE_VERSION = os.getenv('DATA_DATE_VERSION','20250401-30')
DATA_SETTING = os.getenv('DATA_SETTING','v0911_31_all')

PRED_NUM = os.getenv('PRED_NUM', 10)


DATA_PATH={
    'label_path_dev': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING.split('_')[0]}_label_dev.json',
    'label_path_test': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING.split('_')[0]}_label_test.json',
    
    'train_data_path': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_input_train.txt',
    'dev_data_path': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING.split('_')[0]}_{DATA_SETTING.split('_')[1]}_input_dev.txt',
    'test_data_path': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING.split('_')[0]}_{DATA_SETTING.split('_')[1]}_input_test.txt',
    
    'train_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_train_{LLM_MODEL}_{PROMPT_SETTING}',
    'dev_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_dev_{LLM_MODEL}_{PROMPT_SETTING}_{PRED_NUM}_{isScore}',
    'test_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_test_{LLM_MODEL}_{PROMPT_SETTING}_{PRED_NUM}_{isScore}',

    'baseline_metrics_path': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_baseline.json',
    'llm_metrics_path_dev': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_{LLM_MODEL}_{PROMPT_SETTING}_{PRED_NUM}_{isScore}_dev.json',
    'llm_metrics_path_test': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_{LLM_MODEL}_{PROMPT_SETTING}_{PRED_NUM}_{isScore}_test.json',

    'image_mapping_path': f'{DATA_ROOT}/{DATA_DATE_VERSION}_adid2url_mapping.json',
}

if isScore==-1:
        DATA_PATH.update({
        'train_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_train_{LLM_MODEL}_{PROMPT_SETTING}',
        'dev_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_dev_{LLM_MODEL}_{PROMPT_SETTING}_{PRED_NUM}',
        'test_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_test_{LLM_MODEL}_{PROMPT_SETTING}_{PRED_NUM}',
        'llm_metrics_path_dev': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_{LLM_MODEL}_{PROMPT_SETTING}_{PRED_NUM}_dev.json',
        'llm_metrics_path_test': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_{LLM_MODEL}_{PROMPT_SETTING}_{PRED_NUM}_test.json',
        })

if USE_IMAGE:
    if isScore==-1:
        if ONLY_IMAGE:
            DATA_PATH.update({
                'train_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_train_{LLM_MODEL}_{PROMPT_SETTING}Image',
                'dev_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_dev_{LLM_MODEL}_{PROMPT_SETTING}ONLYImage_{PRED_NUM}',
                'test_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_test_{LLM_MODEL}_{PROMPT_SETTING}ONLYImage_{PRED_NUM}',
                'llm_metrics_path_dev': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_{LLM_MODEL}_{PROMPT_SETTING}ONLYImage_{PRED_NUM}_dev.json',
                'llm_metrics_path_test': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_{LLM_MODEL}_{PROMPT_SETTING}ONLYImage_{PRED_NUM}_test.json',
                'baseline_metrics_path': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_baselineImage.json',
            })
        else:
            DATA_PATH.update({
                'train_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_train_{LLM_MODEL}_{PROMPT_SETTING}Image',
                'dev_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_dev_{LLM_MODEL}_{PROMPT_SETTING}Image_{PRED_NUM}',
                'test_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_test_{LLM_MODEL}_{PROMPT_SETTING}Image_{PRED_NUM}',
                'llm_metrics_path_dev': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_{LLM_MODEL}_{PROMPT_SETTING}Image_{PRED_NUM}_dev.json',
                'llm_metrics_path_test': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_{LLM_MODEL}_{PROMPT_SETTING}Image_{PRED_NUM}_test.json',
                'baseline_metrics_path': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_baselineImage.json',
            })
    else:
        if ONLY_IMAGE:
            DATA_PATH.update({
                'train_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_train_{LLM_MODEL}_{PROMPT_SETTING}Image',
                'dev_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_dev_{LLM_MODEL}_{PROMPT_SETTING}ONLYImage_{PRED_NUM}_{isScore}',
                'test_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_test_{LLM_MODEL}_{PROMPT_SETTING}ONLYImage_{PRED_NUM}_{isScore}',
                'llm_metrics_path_dev': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_{LLM_MODEL}_{PROMPT_SETTING}ONLYImage_{PRED_NUM}_{isScore}_dev.json',
                'llm_metrics_path_test': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_{LLM_MODEL}_{PROMPT_SETTING}ONLYImage_{PRED_NUM}_{isScore}_test.json',
                'baseline_metrics_path': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_baselineImage.json',
            })
        else:
            DATA_PATH.update({
                'train_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_train_{LLM_MODEL}_{PROMPT_SETTING}Image',
                'dev_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_dev_{LLM_MODEL}_{PROMPT_SETTING}Image_{PRED_NUM}_{isScore}',
                'test_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_test_{LLM_MODEL}_{PROMPT_SETTING}Image_{PRED_NUM}_{isScore}',
                'llm_metrics_path_dev': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_{LLM_MODEL}_{PROMPT_SETTING}Image_{PRED_NUM}_{isScore}_dev.json',
                'llm_metrics_path_test': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_{LLM_MODEL}_{PROMPT_SETTING}Image_{PRED_NUM}_{isScore}_test.json',
                'baseline_metrics_path': f'result/{DATA_DATE_VERSION}_{DATA_SETTING}_eval_metrics_baselineImage.json',
            })
    if PROMPT_SETTING != 'summary':
         DATA_PATH.update({
            'train_result_dir': f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_train_{LLM_MODEL}_{PROMPT_SETTING}'})
    
os.makedirs(DATA_PATH['train_result_dir'], exist_ok=True)
os.makedirs(DATA_PATH['dev_result_dir'], exist_ok=True)
os.makedirs(DATA_PATH['test_result_dir'], exist_ok=True)

if DATA_DATE_VERSION =='aliAds':
    FEATURES = ['imageId', 'date', 'impression',
            'basicRMean','basicRStd', 'basicRVar', 'basicRMin', 'basicRMax', 'basicRMedian', 
            'basicGMean', 'basicGStd', 'basicGVar','basicGMin', 'basicGMax', 'basicGMedian', 
            'basicBMean','basicBStd', 'basicBVar', 'basicBMin', 'basicBMax','basicBMedian', 
            'hogFeatureDim', 'hogVisShape', 
            'glcmContrast', 'glcmCorrelation', 'glcmEnergy', 'glcmHomogeneity', 
            'shapeArea', 'shapePerimeter', 'shapeAspectRatio', 'shapeCircularity', 
            'cannyEdgePixels', 'cannyEdgeDensity', 'cannyTotalLength', 'cannyContourCount',
            'hsvHMean','hsvHStd', 'hsvSMean', 'hsvSStd', 'hsvVMean', 'hsvVStd'
            ]
elif DATA_DATE_VERSION =='ipinYou': 
    FEATURES =  ['TimeStamp', 'Region', 'City', 'AdExchange', 'DomainID',
                'AdSlotID', 'AdSlotWidth', 'AdSlotHeight', 'AdSlotVisibility',
                'AdSlotFormat', 'AdSlotFloorPrice', 'CreativeID', 'BidPrice',
                'PayPrice', 'LandingPageID', 'AdvertiserID', 'Impression'
                ]
elif DATA_SETTING =='CommAds':
    FEATURES =  []