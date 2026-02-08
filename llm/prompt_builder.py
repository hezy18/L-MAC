# llm/prompt_builder.py
from typing import Dict, List, Any
from config.config import FEATURES, PRED_NUM, USE_IMAGE, USE_FEWSHOT, DATA_PATH, isScore, ONLY_IMAGE
import json

def generate_percentile_description(level, pred_num):
    total_levels = pred_num
    mid_point = (total_levels + 1) // 2
    
    if level <= mid_point:
        percentile = level * (100 / total_levels)
        if percentile.is_integer():
            percent_str = f"{int(percentile)}%"
        else:
            percent_str = f"{level}/{total_levels}"
        if percent_str=="2/6" or percent_str=="3/9":
            percent_str="33%"
        return f"{percent_str}"
    else:
        percentile = (total_levels - level + 1) * (100 / total_levels)
        if percentile.is_integer():
            percent_str = f"{int(percentile)}%"
        else:
            percent_str = f"{total_levels - level + 1}/{total_levels}"
        if percent_str=="2/6" or percent_str=="3/9":
            percent_str="33%"
        return f"{percent_str}"

def generate_level_descrption(pred_num:int):
    # , where level 1-5 are Bottom 5%, 10%, 16.7% (1/6), 33.3% (1/3), 50% (1/2) click volume, level 6-10 are Top 50% (1/2), 33.3% (1/3), 16.7% (1/6), 10%, 5% click volume, level 1 is worst performance (bottom 5%), level 10 is best performance (top 5%)"
    level_description = f'level 1-{pred_num//2} are Bottom'
    for i in range(1, pred_num//2+1):
        if i == 1:
            level_description += f' {generate_percentile_description(i, pred_num)}'
        else:
            level_description += f', {generate_percentile_description(i-1, pred_num)}-{generate_percentile_description(i, pred_num)}'
        
    level_description += f' click volume, level {pred_num//2+1}-{pred_num} are Top'
    for i in range(pred_num//2+1, pred_num+1):
        if i == pred_num//2+1:
            level_description += f' {generate_percentile_description(i+1, pred_num)}-{generate_percentile_description(i, pred_num)} '
        elif i == pred_num:
            level_description += f', {generate_percentile_description(i, pred_num)}'
        else:
            level_description += f', {generate_percentile_description(i+1, pred_num)}-{generate_percentile_description(i, pred_num)}'
    level_description += f' click volume, level 1 is worst performance (bottom {generate_percentile_description(1, pred_num)}), level {pred_num} is best performance (top {generate_percentile_description(pred_num, pred_num)})'
    return level_description

def construct_instruction_prompt(feature_list: str, feature_num: int, summary: str = '', setting: str = 'scoring',phrase: str = 'train'):
    if phrase == 'train':
        TRAINING_INSTRUCTION_BODY = f"You are an expert in analyzing advertisement performance. For each input ad, analyze how its {feature_num} features (listed below in order) contribute to its Daily New Users (DNU) percentile ranking (e.g., 'top X%' or 'bottom Y%-Z%') in the specified region. Feature list: {feature_list}. "
        TRAINING_INSTRUCTION_COMMAND= {
            'ranking':
            """Rank the features by their impact on DNU from most important to least important. The features ranking list is [`most important feature`, `second most important`, ..., `least important feature`]. Generate a JSON dict with {`ad_id`: Input ad ID (e.g., 3001),`features ranking list`:[`most_important_feature`, `second_most_important`, ..., `least_important_feature`]. """
            ,
                'scoring':
            """Provide a list of integer scores ranging from 1 to 100, one score per feature, ordered according to the feature list above. Generate a JSON dict with {`ad_id`: Input ad ID (e.g., 3001),`importance_scores`: A list of each feature's score (integer from 1 to 100), ordered as the feature list above.  """
            ,
                'summary':
            """Generate a brief summary identifying the most important and least important features that affect performance. """
        }
        TRAINING_INSTRUCTION_FINAL = "Following are the inputs.\n"

        if USE_IMAGE:
            TRAINING_INSTRUCTION_BODY = f"You are an expert in analyzing advertisement performance. For each input ad and the corresponding image(in the given order), analyze how its appearance and {feature_num} features (listed below in order) contribute to its Daily New Users (DNU) percentile ranking (e.g., 'top X%' or 'bottom Y%-Z%') in the specified region. Feature list: {feature_list}. Please summarize your analysis in two parts: (1) Feature Importance."
            TRAINING_INSTRUCTION_FINAL = """(2) Visual Characteristics of High-Performing Ads: Describe the common visual traits of ads that perform better. 
            Return a list of dicts: [\{"AdID": xx, "feature important": "...", "visual analysis": "..."\}, ...]
            Following are the inputs.\n"""
        return TRAINING_INSTRUCTION_BODY + TRAINING_INSTRUCTION_COMMAND[setting] + TRAINING_INSTRUCTION_FINAL
    elif phrase == 'test':

        INFER_INSTRUCTION_BODY1 = 'You are an expert in predicting advertisement performance.'
        level_description = generate_level_descrption(int(PRED_NUM))
        INFER_INSTRUCTION_BODY2 = f"You will be given an ad instance represented by a list of {feature_num} feature values corresponding to the above dimensions. Features are listed in order: {feature_list}. Based on the feature values and the feature importance, your task is to classify the ad's popularity into {PRED_NUM} levels (level 1 to {PRED_NUM}), where {level_description} , and explain your reasoning. You should reference the most influential features (those with high importance scores and impactful values) in your explanation."
                              
        INFER_INSTRUCTION_BODY3 = f"Ensure the classification distribution across levels 1-{PRED_NUM} aligns with their corresponding click volume percentiles to avoid excessive concentration in middle levels. "
        
        INFER_INSTRUCTION_BODY4 = """Only generate a JSON dict with {"ad_id": <ad_id>, "Popularity level":  Level 1-%d (1: worst, %d:best), "Reason": "<Brief explanation based on most important features>"}. """ % (int(PRED_NUM), int(PRED_NUM))

        INFER_INSTRUCTION_COMMAND = {
            'scoring': f'The following is a dictionary of average feature importance scores derived from training data: {summary}\\n'
            ,
            'ranking':f'The following is ranking of feature importances from the most important to the list: {summary}\n'
            ,
            'summary': f'The following is the summary of feature importance derived from training data: {summary}\n'
        }
        if USE_IMAGE:
            INFER_INSTRUCTION_COMMAND.update({
                'summary': f'The following is the summary of ad appearance analysis and feature importance derived from training data: {summary}\\n'
            })
            if isScore:
                INFER_INSTRUCTION_BODY2 = f"You will be given ad instances one by one, each paired with its corresponding image and a list of {feature_num} feature values. These features correspond to the dimensions in the order listed: {feature_list}. Based on the feature values and the feature importance, your task is to score the ad's popularity from 1 to {PRED_NUM}, and explain your reasoning. You should reference the most influential features (those with high importance scores and impactful values) in your explanation."
                INFER_INSTRUCTION_BODY3 = f"Ensure the classification distribution across 1-{PRED_NUM} (integer) aligns with their corresponding click volume percentiles to avoid excessive concentration in middle levels. "
                if ONLY_IMAGE:
                    INFER_INSTRUCTION_BODY2 = f"You will be given ad instances one by one with adID and its corresponding image. Based on the feature values and the feature importance, your task is to score the ad's popularity from 1 to {PRED_NUM}, and explain your reasoning. You should reference the most influential features (those with high importance scores and impactful values) in your explanation."
            else:
                INFER_INSTRUCTION_BODY2 = f"You will be given ad instances one by one, each paired with its corresponding image and a list of {feature_num} feature values. These features correspond to the dimensions in the order listed: {feature_list}. Based on the feature values and the feature importance, your task is to classify the ad's popularity into {PRED_NUM} levels (level 1 to {PRED_NUM}), where {level_description} , and explain your reasoning. You should reference the most influential features (those with high importance scores and impactful values) in your explanation."
                if ONLY_IMAGE:
                    INFER_INSTRUCTION_BODY2 = f"You will be given ad instances one by one with adID and its corresponding image. Based on the feature values and the feature importance, your task is to classify the ad's popularity into {PRED_NUM} levels (level 1 to {PRED_NUM}), where {level_description} , and explain your reasoning. You should reference the most influential features (those with high importance scores and impactful values) in your explanation."
        else:
            if isScore:        
                INFER_INSTRUCTION_BODY2 = f"You will be given an ad instance represented by a list of {feature_num} feature values corresponding to the above dimensions. Features are listed in order: {feature_list}. Based on the feature values and the feature importance, your task is to score the ad's popularity from 1 to {PRED_NUM}, and explain your reasoning. You should reference the most influential features (those with high importance scores and impactful values) in your explanation."     
                INFER_INSTRUCTION_BODY3 = f"Ensure the classification distribution across 1-{PRED_NUM} (integer) aligns with their corresponding click volume percentiles to avoid excessive concentration in middle levels. "
        if summary=='':
            return INFER_INSTRUCTION_BODY1+INFER_INSTRUCTION_BODY2+INFER_INSTRUCTION_BODY3+INFER_INSTRUCTION_BODY4 + f'Following are the inputs.\n'
        else:
            return INFER_INSTRUCTION_BODY1+INFER_INSTRUCTION_COMMAND[setting]+INFER_INSTRUCTION_BODY2+INFER_INSTRUCTION_BODY3+INFER_INSTRUCTION_BODY4 + f'Following are the inputs.\n'
    
### ------------FEATURES--------------
feature_list =', '.join(FEATURES)
feature_num = len(FEATURES)

# SUMMARY_INSTRUCTION = "As an expert in ad performance analysis, I need generalized insights into feature importance for ad performance, derived from the collective patterns across all the cases in the following content. Note: Focus on overarching trends and shared characteristics across all cases, rather than summarizing individual cases. Please summarize your analysis in two parts: (1) Feature Importance Ranking and Impact Direction: List the most significant features in order of importance, and briefly note whether each feature has a positive or negative impact on performance. (2) Visual Characteristics of High-Performing Ads: Describe the common visual traits of ads that perform better. \n"
SUMMARY_INSTRUCTION = "You are a seasoned expert specializing in advertisement performance analysis, with deep expertise in identifying core influencing factors of ad performance, extracting universal patterns from multi-case data, and delivering structured, actionable insights. Please summarize your analysis in two parts: (1) Feature Importance Ranking and Impact Direction: List the most significant features in order of importance, and briefly note whether each feature has a positive or negative impact on performance. (2) Visual Characteristics of High-Performing Ads: Describe the common visual traits of ads that perform better. The input cases are as follows: \n"
SUMMARY_INSTRUCTION = "You are a seasoned expert specializing in advertisement performance analysis, with deep expertise in identifying core influencing factors of ad performance, extracting universal patterns from multi-case data, and delivering structured, actionable insights. Please summarize your analysis for Feature Importance Ranking and Impact Direction: List the most significant features in order of importance, and briefly note whether each feature has a positive or negative impact on performance. The input cases are as follows: \n"

def build_training_prompt(input_data: str, setting: str='scoring') -> str:
    return construct_instruction_prompt(feature_list, feature_num, setting=setting, phrase='train') + input_data

def build_inference_prompt(input_data: str, summary, setting: str='scoring') -> str:
    return construct_instruction_prompt(feature_list, feature_num, summary = summary, setting=setting, phrase='test') + input_data

def build_summary_prompt(summary_content: str, limit_tokens: bool=True) -> str:
    if limit_tokens:
        return SUMMARY_INSTRUCTION + summary_content, SUMMARY_INSTRUCTION
    else:
        return SUMMARY_INSTRUCTION + summary_content

