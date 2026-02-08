# main.py
import os
import argparse

def set_env_variable(key, value):
    os.environ[key] = value
    print("Set environment variable:", key, "=", value)
def main():
    parser = argparse.ArgumentParser(description="LLM-ads")
    parser.add_argument("--model", type=str, choices=["gpt4o", "gemini", "gpt5", "seed"], default="gemini")
    parser.add_argument("--setting", type=str, choices=["scoring", "ranking", "summary", "preDefine", "zeroShot","preMRMR","preIG","llmselect","preScoring","preRanking","preSummary","Random", "combine"], default="summary")

    parser.add_argument("--attr_setting", type=str, choices=["gpt4o-ranking", "gpt4o-scoring", "gpt4o-summary", "gemini-ranking", "gemini-scoring", "gemini-summary", "llmselect", "Lasso", "IG", "MRMR", "all", "Random"], default="all")
    # parser.add_argument("--task", type=str, choices=["reg4", "clfSelect", "clfFilter"], default="reg4")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--data_date_version", type=str, default="20250401-30")
    parser.add_argument("--data_setting", type=str, default="v0911_31_all") # v1030_131_all
    parser.add_argument("--run_baselines", type=str, default="")
    parser.add_argument("--stage", type=str, default="", help="combination of rule,merging,inference,evaluation,baseline,comparasion else all stages")
    parser.add_argument("--merge_type", type=str, choices=["all", "by_region"], default="all")
    parser.add_argument("--merge_sampling_rate", type=float, default=1.0)
    parser.add_argument("--pred_num", type=int, default=10)
    parser.add_argument("--use_image", type=int, default=0)
    parser.add_argument("--only_image", type=int, default=0)
    parser.add_argument("--use_fewshot", type=int, default=0)
    parser.add_argument("--llm_temperature", type=float, default=0.1)
    # parser.add_argument("--isUniform", type=int, default=0)

    parser.add_argument("--mapping_focus_cate_weight", type=str, default="{}")
    parser.add_argument("--mapping_IsErrorLoss", type=int, default=1)

    parser.add_argument("--isScore", type=int, default=0)
    
    args = parser.parse_args()
    
    if args.setting:
        set_env_variable("PROMPT_SETTING", args.setting)
    if args.model:
        set_env_variable("LLM_MODEL", args.model)
    if args.data_root:
        set_env_variable("DATA_ROOT", args.data_root)
    if args.data_date_version:
        set_env_variable("DATA_DATE_VERSION", args.data_date_version)
    if args.data_setting:
        set_env_variable("DATA_SETTING", args.data_setting)
    if args.pred_num:
        set_env_variable("PRED_NUM", str(args.pred_num))
    if args.use_image:
        set_env_variable("USE_IMAGE", str(args.use_image))
    if args.only_image:
        set_env_variable("ONLY_IMAGE", str(args.only_image))
    if args.use_fewshot:
        set_env_variable("USE_FEWSHOT", str(args.use_fewshot))
    # if args.isUniform:
    #     set_env_variable("isUniform", str(args.isUniform))
    if args.isScore:
        set_env_variable("isScore", str(args.isScore))
    if args.llm_temperature!=-1:
        set_env_variable("LLM_TEMPERATURE", str(args.llm_temperature))
    import ast
    mapping_focus_cate_weight = ast.literal_eval(args.mapping_focus_cate_weight)
        
    if not isinstance(mapping_focus_cate_weight, dict):
        raise ValueError("Parameter must be a dictionary format") 
    for key, value in mapping_focus_cate_weight.items():
        if not isinstance(key, int) or key < 1 or key > 4:
            raise ValueError(f"Dict key must be an integer between 1 and 4, found: {key}")
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Dict key must be an integer between 1 and 4, found: {key}")

    from modules.rule_generation import run_rule_generation
    from modules.merging import run_merge_training_result
    from modules.merging_by_region import run_merge_training_result_by_region
    from modules.summary import run_merge_training_summary,run_merge_training_summary_by_region
    from modules.inference import run_inference
    from modules.evaluation import run_evaluation
    from baselines.baseline import run_baseline as run_baselines_main
    from llm.llm_client import LLMClient 
    llm_client = LLMClient()
    
    from config.config import PROMPT_SETTING, DATA_PATH, ONLY_IMAGE, USE_IMAGE
    print('PROMPT_SETTING:', PROMPT_SETTING)
    print('DATA_PATH', DATA_PATH)

    stages_to_run = [s.strip() for s in args.stage.split(',')] if args.stage else []
    valid_stages = {'rule', 'merging', 'inference', 'evaluation', 'baseline', 'comparasion'}
    run_all = not stages_to_run or not set(stages_to_run).issubset(valid_stages)

    if args.merge_type=='by_region' and PROMPT_SETTING not in ["scoring", "ranking", "summary"]:
        print(f'Error: PROMPT_SETTING is {PROMPT_SETTING} and merge_type is by_region')
        exit()
    if run_all or 'rule' in stages_to_run:
        if PROMPT_SETTING not in ["scoring", "ranking", "summary"]:
            print(" === skip rule generation stage ===")
        else:
            print("=== start rule generation stage ===")
            run_rule_generation(llm_client, PROMPT_SETTING)

    if run_all or 'merging' in stages_to_run:
        if PROMPT_SETTING not in ["scoring", "ranking", "summary"]:
            print("=== skip rule integration stage ===")
        elif ONLY_IMAGE:
            print("=== skip rule integration stage ===")
            if USE_IMAGE==0:
                raise ValueError("USE_IMAGE must be 1 when ONLY_IMAGE is True")
        else:
            print("=== start rule integration stage ===")
            if PROMPT_SETTING=='summary':
                if args.merge_type == 'all':
                    run_merge_training_summary(llm_client)
                elif args.merge_type == 'by_region':
                    run_merge_training_summary_by_region(llm_client)
            else:
                if args.merge_type == 'all':
                    run_merge_training_result(setting = PROMPT_SETTING)
                elif args.merge_type == 'by_region':
                    run_merge_training_result_by_region(setting = PROMPT_SETTING)

    if run_all or 'inference' in stages_to_run:
        print("\n=== start inference stage ===")
        run_inference(llm_client, phrase='dev', setting = PROMPT_SETTING, merge_type = args.merge_type)
        run_inference(llm_client, phrase='test', setting = PROMPT_SETTING, merge_type = args.merge_type)
    
    if run_all or 'evaluation' in stages_to_run:
        print("\n=== start evaluation stage ===")
        run_evaluation(phrase='dev', merge_type = args.merge_type, mapping_focus_cate_weight = mapping_focus_cate_weight, mapping_IsErrorLoss = args.mapping_IsErrorLoss)
        run_evaluation(phrase='test', merge_type = args.merge_type, mapping_focus_cate_weight = mapping_focus_cate_weight, mapping_IsErrorLoss = args.mapping_IsErrorLoss)
    if run_all or 'baseline' in stages_to_run:
        if len(args.run_baselines)>0:
            
            print("\n=== start baseline model ===")
            run_baselines_main(args.run_baselines, attr_setting=args.attr_setting)
            
    if run_all or 'comparasion' in stages_to_run:
        print("\n=== all results ===")
        from utils.file_utils import read_json
        import glob
        import pandas as pd
        import re
        
        all_data = [] 
        for model in ['gpt4o','gemini']:
            for setting in ['scoring', 'scoringImage', 'scoringONLYImage', 'ranking', 'rankingImage', 'rankingONLYImage', 'summary', 'summaryImage', 'summaryONLYImage',
                            'preDefine', 'preDefineImage','zeroShot','zeroShotImage','zeroShotONLYImage', "preMRMR", "preMRMRImage", "preIG", "preIGImage", 
                            "llmselect", "llmselectImage", "Random", "RandomImage",
                            "preScoring", "preScoringImage", "preRanking", "preRankingImage", "preSummary", "preSummaryImage"]:
                for pred_num in [4, 5, 6, 7, 8, 9, 10, 100]:
                    for isScore in [0, 1, -1]:
                        for merge_type in ['all', 'by_region']:
                            for phrase in ['dev', 'test']:
                                if isScore==-1:
                                    pattern = f'result/{args.data_date_version}_{args.data_setting}_eval_metrics_{model}_{setting}_{pred_num}_{phrase}_*{merge_type}*.json'
                                else:
                                    pattern = f'result/{args.data_date_version}_{args.data_setting}_eval_metrics_{model}_{setting}_{pred_num}_{isScore}_{phrase}_*{merge_type}*.json'

                                matched_files = glob.glob(pattern)
                                file_list = matched_files
                                for path in file_list:
                                    print(path)
                                    if re.match(pattern, path):
                                        optimization = path.split('_')[-1].split('.')[0]
                                    else:
                                        optimization = '\{\}1'
                                    results = read_json(path)
                                    for map_key in results:
                                        result = results[map_key]
                                        prefix_data = {
                                            'model_name': f'{model}_{setting}_{pred_num}_{isScore}_{merge_type}',
                                            'phrase': phrase,
                                            'mapping': map_key,
                                            'optimization': optimization
                                        }
                                    
                                        ordered_result = {**prefix_data, **result}
                                        all_data.append(ordered_result)

        result_data_path = '_'.join(DATA_PATH['baseline_metrics_path'].split('_')[:-1])+'_all.csv'
        baseline_path = DATA_PATH['baseline_metrics_path']
        if 'Image' in baseline_path:
            baseline_path_list = [baseline_path, baseline_path.replace('baselineImage', 'baseline'), 
                                  baseline_path.replace('baselineImage', 'baseline_select'), baseline_path.replace('baselineImage', 'baseline_filter'),
                                  baseline_path.replace('baselineImage', 'baselineImage_select'), baseline_path.replace('baselineImage', 'baselineImage_filter')]
        else:
            baseline_path_list = [baseline_path, baseline_path.replace('baseline', 'baselineImage'),
                                  baseline_path.replace('baseline', 'baselineImage_select'), baseline_path.replace('baseline', 'baselineImage_filter'),
                                  baseline_path.replace('baseline', 'baseline_select'), baseline_path.replace('baseline', 'baseline_filter')]
        for baseline_path in baseline_path_list:
            if os.path.exists(baseline_path):
                results = read_json(baseline_path)
                if 'Image' in baseline_path:
                    image_str = 'Image'
                else:
                    image_str = ''
                if 'select' in baseline_path:
                    label_type = 'select'
                elif 'filter' in baseline_path:
                    label_type = 'filter'
                else:
                    label_type = 'reg'
                for model in results:
                    result = results[model]
                    prefix_data = {
                                'model_name': model + image_str,
                                'phrase': 'test',
                                'mapping': '',
                                'optimization': label_type
                            }
                    ordered_result = {**prefix_data, **result}
                    all_data.append(ordered_result)

        df = pd.DataFrame(all_data)
        df.to_csv(result_data_path, index=False)

        new_df = df[(df['phrase'] == 'test') & (df['mapping'].isin(['clffilter_mapping', 'clfselect_mapping', 'strict_mapping', 'gt_mapping']))]
        baseline_df= df[(df['phrase'] == 'test') & (df['mapping'] == '')]
        new_df = pd.concat([new_df, baseline_df])
        # import pdb; pdb.set_trace()
        new_df.to_csv(f'{args.data_date_version}_{args.data_setting}_result.csv', index=False)


if __name__ == "__main__":
    main()
