import pandas as pd
import numpy as np
import math
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from config.config import DATA_PATH, FEATURES, DATA_ROOT, DATA_SETTING, DATA_DATE_VERSION, USE_IMAGE
import os
import json
import random
RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
from baselines.LassoSelect import run_Lasso
from baselines.MRMRSelect import run_MRMR
from baselines.IGSelect import run_IG
from baselines.baselines_classfier_gridsearch import run_classifier as run_classifier_gridsearch
from baselines.baselines_regressor_gridsearch import run_regressor as run_regressor_gridsearch
from baselines.baseline_FinalMLP import run_FinalMLP_gridsearch
import sys

class MyBinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = pd.DataFrame(X, columns=self.columns)
        transformed_df = self._add_binary_onehot_columns(df, self.columns)
        
        new_columns = [col for col in transformed_df.columns 
                      if any(original_col in col for original_col in self.columns) and '_bit_' in col]
        
        return transformed_df[new_columns].values
    
    def _add_binary_onehot_columns(self, df: pd.DataFrame, id_cols: list) -> pd.DataFrame:
        max_id = max(df[col].max() for col in id_cols)
        n_bits = math.ceil(math.log2(int(max_id) + 1))
        
        bit_positions = np.arange(n_bits)[::-1]  # 例如: [n_bits-1, ..., 0]

        out = df.copy()
        for col in id_cols:
            arr = out[col].to_numpy().astype(int)
            # shape: (n_rows, n_bits)
            bits = ((arr[:, None] >> bit_positions) & 1)
            bit_col_names = [f"{col}_bit_{i}" for i in range(n_bits)]
            out[bit_col_names] = bits

        return out

def calculate_id_binary_columns(encoder, X_id_df, id_cols):
    max_id = max(X_id_df[col].max() for col in id_cols)
    n_bits = math.ceil(math.log2(int(max_id) + 1))
    id_feature_names = []
    for col in id_cols:
        bit_col_names = [f"{col}_bit_{i}" for i in range(n_bits)]
        id_feature_names.extend(bit_col_names)
    return id_feature_names

def ensure_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  
    elif isinstance(obj, (list, tuple)):
        return list(obj) 
    else:
        raise TypeError(f"unsupported type {type(obj)}, should be list/numpy/tuple")

def get_complete_column_mapping(preprocessor, X_train_df, id_cols, cat_cols, num_cols):
    id_pipeline = preprocessor.named_transformers_["id"]
    id_encoder = id_pipeline.named_steps["encoder"]
    X_id_df = X_train_df[id_cols].copy()
    if len(id_cols)>0:
        id_feature_names = calculate_id_binary_columns(id_encoder, X_id_df, id_cols)
        id_feature_names = ensure_list(id_feature_names)

        max_id = max(X_id_df[col].max() for col in id_cols)
        n_bits = math.ceil(math.log2(int(max_id) + 1))
        id_original_mapping = []
        for col in id_cols:
            id_original_mapping.extend([col] * n_bits)
        id_original_mapping = ensure_list(id_original_mapping)
    else:
        id_feature_names = []
        id_original_mapping = []

    cat_pipeline = preprocessor.named_transformers_["cat"]
    cat_encoder = cat_pipeline.named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
    cat_feature_names = ensure_list(cat_feature_names)

    cat_original_mapping = [name.split("_")[0] for name in cat_feature_names]
    cat_original_mapping = ensure_list(cat_original_mapping)

    num_pipeline = preprocessor.named_transformers_["num"]
    num_scaler = num_pipeline.named_steps["scaler"]
    num_feature_names = num_scaler.get_feature_names_out(num_cols)
    num_feature_names = ensure_list(num_feature_names)

    num_original_mapping = num_cols
    num_original_mapping = ensure_list(num_original_mapping)

    all_processed_names = id_feature_names + cat_feature_names + num_feature_names
    all_original_cols = id_original_mapping + cat_original_mapping + num_original_mapping
    all_modules = ["ID"]*len(id_feature_names) + \
                  ["Categorical"]*len(cat_feature_names) + \
                  ["numerical"]*len(num_feature_names)

    df_mapping = pd.DataFrame({
        "post_index": range(len(all_processed_names)),
        "post_colname": all_processed_names,
        "org_colname": all_original_cols,
        "data_type": all_modules,
        "post_cols": [all_original_cols.count(col) for col in all_original_cols]
    })

    df_summary = df_mapping[ ["org_colname", "data_type", "post_cols"]].drop_duplicates()
    df_summary = df_summary.sort_values(["data_type", "org_colname"]).reset_index(drop=True)
    
    return df_mapping, df_summary

def prepare_data(attr_setting:str='all'):
    if USE_IMAGE and DATA_DATE_VERSION not in ['aliAds','ipinYou']:
        train_df = pd.read_csv(f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING.split('_')[0]}_df_Image_train.csv') # {DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_input_train.txt
        dev_df = pd.read_csv(f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING.split('_')[0]}_df_Image_dev.csv')
        test_df = pd.read_csv(f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING.split('_')[0]}_df_Image_test.csv')
    else:
        train_df = pd.read_csv(f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING.split('_')[0]}_df_train.csv') # {DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING}_input_train.txt
        dev_df = pd.read_csv(f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING.split('_')[0]}_df_dev.csv')
        test_df = pd.read_csv(f'{DATA_ROOT}/{DATA_DATE_VERSION}_{DATA_SETTING.split('_')[0]}_df_test.csv')
    # dev_df = pd.read_csv(DATA_PATH['dev_data_path'].split('.txt')[0]+'.csv')
    
    choose_columns = FEATURES[1:]
    if USE_IMAGE and DATA_DATE_VERSION not in ['ipinYou']:
        embedding_columns = [f'emb{i+1}' for i in range(768)]
        choose_columns.extend(embedding_columns)
    

    gt_mapping = {
        1.0: 1, 2.0: 1, 3.0: 1,
        4.0: 2, 5.0: 2,
        6.0: 3, 7.0: 3,
        8.0: 4, 9.0: 4, 10.0: 4
    }
    train_df['ad_label'] = train_df['ad_fine_label'].map(gt_mapping)
    test_df['ad_label'] = test_df['ad_fine_label'].map(gt_mapping)
    dev_df['ad_label'] = dev_df['ad_fine_label'].map(gt_mapping)

    gt_mapping_select = {
            1.0: 0, 2.0: 0, 3.0: 0,
            4.0: 0, 5.0: 0,
            6.0: 0, 7.0: 0,
            8.0: 1, 9.0: 1, 10.0: 1
        }
    gt_mapping_filter = {
            1.0: 1, 2.0: 1, 3.0: 1,
            4.0: 0, 5.0: 0,
            6.0: 0, 7.0: 0,
            8.0: 0, 9.0: 0, 10.0: 0
        }
    
    train_df['ad_label_filter'] = train_df['ad_fine_label'].map(gt_mapping_filter)
    test_df['ad_label_filter'] = test_df['ad_fine_label'].map(gt_mapping_filter)
    dev_df['ad_label_filter'] = dev_df['ad_fine_label'].map(gt_mapping_filter)
    train_df['ad_label_select'] = train_df['ad_fine_label'].map(gt_mapping_select)
    test_df['ad_label_select'] = test_df['ad_fine_label'].map(gt_mapping_select)
    dev_df['ad_label_select'] = dev_df['ad_fine_label'].map(gt_mapping_select)

    X_train_df = train_df[choose_columns]
    y_train_df = train_df[['ad_label']]
    y_train_df_filter = train_df[['ad_label_filter']]
    y_train_df_select = train_df[['ad_label_select']]

    X_test_df = test_df[choose_columns]
    y_test_df = test_df[['ad_label']]
    y_test_df_filter = test_df[['ad_label_filter']]
    y_test_df_select = test_df[['ad_label_select']]

    X_dev_df = dev_df[choose_columns]
    y_dev_df = dev_df[['ad_label']]
    y_dev_df_filter = dev_df[['ad_label_filter']]
    y_dev_df_select = dev_df[['ad_label_select']]
    
    if DATA_DATE_VERSION =='aliAds':
        id_columns = ['goods_id', 'imageId']
        numerical_columns = ['date', 'impression', 'basicRMin', 'basicRMax', 'basicGMin', 'basicGMax', 'basicBMin', 'basicBMax',  
                             'hogFeatureDim', 'cannyEdgePixels', 'cannyContourCount', ]

        categorical_columns = ['basicRMean', 'basicRStd', 'basicRVar', 'basicRMedian', 'basicGMean','basicGStd', 'basicGVar', 'basicGMedian', 'basicBMean', 'basicBStd', 'basicBVar', 'basicBMedian',
                              'hogVisShape', 'glcmContrast', 'glcmCorrelation', 'glcmEnergy', 'glcmHomogeneity', 
                              'shapeArea', 'shapePerimeter', 'shapeAspectRatio', 'shapeCircularity', 'cannyEdgeDensity', 'cannyTotalLength', 
                              'hsvHMean', 'hsvHStd', 'hsvSMean', 'hsvSStd', 'hsvVMean', 'hsvVStd'
                              ]
    elif DATA_DATE_VERSION =='ipinYou':
        id_columns = ['BidID', 'DomainID', 'AdSlotID', 'AdvertiserID', 'CreativeID', 'LandingPageID']
        numerical_columns =  ['TimeStamp', 'AdSlotWidth', 'AdSlotHeight', 'Impression']
        categorical_columns = ['Region', 'City', 'AdExchange', 'AdSlotVisibility',
                'AdSlotFormat', #'AdSlotFloorPrice', 'BidPrice','PayPrice'
                ]
    else:
        id_columns = ['contentID', 'campaignID', 'adsetID', 'accountID','appID']
        
        categorical_columns = ['region', 'creators', 'creativeType', 'company', 'contentType', 'sourceProduction', 'verticalTags', 'stickerTags', 'dailyBudget','UGCCountry']
        
        numerical_columns = ['campaignCreatedTime', 'adsetCreatedTime', 'contentCreatedTime', 'UGCCreatedTime', 'imageWidth','imageHeight','whRatio','imageSize','videoSize','UGCDuration']
    if USE_IMAGE and DATA_DATE_VERSION not in ['ipinYou']:
        numerical_columns.extend(embedding_columns)


    id_columns = [col for col in id_columns if col in choose_columns]
    categorical_columns = [col for col in categorical_columns if col in choose_columns]
    numerical_columns = [col for col in numerical_columns if col in choose_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('id', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)), 
                ('encoder', MyBinaryEncoder(columns=id_columns))
            ]), id_columns),
            
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_columns),
            
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  
                ('scaler', StandardScaler())
            ]), numerical_columns)
        ])
    
    preprocessor.fit(X_train_df)  
    X_train_processed = preprocessor.transform(X_train_df) 
    X_test_processed = preprocessor.transform(X_test_df)
    X_dev_processed = preprocessor.transform(X_dev_df) 
    print(f"data shpe: train={X_train_processed.shape}, valid={X_dev_processed.shape}, test={X_test_processed.shape}")

    if attr_setting in ['Lasso', 'MRMR', 'IG']:
        df_mapping, df_summary = get_complete_column_mapping(
            preprocessor=preprocessor,
            X_train_df=X_train_df,
            id_cols=id_columns,
            cat_cols=categorical_columns,
            num_cols=numerical_columns
        )
        print(df_mapping)

    if attr_setting=='Lasso':
        top_pos = run_Lasso(X_train_processed, y_train_df) 
        print(top_pos)
        import pdb; pdb.set_trace()
        X_train_top = X_train_processed[:,top_pos]
        X_test_top = X_test_processed[:,top_pos]
        X_dev_top = X_dev_processed[:,top_pos]
    elif attr_setting=='MRMR':
        top_pos = run_MRMR(X_train_processed, y_train_df)
        print(top_pos)
        import pdb; pdb.set_trace()
        X_train_top = X_train_processed[:,top_pos]
        X_test_top = X_test_processed[:,top_pos]
        X_dev_top = X_dev_processed[:,top_pos]
    elif attr_setting=='IG':
        top_pos = run_IG(X_train_processed, y_train_df)
        print(top_pos)
        import pdb; pdb.set_trace()
        X_train_top = X_train_processed[:,top_pos]
        X_test_top = X_test_processed[:,top_pos]
        X_dev_top = X_dev_processed[:,top_pos]
    elif attr_setting =='Random':
        X_train_top = []
        X_test_top = []
        X_dev_top = []
        for i in range(5):
            top_pos = np.random.RandomState(RANDOM_SEED + i).choice(X_train_processed.shape[1], 10, replace=False)
            X_train_top.append(X_train_processed[:,top_pos])
            X_test_top.append(X_test_processed[:,top_pos])
            X_dev_top.append(X_dev_processed[:,top_pos])
    else:
        X_train_top = X_train_processed
        X_test_top = X_test_processed
        X_dev_top = X_dev_processed
    return X_train_top, X_test_top, X_dev_top, y_train_df, y_test_df, y_dev_df, y_train_df_filter, y_test_df_filter, y_dev_df_filter, y_train_df_select, y_test_df_select, y_dev_df_select

def run_attr_random(X_train_top, X_test_top, X_dev_top, y_train_df, y_test_df, y_dev_df,
                     y_train_df_filter, y_test_df_filter, y_dev_df_filter, y_train_df_select, y_test_df_select, y_dev_df_select,
                    model_names:str='all',  attr_setting:str='all'):
    all_results = {'select':{}, 'filter':{}, 'reg':{}}
    for i in range(len(X_train_top)):
        results = run_classifier_gridsearch(X_train_top[i], X_test_top[i], X_dev_top[i], y_train_df_select, y_test_df_select, y_dev_df_select,
                                model_names=model_names, label_type='select', attr_setting=attr_setting, save=0)
        for model in results:
            if model not in all_results['select']:
                all_results['select'][model] = []
            all_results['select'][model].append(results[model])
    for i in range(len(X_train_top)):
        results = run_classifier_gridsearch(X_train_top[i], X_test_top[i], X_dev_top[i], y_train_df_filter, y_test_df_filter, y_dev_df_filter,
                                    model_names=model_names, label_type='filter', attr_setting=attr_setting, save=0)
        for model in results:
            if model not in all_results['filter']:
                all_results['filter'][model] = []
            all_results['filter'][model].append(results[model])
    for i in range(len(X_train_top)):
        results = run_regressor_gridsearch(X_train_top[i], X_test_top[i], X_dev_top[i], y_train_df, y_test_df, y_dev_df,
                                model_names=model_names, attr_setting=attr_setting, save=0)
        for model in results:
            if model not in all_results['reg']:
                all_results['reg'][model] = []
            all_results['reg'][model].append(results[model])
    avg_results = {}
    for task_type in all_results:
        avg_results[task_type] = {}
        for model in all_results[task_type]:
            avg_results[task_type][model] = {}
            all_metrics = all_results[task_type][model]
            metrics = all_metrics[0].keys() if all_metrics else []
            for metric in metrics:
                metric_values = [all_metrics[i][metric] for i in range(len(all_metrics))]
                avg_results[task_type][model][metric] = np.mean(metric_values)
    # import pdb; pdb.set_trace()
    return avg_results
            
def run_attr_random_FinalMLP(X_train_top, X_test_top, X_dev_top, y_train_df, y_test_df, y_dev_df,
                     y_train_df_filter, y_test_df_filter, y_dev_df_filter, y_train_df_select, y_test_df_select, y_dev_df_select,
                    model_names:str='all',  attr_setting:str='all'):
    all_results = {'select':{}, 'filter':{}, 'reg':{}}
    for i in range(len(X_train_top)):
        results = run_FinalMLP_gridsearch(X_train_top[i], X_test_top[i], X_dev_top[i], y_train_df_select, y_test_df_select, y_dev_df_select,
                                model_names=model_names, label_type='select', attr_setting=attr_setting)
    for i in range(len(X_train_top)):
        results = run_FinalMLP_gridsearch(X_train_top[i], X_test_top[i], X_dev_top[i], y_train_df_filter, y_test_df_filter, y_dev_df_filter,
                                    model_names=model_names, label_type='filter', attr_setting=attr_setting)
        for metrics in results:
            if metrics not in all_results['filter']:
                all_results['filter'][metrics] = []
            all_results['filter'][metrics].append(results[metrics])
    for i in range(len(X_train_top)):
        results = run_FinalMLP_gridsearch(X_train_top[i], X_test_top[i], X_dev_top[i], y_train_df, y_test_df, y_dev_df,
                                model_names=model_names, attr_setting=attr_setting)
        for metrics in results:
            if metrics not in all_results['reg']:
                all_results['reg'][metrics] = []
            all_results['reg'][metrics].append(results[metrics])
    avg_results = {}
    for task_type in all_results:
        avg_results[task_type] = {}
        for metrics in all_results[task_type]:
            avg_results[task_type][metrics] = np.mean(all_results[task_type][metrics])
    return avg_results

def run_baseline(model_names:str='all',  attr_setting:str='all'):

    X_train_top, X_test_top, X_dev_top, y_train_df, y_test_df, y_dev_df, y_train_df_filter, y_test_df_filter, y_dev_df_filter, y_train_df_select, y_test_df_select, y_dev_df_select  = prepare_data(attr_setting)
    if attr_setting=='Random':
        if model_names=='FinalMLP':
            avg_results = run_attr_random_FinalMLP(X_train_top, X_test_top, X_dev_top, y_train_df, y_test_df, y_dev_df, y_train_df_filter, y_test_df_filter, y_dev_df_filter, y_train_df_select, y_test_df_select, y_dev_df_select)
            output_path = DATA_PATH['baseline_metrics_path']
            with open(f'{output_path.split('.')[0]}_{attr_setting}_FinalMLP_all.{output_path.split('.')[1]}', 'w') as f:
                json.dump(avg_results['reg'], f, indent=4)
            with open( f'{output_path.split('.')[0]}_{attr_setting}_FinalMLP_filter.{output_path.split('.')[1]}', 'w') as f:
                json.dump(avg_results['filter'], f, indent=4)
            
        else:
            avg_results = run_attr_random(X_train_top, X_test_top, X_dev_top, y_train_df, y_test_df, y_dev_df, y_train_df_filter, y_test_df_filter, y_dev_df_filter, y_train_df_select, y_test_df_select, y_dev_df_select)
            output_path = DATA_PATH['baseline_metrics_path']
            
            with open(f'{output_path.split('.')[0]}_{attr_setting}.{output_path.split('.')[1]}', 'w') as f:
                json.dump(avg_results['reg'], f, indent=4)
            with open( f'{output_path.split('.')[0]}_select_{attr_setting}.{output_path.split('.')[1]}', 'w') as f:
                json.dump(avg_results['select'], f, indent=4)
            with open( f'{output_path.split('.')[0]}_filter_{attr_setting}.{output_path.split('.')[1]}', 'w') as f:
                json.dump(avg_results['filter'], f, indent=4)
            
    else:

        if model_names=='FinalMLP':
            run_FinalMLP_gridsearch(X_train_top, X_test_top, X_dev_top, y_train_df_select, y_test_df_select, y_dev_df_select,
                                  model_names=model_names, label_type='select', attr_setting=attr_setting)
        else:
        
            run_classifier_gridsearch(X_train_top, X_test_top, X_dev_top, y_train_df_select, y_test_df_select, y_dev_df_select,
                                    model_names=model_names, label_type='select', attr_setting=attr_setting)
            run_classifier_gridsearch(X_train_top, X_test_top, X_dev_top, y_train_df_filter, y_test_df_filter, y_dev_df_filter,
                                    model_names=model_names, label_type='filter', attr_setting=attr_setting)
            run_regressor_gridsearch(X_train_top, X_test_top, X_dev_top, y_train_df, y_test_df, y_dev_df,
                                    model_names=model_names, attr_setting=attr_setting)

