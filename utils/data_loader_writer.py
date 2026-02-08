# data/data_loader_writer.py
from typing import Dict, List, Tuple
import os
from config.config import DATA_PATH, PROMPT_SETTING
from utils.file_utils import read_file, read_json, write_file, write_json
import json
import pandas as pd

def load_ground_truth(phrase: str = 'dev'):
    return read_json(DATA_PATH[f'label_path_{phrase}'])

def save_train_result(key: str, result: str):
    file_path = os.path.join(DATA_PATH['train_result_dir'], f"{key}.txt")
    write_file(file_path, result)

def save_test_result(key: str, result: str, phrase: str = 'dev'):
    file_path = os.path.join(DATA_PATH[f'{phrase}_result_dir'], f"{key}.txt")
    write_file(file_path, result)

def save_summary_result(result: str, filename: str = "summary.txt"):
    file_path = os.path.join(DATA_PATH['train_result_dir'], filename)
    write_file(file_path, result)

def load_train_input(file_path: str = DATA_PATH['train_data_path']) -> Dict[str, str]:
    if PROMPT_SETTING=='ranking':
        divisor = 5
    else:
        divisor = 6
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    inputs = {}
    i = 0
    start = i
    tmp_input = ''
    
    for line in lines:
        tmp_input = tmp_input + '\n{' + line + '}'
        i += 1
        
        if i % divisor == 0:
            inputs[f'{start}-{i}'] = tmp_input
            tmp_input = ''
            start = i
    if i!=start:
        inputs[f'{start}-{i}'] = tmp_input
    
    return inputs

def load_test_input(phrase: str = 'dev') -> Dict[str, str]:
    divisor = 6
    
    file_path =  DATA_PATH[f'{phrase}_data_path']
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    infers = {}
    i = 0
    start = i
    tmp_input = ''
    
    for line in lines:
        tmp_input = tmp_input + '\n{' + line + '}'
        i += 1
        
        if i % divisor == 0:
            infers[f'{i-6}-{i}'] = tmp_input
            start = i
            tmp_input = ''
    if i!=start:
        infers[f'{start}-{i}'] = tmp_input
    
    return infers

def parse_analysis_data(input_data: str) -> List[Dict]: # TODO?
    ads_data = []
    lines = input_data.strip().split('\n')
    
    for line in lines:
        if line.strip():
            try:
                ad_data = eval(line)
                ads_data.append(ad_data)
            except Exception as e:
                print(f"Parse error: {e}")
    
    return ads_data

import numpy as np

def save_multiple_matrices_to_csv(matrices, names, output_path):
    with open(output_path, 'w') as file:
        for i, (matrix, name) in enumerate(zip(matrices, names)):
            file.write(f"Matrix: {name}\n")
            rows, cols = matrix.shape
            file.write(f"Shape: {rows}x{cols}\n")
            
            np.savetxt(file, matrix, delimiter=',', fmt='%d')
            
            if i < len(matrices) - 1:
                file.write("\n")
    

def read_multiple_matrices_from_csv(file_path):
    matrices = {}
    current_matrix = None
    data = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            if line.startswith("Matrix: "):
                if current_matrix is not None and data:
                    matrices[current_matrix] = np.array(data)
                    data = []
                current_matrix = line.split(": ")[1]
                continue
            
            if line.startswith("Shape: ") or not line:
                continue
            
            row = [int(x) for x in line.split(',')]
            data.append(row)
    
    if current_matrix is not None and data:
        matrices[current_matrix] = np.array(data)
    
    return matrices

# reading example
# loaded_matrices = read_multiple_matrices_from_csv("multiple_matrices.csv")
# for name, matrix in loaded_matrices.items():
#     print(f"\n {name} (shape: {matrix.shape}):")
#     print(matrix)