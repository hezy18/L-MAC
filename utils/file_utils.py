# utils/file_utils.py
import os
import json
import base64
import re

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def check_file_exists(file_path):
    return os.path.exists(file_path)

def create_directory(directory_path):
    os.makedirs(directory_path, exist_ok=True)

def extract_ad_images(mapping_path, text):
    with open (mapping_path, 'r') as f:
        adid2url = json.load(f)
    
    pattern = r'adID: (\d+)'
    ad_ids = re.findall(pattern, text)
    
    image_urls =  [adid2url[ad_id] for ad_id in ad_ids]
    return image_urls
