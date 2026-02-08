# llm/llm_client.py
# llm/llm_client.py
import requests
import json
import os
import base64
from typing import Union, Optional, Tuple, Dict, Any, List
from config.config import LLM_MODEL, API_CONFIG, FEATURES
from utils.file_utils import encode_image
from utils.error_handling import retry

class LLMClient:
    def __init__(self, model_type: str = None):
        self.model_type = model_type or LLM_MODEL
        self._setup_client()
    
    def _setup_client(self):
        if self.model_type.lower() == 'gpt4o':
            config = API_CONFIG['gpt4o']
            self.headers = config['headers']
            self.url = config['url']
            self.model = config['model']
            self.temperature = config.get('temperature', 0.1)
            self.max_output_tokens = config.get('max_output_tokens', 2048)
            self._send_request = self._send_gpt4o_request
        elif self.model_type.lower() == 'gemini':
            config = API_CONFIG['gemini']
            self.headers = config['headers']
            self.url = config['url']
            self.model = config['model']
            self.temperature = config.get('temperature', 0.7)
            self.max_output_tokens = config.get('max_output_tokens', 1000)
            self._send_request = self._send_gemini_request
        else:
            raise ValueError(f"un supported model: {self.model_type}")
        
        self.timeout = API_CONFIG.get('timeout', 20)
    
    @retry(max_retries=API_CONFIG.get('max_retries', 10))
    def chat(self, prompt: str, image_urls: list = None) -> Tuple[Optional[int], Optional[str]]:
        temperature = self.temperature
        max_tokens = self.max_output_tokens
        
        return self._send_request(prompt, temperature, max_tokens, image_urls)
    
    def _send_gpt4o_request(self, prompt: str, temperature: float, max_tokens: int, image_urls: list = None) -> Tuple[Optional[int], Optional[str]]:
        prompt_wrap = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ]
                }
            ],
            "model": self.model,
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }
        if image_urls:
            for image_url in image_urls:
                if isinstance(image_url, str):
                    if not image_url.startswith("http"):
                        assert os.path.isfile(image_url)
                        base64_image = encode_image(image_url)
                        image_url = f"data:image/jpeg;base64,{base64_image}"
                elif isinstance(image_url, bytes):    
                    image_url = f"data:image/jpeg;base64,{base64.b64encode(image_url).decode('utf-8')}"

                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
                prompt_wrap["messages"][0]["content"].append(image_content)

        data = {
            "prompt": json.dumps(prompt_wrap)
        }

        response = requests.post(
            self.url,
            headers=self.headers,
            json=data,
            timeout=self.timeout
        )
        status = response.status_code
        # import pdb;pdb.set_trace()
        result = response.json()["data"]["resp"]
        
        if status == 200:
            return status, result
        else:
            print(f"fail, state code: {status}, error message: {response.text}")
            return status, None

  
    def _send_gemini_request(self, prompt: str, temperature: float, max_tokens: int, image_urls: list = None) -> Tuple[Optional[int], Optional[str]]:
        messages = [{"role": "user", "content": prompt}]
        data = {
            "stream": False,
            "model":self.model,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        if image_urls:
            for image_url in image_urls:
                if isinstance(image_url, str):
                    if not image_url.startswith("http"):
                        assert os.path.isfile(image_url)
                        base64_image = encode_image(image_url)
                        image_url = f"data:image/jpeg;base64,{base64_image}"
                elif isinstance(image_url, bytes):    
                    image_url = f"data:image/jpeg;base64,{base64.b64encode(image_url).decode('utf-8')}"

                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
                data["messages"][0]["content"].append(image_content)

        response = requests.post(
            self.url,
            headers=self.headers,
            data=json.dumps(data),
            timeout=self.timeout
        )
        
        status = response.status_code
        if status == 200:
            result = response.json()["choices"][0]["message"]["content"]
            return status, result
        else:
            print(f"fail, state code: {status}, error message: {response.text}")
            return status, None
    
    def __str__(self):
        return f"LLMClient(model_type={self.model_type}, model={self.model})"
    
if __name__ == '__main__':
    client = LLMClient()

    prompt = '''
        Using data collected at a advertising platform, we wish to build a machine learning model that can accurately predict whether the ad can have high DNU or low DNU. The dataset contains of total of 17 features. Prior to training the model, we first want to identify a subset that are most important for reliable prediction of the target variable.
        Given a list of features, rank them according to their importances in predicting whether an individual can have high DNU. The ranking should be in descending order, starting with the most important feature. 
        Your response should be a numbered list with each item on a new line. For example: 1. foo 2. bar 3.baz 
        Only output the ranking. Do not output dialogue or explanations for the ranking. Do not exclude any features in the ranking.
        Rank all 17 features in the following list: 
        “['TimeStamp', 'Region', 'City', 'AdExchange', 'DomainID', 'AdSlotID', 'AdSlotWidth', 'AdSlotHeight', 'AdSlotVisibility', 'AdSlotFormat', 'AdSlotFloorPrice', 'CreativeID', 'BidPrice', 'PayPrice', 'LandingPageID', 'AdvertiserID', 'Impression']”
        '''
    
    prompt = '''
        Using data collected at a advertising platform, we wish to build a machine learning model that can accurately predict whether the ad can have high DNU or low DNU. The dataset contains of total of 31 features. Prior to training the model, we first want to identify a subset that are most important for reliable prediction of the target variable.
        Given a list of features, rank them according to their importances in predicting whether an individual can have high DNU. The ranking should be in descending order, starting with the most important feature. 
        Your response should be a numbered list with each item on a new line. For example: 1. foo 2. bar 3.baz 
        Only output the ranking. Do not output dialogue or explanations for the ranking. Do not exclude any features in the ranking.
        Rank all 31 features in the following list: 
        “[]”
        '''
    
    prompt = '''
        Using data collected at a advertising platform, we wish to build a machine learning model that can accurately predict whether the ad can have high DNU or low DNU. The dataset contains of total of 42 features. Prior to training the model, we first want to identify a subset that are most important for reliable prediction of the target variable.
        Given a list of features, rank them according to their importances in predicting whether an individual can have high DNU. The ranking should be in descending order, starting with the most important feature. 
        Your response should be a numbered list with each item on a new line. For example: 1. foo 2. bar 3.baz 
        Only output the ranking. Do not output dialogue or explanations for the ranking. Do not exclude any features in the ranking.
        Rank all 42 features in the following list: 
        “['imageId', 'date', 'impression', 'basicRMean','basicRStd', 'basicRVar', 'basicRMin', 'basicRMax', 'basicRMedian', 'basicGMean', 'basicGStd', 'basicGVar','basicGMin', 'basicGMax', 'basicGMedian', 'basicBMean','basicBStd', 'basicBVar', 'basicBMin', 'basicBMax','basicBMedian', 'hogFeatureDim', 'hogVisShape', 'glcmContrast', 'glcmCorrelation', 'glcmEnergy', 'glcmHomogeneity', 'shapeArea', 'shapePerimeter', 'shapeAspectRatio', 'shapeCircularity', 'cannyEdgePixels', 'cannyEdgeDensity', 'cannyTotalLength', 'cannyContourCount','hsvHMean','hsvHStd', 'hsvSMean', 'hsvSStd', 'hsvVMean', 'hsvVStd']"
        '''
    
    status, result = client.chat(prompt)

    if status == 200:
        print("result:", result)
    else:
        print("fail")
