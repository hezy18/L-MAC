# llm/token_counter.py
import tiktoken
from typing import Optional
from config.config import LLM_MODEL

def count_tokens(text: str) -> int:
    try:
        if LLM_MODEL.startswith("gpt-4") or LLM_MODEL.startswith("gpt4"):
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif LLM_MODEL.startswith("gpt-3.5"):
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        elif LLM_MODEL.startswith("gemini"):
            encoding = tiktoken.get_encoding("cl100k_base")
        elif LLM_MODEL.startswith("claude"):
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.encoding_for_model("gpt-4")
        
        return len(encoding.encode(text))
    except Exception as e:
        return '', len(text) // 4

def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    
    if len(text) < max_tokens * 4:
        return text
    
    if LLM_MODEL.startswith("gemini"):
        encoding = tiktoken.get_encoding("cl100k_base")
    else:
        encoding = tiktoken.encoding_for_model(
            "gpt-4" if LLM_MODEL == "gpt4o" else LLM_MODEL
        )
    
    encoded = encoding.encode(text)
    
    if len(encoded) > max_tokens:
        encoded = encoded[:max_tokens]
        try:
            return encoding.decode(encoded)
        except:
            return encoding.decode(encoded[:-1]) 
    
    return text