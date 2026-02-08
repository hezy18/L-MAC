# utils/error_handling.py
import time
from typing import Callable, Any, Optional

def retry(max_retries: int = 3, loudly: bool = True):
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    if result is not None:
                        return result
                except Exception as e:
                    if loudly:
                        print(f"Error: {e}, retrying ({attempt+1}/{max_retries})")
                    time.sleep(1)
            
            if loudly:
                print(f"Waring: Reached maximum retry attempts {max_retries}, skipping operation.")
            return None
        
        return wrapper
    
    return decorator
