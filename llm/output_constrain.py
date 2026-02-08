import json
from config.config import LLM_MODEL

def check_data_format(result, format = 'json', ):
    if LLM_MODEL =='gpt4o':
        try:
            content = result[8:-3]
            parts = content.strip().split("\n\n")
            ads_part = parts[0]
            try:
                ads_data = json.loads(ads_part)
                return True
            except json.JSONDecodeError as e:
                error_pos = e.pos
                truncated_part = ads_part[:error_pos]
                end_bracket = truncated_part.rfind(']')
                if end_bracket != -1:
                    ads_part_refined = truncated_part[:end_bracket + 1]
                else:
                    ads_part_refined = truncated_part
                try:
                    ads_data = json.loads(ads_part_refined)
                    return True
                except json.JSONDecodeError:
                    return False
        except IndexError:
            print("Error: unsupported input type")
            return False
        except json.JSONDecodeError as e:
            print(f"Json decoder error: {e}")
            return False
    
    elif LLM_MODEL =='gemini':
        try:
            content = result[8:-3]  
            parts = content.strip().split("\n\n")
            ads_part = parts[0]
            ads_part = ads_part.strip()
            if '```' == ads_part[-3:]:
                ads_part = ads_part[:-3]
            try:
                ads_data = json.loads(ads_part)
            except json.JSONDecodeError as e:
                ads_part_list = ','.join(ads_part.split("\n"))
                ads_part_list  = '['+ads_part_list+']'
                try:
                    ads_data = json.loads(ads_part_list)
                except json.JSONDecodeError as e:
                    print(f"Json decode error: {e}")
                    return False
            return True
        except IndexError:
            print("Error: unsupported input type")
            return False
        except json.JSONDecodeError as e:
            print(f"Json decode error: {e}")
            return False
    
def load_formatted_data(result, format = 'json'):
    if LLM_MODEL =='gpt4o':
        try:
            content = result[8:-3] 
            parts = content.strip().split("\n\n")
            ads_part = parts[0]
            try:
                return json.loads(ads_part)
            except json.JSONDecodeError as e:
                error_pos = e.pos
                truncated_part = ads_part[:error_pos]
                end_bracket = truncated_part.rfind(']')
                if end_bracket != -1:
                    ads_part_refined = truncated_part[:end_bracket + 1]
                else:
                    ads_part_refined = truncated_part
                try:
                    return json.loads(ads_part_refined)
                except json.JSONDecodeError:
                    print("try ads_part, but still json decode error")
        except IndexError:
            print("Error: unsupported input type")
        except json.JSONDecodeError as e:
            print(f"Json decode error: {e}")
    
    elif LLM_MODEL =='gemini':
        try:
            content = result[8:-3]  
            parts = content.strip().split("\n\n")
            ads_part = parts[0]
            ads_part = ads_part.strip()
            if '\n```' == ads_part[-4:]:
                ads_part = ads_part[:-4]
            try:
                ads_data = json.loads(ads_part)
                return ads_data
            except json.JSONDecodeError as e:
                ads_part_list = ','.join(ads_part.split("\n"))
                ads_part_list  = '['+ads_part_list+']'
                try:
                    ads_data = json.loads(ads_part_list)
                    return ads_data
                except json.JSONDecodeError as e:
                    print(f"Json decode error: {e}")
            
        except IndexError:
            print("Error: unsupported input type")
        except json.JSONDecodeError as e:
            print(f"Json decode error: {e}")
