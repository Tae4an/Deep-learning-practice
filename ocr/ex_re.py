import re
from difflib import SequenceMatcher

# 기존에 저장된 영양 성분 이름
stored_nutrition_names = [
    "칼로리", "나트륨", "탄수화물", "당류", "지방", "트랜스지방", 
    "포화지방", "콜레스테롤", "단백질"
]

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def correct_text(extracted_text):
    corrected_info = {}
    for line in extracted_text.split('\n'):
        best_match = None
        highest_similarity = 0
        for name in stored_nutrition_names:
            similarity = similar(line, name)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = name
        if highest_similarity > 0.6:  # 유사도 임계값 설정
            corrected_info[best_match] = line
    return corrected_info

def normalize_text(text):
    text = text.replace("′", "").replace("′", "").replace("?", "").replace(",", "").replace("_", " ")
    text = re.sub(r'\s+', ' ', text)  # 다중 공백을 단일 공백으로 치환
    return text

def parse_nutrition_info(text):
    normalized_text = normalize_text(text)
    corrected_text = correct_text(normalized_text)
    nutrition_info = {}
    patterns = {
        '칼로리': r'칼로리\s*:\s*(\d+\.?\d*)\s*kcal',
        '나트륨': r'나트륨\s*(\d+\.?\d*)\s*mg\s*\((\d+\.?\d*)%\)',
        '탄수화물': r'탄수화물\s*(\d+\.?\d*)\s*g\s*\((\d+\.?\d*)%\)',
        '당류': r'당류\s*(\d+\.?\d*)\s*g',
        '지방': r'지방\s*(\d+\.?\d*)\s*g\s*\((\d+\.?\d*)%\)',
        '트랜스지방': r'트랜스지방\s*(\d+\.?\d*)\s*g',
        '포화지방': r'포화지방\s*(\d+\.?\d*)\s*g\s*\((\d+\.?\d*)%\)',
        '콜레스테롤': r'콜레스테롤\s*(\d+\.?\d*)\s*mg\s*\((\d+\.?\d*)%\)',
        '단백질': r'단백질\s*(\d+\.?\d*)\s*g\s*\((\d+\.?\d*)%\)'
    }
    for key, line in corrected_text.items():
        if key in patterns:
            match = re.search(patterns[key], line)
            if match:
                nutrition_info[key] = match.groups()
    return nutrition_info
