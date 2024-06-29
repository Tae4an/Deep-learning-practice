# STEP 1
from transformers import pipeline

# STEP 2
classifier = pipeline('text-classification',"snunlp/KR-FinBert-SC", device = 0)

# STEP 3
text = "이준석 “참 나쁜 대통령”…한동훈 “민주당, 말 같지 않은 것도 정치공세” [금주의 말말말]"

# STEP 4
result = classifier(text)

# STEP 5
print(result)