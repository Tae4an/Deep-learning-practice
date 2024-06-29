from fastapi import FastAPI, Form
# STEP 1: 필요한 모듈 임포트
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

app = FastAPI()

# STEP 2: 사전 학습된 토크나이저와 모델 로드
# "stevhliu/my_awesome_model"은 사전 학습된 감정 분석 모델로, 문장의 감정을 분류하는 모델
tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
model.to("cuda:0")  # 모델을 GPU로 이동

@app.post("/login/")
async def predict(text: str = Form()):
    # STEP 4: 입력 텍스트를 예측하는 함수
    # 4-1 전처리: 텍스트를 텐서로 변환
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    # 4-2 추론
    with torch.no_grad():  # 추론 시에는 기울기 계산을 비활성화
        logits = model(**inputs).logits
    # 4-3 후처리: 가장 높은 점수의 클래스를 예측
    predicted_class_id = logits.argmax().item()
    result = model.config.id2label[predicted_class_id]

    # STEP 5: 예측 결과 반환
    return {"result": result}
