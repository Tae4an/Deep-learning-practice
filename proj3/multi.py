# STEP 1: 필요한 라이브러리 임포트
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice
import torch

# STEP 2: 사전 학습된 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_swag_model")
model = AutoModelForMultipleChoice.from_pretrained("stevhliu/my_awesome_swag_model")

# STEP 3: 프롬프트와 후보 문장 설정
prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
candidate1 = "The law does not apply to croissants and brioche."
candidate2 = "The law applies to baguettes."

# STEP 4: 입력을 토큰화하고 모델에 입력
# 입력을 토큰화하여 필요한 텐서 형식으로 변환하고 패딩을 추가.
inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
# 정답 라벨을 텐서로 정의하고 차원을 맞추기 위해 unsqueeze로 차원을 추가.
labels = torch.tensor(0).unsqueeze(0)
# 모델에 입력을 전달하고 출력을 얻음. 각 입력 텐서의 차원을 맞추기 위해 unsqueeze를 사용.
outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
# 모델의 로짓(logits)을 얻음.
logits = outputs.logits
# 로짓 중 가장 높은 값을 가지는 인덱스를 예측된 클래스(predicted_class)로 설정.
predicted_class = logits.argmax().item()


# STEP 5: 예측된 클래스를 출력
# 예측된 클래스 (0이면 candidate1, 1이면 candidate2)를 출력합니다.
print(predicted_class)
