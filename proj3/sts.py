# STEP 1: 필요한 모듈 임포트
from sentence_transformers import SentenceTransformer

# STEP 2: 사전 학습된 Sentence Transformer 모델 로드
# "all-MiniLM-L6-v2"는 문장을 임베딩 벡터로 변환하는 데 사용되는 사전 학습된 모델
model = SentenceTransformer("all-MiniLM-L6-v2")

# STEP 3: 인코딩할 문장 리스트 정의
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# STEP 4: 문장 임베딩 계산
# 2. model.encode()를 호출하여 임베딩 벡터 계산
embeddings = model.encode(sentences)
print(embeddings.shape)  # 임베딩 벡터의 크기 출력
# [3, 384]

# STEP 5: 임베딩 벡터 추출 및 유사도 계산
# 첫 번째 문장의 임베딩 벡터 추출
emb1 = embeddings[0]
# 두 번째 문장의 임베딩 벡터 추출
emb2 = embeddings[1]
# 3. 임베딩 유사도 계산
similarities = model.similarity(emb1, emb2)
print(similarities)  # 유사도 행렬 출력
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])
