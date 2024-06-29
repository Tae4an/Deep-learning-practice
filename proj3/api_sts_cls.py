from fastapi import FastAPI, Form

# STEP 1: 필요한 모듈 임포트
from sentence_transformers import SentenceTransformer
import torch

# STEP 2: 사전 학습된 Sentence Transformer 모델 로드
# "paraphrase-multilingual-MiniLM-L12-v2"는 다국어 문장 임베딩을 위한 사전 학습된 모델로, 문장의 의미를 벡터로 변환하여 유사도 계산에 사용
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

app = FastAPI()

# 저장된 문장과 임베딩 벡터를 저장할 리스트
contents = []
contents_emb = []

@app.post("/add/")
async def add(text: str = Form()):
    # STEP 3: 입력 문장을 받아 처리

    # STEP 4: 문장 임베딩 계산
    embedding = model.encode(text)

    # STEP 5: 문장과 임베딩 벡터를 리스트에 추가
    contents.append(text)
    contents_emb.append(embedding)

    # 현재 저장된 문장과 임베딩 벡터의 개수 출력
    print(len(contents), len(contents_emb))
    return {"result": "OK"}

@app.post("/search/")
async def search(query: str = Form()):
    # STEP 3: 검색할 쿼리 문장을 받아 처리

    # STEP 4: 쿼리 문장의 임베딩 벡터 계산
    embedding = model.encode(query)

    # STEP 5: 저장된 문장들 중에서 쿼리 문장과 가장 유사한 문장 찾기
    # 코사인 유사도를 사용하여 유사도 계산
    sims = torch.nn.functional.cosine_similarity(torch.tensor(contents_emb), torch.tensor([embedding]), dim=1)
    result_index = torch.argmax(sims).item()

    # 유사도 값과 결과 출력
    print(sims)
    print(result_index)

    return {"result": contents[result_index]}
