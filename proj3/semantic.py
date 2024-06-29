"""
이것은 문장 임베딩을 위한 간단한 애플리케이션입니다: 시맨틱 검색

우리는 다양한 문장이 포함된 코퍼스를 가지고 있습니다. 주어진 쿼리 문장에 대해
코퍼스에서 가장 유사한 문장을 찾고자 합니다.

이 스크립트는 다양한 쿼리에 대해 코퍼스 내에서 가장 유사한 문장 상위 5개를 출력합니다.
"""
# STEP 1
import torch
from sentence_transformers import SentenceTransformer

# STEP 2
embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")

# STEP 3
corpus = [
    "별로야"
    # "남자가 음식을 먹고 있다.",
    # "남자가 빵 조각을 먹고 있다.",
    # "여자아이가 아기를 안고 있다.",
    # "남자가 말을 타고 있다.",
    # "여자가 바이올린을 연주하고 있다.",
    # "두 남자가 숲 속에서 카트를 밀고 있다.",
    # "남자가 흰 말을 타고 울타리 안에 있다.",
    # "원숭이가 드럼을 연주하고 있다.",
    # "치타가 먹이를 쫓고 있다.",
]
# STEP 4
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# 쿼리 문장들:
queries = [
    "맛없어"
    # "남자가 파스타를 먹고 있다.",
    # "고릴라 복장을 한 사람이 드럼을 연주하고 있다.",
    # "치타가 들판에서 먹이를 쫓고 있다.",
]

# 각 쿼리 문장에 대해 코퍼스에서 코사인 유사도에 기반하여 가장 유사한 상위 5개 문장을 찾습니다.
top_k = min(5, len(corpus))  # 코퍼스의 길이와 5 중에서 더 작은 값을 선택합니다.
for query in queries:
    # 쿼리 문장을 인코딩하여 임베딩을 생성합니다.
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # 코사인 유사도와 torch.topk를 사용하여 가장 높은 5개의 점수를 찾습니다.
    similarity_scores = torch.nn.functional.cosine_similarity(query_embedding, corpus_embeddings)
    scores, indices = torch.topk(similarity_scores, k=top_k)

    print("\nQuery:", query)
    print("Top 5 most similar sentences in corpus:")

    # 상위 5개 문장과 그 유사도 점수를 출력합니다.
    for score, idx in zip(scores, indices):
        print(corpus[idx], "(Score: {:.4f})".format(score))

    """
    # 또는, util.semantic_search를 사용하여 코사인 유사도 + topk를 수행할 수 있습니다.
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]  # 첫 번째 쿼리에 대한 결과를 가져옵니다.
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """
