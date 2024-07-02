# scripts/recommend_songs.py
from extract_vggish_features import extract_vggish_features
from sklearn.metrics.pairwise import cosine_similarity
import os

# 예시로 사용할 음악 데이터베이스
music_database = {
    "song1": "../data/song1.wav",
    "song2": "../data/song2.wav",
    "song3": "../data/song3.wav"
}

# 데이터베이스의 모든 음악 특징 추출
db_features = {song: extract_vggish_features(path) for song, path in music_database.items()}

def recommend_similar_songs(audio_path, db_features):
    # 입력된 음악의 특징 추출
    query_features = extract_vggish_features(audio_path)

    # 유사도 계산
    similarities = {song: cosine_similarity(query_features, features).mean() for song, features in db_features.items()}
    
    # 유사도가 높은 순으로 정렬
    recommended_songs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    
    return recommended_songs
