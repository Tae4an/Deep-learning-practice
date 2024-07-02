# scripts/transcribe_audio.py
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import librosa

# 음성 인식 모델과 토크나이저 로드
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

def transcribe_audio(audio_path):
    # 오디오 파일 로드
    speech, rate = librosa.load(audio_path, sr=16000)

    # 입력을 토크나이즈
    input_values = tokenizer(speech, return_tensors="pt", padding="longest").input_values

    # 모델을 통해 예측
    with torch.no_grad():
        logits = model(input_values).logits

    # 결과를 디코딩
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]

    return transcription
