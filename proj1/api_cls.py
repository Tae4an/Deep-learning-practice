from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
# STEP 1: 필요한 모듈 임포트
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# FastAPI 애플리케이션 객체 생성
app = FastAPI()

# STEP 2: ImageClassifier 객체 생성
# 모델 경로 설정하여 BaseOptions 객체 생성
base_options = python.BaseOptions(model_asset_path='model/efficientnet_lite0.tflite')
# ImageClassifierOptions 객체 생성
options = vision.ImageClassifierOptions(base_options=base_options, max_results=4)
# ImageClassifier 객체 생성
classifier = vision.ImageClassifier.create_from_options(options)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # STEP 3: 입력 이미지 로드
    # 클라이언트로부터 데이터 읽기
    contents = await file.read()
    # 문자열로부터 바이너리 변환
    nparr = np.fromstring(contents, np.uint8)
    # 바이너리 이미지 배열로부터 이미지 디코드
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # OpenCV 매트릭스로부터 mp 이미지 생성
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    
    # STEP 4: 입력 이미지 분류
    classification_result = classifier.classify(rgb_frame)
    
    # STEP 5: 분류 결과 처리
    # 최상위 카테고리 추출
    top_category = classification_result.classifications[0].categories[0]
    # 결과 문자열 생성
    result = f"{top_category.category_name} ({top_category.score:.2f})"
    return {"result": result}
