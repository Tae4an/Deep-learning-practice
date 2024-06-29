from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import io

# 시각화에 필요한 설정
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def visualize(image, detection_result) -> np.ndarray:
    """입력 이미지에 경계 상자를 그리고 반환합니다.
    Args:
      image: 입력 RGB 이미지.
      detection_result: 시각화할 모든 "Detection" 엔티티의 리스트.
    Returns:
      경계 상자가 있는 이미지.
    """
    for detection in detection_result.detections:
        # 경계 상자 그리기
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # 레이블과 점수 그리기
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image

# FastAPI 애플리케이션 객체 생성
app = FastAPI()

# STEP 2: ObjectDetector 객체 생성
base_options = python.BaseOptions(model_asset_path='model/efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # STEP 3: 입력 이미지 로드
    # 클라이언트로부터 데이터 읽기
    contents = await file.read()
    # 문자열로부터 바이너리 변환
    nparr = np.frombuffer(contents, np.uint8)
    # 바이너리 이미지 배열로부터 이미지 디코드
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # OpenCV 매트릭스로부터 mp 이미지 생성
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    
    # STEP 4: 입력 이미지에서 객체 탐지
    detection_result = detector.detect(rgb_frame)
    
    # # STEP 5: 탐지 결과 처리
    # # 탐지 결과 시각화
    # image_copy = np.copy(img)
    # annotated_image = visualize(image_copy, detection_result)
        
    # # 이미지 반환을 위해 BGR에서 RGB로 변환
    # _, img_encoded = cv2.imencode('.png', annotated_image)
    # return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")
    
    # STEP 5
    # object_list = []
    # for det in detection_result.detections:
    #    object_list.append(det.categories[0].category_name)
       
    # return {"objects":object_list}

    # STEP 5: box가 그려진 이미지를 반환
    # https://stackoverflow.com/a/59618249
    image_copy = np.copy(img)
    annotated_image = visualize(image_copy, detection_result)
    res, im_png = cv2.imencode(".png", annotated_image)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
