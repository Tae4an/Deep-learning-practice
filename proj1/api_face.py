from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
# STEP 1: 필요한 모듈 임포트
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision
from fastapi.responses import StreamingResponse
from typing import Tuple, Union
import io
import math


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                    width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                    MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image


# FastAPI 애플리케이션 객체 생성
app = FastAPI()

# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='model\blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

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
    
    # STEP 4: Detect faces in the input image.
    detection_result = detector.detect(rgb_frame)
    print(detection_result)
    
    # STEP 5: Process the detection result. In this case, visualize it.
    image_copy = np.copy(img)
    annotated_image = visualize(image_copy, detection_result)
    res, im_png = cv2.imencode(".png", annotated_image)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
