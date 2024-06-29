from fastapi.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import io

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

# FastAPI 애플리케이션 객체 생성
app = FastAPI()

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='model\pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

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
    im_copy = np.copy(img)
    annotated_image = draw_landmarks_on_image(rgb_frame.numpy_view(), detection_result)
    res, im_png = cv2.imencode(".png", annotated_image)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


