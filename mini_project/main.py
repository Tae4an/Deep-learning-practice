import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pytesseract
import cv2
import numpy as np
from ex_re import parse_nutrition_info
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

app = FastAPI()

# Miniforge 환경에서 설치된 Tesseract 실행 파일의 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\admin\miniforge3\envs\mini_project\Library\bin\tesseract.exe'

# TESSDATA_PREFIX 환경 변수 설정
os.environ['TESSDATA_PREFIX'] = r'C:\Users\admin\miniforge3\envs\mini_project\Library\share\tessdata'

templates = Jinja2Templates(directory="templates")

# MongoDB 연결 설정
MONGO_DETAILS = "mongodb+srv://minjuking:GAdeWGF35XjDLId8@cluster0.qvqzpec.mongodb.net/"
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client.nutrition_db
nutrition_collection = database.get_collection("nutrition_info")

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def extract_text_from_image(image):
    preprocessed_image = preprocess_image(image)
    text = pytesseract.image_to_string(preprocessed_image, lang='kor+eng')  # 한국어와 영어 모두 인식
    return text

@app.post("/uploadfile/")
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    text = extract_text_from_image(image)
    parsed_nutrition_info = parse_nutrition_info(text)
    
    # MongoDB에 저장
    nutrition_info = {
        "text": text,
        "parsed_nutrition_info": parsed_nutrition_info
    }
    result = await nutrition_collection.insert_one(nutrition_info)
    
    return templates.TemplateResponse("result.html", {"request": request, "text": text, "parsed_nutrition_info": parsed_nutrition_info})

@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
