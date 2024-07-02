# main.py
from fastapi import FastAPI, UploadFile, File
from scripts.transcribe_audio import transcribe_audio
from scripts.recommend_songs import recommend_songs, db_features
import os

app = FastAPI()

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    transcription = transcribe_audio(file_location)
    return {"transcription": transcription}

@app.post("/recommend/")
async def recommend(file: UploadFile = File(...)):
    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    recommendations = recommend_songs(file_location, db_features)
    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
