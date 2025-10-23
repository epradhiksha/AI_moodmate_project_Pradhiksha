from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil, os
from detectface import predict_emotion_image, predict_emotion_webcam
from TEXT import detect_emotion
from recommendation import recommend_songs_by_emotion

app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure uploads directory exists
os.makedirs("static/uploads", exist_ok=True)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/emotion", response_class=HTMLResponse)
async def emotion_page(request: Request):
    return templates.TemplateResponse("emotion.html", {"request": request})

@app.get("/sentiment", response_class=HTMLResponse)
async def sentiment_page(request: Request):
    return templates.TemplateResponse("sentiment.html", {"request": request})

@app.post("/predict_emotion/", response_class=HTMLResponse)
async def predict_emotion(request: Request, method: str = Form(...), file: UploadFile = File(None), count: int = Form(5)):
    image_url = None
    emotion = None
    songs = None

    if method == "upload" and file:
        # Save uploaded image to static/uploads
        file_path = f"static/uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        emotion = predict_emotion_image(file_path)
        image_url = f"/{file_path}"  # For displaying in HTML

    elif method == "webcam":
        emotion = predict_emotion_webcam()
        image_url = "/static/uploads/temp.jpg" if os.path.exists("static/uploads/temp.jpg") else None

    else:
        emotion = "No input detected"

    # Get song recommendations based on emotion
    if emotion and emotion != "No input detected":
        songs = recommend_songs_by_emotion(emotion,n=count)
    else:
        songs = []

    return templates.TemplateResponse(
        "emotion.html",
        {"request": request, "emotion": emotion, "image_url": image_url, "songs": songs}
    )

@app.post("/predict_sentiment/", response_class=HTMLResponse)
async def predict_sentiment(request: Request, text: str = Form(...), count: int = Form(5)):
    sentiment = detect_emotion(text)
    songs = recommend_songs_by_emotion(sentiment,n=count)
    return templates.TemplateResponse("sentiment.html", {"request": request, "sentiment": sentiment, "songs": songs})
