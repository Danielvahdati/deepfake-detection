from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import sys
from pathlib import Path
import shutil
import tempfile

sys.path.append(str(Path(__file__).parent.parent))  # add parent dir to path for imports
from deepfake_detector.detector import DeepfakeDetector


app = FastAPI(
    title="Deepfake Detector API",
    description="Real-time deepfake detection for images and videos",
    version="0.1.0"
)

detector = DeepfakeDetector()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    result = detector.predict_image(tmp_path)
    Path(tmp_path).unlink()  # clean up temp file
    return JSONResponse(result)


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    result = detector.predict_video(tmp_path)
    Path(tmp_path).unlink()  # clean up temp file
    return JSONResponse(result)