from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import sys
from pathlib import Path
import shutil
import tempfile

sys.path.append(str(Path(__file__).parent.parent))  # add parent dir to path for imports
from deepfake-detection.detector import DeepfakeDetector  #