# Deepfake Detector
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

Real-time deepfake detection for images and videos, served via a REST API. Built with PyTorch, FastAPI, and a ViT backbone pretrained on real/fake face data.

## Features
- Detection of deepfake videos and synthetic images with confidence scores
- REST API built with FastAPI , accepts image/video uploads, returns JSON predictions
- Real-time video demo with visual overlay of confidence score
- Automatic handling of face detection using MTCNN

## Installation
```bash
git clone https://github.com/Danielvahdati/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
```

## Usage

**Run the demo:**
```bash
python demo/run.py --input path/to/image.jpg
python demo/run.py --input path/to/video.mp4 --output result.mp4 --threshold 0.6
```

**Start the API:**
```bash
uvicorn serve.app:app --host 0.0.0.0 --port 8000
```
Then visit `http://localhost:8000/docs` to test the API interactively.

## Project Structure
```
deepfake-detector/
├── deepfake_detector/
│   ├── detector.py        # core detection logic
│   ├── models/
│   └── utils/
├── serve/
│   └── app.py             # FastAPI REST API
├── demo/
│   └── run.py             # local demo script
├── tests/
├── requirements.txt
└── README.md
```

## Model

The detection backbone is a Vision Transformer (ViT) fine-tuned for binary classification (real vs fake). We use the [`prithivMLmods/Deep-Fake-Detector-v2-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) checkpoint from HuggingFace, trained on a dataset of real human faces and AI-generated fake images.

The model is downloaded automatically on first run and cached locally.

| Metric | Score |
|--------|-------|
| Accuracy | 92.1% |
| Precision (Fake) | 88.3% |
| Recall (Fake) | 97.2% |

## Author

Built by Danial Samadi  Vahdati, PhD candidate in Electrical & Computer Engineering at Drexel University, specializing in AI security and synthetic media forensics.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/danial-vahdati-6a652b169/)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-Profile-green)](https://scholar.google.com/citations?user=FzqABy8AAAAJ&hl=en)