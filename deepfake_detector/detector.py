from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from transformers import ViTForImageClassification, ViTImageProcessor


class DeepfakeDetector:

    MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"

    def __init__(self, threshold=0.5, confidence_cutoff=0.7):
        # no more model_path — we load from HuggingFace directly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.confidence_cutoff = confidence_cutoff
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.processor = None  # gets set inside _load_model
        self.model = self._load_model()

    def _load_model(self):
        # downloads model and processor from HuggingFace on first run, cached after
        print(f"Loading model from HuggingFace: {self.MODEL_NAME}")
        self.processor = ViTImageProcessor.from_pretrained(self.MODEL_NAME)
        model = ViTForImageClassification.from_pretrained(self.MODEL_NAME)
        model = model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, pil_image):
        # detect and crop face using MTCNN
        face = self.mtcnn(pil_image)
        if face is None:
            return None

        # MTCNN returns a tensor, convert back to PIL for the ViT processor
        face_pil = Image.fromarray(
            (face.permute(1, 2, 0).numpy() * 128 + 127.5).clip(0, 255).astype(np.uint8)
        )

        # processor handles all normalization and resizing for this specific model
        inputs = self.processor(images=face_pil, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def _predict_from_pil(self, pil_image):
        face_tensor = self.preprocess_image(pil_image)
        if face_tensor is None:
            return {"label": None, "confidence": None, "score": None, "error": "no face detected"}

        with torch.no_grad():
            outputs = self.model(pixel_values=face_tensor)
            # model outputs logits for [real, fake] — softmax to get probabilities
            probs = torch.softmax(outputs.logits, dim=1)
            fake_score = probs[0][1].item()  # index 1 is "fake" class

        label = "fake" if fake_score >= self.threshold else "real"
        return {
            "label": label,
            "confidence": round(fake_score, 4),
            "score": round(fake_score, 4),
        }

    def predict_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self._predict_from_pil(image)

    def predict_video(self, video_path, num_frames=100):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

        frame_results = []
        faces_detected = 0

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # cv2 reads BGR, PIL expects RGB
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self._predict_from_pil(frame_pil)

            if result.get("error"):
                continue

            faces_detected += 1
            frame_results.append({
                "frame": int(idx),
                "label": result["label"],
                "score": result["score"]
            })

        cap.release()

        if not frame_results:
            return {"error": "no faces detected in video", "label": None, "confidence": None, "score": None}

        scores = [r["score"] for r in frame_results]
        confident_scores = [s for s in scores if s >= self.confidence_cutoff or s <= (1 - self.confidence_cutoff)]
        final_score = float(np.mean(confident_scores)) if confident_scores else float(np.mean(scores))
        label = "fake" if final_score >= self.threshold else "real"

        return {
            "label": label,
            "confidence": round(final_score, 4),
            "score": round(final_score, 4),
            "num_frames_analyzed": len(frame_results),
            "num_faces_detected": faces_detected,
            "frame_results": frame_results
        }