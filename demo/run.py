import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
sys.path.append(str(Path(__file__).parent.parent))
from deepfake_detector.detector import DeepfakeDetector


parser = argparse.ArgumentParser(description="Deepfake Detection Demo")
parser.add_argument("--input", type=str, required=True, help="Path to input image/video")
parser.add_argument("--output", type=str, required=False, help="Path to save output")
parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
args = parser.parse_args()

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
video_extensions = ['.mp4', '.avi', '.mov']
extensions = image_extensions + video_extensions
suffix = Path(args.input).suffix.lower()
if suffix not in extensions:
    print("Unsupported file format. Please provide an image or video file.")
    exit(1)

is_image = suffix in image_extensions
is_video = suffix in video_extensions

detector = DeepfakeDetector(threshold=args.threshold)


if is_image:
    result = detector.predict_image(args.input)
    image = cv2.imread(args.input)
    label = result['label'].upper()
    confidence = result['confidence']
    text = f"{label}: {confidence:.1%}"
    color = (0,0,255) if label == 'FAKE' else (0,255,0)
    cv2.putText(image, text, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Deepfake Detector", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if args.output:
        cv2.imwrite(args.output, image)
        print(f"Output saved to {args.output}")

elif is_video:
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector._predict_from_pil(frame_pil)
        label = result['label'].upper()
        confidence = result['confidence']
        text = f"{label}: {confidence:.1%}"
        color = (0,0,255) if label == 'FAKE' else (0,255,0)
        cv2.putText(frame, text, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Deepfake Detector", frame)
        if args.output:
            writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    if writer:
        writer.release()
        print(f"Output saved to {args.output}")
    cv2.destroyAllWindows()
