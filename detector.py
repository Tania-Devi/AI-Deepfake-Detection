import os
import tempfile

import cv2
from PIL import Image
from transformers import pipeline

_model = None


def load_model():
    global _model

    if _model is None:
        _model = pipeline(
            "image-classification",
            model="hamzenium/ViT-Deepfake-Classifier",
            device=-1
        )

    return _model


def analyze_image(image):
    """Analyze one image using the pretrained model."""
    image = image.convert("RGB")
    predictions = load_model()(image)

    return {
        "width": image.width,
        "height": image.height,
        "predictions": predictions,
        "status": "Analysis completed"
    }


def analyze_video(video_bytes, max_frames=10):
    """Analyze up to 10 evenly spaced frames from a video."""
    temp_path = None
    cap = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise ValueError(
                "Could not open this video. Try an MP4 video."
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise ValueError("No readable video frames were found.")

        frame_count = min(total_frames, max_frames)
        frame_indexes = [
            int(i * (total_frames - 1) / max(frame_count - 1, 1))
            for i in range(frame_count)
        ]

        model = load_model()
        frame_results = []
        score_totals = {}

        for frame_number in frame_indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            predictions = model(image)

            frame_results.append({
                "frame": frame_number,
                "predictions": predictions
            })

            for item in predictions:
                label = item["label"].lower()
                score_totals[label] = (
                    score_totals.get(label, 0.0) + item["score"]
                )

        if not frame_results:
            raise ValueError("Could not analyze any video frames.")

        average_scores = {
            label: total / len(frame_results)
            for label, total in score_totals.items()
        }

        return {
            "frames_analyzed": len(frame_results),
            "average_scores": average_scores,
            "frame_results": frame_results
        }

    finally:
        if cap is not None:
            cap.release()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)