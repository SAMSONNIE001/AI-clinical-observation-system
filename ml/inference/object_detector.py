from dataclasses import dataclass
from pathlib import Path

import cv2


DEFAULT_YOLO_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolov8n.pt"

DANGEROUS_COCO_OBJECTS = {
    "knife": "knife",
    "scissors": "scissors",
}


@dataclass(frozen=True)
class ObjectDetection:
    label: str
    confidence: float
    frame_index: int


class PretrainedObjectDetector:
    model_version = "yolo-object-v1"

    def __init__(
        self,
        model_name: str | Path = DEFAULT_YOLO_MODEL_PATH,
        confidence_threshold: float = 0.35,
    ) -> None:
        from ultralytics import YOLO

        model_path = Path(model_name)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model weights not found: {model_path}. "
                "Run `python scripts/download_yolo_weights.py` first."
            )

        self.model = YOLO(str(model_path))
        self.confidence_threshold = confidence_threshold

    def detect_video(
        self,
        video_path: Path,
        sample_count: int = 12,
    ) -> list[ObjectDetection]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            capture.release()
            raise ValueError(f"Could not read frame count: {video_path}")

        step = max(1, frame_count // sample_count)
        detections: list[ObjectDetection] = []

        try:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % step == 0:
                    detections.extend(self.detect_frame(frame, frame_index))

                frame_index += 1
        finally:
            capture.release()

        return detections

    def detect_frame(self, frame, frame_index: int = 0) -> list[ObjectDetection]:
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
        )
        detections: list[ObjectDetection] = []

        for result in results:
            names = result.names
            for box in result.boxes:
                label = names[int(box.cls[0])]
                confidence = float(box.conf[0])
                detections.append(
                    ObjectDetection(
                        label=label,
                        confidence=confidence,
                        frame_index=frame_index,
                    )
                )

        return detections


def dangerous_objects_from_detections(
    detections: list[ObjectDetection],
) -> list[str]:
    dangerous_objects = {
        DANGEROUS_COCO_OBJECTS[detection.label]
        for detection in detections
        if detection.label in DANGEROUS_COCO_OBJECTS
    }
    return sorted(dangerous_objects)
