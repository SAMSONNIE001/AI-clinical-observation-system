from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MEDIAPIPE_POSE_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_full.task"
)


@dataclass(frozen=True)
class PoseMovementSummary:
    frames_sampled: int
    pose_frames: int
    pose_coverage: float
    mean_motion: float
    max_motion: float
    vertical_motion: float
    horizontal_motion: float
    posture_change: float


class MediaPipePoseMovementAnalyzer:
    model_version = "mediapipe-pose-v1"

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Path = DEFAULT_MEDIAPIPE_POSE_MODEL_PATH,
    ) -> None:
        import mediapipe as mp

        self.mp = mp
        self.backend = "solutions"
        self.pose = None
        self.landmarker = None
        self._next_timestamp_ms = 0

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        if not hasattr(mp, "tasks") or not hasattr(mp.tasks, "vision"):
            version = getattr(mp, "__version__", "unknown")
            raise RuntimeError(
                "MediaPipe Pose is unavailable in the installed mediapipe "
                f"package (version={version})."
            )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe pose model not found: {model_path}. "
                "Run `python scripts/download_mediapipe_pose_model.py` first."
            )

        self.backend = "tasks"
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def analyze_video(
        self,
        video_path: Path,
        sample_count: int = 24,
    ) -> PoseMovementSummary:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            capture.release()
            raise ValueError(f"Could not read frame count: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, frame_count // sample_count)
        sampled = 0
        landmarks_by_frame: list[np.ndarray] = []

        try:
            frame_index = 0
            latest_timestamp_ms = self._next_timestamp_ms
            while sampled < sample_count:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % step == 0:
                    sampled += 1
                    timestamp_ms = self._next_timestamp_ms + int(
                        (frame_index / fps) * 1000
                    )
                    latest_timestamp_ms = timestamp_ms
                    landmarks = self._extract_landmarks_for_video(
                        frame,
                        timestamp_ms,
                    )
                    if landmarks is not None:
                        landmarks_by_frame.append(landmarks)

                frame_index += 1
        finally:
            capture.release()
            self._next_timestamp_ms = latest_timestamp_ms + 1

        return summarize_pose_movement(sampled, landmarks_by_frame)

    def _extract_landmarks(self, frame) -> np.ndarray | None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.backend == "tasks":
            raise RuntimeError(
                "MediaPipe tasks backend requires frame timestamps. "
                "Use `_extract_landmarks_for_video`."
            )

        result = self.pose.process(rgb_frame)
        if not result.pose_landmarks:
            return None

        return np.array(
            [
                [landmark.x, landmark.y, landmark.z, landmark.visibility]
                for landmark in result.pose_landmarks.landmark
            ],
            dtype=np.float32,
        )

    def _extract_landmarks_for_video(
        self,
        frame,
        timestamp_ms: int,
    ) -> np.ndarray | None:
        if self.backend != "tasks":
            return self._extract_landmarks(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return None

        return np.array(
            [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    getattr(landmark, "visibility", 1.0),
                ]
                for landmark in result.pose_landmarks[0]
            ],
            dtype=np.float32,
        )


def summarize_pose_movement(
    frames_sampled: int,
    landmarks_by_frame: list[np.ndarray],
) -> PoseMovementSummary:
    pose_frames = len(landmarks_by_frame)
    if frames_sampled <= 0 or pose_frames == 0:
        return PoseMovementSummary(
            frames_sampled=frames_sampled,
            pose_frames=pose_frames,
            pose_coverage=0.0,
            mean_motion=0.0,
            max_motion=0.0,
            vertical_motion=0.0,
            horizontal_motion=0.0,
            posture_change=0.0,
        )

    coordinates = np.stack([frame[:, :2] for frame in landmarks_by_frame])
    if pose_frames < 2:
        motion = np.zeros((1, coordinates.shape[1]), dtype=np.float32)
    else:
        motion = np.linalg.norm(np.diff(coordinates, axis=0), axis=2)

    nose_y = coordinates[:, 0, 1]
    left_shoulder_y = coordinates[:, 11, 1]
    right_shoulder_y = coordinates[:, 12, 1]
    shoulder_y = (left_shoulder_y + right_shoulder_y) / 2.0
    posture = np.abs(nose_y - shoulder_y)

    return PoseMovementSummary(
        frames_sampled=frames_sampled,
        pose_frames=pose_frames,
        pose_coverage=pose_frames / frames_sampled,
        mean_motion=float(motion.mean()),
        max_motion=float(motion.max()),
        vertical_motion=float(np.ptp(coordinates[:, :, 1])),
        horizontal_motion=float(np.ptp(coordinates[:, :, 0])),
        posture_change=float(np.ptp(posture)),
    )
