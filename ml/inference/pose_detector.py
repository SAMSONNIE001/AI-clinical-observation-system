from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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
    ) -> None:
        import mediapipe as mp

        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "pose"):
            version = getattr(mp, "__version__", "unknown")
            raise RuntimeError(
                "MediaPipe Pose is unavailable in the installed mediapipe "
                f"package (version={version}). Install a MediaPipe build that "
                "provides mp.solutions.pose, or run the pipeline with pose "
                "analysis disabled."
            )

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

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

        step = max(1, frame_count // sample_count)
        sampled = 0
        landmarks_by_frame: list[np.ndarray] = []

        try:
            frame_index = 0
            while sampled < sample_count:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % step == 0:
                    sampled += 1
                    landmarks = self._extract_landmarks(frame)
                    if landmarks is not None:
                        landmarks_by_frame.append(landmarks)

                frame_index += 1
        finally:
            capture.release()

        return summarize_pose_movement(sampled, landmarks_by_frame)

    def _extract_landmarks(self, frame) -> np.ndarray | None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
