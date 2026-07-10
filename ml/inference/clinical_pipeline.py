from pathlib import Path

from app.domain.enums import BehaviourType
from app.domain.risk import (
    alarm_required_for_behaviour,
    alarm_required_for_objects,
    risk_level_for_behaviour,
)
from ml.inference.predictor import BaselineVideoClassifierPredictor, PredictionResult


class PretrainedClinicalObservationPipeline:
    model_version = "clinical-pipeline-v1"

    def __init__(
        self,
        baseline_model_path: Path,
        enable_objects: bool = True,
        enable_pose: bool = True,
    ) -> None:
        self.baseline = BaselineVideoClassifierPredictor(model_path=baseline_model_path)
        self.enable_objects = enable_objects
        self.enable_pose = enable_pose
        self._object_detector = None
        self._pose_analyzer = None
        self._object_detector_error: str | None = None
        self._pose_analyzer_error: str | None = None

    def predict(self, video_path: Path | None = None) -> PredictionResult:
        baseline_result = self.baseline.predict(video_path)
        if video_path is None:
            return baseline_result

        dangerous_objects, object_status = self._detect_dangerous_objects(video_path)
        pose_status = self._analyze_pose(video_path)

        behaviour = (
            BehaviourType(baseline_result.predicted_behaviour)
            if baseline_result.predicted_behaviour is not None
            else None
        )
        object_alarm_required = alarm_required_for_objects(dangerous_objects)
        behaviour_alarm_required = alarm_required_for_behaviour(behaviour)
        alarm_required = behaviour_alarm_required or object_alarm_required
        alarm_reason = self._alarm_reason(
            behaviour_alarm_required=behaviour_alarm_required,
            object_alarm_required=object_alarm_required,
        )

        return PredictionResult(
            predicted_behaviour=baseline_result.predicted_behaviour,
            confidence=baseline_result.confidence,
            dangerous_objects_detected=dangerous_objects,
            alarm_required=alarm_required,
            alarm_reason=alarm_reason,
            model_version=self._model_version(object_status, pose_status),
            status=baseline_result.status,
            message=(
                f"{baseline_result.message} "
                f"Object detector status: {object_status}. "
                f"Pose analyzer status: {pose_status}."
            ),
        )

    def _detect_dangerous_objects(self, video_path: Path) -> tuple[list[str], str]:
        if not self.enable_objects:
            return [], "disabled"
        if self._object_detector_error is not None:
            return [], self._object_detector_error

        try:
            if self._object_detector is None:
                from ml.inference.object_detector import PretrainedObjectDetector

                self._object_detector = PretrainedObjectDetector()

            from ml.inference.object_detector import dangerous_objects_from_detections

            detections = self._object_detector.detect_video(video_path)
            return dangerous_objects_from_detections(detections), "ok"
        except Exception as exc:
            self._object_detector_error = f"unavailable ({exc})"
            return [], self._object_detector_error

    def _analyze_pose(self, video_path: Path) -> str:
        if not self.enable_pose:
            return "disabled"
        if self._pose_analyzer_error is not None:
            return self._pose_analyzer_error

        try:
            if self._pose_analyzer is None:
                from ml.inference.pose_detector import MediaPipePoseMovementAnalyzer

                self._pose_analyzer = MediaPipePoseMovementAnalyzer()

            summary = self._pose_analyzer.analyze_video(video_path)
            return (
                "ok "
                f"coverage={summary.pose_coverage:.2f} "
                f"motion={summary.mean_motion:.3f} "
                f"posture_change={summary.posture_change:.3f}"
            )
        except Exception as exc:
            self._pose_analyzer_error = f"unavailable ({exc})"
            return self._pose_analyzer_error

    def _model_version(self, object_status: str, pose_status: str) -> str:
        versions = [self.baseline.model_version]
        if object_status.startswith("ok"):
            versions.append("yolo-object-v1")
        if pose_status.startswith("ok"):
            versions.append("mediapipe-pose-v1")
        return "+".join(versions)

    def _alarm_reason(
        self,
        behaviour_alarm_required: bool,
        object_alarm_required: bool,
    ) -> str | None:
        if behaviour_alarm_required and object_alarm_required:
            return "High-risk behaviour and dangerous object detected."
        if object_alarm_required:
            return "Dangerous object detected."
        if behaviour_alarm_required:
            return "High-risk behaviour detected."
        return None

    def risk_level(self, label: str | None) -> str:
        if label is None:
            return "low"
        return risk_level_for_behaviour(BehaviourType(label)).value
