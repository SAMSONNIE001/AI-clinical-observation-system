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
        action_model_path: Path | None = None,
        enable_objects: bool = True,
        enable_pose: bool = True,
    ) -> None:
        self.baseline = BaselineVideoClassifierPredictor(model_path=baseline_model_path)
        self.action_model_path = action_model_path
        self.enable_objects = enable_objects
        self.enable_pose = enable_pose
        self._action_classifier = None
        self._action_classifier_error: str | None = None
        self._object_detector = None
        self._pose_analyzer = None
        self._object_detector_error: str | None = None
        self._pose_analyzer_error: str | None = None

    def predict(self, video_path: Path | None = None) -> PredictionResult:
        baseline_result = self.baseline.predict(video_path)
        if video_path is None:
            return baseline_result

        action_label, action_confidence, action_status, action_version = (
            self._predict_action(video_path)
        )
        dangerous_objects, object_status = self._detect_dangerous_objects(video_path)
        pose_status = self._analyze_pose(video_path)
        predicted_label = action_label or baseline_result.predicted_behaviour
        confidence = (
            action_confidence
            if action_label is not None
            else baseline_result.confidence
        )

        behaviour = (
            BehaviourType(predicted_label)
            if predicted_label is not None
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
            predicted_behaviour=predicted_label,
            confidence=confidence,
            dangerous_objects_detected=dangerous_objects,
            alarm_required=alarm_required,
            alarm_reason=alarm_reason,
            model_version=self._model_version(
                action_version=action_version,
                object_status=object_status,
                pose_status=pose_status,
            ),
            status=baseline_result.status,
            message=(
                f"{baseline_result.message} "
                f"Action classifier status: {action_status}. "
                f"Object detector status: {object_status}. "
                f"Pose analyzer status: {pose_status}."
            ),
        )

    def _predict_action(self, video_path: Path) -> tuple[str | None, float, str, str | None]:
        if self.action_model_path is None:
            return None, 0.0, "not_configured", None
        if not self.action_model_path.exists():
            return None, 0.0, "checkpoint_not_found", None
        if self._action_classifier_error is not None:
            return None, 0.0, self._action_classifier_error, None

        try:
            if self._action_classifier is None:
                from ml.inference.video_action_classifier import (
                    TorchVisionVideoActionClassifier,
                )

                self._action_classifier = TorchVisionVideoActionClassifier(
                    checkpoint_path=self.action_model_path
                )

            prediction = self._action_classifier.predict(video_path)
            return (
                prediction.label,
                prediction.confidence,
                "ok",
                prediction.model_version,
            )
        except Exception as exc:
            self._action_classifier_error = f"unavailable ({exc})"
            return None, 0.0, self._action_classifier_error, None

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

    def _model_version(
        self,
        action_version: str | None,
        object_status: str,
        pose_status: str,
    ) -> str:
        versions = [action_version or self.baseline.model_version]
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
