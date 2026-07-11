from pathlib import Path

from app.domain.enums import BehaviourType
from app.domain.risk import (
    alarm_required_for_behaviour,
    risk_level_for_behaviour,
)
from ml.inference.predictor import BaselineVideoClassifierPredictor, PredictionResult
from ml.inference.risk_engine import RiskSignalSnapshot, assess_risk


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

        (
            action_label,
            action_confidence,
            action_status,
            action_version,
            action_label_mode,
        ) = (
            self._predict_action(video_path)
        )
        dangerous_objects, clinical_object_cues, object_status = (
            self._detect_dangerous_objects(video_path)
        )
        pose_summary, pose_status = self._analyze_pose(video_path)
        predicted_label = (
            action_label
            if action_label_mode == "detailed"
            else baseline_result.predicted_behaviour
        )
        confidence = (
            action_confidence
            if action_label_mode == "detailed" and action_label is not None
            else baseline_result.confidence
        )

        behaviour = (
            BehaviourType(predicted_label)
            if predicted_label is not None
            else None
        )
        behaviour_alarm_required = alarm_required_for_behaviour(behaviour)
        risk_assessment = assess_risk(
            RiskSignalSnapshot(
                action_group=action_label if action_label_mode == "grouped" else None,
                action_confidence=action_confidence,
                dangerous_objects=dangerous_objects,
                clinical_object_cues=clinical_object_cues,
                pose_summary=pose_summary,
            )
        )
        alarm_required = behaviour_alarm_required or risk_assessment.alarm_required
        alarm_reason = self._alarm_reason(
            behaviour_alarm_required=behaviour_alarm_required,
            risk_assessment=risk_assessment,
        )

        return PredictionResult(
            predicted_behaviour=predicted_label,
            confidence=confidence,
            dangerous_objects_detected=dangerous_objects,
            alarm_required=alarm_required,
            alarm_reason=alarm_reason,
            risk_group=risk_assessment.risk_group,
            risk_level=(
                "high"
                if behaviour_alarm_required
                else risk_assessment.risk_level
            ),
            risk_reasons=risk_assessment.reasons,
            observation_summary=risk_assessment.observation_summary,
            model_version=self._model_version(
                action_version=action_version,
                object_status=object_status,
                pose_status=pose_status,
            ),
            status=baseline_result.status,
            message=(
                f"{baseline_result.message} "
                f"Action classifier status: {action_status}"
                f"{self._action_label_message(action_label, action_label_mode)}. "
                f"Object detector status: {object_status}. "
                f"Pose analyzer status: {pose_status}. "
                f"Risk engine: group={risk_assessment.risk_group} "
                f"level={risk_assessment.risk_level} "
                f"summary={risk_assessment.observation_summary}"
            ),
        )

    def _predict_action(
        self,
        video_path: Path,
    ) -> tuple[str | None, float, str, str | None, str | None]:
        if self.action_model_path is None:
            return None, 0.0, "not_configured", None, None
        if not self.action_model_path.exists():
            return None, 0.0, "checkpoint_not_found", None, None
        if self._action_classifier_error is not None:
            return None, 0.0, self._action_classifier_error, None, None

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
                prediction.label_mode,
            )
        except Exception as exc:
            self._action_classifier_error = f"unavailable ({exc})"
            return None, 0.0, self._action_classifier_error, None, None

    def _detect_dangerous_objects(
        self,
        video_path: Path,
    ) -> tuple[list[str], list[str], str]:
        if not self.enable_objects:
            return [], [], "disabled"
        if self._object_detector_error is not None:
            return [], [], self._object_detector_error

        try:
            if self._object_detector is None:
                from ml.inference.object_detector import PretrainedObjectDetector

                self._object_detector = PretrainedObjectDetector()

            from ml.inference.object_detector import (
                clinical_object_cues_from_detections,
                dangerous_objects_from_detections,
            )

            detections = self._object_detector.detect_video(video_path)
            return (
                dangerous_objects_from_detections(detections),
                clinical_object_cues_from_detections(detections),
                "ok",
            )
        except Exception as exc:
            self._object_detector_error = f"unavailable ({exc})"
            return [], [], self._object_detector_error

    def _analyze_pose(self, video_path: Path):
        if not self.enable_pose:
            return None, "disabled"
        if self._pose_analyzer_error is not None:
            return None, self._pose_analyzer_error

        try:
            if self._pose_analyzer is None:
                from ml.inference.pose_detector import MediaPipePoseMovementAnalyzer

                self._pose_analyzer = MediaPipePoseMovementAnalyzer()

            summary = self._pose_analyzer.analyze_video(video_path)
            return summary, (
                "ok "
                f"coverage={summary.pose_coverage:.2f} "
                f"motion={summary.mean_motion:.3f} "
                f"posture_change={summary.posture_change:.3f}"
            )
        except Exception as exc:
            self._pose_analyzer_error = f"unavailable ({exc})"
            return None, self._pose_analyzer_error

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
        risk_assessment,
    ) -> str | None:
        if behaviour_alarm_required and risk_assessment.alarm_required:
            return "High-risk behaviour and structured risk signal detected."
        if risk_assessment.alarm_required:
            return risk_assessment.observation_summary
        if behaviour_alarm_required:
            return "High-risk behaviour detected."
        return None

    def _action_label_message(
        self,
        action_label: str | None,
        action_label_mode: str | None,
    ) -> str:
        if action_label is None:
            return ""
        return f" label={action_label} label_mode={action_label_mode}"

    def risk_level(self, label: str | None) -> str:
        if label is None:
            return "low"
        return risk_level_for_behaviour(BehaviourType(label)).value
