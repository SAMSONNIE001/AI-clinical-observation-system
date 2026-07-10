import argparse
import sys
import tempfile
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "baseline_video_classifier.json"
ACTION_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "video_action_classifier.pt"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_PATH = PROJECT_ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from ml.inference.clinical_pipeline import PretrainedClinicalObservationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the clinical observation pipeline on short live camera clips."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open.")
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=6.0,
        help="Seconds of video to classify at a time.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=3.0,
        help="Minimum seconds between predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional folder to keep camera clips instead of using temporary files.",
    )
    parser.add_argument(
        "--alarm-threshold",
        type=float,
        default=0.70,
        help="Minimum confidence required before a high-risk prediction can alarm.",
    )
    parser.add_argument(
        "--alarm-confirmations",
        type=int,
        default=2,
        help="Number of repeated high-risk predictions required before alarming.",
    )
    parser.add_argument(
        "--disable-objects",
        action="store_true",
        help="Disable YOLO object detection.",
    )
    parser.add_argument(
        "--disable-pose",
        action="store_true",
        help="Disable MediaPipe pose movement analysis.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not MODEL_PATH.exists():
        print(f"Model file not found: {MODEL_PATH}")
        return 1

    predictor = PretrainedClinicalObservationPipeline(
        baseline_model_path=MODEL_PATH,
        action_model_path=ACTION_MODEL_PATH,
        enable_objects=not args.disable_objects,
        enable_pose=not args.disable_pose,
    )
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Could not open camera index {args.camera}.")
        return 1

    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    frame_limit = max(16, int(fps * args.clip_seconds))
    frames = []
    last_prediction_at = 0.0
    latest_label = "warming_up"
    latest_confidence = 0.0
    latest_alarm = False
    latest_status = "collecting_frames"
    confirmed_label = None
    confirmed_count = 0

    print("Live camera detection started. Press q in the camera window to stop.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera frame could not be read.")
                break

            frames.append(frame.copy())
            if len(frames) > frame_limit:
                frames.pop(0)

            now = time.monotonic()
            can_predict = (
                len(frames) >= frame_limit
                and now - last_prediction_at >= args.interval_seconds
            )
            if can_predict:
                clip_path = write_clip(
                    frames=frames,
                    fps=fps,
                    size=(width, height),
                    output_dir=args.output_dir,
                )
                result = predictor.predict(clip_path)
                latest_label = result.predicted_behaviour or "unknown"
                latest_confidence = result.confidence
                eligible_alarm = (
                    result.alarm_required
                    and latest_confidence >= args.alarm_threshold
                )
                if eligible_alarm and latest_label == confirmed_label:
                    confirmed_count += 1
                elif eligible_alarm:
                    confirmed_label = latest_label
                    confirmed_count = 1
                else:
                    confirmed_label = None
                    confirmed_count = 0

                latest_alarm = (
                    eligible_alarm
                    and confirmed_count >= args.alarm_confirmations
                )
                latest_status = result.status
                last_prediction_at = now

                if latest_alarm:
                    sound_alarm()
                    print(
                        f"ALARM: {latest_label} "
                        f"confidence={latest_confidence:.2f} "
                        f"confirmations={confirmed_count} status={latest_status}"
                    )
                elif result.alarm_required:
                    print(
                        f"review: {latest_label} "
                        f"confidence={latest_confidence:.2f} "
                        f"confirmations={confirmed_count}/"
                        f"{args.alarm_confirmations} status={latest_status}"
                    )
                else:
                    print(
                        f"{latest_label} "
                        f"confidence={latest_confidence:.2f} status={latest_status}"
                    )

                if args.output_dir is None:
                    clip_path.unlink(missing_ok=True)

            draw_overlay(
                frame=frame,
                label=latest_label,
                confidence=latest_confidence,
                alarm=latest_alarm,
                status=latest_status,
            )
            cv2.imshow("AI Clinical Observation - Live Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


def write_clip(
    frames: list,
    fps: float,
    size: tuple[int, int],
    output_dir: Path | None,
) -> Path:
    if output_dir is None:
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".avi",
            prefix="live_camera_",
            delete=False,
        )
        clip_path = Path(temp_file.name)
        temp_file.close()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        clip_path = output_dir / f"live_camera_{int(time.time() * 1000)}.avi"

    writer = cv2.VideoWriter(
        str(clip_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise ValueError(f"Could not create video writer for {clip_path}")

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    return clip_path


def draw_overlay(
    frame,
    label: str,
    confidence: float,
    alarm: bool,
    status: str,
) -> None:
    colour = (0, 0, 255) if alarm else (0, 180, 0)
    alarm_text = "ALARM" if alarm else "monitoring"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 86), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"{alarm_text}: {label} ({confidence:.2f})",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        colour,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"status={status} | press q to stop",
        (16, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def sound_alarm() -> None:
    try:
        import winsound

        winsound.Beep(1200, 300)
    except Exception:
        print("\a", end="")


if __name__ == "__main__":
    raise SystemExit(main())
