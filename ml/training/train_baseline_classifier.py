import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ml.training.dataset_manifest import TrainingManifestItem, load_training_manifest
from ml.training.video_features import FEATURE_VERSION, extract_video_features


DEFAULT_MANIFEST = Path("dataset/training_manifest.jsonl")
DEFAULT_OUTPUT = Path("ml/models/baseline_video_classifier.json")
DEFAULT_FEATURE_CACHE = Path("dataset/features_baseline.jsonl")


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    test_count: int
    correct_count: int
    per_label: dict[str, dict[str, int | float]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline video classifier.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--test-every", type=int, default=5)
    parser.add_argument("--max-per-label", type=int, default=6)
    parser.add_argument("--sample-count", type=int, default=8)
    args = parser.parse_args()

    items = load_training_manifest(args.manifest)
    if not items:
        raise ValueError(f"No training items found in {args.manifest}")

    feature_rows = _extract_features(
        items=items,
        feature_cache_path=args.feature_cache,
        max_per_label=args.max_per_label,
        sample_count=args.sample_count,
    )
    train_rows, test_rows = _split_rows(feature_rows, test_every=args.test_every)
    model = _train_nearest_centroid(train_rows)
    evaluation = _evaluate(model, test_rows)
    _write_model(args.output, model, evaluation, len(items))

    print(f"Trained baseline model with {len(train_rows)} training clips")
    print(f"Evaluated on {evaluation.test_count} clips")
    print(f"Accuracy: {evaluation.accuracy:.3f}")
    print(f"Wrote model to {args.output}")


def _extract_features(
    items: list[TrainingManifestItem],
    feature_cache_path: Path,
    max_per_label: int,
    sample_count: int,
) -> list[dict[str, object]]:
    cached_rows = _read_feature_cache(feature_cache_path)
    cache_key = {
        str(row["video_id"]): row
        for row in cached_rows
        if row.get("feature_version") == FEATURE_VERSION
    }

    rows = []
    included_counts: Counter[str] = Counter()
    for index, item in enumerate(items, start=1):
        if included_counts[item.label] >= max_per_label:
            continue

        if item.relative_path is None:
            raise ValueError(f"Manifest item is missing relative_path: {item.video_id}")

        if item.video_id in cache_key:
            row = cache_key[item.video_id]
            rows.append(row)
            included_counts[item.label] += 1
            continue

        video_path = Path(item.relative_path)
        print(f"Extracting features {index}/{len(items)}: {item.video_id}")
        row = {
            "video_id": item.video_id,
            "label": item.label,
            "feature_version": FEATURE_VERSION,
            "features": extract_video_features(
                video_path,
                sample_count=sample_count,
            ),
        }
        rows.append(row)
        included_counts[item.label] += 1
        _append_feature_cache(feature_cache_path, row)
    return rows


def _read_feature_cache(feature_cache_path: Path) -> list[dict[str, object]]:
    if not feature_cache_path.exists():
        return []
    with feature_cache_path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _append_feature_cache(
    feature_cache_path: Path,
    row: dict[str, object],
) -> None:
    feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_cache_path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(row) + "\n")


def _split_rows(
    rows: list[dict[str, object]],
    test_every: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if test_every < 2:
        raise ValueError("test_every must be at least 2")

    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label"])].append(row)

    train_rows = []
    test_rows = []
    for label, label_rows in sorted(by_label.items()):
        for index, row in enumerate(label_rows, start=1):
            if len(label_rows) > 1 and index % test_every == 0:
                test_rows.append(row)
            else:
                train_rows.append(row)

    if not test_rows:
        raise ValueError("No test rows were selected; lower --test-every")
    return train_rows, test_rows


def _train_nearest_centroid(rows: list[dict[str, object]]) -> dict[str, object]:
    by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label"])].append(np.array(row["features"], dtype=np.float32))

    centroids = {
        label: np.stack(vectors).mean(axis=0).tolist()
        for label, vectors in sorted(by_label.items())
    }

    return {
        "model_type": "nearest_centroid",
        "feature_version": FEATURE_VERSION,
        "labels": sorted(centroids),
        "centroids": centroids,
        "train_counts": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
    }


def _evaluate(
    model: dict[str, object],
    rows: list[dict[str, object]],
) -> EvaluationResult:
    correct_count = 0
    per_label_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    for row in rows:
        expected = str(row["label"])
        predicted, _ = predict_with_model(model, list(row["features"]))
        per_label_counts[expected]["total"] += 1
        if predicted == expected:
            correct_count += 1
            per_label_counts[expected]["correct"] += 1

    per_label = {}
    for label, counts in sorted(per_label_counts.items()):
        total = counts["total"]
        per_label[label] = {
            "correct": counts["correct"],
            "total": total,
            "accuracy": counts["correct"] / total if total else 0.0,
        }

    return EvaluationResult(
        accuracy=correct_count / len(rows),
        test_count=len(rows),
        correct_count=correct_count,
        per_label=per_label,
    )


def predict_with_model(
    model: dict[str, object],
    features: list[float],
) -> tuple[str, float]:
    feature_vector = np.array(features, dtype=np.float32)
    distances = {}
    for label, centroid in dict(model["centroids"]).items():
        centroid_vector = np.array(centroid, dtype=np.float32)
        distances[str(label)] = float(np.linalg.norm(feature_vector - centroid_vector))

    predicted_label = min(distances, key=distances.get)
    sorted_distances = sorted(distances.values())
    best_distance = sorted_distances[0]
    second_best = sorted_distances[1] if len(sorted_distances) > 1 else best_distance
    confidence = second_best / (best_distance + second_best + 1e-6)
    return predicted_label, float(confidence)


def _write_model(
    output_path: Path,
    model: dict[str, object],
    evaluation: EvaluationResult,
    dataset_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **model,
        "dataset_size": dataset_size,
        "evaluation": asdict(evaluation),
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


if __name__ == "__main__":
    main()
