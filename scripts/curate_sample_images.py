"""Build a verified, deployable sample-image library from the *_yl folders."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CLASS_SOURCE_DIRS = {
    "tb": "tb_yl",
    "loai1": "loai1_yl",
    "loai2": "loai2_yl",
    "loai3": "loai3_yl",
    "lbw": "lbw_yl",
    "bad_output": "badoutput_yl",
}
DISPLAY_NAMES = {
    "tb": "Tốt",
    "loai1": "Loại 1",
    "loai2": "Loại 2",
    "loai3": "Loại 3",
    "lbw": "LBW",
    "bad_output": "Lỗi",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def class_source_map(source_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    """Return the known class-to-source-folder mapping without reading images."""
    return {name: source_root / directory for name, directory in CLASS_SOURCE_DIRS.items()}


def evaluate_candidate(path: Path, expected_class: str, pipeline: Any) -> dict[str, Any] | None:
    """Return verification metadata when a readable frame predicts its source class."""
    image = cv2.imread(str(path))
    if image is None:
        return None
    predictions = pipeline.predict(image)
    matching = [prediction for prediction in predictions if prediction["class"] == expected_class]
    if not matching:
        return None
    top = max(matching, key=lambda prediction: float(prediction["conf_class"]))
    return {
        "source": str(path),
        "expected_class": expected_class,
        "predicted_class": top["class"],
        "confidence": round(float(top["conf_class"]), 4),
        "detected_count": len(predictions),
    }


def select_samples(records: list[dict[str, Any]], per_class: int = 2) -> list[dict[str, Any]]:
    """Choose highest-confidence, distinct-source records per class deterministically."""
    selected: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["expected_class"] == record["predicted_class"] and record["detected_count"] > 0:
            grouped[record["expected_class"]].append(record)

    for class_name in CLASS_SOURCE_DIRS:
        candidates = sorted(
            grouped[class_name],
            key=lambda item: (-float(item["confidence"]), str(item["source"])),
        )
        distinct = []
        seen_sources = set()
        for item in candidates:
            if item["source"] not in seen_sources:
                distinct.append(item)
                seen_sources.add(item["source"])
            if len(distinct) == per_class:
                break
        if len(distinct) != per_class:
            raise RuntimeError(f"{class_name} only has {len(distinct)} verified sample(s); need {per_class}")
        selected.extend(distinct)
    return selected


def write_manifest(samples: list[dict[str, Any]], output_dir: Path) -> Path:
    """Copy selected images and write the static manifest consumed by the webapp."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_samples = []
    counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        class_name = sample["expected_class"]
        counts[class_name] += 1
        number = counts[class_name]
        stem = "bad-output" if class_name == "bad_output" else class_name
        filename = f"{stem}-{number:02d}.png"
        destination = output_dir / filename
        shutil.copy2(sample["source"], destination)
        manifest_samples.append(
            {
                "id": f"{stem}-{number:02d}",
                "path": f"/samples/{filename}",
                "expected_class": class_name,
                "display_name": DISPLAY_NAMES[class_name],
                "detected_count": sample["detected_count"],
                "predicted_class": sample["predicted_class"],
                "confidence": sample["confidence"],
            }
        )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"samples": manifest_samples}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def iter_source_images(source_root: Path, max_per_class: int | None = None):
    for class_name, folder in class_source_map(source_root).items():
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing source folder: {folder}")
        yielded = 0
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield class_name, path
                yielded += 1
                if max_per_class is not None and yielded >= max_per_class:
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="Create verified Cashew QC sample images")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "static" / "samples")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--model", choices=("resnet50", "efficientnet_b3"), default="efficientnet_b3")
    args = parser.parse_args()

    from api import get_pipeline

    pipeline, model_id = get_pipeline(args.model)
    records = []
    for expected_class, path in iter_source_images(args.source_root, args.max_per_class):
        record = evaluate_candidate(path, expected_class, pipeline)
        if record is not None:
            records.append(record)
    selected = select_samples(records, args.per_class)
    manifest_path = write_manifest(selected, args.output)
    print(f"Wrote {len(selected)} verified samples with {model_id}: {manifest_path}")


if __name__ == "__main__":
    main()
