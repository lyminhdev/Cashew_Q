from collections import Counter

import cv2
import numpy as np

from scripts.curate_sample_images import (
    CLASS_SOURCE_DIRS,
    class_source_map,
    evaluate_candidate,
    select_samples,
)


class FakePipeline:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, image):
        return self.predictions


def test_class_source_map_uses_only_yl_folders(tmp_path):
    sources = class_source_map(tmp_path)

    assert set(sources) == set(CLASS_SOURCE_DIRS)
    assert sources["bad_output"].name == "badoutput_yl"
    assert all(path.parent == tmp_path for path in sources.values())


def test_candidate_requires_matching_top_prediction(tmp_path):
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    pipeline = FakePipeline([{"class": "loai1", "conf_class": 0.97}])

    assert evaluate_candidate(image_path, "loai2", pipeline) is None


def test_select_samples_returns_two_distinct_records_per_class():
    records = []
    for class_name in CLASS_SOURCE_DIRS:
        records.extend(
            {
                "source": f"{class_name}-{index}.png",
                "expected_class": class_name,
                "predicted_class": class_name,
                "confidence": 0.9 - index / 100,
                "detected_count": 1,
            }
            for index in range(3)
        )

    selected = select_samples(records, per_class=2)

    assert Counter(item["expected_class"] for item in selected) == {
        class_name: 2 for class_name in CLASS_SOURCE_DIRS
    }
    assert len({item["source"] for item in selected}) == 12
