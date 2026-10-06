from pathlib import Path

import pytest

from model_eval import evaluate_classifier

BASE_DIR = Path(__file__).parent.parent
RESNET_PATH = BASE_DIR / "resnet50_cashew.pt"
VAL_DIR = BASE_DIR / "resnet_dataset_v4" / "val"

pytestmark = pytest.mark.skipif(
    not RESNET_PATH.exists() or not VAL_DIR.exists(),
    reason="model/dataset không có sẵn trong môi trường này",
)


def test_evaluate_classifier_invariants():
    result = evaluate_classifier(str(RESNET_PATH), 224, str(VAL_DIR), device="cpu")

    n_classes = len(result["class_names"])
    total_support = sum(c["support"] for c in result["per_class"].values())
    cm_total = sum(sum(row) for row in result["confusion_matrix"])

    assert n_classes == 6
    assert total_support == cm_total
    assert len(result["predictions"]) == total_support
    for c in result["per_class"].values():
        assert 0.0 <= c["precision"] <= 1.0
        assert 0.0 <= c["recall"] <= 1.0
        assert 0.0 <= c["f1"] <= 1.0
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["latency_ms_per_image"] > 0

    correct = sum(result["confusion_matrix"][i][i] for i in range(n_classes))
    assert correct == pytest.approx(result["accuracy"] * cm_total, abs=1)
