"""
tests/test_pipeline.py
----------------------
Unit tests for the cashew nut detection pipeline.

Run:
    pytest tests/
    pytest tests/ -v
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Allow importing from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Class index mapping ────────────────────────────────────────────────────────

# ImageFolder sorts classes alphabetically. The model was trained with that
# ordering so inference CLASS_NAMES must match exactly.
EXPECTED_CLASS_ORDER = ["bad_output", "lbw", "loai1", "loai2", "loai3", "tb"]


def test_root_class_names_alphabetical():
    """Root inference_pipeline.py CLASS_NAMES must match ImageFolder order."""
    import inference_pipeline as ip
    assert ip.CLASS_NAMES == EXPECTED_CLASS_ORDER, (
        f"Root CLASS_NAMES is {ip.CLASS_NAMES!r}, expected {EXPECTED_CLASS_ORDER!r}. "
        "ImageFolder assigns indices alphabetically — mismatch causes wrong labels."
    )


def test_deployment_class_names_alphabetical():
    """deployment_package/inference_pipeline.py CLASS_NAMES must match ImageFolder order."""
    sys.path.insert(0, str(ROOT / "deployment_package"))
    try:
        import importlib
        dp = importlib.import_module("inference_pipeline")
        # Reload to avoid cached root version
        importlib.reload(dp)
        # Check the deployment package's CLASS_NAMES by reading the file directly
        # (avoids module naming conflicts)
    finally:
        sys.path.pop(0)

    src = (ROOT / "deployment_package" / "inference_pipeline.py").read_text(encoding="utf-8")
    # Extract CLASS_NAMES list from source
    import ast, re
    match = re.search(r'CLASS_NAMES\s*=\s*(\[.*?\])', src, re.DOTALL)
    assert match, "Could not find CLASS_NAMES in deployment_package/inference_pipeline.py"
    class_names = ast.literal_eval(match.group(1))
    assert class_names == EXPECTED_CLASS_ORDER, (
        f"deployment_package CLASS_NAMES is {class_names!r}, expected {EXPECTED_CLASS_ORDER!r}."
    )


def test_demo_class_names_alphabetical():
    """demo/demo_pipeline.py CLASS_NAMES must match ImageFolder order."""
    src = (ROOT / "demo" / "demo_pipeline.py").read_text(encoding="utf-8")
    import ast, re
    match = re.search(r'CLASS_NAMES\s*=\s*(\[.*?\])', src, re.DOTALL)
    assert match, "Could not find CLASS_NAMES in demo/demo_pipeline.py"
    class_names = ast.literal_eval(match.group(1))
    assert class_names == EXPECTED_CLASS_ORDER, (
        f"demo CLASS_NAMES is {class_names!r}, expected {EXPECTED_CLASS_ORDER!r}."
    )


def test_class_names_length():
    """All CLASS_NAMES lists must have exactly 6 entries."""
    import inference_pipeline as ip
    assert len(ip.CLASS_NAMES) == 6


# ── Preprocessing ──────────────────────────────────────────────────────────────

def _make_dummy_pipeline():
    """Return a CashewPipeline with mocked models (no real .pt files needed)."""
    import inference_pipeline as ip

    with patch.object(ip, "YOLO", return_value=MagicMock()):
        with patch("torch.jit.load", return_value=MagicMock()):
            pipeline = ip.CashewPipeline.__new__(ip.CashewPipeline)
            pipeline.device = torch.device("cpu")
            pipeline._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            pipeline._std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return pipeline


def test_preprocess_crops_output_shape():
    """_preprocess_crops must return tensor [N, 3, 224, 224]."""
    import inference_pipeline as ip
    pipeline = _make_dummy_pipeline()

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    boxes = np.array([
        [10, 10, 100, 100],
        [200, 50, 350, 200],
        [400, 300, 600, 450],
    ], dtype=np.float32)

    batch = pipeline._preprocess_crops(image, boxes)
    assert batch.shape == (3, 3, 224, 224), f"Expected (3, 3, 224, 224), got {batch.shape}"


def test_preprocess_crops_dtype():
    """_preprocess_crops must return float32 tensor."""
    import inference_pipeline as ip
    pipeline = _make_dummy_pipeline()

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    boxes = np.array([[50, 50, 200, 200]], dtype=np.float32)

    batch = pipeline._preprocess_crops(image, boxes)
    assert batch.dtype == torch.float32


def test_preprocess_crops_normalized():
    """Output tensor values should be normalized (not in [0,1] raw range)."""
    import inference_pipeline as ip
    pipeline = _make_dummy_pipeline()

    # Solid white image — raw pixel 1.0, after ImageNet norm should be ~2.6
    image = np.full((300, 300, 3), 255, dtype=np.uint8)
    boxes = np.array([[10, 10, 290, 290]], dtype=np.float32)

    batch = pipeline._preprocess_crops(image, boxes)
    # After ImageNet normalization, white pixels should yield values > 1.0
    assert batch.max().item() > 1.0, "Tensor appears un-normalized (values still in [0,1])"


def test_preprocess_crops_skips_empty_bbox(caplog):
    """_preprocess_crops must skip (not crash) on zero-size boxes."""
    import inference_pipeline as ip
    import logging
    pipeline = _make_dummy_pipeline()

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Second box is degenerate (x1==x2, y1==y2)
    boxes = np.array([
        [50, 50, 200, 200],
        [100, 100, 100, 100],  # zero-area
        [300, 100, 500, 300],
    ], dtype=np.float32)

    with caplog.at_level(logging.WARNING, logger="inference_pipeline"):
        batch = pipeline._preprocess_crops(image, boxes)

    # Should produce 2 valid crops (skipping the zero-area one)
    assert batch.shape[0] == 2, f"Expected 2 crops, got {batch.shape[0]}"
    assert any("skipping" in r.message.lower() for r in caplog.records), (
        "Expected a warning log for the empty crop"
    )


# ── predict output schema ──────────────────────────────────────────────────────

def _make_mock_yolo_result(xyxy: torch.Tensor, conf: torch.Tensor):
    """Return a mock YOLO result whose boxes.xyxy / boxes.conf / len() are set correctly."""
    mock_box = MagicMock()
    mock_box.xyxy = xyxy
    mock_box.conf = conf
    mock_box.__len__ = MagicMock(return_value=len(xyxy))  # prevent early-exit on len() == 0
    mock_result = MagicMock()
    mock_result.boxes = mock_box
    return mock_result


def test_predict_returns_list_of_dicts():
    """predict() must return a list of dicts with required keys."""
    import inference_pipeline as ip

    pipeline = _make_dummy_pipeline()

    mock_result = _make_mock_yolo_result(
        xyxy=torch.tensor([[10., 10., 100., 100.], [200., 50., 350., 200.]]),
        conf=torch.tensor([0.9, 0.8]),
    )
    pipeline.yolo = MagicMock(return_value=[mock_result])

    # Fake ResNet returning logits for 2 crops
    fake_logits = torch.zeros(2, 6)
    fake_logits[0, 2] = 5.0   # loai1 wins for crop 0
    fake_logits[1, 5] = 5.0   # tb wins for crop 1
    pipeline.resnet = MagicMock(return_value=fake_logits)

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    results = pipeline.predict(image)

    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert "bbox" in r
        assert "class" in r
        assert "conf_detect" in r
        assert "conf_class" in r
        assert r["class"] in ip.CLASS_NAMES
        assert 0.0 <= r["conf_class"] <= 1.0


def test_predict_per_nut_classification():
    """Each nut must receive its own individual class (no majority vote)."""
    import inference_pipeline as ip

    pipeline = _make_dummy_pipeline()

    mock_result = _make_mock_yolo_result(
        xyxy=torch.tensor([[10., 10., 100., 100.], [200., 50., 350., 200.]]),
        conf=torch.tensor([0.9, 0.8]),
    )
    pipeline.yolo = MagicMock(return_value=[mock_result])

    # Nut 0 → bad_output (index 0), Nut 1 → tb (index 5)
    fake_logits = torch.zeros(2, 6)
    fake_logits[0, 0] = 5.0  # bad_output
    fake_logits[1, 5] = 5.0  # tb
    pipeline.resnet = MagicMock(return_value=fake_logits)

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    results = pipeline.predict(image)

    assert len(results) == 2
    assert results[0]["class"] == "bad_output", (
        f"Nut 0 should be 'bad_output', got {results[0]['class']!r}. "
        "Majority vote may still be active."
    )
    assert results[1]["class"] == "tb", (
        f"Nut 1 should be 'tb', got {results[1]['class']!r}. "
        "Majority vote may still be active."
    )


def test_predict_empty_image_returns_empty_list():
    """predict() must return [] when YOLO finds no detections."""
    import inference_pipeline as ip

    pipeline = _make_dummy_pipeline()

    mock_result = MagicMock()
    mock_result.boxes = None
    pipeline.yolo = MagicMock(return_value=[mock_result])

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    results = pipeline.predict(image)

    assert results == []
