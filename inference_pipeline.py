"""
inference_pipeline.py
---------------------
Pipeline kết hợp YOLO + ResNet-50 để phát hiện và phân loại hạt điều.

Luồng xử lý tối ưu (độ trễ thấp):
    Input Image
        ↓
    YOLO (FP16, single-class)  →  danh sách bounding boxes
        ↓
    Crop tất cả bboxes → batch tensor [N, 3, 224, 224]
        ↓
    ResNet-50 (FP16, batch inference)  →  [N, 6] predictions
        ↓
    Output: danh sách {bbox, class, confidence}

Tối ưu:
  - torch.inference_mode()   : tắt autograd graph
  - Batch ResNet crops       : 1 forward pass cho N hạt
  - FP16 mixed precision     : giảm ~50% memory, tăng throughput
  - TorchScript models       : loại bỏ Python overhead
  - YOLO verbose=False       : bỏ print logging

Chạy:
    python inference_pipeline.py --image path/to/image.jpg
    python inference_pipeline.py --image path/to/image.jpg --show
    python inference_pipeline.py --benchmark  (đo latency 100 frame)
"""

import argparse
import logging
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from ultralytics import YOLO

# ── Cấu hình ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DEFAULT_YOLO_PATH   = BASE_DIR / "runs" / "yolo" / "cashew_detect" / "weights" / "best.pt"

# EfficientNet-B3 (mới, tốt hơn) — fallback về ResNet nếu không tồn tại
_EFFNET_PATH = BASE_DIR / "efficientnet_cashew.pt"
_RESNET_PATH = BASE_DIR / "resnet50_cashew.pt"
DEFAULT_CLASSIFIER_PATH = _EFFNET_PATH if _EFFNET_PATH.exists() else _RESNET_PATH
DEFAULT_CLASSIFIER_SIZE = 300 if _EFFNET_PATH.exists() else 224

CLASS_NAMES = ["bad_output", "lbw", "loai1", "loai2", "loai3", "tb"]
CLASS_COLORS = {
    "tb":         (0, 200, 0),    # Xanh lá - Tốt bán
    "loai1":      (0, 140, 255),  # Cam - Loại 1
    "loai2":      (0, 200, 255),  # Vàng - Loại 2
    "loai3":      (200, 200, 0),  # Cyan đậm - Loại 3
    "lbw":        (200, 0, 200),  # Tím - LBW
    "bad_output": (0, 0, 200),    # Đỏ - Bad
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

YOLO_CONF_THRESH = 0.25
MIN_BBOX_SIDE    = 30    # px — lọc edge strip (bbox quá nhỏ ở 1 chiều)
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


class CashewPipeline:
    def __init__(
        self,
        yolo_path: str = str(DEFAULT_YOLO_PATH),
        classifier_path: str = str(DEFAULT_CLASSIFIER_PATH),
        classifier_size: int = DEFAULT_CLASSIFIER_SIZE,
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.classifier_size = classifier_size

        logger.info("Device: %s", self.device)
        logger.info("Loading YOLO: %s", yolo_path)
        self.yolo = YOLO(yolo_path)

        logger.info("Loading classifier: %s (input=%dpx)", classifier_path, classifier_size)
        self.resnet = torch.jit.load(classifier_path, map_location=self.device)
        self.resnet.eval()

        # Pre-compute normalize tensors trên device
        self._mean = IMAGENET_MEAN.view(3, 1, 1).to(self.device)
        self._std  = IMAGENET_STD.view(3, 1, 1).to(self.device)

        # Warmup
        self._warmup()
        logger.info("Pipeline ready")

    def _warmup(self):
        """Warmup để GPU JIT-compile trước khi inference thực."""
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.yolo(dummy_img, verbose=False)
        dummy_batch = torch.zeros(1, 3, self.classifier_size, self.classifier_size).to(self.device)
        with torch.inference_mode():
            self.resnet(dummy_batch)

    def _preprocess_crops(self, image_rgb: np.ndarray, boxes: np.ndarray) -> torch.Tensor:
        """
        Crop + resize + normalize tất cả bboxes thành 1 batch tensor.
        Args:
            image_rgb: [H, W, 3] uint8 RGB
            boxes: [N, 4] xyxy format
        Returns:
            tensor [N, 3, 224, 224] trên self.device
        """
        h, w = image_rgb.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes:
            # Clamp bbox vào trong ảnh
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            crop = image_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                logger.warning("Empty crop from bbox (%d,%d,%d,%d) — skipping", x1, y1, x2, y2)
                continue

            # Resize về 224x224
            crop_pil = Image.fromarray(crop)
            crop_pil = crop_pil.resize((self.classifier_size, self.classifier_size), Image.Resampling.BILINEAR)

            # To tensor [3, 224, 224], float32 [0,1]
            t = TF.to_tensor(crop_pil)
            crops.append(t)

        # Stack → [N, 3, 224, 224]
        batch = torch.stack(crops).to(self.device)

        # Normalize (broadcast)
        batch = (batch - self._mean) / self._std
        return batch

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> list[dict]:
        """
        Phát hiện và phân loại hạt điều trong ảnh.

        Args:
            image: numpy array [H, W, 3] BGR (OpenCV format)

        Returns:
            list of {
                "bbox":  [x1, y1, x2, y2],
                "class": str,
                "conf_detect": float,   # YOLO confidence
                "conf_class":  float,   # ResNet confidence
            }
        """
        # ── Step 1: YOLO Detection ─────────────────────────────────────────
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.yolo(
            image_rgb,
            half=(self.device.type == "cuda"),
            conf=YOLO_CONF_THRESH,
            verbose=False,
        )[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()           # [N, 4]
        detect_confs = boxes.conf.cpu().numpy()   # [N]

        # Lọc edge strip — bbox có chiều nào < MIN_BBOX_SIDE là nhiễu YOLO
        keep = (xyxy[:, 2] - xyxy[:, 0] >= MIN_BBOX_SIDE) & \
               (xyxy[:, 3] - xyxy[:, 1] >= MIN_BBOX_SIDE)
        xyxy = xyxy[keep]
        detect_confs = detect_confs[keep]
        if len(xyxy) == 0:
            return []

        # ── Step 2: Crop + Batch Preprocess ───────────────────────────────
        batch = self._preprocess_crops(image_rgb, xyxy)  # [N, 3, 224, 224]

        # ── Step 3: ResNet Batch Inference ────────────────────────────────
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda"):
                logits = self.resnet(batch)   # [N, 6]
        else:
            logits = self.resnet(batch)

        probs = torch.softmax(logits, dim=1)
        class_ids   = probs.argmax(dim=1).cpu().numpy()
        class_confs = probs.max(dim=1).values.cpu().numpy()

        # ── Step 4: Assemble Results ──────────────────────────────────────
        all_probs = probs.cpu().numpy()   # [N, 6]
        return [
            {
                "bbox":        xyxy[i].tolist(),
                "class":       CLASS_NAMES[class_ids[i]],
                "conf_detect": float(detect_confs[i]),
                "conf_class":  float(class_confs[i]),
                "probs":       {CLASS_NAMES[j]: float(all_probs[i, j]) for j in range(len(CLASS_NAMES))},
            }
            for i in range(len(xyxy))
        ]

    def draw(self, image: np.ndarray, predictions: list[dict]) -> np.ndarray:
        """Vẽ bounding boxes + labels lên ảnh."""
        out = image.copy()
        for pred in predictions:
            x1, y1, x2, y2 = [int(v) for v in pred["bbox"]]
            cls   = pred["class"]
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            label = f"{cls} {pred['conf_class']:.2f}"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return out


def run_benchmark(pipeline: CashewPipeline, n_frames: int = 100):
    """Đo latency trung bình trên n_frames ảnh giả."""
    print(f"\n[Benchmark] {n_frames} frames (640x640)...")
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    latencies = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        pipeline.predict(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = sorted(latencies)
    print(f"  Avg     : {sum(latencies)/len(latencies):.1f} ms")
    print(f"  P50     : {latencies[len(latencies)//2]:.1f} ms")
    print(f"  P95     : {latencies[int(len(latencies)*0.95)]:.1f} ms")
    print(f"  FPS     : {1000 / (sum(latencies)/len(latencies)):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Cashew Nut Detection + Classification")
    parser.add_argument("--image", type=str, help="Đường dẫn ảnh đầu vào")
    parser.add_argument("--yolo",       default=str(DEFAULT_YOLO_PATH))
    parser.add_argument("--classifier", default=str(DEFAULT_CLASSIFIER_PATH),
                        help="Path to TorchScript classifier (.pt)")
    parser.add_argument("--input_size", type=int, default=DEFAULT_CLASSIFIER_SIZE,
                        help="Classifier input size (224 for ResNet, 300 for EfficientNet)")
    parser.add_argument("--resnet", default=None, help="(legacy) alias for --classifier")
    parser.add_argument("--device", default=None)
    parser.add_argument("--show",   action="store_true", help="Hiển thị kết quả")
    parser.add_argument("--output", type=str, help="Lưu ảnh kết quả ra file")
    parser.add_argument("--benchmark", action="store_true", help="Đo latency")
    args = parser.parse_args()

    print("=" * 60)
    print("  Cashew Pipeline: YOLO + ResNet-50")
    print("=" * 60)

    classifier = args.resnet or args.classifier
    pipeline = CashewPipeline(args.yolo, classifier, args.input_size, args.device)

    if args.benchmark:
        run_benchmark(pipeline)
        return

    if not args.image:
        parser.error("Cần --image hoặc --benchmark")

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")

    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {img_path}")

    # Inference
    t0 = time.perf_counter()
    predictions = pipeline.predict(image)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # In kết quả
    print(f"  Thời gian: {elapsed_ms:.1f} ms")
    print(f"  Số hạt  : {len(predictions)}")
    print()
    for i, pred in enumerate(predictions):
        bbox = [int(v) for v in pred["bbox"]]
        print(f"  [{i+1}] {pred['class']:12s} | "
              f"detect={pred['conf_detect']:.2f} | "
              f"class={pred['conf_class']:.2f} | "
              f"bbox={bbox}")

    # Thống kê theo class
    if predictions:
        counts = Counter(p["class"] for p in predictions)
        print("\n  Tổng hợp:")
        for cls, count in sorted(counts.items()):
            print(f"    {cls:15s}: {count} hạt")

    # Visualize
    if args.show or args.output:
        result_img = pipeline.draw(image, predictions)
        if args.output:
            cv2.imwrite(args.output, result_img)
            print(f"\n  Đã lưu: {args.output}")
        if args.show:
            cv2.imshow("Cashew Detection", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
