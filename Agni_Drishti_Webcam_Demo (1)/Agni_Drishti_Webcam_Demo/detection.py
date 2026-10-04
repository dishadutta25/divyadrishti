"""Detector contract and checkpoint-backed custom detector."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol, Sequence, Tuple


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    # (x, y, width, height) in source-frame pixels
    bounding_box: Tuple[float, float, float, float]
    timestamp: float


class DetectorInterface(Protocol):
    def detect(self, image, timestamp: float = 0.0) -> List[Detection]: ...


class CustomGridDetector:
    """Runs the local, project-owned grid detector; never downloads weights."""
    def __init__(self, checkpoint: str, class_names: Sequence[str], confidence: float = 0.35,
                 nms_iou: float = 0.45, device: str = "cpu"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is needed for detector inference; no remote fallback is used") from exc
        from visionx.model import GridDetector, decode
        self.torch = torch
        self.classes = list(class_names)
        self.confidence, self.nms_iou = confidence, nms_iou
        self.device = torch.device(device)
        state = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        if state.get("class_names") != self.classes:
            raise ValueError("Checkpoint class list does not match configured class list")
        self.size, self.grid = int(state["image_size"]), int(state["grid_size"])
        self.model = GridDetector(len(self.classes), self.grid).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

    def detect(self, image, timestamp: float = 0.0) -> List[Detection]:
        import numpy as np
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype("uint8"))
        w, h = image.size
        resized = image.convert("RGB").resize((self.size, self.size))
        tensor = self.torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
        with self.torch.no_grad():
            pred = self.model(tensor.to(self.device))[0].cpu()
        items = decode(pred, self.classes, w, h, self.confidence)
        items = _nms(items, self.nms_iou)
        return [Detection(i, self.classes[i], float(score), (x, y, bw, bh), timestamp)
                for i, score, x, y, bw, bh in items]


def _nms(items, threshold):
    """Small dependency-free class-aware NMS. Items are (class, score, x,y,w,h)."""
    kept = []
    for cls in sorted({it[0] for it in items}):
        pending = sorted((it for it in items if it[0] == cls), key=lambda x: x[1], reverse=True)
        while pending:
            top = pending.pop(0); kept.append(top)
            def iou(a, b):
                ix1, iy1 = max(a[2], b[2]), max(a[3], b[3])
                ix2, iy2 = min(a[2]+a[4], b[2]+b[4]), min(a[3]+a[5], b[3]+b[5])
                inter = max(0, ix2-ix1)*max(0, iy2-iy1)
                return inter / max(1e-9, a[4]*a[5]+b[4]*b[5]-inter)
            pending = [x for x in pending if iou(top, x) <= threshold]
    return sorted(kept, key=lambda x: x[1], reverse=True)
