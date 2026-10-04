"""JSONL detection dataset loader."""
import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from visionx.model import encode


class JsonlDetectionDataset(Dataset):
    def __init__(self, manifest, class_names, image_size=320, grid_size=None):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.records = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.classes, self.image_size = list(class_names), image_size
        self.grid_size = grid_size or max(1, image_size // 32)
        for n, rec in enumerate(self.records, 1):
            if not (self.root / rec["image"]).is_file():
                raise FileNotFoundError(f"{self.manifest}:{n}: missing image {rec['image']}")
            for box in rec.get("boxes", []):
                if not 0 <= int(box["class_id"]) < len(self.classes):
                    raise ValueError(f"{self.manifest}:{n}: class_id outside classes.txt")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(self.root / rec["image"]).convert("RGB")
        ow, oh = image.size
        image = image.resize((self.image_size, self.image_size))
        scale_x, scale_y = self.image_size / ow, self.image_size / oh
        boxes = [(int(b["class_id"]), b["x"]*scale_x, b["y"]*scale_y,
                  b["width"]*scale_x, b["height"]*scale_y) for b in rec.get("boxes", [])]
        x = torch.from_numpy(__import__("numpy").array(image).copy()).permute(2,0,1).float().div_(255)
        return x, encode(boxes, self.classes, self.grid_size, self.image_size)
