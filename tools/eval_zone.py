"""Оценка на уровне задачи: правильно ли система решает "есть ролик в зоне".

mAP детектора - полезная, но косвенная метрика: реле важны не точные рамки,
а бинарный ответ по каждому кадру. Здесь для каждого размеченного кадра
сравниваем ответ "в зоне есть ролик" по разметке и по модели.

    python tools/eval_zone.py --weights models/roll_yolo11n.pt
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from detector import RollDetector  # noqa: E402
from zone import Zone  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/roll_yolo11n.pt")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--boxes", default="data/labels/boxes.json")
    ap.add_argument("--dataset", default="data/dataset")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    zone = Zone(**cfg["zone"])
    det = RollDetector(args.weights, conf=cfg["model"]["conf"], imgsz=cfg["model"]["imgsz"])
    gt_boxes = json.loads(Path(args.boxes).read_text())

    stats = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    errors = []
    for split in ("train", "val"):
        for img_path in sorted((Path(args.dataset) / "images" / split).glob("*.jpg")):
            frame = cv2.imread(str(img_path))
            h, w = frame.shape[:2]
            gt = any(zone.contains(b, w, h) for b in gt_boxes[img_path.name])
            pred = any(zone.contains(d.box, w, h) for d in det(frame))
            key = "tp" if gt and pred else "tn" if not gt and not pred else "fp" if pred else "fn"
            stats[split][key] += 1
            if gt != pred:
                errors.append((split, img_path.name, "лишний" if pred else "пропуск"))

    for split, s in stats.items():
        n = sum(s.values())
        print(f"{split}: кадров {n}, верно {s['tp'] + s['tn']} ({(s['tp'] + s['tn']) / n:.1%}); "
              f"ролик в зоне: {s['tp'] + s['fn']}, пропусков {s['fn']}; "
              f"пустая зона: {s['tn'] + s['fp']}, ложных {s['fp']}")
    for e in errors:
        print("  ошибка:", *e)


if __name__ == "__main__":
    main()
