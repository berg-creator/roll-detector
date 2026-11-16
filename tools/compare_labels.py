"""Сверка разметки с предсказаниями обученной модели (второй проход разметки).

Модель первой итерации, обученная на черновой разметке, часто находит то,
что пропустил zero-shot разметчик (например, ролик в тени внутри машины).
Скрипт находит кадры, где разметка и модель расходятся, и рисует их
(зеленый - разметка, красный - модель). Каждое расхождение решается
глазами; принятые исправления записываются в data/labels/fixes.json.

    python tools/compare_labels.py --weights runs/.../best.pt --out review_diff/
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from make_labels import iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--boxes", default="data/labels/boxes.json")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--out", default="review_diff")
    args = ap.parse_args()

    model = YOLO(args.weights)
    labels = json.loads(Path(args.boxes).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    diffs = {}
    for name, gt in sorted(labels.items()):
        r = model.predict(str(Path(args.frames) / name), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        pred = [[round(float(v)) for v in b] + [round(float(s), 2)] for b, s in zip(r.boxes.xyxy, r.boxes.conf)]
        missed = [g for g in gt if all(iou(g, p) < 0.5 for p in pred)]
        extra = [p for p in pred if all(iou(p, g) < 0.5 for g in gt)]
        if missed or extra:
            diffs[name] = {"labels": gt, "pred": pred, "missed": missed, "extra": extra}
    (out / "diffs.json").write_text(json.dumps(diffs, indent=1))

    tiles = []
    for name, d in diffs.items():
        img = cv2.imread(str(Path(args.frames) / name))
        for b in d["labels"]:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 4)
        for p in d["extra"]:
            cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 4)
            cv2.putText(img, str(p[4]), (min(p[0] + 5, 1150), p[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        img = cv2.resize(img, (480, 270))
        cv2.putText(img, name[:-4], (5, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(img)
    while tiles and len(tiles) % 12:
        tiles.append(np.zeros_like(tiles[0]))
    for s in range(0, len(tiles), 12):
        sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(s, s + 12, 3)])
        cv2.imwrite(str(out / f"diff_{s // 12:03d}.jpg"), sheet)
    print(f"расхождений: {len(diffs)} -> {out}")


if __name__ == "__main__":
    main()
