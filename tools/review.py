"""Контрольные листы разметки: кадры с нарисованными рамками, по 12 на лист.

    python tools/review.py --boxes data/labels/boxes.json --out review/

Каждый кадр подписан именем, чтобы ошибку можно было исправить в fixes.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", default="data/labels/boxes.json")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--out", default="review")
    ap.add_argument("--only", default="", help="только кадры, начинающиеся с этой строки")
    ap.add_argument("--per-sheet", type=int, default=12)
    args = ap.parse_args()

    boxes = json.loads(Path(args.boxes).read_text())
    names = [n for n in sorted(boxes) if n.startswith(args.only)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tw, th = 480, 270
    for s in range(0, len(names), args.per_sheet):
        tiles = []
        for name in names[s:s + args.per_sheet]:
            img = cv2.imread(str(Path(args.frames) / name))
            for b in boxes[name]:
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 4)
            img = cv2.resize(img, (tw, th))
            cv2.putText(img, name[:-4], (5, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(img, name[:-4], (5, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(img)
        while len(tiles) % 3:
            tiles.append(np.zeros((th, tw, 3), np.uint8))
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        cv2.imwrite(str(out / f"sheet_{s // args.per_sheet:03d}.jpg"), np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"{len(names)} кадров -> {out}")


if __name__ == "__main__":
    main()
