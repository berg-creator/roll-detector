"""Черновые детекции Grounding DINO -> разметка YOLO + датасет.

Шаги:
1. отбрасываем рамки с низкой уверенностью;
2. убираем дубли (NMS) и "объединяющие" рамки, которые охватывают
   сразу несколько роликов - типичная ошибка zero-shot моделей;
3. накладываем ручные правки из data/labels/fixes.json - там, где
   при проверке глазами автомат ошибся, рамки кадра заданы вручную;
4. пишем YOLO-файлы и раскладываем кадры на train/val ПО ВИДЕО
   (кадры одного ролика почти одинаковые: если делить случайно,
   валидация будет мерить запоминание, а не обобщение).

Формат fixes.json: {"имя_кадра.jpg": [[x1, y1, x2, y2], ...]} - полный
список рамок кадра, кроме складских STATIC_BOXES (они добавляются всегда);
пустой список = на площадке роликов нет; null = кадр исключить из датасета
(когда честно не сказать, ролик это или нет - например, видна полоска в 20 px).

    python tools/make_labels.py --val demo_4 demo_6
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2

W, H = 1280, 720

# Складские ролики у дальней стены (верх кадра). Камера и штабель неподвижны,
# а zero-shot модель находит их то с уверенностью 0.6, то 0.2 - разметка
# "то есть, то нет" учит сеть противоречию. Поэтому эти рамки задаем
# постоянными и добавляем в каждый кадр, а автоматические рамки рядом убираем.
STATIC_BOXES = [
    [372, 0, 600, 157],  # передний ролик штабеля
    [278, 0, 393, 80],   # ролик за ним, виден частично
]


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def inside(inner, outer, tol=0.85):
    """Доля площади inner, лежащая внутри outer, не меньше tol."""
    iw = min(inner[2], outer[2]) - max(inner[0], outer[0])
    ih = min(inner[3], outer[3]) - max(inner[1], outer[1])
    if iw <= 0 or ih <= 0:
        return False
    return iw * ih >= tol * (inner[2] - inner[0]) * (inner[3] - inner[1])


# Области, где роликов не бывает, но есть похожие объекты: стопка плоских
# листов картона у стены (demo_8) - zero-shot модель принимает ее за ролик.
# Для детектора это важный "отрицательный" пример, поэтому рамки там убираем.
IGNORE_REGIONS = [
    [640, 0, 930, 160],
]

MAX_ROLL_WIDTH = 500  # px; самый широкий ролик на записях ~400 px


def is_edge_roll(d, edge_score):
    """Ролик, обрезанный левым/правым краем кадра: узкий, стоит на уровне пола.
    Модель дает таким пониженную уверенность (видна только часть), поэтому
    для них порог ниже. Левый край - это выход транспортера, пропускать их нельзя."""
    touches = d[0] < 8 or d[2] > W - 8
    return touches and d[4] >= edge_score and d[2] - d[0] < 350 and d[3] > 400 and d[3] - d[1] > 150


def clean(dets, min_score, edge_score=0.2):
    dets = [d for d in dets if d[2] - d[0] <= MAX_ROLL_WIDTH]  # "вся площадка" - частая ошибка
    dets = sorted((d for d in dets if d[4] >= min_score or is_edge_roll(d, edge_score)),
                  key=lambda d: -d[4])
    kept = []
    for d in dets:  # NMS
        if all(iou(d, k) < 0.5 for k in kept):
            kept.append(d)
    # рамка, внутри которой целиком лежат две другие, - это "группа", а не ролик
    return [d[:4] for d in kept if sum(inside(o, d) for o in kept if o is not d) < 2]


def to_yolo(box):
    x1, y1, x2, y2 = (max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3]))
    return f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/labels/raw_gdino.json")
    ap.add_argument("--fixes", default="data/labels/fixes.json")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--boxes", default="data/labels/boxes.json", help="куда записать итоговые рамки")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--val", nargs="+", default=["demo_4", "demo_6"], help="видео для валидации")
    args = ap.parse_args()

    raw = json.loads(Path(args.raw).read_text())
    fixes = json.loads(Path(args.fixes).read_text()) if Path(args.fixes).exists() else {}

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    boxes_all = {}
    for name, dets in sorted(raw.items()):
        if name in fixes and fixes[name] is None:
            continue  # спорный кадр (виден край ролика в пару десятков пикселей) - не берем
        if name in fixes:
            boxes = fixes[name]
        else:
            boxes = [b for b in clean(dets, args.min_score)
                     if all(iou(b, s) < 0.3 and not inside(b, s, 0.6) for s in STATIC_BOXES)
                     and not any(inside(b, r, 0.6) for r in IGNORE_REGIONS)]
        boxes = STATIC_BOXES + boxes
        boxes_all[name] = boxes
        split = "val" if any(name.startswith(v + "_") for v in args.val) else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(args.frames) / name, out / "images" / split / name)
        (out / "labels" / split / name.replace(".jpg", ".txt")).write_text(
            "".join(to_yolo(b) + "\n" for b in boxes))

    # итоговые рамки в пикселях - для проверки глазами (tools/review.py)
    Path(args.boxes).write_text(json.dumps(boxes_all, indent=0))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: roll\n")
    for split in ("train", "val"):
        imgs = list((out / "images" / split).glob("*.jpg"))
        n_boxes = sum(len(boxes_all[p.name]) for p in imgs)
        print(f"{split}: {len(imgs)} кадров, {n_boxes} рамок")


if __name__ == "__main__":
    main()
