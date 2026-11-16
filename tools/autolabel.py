"""Черновая авторазметка кадров zero-shot детектором Grounding DINO.

Модель по текстовому запросу ("a cardboard roll.") находит объекты без обучения.
Она медленная и ошибается, поэтому ее выход - только черновик:
дальше рамки фильтруются (tools/make_labels.py), проверяются глазами
на контрольных листах (tools/review.py) и правятся вручную.

Результат - JSON {имя_кадра: [[x1, y1, x2, y2, score], ...]} в пикселях.

    python tools/autolabel.py --frames data/frames --out data/labels/raw_gdino.json
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-base"
PROMPT = "a cardboard roll."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--out", default="data/labels/raw_gdino.json")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--short-side", type=int, default=480,
                    help="размер, до которого модель сжимает кадр (меньше = быстрее)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {}  # можно продолжить после обрыва

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    size = {"shortest_edge": args.short_side, "longest_edge": args.short_side * 16 // 9}
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).eval()

    frames = sorted(Path(args.frames).glob("*.jpg"))
    for i, path in enumerate(frames):
        if path.name in results:
            continue
        img = Image.open(path).convert("RGB")
        inputs = proc(images=img, text=PROMPT, size=size, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        det = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=args.threshold, text_threshold=args.threshold,
            target_sizes=[img.size[::-1]])[0]
        results[path.name] = [[round(v, 1) for v in box.tolist()] + [round(score.item(), 3)]
                              for box, score in zip(det["boxes"], det["scores"])]
        if i % 10 == 0:
            out.write_text(json.dumps(results, indent=0))
            print(f"{i + 1}/{len(frames)}", flush=True)
    out.write_text(json.dumps(results, indent=0))
    print("готово:", len(results))


if __name__ == "__main__":
    main()
