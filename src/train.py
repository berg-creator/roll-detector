"""Дообучение YOLO11n на одном классе «roll».

Берём модель, предобученную на COCO (там уже есть общие признаки:
края, текстуры, формы), и дообучаем на наших кадрах. С нуля на ~400
кадрах сеть не обучить, а дообучение сходится за десятки эпох.

    python src/train.py --data data/dataset/data.yaml --epochs 60
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset/data.yaml")
    ap.add_argument("--base", default="models/yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=480)
    # 8 укладывается в ~1.5 ГБ свободной памяти; на качество не влияет:
    # ultralytics копит градиенты до номинального батча 64 (nbs)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="roll")
    ap.add_argument("--out", default="models/roll_yolo11n.pt")
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=2,
        patience=15,       # остановиться, если 15 эпох подряд нет улучшения на val
        project="runs",
        name=args.name,
        exist_ok=True,
        seed=0,
        # аугментации: камера стоит неподвижно, поэтому сильные повороты и
        # перспектива не нужны; яркость/цвет — нужны (освещение в цехе меняется)
        degrees=0.0,
        perspective=0.0,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
    )
    best = Path(model.trainer.best)  # лучшая эпоха по метрике на val
    shutil.copy(best, args.out)
    print("лучшие веса:", args.out)

    # экспорт для работы: OpenVINO на CPU вдвое быстрее PyTorch (~27 против ~55 мс/кадр)
    # появится папка рядом с .pt: models/roll_yolo11n_openvino_model
    YOLO(args.out).export(format="openvino", imgsz=args.imgsz)


if __name__ == "__main__":
    main()
