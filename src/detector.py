"""Обертка над обученной моделью YOLO: кадр -> список рамок роликов."""
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)


class RollDetector:
    def __init__(self, weights, conf=0.5, imgsz=480, device="cpu"):
        # weights может быть .pt или экспортированная модель (.onnx, папка openvino) -
        # ultralytics сам выберет движок по расширению.
        self.model = YOLO(weights, task="detect")
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        # прогрев: первый вызов в несколько раз медленнее (загрузка, компиляция графа),
        # пусть это случится при запуске, а не на первом кадре с камеры
        self(np.zeros((720, 1280, 3), dtype=np.uint8))

    def __call__(self, frame):
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                    device=self.device, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return [Detection(*map(float, b), float(s)) for b, s in zip(boxes, scores)]
