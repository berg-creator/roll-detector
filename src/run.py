"""Управляющая программа: камера -> детектор -> зона -> логика -> виртуальное реле.

    # запись с завода, кадры берутся как с камеры 20 fps:
    python src/run.py --source data/videos/demo_1.mp4 --save out.mp4
    # живая камера:
    python src/run.py --source 0
    python src/run.py --source rtsp://user:pass@10.0.0.5/stream

Параметры X, Y, Z можно задать в config.yaml или флагами --x --y --z.
"""
import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import cv2
import yaml

from detector import RollDetector
from relay import RelayLogic
from sources import CameraSource, VideoFileSource
from zone import Zone

log = logging.getLogger("run")


class VirtualRelay:
    """Виртуальное реле. На реальном объекте здесь будет запись в ПЛК
    (Modbus/OPC UA) или дискретный выход; остальной код от этого не меняется."""

    def __init__(self, events_csv=None):
        self.state = None
        self._csv = None
        if events_csv:
            f = open(events_csv, "w", newline="")
            self._csv = (f, csv.writer(f))
            self._csv[1].writerow(["t", "state"])

    def set(self, locked, t):
        if locked == self.state:
            return
        self.state = locked
        print(f"[{t:8.2f} с] РЕЛЕ: {'БЛОКИРОВКА' if locked else 'разблокировано'}", flush=True)
        if self._csv:
            self._csv[1].writerow([f"{t:.3f}", int(locked)])
            self._csv[0].flush()


def draw(frame, zone, detections, in_zone_flags, locked, t):
    h, w = frame.shape[:2]
    zx1, zy1, zx2, zy2 = map(int, zone.to_pixels(w, h))
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 200, 0), 2)
    for d, inside in zip(detections, in_zone_flags):
        color = (0, 0, 255) if inside else (0, 255, 0)
        cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 3)
        cv2.putText(frame, f"{d.score:.2f}", (int(d.x1) + 4, int(d.y1) + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    label = "LOCKED" if locked else "UNLOCKED"
    cv2.rectangle(frame, (0, 0), (330, 50), (0, 0, 180) if locked else (0, 140, 0), -1)
    cv2.putText(frame, f"{label}  {t:6.1f}s", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="видеофайл, номер камеры или rtsp-адрес")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--x", type=float, help="X: граница зоны, доля ширины кадра")
    ap.add_argument("--y", type=float, help="Y: секунд в зоне до блокировки")
    ap.add_argument("--z", type=float, help="Z: секунд пустой зоны до разблокировки")
    ap.add_argument("--weights", help="путь к весам модели")
    ap.add_argument("--save", help="записать видео с разметкой")
    ap.add_argument("--events", help="CSV с переключениями реле")
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.x is not None:
        cfg["zone"]["x_max"] = args.x
    if args.y is not None:
        cfg["relay"]["lock_after"] = args.y
    if args.z is not None:
        cfg["relay"]["unlock_after"] = args.z
    if args.weights:
        cfg["model"]["weights"] = args.weights

    detector = RollDetector(**cfg["model"])
    zone = Zone(**cfg["zone"])
    logic = RelayLogic(**cfg["relay"])
    relay = VirtualRelay(args.events)

    is_file = Path(args.source).is_file()
    source = (VideoFileSource(args.source, fps=cfg["camera"]["fps"]) if is_file
              else CameraSource(args.source, timeout=cfg["camera"]["timeout"]))
    log.info("источник: %s, X=%.2f Y=%.1f c Z=%.1f c", args.source,
             zone.x_max, logic.lock_after, logic.unlock_after)

    writer = None
    n, busy = 0, 0.0
    try:
        for t, frame in source:
            if frame is None:  # камера молчит дольше timeout
                logic.force_lock(t, "нет кадров с камеры")
                relay.set(logic.locked, t)
                continue

            t0 = time.perf_counter()
            h, w = frame.shape[:2]
            try:
                detections = detector(frame)
            except Exception:  # битый кадр, сбой модели — не падаем, а блокируем
                log.exception("ошибка обработки кадра")
                logic.force_lock(t, "ошибка обработки кадра")
                relay.set(logic.locked, t)
                continue
            flags = [zone.contains(d.box, w, h) for d in detections]
            locked = logic.update(t, any(flags))
            relay.set(locked, t)
            busy += time.perf_counter() - t0
            n += 1

            if args.save:
                if writer is None:
                    fps = cfg["camera"]["fps"] if is_file else 20
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(draw(frame, zone, detections, flags, locked, t))
    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.release()
        if not is_file:
            source.close()
    if n:
        log.info("кадров: %d, среднее время обработки %.1f мс (%.1f кадр/с)",
                 n, 1000 * busy / n, n / busy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
