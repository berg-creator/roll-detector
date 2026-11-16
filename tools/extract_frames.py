"""Нарезка кадров из видео для разметки.

Берем каждый N-й кадр (соседние кадры при 60 fps почти одинаковые).
Если сцена почти не изменилась с последнего сохраненного кадра
(пустая площадка, ролик стоит), такой кадр сохраняем реже, чтобы
датасет не забивался копиями.

    python tools/extract_frames.py --step 12
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--out", default="data/frames")
    ap.add_argument("--step", type=int, default=12, help="брать каждый N-й кадр")
    ap.add_argument("--static-diff", type=float, default=3.0,
                    help="средняя разница пикселей, ниже которой кадр считается 'тем же'")
    ap.add_argument("--static-keep", type=int, default=5,
                    help="из подряд идущих 'тех же' кадров сохранять каждый K-й")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for video in sorted(Path(args.videos).glob("*.mp4")):
        cap = cv2.VideoCapture(str(video))
        last_small = None
        static_run = 0
        saved = 0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.step == 0:
                small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                same = last_small is not None and np.abs(small - last_small).mean() < args.static_diff
                static_run = static_run + 1 if same else 0
                if not same or static_run % args.static_keep == 0:
                    cv2.imwrite(str(out / f"{video.stem}_{idx:05d}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    last_small = small
                    saved += 1
            idx += 1
        cap.release()
        print(f"{video.name}: {idx} кадров, сохранено {saved}")


if __name__ == "__main__":
    main()
