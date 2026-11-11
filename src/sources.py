"""Источники кадров: видеофайл и живая камера.

Оба отдают пары (t, frame), где t — время кадра в секундах.
Вся логика таймеров работает от этого t, поэтому одинаково ведёт себя
и на камере, и на записи (даже если запись обрабатывается быстрее
реального времени).
"""
import threading
import time

import cv2


class VideoFileSource:
    """Видеофайл. Время берётся из номера кадра, а не из часов.

    fps — с какой частотой брать кадры (камера на заводе 20 fps, записи 60 fps:
    при fps=20 берём каждый третий кадр, т.е. моделируем реальную камеру).
    """

    def __init__(self, path, fps=None):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"не открывается видео: {path}")
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.period = 1.0 / fps if fps else 0.0
        self.live = False

    def __iter__(self):
        idx = 0
        next_t = 0.0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            t = idx / self.video_fps
            idx += 1
            if t + 1e-9 < next_t:
                continue
            next_t += self.period
            yield t, frame
        self.cap.release()


class CameraSource:
    """Живая камера (номер устройства или rtsp/http-адрес).

    Кадры читаются в отдельном потоке, наружу отдаётся только самый свежий:
    если обработка кадра медленнее камеры, старые кадры выбрасываются,
    а не копятся в очереди (иначе решение принималось бы по картинке
    многосекундной давности).

    Если новых кадров нет дольше timeout секунд, итератор отдаёт (t, None) —
    управляющая программа по этому сигналу блокирует реле. Поток при этом
    пытается переподключиться к камере.
    """

    def __init__(self, src, timeout=1.0, reconnect_delay=2.0):
        self.src = int(src) if str(src).isdigit() else src
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay
        self.live = True
        self._frame = None
        self._t = None
        self._new = threading.Event()
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stop:
            cap = cv2.VideoCapture(self.src)
            while not self._stop and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                with self._lock:
                    self._frame, self._t = frame, time.monotonic()
                    self._new.set()
            cap.release()
            if not self._stop:
                time.sleep(self.reconnect_delay)

    def __iter__(self):
        while not self._stop:
            if not self._new.wait(self.timeout):
                yield time.monotonic(), None  # камера молчит
                continue
            with self._lock:
                t, frame = self._t, self._frame
                self._new.clear()
            yield t, frame

    def close(self):
        self._stop = True
