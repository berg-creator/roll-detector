"""Логика блокировки реле (без компьютерного зрения).

Два состояния: UNLOCKED (транспортёр работает) и LOCKED (заблокирован).

UNLOCKED -> LOCKED: ролик находится в зоне непрерывно lock_after секунд (Y).
LOCKED -> UNLOCKED: в зоне нет ролика дольше unlock_after секунд (Z).

Детектор иногда пропускает ролик на 1-2 кадра (блик, рука, захват погрузчика).
Чтобы такой пропуск не сбрасывал отсчёт Y, короткие разрывы присутствия
не длиннее gap_tolerance секунд считаются продолжением присутствия.
Для разблокировки разрывы не прощаются: любое появление ролика
перезапускает отсчёт Z — ошибка в эту сторону безопаснее
(лишний раз постоять, чем толкнуть ролик в занятую зону).

Время передаётся снаружи (секунды, монотонно растущие), поэтому
логику можно проверять тестами без камеры и без ожидания.
"""
import logging

log = logging.getLogger("relay")


class RelayLogic:
    def __init__(self, lock_after, unlock_after, gap_tolerance=0.3, start_locked=True):
        if lock_after < 0 or unlock_after < 0 or gap_tolerance < 0:
            raise ValueError("времена не могут быть отрицательными")
        self.lock_after = lock_after
        self.unlock_after = unlock_after
        self.gap_tolerance = gap_tolerance
        self.locked = start_locked
        self._presence_start = None  # начало текущего непрерывного присутствия
        self._last_seen = None       # когда ролик был в зоне последний раз
        self._last_t = None

    def update(self, t, present, reason=""):
        """Подать результат очередного кадра. Возвращает True, если реле заблокировано."""
        if self._last_t is not None and t < self._last_t:
            raise ValueError(f"время пошло назад: {t} < {self._last_t}")
        if self._last_t is None and self.locked:
            # Старт в заблокированном состоянии: считаем, что ролик «был виден» сейчас,
            # и разблокируемся только после Z секунд пустой зоны.
            self._last_seen = t
        self._last_t = t

        if present:
            if self._presence_start is None or t - self._last_seen > self.gap_tolerance:
                self._presence_start = t
            self._last_seen = t
        elif self._presence_start is not None and t - self._last_seen > self.gap_tolerance:
            self._presence_start = None  # разрыв слишком длинный — присутствие закончилось

        if not self.locked:
            if present and t - self._presence_start >= self.lock_after:
                self._set(True, t, reason or f"ролик в зоне {t - self._presence_start:.2f} с")
        else:
            if t - self._last_seen >= self.unlock_after:
                self._presence_start = None  # после разблокировки отсчёт Y начинается заново
                self._set(False, t, reason or f"зона пуста {t - self._last_seen:.2f} с")
        return self.locked

    def force_lock(self, t, reason):
        """Аварийная блокировка (камера пропала и т.п.). Снять её можно только
        обычным путём — после Z секунд нормальной работы с пустой зоной."""
        self._last_t = t if self._last_t is None else max(self._last_t, t)
        self._last_seen = self._last_t
        self._presence_start = None
        if not self.locked:
            self._set(True, t, reason)

    def _set(self, locked, t, reason):
        self.locked = locked
        log.info("t=%.2f реле %s (%s)", t, "ЗАБЛОКИРОВАНО" if locked else "РАЗБЛОКИРОВАНО", reason)
