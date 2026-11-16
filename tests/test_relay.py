"""Тесты логики реле и зоны. Камера и модель не нужны: подаем (время, есть ролик)."""
import pytest

from relay import RelayLogic
from zone import Zone

FPS = 20
DT = 1 / FPS


def feed(logic, pattern, t0=0.0):
    """pattern - список (длительность_с, есть_ролик). Возвращает (время, состояние) на каждом кадре."""
    out, t = [], t0
    for duration, present in pattern:
        for _ in range(round(duration * FPS)):
            out.append((t, logic.update(t, present)))
            t += DT
    return out


def first_change(trace, to_state):
    return next(t for t, s in trace if s == to_state)


def test_lock_after_y_seconds():
    logic = RelayLogic(lock_after=3, unlock_after=5, start_locked=False)
    trace = feed(logic, [(1, False), (5, True)])
    assert first_change(trace, True) == pytest.approx(1 + 3, abs=DT)


def test_thresholds_exact_with_float_time():
    # время как у VideoFileSource: номер кадра / 60 fps, каждый 3-й кадр (20 fps);
    # 9.7 - 6.7 в float меньше 3 - блокировка все равно должна быть ровно на 9.7
    logic = RelayLogic(lock_after=3, unlock_after=5, start_locked=False)
    events = []
    for i in range(0, 1200, 3):
        t = i / 60
        was = logic.locked
        if logic.update(t, 402 <= i < 627) != was:  # ролик в зоне 6.7–10.4 с
            events.append(round(t, 2))
    assert events == [9.7, 15.4]  # 6.7 + Y; последний кадр с роликом 10.4 + Z


def test_short_presence_does_not_lock():
    logic = RelayLogic(lock_after=3, unlock_after=5, start_locked=False)
    trace = feed(logic, [(2.9, True), (1, False), (2.9, True)])
    assert not any(s for _, s in trace)


def test_detection_dropouts_do_not_reset_y():
    # ролик стоит, но детектор теряет его на 1-2 кадра каждые полсекунды
    logic = RelayLogic(lock_after=3, unlock_after=5, gap_tolerance=0.3, start_locked=False)
    pattern = [(0.4, True), (0.1, False)] * 10
    trace = feed(logic, pattern)
    assert first_change(trace, True) == pytest.approx(3, abs=0.15)


def test_long_gap_resets_y():
    logic = RelayLogic(lock_after=3, unlock_after=5, gap_tolerance=0.3, start_locked=False)
    trace = feed(logic, [(2, True), (0.5, False), (2, True)])
    assert not any(s for _, s in trace)


def test_unlock_after_z_seconds_of_empty_zone():
    logic = RelayLogic(lock_after=1, unlock_after=5, start_locked=False)
    trace = feed(logic, [(2, True), (10, False)])
    last_seen = 2 - DT
    assert first_change(trace, True) == pytest.approx(1, abs=DT)
    t_unlock = next(t for t, s in trace if t > 2 and not s)
    assert t_unlock == pytest.approx(last_seen + 5, abs=DT)


def test_single_detection_restarts_z():
    logic = RelayLogic(lock_after=1, unlock_after=5, start_locked=False)
    trace = feed(logic, [(2, True), (4, False), (DT, True), (4.5, False)])
    assert all(s for t, s in trace if t >= 1)  # так и не разблокировалось


def test_starts_locked_and_unlocks_after_z():
    logic = RelayLogic(lock_after=3, unlock_after=5, start_locked=True)
    trace = feed(logic, [(10, False)])
    assert trace[0][1] is True
    assert first_change(trace, False) == pytest.approx(5, abs=DT)


def test_force_lock_then_unlock_only_after_z():
    logic = RelayLogic(lock_after=3, unlock_after=5, start_locked=False)
    feed(logic, [(1, False)])
    logic.force_lock(1.0, "нет кадров")
    assert logic.locked
    trace = feed(logic, [(10, False)], t0=1.0 + DT)
    assert first_change(trace, False) == pytest.approx(6, abs=DT)


def test_relock_after_unlock_needs_full_y():
    logic = RelayLogic(lock_after=3, unlock_after=1, start_locked=False)
    trace = feed(logic, [(4, True), (2, False), (2.5, True)])
    assert trace[-1][1] is False


def test_time_going_back_is_error():
    logic = RelayLogic(lock_after=1, unlock_after=1)
    logic.update(5.0, False)
    with pytest.raises(ValueError):
        logic.update(4.0, False)


def test_zone_overlap():
    z = Zone(x_max=0.5, y_min=0.5, y_max=1.0, min_overlap=0.25)
    w, h = 1000, 1000
    assert z.contains((100, 600, 300, 900), w, h)        # целиком в зоне
    assert not z.contains((600, 600, 900, 900), w, h)    # правее X
    assert not z.contains((100, 0, 300, 300), w, h)      # выше зоны (склад)
    assert z.overlap((400, 600, 600, 900), w, h) == pytest.approx(0.5)
    assert not z.contains((480, 600, 680, 900), w, h)    # заходит в зону на 10%
