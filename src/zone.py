"""Зона выхода из транспортёра и проверка «ролик в зоне».

Зона — прямоугольник в относительных координатах кадра (0..1):
    x от 0 до X (X — граница из задания, ролик должен быть левее неё),
    y от y_min до y_max (отсекает всё, что выше площадки, например
    складские ролики у дальней стены — они тоже левее X, но к выходу
    транспортёра отношения не имеют).

Ролик считается в зоне, если внутри зоны лежит не меньше min_overlap
от площади его рамки. Так ролик, который только краем задел линию X,
не считается, а ролик, наполовину въехавший в зону, — считается.
"""
from dataclasses import dataclass


@dataclass
class Zone:
    x_max: float = 0.4       # X из задания, доля ширины кадра
    y_min: float = 0.35      # доля высоты кадра
    y_max: float = 1.0
    min_overlap: float = 0.25

    def to_pixels(self, width, height):
        """Прямоугольник зоны в пикселях: (x1, y1, x2, y2)."""
        return (0.0, self.y_min * height, self.x_max * width, self.y_max * height)

    def overlap(self, box, width, height):
        """Доля площади рамки box=(x1, y1, x2, y2), попавшая в зону."""
        zx1, zy1, zx2, zy2 = self.to_pixels(width, height)
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            return 0.0
        iw = min(x2, zx2) - max(x1, zx1)
        ih = min(y2, zy2) - max(y1, zy1)
        if iw <= 0 or ih <= 0:
            return 0.0
        return iw * ih / area

    def contains(self, box, width, height):
        return self.overlap(box, width, height) >= self.min_overlap
