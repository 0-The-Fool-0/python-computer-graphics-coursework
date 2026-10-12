from __future__ import annotations

from typing import List, Tuple


Point = Tuple[int, int]


def line_step_by_step(p0: Point, p1: Point) -> List[Point]:
    x0, y0 = p0
    x1, y1 = p1
    points: List[Point] = []
    if x0 == x1:
        step = 1 if y1 >= y0 else -1
        for y in range(y0, y1 + step, step):
            points.append((x0, y))
        return points
    m = (y1 - y0) / (x1 - x0)
    b = y0 - m * x0
    step = 1 if x1 >= x0 else -1
    x = x0
    while True:
        y = round(m * x + b)
        points.append((x, y))
        if x == x1:
            break
        x += step
    return points


def line_dda(p0: Point, p1: Point) -> List[Point]:
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy)))
    if steps == 0:
        return [(x0, y0)]
    x_inc = dx / steps
    y_inc = dy / steps
    x = x0
    y = y0
    points: List[Point] = []
    for _ in range(steps + 1):
        points.append((round(x), round(y)))
        x += x_inc
        y += y_inc
    return points


def line_bresenham(p0: Point, p1: Point) -> List[Point]:
    x0, y0 = p0
    x1, y1 = p1
    points: List[Point] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def circle_bresenham(center: Point, radius: int) -> List[Point]:
    cx, cy = center
    x = 0
    y = radius
    d = 3 - 2 * radius
    points: List[Point] = []
    while y >= x:
        points.extend([
            (cx + x, cy + y), (cx - x, cy + y), (cx + x, cy - y), (cx - x, cy - y),
            (cx + y, cy + x), (cx - y, cy + x), (cx + y, cy - x), (cx - y, cy - x),
        ])
        if d > 0:
            d += 4 * (x - y) + 10
            y -= 1
        else:
            d += 4 * x + 6
        x += 1
    # Deduplicate while preserving order
    seen = set()
    uniq: List[Point] = []
    for pt in points:
        if pt not in seen:
            seen.add(pt)
            uniq.append(pt)
    return uniq

 


