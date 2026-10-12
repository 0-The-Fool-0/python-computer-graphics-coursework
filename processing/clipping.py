from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Segment = Tuple[Point, Point]
Rect = Tuple[float, float, float, float]


def _ensure_rect(rect: Rect) -> Rect:
    xmin, ymin, xmax, ymax = rect
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    return xmin, ymin, xmax, ymax


def liang_barsky(segment: Segment, rect: Rect) -> Optional[Segment]:
    """
    Clip a segment by an axis-aligned rectangle using the Liang–Barsky algorithm.
    Returns the clipped segment or None if fully outside.
    """
    (x0, y0), (x1, y1) = segment
    xmin, ymin, xmax, ymax = _ensure_rect(rect)

    dx = x1 - x0
    dy = y1 - y0

    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]

    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
            continue
        t = qi / pi
        if pi < 0:
            if t > u1:
                return None
            u0 = max(u0, t)
        else:
            if t < u0:
                return None
            u1 = min(u1, t)
    if u0 > u1:
        return None
    clipped = ((x0 + u0 * dx, y0 + u0 * dy), (x0 + u1 * dx, y0 + u1 * dy))
    return clipped


def polygon_area(poly: Sequence[Point]) -> float:
    area = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        area += x0 * y1 - x1 * y0
    return area / 2.0


def _edge_normal(p0: Point, p1: Point, outward: bool) -> Point:
    ex = p1[0] - p0[0]
    ey = p1[1] - p0[1]
    if outward:
        return (ey, -ex)
    return (-ey, ex)


def clip_segment_convex_polygon(segment: Segment, polygon: Sequence[Point]) -> Optional[Segment]:
    """
    Cyrus–Beck style clipping of a segment by a convex polygon defined in either CW or CCW order.
    """
    (x0, y0), (x1, y1) = segment
    d = (x1 - x0, y1 - y0)
    t_enter, t_leave = 0.0, 1.0
    outward = polygon_area(polygon) > 0 
    for i in range(len(polygon)):
        pk = polygon[i]
        pk1 = polygon[(i + 1) % len(polygon)]
        normal = _edge_normal(pk, pk1, outward)
        w = (x0 - pk[0], y0 - pk[1])
        denom = normal[0] * d[0] + normal[1] * d[1]
        numer = -(normal[0] * w[0] + normal[1] * w[1])
        if denom == 0:
            if numer < 0:
                return None
            continue
        t = numer / denom
        if denom < 0:
            if t > t_leave:
                return None
            t_enter = max(t_enter, t)
        else:
            if t < t_enter:
                return None
            t_leave = min(t_leave, t)
        if t_enter > t_leave:
            return None
    clipped = (
        (x0 + t_enter * d[0], y0 + t_enter * d[1]),
        (x0 + t_leave * d[0], y0 + t_leave * d[1]),
    )
    return clipped


def clip_polygon_convex(subject: Sequence[Point], clipper: Sequence[Point]) -> List[Point]:
    """
    Sutherland–Hodgman polygon clipping for a convex clip window.
    """
    def inside(p: Point, edge_start: Point, edge_end: Point, outward_normal: Point) -> bool:
        return (p[0] - edge_start[0]) * outward_normal[0] + (p[1] - edge_start[1]) * outward_normal[1] >= 0

    def intersection(p1: Point, p2: Point, edge_start: Point, edge_end: Point, outward_normal: Point) -> Point:
        d = (p2[0] - p1[0], p2[1] - p1[1])
        w = (p1[0] - edge_start[0], p1[1] - edge_start[1])
        denom = outward_normal[0] * d[0] + outward_normal[1] * d[1]
        if denom == 0:
            return p2
        t = -(outward_normal[0] * w[0] + outward_normal[1] * w[1]) / denom
        return (p1[0] + t * d[0], p1[1] + t * d[1])

    if not subject:
        return []

    outward = polygon_area(clipper) > 0
    output = list(subject)
    for i in range(len(clipper)):
        edge_start = clipper[i]
        edge_end = clipper[(i + 1) % len(clipper)]
        normal = _edge_normal(edge_start, edge_end, outward)
        input_list = output
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if inside(e, edge_start, edge_end, normal):
                if not inside(s, edge_start, edge_end, normal):
                    output.append(intersection(s, e, edge_start, edge_end, normal))
                output.append(e)
            elif inside(s, edge_start, edge_end, normal):
                output.append(intersection(s, e, edge_start, edge_end, normal))
            s = e
    return output


def rect_to_polygon(rect: Rect) -> List[Point]:
    xmin, ymin, xmax, ymax = _ensure_rect(rect)
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def batch_clip_segments(segments: Iterable[Segment], polygon: Sequence[Point]) -> List[Segment]:
    clipped: List[Segment] = []
    for seg in segments:
        clipped_seg = clip_segment_convex_polygon(seg, polygon)
        if clipped_seg:
            clipped.append(clipped_seg)
    return clipped


def batch_liang_barsky(segments: Iterable[Segment], rect: Rect) -> List[Segment]:
    clipped: List[Segment] = []
    for seg in segments:
        clipped_seg = liang_barsky(seg, rect)
        if clipped_seg:
            clipped.append(clipped_seg)
    return clipped



