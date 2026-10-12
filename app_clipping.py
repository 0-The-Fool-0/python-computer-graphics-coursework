from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import streamlit as st
from PIL import Image, ImageDraw

from processing.clipping import (
    Point,
    Segment,
    Rect,
    batch_liang_barsky,
    batch_clip_segments,
    clip_polygon_convex,
    rect_to_polygon,
)


st.set_page_config(page_title="Отсечение отрезков и многоугольников", layout="wide")

BACKGROUND = (25, 27, 31)
GRID = (55, 58, 64)
AXIS = (230, 230, 235)
SEGMENT_COLOR = (200, 200, 200)
WINDOW_COLOR = (255, 165, 0)
CLIPPED_COLOR = (67, 220, 140)
POLYGON_FILL = (67, 116, 220, 60)
SUBJECT_COLOR = (180, 100, 255, 90)


@dataclass
class Viewport:
    min_x: float
    min_y: float
    span_x: float
    span_y: float
    scale: float
    pad_x: float
    pad_y: float
    width: int
    height: int
    margin: int

    def to_screen(self, pt: Point) -> Tuple[int, int]:
        sx = self.margin + self.pad_x + (pt[0] - self.min_x) * self.scale
        sy = self.height - (self.margin + self.pad_y + (pt[1] - self.min_y) * self.scale)
        return int(round(sx)), int(round(sy))


def compute_viewport(points: Sequence[Point], width: int, height: int, margin: int = 60) -> Viewport:
    if not points:
        points = [(0.0, 0.0), (10.0, 10.0)]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    if min_x == max_x:
        min_x -= 1
        max_x += 1
    if min_y == max_y:
        min_y -= 1
        max_y += 1
    span_x = max_x - min_x
    span_y = max_y - min_y
    usable_w = max(1, width - 2 * margin)
    usable_h = max(1, height - 2 * margin)
    scale_x = usable_w / span_x
    scale_y = usable_h / span_y
    scale = min(scale_x, scale_y)
    span_px_x = span_x * scale
    span_px_y = span_y * scale
    pad_x = (usable_w - span_px_x) / 2
    pad_y = (usable_h - span_px_y) / 2
    return Viewport(min_x, min_y, span_x, span_y, scale, pad_x, pad_y, width, height, margin)


def draw_grid(canvas: Image.Image, viewport: Viewport) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(0, 0), canvas.size], fill=BACKGROUND)
    min_x = math.floor(viewport.min_x)
    max_x = math.ceil(viewport.min_x + viewport.span_x)
    min_y = math.floor(viewport.min_y)
    max_y = math.ceil(viewport.min_y + viewport.span_y)

    for x in range(min_x, max_x + 1):
        sx, _ = viewport.to_screen((x, viewport.min_y))
        draw.line([(sx, viewport.margin // 2), (sx, viewport.height - viewport.margin // 2)], fill=GRID)
        draw.text((sx + 2, viewport.height - viewport.margin + 5), str(x), fill=GRID)

    for y in range(min_y, max_y + 1):
        _, sy = viewport.to_screen((viewport.min_x, y))
        draw.line([(viewport.margin // 2, sy), (viewport.width - viewport.margin // 2, sy)], fill=GRID)
        draw.text((5, sy - 12), str(y), fill=GRID)

    # axes
    if viewport.min_y <= 0 <= viewport.min_y + viewport.span_y:
        y_axis = [viewport.to_screen((x, 0)) for x in (viewport.min_x, viewport.min_x + viewport.span_x)]
        draw.line(y_axis, fill=AXIS, width=2)
    if viewport.min_x <= 0 <= viewport.min_x + viewport.span_x:
        x_axis = [viewport.to_screen((0, y)) for y in (viewport.min_y, viewport.min_y + viewport.span_y)]
        draw.line(x_axis, fill=AXIS, width=2)


def draw_polygon(canvas: Image.Image, polygon: Sequence[Point], viewport: Viewport, outline: Tuple[int, int, int], fill: Optional[Tuple[int, int, int, int]] = None) -> None:
    if not polygon:
        return
    draw = ImageDraw.Draw(canvas)
    pts = [viewport.to_screen(p) for p in polygon]
    if fill:
        fill_img = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        fill_draw = ImageDraw.Draw(fill_img)
        fill_draw.polygon(pts, fill=fill)
        canvas.alpha_composite(fill_img)
    draw.line(pts + [pts[0]], fill=outline, width=2)


def draw_segments(draw: ImageDraw.ImageDraw, segments: Sequence[Segment], viewport: Viewport, color: Tuple[int, int, int], width: int = 2) -> None:
    for p0, p1 in segments:
        draw.line([viewport.to_screen(p0), viewport.to_screen(p1)], fill=color, width=width)


def collect_points(*collections: Iterable[Point]) -> List[Point]:
    pts: List[Point] = []
    for col in collections:
        pts.extend(col)
    return pts


def parse_segments(text: str) -> List[Segment]:
    segments: List[Segment] = []
    for line in text.strip().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) == 0:
            continue
        if len(parts) != 4:
            raise ValueError(f"Строка '{line}' должна содержать 4 числа (x0 y0 x1 y1)")
        x0, y0, x1, y1 = map(float, parts)
        segments.append(((x0, y0), (x1, y1)))
    return segments


def parse_points(text: str) -> List[Point]:
    pts: List[Point] = []
    for line in text.strip().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) == 0:
            continue
        if len(parts) != 2:
            raise ValueError(f"Строка '{line}' должна содержать 2 числа (x y)")
        x, y = map(float, parts)
        pts.append((x, y))
    return pts


def parse_input_file(text: str) -> Tuple[List[Segment], Rect]:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Пустой файл")
    n = int(lines[0])
    if len(lines) < n + 2:
        raise ValueError("Недостаточно строк с отрезками")
    segments = []
    for i in range(1, n + 1):
        x0, y0, x1, y1 = map(float, lines[i].split())
        segments.append(((x0, y0), (x1, y1)))
    xmin, ymin, xmax, ymax = map(float, lines[n + 1].split())
    return segments, (xmin, ymin, xmax, ymax)


def render_scene(width: int, height: int, base_segments: Sequence[Segment], clipped_segments: Sequence[Segment], clip_polygon: Sequence[Point], subject_polygon: Optional[Sequence[Point]] = None) -> Image.Image:
    all_points = collect_points(
        [pt for seg in base_segments for pt in seg],
        [pt for seg in clipped_segments for pt in seg],
        clip_polygon,
        subject_polygon or [],
    )
    viewport = compute_viewport(all_points, width, height)
    canvas = Image.new("RGBA", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw_grid(canvas, viewport)
    if subject_polygon:
        draw_polygon(canvas, subject_polygon, viewport, outline=(180, 100, 255), fill=SUBJECT_COLOR)
    draw_polygon(canvas, clip_polygon, viewport, outline=WINDOW_COLOR, fill=POLYGON_FILL)
    draw_segments(draw, base_segments, viewport, SEGMENT_COLOR, width=2)
    draw_segments(draw, clipped_segments, viewport, CLIPPED_COLOR, width=4)
    return canvas.convert("RGB")


def canvas_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def sidebar_controls():
    with st.sidebar:
        st.header("Параметры сцены")
        width = st.slider("Ширина полотна", 500, 1400, 900, 10)
        height = st.slider("Высота полотна", 400, 900, 600, 10)
    return width, height


def main() -> None:
    st.title("Алгоритмы отсечения отрезков и многоугольников")
    st.caption("Лианг–Барски, Cyrus–Beck (отрезки) и Сазерленд–Ходжман (многоугольники)")

    width, height = sidebar_controls()

    tab_lines, tab_convex, tab_poly = st.tabs(
        [
            "Отрезки + прямоугольное окно (Лианг–Барски)",
            "Отрезки + выпуклый многоугольник",
            "Отсечение многоугольника",
        ]
    )

    with tab_lines:
        st.subheader("Лианг–Барски")
        default_segments = "-10 -8 12 10\n-5 6 12 -6\n-14 2 -6 8\n-12 -4 4 6"
        segments_text = st.text_area("Координаты отрезков (x0 y0 x1 y1)", value=default_segments, height=120, key="lb_segments")
        xmin, ymin, xmax, ymax = st.columns(4)
        with xmin:
            x_min = st.number_input("Xmin", value=-8.0)
        with ymin:
            y_min = st.number_input("Ymin", value=-6.0)
        with xmax:
            x_max = st.number_input("Xmax", value=8.0)
        with ymax:
            y_max = st.number_input("Ymax", value=6.0)
        uploaded = st.file_uploader("Загрузить файл входных данных", type=["txt"])

        try:
            if uploaded:
                content = uploaded.read().decode("utf-8")
                segments, rect = parse_input_file(content)
                x_min, y_min, x_max, y_max = rect
            else:
                segments = parse_segments(segments_text)
            clipped = batch_liang_barsky(segments, (x_min, y_min, x_max, y_max))
            clip_poly = rect_to_polygon((x_min, y_min, x_max, y_max))
            canvas = render_scene(width, height, segments, clipped, clip_poly)
            st.image(canvas, caption=f"Видимых отрезков: {len(clipped)} из {len(segments)}", use_container_width=True)
            st.download_button("Скачать PNG", canvas_to_bytes(canvas), file_name="liang-barsky.png", mime="image/png", key="dl_lb")
        except ValueError as err:
            st.error(f"Ошибка ввода: {err}")

    with tab_convex:
        st.subheader("Отсечение отрезков выпуклым многоугольником (Cyrus–Beck)")
        segments_text = st.text_area(
            "Отрезки (x0 y0 x1 y1)",
            value="-8 -2 8 6\n-6 -4 12 0\n-10 6 4 -6\n2 -8 2 8",
            height=120,
            key="poly_segments",
        )
        clipper_text = st.text_area(
            "Вершины выпуклого окна (x y, по порядку обхода)",
            value="-6 -6\n8 -4\n10 4\n0 9\n-8 4",
            height=140,
            key="clipper_vertices",
        )
        try:
            segments = parse_segments(segments_text)
            clip_polygon = parse_points(clipper_text)
            if len(clip_polygon) < 3:
                raise ValueError("Нужно минимум 3 вершины")
            clipped = batch_clip_segments(segments, clip_polygon)
            canvas = render_scene(width, height, segments, clipped, clip_polygon)
            st.image(canvas, caption=f"Видимых отрезков: {len(clipped)} из {len(segments)}", use_container_width=True)
            st.download_button("Скачать PNG", canvas_to_bytes(canvas), file_name="convex-segments.png", mime="image/png", key="dl_convex")
        except ValueError as err:
            st.error(f"Ошибка ввода: {err}")

    with tab_poly:
        st.subheader("Отсечение многоугольника (Сазерленд–Ходжман)")
        subject_text = st.text_area(
            "Исходный многоугольник (x y)",
            value="-10 -2\n-4 8\n0 4\n4 10\n10 -4\n0 -8",
            height=160,
            key="subject_poly",
        )
        clipper_text = st.text_area(
            "Клиппирующий выпуклый многоугольник (x y)",
            value="-6 -6\n8 -6\n8 6\n-6 6",
            height=120,
            key="clipper_poly",
        )
        try:
            subject = parse_points(subject_text)
            clipper = parse_points(clipper_text)
            if len(subject) < 3 or len(clipper) < 3:
                raise ValueError("Укажите минимум по 3 вершины для обоих многоугольников")
            clipped_poly = clip_polygon_convex(subject, clipper)
            base_segments = list(zip(subject, subject[1:] + subject[:1]))
            clipped_segments = list(zip(clipped_poly, clipped_poly[1:] + clipped_poly[:1])) if clipped_poly else []
            canvas = render_scene(width, height, base_segments, clipped_segments, clipper, subject_polygon=subject)
            st.image(canvas, caption=f"Вершин после отсечения: {len(clipped_poly)}", use_container_width=True)
            st.download_button("Скачать PNG", canvas_to_bytes(canvas), file_name="polygon-clipping.png", mime="image/png", key="dl_poly")
        except ValueError as err:
            st.error(f"Ошибка ввода: {err}")


if __name__ == "__main__":
    main()


