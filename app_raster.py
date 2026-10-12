import time
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from processing.raster import line_step_by_step, line_dda, line_bresenham, circle_bresenham


st.set_page_config(page_title="Rasterization Lab", layout="wide")


def draw_grid(canvas: Image.Image, grid: int, margin: int = 40) -> None:
    w, h = canvas.size
    draw = ImageDraw.Draw(canvas)
    # background
    draw.rectangle([(0, 0), (w, h)], fill=(32, 34, 37))
    # grid
    color_grid = (60, 62, 68)
    for x in range(margin, w - margin + 1, grid):
        draw.line([(x, margin), (x, h - margin)], fill=color_grid)
    for y in range(margin, h - margin + 1, grid):
        draw.line([(margin, y), (w - margin, y)], fill=color_grid)
    # axes
    draw.line([(margin, h - margin), (w - margin, h - margin)], fill=(220, 220, 220), width=2)
    draw.line([(margin, margin), (margin, h - margin)], fill=(220, 220, 220), width=2)
    draw.text((w - margin + 5, h - margin + 5), "x", fill=(220, 220, 220))
    draw.text((margin - 15, margin - 15), "y", fill=(220, 220, 220))


def to_screen(pt: Tuple[int, int], canvas: Image.Image, grid: int, margin: int = 40) -> Tuple[int, int]:
    w, h = canvas.size
    x, y = pt
    sx = margin + x * grid
    sy = h - (margin + y * grid)
    return sx, sy


def draw_pixels(canvas: Image.Image, points: List[Tuple[int, int]], grid: int, margin: int = 40) -> None:
    draw = ImageDraw.Draw(canvas)
    for x, y in points:
        sx, sy = to_screen((x, y), canvas, grid, margin)
        draw.rectangle([(sx, sy - grid), (sx + grid - 1, sy)], fill=(0, 200, 255))


def main() -> None:
    st.title("Растеризация: базовые алгоритмы")
    st.caption("Пошаговый, ЦДА, Брезенхем (линия/окружность) + сетка/оси и измерение времени")

    with st.sidebar:
        alg = st.selectbox("Алгоритм", ["Пошаговый", "ЦДА", "Брезенхем (линия)", "Брезенхем (окружность)"])
        x0 = st.number_input("x0", value=0, step=1)
        y0 = st.number_input("y0", value=0, step=1)
        x1 = st.number_input("x1", value=10, step=1)
        y1 = st.number_input("y1", value=6, step=1)
        r = st.number_input("radius", value=8, min_value=0, step=1)
        grid = st.slider("Размер клетки", 5, 60, 20, 1)
        w = st.slider("Ширина", 400, 1400, 900, 10)
        h = st.slider("Высота", 300, 1000, 600, 10)

    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    draw_grid(canvas, grid)

    t0 = time.perf_counter()
    if alg == "Брезенхем (окружность)":
        pts = circle_bresenham((int(x0), int(y0)), int(r))
    else:
        p0 = (int(x0), int(y0))
        p1 = (int(x1), int(y1))
        if alg == "Пошаговый":
            pts = line_step_by_step(p0, p1)
        elif alg == "ЦДА":
            pts = line_dda(p0, p1)
        else:
            pts = line_bresenham(p0, p1)
    t1 = time.perf_counter()

    draw_pixels(canvas, pts, grid)
    st.image(canvas, caption=f"pixels: {len(pts)}, time: {(t1 - t0) * 1e6:.0f} μs", use_container_width=True)

    st.download_button("Скачать PNG", data=canvas_to_bytes(canvas), file_name="raster.png", mime="image/png")


def canvas_to_bytes(img: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    main()




