import math
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from processing.transform3d import (
    Point3D,
    Edge,
    create_letter_b_model,
    identity_matrix,
    translation_matrix,
    scaling_matrix,
    rotation_x_matrix,
    rotation_y_matrix,
    rotation_z_matrix,
    rotation_around_axis_matrix,
    apply_transformation,
    orthographic_projection_xy,
    orthographic_projection_xz,
    orthographic_projection_yz,
)

st.set_page_config(page_title="3D Визуализация буквы Б", layout="wide")


def draw_axes(fig: go.Figure, max_range: float = 15) -> None:
    """Добавляет оси координат на график."""
    # Ось X (красная)
    fig.add_trace(go.Scatter3d(
        x=[0, max_range], y=[0, 0], z=[0, 0],
        mode='lines+markers+text',
        line=dict(color='red', width=5),
        marker=dict(size=4, color='red'),
        text=['', 'X'],
        textposition='top center',
        name='Ось X',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Ось Y (зеленая)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, max_range], z=[0, 0],
        mode='lines+markers+text',
        line=dict(color='green', width=5),
        marker=dict(size=4, color='green'),
        text=['', 'Y'],
        textposition='top center',
        name='Ось Y',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Ось Z (синяя)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, max_range],
        mode='lines+markers+text',
        line=dict(color='blue', width=5),
        marker=dict(size=4, color='blue'),
        text=['', 'Z'],
        textposition='top center',
        name='Ось Z',
        showlegend=False,
        hoverinfo='skip'
    ))


def plot_3d_object(vertices: List[Point3D], edges: List[Edge], title: str = "3D Объект") -> go.Figure:
    """Создает 3D график объекта."""
    fig = go.Figure()
    
    # Рисуем рёбра
    for edge in edges:
        i, j = edge
        if i < len(vertices) and j < len(vertices):
            v1 = vertices[i]
            v2 = vertices[j]
            fig.add_trace(go.Scatter3d(
                x=[v1[0], v2[0]],
                y=[v1[1], v2[1]],
                z=[v1[2], v2[2]],
                mode='lines',
                line=dict(color='cyan', width=3),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Рисуем вершины
    if vertices:
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers',
            marker=dict(size=5, color='yellow'),
            name='Вершины',
            showlegend=True,
            hovertemplate='(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))
    
    # Добавляем оси
    max_coord = max(
        max(abs(v[i]) for v in vertices) if vertices else 0
        for i in range(3)
    )
    draw_axes(fig, max(max_coord * 1.2, 10))
    
    # Настройка сцены
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', showbackground=True, backgroundcolor='rgba(230, 230, 230, 0.5)'),
            yaxis=dict(title='Y', showbackground=True, backgroundcolor='rgba(230, 230, 230, 0.5)'),
            zaxis=dict(title='Z', showbackground=True, backgroundcolor='rgba(230, 230, 230, 0.5)'),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def plot_2d_projection(vertices_2d: List[Tuple[float, float]], edges: List[Edge], 
                       original_vertices: List[Point3D], title: str, xlabel: str, ylabel: str) -> go.Figure:
    """Создает 2D график проекции."""
    fig = go.Figure()
    
    # Рисуем рёбра проекции
    for edge in edges:
        i, j = edge
        if i < len(vertices_2d) and j < len(vertices_2d):
            v1 = vertices_2d[i]
            v2 = vertices_2d[j]
            fig.add_trace(go.Scatter(
                x=[v1[0], v2[0]],
                y=[v1[1], v2[1]],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Рисуем вершины
    if vertices_2d:
        xs = [v[0] for v in vertices_2d]
        ys = [v[1] for v in vertices_2d]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='markers',
            marker=dict(size=6, color='red'),
            name='Вершины',
            showlegend=True,
            hovertemplate='(%{x:.2f}, %{y:.2f})<extra></extra>'
        ))
    
    # Добавляем оси
    max_coord = max(
        max(abs(v[i]) for v in original_vertices) if original_vertices else 0
        for i in range(3)
    )
    max_range = max(max_coord * 1.2, 10)
    
    # Ось X
    fig.add_trace(go.Scatter(
        x=[0, max_range], y=[0, 0],
        mode='lines+markers+text',
        line=dict(color='red', width=2),
        marker=dict(size=4, color='red'),
        text=['', xlabel],
        textposition='top center',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Ось Y
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, max_range],
        mode='lines+markers+text',
        line=dict(color='green', width=2),
        marker=dict(size=4, color='green'),
        text=['', ylabel],
        textposition='top center',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(title=xlabel, showgrid=True, zeroline=True),
        yaxis=dict(title=ylabel, showgrid=True, zeroline=True),
        width=400,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def format_matrix(matrix: np.ndarray) -> str:
    """Форматирует матрицу для отображения."""
    rows = []
    for i in range(matrix.shape[0]):
        row = ' | '.join(f'{matrix[i, j]:8.3f}' for j in range(matrix.shape[1]))
        rows.append(f'| {row} |')
    return '\n'.join(rows)


def display_matrix_table(matrix: np.ndarray):
    """Отображает матрицу в виде таблицы Streamlit."""
    st.write("**Матрица преобразования 4×4:**")
    cols = st.columns(4)
    labels = ['X', 'Y', 'Z', 'W']
    
    # Заголовки
    for i, col in enumerate(cols):
        col.write(f"**{labels[i]}**")
    
    # Строки матрицы
    for i in range(matrix.shape[0]):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            col.write(f"{matrix[i, j]:.3f}")


def main() -> None:
    st.title("Лабораторная работа 6: Трехмерная визуализация")
    st.caption("Построение и визуализация трехмерной буквы Б с преобразованиями и проекциями")
    
    # Инициализация состояния
    if 'vertices' not in st.session_state:
        st.session_state.vertices, st.session_state.edges = create_letter_b_model()
        st.session_state.transform_matrix = identity_matrix()
    
    # Боковая панель с параметрами
    with st.sidebar:
        st.header("Параметры преобразований")
        
        st.subheader("Перенос")
        tx = st.number_input("Tx", value=0.0, step=0.5)
        ty = st.number_input("Ty", value=0.0, step=0.5)
        tz = st.number_input("Tz", value=0.0, step=0.5)
        
        st.subheader("Масштабирование")
        sx = st.number_input("Sx", value=1.0, step=0.1, min_value=0.1)
        sy = st.number_input("Sy", value=1.0, step=0.1, min_value=0.1)
        sz = st.number_input("Sz", value=1.0, step=0.1, min_value=0.1)
        
        st.subheader("Вращение вокруг осей")
        rot_x = st.slider("Вращение вокруг X (градусы)", -180, 180, 0)
        rot_y = st.slider("Вращение вокруг Y (градусы)", -180, 180, 0)
        rot_z = st.slider("Вращение вокруг Z (градусы)", -180, 180, 0)
        
        st.subheader("Вращение вокруг произвольной оси")
        use_custom_axis = st.checkbox("Использовать произвольную ось")
        if use_custom_axis:
            axis_x = st.number_input("Ось X", value=1.0, step=0.1)
            axis_y = st.number_input("Ось Y", value=1.0, step=0.1)
            axis_z = st.number_input("Ось Z", value=1.0, step=0.1)
            rot_custom = st.slider("Угол вращения (градусы)", -180, 180, 0)
        else:
            axis_x, axis_y, axis_z, rot_custom = 1.0, 1.0, 1.0, 0
        
        if st.button("Сбросить преобразования"):
            st.session_state.transform_matrix = identity_matrix()
            st.rerun()
    
    # Применяем преобразования
    transform = identity_matrix()
    
    # Перенос
    if tx != 0 or ty != 0 or tz != 0:
        T = translation_matrix(tx, ty, tz)
        transform = T @ transform
    
    # Масштабирование
    if sx != 1.0 or sy != 1.0 or sz != 1.0:
        S = scaling_matrix(sx, sy, sz)
        transform = S @ transform
    
    # Вращение вокруг осей
    if rot_x != 0:
        Rx = rotation_x_matrix(math.radians(rot_x))
        transform = Rx @ transform
    if rot_y != 0:
        Ry = rotation_y_matrix(math.radians(rot_y))
        transform = Ry @ transform
    if rot_z != 0:
        Rz = rotation_z_matrix(math.radians(rot_z))
        transform = Rz @ transform
    
    # Вращение вокруг произвольной оси
    if use_custom_axis and rot_custom != 0:
        R_custom = rotation_around_axis_matrix((axis_x, axis_y, axis_z), math.radians(rot_custom))
        transform = R_custom @ transform
    
    # Применяем преобразование к вершинам
    transformed_vertices = apply_transformation(st.session_state.vertices, transform)
    
    # Вкладки
    tab1, tab2, tab3 = st.tabs([
        "6a: Визуализация 3D объекта",
        "6b: Трехмерные преобразования",
        "6c: Ортографические проекции"
    ])
    
    with tab1:
        st.subheader("Трехмерная каркасная модель буквы Б")
        st.write("Исходный объект без преобразований")
        
        fig_original = plot_3d_object(st.session_state.vertices, st.session_state.edges, 
                                      "Исходная модель буквы Б")
        st.plotly_chart(fig_original, use_container_width=True)
        
        # Показываем координаты вершин
        with st.expander("Координаты вершин"):
            st.write("Массив вершин (x, y, z):")
            for i, vertex in enumerate(st.session_state.vertices):
                st.text(f"Вершина {i}: ({vertex[0]:.2f}, {vertex[1]:.2f}, {vertex[2]:.2f})")
    
    with tab2:
        st.subheader("Трехмерные преобразования")
        
        # Показываем преобразованный объект
        fig_transformed = plot_3d_object(transformed_vertices, st.session_state.edges,
                                        "Преобразованная модель буквы Б")
        st.plotly_chart(fig_transformed, use_container_width=True)
        
        # Показываем матрицу преобразования
        st.subheader("Итоговая матрица преобразования")
        display_matrix_table(transform)
        with st.expander("Матрица в текстовом формате"):
            st.code(format_matrix(transform), language=None)
        
        # Показываем координаты преобразованных вершин
        with st.expander("Координаты преобразованных вершин"):
            for i, vertex in enumerate(transformed_vertices):
                st.text(f"Вершина {i}: ({vertex[0]:.2f}, {vertex[1]:.2f}, {vertex[2]:.2f})")
    
    with tab3:
        st.subheader("Ортографические проекции")
        st.write("Проекции преобразованного объекта на координатные плоскости")
        
        # Проекции
        proj_xy = orthographic_projection_xy(transformed_vertices)
        proj_xz = orthographic_projection_xz(transformed_vertices)
        proj_yz = orthographic_projection_yz(transformed_vertices)
        
        # Отображаем проекции в колонках
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_xy = plot_2d_projection(proj_xy, st.session_state.edges, transformed_vertices,
                                       "Проекция на Oxy", "X", "Y")
            st.plotly_chart(fig_xy, use_container_width=True)
        
        with col2:
            fig_xz = plot_2d_projection(proj_xz, st.session_state.edges, transformed_vertices,
                                       "Проекция на Oxz", "X", "Z")
            st.plotly_chart(fig_xz, use_container_width=True)
        
        with col3:
            fig_yz = plot_2d_projection(proj_yz, st.session_state.edges, transformed_vertices,
                                       "Проекция на Oyz", "Y", "Z")
            st.plotly_chart(fig_yz, use_container_width=True)
        
        # Показываем координаты проекций
        with st.expander("Координаты проекций"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Проекция на Oxy:**")
                for i, (x, y) in enumerate(proj_xy):
                    st.text(f"{i}: ({x:.2f}, {y:.2f})")
            with col2:
                st.write("**Проекция на Oxz:**")
                for i, (x, z) in enumerate(proj_xz):
                    st.text(f"{i}: ({x:.2f}, {z:.2f})")
            with col3:
                st.write("**Проекция на Oyz:**")
                for i, (y, z) in enumerate(proj_yz):
                    st.text(f"{i}: ({y:.2f}, {z:.2f})")


if __name__ == "__main__":
    main()

