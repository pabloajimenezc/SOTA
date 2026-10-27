"""Plotly figures. Every function takes plain DataFrames/Series and returns a Figure."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_white"
SEQ = px.colors.qualitative.Safe + px.colors.qualitative.Pastel


def _order(values) -> dict:
    """Natural order (T2 before T10) so each topic always gets the same color and legend position."""
    key = lambda v: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(v))]
    return {"grupo": sorted(pd.unique(pd.Series(values).astype(str)), key=key)}


def _style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(font=dict(size=10)), font=dict(family="Arial"))
    return fig


def per_year(df: pd.DataFrame, color: pd.Series | None = None, share: bool = False, title: str = "") -> go.Figure:
    d = pd.DataFrame({"año": df["publication_year"], "grupo": color if color is not None else "Total"})
    d = d.dropna().groupby(["año", "grupo"]).size().rename("docs").reset_index()
    if share:
        d["docs"] = d["docs"] / d.groupby("año")["docs"].transform("sum") * 100
    fig = px.area(d, x="año", y="docs", color="grupo", color_discrete_sequence=SEQ, title=title,
                  category_orders=_order(d["grupo"]),
                  labels={"docs": "% de publicaciones" if share else "Publicaciones", "grupo": ""})
    return _style(fig)


def hbar(s: pd.Series, title: str = "", xlabel: str = "Publicaciones") -> go.Figure:
    s = s[::-1]
    fig = px.bar(x=s.values, y=s.index.astype(str), orientation="h", title=title, labels={"x": xlabel, "y": ""})
    fig.update_traces(marker_color="#3b6ea5")
    return _style(fig, height=max(300, 24 * len(s) + 80))


def top_bar(values: pd.Series, n: int = 15, title: str = "") -> go.Figure:
    return hbar(values.value_counts().head(n), title)


def topic_map(df: pd.DataFrame, xy: np.ndarray, color: pd.Series, title: str = "") -> go.Figure:
    d = df.assign(x=xy[:, 0], y=xy[:, 1], grupo=color.astype(str).values,
                  tam=np.log1p(df["article_citation_count"]) + 2)
    fig = px.scatter(d, x="x", y="y", color="grupo", size="tam", size_max=9, title=title,
                     category_orders=_order(d["grupo"]),
                     hover_name="document_title", color_discrete_sequence=SEQ,
                     hover_data={"publication_year": True, "publication_title": True,
                                 "article_citation_count": True, "x": False, "y": False, "tam": False},
                     custom_data=["id"])
    fig.update_traces(marker=dict(line=dict(width=0), opacity=0.6))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height=620)


def heatmap(table: pd.DataFrame, title: str = "", fmt: str = ".0f", zlabel: str = "") -> go.Figure:
    fig = px.imshow(table, text_auto=fmt, aspect="auto", color_continuous_scale="Viridis", title=title,
                    labels=dict(color=zlabel))
    return _style(fig, height=max(350, 28 * len(table) + 120))


def squarify(sizes: list[float], x: float = 0, y: float = 0, w: float = 100, h: float = 100) -> list[tuple]:
    """Squarified treemap layout (Bruls et al.). Sizes sorted descending -> [(x, y, w, h)]."""
    total = sum(sizes)
    areas = [s * w * h / total for s in sizes]

    def worst(row, side):
        s = sum(row)
        return max(max(side ** 2 * r / s ** 2, s ** 2 / (side ** 2 * r)) for r in row)

    rects, i = [], 0
    while i < len(areas):
        side, row = min(w, h), [areas[i]]
        i += 1
        while i < len(areas) and worst(row + [areas[i]], side) <= worst(row, side):
            row.append(areas[i])
            i += 1
        s = sum(row)
        if w >= h:  # column on the left
            rw, yy = s / h, y
            for r in row:
                rects.append((x, yy, rw, r / rw))
                yy += r / rw
            x, w = x + rw, w - rw
        else:  # row on top
            rh, xx = s / w, x
            for r in row:
                rects.append((xx, y, r / rh, rh))
                xx += r / rh
            y, h = y + rh, h - rh
    return rects


def treemap_flat(sizes: list[int], codes: list[str], tags: list[str], title: str = "") -> go.Figure:
    """One level of the topic tree, drawn with bars so a click selects a box (Plotly treemaps can't be selected).

    Fixed text size: labels that don't fit are hidden (shown on hover) instead of shrinking.
    """
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i])
    rects = dict(zip(order, squarify([sizes[i] for i in order], 0, 0, 100, 60)))
    pad = 0.25
    x, y, w, h = zip(*[rects[i] for i in range(len(sizes))])
    fig = go.Figure(go.Bar(
        x=[a + b / 2 for a, b in zip(x, w)], y=[b - 2 * pad for b in h], base=[a + pad for a in y],
        width=[b - 2 * pad for b in w], marker=dict(color=(SEQ * (len(sizes) // len(SEQ) + 1))[:len(sizes)]),
        customdata=list(zip(codes, [t.replace(", ", "<br>", 3) for t in tags], sizes, tags)),
        texttemplate="<b>%{customdata[0]}</b> · %{customdata[2]}<br>%{customdata[1]}",
        textposition="inside", insidetextanchor="start", textangle=0, textfont=dict(size=14),
        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[2]} artículos<br>%{customdata[3]}<extra></extra>",
    ))
    fig.update_xaxes(visible=False, range=[0, 100], fixedrange=True)
    fig.update_yaxes(visible=False, range=[60, 0], fixedrange=True)
    fig.update_layout(title=title, uniformtext=dict(minsize=11, mode="hide"), bargap=0, showlegend=False,
                      clickmode="event+select", dragmode=False)
    return _style(fig, height=540)


def citations(df: pd.DataFrame, color: pd.Series, title: str = "") -> go.Figure:
    d = df.assign(grupo=color.astype(str).values, citas=df["article_citation_count"] + 1)
    fig = px.strip(d, x="publication_year", y="citas", color="grupo", log_y=True, title=title,
                   category_orders=_order(d["grupo"]),
                   hover_name="document_title", color_discrete_sequence=SEQ,
                   labels={"citas": "Citas + 1 (log)", "publication_year": "Año", "grupo": ""})
    return _style(fig, height=480)


def emerging(d: pd.DataFrame, title: str = "") -> go.Figure:
    d = d.iloc[::-1]
    fig = go.Figure([
        go.Bar(y=d["término"], x=d["antes"], name="Antes", orientation="h", marker_color="#b0bec5"),
        go.Bar(y=d["término"], x=d["reciente"], name="Reciente", orientation="h", marker_color="#e4572e"),
    ])
    fig.update_layout(barmode="group", title=title, xaxis_title="% de documentos que lo mencionan")
    return _style(fig, height=max(350, 30 * len(d) + 100))
