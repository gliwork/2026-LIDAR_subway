#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_fast_ru.py — онлайн-измерение дистанции по 3 лазерным линиям,
быстрый режим, рассчитанный на 30+ к/с.

Тот же стенд (3 лазера + камера) и та же калибровка, но минимально
дорогой поскадовый конвейер:

  1. Ч/б + гистограммовый порог яркости (O(256), без np.sort)   ~1 мс
  2. Маска ярких пикселей, центральное окно, разреженные
     координаты яркой области                                   ~3 мс
  3. Для каждого лазера: прямая по яркой полосе вокруг
     прогноза из предыдущего кадра (1–5 попыток, каждая ~1 мс)
  4. Векторизованный поиск дистанции на калибровочных картах
     (предвычисленная сетка, ~0.3–0.5 мс вместо ~6 мс)
  5. Ближние препятствия: компоненты связности маски с рамкой >= 5x5 px
     (в ЛЮБОЙ точке изображения); признак считается БЛИЖНИМ, если он на
     ближней стороне измеренной лазерной линии (ниже B, левее L, правее
     R) и не лежит на самой линии; кросс-проверка точек по лазерам
  6. Аннотация кадра + imshow

Типичная цена кадра ~20-25 мс -> 40+ к/с в одиночку.
Если по какому-то лазеру наведение не сработало (смена сцены,
затенение) — для НЕГО только один раз выполняется полный проход
детекции (~35 мс), с кулдауном повторений.

Обрабатываются только «реальные» признаки: яркие пятна меньше 5x5 px
полностью игнорируются (шум/зерно), а ближние признаки считаются
подтверждёнными только после 3 подряд кадров (NearTracker).

Запуск:
    python3 camera_fast_ru.py                 # камера 0
    python3 camera_fast_ru.py --cam 1
    python3 camera_fast_ru.py --min-brightness 40
    python3 camera_fast_ru.py --video f.mp4   # тот же конвейер, файл
Клавиши: q / ESC — выход.
"""

import argparse
from typing import Optional
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laser_sensor as ls
import measure as mm

# Размер изображения, на котором делалась калибровка
REF_W, REF_H = 1280, 720
# Сколько раз подряд можно пускать полный (дорогой) проход детекции
# для «потерявшегося» лазера, прежде чем временно от него отказаться
RECOVERY_MAX_STREAK = 6


# ----------------------------------------------------------------------------
# Быстрые примитивы
# ----------------------------------------------------------------------------

def fast_threshold(gray: np.ndarray) -> float:
    """Гистограммовая версия ls.image_threshold: то же правило
    30/80-процентилей, но O(256) вместо O(N log N) (~1 мс)."""
    hist = np.bincount(gray.ravel(), minlength=256)
    total = int(hist.sum())
    if total == 0:
        return 25.0
    cum = np.cumsum(hist)
    i30 = int(np.searchsorted(cum, 0.30 * total))
    i80 = int(np.searchsorted(cum, 0.80 * total))
    return max(25.0, 0.5 * (i30 + i80))


class FastMap:
    """Векторизованный поиск дистанции по калибровочной карте (LineMap).

    Карта — эмпирическая пара кривых (наклон m(d), позиция x0(d)).
    Мы один раз предвыбираем их на плотной сетке d и приводим каждую
    точку сетки к нормализованным коэффициентам прямой (a,b,c).
    Тогда «подгонка» измеряемой прямой к карте — одно векторное
    вычисление остатка по сотне-двум сеточным точкам (~0.3 мс)
    вместо ~350-итерационного python-цикла LineMap.fit_distance (~6 мс).
    """

    def __init__(self, m: ls.LineMap, step: float = 0.02):
        self.m = m
        self.lo, self.hi = float(m.ds[0]), float(m.ds[-1])
        n = max(64, int(round((self.hi - self.lo) / step)) + 1)
        # сетка расстояний и интерполяция кривых карты на неё
        self.ds = np.linspace(self.lo, self.hi, n)
        self.x0 = np.interp(self.ds, m.ds, m.x0s)
        self.slopes = np.interp(self.ds, m.ds, m.ms)
        # нормализованные коэффициенты прямых для каждой точки сетки
        if m.vertical:
            # вертикальная: x = s*y + x0  ->  a=1, b=-s, c=s*360 - x0
            c = self.slopes * 360.0 - self.x0
            nrm = np.hypot(1.0, self.slopes)
            self.a = 1.0 / nrm
            self.b = -self.slopes / nrm
            self.c = c / nrm
        else:
            # горизонтальная: y = s*x + y0  ->  a=-s, b=1, c=s*640 - y0
            c = self.slopes * 640.0 - self.x0
            nrm = np.hypot(self.slopes, 1.0)
            self.a = -self.slopes / nrm
            self.b = 1.0 / nrm
            self.c = c / nrm

    def fit(self, line: ls.ImageLine, prior=None) -> tuple:
        """(d, err_px) для измеренной прямой, либо (None, inf)."""
        if not line.ok:
            return None, float("inf")
        # нормализуем коэффициенты измеренной прямой
        am, bm, cm = line.a, line.b, line.c
        nrm = math.hypot(am, bm)
        if nrm < 1e-9:
            return None, float("inf")
        am, bm, cm = am / nrm, bm / nrm, cm / nrm
        # контрольные точки на измеренной прямой (±100 px от точки,
        # ближайшей к началу координат) — та же метрика, что в
        # ls.line_residual
        cx0, cy0 = -am * cm, -bm * cm
        dvx, dvy = bm, -am
        p1x, p1y = cx0 - 100.0 * dvx, cy0 - 100.0 * dvy
        p2x, p2y = cx0 + 100.0 * dvx, cy0 + 100.0 * dvy
        # ориентируем прогнозированные прямые как измеренную
        sgn = np.sign(self.a * am + self.b * bm)
        sgn = np.where(sgn == 0.0, 1.0, sgn)
        a, b, c = self.a * sgn, self.b * sgn, self.c * sgn
        # остаток по всем точкам сетки сразу (векторно)
        e1 = np.abs(a * p1x + b * p1y + c)
        e2 = np.abs(a * p2x + b * p2y + c)
        err = np.maximum(e1, e2)
        ds = self.ds

        def best(i):
            return float(ds[i]), float(err[i])

        i = int(np.argmin(err))
        if prior is not None and self.lo <= prior <= self.hi:
            # если есть априори (из прошлого кадра) — сначала ищем
            # минимум в окне ±1.5 м от него: стабильнее при нескольких
            # локальных минимумах
            lo_i = max(0, int((prior - 1.5) / (ds[1] - ds[0])))
            hi_i = min(len(ds) - 1, int((prior + 1.5) / (ds[1] - ds[0])) + 1)
            if hi_i > lo_i:
                jl = int(np.argmin(err[lo_i:hi_i + 1])) + lo_i
                if err[jl] < 8.0:
                    i = jl
                else:
                    i = int(np.argmin(err))
        return best(i)


def _fit_strip_line(x: np.ndarray, y: np.ndarray, pred,
                    strip: float = 60.0) -> tuple:
    """Прямая по ярким точкам в полосе ±strip px вокруг прогнозируемой.

    Возвращает (line, err); line.ok = False, если подгонка
    ненадёжна (мало вайлеров, слишком короткий размах).

    Если вайлеры заметно «гофрированы» (ближнее препятствие гнёт
    лазерный луч), кластеры разделяются по знаку остатка; дальше
    используется главный луч (ближе всего к прогнозу), а ближний
    «загнутый» участок ловит детектор ближних признаков.
    """
    a, b, c = pred.a, pred.b, pred.c
    # выбираем яркую область внутри полосы
    sel = np.abs(a * x + b * y + c) < strip
    n = int(sel.sum())
    if n < 40:
        return ls.ImageLine(ok=False), float("inf")
    xs, ys = x[sel], y[sel]
    line = ls._fit_line_to_points(xs, ys)
    if not line.ok:
        return line, float("inf")
    # второй проход по вайлерам: отбрасываем точки далеко от прямой
    d2 = np.abs(line.a * xs + line.b * ys + line.c)
    k = d2 < 8.0
    if int(k.sum()) < 30:
        line.ok = False
        return line, float("inf")
    line = ls._fit_line_to_points(xs[k], ys[k])
    if not line.ok:
        return line, float("inf")
    # разделение «ломаного» луча: большой разброс остатков -> две
    # подпрямые; обе должны быть достаточными (>=30 точек, размах
    # >=60 px), иначе оставляем единую подгонку
    dv = line.a * xs[k] + line.b * ys[k] + line.c
    if dv.max() - dv.min() > 16.0:
        km = dv < 0
        parts = [(xs[k][km], ys[k][km]), (xs[k][~km], ys[k][~km])]
        sub = []
        for cjx, cjy in parts:
            if len(cjx) >= 30:
                lj = ls._fit_line_to_points(cjx, cjy)
                if lj.ok:
                    ddj = np.array([lj.b, -lj.a])
                    tv = cjx * ddj[0] + cjy * ddj[1]
                    if tv.max() - tv.min() >= 60.0:
                        sub.append((len(cjx), lj))
        if len(sub) == 2 and min(s[0] for s in sub) >= 30:
            # две настоящие подпрямые: берём ту, что ближе к прогнозу
            line = (sub[0][1] if _pred_dist(pred, sub[0][1]) <=
                    _pred_dist(pred, sub[1][1]) else sub[1][1])
    # настоящая лазерная линия занимает > 120 px по длине
    dd = np.array([line.b, -line.a])
    tv = xs[k] * dd[0] + ys[k] * dd[1]
    if (tv.max() - tv.min()) < 120.0:
        line.ok = False
        return line, float("inf")
    # отклонение от прогноза: в 3 точках на подогнанной прямой
    # (та же конвенция, что в extract_line_guided)
    H_, W_ = 720, 1280
    cxp, cyp = W_ / 2.0, H_ / 2.0
    s = line.a * cxp + line.b * cyp + line.c
    fx, fy = cxp - line.a * s, cyp - line.b * s
    dx, dy = line.b, -line.a
    err = 0.0
    for t in (-100.0, 0.0, 100.0):
        px, py = fx + t * dx, fy + t * dy
        err = max(err, abs(a * px + b * py + c))
    return line, err


def _pred_dist(pred, line) -> float:
    """Максимальное отклонение (px) прямой `line` от прогноза `pred`
    в окрестности центра изображения."""
    H_, W_ = 720, 1280
    cxp, cyp = W_ / 2.0, H_ / 2.0
    s = line.a * cxp + line.b * cyp + line.c
    fx, fy = cxp - line.a * s, cyp - line.b * s
    dx, dy = line.b, -line.a
    a, b, c = pred.a, pred.b, pred.c
    e = 0.0
    for t in (-100.0, 0.0, 100.0):
        px, py = fx + t * dx, fy + t * dy
        e = max(e, abs(a * px + b * py + c))
    return e


# ----------------------------------------------------------------------------
# Ближние признаки из пятен >= 5x5
# ----------------------------------------------------------------------------

def near_features_fast(mask: np.ndarray, maps: dict,
                       b_line, v_lines, laser_ds, model=None,
                       min_box: int = 5) -> list:
    """Компактный блоб-детектор ближних препятствий.

    Компоненты связности яркой маски с рамкой >= min_box x min_box px
    (по умолчанию 5x5) — фильтр размера действует на признаки в ЛЮБОЙ
    точке изображения. Признак считается кандидатом на БЛИЖНЕЕ
    препятствие, если он лежит на БЛИЖНЕЙ стороне хотя бы одной
    измеренной лазерной линии: более близкая поверхность выгибает
    след линии наружу (от центра изображения):

        B (горизонтальный лист): признак НИЖЕ линии B
        L (вертикальный веер):   признак ЛЕВЕЕ линии L
        R (вертикальный веер):   признак ПРАВЕЕ линии R

    Признаки НА измеренной линии (собственные пиксели линии) или на
    ДАЛЬНЕЙ стороне всех линий (фон за измеренными поверхностями)
    отбрасываются. Дистанция: точка признака проецируется на
    калибровочную карту каждой подходящей линии, побеждает наиболее
    надёжное проецирование. Формы «сгиба листа» (hline/vline)
    освобождены от кросс-проверки по лазерам; точки должны
    подтверждаться измеренным лазером."""
    H, W = mask.shape
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    feats = []
    if n <= 1:
        return feats

    # вертикальные лазерные линии (для ограничения коридором)
    vl = [v_lines[k] for k in ("L", "R")
          if v_lines.get(k) is not None and v_lines[k].ok]

    for i in range(1, n):
        x0, y0, w, h, area = int(stats[i][0]), int(stats[i][1]), \
            int(stats[i][2]), int(stats[i][3]), int(stats[i][4])
        # ---- ГЛАВНЫЙ ФИЛЬТР ЗАПРОСА: игнорируем всё меньше 5x5 px ----
        if w < min_box or h < min_box:
            continue                        # зерно/шум
        if w > 250 or h > 250 or area > 30000:
            continue                        # кромки комнаты / слившийся луч
        cx, cy = float(cents[i][0]), float(cents[i][1])
        kind = "hline" if w > h else ("vline" if h > w else "dot")
        # отсеки по квадрантам (заточены на оффлайн-данных: кромки фона)
        if kind == "hline" and (cx < 0.25 * W or cx > 0.75 * W):
            continue
        if kind == "vline" and (cy < 0.5 * H or cy > 0.9 * H):
            continue
        # коридор: между V-линиями в этом ряду (запас 60 px, чтобы
        # наружный сгиб луча от близкого объекта проходил)
        if vl:
            xs_line = [(-v.b * cy - v.c) / v.a for v in vl if abs(v.a) > 1e-6]
            if xs_line:
                lo_x, hi_x = min(xs_line) - 60.0, max(xs_line) + 60.0
                if not (lo_x <= cx <= hi_x):
                    continue
        # не НА измеренной лазерной линии (это сам луч — он уже учтён
        # подгонкой прямой)
        on_line = False
        for k in ("L", "R", "B"):
            v = v_lines.get(k)
            if v is None or not v.ok:
                continue
            if abs(v.a * cx + v.b * cy + v.c) < 10.0:
                on_line = True
                break
        if on_line:
            continue
        # тест БЛИЖНЕЙ стороны по каждой линии:
        #   ниже B / левее L / правее R.
        # Для каждой подходящей линии проецируем точку на её
        # калибровочную карту -> кандидат (d, err).
        cand = []
        bv = v_lines.get("B")
        if bv is not None and bv.ok and abs(bv.b) > 1e-6:
            yb = (-bv.a * cx - bv.c) / bv.b
            if cy > yb + 6.0:
                rb = _point_b_distance_fast(maps["B"], cx, cy, model)
                if rb[0] is not None:
                    cand.append(rb)
        lv = v_lines.get("L")
        if lv is not None and lv.ok and abs(lv.a) > 1e-6:
            xl = (-lv.b * cy - lv.c) / lv.a
            if cx < xl - 6.0:
                rl = _point_v_distance_fast(maps["L"], cx, cy, model,
                                            "L", lv)
                if rl[0] is not None:
                    cand.append(rl)
        rv = v_lines.get("R")
        if rv is not None and rv.ok and abs(rv.a) > 1e-6:
            xr = (-rv.b * cy - rv.c) / rv.a
            if cx > xr + 6.0:
                rr = _point_v_distance_fast(maps["R"], cx, cy, model,
                                            "R", rv)
                if rr[0] is not None:
                    cand.append(rr)
        if not cand:
            continue                        # дальняя сторона всех линий: фон
        # побеждает наиболее надёжное проецирование
        # (наименьший err, при равенстве — ближайшее d)
        d, derr = min(cand, key=lambda t: (t[1] if t[1] is not None
                                           else 99.0, t[0]))
        near = (derr is not None and derr <= 6.0)
        # кросс-проверка: точка должна подтверждаться измеренным
        # лазером; сгибы листа (hline/vline) — подлинные, освобождены
        if near and kind == "dot" and laser_ds:
            if not any((d - 0.5) <= ld <= (d + 2.0)
                       for ld in laser_ds if ld is not None and ld == ld):
                near = False
        feats.append({"kind": kind, "x0": x0, "y0": y0,
                      "x1": x0 + w, "y1": y0 + h,
                      "near": near, "d": d, "n": int(area)})
    return feats


def _point_b_distance_fast(b_map: ls.LineMap, x: float, y: float,
                           model=None) -> tuple:
    """Векторная дистанция точки по карте B (быстрый аналог
    ls._point_distance_to_b).

    Возвращает (d, err_px); err = None, если d получена
    экстраполяцией/3D, а не прямым совпадением с картой.
    """
    if b_map is None or not b_map.ds:
        return None, None
    ds_a = np.asarray(b_map.ds, float)
    slopes = np.asarray(b_map.ms, float)
    y0s = np.asarray(b_map.x0s, float)
    # y-позиция линии B при данном x для каждой точки калибровки
    y_pred = slopes * (x - 640.0) + y0s
    err = np.abs(y - y_pred)
    i = int(np.argmin(err))
    if err[i] <= 10.0:
        d = float(ds_a[i])
        if d < 2.02:
            # ближайшая калибровочная дистанция: сканирование карты
            # насыщено — экстраполируем эмпирическую карту, иначе 3D
            d3 = ls._extrapolate_b_distance(b_map, x, y)
            if d3 is None:
                d3 = ls._point_b_plane_distance(model, x, y)
            if d3 is not None and 0.3 <= d3 <= 15.0:
                return float(d3), None
        return d, float(err[i])
    if model is not None:
        d3 = ls._point_b_plane_distance(model, x, y)
        if d3 is not None and 0.3 <= d3 <= 15.0:
            return float(d3), None
    return None, None


def _point_v_distance_fast(v_map: ls.LineMap, x: float, y: float,
                           model=None, fan_name: str = "L",
                           v_line=None) -> tuple:
    """Векторная дистанция точки по карте V-веера (L/R) — «перевёрнутый»
    аналог _point_b_distance_fast.

    Карта V: x = m(d)*(y-360) + x0(d).  Возвращает (d, err_px); err = None,
    если d получена экстраполяцией/3D, а не прямым совпадением с картой.
    v_line — подогнанная V-линия текущего кадра: точка далеко вне её
    следа (зона «сгиба» ~60 px) — это не ближний объект.
    """
    if v_map is None or not v_map.ds:
        return None, None
    ds_a = np.asarray(v_map.ds, float)
    slopes = np.asarray(v_map.ms, float)
    x0s = np.asarray(v_map.x0s, float)
    # x-позиция V-линии в этом ряду для каждой калибровочной дистанции
    x_pred = slopes * (y - 360.0) + x0s
    err = np.abs(x - x_pred)
    i = int(np.argmin(err))
    # ограничитель «зоны сгиба»: не более 60 px от подогнанного следа
    out_of_zone = False
    if v_line is not None and abs(v_line.a) > 1e-6:
        x_fit = (-v_line.b * y - v_line.c) / v_line.a
        out_of_zone = abs(x - x_fit) > 60.0
    if err[i] <= 10.0:
        d = float(ds_a[i])
        if d - float(ds_a[0]) < 0.1 and not out_of_zone:
            # ближе ближайшей калибровочной дистанции: экстраполируем
            # эмпирическую карту, иначе — 3D-плоскость веера
            d3 = _extrapolate_v_distance(v_map, x, y)
            if d3 is None:
                d3 = _point_fan_plane_distance(model, fan_name, x, y)
            if d3 is not None and 0.3 <= d3 <= 15.0:
                return float(d3), None
        return d, float(err[i])
    if model is not None and not out_of_zone:
        # 3D-пересечение осмысленно только в зоне сгиба: далёкие точки
        # на плоскости веера — это просто плоскость, а не близкий объект
        d3 = _point_fan_plane_distance(model, fan_name, x, y)
        if d3 is not None and 0.3 <= d3 <= 15.0:
            return float(d3), None
    return None, None


def _extrapolate_v_distance(v_map: ls.LineMap, x: float, y: float,
                            lo: float = 0.4) -> Optional[float]:
    """Дистанция для точки чуть левее/правее ближайшей калибровочной
    V-линии: локальная линейная подгонка x0_est(d) = a*d + b по 3-4
    ближайшим узлам карты, решение относительно d.  None, если оценка
    не лежит чуть ниже ближнего края диапазона калибровки."""
    try:
        ds = np.asarray(v_map.ds, float)
        x0s = np.asarray(v_map.x0s, float)
        n = min(4, len(ds))
        m_avg = float(np.mean(v_map.ms))
        x0_est = x - m_avg * (y - 360.0)
        A = np.vstack([ds[:n], np.ones(n)]).T
        coef = np.linalg.lstsq(A, x0s[:n], rcond=None)[0]
        if abs(coef[0]) < 1e-9:
            return None
        d = (x0_est - coef[1]) / coef[0]
        if lo < d < ds[0] + 0.15:
            return float(d)
    except Exception:
        pass
    return None


def _point_fan_plane_distance(model, fan_name: str, x: float,
                              y: float) -> Optional[float]:
    """Мировое Z точки пересечения луча камеры через (x, y) с плоскостью
    веера с именем fan_name — обобщение ls._point_b_plane_distance на
    любой веер (L/R/B)."""
    if model is None:
        return None
    try:
        f = next(f for f in model.fans if f.name == fan_name)
        cam = model.cam
        v_cam = np.array([(x - cam.cx) / cam.fx,
                          (y - cam.cy) / cam.fy, 1.0])
        v_w = cam.R @ v_cam
        denom = float(f.normal @ v_w)
        if abs(denom) < 1e-6:
            return None
        t = (f.c - float(f.normal @ cam.C)) / denom
        if t <= 0:
            return None
        P = cam.C + t * v_w
        if 0.3 < P[2] < 25.0:
            return float(P[2])
    except Exception:
        pass
    return None


def _seg_from_line_fast(line: ls.ImageLine, g: np.ndarray,
                        wx: np.ndarray, wy: np.ndarray,
                        margin: float = 6.0) -> ls.LineSegment:
    """Сегмент вокруг подогнанной прямой из разреженных ярких пикселей
    (дешёвый аналог mm.segment_from_line, без сетки на весь кадр)."""
    dd = np.array([line.b, -line.a])
    d = np.abs(line.a * wx + line.b * wy + line.c)
    m = d < margin
    t = wx * dd[0] + wy * dd[1]
    n = int(m.sum())
    if n < 30:
        return ls.LineSegment(line=line, t_start=float(t.min()),
                              t_stop=float(t.max()), brightness=0.0, n_points=n)
    return ls.LineSegment(line=line, t_start=float(t[m].min()),
                          t_stop=float(t[m].max()),
                          brightness=float(g[wy[m], wx[m]].mean()),
                          n_points=n)


# ----------------------------------------------------------------------------
# Быстрое измерение: один кадр
# ----------------------------------------------------------------------------

def fast_measure(model: ls.Model, maps: dict, fmaps: dict, img: np.ndarray,
                 state, fan_by_name: dict, min_brightness=None) -> ls.Measurement:
    """Одно быстрое измерение.

    `state` — словарь состояния между кадрами:
      prior        — сглаженный априори дистанции (якорь наведения)
      last_d       — свежее расстояние текущего кадра
      fail_streak  — сколько кадров подряд не удалось навести
      still_frames — сколько кадров д≈const (запуск повторного
                     грубого наведения)
      tracker      — NearTracker (пersistence 3 кадра)
      recovery     — счётчик полных проходов восстановления
    """
    # ---- 1. ч/б и адаптивный порог яркости -------------------------------
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    thr = fast_threshold(g)
    if min_brightness is not None:
        thr = max(thr, float(min_brightness))
    # ---- 2. маска ярких пикселей + центральное окно ---------------------
    mask = (g > thr).astype(np.uint8)
    H, W = g.shape
    ys, xs = np.where(mask)
    k = (xs > 0.25 * W) & (xs < 0.75 * W) & (ys > 0.15 * H) & (ys < 0.85 * H)
    wx, wy = xs[k], ys[k]

    # ---- 3. наведение и подгонка прямых для L/R/B ------------------------
    lines = {}
    failed = []
    for fan in ("L", "R", "B"):
        prior = state["prior"].get(fan)
        streak = state["fail_streak"].get(fan, 0)
        lo, hi = maps[fan].ds[0], maps[fan].ds[-1]
        # «залипание» на краю калибровочного диапазона или застой
        # измерения — повод добавить грубые попытки
        at_boundary = prior is not None and (prior - lo < 0.3 or hi - prior < 0.3)
        stalled = state["still_frames"].get(fan, 0) >= 20
        tries = []
        if prior is not None:
            # сначала — окрестность априори (от ближайшего к дальнейшему)
            tries = [prior, prior - 0.5, prior + 0.5, prior - 1.0, prior + 1.0]
        if prior is None or at_boundary or stalled or streak >= 2:
            # нет якоря / застряли на краю / сцена изменилась:
            # дополнительно пробуем 3 грубые точки диапазона
            # (побеждает лучшее по сумме ошибок, см. ниже)
            tries += [lo + 0.25 * (hi - lo), (lo + hi) / 2.0,
                      lo + 0.75 * (hi - lo)]
        best = None  # (score, d, line)
        for d_try in tries:
            dc = min(max(d_try, lo), hi)
            pred = maps[fan].predict(dc)          # прогноз положения линии
            cand, err = _fit_strip_line(wx, wy, pred, strip=60.0)
            if not (cand.ok and err < 12.0):
                continue                          # наведение не сработало
            d, ferr = fmaps[fan].fit(cand, prior) # дистанция по карте
            if d is None or ferr > 15.0:
                continue
            # выбор лучшей из прошедших попыток:
            # сумма «ошибка наведения + ошибка карты», со штрафом
            # подгонкам на краю (насыщенных) диапазона
            score = err + ferr
            if (d - lo < 0.3 or hi - d < 0.3) and ferr > 3.0:
                score += 5.0
            if best is None or score < best[0]:
                best = (score, d, cand)
        if best is None:
            # лазер не наведён: наращиваем счётчик провалов; пока
            # streak <= RECOVERY_MAX_STREAK — запустим полный проход
            state["fail_streak"][fan] = streak + 1
            if streak + 1 <= RECOVERY_MAX_STREAK:
                failed.append(fan)
            # после 30 кадров провала сбрасываем якорь (сцена реально
            # изменилась) — но текущий кадр этот лазер просто не
            # измеряется
            if state["fail_streak"][fan] > 30:
                state["prior"][fan] = None
            continue
        state["fail_streak"][fan] = 0
        d, line = best[1], best[2]
        # следим за «застоем» дистанции (запуск грубого перенаведения)
        if prior is not None and abs(d - prior) < 0.05:
            state["still_frames"][fan] = state["still_frames"].get(fan, 0) + 1
        else:
            state["still_frames"][fan] = 0
        # сглаживаем ЯКОРЬ (EMA 0.7/0.3); само измерение остаётся «сырым»
        state["prior"][fan] = d if prior is None else 0.7 * prior + 0.3 * d
        lines[fan] = line
        state["last_d"][fan] = d

    if failed:
        # ---- 4. восстановление: полный проход детекции для провалившихся
        # лазеров (реже всего; один раз в несколько кадров) ---------------
        segs, cands = ls.detect_line_segments(img, thr=thr,
                                              return_candidates=True)
        state["recovery"] += 1
        for fan in failed:
            got = None
            if fan == "B":
                # горизонталь: лучший сегмент по ошибке карты
                best = None
                for s in segs.get("H", []):
                    r = maps[fan].fit_distance(s.line)
                    if r and r[1] < 15.0:
                        if best is None or r[1] < best[0]:
                            best = (r[1], r[0], s.line)
                if best:
                    got = (best[2], best[1])
            else:
                # вертикаль: классификация кандидатов L/R по ошибкам
                # обеих карт
                scored = []
                for cx, gl in cands:
                    rep = max(gl, key=lambda s: s.n_points)
                    eL = ls._map_err(maps["L"], rep.line)
                    eR = ls._map_err(maps["R"], rep.line)
                    scored.append((fan, rep.line,
                                   eL if fan == "L" else eR, cx))
                for fan2, line, e, cx in sorted(scored, key=lambda t: t[2]):
                    if fan2 == fan and e < 10.0:
                        r = maps[fan].fit_distance(line)
                        if r:
                            got = (line, r[0])
                        break
            if got:
                lines[fan] = got[0]
                state["last_d"][fan] = got[1]
                state["prior"][fan] = got[1]
                state["fail_streak"][fan] = 0
            # иначе: якорь (последний prior) остаётся для поиска

    # ---- 5. стена или объект: то же правило, что в measure() ------------
    # в MIN идут ТОЛЬКО значения текущего кадра
    ds_found = [state["last_d"][f] for f in lines]
    distances = {f: state["last_d"][f] for f in lines}
    if ds_found:
        med = float(np.median(ds_found))
        mn = min(ds_found)
        # если минимальное заметно меньше медианы (<=0.75) — это объект,
        # иначе (стена) отчётная дистанция — медиана
        closest = mn if (mn / med) <= 0.75 else med
    else:
        closest = None

    # ---- 6. углы границы коридора: пересечения V x B (2D) и 3D ---------
    pairs = []
    if "B" in lines and state["last_d"].get("B") is not None:
        for vname in ("L", "R"):
            if vname in lines:
                p2 = ls.line_intersection(lines[vname], lines["B"])
                if p2 is not None:
                    p3 = ls._fan_intersection_3d(fan_by_name[vname],
                                                 fan_by_name["B"],
                                                 state["last_d"]["B"])
                    pairs.append((p2, p3))
    corners = tuple(p[0] for p in pairs) or None
    corners_3d = tuple(p[1] for p in pairs) or None

    # ---- 7. ближние признаки (на ближней стороне любой линии) ----------
    #      + подтверждение за 3 кадра
    b_line = lines.get("B")
    laser_ds = ds_found
    raw_near = near_features_fast(mask, maps, b_line, lines,
                                  laser_ds, model)
    confirmed = state["tracker"].update(raw_near)
    near_min = min((f["d"] for f in confirmed if f["near"] and f["d"]),
                   default=None)
    if near_min is not None and closest is not None:
        closest = min(closest, near_min)
    elif near_min is not None:
        closest = near_min

    # ---- 8. сегменты для аннотатора (разреженная версия) ----------------
    segs_out = {}
    for key, fan in (("V1", "L"), ("V2", "R"), ("H", "B")):
        if fan in lines:
            s = _seg_from_line_fast(lines[fan], g, wx, wy, margin=6.0)
            segs_out[key] = [s]

    # ---- 9. итоговый Measurement ----------------------------------------
    # в distances попадают только реально измеренные в этом кадре лазеры
    res = ls.Measurement(distances={f: distances[f] for f in distances
                                    if distances[f] is not None},
                         closest=closest,
                         corners=corners, corners_3d=corners_3d,
                         lines={f: lines[f] for f in lines},
                         near_features=confirmed, near_min=near_min)
    res._segs = segs_out
    res.details = {f: [{"d": state["last_d"][f], "n": int(lines[f].n_points)}]
                   for f in lines}
    return res


# ----------------------------------------------------------------------------
# Камера (та же политика, что в camera_online.py)
# ----------------------------------------------------------------------------

def open_camera(index: int, width: int, height: int):
    """Открывает камеру и запрашивает MJPG в нужном разрешении."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        dev = [d for d in os.listdir("/dev") if d.startswith("video")]
        raise SystemExit(
            f"не удалось открыть камеру {index}.  Устройства /dev/video*: "
            f"{dev or 'нет'}  (попробуйте --cam 1..3, проверьте права)")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main():
    ap = argparse.ArgumentParser(
        description="Онлайн-измерение дистанции по 3 лазерам, 30+ к/с")
    ap.add_argument("--cam", type=int, default=0, help="индекс камеры (0)")
    ap.add_argument("--video", default=None, help="видеофайл как источник")
    ap.add_argument("--cal", default="calibration.json")
    ap.add_argument("--width", type=int, default=REF_W)
    ap.add_argument("--height", type=int, default=REF_H)
    ap.add_argument("--min-brightness", type=float, default=None,
                    help="поднять порог яркости (пересвеченная сцена)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-smooth", action="store_true",
                    help="выключить сглаживание априори (сырые значения)")
    args = ap.parse_args()

    # ---- загрузка калибровки --------------------------------------------
    model, doc, maps = ls.load_calibration(args.cal)
    fmaps = {k: FastMap(m) for k, m in maps.items()}   # быстрые карты
    fan_by_name = {f.name: f for f in model.fans}
    traj = ls.corner_trajectories(maps)                # траектории углов
    print("точки исчезания: "
          + "  ".join(f"{k} vp=({t['vp'][0]:.0f},{t['vp'][1]:.0f})"
                      for k, t in traj.items()))

    # ---- источник: камера или видеофайл ---------------------------------
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"не удалось открыть видеофайл {args.video}")
    else:
        cap = open_camera(args.cam, args.width, args.height)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS)
    print(f"источник: {args.video or f'камера {args.cam}'}  {w}x{h} "
          f"@ {fps_cam:.1f} к/с" + (f"  (масштабируется в {REF_W}x{REF_H})"
                                    if (w, h) != (REF_W, REF_H) else ""))

    # ---- состояние между кадрами ----------------------------------------
    state = {"prior": {"L": None, "R": None, "B": None},
             "last_d": {"L": None, "R": None, "B": None},
             "fail_streak": {"L": 0, "R": 0, "B": 0},
             "still_frames": {"L": 0, "R": 0, "B": 0},
             "tracker": mm.NearTracker(persist=3),
             "recovery": 0}
    win = "laser distance 30fps (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, REF_W, REF_H)

    if not args.quiet:
        # ---- замер РЕАЛЬНОЙ частоты доставки камеры ---------------------
        # (UVC-камеры часто сообщают 30 к/с, а по USB отдают меньше;
        # итоговая частота ограничена тем из двух: камера или обработка)
        t0 = time.perf_counter()
        nf = 0
        while time.perf_counter() - t0 < 2.0:
            ok, _ = cap.read()
            nf += ok
        time.sleep(0.2)
        print(f"камера фактически отдаёт {nf/2.0:.1f} к/с "
              f"(запрошено {fps_cam:.0f}); если это меньше частоты "
              f"обработки — ограничитель камеры", flush=True)

    last_print = 0.0
    n_frames = 0
    t_loop_start = time.perf_counter()
    proc_times = []

    try:
        # ---- главный цикл ------------------------------------------------
        while True:
            t0 = time.perf_counter()
            ok, img = cap.read()
            if not ok:
                if args.video:
                    break                       # конец файла
                # камера «выпала»: переподключаем
                time.sleep(0.01)
                cap.release()
                cap = open_camera(args.cam, args.width, args.height)
                continue
            # нормализация под размер калибровки
            if (img.shape[1], img.shape[0]) != (REF_W, REF_H):
                img = cv2.resize(img, (REF_W, REF_H))
            # измерение + аннотация
            res = fast_measure(model, maps, fmaps, img, state,
                               fan_by_name,
                               min_brightness=args.min_brightness)
            if args.no_smooth:
                state["prior"] = dict(state["last_d"])
            out = mm.annotate(img, res, traj)
            cv2.imshow(win, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            n_frames += 1
            proc_times.append((time.perf_counter() - t0) * 1000.0)
            # ---- консольный лог (раз в 0.25 с) ---------------------------
            if not args.quiet and time.time() - last_print > 0.25:
                last_print = time.time()
                d = res.distances
                ds = {k: (f"{d[k]:5.2f}" if k in d and d[k] is not None else "   --")
                      for k in ("L", "R", "B")}
                nm = f"{res.near_min:5.2f}" if res.near_min is not None else " None"
                cl = f"{res.closest:5.2f}" if res.closest is not None else "   --"
                avg = float(np.mean(proc_times[-30:])) if proc_times else 0.0
                fps = f"{1000/avg:4.1f} к/с" if avg else "--"
                print(f"кадр {n_frames:5d}  L={ds['L']} R={ds['R']} "
                      f"B={ds['B']}  MIN={cl}  NEAR={nm}   "
                      f"обр. {avg:5.1f} мс ({fps})", flush=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # ---- итоговая статистика -----------------------------------------
        if proc_times:
            arr = np.array(proc_times)
            print(f"\nn = {n_frames} кадров   обработка: ср. {arr.mean():.1f} мс, "
                  f"p95 {np.percentile(arr, 95):.1f} мс, "
                  f"макс {arr.max():.1f} мс  ->  {1000/arr.mean():.1f} к/с")


if __name__ == "__main__":
    main()
