#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Draw the PhysioWave series logo: assets/logo/{mark,banner_light,banner_dark}.svg

    python scripts/make_logo.py

The mark is a Morlet wavelet. Read left to right it traces a W -- down, up to
the central peak, down, up -- so the W of PhysioWave is the wavelet itself.
The stroke swells at the peak and tapers into the tails, and its colour runs
through the three scale colours of the paper's figures: pale blue, gold,
mauve. One shape, no joins; it reads at 16 px and at 1024 px.
"""

from __future__ import annotations

import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "logo")

INDIGO, INDIGO_DEEP = "#3F4280", "#262A5C"
PALE, GOLD, MAUVE = "#BBD0E8", "#DCA94F", "#C3609E"


def morlet(t: float, sigma: float = 0.62, freq: float = 1.0) -> float:
    return math.exp(-t * t / (2 * sigma * sigma)) * math.cos(2 * math.pi * freq * t)


def wavelet_ribbon(cx, cy, half_w, amp, w_min=1.6, w_max=7.4, n=260, t_max=1.55):
    """A closed path: the Morlet curve offset along its normal by a width that
    follows the Gaussian envelope, so the ribbon is thick at the peak and
    fades into the tails."""
    pts = []
    for i in range(n + 1):
        t = -t_max + 2 * t_max * i / n
        x = cx + half_w * t / t_max
        y = cy - amp * morlet(t)
        pts.append((t, x, y))
    upper, lower = [], []
    for k, (t, x, y) in enumerate(pts):
        x0, y0 = pts[max(k - 1, 0)][1:]
        x1, y1 = pts[min(k + 1, n)][1:]
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / norm, dx / norm
        env = math.exp(-t * t / (2 * 0.62 ** 2))
        w = w_min + (w_max - w_min) * env ** 1.3
        upper.append((x + nx * w / 2, y + ny * w / 2))
        lower.append((x - nx * w / 2, y - ny * w / 2))
    path = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in upper)
    path += " L" + " L".join(f"{x:.2f} {y:.2f}" for x, y in reversed(lower)) + " Z"
    peak = (cx, cy - amp)
    return path, peak


def mark(ox=0.0, oy=0.0, s=1.0, uid="m"):
    """The 128 x 128 mark, translated and scaled."""
    cx, cy = 64.0, 68.0
    ribbon, (px, py) = wavelet_ribbon(cx, cy, half_w=50.0, amp=30.0)
    g = [f'<g transform="translate({ox} {oy}) scale({s})">',
         "<defs>",
         f'<linearGradient id="{uid}bg" x1="0" y1="0" x2="1" y2="1">'
         f'<stop offset="0" stop-color="{INDIGO}"/><stop offset="1" stop-color="{INDIGO_DEEP}"/>'
         f'</linearGradient>',
         f'<linearGradient id="{uid}sheen" x1="0" y1="0" x2="0" y2="1">'
         f'<stop offset="0" stop-color="#FFFFFF" stop-opacity="0.14"/>'
         f'<stop offset="0.6" stop-color="#FFFFFF" stop-opacity="0"/></linearGradient>',
         f'<linearGradient id="{uid}wave" gradientUnits="userSpaceOnUse" '
         f'x1="{cx - 50}" y1="0" x2="{cx + 50}" y2="0">'
         f'<stop offset="0" stop-color="{PALE}"/><stop offset="0.5" stop-color="{GOLD}"/>'
         f'<stop offset="1" stop-color="{MAUVE}"/></linearGradient>',
         f'<radialGradient id="{uid}glow" cx="0.5" cy="0.5" r="0.5">'
         f'<stop offset="0" stop-color="{GOLD}" stop-opacity="0.45"/>'
         f'<stop offset="1" stop-color="{GOLD}" stop-opacity="0"/></radialGradient>',
         "</defs>",
         f'<rect width="128" height="128" rx="30" fill="url(#{uid}bg)"/>',
         f'<rect width="128" height="128" rx="30" fill="url(#{uid}sheen)"/>',
         # the baseline the wavelet oscillates about
         f'<line x1="16" y1="{cy}" x2="112" y2="{cy}" stroke="#FFFFFF" stroke-opacity="0.16" '
         f'stroke-width="1.2" stroke-linecap="round"/>',
         f'<ellipse cx="{px}" cy="{py + 10}" rx="30" ry="22" fill="url(#{uid}glow)"/>',
         f'<path d="{ribbon}" fill="url(#{uid}wave)"/>',
         "</g>"]
    return "\n".join(g)


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{title}">\n<title>{title}</title>\n'
            f'{body}\n</svg>\n')


FONT = "font-family=\"'Inter','Helvetica Neue','Segoe UI',Arial,sans-serif\""


def banner(dark: bool):
    ink = "#EEF1F6" if dark else "#23264F"
    sub = "#AEB6CC" if dark else "#5A6078"
    accent = "#E58BC1" if dark else MAUVE
    body = [mark(0, 8, 1.0, uid="d" if dark else "l"),
            f'<text x="152" y="86" {FONT} font-size="60" font-weight="700" fill="{ink}" '
            f'letter-spacing="-1">Physio<tspan fill="{accent}">Wave</tspan></text>',
            f'<text x="155" y="122" {FONT} font-size="19" font-weight="500" fill="{sub}" '
            f'letter-spacing="0.4">Wavelet foundation models for EEG · ECG · sEMG</text>']
    return svg(640, 144, "\n".join(body), "PhysioWave")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    files = {"mark.svg": svg(128, 128, mark(), "PhysioWave"),
             "banner_light.svg": banner(False), "banner_dark.svg": banner(True)}
    for name, text in files.items():
        with open(os.path.join(OUT, name), "w") as f:
            f.write(text)
        print("wrote", os.path.join("assets", "logo", name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
