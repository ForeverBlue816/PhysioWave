#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Draw the PhysioWave series logo: assets/logo/{mark,banner_light,banner_dark}.svg

    python scripts/make_logo.py

The mark is scale folding made visible: three wavelet bands -- a fast,
a middle and a slow one, the detail and approximation scales -- run in from
the left, meet in the fold, and leave as one physiological waveform with a
QRS-like spike. Colours are the journal palette of the paper's figures.
"""

from __future__ import annotations

import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "logo")

INDIGO, INDIGO_DEEP = "#3F4280", "#262A5C"
LAVENDER, BLUE, PALE = "#8A84BF", "#7F97C2", "#BBD0E8"
MAUVE, GOLD = "#C3609E", "#DCA94F"


def band(x0, x1, y, amp, cycles, fold_x, fold_y, n=120):
    """A sine band from x0 to x1, then a smooth bend into the fold point."""
    pts = []
    for i in range(n + 1):
        t = i / n
        x = x0 + (x1 - x0) * t
        env = math.sin(math.pi * min(1.0, t * 1.15)) ** 0.6     # taper at both ends
        pts.append((x, y + amp * env * math.sin(2 * math.pi * cycles * t)))
    d = "M" + " L".join(f"{x:.2f} {yy:.2f}" for x, yy in pts)
    # cubic into the fold: leave horizontally, arrive horizontally
    cx = (x1 + fold_x) / 2
    d += f" C{cx:.2f} {y:.2f} {cx:.2f} {fold_y:.2f} {fold_x:.2f} {fold_y:.2f}"
    return d


def signal(x0, x1, y):
    """The folded output: baseline, a P wave, a QRS spike, a T wave."""
    w = x1 - x0
    k = lambda f: x0 + w * f
    return (f"M{x0:.2f} {y:.2f} "
            f"C{k(.06):.2f} {y:.2f} {k(.08):.2f} {y - 5:.2f} {k(.13):.2f} {y - 5:.2f} "
            f"C{k(.18):.2f} {y - 5:.2f} {k(.20):.2f} {y:.2f} {k(.25):.2f} {y:.2f} "
            f"L{k(.31):.2f} {y + 5:.2f} L{k(.38):.2f} {y - 30:.2f} L{k(.45):.2f} {y + 13:.2f} "
            f"L{k(.50):.2f} {y:.2f} "
            f"C{k(.58):.2f} {y:.2f} {k(.62):.2f} {y - 9:.2f} {k(.70):.2f} {y - 9:.2f} "
            f"C{k(.78):.2f} {y - 9:.2f} {k(.82):.2f} {y:.2f} {k(.90):.2f} {y:.2f} "
            f"L{x1:.2f} {y:.2f}")


def mark(ox=0.0, oy=0.0, s=1.0, uid="m"):
    """The 128 x 128 mark, translated and scaled."""
    fx, fy = 62.0, 66.0
    bands = [  # (y, amplitude, cycles, colour, width): fast detail to slow approximation
        (40.0, 3.4, 5.0, PALE, 2.8),
        (66.0, 5.5, 3.5, GOLD, 3.4),
        (92.0, 7.0, 1.6, MAUVE, 3.6),
    ]
    g = [f'<g transform="translate({ox} {oy}) scale({s})">',
         f'<defs><linearGradient id="{uid}bg" x1="0" y1="0" x2="1" y2="1">'
         f'<stop offset="0" stop-color="{INDIGO}"/><stop offset="1" stop-color="{INDIGO_DEEP}"/>'
         f'</linearGradient>'
         f'<radialGradient id="{uid}glow" cx="0.5" cy="0.5" r="0.5">'
         f'<stop offset="0" stop-color="#FFFFFF" stop-opacity="0.55"/>'
         f'<stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/></radialGradient></defs>',
         f'<rect x="0" y="0" width="128" height="128" rx="30" fill="url(#{uid}bg)"/>',
         # faint scale grid, three rows, like the subband rows of a decomposition
         *[f'<line x1="14" y1="{y}" x2="56" y2="{y}" stroke="#FFFFFF" stroke-opacity="0.08" '
           f'stroke-width="1"/>' for y, *_ in bands]]
    for y, amp, cyc, col, wid in bands:
        g.append(f'<path d="{band(14, 44, y, amp, cyc, fx, fy)}" fill="none" stroke="{col}" '
                 f'stroke-width="{wid}" stroke-linecap="round" stroke-linejoin="round"/>')
    g.append(f'<circle cx="{fx}" cy="{fy}" r="13" fill="url(#{uid}glow)"/>')
    g.append(f'<circle cx="{fx}" cy="{fy}" r="5.2" fill="#FFFFFF"/>')
    g.append(f'<path d="{signal(fx, 114, fy)}" fill="none" stroke="#FFFFFF" stroke-width="4.6" '
             f'stroke-linecap="round" stroke-linejoin="round"/>')
    g.append("</g>")
    return "\n".join(g)


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{title}">\n<title>{title}</title>\n'
            f'{body}\n</svg>\n')


FONT = ("font-family=\"'Inter','Helvetica Neue','Segoe UI',Arial,sans-serif\"")


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
