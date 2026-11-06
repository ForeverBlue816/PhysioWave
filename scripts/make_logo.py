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
    """A sine band from x0 to x1 that tapers to a flat line, then a long S-curve
    into the fold, level where it leaves and level where it arrives."""
    pts = []
    for i in range(n + 1):
        t = i / n
        x = x0 + (x1 - x0) * t
        env = math.sin(math.pi * min(1.0, t * 1.12)) ** 0.7     # taper at both ends
        pts.append((x, y + amp * env * math.sin(2 * math.pi * cycles * t)))
    d = "M" + " L".join(f"{x:.2f} {yy:.2f}" for x, yy in pts)
    span = fold_x - x1
    d += (f" C{x1 + 0.55 * span:.2f} {y:.2f} {fold_x - 0.45 * span:.2f} {fold_y:.2f} "
          f"{fold_x:.2f} {fold_y:.2f}")
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
    """The 128 x 128 mark, translated and scaled.

    Each band fades from its own colour to white over its curve into the
    fold, so the three arrive as one white line and leave as the waveform;
    no node marks the junction, only a soft glow.
    """
    x0, x1, fx, fy = 14.0, 37.0, 66.0, 66.0
    bands = [  # (y, amplitude, cycles, colour, width): fast detail to slow approximation
        (38.0, 3.0, 4.5, PALE, 2.9),
        (66.0, 5.0, 3.0, GOLD, 3.2),
        (94.0, 6.5, 1.5, MAUVE, 3.4),
    ]
    defs = [f'<linearGradient id="{uid}bg" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0" stop-color="{INDIGO}"/><stop offset="1" stop-color="{INDIGO_DEEP}"/>'
            f'</linearGradient>',
            f'<linearGradient id="{uid}sheen" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="#FFFFFF" stop-opacity="0.13"/>'
            f'<stop offset="0.55" stop-color="#FFFFFF" stop-opacity="0"/></linearGradient>',
            f'<radialGradient id="{uid}glow" cx="0.5" cy="0.5" r="0.5">'
            f'<stop offset="0" stop-color="#FFFFFF" stop-opacity="0.32"/>'
            f'<stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/></radialGradient>']
    for k, (_y, _a, _c, col, _w) in enumerate(bands):
        defs.append(f'<linearGradient id="{uid}b{k}" gradientUnits="userSpaceOnUse" '
                    f'x1="{x1 - 2}" y1="0" x2="{fx}" y2="0">'
                    f'<stop offset="0" stop-color="{col}"/>'
                    f'<stop offset="1" stop-color="#FFFFFF"/></linearGradient>')
    g = [f'<g transform="translate({ox} {oy}) scale({s})">',
         "<defs>" + "".join(defs) + "</defs>",
         f'<rect x="0" y="0" width="128" height="128" rx="30" fill="url(#{uid}bg)"/>',
         f'<rect x="0" y="0" width="128" height="128" rx="30" fill="url(#{uid}sheen)"/>',
         f'<circle cx="{fx}" cy="{fy}" r="16" fill="url(#{uid}glow)"/>']
    for k, (y, amp, cyc, _col, wid) in enumerate(bands):
        g.append(f'<path d="{band(x0, x1, y, amp, cyc, fx, fy)}" fill="none" '
                 f'stroke="url(#{uid}b{k})" stroke-width="{wid}" stroke-linecap="round" '
                 f'stroke-linejoin="round"/>')
    g.append(f'<path d="{signal(fx, 114, fy)}" fill="none" stroke="#FFFFFF" stroke-width="4.2" '
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
