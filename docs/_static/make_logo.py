# /// script
# requires-python = ">=3.11"
# dependencies = ["resvg-py"]
# ///
# A standalone script, not part of a package; its PEP 723 header above is metadata,
# not commented-out code.
# ruff: noqa: INP001, ERA001
"""Draw the trackstream logo: a railway track laid on a circular stream.

A stellar stream wraps round in a ring of grey stars, scattered from a fixed
seed. Along three quarters of it runs a black railway track, the stream's track,
broken at the bottom by the progenitor; past the track's ends the stars run on.
It redraws the 2020 logo (``stream-track.png`` on the ``logo`` branch) as a
circle. The shapes are vector, so the logo is written as an SVG, sharp at any
size; for a bitmap, name a .png and give its size::

    uv run docs/_static/make_logo.py                     # favicon.svg
    uv run docs/_static/make_logo.py --size 2048 big.png
"""

import argparse
import math
from pathlib import Path
import random

INK, STARS_COLOUR = "#000000", "#8b949e"

# In a 64-unit square: the stream's centre and radius.
CENTRE, RADIUS = 32, 21
STARS, SEED, SPREAD = 260, 1, 2.2  # the stars, their draw, their spread about it
STAR_SIZE = (0.5, 1.3)  # the stars' smallest and largest radius
# The track: where it starts and ends (degrees, clockwise from the right), and
# the gap round the progenitor at the bottom. Its rails' half-separation, its
# sleepers' half-length and spacing.
TRACK, GAP = (-45, 225), (84, 96)
GAUGE, SLEEPER, SLEEPER_STEP = 1.6, 2.6, 8
PROGENITOR = (90, 2.4)  # where (degrees) and its radius

SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="512" height="512">
{stars}
  <path d="{rails}" fill="none" stroke="{ink}" stroke-width="0.9"/>
  <path d="{sleepers}" stroke="{ink}" stroke-width="1.1" stroke-linecap="round"/>
  <circle cx="{px:.2f}" cy="{py:.2f}" r="{pr:g}" fill="{ink}"/>
</svg>
"""


def at(radius: float, degrees: float) -> tuple[float, float]:
    """Return the point at ``radius`` and angle ``degrees`` about the centre."""
    a = math.radians(degrees)
    return CENTRE + radius * math.cos(a), CENTRE + radius * math.sin(a)


def stars() -> str:
    """Return the stream: stars scattered about the ring, as SVG circles."""
    rng = random.Random(SEED)  # noqa: S311  # scatters stars; not cryptographic
    circles = []
    for _ in range(STARS):
        angle = rng.uniform(0, 360)
        x, y = at(RADIUS + rng.gauss(0, SPREAD), angle)
        size = rng.uniform(*STAR_SIZE)
        circles.append(
            f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" fill="{STARS_COLOUR}" opacity="0.8"/>',
        )
    return "\n".join(circles)


def spans() -> list[tuple[float, float]]:
    """Return the track's stretches either side of the progenitor's gap."""
    (start, end), (gap_start, gap_end) = TRACK, GAP
    return [(start, gap_start), (gap_end, end)]


def rails() -> str:
    """Return both rails, each an arc, along every stretch of track."""
    paths = []
    for start, end in spans():
        for r in (RADIUS - GAUGE, RADIUS + GAUGE):
            (x0, y0), (x1, y1) = at(r, start), at(r, end)
            large = 1 if end - start > 180 else 0
            paths.append(f"M{x0:.2f} {y0:.2f}A{r:g} {r:g} 0 {large} 1 {x1:.2f} {y1:.2f}")
    return "".join(paths)


def sleepers() -> str:
    """Return the sleepers across the rails, evenly spaced along each stretch."""
    paths = []
    for start, end in spans():
        angle = start + SLEEPER_STEP / 2
        while angle < end - 2:
            (x0, y0), (x1, y1) = at(RADIUS - SLEEPER, angle), at(RADIUS + SLEEPER, angle)
            paths.append(f"M{x0:.2f} {y0:.2f}L{x1:.2f} {y1:.2f}")
            angle += SLEEPER_STEP
    return "".join(paths)


def svg() -> str:
    """Return the logo as SVG text."""
    where, size = PROGENITOR
    px, py = at(RADIUS, where)
    return SVG.format(
        stars=stars(),
        rails=rails(),
        sleepers=sleepers(),
        ink=INK,
        px=px,
        py=py,
        pr=size,
    )


def main() -> None:
    """Parse the command line and save the logo."""
    parser = argparse.ArgumentParser(description="Draw the trackstream logo.")
    parser.add_argument(
        "out",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("favicon.svg"),
        help="output file, SVG or PNG by its extension (default: favicon.svg)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="pixels per side, for a PNG",
    )
    args = parser.parse_args()

    if args.out.suffix == ".svg":
        args.out.write_text(svg())
    else:
        import resvg_py  # only a PNG needs a renderer

        png = resvg_py.svg_to_bytes(svg_string=svg(), width=args.size)
        args.out.write_bytes(bytes(png))


if __name__ == "__main__":
    main()
