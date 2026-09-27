#!/usr/bin/env python3
"""Seeded, collision-safe random initial layouts for the cup-on-plate task.

The plate center is uniform over a central disk. The cup center is uniform over
an annulus around that plate, subject to tabletop clearance. Colors are sampled
independently. The 32-bit generator is mirrored in the web preview for repeatable
layouts across Python and JavaScript.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
UINT32 = 0xFFFFFFFF


class Mulberry32:
    def __init__(self, seed: int):
        self.state = int(seed) & UINT32

    def random(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & UINT32
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & UINT32
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & UINT32)) & UINT32)) & UINT32
        return ((t ^ (t >> 14)) & UINT32) / 4294967296.0


def _inside_table(x: float, y: float, radius: float, edge_margin: float, table: dict) -> bool:
    return (
        abs(x) + radius + edge_margin <= table["depth"] / 2
        and abs(y) + radius + edge_margin <= table["width"] / 2
    )


def generate_random_scenarios(config: dict, count: int, seed: int) -> list[dict]:
    """Return initial poses/colors; raises if requested region is infeasible.

    Draw order per scenario (for cross-language reproducibility): plate radial U,
    plate angle U, then cup radial/angle U pairs until feasible, then cup and
    tray color index U. Radial coordinates use square-root area sampling.
    """
    if count < 0:
        raise ValueError("count must be nonnegative")
    table, cup, tray, spec = (config[key] for key in ("table", "cup", "tray", "randomization"))
    center_radius = float(spec["plate_center_region_radius"])
    clearance = float(spec["cup_clearance_m"])
    margin = float(spec["table_edge_margin_m"])
    r_min = float(tray["outer_radius"]) + float(cup["top_radius"]) + clearance
    r_max = float(spec["cup_max_distance_m"])
    if center_radius < 0 or clearance < 0 or margin < 0 or r_min > r_max:
        raise ValueError("Invalid randomization radii or clearances")
    for name in ("cup_colors", "tray_colors"):
        if not spec.get(name):
            raise ValueError(f"randomization.{name} cannot be empty")
    if not _inside_table(center_radius, 0, tray["outer_radius"], margin, table):
        raise ValueError("Central plate region does not fit inside the table along X")
    if not _inside_table(0, center_radius, tray["outer_radius"], margin, table):
        raise ValueError("Central plate region does not fit inside the table along Y")

    rng = Mulberry32(seed)
    results = []
    for i in range(count):
        plate_r = center_radius * math.sqrt(rng.random())
        plate_theta = 2 * math.pi * rng.random()
        px = plate_r * math.cos(plate_theta)
        py = plate_r * math.sin(plate_theta)
        for _ in range(10000):
            cup_r = math.sqrt(r_min * r_min + rng.random() * (r_max * r_max - r_min * r_min))
            cup_theta = 2 * math.pi * rng.random()
            cx = px + cup_r * math.cos(cup_theta)
            cy = py + cup_r * math.sin(cup_theta)
            if _inside_table(cx, cy, cup["top_radius"], margin, table):
                break
        else:
            raise ValueError("No feasible cup pose found; adjust table size or randomization region")
        cup_palette = spec["cup_colors"]
        tray_palette = spec["tray_colors"]
        results.append(
            {
                "id": f"random_{i:03d}_seed_{int(seed) & UINT32}",
                "seed": int(seed) & UINT32,
                "index": i,
                "cup_xy": [round(cx, 6), round(cy, 6)],
                "tray_xy": [round(px, 6), round(py, 6)],
                "cup_color": cup_palette[min(len(cup_palette) - 1, int(rng.random() * len(cup_palette)))],
                "tray_color": tray_palette[min(len(tray_palette) - 1, int(rng.random() * len(tray_palette)))],
                "placement_status": "seeded synthetic initial state; no depth or measured camera pose",
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads((PROJECT / "scene_config.json").read_text())
    seed = args.seed if args.seed is not None else config["randomization"]["seed"]
    result = generate_random_scenarios(config, args.count, seed)
    body = json.dumps({"seed": seed, "scenarios": result}, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body)
        print(args.output)
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
