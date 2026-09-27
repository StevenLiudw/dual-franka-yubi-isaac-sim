"""Resolve reusable cup/plate starting layouts for the Isaac Sim environment.

All packaged layouts are data files under ``setups/``. A custom JSON file or
dictionary can use the same flat fields, or ``cup``/``plate`` objects with
``xy_m`` or ``position_m``, ``quaternion_wxyz``, and ``color`` fields.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

from .random_scenarios import generate_random_scenarios


PACKAGE_ROOT = Path(__file__).resolve().parent
SETUP_DIR = PACKAGE_ROOT / "setups"
DEFAULT_SEED = 20260924


def _color(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 7 or not value.startswith("#"):
        raise ValueError(f"{label} must be a #RRGGBB color")
    try:
        int(value[1:], 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a #RRGGBB color") from exc
    return value.lower()


def _vector(value: Any, size: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must have {size} numbers")
    values = [float(item) for item in value]
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{label} must contain finite numbers")
    return values


def normalize_setup(raw: dict, scene_config: dict, default_id: str = "custom") -> dict:
    """Validate a setup and return the legacy scenario fields plus full poses."""
    if not isinstance(raw, dict):
        raise TypeError("A setup must be a JSON object/dictionary")
    table_z = float(scene_config["table"]["top_z"])
    result = dict(raw)
    result.setdefault("id", default_id)
    result.setdefault("placement_status", "user-specified object setup; not a measured pose")
    for name, flat_name, color_name in (
        ("cup", "cup", "cup_color"),
        ("plate", "tray", "tray_color"),
    ):
        object_spec = raw.get(name, {})
        if object_spec is None:
            object_spec = {}
        if not isinstance(object_spec, dict):
            raise ValueError(f"{name} must be an object")
        position = object_spec.get("position_m", raw.get(f"{flat_name}_position_m"))
        xy = object_spec.get("xy_m", raw.get(f"{flat_name}_xy"))
        if position is None:
            if xy is None:
                raise ValueError(f"{name} requires xy_m/position_m (or {flat_name}_xy)")
            position = [*_vector(xy, 2, f"{name}.xy_m"), table_z]
        position = _vector(position, 3, f"{name}.position_m")
        quaternion = object_spec.get(
            "quaternion_wxyz", raw.get(f"{flat_name}_quaternion_wxyz", [1, 0, 0, 0])
        )
        quaternion = _vector(quaternion, 4, f"{name}.quaternion_wxyz")
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm < 1e-9:
            raise ValueError(f"{name}.quaternion_wxyz must be nonzero")
        quaternion = [value / norm for value in quaternion]
        default_color = (
            scene_config["cup"]["color"]
            if name == "cup"
            else scene_config["randomization"]["tray_colors"][0]
        )
        color = _color(object_spec.get("color", raw.get(color_name, default_color)), f"{name}.color")
        result[f"{flat_name}_xy"] = position[:2]
        result[f"{flat_name}_position_m"] = position
        result[f"{flat_name}_quaternion_wxyz"] = quaternion
        result[color_name] = color
        if position[2] < table_z - 0.001:
            raise ValueError(f"{name} bottom is below the tabletop by more than 1 mm")
        radius = float(scene_config["cup"]["top_radius"] if name == "cup" else scene_config["tray"]["outer_radius"])
        table = scene_config["table"]
        if abs(position[0]) + radius > float(table["depth"]) / 2 or abs(position[1]) + radius > float(table["width"]) / 2:
            raise ValueError(f"{name} footprint extends beyond the tabletop")
    horizontal_separation = math.dist(result["cup_xy"], result["tray_xy"])
    overlap_distance = float(scene_config["cup"]["top_radius"]) + float(scene_config["tray"]["outer_radius"])
    low_cup = result["cup_position_m"][2] < result["tray_position_m"][2] + float(scene_config["tray"]["height"])
    cup_inside_opening = (
        horizontal_separation + float(scene_config["cup"]["top_radius"])
        <= float(scene_config["tray"]["inner_radius"])
        and result["cup_position_m"][2]
        >= result["tray_position_m"][2] + float(scene_config["tray"]["base_thickness"]) - 0.001
    )
    if horizontal_separation < overlap_distance and low_cup and not cup_inside_opening:
        warning = (
            f"cup/plate footprints overlap by {overlap_distance - horizontal_separation:.4f} m "
            "near tabletop height; this starting pose may be unsafe for physics"
        )
        result.setdefault("safety_warnings", []).append(warning)
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
    return result


def list_setups() -> list[str]:
    """List the packaged setup names, such as ``random:0``."""
    names = []
    for path in sorted(SETUP_DIR.glob("*.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        names.append(str(body.get("name", path.stem)))
    return names


def resolve_setup(
    setup: str | Path | dict | None = None,
    *,
    scene_config: dict | None = None,
    seed: int = DEFAULT_SEED,
    scenario_index: int = 0,
) -> dict:
    """Select a named, custom, or deterministic random object setup.

    ``setup=None`` preserves the original ``reset(seed, scenario_index)``
    behavior. ``random:N`` uses the provided seed, with the packaged JSON
    preset used as a reproducibility reference when that seed is the default.
    """
    if scene_config is None:
        scene_config = json.loads((PACKAGE_ROOT / "scene_config.json").read_text(encoding="utf-8"))
    if isinstance(setup, dict):
        raw = setup
        default_id = "custom"
    elif setup is None:
        if scenario_index < 0:
            raise ValueError("scenario_index must be nonnegative")
        raw = generate_random_scenarios(scene_config, scenario_index + 1, int(seed))[-1]
        default_id = raw["id"]
    else:
        name = str(setup)
        candidate = Path(name).expanduser()
        if candidate.is_file():
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            default_id = candidate.stem
        elif name.startswith("random:"):
            try:
                index = int(name.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid random setup name: {name}") from exc
            if index < 0:
                raise ValueError("Random setup index must be nonnegative")
            raw = generate_random_scenarios(scene_config, index + 1, int(seed))[-1]
            default_id = raw["id"]
        elif name.startswith(("episode:", "physics_safe_episode:")):
            prefix, episode_id = name.split(":", 1)
            candidate = SETUP_DIR / f"{prefix}_{episode_id}.json"
            if not candidate.is_file():
                raise KeyError(f"Unknown episode setup: {name}")
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            default_id = f"{prefix}_{episode_id}"
        else:
            candidate = SETUP_DIR / f"{name}.json"
            if not candidate.is_file():
                options = ", ".join(list_setups())
                raise KeyError(f"Unknown setup {name!r}. Packaged setups: {options}")
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            default_id = candidate.stem
    return normalize_setup(raw, scene_config, default_id=default_id)
