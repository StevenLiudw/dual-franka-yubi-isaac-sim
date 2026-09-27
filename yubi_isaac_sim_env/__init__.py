"""Isolated GPU Isaac Sim launcher for the dual-Franka/YUBI cup scene.

The sibling package owns its scenes, assets, configuration, and setup catalog.
Importing this module does not start Isaac Sim.
"""

from __future__ import annotations

import os
from pathlib import Path

from .env import DualFrankaYubiCupPlateEnv
from .head_camera import configure_head_camera
from .setup_catalog import list_setups


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = "dual_franka_yubi_random_000_seed_20260924"


def resolve_scene(scene: str | Path = DEFAULT_SCENE) -> Path:
    """Resolve a bundled YUBI scene name or an explicit stage path."""
    requested = Path(scene).expanduser()
    if requested.is_file():
        return requested.resolve()
    if requested.is_absolute():
        raise FileNotFoundError(requested)
    candidate = PACKAGE_ROOT / "scenes" / f"{requested.stem}.usda"
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"YUBI scene {scene!r} not found under {PACKAGE_ROOT / 'scenes'}")


def create_sim(
    scene: str | Path = DEFAULT_SCENE,
    *,
    gui: bool = True,
    setup: str | Path | dict | None = None,
    seed: int = 20260924,
    scenario_index: int = 0,
) -> tuple[object, DualFrankaYubiCupPlateEnv]:
    """Launch Isaac Sim with GPU physics and return ``(app, env)``.

    The caller owns the returned SimulationApp and should close it when done.
    Each ``env.reset(setup=...)`` starts another episode without restarting it.
    """
    scene_path = resolve_scene(scene)
    if gui and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("GUI requested but no DISPLAY or WAYLAND_DISPLAY is set")
    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": not gui, "active_gpu": 0, "physics_gpu": 0, "renderer": "RaytracedLighting"}
    )
    import omni.usd

    context = omni.usd.get_context()
    if not context.open_stage(str(scene_path)):
        app.close()
        raise RuntimeError(f"Isaac Sim could not open YUBI scene: {scene_path}")
    configure_head_camera(context.get_stage())
    for _ in range(10):
        app.update()
    env = DualFrankaYubiCupPlateEnv(render=gui)
    env.reset(seed=seed, scenario_index=scenario_index, setup=setup)
    return app, env


__all__ = ["DualFrankaYubiCupPlateEnv", "create_sim", "resolve_scene", "list_setups"]
