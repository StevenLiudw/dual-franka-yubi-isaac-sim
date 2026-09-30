"""Conservative dual Panda link clearance guard for one 10 Hz command.

The controller checks the swept motion of padded link center points using
the current articulation Jacobians.  It scales joint increments when their
linearized motion would reduce inter-arm clearance below the margin.  This is
an online safety layer, not a substitute for mesh-level collision checking or
hardware emergency stops.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


INTERARM_MARGIN_M = 0.02
SCALES = (1.0, 0.75, 0.5, 0.25, 0.0)
SWEEP_FRACTIONS = np.linspace(0.0, 1.0, 5)


def _points(robot: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entries = robot.get("collision_points")
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError("collision_points must contain the current Panda link samples")
    positions = np.asarray([entry["position_m"] for entry in entries], dtype=np.float64)
    radii = np.asarray([entry["radius_m"] for entry in entries], dtype=np.float64)
    jacobians = np.asarray([entry["arm_translation_jacobian"] for entry in entries], dtype=np.float64)
    if (positions.shape != (len(entries), 3) or radii.shape != (len(entries),)
            or jacobians.shape != (len(entries), 3, 7)
            or not np.isfinite(positions).all() or not np.isfinite(radii).all()
            or not np.isfinite(jacobians).all() or np.any(radii <= 0)):
        raise ValueError("collision_points contain invalid positions, radii, or Jacobians")
    # Midpoints close gaps between sampled arm-link origins.  Their Jacobian
    # and radius are interpolated from the endpoint samples.
    names = [str(entry.get("name", "")) for entry in entries]
    for first, second in (
        ("panda_link2", "panda_link3"), ("panda_link3", "panda_link4"),
        ("panda_link4", "panda_link5"), ("panda_link5", "panda_link6"),
        ("panda_link6", "panda_link7"), ("panda_link7", "yubi_base"),
        ("yubi_base", "yubi_leftfinger"), ("yubi_base", "yubi_rightfinger"),
    ):
        if first not in names or second not in names:
            continue
        a, b = names.index(first), names.index(second)
        positions = np.vstack((positions, (positions[a] + positions[b]) / 2))
        radii = np.append(radii, (radii[a] + radii[b]) / 2)
        jacobians = np.concatenate((jacobians, ((jacobians[a] + jacobians[b]) / 2)[None]))
    return positions, radii, jacobians


def _clearance(left: np.ndarray, right: np.ndarray, radii: np.ndarray) -> float:
    distance = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float(np.min(distance - radii))


def observed_interarm_clearance_m(observation: Mapping) -> float | None:
    """Minimum padded-link clearance in a measured two-arm observation."""
    robots = observation.get("robots", {})
    if not all(side in robots and "collision_points" in robots[side] for side in ("left", "right")):
        return None
    left_p, left_r, _ = _points(robots["left"])
    right_p, right_r, _ = _points(robots["right"])
    return _clearance(left_p, right_p, left_r[:, None] + right_r[None, :])


def limit_interarm_motion(observation: Mapping, action: dict[str, dict],
                          *, margin_m: float = INTERARM_MARGIN_M) -> tuple[dict[str, dict], dict]:
    """Return guarded joint commands and a compact clearance diagnostic.

    An already close configuration may move only if its predicted clearance
    never decreases.  Candidate scales are checked through the full 10 Hz
    interval, so simply safe endpoints do not hide a mid-step crossing.
    """
    if not math.isfinite(margin_m) or margin_m < 0:
        raise ValueError("margin_m must be finite and nonnegative")
    robots = observation.get("robots", {})
    available = [side in robots and "collision_points" in robots[side] for side in ("left", "right")]
    if not any(available):
        return action, {"active": False}
    if not all(available):
        raise ValueError("Both Panda arms need collision_points for the inter-arm guard")

    prepared = {}
    for side in ("left", "right"):
        robot = robots[side]
        position, radius, jacobian = _points(robot)
        names = robot.get("joint_names", ())
        q = np.asarray(robot.get("joint_positions"), dtype=np.float64)
        if q.shape != (len(names),):
            raise ValueError(f"{side} joint_positions and joint_names differ")
        arm_indices = [names.index(f"panda_joint{i}") for i in range(1, 8)]
        arm_q = q[arm_indices]
        target = np.asarray(action.get(side, {}).get("arm_joint_targets_rad", arm_q), dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError(f"{side} arm target must contain seven finite joint angles")
        prepared[side] = (position, radius, jacobian @ (target - arm_q), arm_q, target)

    left_p, left_r, left_delta, left_q, left_target = prepared["left"]
    right_p, right_r, right_delta, right_q, right_target = prepared["right"]
    combined_radii = left_r[:, None] + right_r[None, :]
    current = _clearance(left_p, right_p, combined_radii)

    candidates = sorted(((l, r) for l in SCALES for r in SCALES),
                        key=lambda pair: (pair[0] + pair[1], min(pair)), reverse=True)
    chosen = (0.0, 0.0)
    predicted = current
    for left_scale, right_scale in candidates:
        minimum = min(
            _clearance(left_p + fraction * left_scale * left_delta,
                       right_p + fraction * right_scale * right_delta, combined_radii)
            for fraction in SWEEP_FRACTIONS
        )
        if minimum >= (margin_m if current >= margin_m else current - 1e-4):
            chosen = (left_scale, right_scale)
            predicted = minimum
            break

    guarded = {side: dict(command) for side, command in action.items()}
    for side, scale, q, target in (
        ("left", chosen[0], left_q, left_target),
        ("right", chosen[1], right_q, right_target),
    ):
        if side in guarded and "arm_joint_targets_rad" in guarded[side]:
            guarded[side]["arm_joint_targets_rad"] = (q + scale * (target - q)).tolist()
        if scale == 0.0 and current < margin_m and side in guarded and "gripper_open_fraction" in guarded[side]:
            guarded[side]["gripper_open_fraction"] = float(np.clip(
                robots[side]["gripper_open_fraction"], 0.0, 1.0
            ))
    return guarded, {
        "active": True,
        "current_clearance_m": current,
        "predicted_swept_clearance_m": predicted,
        "left_scale": chosen[0],
        "right_scale": chosen[1],
    }


__all__ = ["INTERARM_MARGIN_M", "limit_interarm_motion", "observed_interarm_clearance_m"]
