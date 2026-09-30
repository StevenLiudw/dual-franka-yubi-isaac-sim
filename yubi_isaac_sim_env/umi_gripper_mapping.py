"""Fixed operator-angle to motorized-aperture mapping, independent of episode."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np

CALIBRATION_PATH = Path(__file__).with_name("gripper_aperture_calibration.json")
CALIBRATION = json.loads(CALIBRATION_PATH.read_text())
SOURCE_Q = np.array(CALIBRATION["source_angle_rad"])
SOURCE_GAP = np.array(CALIBRATION["source_aperture_m"])
MOTOR_Q = np.array(CALIBRATION["motorized_angle_rad"])
MOTOR_GAP = np.array(CALIBRATION["motorized_aperture_m"])
if not (len(SOURCE_Q) == len(SOURCE_GAP) and len(MOTOR_Q) == len(MOTOR_GAP)
        and np.all(np.diff(SOURCE_Q) > 0) and np.all(np.diff(MOTOR_Q) > 0)
        and np.all(np.diff(SOURCE_GAP) >= 0) and np.all(np.diff(MOTOR_GAP) > 0)):
    raise ValueError("Invalid monotone aperture calibration")


def source_aperture_m(source_angles_rad) -> np.ndarray:
    source = np.asarray(source_angles_rad, dtype=np.float64)
    if source.ndim != 1 or not len(source) or not np.isfinite(source).all():
        raise ValueError("source_angles_rad must be a nonempty finite vector")
    if np.any((source < SOURCE_Q[0] - 1e-6) | (source > SOURCE_Q[-1] + 1e-6)):
        warnings.warn("Source angle exceeds the fixed operator calibration; clamping to its range", RuntimeWarning, stacklevel=2)
    return np.interp(source, SOURCE_Q, SOURCE_GAP)


def motorized_aperture_m(jaw_angles_rad) -> np.ndarray:
    return np.interp(jaw_angles_rad, MOTOR_Q, MOTOR_GAP)


def source_angles_to_open_fraction(source_angles_rad, closed_jaw_rad: float,
                                   open_jaw_rad: float) -> np.ndarray:
    """Preserve the CAD distal aperture, without subtracting episode minima.

    Official joint states already carry the device's closed-position offset.
    A held-object minimum must remain partially open. This fixed lookup is
    identical for a single model output, a chunk, or a complete recording.
    The operator and motorized jaws have different CAD zero geometries.
    """
    if not np.isfinite([closed_jaw_rad, open_jaw_rad]).all() or open_jaw_rad <= closed_jaw_rad:
        raise ValueError("Invalid simulated jaw limits")
    if closed_jaw_rad < MOTOR_Q[0] - 1e-6 or open_jaw_rad > MOTOR_Q[-1] + 1e-6:
        raise ValueError("Jaw limits exceed the calibrated motorized aperture range")
    aperture = source_aperture_m(source_angles_rad)
    supported = motorized_aperture_m([closed_jaw_rad, open_jaw_rad])
    # The closed model leaves <0.5mm CAD clearance; tolerate that approximation.
    if np.any((aperture < supported[0] - .0005) | (aperture > supported[1] + .0005)):
        warnings.warn("Requested aperture exceeds motorized stroke; clipping is required", RuntimeWarning, stacklevel=2)
    angle = np.clip(np.interp(aperture, MOTOR_GAP, MOTOR_Q), closed_jaw_rad, open_jaw_rad)
    return (angle - closed_jaw_rad) / (open_jaw_rad - closed_jaw_rad)


__all__ = ["source_angles_to_open_fraction", "source_aperture_m", "motorized_aperture_m"]
