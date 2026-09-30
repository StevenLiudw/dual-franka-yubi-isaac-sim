"""Rigid, shared-frame conversion of recorded hand-root poses to YUBI TCPs.

The operator URDF uses X toward the fingertips, Y across the jaws, Z toward
the camera. The motorized CAD uses Z toward the fingertips, X across the
jaws, Y toward the camera. These are proper rotations, never RGB reflections.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# Columns are the motorized tool axes expressed in the operator hand frame.
HAND_FROM_TOOL = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
# Source +X is the operator's right; world -Y is image-right in the unmirrored
# head camera. Both hands use exactly the same table-to-world rotation.
WORLD_FROM_TABLE = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
# Official operator URDF: hinge X=-0.016, nominal fingertip X=0.10943.
# This is a fixed reference TCP, not an angle-dependent contact estimate.
HAND_TCP_M = np.array([0.09343, 0., 0.])


def source_to_tool(poses_xyz_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return world-oriented TCP positions (no translation) and wxyz rotations."""
    values = np.asarray(poses_xyz_xyzw, dtype=float)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("Expected finite hand-root poses with shape (N, 7)")
    norms = np.linalg.norm(values[:, 3:], axis=1)
    if np.any(np.abs(norms - 1.) > 1e-3):
        raise ValueError("Source orientation must be a unit quaternion in xyzw order")
    source = Rotation.from_quat(values[:, 3:])
    tcp = (values[:, :3] + source.apply(HAND_TCP_M)) @ WORLD_FROM_TABLE.T
    rotation = Rotation.from_matrix(WORLD_FROM_TABLE) * source * Rotation.from_matrix(HAND_FROM_TOOL)
    xyzw = rotation.as_quat()
    for i in range(1, len(xyzw)):
        if np.dot(xyzw[i - 1], xyzw[i]) < 0:
            xyzw[i] *= -1
    return tcp, xyzw[:, [3, 0, 1, 2]]


def interpolate_orientation(start_wxyz, end_wxyz, fractions) -> np.ndarray:
    values = np.asarray([start_wxyz, end_wxyz], dtype=float)
    xyzw = Slerp([0., 1.], Rotation.from_quat(values[:, [1, 2, 3, 0]]))(fractions).as_quat()
    return xyzw[:, [3, 0, 1, 2]]
