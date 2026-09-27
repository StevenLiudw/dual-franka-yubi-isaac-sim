"""Dataset-aligned overhead camera authored into an open Isaac Sim USD stage.

The calibration is deliberately separate from the scene and can be replaced
when a physical camera calibration becomes available. Importing this module
does not start Isaac Sim or import USD; call ``configure_head_camera`` after
``SimulationApp`` has opened the stage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


HEAD_CAMERA_PATH = "/World/Sensors/HeadCamera"
CALIBRATION_PATH = Path(__file__).resolve().with_name("head_camera_calibration.json")


def load_head_camera_calibration(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the overhead camera estimate in SI units.

    ``horizontal_flip_for_dataset`` documents the reflection between the
    legacy image-to-world convention and a physical right-handed USD camera.
    It does not change the authored camera; consumers flip RGB when needed.
    """
    source = Path(path).expanduser() if path is not None else CALIBRATION_PATH
    data = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "camera_path", "resolution_px", "eye_world_m", "quaternion_wxyz",
        "focal_length_m", "horizontal_aperture_m", "horizontal_flip_for_dataset",
        "height_above_table_m", "table_top_z_m", "provenance",
        "lens_model", "intrinsics_px", "distortion_coefficients",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"Head-camera calibration is missing {missing}: {source}")
    if data["camera_path"] != HEAD_CAMERA_PATH:
        raise ValueError(f"Head-camera path must be {HEAD_CAMERA_PATH}")
    resolution = data["resolution_px"]
    if len(resolution) != 2 or any(not isinstance(v, int) or v <= 0 for v in resolution):
        raise ValueError("resolution_px must contain two positive integers")
    eye = data["eye_world_m"]
    quat = data["quaternion_wxyz"]
    if len(eye) != 3 or len(quat) != 4:
        raise ValueError("eye_world_m needs 3 values and quaternion_wxyz needs 4")
    values = [*eye, *quat, data["focal_length_m"], data["horizontal_aperture_m"],
              data["height_above_table_m"], data["table_top_z_m"]]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError("Head-camera pose and optics must be finite numeric values")
    if data["focal_length_m"] <= 0 or data["horizontal_aperture_m"] <= 0:
        raise ValueError("Focal length and aperture must be positive")
    if abs(sum(v * v for v in quat) - 1.0) > 1e-4:
        raise ValueError("Camera quaternion must be normalized")
    if abs(eye[2] - data["table_top_z_m"] - data["height_above_table_m"]) > 1e-4:
        raise ValueError("Eye Z must equal tabletop Z plus calibrated height")
    if not isinstance(data["horizontal_flip_for_dataset"], bool):
        raise ValueError("horizontal_flip_for_dataset must be boolean")
    if data["lens_model"] != "OpenCvPinhole":
        raise ValueError("Head camera requires Isaac RTX OpenCvPinhole lens model")
    intrinsics = data["intrinsics_px"]
    if set(intrinsics) != {"fx", "fy", "cx", "cy"} or not all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in intrinsics.values()
    ):
        raise ValueError("intrinsics_px must contain finite fx, fy, cx, cy")
    if intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0:
        raise ValueError("fx and fy must be positive")
    if len(data["distortion_coefficients"]) != 12 or any(
        not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in data["distortion_coefficients"]
    ):
        raise ValueError("OpenCV distortion requires twelve finite coefficients")
    return data


def configure_head_camera(stage: Any, calibration: dict[str, Any] | None = None) -> Any:
    """Author ``/World/Sensors/HeadCamera`` into the stage session layer.

    Returns a ``UsdGeom.Camera``. Its focal length/aperture are converted from
    metres to USD camera tenths-of-stage-unit values, matching Isaac Sim 5.1's
    ``Camera.set_focal_length`` and ``set_horizontal_aperture`` implementation.
    The session layer keeps the packaged scene asset immutable.
    """
    from pxr import Gf, UsdGeom

    data = calibration if calibration is not None else load_head_camera_calibration()
    if stage is None or not stage.GetPrimAtPath("/World").IsValid():
        raise ValueError("Open a scene containing /World before configuring HeadCamera")
    previous_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    try:
        UsdGeom.Xform.Define(stage, "/World/Sensors")
        camera = UsdGeom.Camera.Define(stage, data["camera_path"])
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*map(float, data["eye_world_m"]))
        )
        w, x, y, z = map(float, data["quaternion_wxyz"])
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(w, Gf.Vec3d(x, y, z))
        )
        camera.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
        camera.GetFocalLengthAttr().Set(float(data["focal_length_m"]) * 10.0)
        camera.GetHorizontalApertureAttr().Set(float(data["horizontal_aperture_m"]) * 10.0)
        width, height = data["resolution_px"]
        camera.GetVerticalApertureAttr().Set(
            float(data["horizontal_aperture_m"]) * 10.0 * height / width
        )
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 100.0))
        # Isaac RTX's legacy USD camera projection rendered a 19 cm plate about
        # 1.5x too large even though the Camera wrapper reported fx=564 px.
        # Its explicit OpenCV pinhole lens matches the source image geometry.
        prim = camera.GetPrim()
        prim.ApplyAPI("OmniLensDistortionOpenCvPinholeAPI")
        prim.GetAttribute("omni:lensdistortion:model").Set("opencvPinhole")
        intrinsics = data["intrinsics_px"]
        for name in ("fx", "fy", "cx", "cy"):
            prim.GetAttribute(f"omni:lensdistortion:opencvPinhole:{name}").Set(float(intrinsics[name]))
        prim.GetAttribute("omni:lensdistortion:opencvPinhole:imageSize").Set(Gf.Vec2i(width, height))
        names = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4")
        for name, value in zip(names, data["distortion_coefficients"]):
            prim.GetAttribute(f"omni:lensdistortion:opencvPinhole:{name}").Set(float(value))
        return camera
    finally:
        stage.SetEditTarget(previous_target)


__all__ = ["HEAD_CAMERA_PATH", "load_head_camera_calibration", "configure_head_camera"]
