"""Rebuild the isolated Panda/YUBI USD from pinned official CAD meshes.

Run this with an Isaac Sim 5.1 Python environment that exposes ``pxr`` and
``trimesh``.  It writes only inside this package.  The source mesh groups were
extracted from the v2.0.0 motorized YUBI STEP assembly; see
``assets/yubi/source_manifest.json`` for provenance and frame definitions.
The direct Franka adapter is the guide-linked v1.1.2 FR_FLANGE geometry.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MESHES = ASSETS / "yubi" / "meshes"
OUTPUT = ASSETS / "franka_yubi_panda.usdc"
SCENE = ROOT / "scenes" / "dual_franka_yubi_random_000_seed_20260924.usda"
TASK_CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
CONFIG = TASK_CONFIG["yubi"]
WRIST_CAMERA = json.loads((ROOT / "wrist_camera_model.json").read_text(encoding="utf-8"))

# These are initial dynamics estimates, not measured hardware properties.
# The pinned STEP/BOM supply geometry/material categories and ROBOTIS supplies
# the 82 g servo mass; the assembled adapter/camera/fasteners were not weighed.
MASS_KG = CONFIG["mass_kg"]
PIVOTS_M = {"left": (0.015, 0.025, 0.045), "right": (-0.015, 0.025, 0.045)}
# The guide-linked v1 direct Franka flange mates to the gripper bracket with
# its adapter axis 18 mm toward +Y in the assembly mount frame.
# The v1 direct flange's central pilot extends 2.5 mm toward the robot. Its
# planar mating face is gripper Z=+2.5 mm, not the pilot tip at Z=0.
MOUNT_POINT_M = (0.0, 0.018, 0.0025)
TOOL_M = tuple(CONFIG["tool_frame_xyz_m"])
Q_CLOSED_DEG = math.degrees(CONFIG["q_closed_rad"])
Q_OPEN_DEG = math.degrees(CONFIG["q_open_rad"])


def _xform(stage: Usd.Stage, path: str, xyz: tuple[float, ...], quat: Gf.Quatd | None = None):
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*xyz))
    if quat is not None:
        body.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quat)
    return body


def _mass(prim, mass_kg: float, center_m: tuple[float, ...], lengths_m: tuple[float, ...]) -> None:
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(mass_kg)
    api.CreateCenterOfMassAttr(Gf.Vec3f(*center_m))
    a, b, c = lengths_m
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(
        mass_kg * (b * b + c * c) / 12.0,
        mass_kg * (a * a + c * c) / 12.0,
        mass_kg * (a * a + b * b) / 12.0,
    ))


def _physics_material(stage: Usd.Stage, path: str, static: float, dynamic: float):
    prim = UsdShade.Material.Define(stage, path).GetPrim()
    api = UsdPhysics.MaterialAPI.Apply(prim)
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    return UsdShade.Material(prim)


def _visual_material(stage: Usd.Stage, path: str, rgb: tuple[float, ...]):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.68)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _mesh(stage: Usd.Stage, path: str, source: Path, origin: tuple[float, ...],
          visual, contact=None):
    if not source.is_file():
        raise FileNotFoundError(source)
    tri = trimesh.load_mesh(source, process=False)
    tri.merge_vertices()
    xyz = np.asarray(tri.vertices, dtype=np.float32) - np.asarray(origin, dtype=np.float32)
    faces = np.asarray(tri.faces, dtype=np.int32)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(xyz))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(visual)
    if contact is not None:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            contact, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
    return tri.bounds - np.asarray(origin, dtype=float)


def _fixed_joint(stage: Usd.Stage, path: str, parent: str, child: str,
                 local0: tuple[float, ...] = (0.0, 0.0, 0.0),
                 local1: tuple[float, ...] = (0.0, 0.0, 0.0),
                 rot0: Gf.Quatf | None = None) -> None:
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(parent)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*local0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*local1))
    joint.CreateLocalRot0Attr(rot0 or Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateCollisionEnabledAttr(False)


def _jaw_joint(stage: Usd.Stage, name: str, child: str,
               pivot: tuple[float, ...], lower: float, upper: float):
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"/panda/joints/{name}")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/panda/yubi_base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child)])
    joint.CreateAxisAttr("Y")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pivot))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    joint.CreateCollisionEnabledAttr(False)
    return joint


def build() -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(OUTPUT))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("kilogramsPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    panda = UsdGeom.Xform.Define(stage, "/panda")
    panda.GetPrim().GetReferences().AddReference("franka_panda/franka_panda.usd")
    stage.SetDefaultPrim(panda.GetPrim())

    # The stock Panda asset uses acceleration drives with K=625 and D=0.  The
    # zero damping lets a 10 Hz replay target excite joint oscillation, even
    # when the target trajectory itself is smooth.  Author a higher-bandwidth,
    # well-damped local drive so each 100 ms waypoint interval settles without
    # adding replay delay.  This leaves the vendored NVIDIA asset untouched.
    panda_drive = TASK_CONFIG["panda_joint_drive"]
    for joint_number in range(1, 8):
        joint = stage.GetPrimAtPath(f"/panda/joints/panda_joint{joint_number}")
        if not joint.IsValid():
            raise RuntimeError(f"Stock Panda joint {joint_number} is missing")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            raise RuntimeError(f"Stock Panda joint {joint_number} has no angular drive")
        stiffness = drive.GetStiffnessAttr().Get()
        if abs(stiffness - panda_drive["source_stiffness_s_inv2"]) > 1e-6:
            raise RuntimeError(
                f"Unexpected Panda joint stiffness {stiffness}; refusing stale damping tuning"
            )
        drive.GetStiffnessAttr().Set(panda_drive["stiffness_s_inv2"])
        drive.GetDampingAttr().Set(panda_drive["damping_s_inv"])

    # The source Franka remains untouched. These stronger local opinions remove
    # its hand and sliders only in this derived YUBI-equipped robot asset.
    for path in (
        "/panda/joints/panda_hand_joint",
        "/panda/joints/panda_finger_joint1",
        "/panda/joints/panda_finger_joint2",
        "/panda/panda_hand",
        "/panda/panda_leftfinger",
        "/panda/panda_rightfinger",
    ):
        stage.OverridePrim(path).SetActive(False)

    body_color = _visual_material(stage, "/panda/YubiLooks/Body", (0.21, 0.24, 0.27))
    metal_color = _visual_material(stage, "/panda/YubiLooks/Adapter", (0.64, 0.67, 0.70))
    pad_color = _visual_material(stage, "/panda/YubiLooks/Jaws", (0.28, 0.43, 0.55))
    grip_material = _physics_material(
        stage, "/panda/YubiLooks/GripPhysics",
        CONFIG["friction"]["static"], CONFIG["friction"]["dynamic"],
    )
    body_material = _physics_material(stage, "/panda/YubiLooks/BodyPhysics", 0.5, 0.4)

    # panda_link8 is already Rx(180 deg), with +Z facing down at rest. The
    # guide-linked v1 adapter is Rz(-90 deg) in the gripper CAD frame, so the
    # nominal inverse Rz(+90 deg) aligns its native flange axes to link8.
    # Combining Rx(180) * Rz(+90) keeps YUBI +Z toward the table. The +18 mm
    # local Y/Z mating point coincides with the link8 origin. Physical locating
    # key/dowel yaw is not given by the source and still needs verification.
    base_xyz = (0.1059999772, 0.0, 0.9285)
    sine = math.sqrt(0.5)
    base_quat = Gf.Quatd(0.0, sine, -sine, 0.0)
    base = _xform(stage, "/panda/yubi_base", base_xyz, base_quat)
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    b0 = _mesh(stage, "/panda/yubi_base/fixed_visual", MESHES / "fixed.stl", (0, 0, 0),
               body_color, body_material)
    b1 = _mesh(stage, "/panda/yubi_base/franka_adapter", MESHES / "franka_adapter_v1.stl", (0, 0, 0),
               metal_color, body_material)
    bmin = np.minimum(b0[0], b1[0]); bmax = np.maximum(b0[1], b1[1])
    _mass(base.GetPrim(), MASS_KG["base"], tuple((bmin + bmax) / 2), tuple(bmax - bmin))
    _fixed_joint(stage, "/panda/joints/yubi_mount_joint", "/panda/panda_link8",
                 "/panda/yubi_base", local1=MOUNT_POINT_M,
                 rot0=Gf.Quatf(sine, 0, 0, sine))

    for side, name in (("left", "yubi_leftfinger"), ("right", "yubi_rightfinger")):
        pivot = PIVOTS_M[side]
        # The bodies share the base orientation at zero jaw angle. Their mesh
        # coordinates are expressed around the CAD-derived hinge center.
        translated = (base_xyz[0] - pivot[1], base_xyz[1] - pivot[0], base_xyz[2] - pivot[2])
        body = _xform(stage, f"/panda/{name}", translated, base_quat)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        bounds = _mesh(stage, f"/panda/{name}/geometry", MESHES / f"jaw_{side}.stl",
                       pivot, pad_color, grip_material)
        _mass(body.GetPrim(), MASS_KG[side], tuple(bounds.mean(axis=0)), tuple(bounds[1] - bounds[0]))

    driven = _jaw_joint(stage, "yubi_finger_joint", "/panda/yubi_leftfinger",
                        PIVOTS_M["left"], Q_CLOSED_DEG, Q_OPEN_DEG)
    mirrored = _jaw_joint(stage, "yubi_finger_mimic_joint", "/panda/yubi_rightfinger",
                          PIVOTS_M["right"], -Q_OPEN_DEG, -Q_CLOSED_DEG)
    # PhysX 5.1's articulation mimic constraint can lock this pair at the
    # shared zero stop when its limits have opposite signs.  Drive both sides
    # from the same normalized command instead.  The environment always sends
    # q and -q together, preserving the one-actuator hardware API while making
    # the simulated gearing deterministic.  Split the motor torque clamp over
    # the two simulated drives so their combined clamp remains XM430-sized.
    for joint, target in ((driven, Q_CLOSED_DEG), (mirrored, -Q_CLOSED_DEG)):
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(CONFIG["grip_drive_stiffness_Nm_per_rad"])
        drive.CreateDampingAttr(CONFIG["grip_drive_damping_Nm_s_per_rad"])
        drive.CreateMaxForceAttr(CONFIG["grip_drive_max_force_Nm"] / 2.0)
        drive.CreateTargetPositionAttr(target)
    mirrored.GetPrim().SetCustomDataByKey("yubi:mirrorsJoint", "yubi_finger_joint")
    # The geared jaws can touch each other at the closed stop. Filter only
    # their mutual collision; both remain collidable with the cup and plate.
    UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/panda/yubi_leftfinger")) \
        .CreateFilteredPairsRel().AddTarget(Sdf.Path("/panda/yubi_rightfinger"))

    tool_xyz = (base_xyz[0] - TOOL_M[1], base_xyz[1] - TOOL_M[0], base_xyz[2] - TOOL_M[2])
    tool = _xform(stage, "/panda/yubi_tool", tool_xyz, base_quat)
    UsdPhysics.RigidBodyAPI.Apply(tool.GetPrim())
    _mass(tool.GetPrim(), MASS_KG["tool"], (0, 0, 0), (0.005, 0.005, 0.005))
    _fixed_joint(stage, "/panda/joints/yubi_tool_joint", "/panda/yubi_base",
                 "/panda/yubi_tool", TOOL_M)

    camera = UsdGeom.Camera.Define(stage, "/panda/yubi_base/wrist_camera")
    mount = WRIST_CAMERA["nominal_mount_in_yubi_frame"]
    camera.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*mount["translation_m"]))
    angle = math.radians(mount["rotation_x_deg"]) / 2.0
    roll = math.radians(mount["roll_about_optical_axis_deg"]) / 2.0
    # CAD fixes the lens position and forward ray, but not which sensor edge
    # appears at the top of the replay image. A local optical-axis roll keeps
    # that forward ray while aligning both wrist views to the dataset.
    forward_rotation = Gf.Quatd(math.cos(angle), math.sin(angle), 0.0, 0.0)
    image_roll = Gf.Quatd(math.cos(roll), 0.0, 0.0, math.sin(roll))
    camera.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        forward_rotation * image_roll
    )
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    camera.CreateFocalLengthAttr(float(WRIST_CAMERA["usd_focal_length_m"]) * 10.0)
    camera.CreateHorizontalApertureAttr(float(WRIST_CAMERA["usd_horizontal_aperture_m"]) * 10.0)
    width, height = WRIST_CAMERA["render_resolution_px"]
    camera.CreateVerticalApertureAttr(
        float(WRIST_CAMERA["usd_horizontal_aperture_m"]) * 10.0 * height / width
    )
    # Author the renderer's actual fisheye schema in USD so opening the scene
    # directly gives the same lens as the Python runner. These are nominal FOV
    # fits, not calibrated intrinsics for either physical camera.
    camera_prim = camera.GetPrim()
    if not camera_prim.ApplyAPI("OmniLensDistortionOpenCvFisheyeAPI"):
        raise RuntimeError("Isaac Sim's OmniLensDistortionOpenCvFisheyeAPI is unavailable")
    camera_prim.GetAttribute("omni:lensdistortion:model").Set("opencvFisheye")
    lens_prefix = "omni:lensdistortion:opencvFisheye:"
    for name, value in WRIST_CAMERA["intrinsics_px"].items():
        camera_prim.GetAttribute(lens_prefix + name).Set(float(value))
    for number, value in enumerate(WRIST_CAMERA["distortion_k1_k4"], start=1):
        camera_prim.GetAttribute(lens_prefix + f"k{number}").Set(float(value))
    camera_prim.GetAttribute(lens_prefix + "imageSize").Set(Gf.Vec2i(width, height))
    camera_prim.SetCustomDataByKey("yubi:cameraStatus", WRIST_CAMERA["intrinsics_status"])
    camera_prim.SetCustomDataByKey("yubi:cameraModel", WRIST_CAMERA["model"])
    panda.GetPrim().SetCustomDataByKey("yubi:sourceTag", "Toyota/yubi-hw v2.0.0 motorized gripper and v1.1.2 direct Franka flange")
    panda.GetPrim().SetCustomDataByKey(
        "yubi:physicsStatus",
        "provisional masses, friction, jaw limits; damped Panda acceleration drives",
    )
    stage.GetRootLayer().Save()

    # A composition check catches broken relative references and inactive
    # stock joints without requiring the RTX renderer or a CUDA device.
    verify = Usd.Stage.Open(str(OUTPUT))
    required = ["/panda/root_joint", "/panda/panda_link8", "/panda/yubi_base",
                "/panda/yubi_leftfinger", "/panda/yubi_rightfinger",
                "/panda/joints/yubi_finger_joint", "/panda/joints/yubi_finger_mimic_joint"]
    if any(not verify.GetPrimAtPath(p).IsValid() for p in required):
        raise RuntimeError("Generated robot USD is missing a required prim")
    if verify.GetPrimAtPath("/panda/panda_hand").IsActive():
        raise RuntimeError("Stock Panda hand remained active in YUBI robot")
    # The cup scene is reused by reference, and its PhysicsScene has no
    # explicit PhysX GPU settings. Author them only in this derived scene.
    scene = Usd.Stage.Open(str(SCENE))
    if scene is None:
        raise RuntimeError(f"Cannot open YUBI scene: {SCENE}")
    scene.SetEditTarget(scene.GetRootLayer())
    physics_prim = scene.OverridePrim("/World/PhysicsScene")
    if not physics_prim.IsA(UsdPhysics.Scene):
        raise RuntimeError("YUBI scene is missing a PhysicsScene")
    physx = PhysxSchema.PhysxSceneAPI.Apply(physics_prim)
    physx.CreateEnableGPUDynamicsAttr(True)
    physx.CreateBroadphaseTypeAttr("GPU")
    scene.GetRootLayer().Save()
    return OUTPUT


if __name__ == "__main__":
    print(build())
