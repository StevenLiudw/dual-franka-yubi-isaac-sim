"""Offline checks for the isolated YUBI USD package (no Kit or GPU needed)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "scenes" / "dual_franka_yubi_random_000_seed_20260924.usda"
SOURCES = ROOT / "assets" / "yubi"
MESHES = ROOT / "assets" / "yubi" / "meshes"


def require(test: bool, message: str) -> None:
    if not test:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate() -> dict:
    manifest = json.loads((ROOT / "assets" / "yubi" / "source_manifest.json").read_text())
    camera_model = json.loads((ROOT / "wrist_camera_model.json").read_text())
    task_config = json.loads((ROOT / "config.json").read_text())
    yubi_config = task_config["yubi"]
    panda_drive_config = task_config["panda_joint_drive"]
    assembly = json.loads((ROOT / "assets" / "yubi" / "assembly_manifest.json").read_text())
    width, height = camera_model["render_resolution_px"]
    intrinsics = camera_model["intrinsics_px"]
    fov = camera_model["nominal_fov_deg"]
    require(abs(2 * math.degrees((width / 2) / intrinsics["fx"]) - fov["horizontal"]) < 1e-3,
            "Wrist horizontal FOV fit differs from camera model")
    require(abs(2 * math.degrees((height / 2) / intrinsics["fy"]) - fov["vertical"]) < 1e-3,
            "Wrist vertical FOV fit differs from camera model")
    for item in manifest["source_files"]:
        path = SOURCES / item.get("source_root", "source_v2.0.0") / item["path"]
        require(path.is_file() and sha256(path) == item["sha256"], f"Source mismatch: {path}")
    for item in manifest["generated_files"]:
        path = MESHES / item["path"]
        require(path.is_file() and sha256(path) == item["sha256"], f"Mesh mismatch: {path}")

    stage = Usd.Stage.Open(str(SCENE))
    require(stage is not None, f"Cannot open {SCENE}")
    for layer in stage.GetUsedLayers():
        if layer.realPath:
            require(Path(layer.realPath).is_relative_to(ROOT), f"External USD layer: {layer.realPath}")
    scene = stage.GetPrimAtPath("/World/PhysicsScene")
    require(scene.IsA(UsdPhysics.Scene), "PhysicsScene missing")
    physx = PhysxSchema.PhysxSceneAPI(scene)
    require(physx.GetEnableGPUDynamicsAttr().Get() is True, "GPU dynamics is disabled")
    require(physx.GetBroadphaseTypeAttr().Get() == "GPU", "GPU broadphase is disabled")

    cache = UsdGeom.XformCache()
    arms = {}
    local_mounts = []
    local_cameras = []
    for side in ("LeftMount", "RightMount"):
        root = f"/World/Robots/{side}/Panda"
        def prim(name: str):
            result = stage.GetPrimAtPath(f"{root}/{name}")
            require(result.IsValid() and result.IsActive(), f"Missing {root}/{name}")
            return result

        root_joint = prim("root_joint")
        require(root_joint.HasAPI(UsdPhysics.ArticulationRootAPI), f"{side} articulation root missing")
        for joint_number in range(1, 8):
            arm_joint = prim(f"joints/panda_joint{joint_number}")
            drive = UsdPhysics.DriveAPI.Get(arm_joint, "angular")
            require(bool(drive), f"{side} Panda joint {joint_number} angular drive missing")
            require(drive.GetTypeAttr().Get() == "acceleration",
                    f"{side} Panda joint {joint_number} drive type changed")
            require(abs(drive.GetStiffnessAttr().Get() -
                        panda_drive_config["stiffness_s_inv2"]) < 1e-6,
                    f"{side} Panda joint {joint_number} stiffness mismatch")
            require(abs(drive.GetDampingAttr().Get() -
                        panda_drive_config["damping_s_inv"]) < 1e-6,
                    f"{side} Panda joint {joint_number} damping mismatch")
        link8 = prim("panda_link8")
        base = prim("yubi_base")
        for name in ("yubi_leftfinger", "yubi_rightfinger", "yubi_tool"):
            body = prim(name)
            require(body.HasAPI(UsdPhysics.RigidBodyAPI), f"Missing rigid body: {body.GetPath()}")
            require(UsdPhysics.MassAPI(body).GetMassAttr().Get() > 0, f"Missing mass: {body.GetPath()}")
        require(base.HasAPI(UsdPhysics.RigidBodyAPI), f"{side} base is not rigid")
        camera = prim("yubi_base/wrist_camera")
        require(camera.IsA(UsdGeom.Camera), f"{side} wrist camera missing")
        require("OmniLensDistortionOpenCvFisheyeAPI" in camera.GetAppliedSchemas(),
                f"{side} wrist fisheye schema missing")
        prefix = "omni:lensdistortion:opencvFisheye:"
        require(camera.GetAttribute("omni:lensdistortion:model").Get() == "opencvFisheye",
                f"{side} wrist lens model mismatch")
        require(tuple(camera.GetAttribute(prefix + "imageSize").Get()) == (width, height),
                f"{side} wrist image size mismatch")
        for name, value in intrinsics.items():
            require(abs(camera.GetAttribute(prefix + name).Get() - value) < 1e-4,
                    f"{side} wrist {name} mismatch")
        for number, value in enumerate(camera_model["distortion_k1_k4"], start=1):
            require(abs(camera.GetAttribute(prefix + f"k{number}").Get() - value) < 1e-6,
                    f"{side} wrist k{number} mismatch")
        camera_local = UsdGeom.Xformable(camera).GetLocalTransformation()
        nominal_camera_mount = camera_model["nominal_mount_in_yubi_frame"]
        require((camera_local.ExtractTranslation() -
                 Gf.Vec3d(*nominal_camera_mount["translation_m"])).GetLength() < 1e-8,
                f"{side} wrist camera local translation mismatch")
        camera_quat = camera_local.ExtractRotationQuat()
        half_angle = math.radians(nominal_camera_mount["rotation_x_deg"]) / 2
        half_roll = math.radians(nominal_camera_mount["roll_about_optical_axis_deg"]) / 2
        camera_quat_expected = (Gf.Quatd(math.cos(half_angle), math.sin(half_angle), 0, 0) *
                                Gf.Quatd(math.cos(half_roll), 0, 0, math.sin(half_roll)))
        camera_dot = (camera_quat.GetReal() * camera_quat_expected.GetReal() +
                      sum(a * b for a, b in zip(camera_quat.GetImaginary(),
                                                camera_quat_expected.GetImaginary())))
        require(abs(abs(camera_dot) - 1) < 1e-8,
                f"{side} wrist camera local rotation mismatch")
        forward = camera_local.TransformDir(Gf.Vec3d(0, 0, -1))
        cad_forward = Gf.Vec3d(*assembly["camera"]["optical_forward_mount"])
        require((forward - cad_forward).GetLength() < 1e-4,
                f"{side} wrist camera no longer points along the CAD optical axis")
        tool_in_camera = camera_local.GetInverse().Transform(
            Gf.Vec3d(*yubi_config["tool_frame_xyz_m"]))
        require(tool_in_camera[1] < 0 and tool_in_camera[2] < 0,
                f"{side} YUBI tool is not below the camera image center")
        local_cameras.append((tuple(camera_local.ExtractTranslation()),
                              camera_quat.GetReal(), tuple(camera_quat.GetImaginary())))
        for name in ("panda_hand", "panda_leftfinger", "panda_rightfinger",
                     "joints/panda_hand_joint", "joints/panda_finger_joint1", "joints/panda_finger_joint2"):
            old = stage.GetPrimAtPath(f"{root}/{name}")
            require(old.IsValid() and not old.IsActive(), f"Stock gripper still active: {old.GetPath()}")
        mount = UsdPhysics.Joint(prim("joints/yubi_mount_joint"))
        require(mount.GetBody0Rel().GetTargets() == [link8.GetPath()], f"{side} mount parent mismatch")
        require(mount.GetBody1Rel().GetTargets() == [base.GetPath()], f"{side} mount child mismatch")
        local_mounts.append((tuple(mount.GetLocalPos0Attr().Get()),
                             tuple(mount.GetLocalPos1Attr().Get()),
                             tuple(mount.GetLocalRot0Attr().Get().GetImaginary()),
                             mount.GetLocalRot0Attr().Get().GetReal()))
        mount_yaw = mount.GetLocalRot0Attr().Get()
        require(abs(mount_yaw.GetReal() - math.sqrt(0.5)) < 1e-6 and
                (Gf.Vec3d(mount_yaw.GetImaginary()) -
                 Gf.Vec3d(0, 0, math.sqrt(0.5))).GetLength() < 1e-6,
                f"{side} YUBI mount yaw differs from +90 degrees in link8")
        mount_child_rotation = mount.GetLocalRot1Attr().Get()
        require(abs(mount_child_rotation.GetReal() - 1) < 1e-6 and
                Gf.Vec3d(mount_child_rotation.GetImaginary()).GetLength() < 1e-6,
                f"{side} YUBI mount child rotation is not identity")
        require((Gf.Vec3d(mount.GetLocalPos1Attr().Get()) - Gf.Vec3d(0, 0.018, 0.0025)).GetLength() < 1e-6,
                f"{side} adapter mating plane is not aligned to link8")
        for name in ("yubi_finger_joint", "yubi_finger_mimic_joint", "yubi_tool_joint"):
            prim(f"joints/{name}")
        driven = UsdPhysics.RevoluteJoint(prim("joints/yubi_finger_joint"))
        mimic = UsdPhysics.RevoluteJoint(prim("joints/yubi_finger_mimic_joint"))
        require(driven.GetAxisAttr().Get() == "Y" and mimic.GetAxisAttr().Get() == "Y",
                f"{side} jaw hinge axis mismatch")
        require(abs(driven.GetUpperLimitAttr().Get() - math.degrees(yubi_config["q_open_rad"])) < 1e-3,
                f"{side} jaw opening limit mismatch")
        require(abs(driven.GetLowerLimitAttr().Get() - math.degrees(yubi_config["q_closed_rad"])) < 1e-3,
                f"{side} jaw closing limit mismatch")
        require(mimic.GetLowerLimitAttr().Get() == -driven.GetUpperLimitAttr().Get(),
                f"{side} mimic joint lower limit mismatch")
        require(mimic.GetUpperLimitAttr().Get() == -driven.GetLowerLimitAttr().Get(),
                f"{side} mimic joint upper limit mismatch")
        for name in ("yubi_finger_joint", "yubi_finger_mimic_joint"):
            jaw_prim = prim(f"joints/{name}")
            drive = UsdPhysics.DriveAPI.Get(jaw_prim, "angular")
            require(bool(drive), f"{side} {name} angular drive missing")
        require(prim("joints/yubi_finger_mimic_joint").GetCustomDataByKey("yubi:mirrorsJoint") ==
                "yubi_finger_joint", f"{side} mirrored jaw metadata missing")
        # The adapter mating plane is +18 mm Y, +2.5 mm Z in the YUBI frame.
        # It must coincide with the stock Franka flange center. YUBI +Z
        # points down at rest, and its central pilot enters the flange recess.
        link_matrix = cache.GetLocalToWorldTransform(link8)
        link_origin = link_matrix.ExtractTranslation()
        base_matrix = cache.GetLocalToWorldTransform(base)
        mount_point = base_matrix.Transform(Gf.Vec3d(0, 0.018, 0.0025))
        require((link_origin - mount_point).GetLength() < 1e-5, f"{side} mount frames do not coincide")
        base_in_link8 = link_matrix.GetInverse().Transform(base_matrix.ExtractTranslation())
        require((base_in_link8 - Gf.Vec3d(0.018, 0, -0.0025)).GetLength() < 1e-5,
                f"{side} link8-to-YUBI translation mismatch")
        tip_direction = base_matrix.TransformDir(Gf.Vec3d(0, 0, 1))
        require(tip_direction[2] < -0.99, f"{side} YUBI fingertips do not face downward")
        arms[side] = {"mount_error_m": (link_origin - mount_point).GetLength(),
                      "link8_to_yubi_translation_m": tuple(base_in_link8),
                      "tip_direction_z": tip_direction[2]}

    require(local_mounts[0] == local_mounts[1], "Left and right YUBI mounts differ in local flange frame")
    require(local_cameras[0] == local_cameras[1], "Left and right wrist cameras differ in local YUBI frame")

    return {"scene": str(SCENE), "arms": arms,
            "usd_layers": len([layer for layer in stage.GetUsedLayers() if layer.realPath]),
            "source_files": len(manifest["source_files"]),
            "generated_meshes": len(manifest["generated_files"]),
            "wrist_camera_resolution_px": [width, height],
            "wrist_nominal_fov_deg": fov,
            "wrist_identical_local_mounts": True,
            "yubi_mount_yaw_link8_deg": 90,
            "identical_local_mounts": True,
            "panda_joint_drive_damping_s_inv": panda_drive_config["damping_s_inv"],
            "gpu_settings_authored": True}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
