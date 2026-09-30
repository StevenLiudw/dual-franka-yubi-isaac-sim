#!/usr/bin/env python3
"""Launch the packaged dual-Franka YUBI scene in a visible GPU Isaac Sim window.

Run with ``python -m yubi_isaac_sim_env.run`` from the repository root.
``--record-run DIR`` captures RGB video and named joint states for each episode.
Custom policies export ``act(observation, step, episode) -> action``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yubi_isaac_sim_env import create_sim, resolve_scene
from yubi_isaac_sim_env.command_conditioning import (
    JOINT_REFERENCE_PROFILES,
    get_joint_reference_profile,
)
from yubi_isaac_sim_env.interarm_guard import observed_interarm_clearance_m
from yubi_isaac_sim_env.recording import JointStateWriter, VideoWriter
from yubi_isaac_sim_env.policy_adapter import TRAJECTORY_EXECUTOR_PROFILES


PACKAGE_DIR = Path(__file__).resolve().parent
OVERVIEW_CAMERA_PATH = "/World/OverviewCamera"
OVERVIEW_CAMERA_EYE = (1.9, 0.0, 1.85)
OVERVIEW_CAMERA_TARGET = (-0.2, 0.0, 1.1)
HEAD_CALIBRATION_PATH = PACKAGE_DIR / "head_camera_calibration.json"
WRIST_CAMERA_MODEL_PATH = PACKAGE_DIR / "wrist_camera_model.json"


def _camera_spec(name: str, head_calibration: Path | None = None) -> dict:
    if name == "head":
        calibration_path = head_calibration or HEAD_CALIBRATION_PATH
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        return {"name": "head", "prim_path": calibration["camera_path"],
                "resolution": calibration["resolution_px"],
                "eye_m": calibration["eye_world_m"],
                "quaternion_wxyz": calibration["quaternion_wxyz"],
                "lens_model": calibration["lens_model"],
                "intrinsics_px": calibration["intrinsics_px"],
                "focal_length_m": calibration["focal_length_m"],
                "horizontal_aperture_m": calibration["horizontal_aperture_m"],
                "horizontal_flip_for_dataset": calibration["horizontal_flip_for_dataset"],
                "calibration_file": str(calibration_path)}
    if name in ("left_wrist", "right_wrist"):
        side = "LeftMount" if name == "left_wrist" else "RightMount"
        model = json.loads(WRIST_CAMERA_MODEL_PATH.read_text(encoding="utf-8"))
        return {
            "name": name,
            "prim_path": f"/World/Robots/{side}/Panda/yubi_base/wrist_camera",
            "camera_model": model["model"],
            "resolution": model["render_resolution_px"],
            "lens_model": model["projection"],
            "intrinsics_px": model["intrinsics_px"],
            "distortion": model["distortion_k1_k4"],
            "nominal_fov_deg": model["nominal_fov_deg"],
            "focal_length_m": model["usd_focal_length_m"],
            "horizontal_aperture_m": model["usd_horizontal_aperture_m"],
            "horizontal_flip_for_dataset": False,
            "calibration_status": model["intrinsics_status"],
            "calibration_file": str(WRIST_CAMERA_MODEL_PATH),
        }
    return {"name": "overview", "prim_path": OVERVIEW_CAMERA_PATH,
            "resolution": [960, 540], "eye_m": OVERVIEW_CAMERA_EYE,
            "target_m": OVERVIEW_CAMERA_TARGET, "focal_length_m": 0.015,
            "horizontal_aperture_m": 0.036,
            "horizontal_flip_for_dataset": False}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_path(name: str) -> Path:
    requested = Path(name).expanduser()
    candidates = [requested, PACKAGE_DIR / requested, PACKAGE_DIR / "policies" / requested]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Policy script not found: {name}")


def _setup_file(name: str | None) -> Path | None:
    if name is None:
        return None
    requested = Path(name).expanduser()
    if requested.is_file():
        return requested.resolve()
    bundled = PACKAGE_DIR / "setups" / f"{name.replace(':', '_')}.json"
    return bundled.resolve() if bundled.is_file() else None


def _load_callable(path: Path, name: str):
    spec = importlib.util.spec_from_file_location("cup_plate_user_policy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import policy script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    action_fn = getattr(module, name, None)
    if not callable(action_fn):
        raise RuntimeError(f"Policy script must define {name}(observation, step, episode): {path}")
    return action_fn


def _load_policy(path: Path):
    return _load_callable(path, "act")


def _builtin_action(observation: dict, step: int, mode: str) -> dict:
    if mode == "hold":
        return {}
    names = observation["robots"]["left"]["joint_names"]
    indices = [names.index(f"panda_joint{i}") for i in range(1, 8)]
    offset = 0.12 * math.sin(2 * math.pi * (step + 1) / 20)
    result = {}
    for side, sign in (("left", 1), ("right", -1)):
        q = observation["robots"][side]["joint_positions"]
        targets = [float(q[i]) for i in indices]
        targets[0] = sign * offset
        result[side] = {"arm_joint_targets_rad": targets, "gripper_open_fraction": 1.0}
    return result


def _look_at_quaternion(eye: tuple[float, ...], target: tuple[float, ...]) -> np.ndarray:
    from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats

    eye_vec = np.asarray(eye, dtype=float)
    forward = np.asarray(target, dtype=float) - eye_vec
    forward /= np.linalg.norm(forward)
    up = np.array((0.0, 0.0, 1.0))
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # USD camera coordinates: +X right, +Y up, and -Z forward.
    rotation = np.column_stack((right, up, -forward))
    return rot_matrices_to_quats(rotation)


def _make_camera(spec: dict, gui: bool, recording_video: bool):
    from isaacsim.sensors.camera import Camera
    import omni.usd

    # An inspection-only lighting override applies to the chosen camera.
    stage = omni.usd.get_context().get_stage()
    light = stage.GetPrimAtPath("/World/Lights/InspectionLight")
    if light.IsValid():
        stage.SetEditTarget(stage.GetSessionLayer())
        light.GetAttribute("inputs:intensity").Set(220.0)

    width, height = spec["resolution"]
    camera = Camera(
        prim_path=spec["prim_path"],
        name=f"cup_plate_{spec['name']}",
        resolution=(width, height),
        annotator_device="cpu",
    )
    if spec["name"] == "overview":
        camera.set_world_pose(
            position=np.asarray(spec["eye_m"], dtype=float),
            orientation=_look_at_quaternion(spec["eye_m"], spec["target_m"]),
            camera_axes="usd",
        )
    # Keep the RTX sensor intrinsics consistent with the authored USD camera.
    camera.set_focal_length(spec["focal_length_m"])
    camera.set_horizontal_aperture(spec["horizontal_aperture_m"])
    if spec["name"] in ("left_wrist", "right_wrist"):
        intrinsics = spec["intrinsics_px"]
        camera.set_opencv_fisheye_properties(
            cx=intrinsics["cx"], cy=intrinsics["cy"],
            fx=intrinsics["fx"], fy=intrinsics["fy"],
            fisheye=spec["distortion"],
        )
    if gui:
        from isaacsim.core.utils.viewports import create_viewport_for_camera, set_active_viewport_camera
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            create_viewport_for_camera("Cup and plate", spec["prim_path"], width, height)
            viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("Isaac Sim GUI started without an active viewport")
        set_active_viewport_camera(spec["prim_path"])
        if str(viewport.camera_path) != spec["prim_path"]:
            raise RuntimeError(f"GUI viewport did not switch to {spec['prim_path']}: {viewport.camera_path}")
    if recording_video:
        camera.initialize()
    return camera


def _rendered_rgb(camera, world, spec: dict) -> np.ndarray:
    # RTX annotators can be several frames behind a just-reset state.
    for _ in range(5):
        world.render()
    height, width = spec["resolution"][1], spec["resolution"][0]
    for _ in range(24):
        world.render()
        rgb = camera.get_rgb(device="cpu")
        if rgb is None:
            continue
        frame = np.asarray(rgb)
        if frame.shape == (height, width, 3) and np.isfinite(frame).all():
            if float(frame.max()) - float(frame.min()) >= 2:
                # The source scene's 2D-to-world projection reverses handedness.
                # Flip the rendered image so plate/cup positions match center.mp4.
                return np.ascontiguousarray(frame[:, ::-1]) if spec["horizontal_flip_for_dataset"] else frame
    raise RuntimeError(f"{spec['name']} camera did not produce a nonblank RTX RGB image")


def _policy_rgb(frame: np.ndarray, spec: dict) -> np.ndarray:
    """Return an owned uint8 image in the model-facing H×W×RGB convention."""
    height, width = spec["resolution"][1], spec["resolution"][0]
    value = np.asarray(frame)
    if value.shape != (height, width, 3) or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{spec['name']} RGB shape/dtype mismatch: {value.shape}, {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{spec['name']} RGB contains non-finite pixels")
    if value.dtype != np.uint8:
        value = np.clip(value * 255 if float(value.max()) <= 1 else value, 0, 255).astype(np.uint8)
    result = np.array(value, dtype=np.uint8, copy=True, order="C")
    return result


def _policy_observation(state: dict, cameras: dict, specs: dict, world) -> dict:
    """Add ephemeral RGB input to a state observation without logging pixels."""
    images = {}
    metadata = {}
    for name in ("head", "left_wrist", "right_wrist"):
        camera = cameras[name]
        spec = specs[name]
        images[name] = _policy_rgb(_rendered_rgb(camera, world, spec), spec)
        position, orientation = camera.get_world_pose(camera_axes="usd")
        frame = camera.get_current_frame()
        metadata[name] = {
            "resolution_px": list(spec["resolution"]),
            "lens_model": spec.get("lens_model", "pinhole"),
            "intrinsics_px": spec.get("intrinsics_px"),
            "distortion_k1_k4": spec.get("distortion"),
            "horizontal_flip_for_dataset": bool(spec.get("horizontal_flip_for_dataset", False)),
            "pose_world": {
                "position_m": np.asarray(position, dtype=float).tolist(),
                "quaternion_wxyz": np.asarray(orientation, dtype=float).tolist(),
            },
            "physics_time_s": float(state["physics_time_s"]),
            "rendering_time_s": float(frame["rendering_time"]) if frame and frame.get("rendering_time") is not None else None,
        }
    return {**state, "images": images, "image_metadata": metadata}


def _spatial_alignment_state(observation: dict, side: str = "right") -> dict:
    """Compact omniscient state for replay/object alignment diagnostics."""
    robot = observation["robots"][side]
    links = robot["link_poses"]
    left = np.asarray(links["left_finger"]["position_m"], dtype=np.float64)
    right = np.asarray(links["right_finger"]["position_m"], dtype=np.float64)
    return {
        "cup_position_m": observation["objects"]["cup"]["position_m"],
        "cup_quaternion_wxyz": observation["objects"]["cup"]["quaternion_wxyz"],
        "tool_position_m": robot["tool_pose"]["position_m"],
        "finger_body_midpoint_m": ((left + right) * 0.5).tolist(),
        "gripper_open_fraction": robot["gripper_open_fraction"],
    }


def _sample(video: VideoWriter | None, joints: JointStateWriter | None, camera, world,
            spec: dict, episode: int, sample_index: int, phase: str, policy_step: int,
            observation: dict) -> None:
    if video is not None:
        video.write(_rendered_rgb(camera, world, spec))
    if joints is not None:
        joints.write(episode, sample_index, phase, policy_step, observation)


def _episode_path(path: Path | None, episode: int, total: int) -> Path | None:
    if path is None:
        return None
    if total == 1:
        return path
    return path.with_name(f"{path.stem}_episode_{episode:03d}{path.suffix}")


def _paths(args, episode: int) -> tuple[Path | None, Path | None]:
    if args.record_run is not None:
        if args.episodes == 1:
            return args.record_run / "video.mp4", args.record_run / "joints.csv"
        return (
            args.record_run / f"episode_{episode:03d}.mp4",
            args.record_run / f"episode_{episode:03d}_joints.csv",
        )
    return (
        _episode_path(args.record_video, episode, args.episodes),
        _episode_path(args.record_joints, episode, args.episodes),
    )


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="dual_franka_yubi_random_000_seed_20260924", help="Packaged scene name or USD path")
    parser.add_argument("--setup", help="Named, JSON-file, or custom object setup for reset")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50, help="Maximum 10 Hz policy steps per episode")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--index-start", type=int, default=0)
    parser.add_argument("--fixed-index", type=int)
    parser.add_argument("--policy", choices=("hold", "oscillate"), default="hold")
    parser.add_argument("--camera", choices=("head", "overview", "left_wrist", "right_wrist"), default="head",
                        help="GUI and recorded view; head matches the center videos approximately")
    parser.add_argument("--policy-script", help="Python file exporting act(observation, step, episode)")
    parser.add_argument("--trajectory-policy-script",
                        help="Python file exporting predict(observation, step, episode) -> world-frame tool waypoint chunk")
    parser.add_argument(
        "--trajectory-controller-profile",
        choices=tuple(TRAJECTORY_EXECUTOR_PROFILES),
        default="default",
        help="Differential-IK tuning shared by simulation and hardware trajectory adapters",
    )
    parser.add_argument("--policy-images", choices=("none", "all"), default="none",
                        help="Pass head and both wrist RGB arrays to a custom policy (default: state only)")
    parser.add_argument("--head-camera-calibration", type=Path, help="Optional estimated/measured head-camera JSON")
    parser.add_argument("--headless", action="store_true", help="Disable GUI window, retaining GPU physics and RTX recording")
    parser.add_argument("--linger-seconds", type=float, help="Keep the GUI visible this long after the run (default: 5)")
    parser.add_argument("--keep-open", action="store_true", help="Keep rendering the GUI until its window is closed")
    parser.add_argument("--continue-after-success", action="store_true",
                        help="Continue recording through --steps after success; preserve success events in the report")
    parser.add_argument("--record-run", type=Path, help="Shortcut: capture MP4, joint CSV, manifest, and report in DIR")
    parser.add_argument("--record-video", type=Path, help="Capture rendered camera MP4")
    parser.add_argument("--record-joints", type=Path, help="Capture named joint position and velocity CSV")
    parser.add_argument("--record-fps", type=int, default=30,
                        help="Camera/joint samples per simulated second (default: 30, matching source videos)")
    parser.add_argument("--interpolate-targets", action="store_true",
                        help="Linearly interpolate each 10 Hz joint target over the six 60 Hz physics steps")
    parser.add_argument(
        "--joint-command-profile",
        choices=("direct", *JOINT_REFERENCE_PROFILES),
        default="direct",
        help="Servo-rate joint reference conditioning shared with hardware integrations",
    )
    parser.add_argument("--report", type=Path, help="Write JSON episode report (automatic with --record-run)")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.steps < 1 or args.index_start < 0 or (args.fixed_index is not None and args.fixed_index < 0):
        parser.error("episodes and steps must be positive; scenario indices must be nonnegative")
    if args.linger_seconds is not None and args.linger_seconds < 0:
        parser.error("--linger-seconds must be nonnegative")
    if args.keep_open and args.headless:
        parser.error("--keep-open requires the GUI")
    if args.policy_script and args.trajectory_policy_script:
        parser.error("Choose one of --policy-script and --trajectory-policy-script")
    if args.interpolate_targets and args.joint_command_profile != "direct":
        parser.error("Choose one of --interpolate-targets and --joint-command-profile")
    if args.trajectory_controller_profile == "panda-pose-replay" and args.joint_command_profile != "franka-panda-interface":
        parser.error("panda-pose-replay requires --joint-command-profile franka-panda-interface")
    if args.policy_images == "all" and not (args.policy_script or args.trajectory_policy_script):
        parser.error("--policy-images all requires a custom policy script")
    if args.record_run and (args.record_video or args.record_joints or args.report):
        parser.error("--record-run already names the video, joint CSV, and report outputs")
    if args.record_video and args.record_video.suffix.lower() != ".mp4":
        parser.error("--record-video must end in .mp4")
    if args.record_joints and args.record_joints.suffix.lower() != ".csv":
        parser.error("--record-joints must end in .csv")
    if args.record_run:
        args.record_run = args.record_run.expanduser().resolve()
        if args.record_run.exists() and any(args.record_run.iterdir()):
            parser.error(f"Recording directory is not empty: {args.record_run}")
    for key in ("record_video", "record_joints", "report"):
        value = getattr(args, key)
        if value is not None:
            value = value.expanduser().resolve()
            if value.exists():
                parser.error(f"Output already exists: {value}")
            setattr(args, key, value)
    return args


def _linger(app, world, seconds: float, keep_open: bool) -> None:
    end = time.monotonic() + seconds
    try:
        while app.is_running() and (keep_open or time.monotonic() < end):
            world.render()
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    scene_path = Path(resolve_scene(args.scene)).resolve()
    selected_policy_script = args.trajectory_policy_script or args.policy_script
    policy_path = _policy_path(selected_policy_script) if selected_policy_script else None
    policy = None
    trajectory_executor = None
    config = json.loads((PACKAGE_DIR / "config.json").read_text(encoding="utf-8"))
    robot_model = str(config.get("robot_model", "unknown"))
    policy_fps = int(config["policy_hz"])
    from yubi_isaac_sim_env.policy_adapter import trajectory_executor_profile
    trajectory_controller = {
        "name": args.trajectory_controller_profile,
        **trajectory_executor_profile(args.trajectory_controller_profile),
    }
    physics_fps = int(config["physics_hz"])
    record_fps = int(args.record_fps)
    if record_fps <= 0 or physics_fps % record_fps or record_fps % policy_fps:
        raise ValueError(
            f"--record-fps must be a positive multiple of policy_hz={policy_fps} "
            f"and divide physics_hz={physics_fps}"
        )
    capture_interval = physics_fps // record_fps
    video_enabled = args.record_run is not None or args.record_video is not None
    head_calibration = args.head_camera_calibration.expanduser().resolve() if args.head_camera_calibration else None
    camera_spec = _camera_spec(args.camera, head_calibration)
    camera_width, camera_height = camera_spec["resolution"]
    report_path = args.record_run / "report.json" if args.record_run else args.report
    manifest_path = args.record_run / "manifest.json" if args.record_run else None
    report: dict = {
        "status": "started",
        "scene": str(scene_path),
        "setup_requested": args.setup,
        "seed": args.seed,
        "episodes_requested": args.episodes,
        "steps_requested": args.steps,
        "policy": str(policy_path) if policy_path else args.policy,
        "policy_input_views": ["head", "left_wrist", "right_wrist"] if args.policy_images == "all" else [],
        "policy_output": "world_tool_waypoint_chunks" if args.trajectory_policy_script else "joint_targets",
        "robot_model": robot_model,
        "trajectory_controller_profile": trajectory_controller,
        "backend": None,
        "camera": camera_spec,
        "record_fps": record_fps,
        "interpolate_targets": args.interpolate_targets,
        "joint_command_profile": (
            get_joint_reference_profile(args.joint_command_profile).as_dict()
            if args.joint_command_profile != "direct"
            else {"name": "direct"}
        ),
        "episodes": [],
    }
    input_paths = [
        scene_path,
        PACKAGE_DIR / "assets" / "franka_yubi_panda.usdc",
        PACKAGE_DIR / "assets" / "yubi" / "source_manifest.json",
        PACKAGE_DIR / "config.json",
        PACKAGE_DIR / "scene_config.json",
    ]
    if args.camera == "head" or args.policy_images == "all":
        input_paths.append(head_calibration or HEAD_CALIBRATION_PATH)
    if args.camera in ("left_wrist", "right_wrist") or args.policy_images == "all":
        input_paths.append(WRIST_CAMERA_MODEL_PATH)
    if policy_path is not None:
        input_paths.append(policy_path)
    if args.trajectory_policy_script:
        input_paths.append(PACKAGE_DIR / "policy_adapter.py")
    if args.joint_command_profile != "direct":
        input_paths.append(PACKAGE_DIR / "command_conditioning.py")
    setup_file = _setup_file(args.setup)
    if setup_file is not None:
        input_paths.append(setup_file)
    manifest: dict = {
        "format_version": 1,
        "scene": str(scene_path),
        "setup_requested": args.setup,
        "seed": args.seed,
        "policy": str(policy_path) if policy_path else args.policy,
        "policy_input_views": ["head", "left_wrist", "right_wrist"] if args.policy_images == "all" else [],
        "policy_output": "world_tool_waypoint_chunks" if args.trajectory_policy_script else "joint_targets",
        "robot_model": robot_model,
        "trajectory_controller_profile": trajectory_controller,
        "sampling": f"reset state plus one sample every {capture_interval} physics frames",
        "frame_alignment": "video frame n corresponds to joint CSV sample_index n within each episode",
        "video_fps": record_fps,
        "policy_fps": policy_fps,
        "interpolate_targets": args.interpolate_targets,
        "joint_command_profile": (
            get_joint_reference_profile(args.joint_command_profile).as_dict()
            if args.joint_command_profile != "direct"
            else {"name": "direct"}
        ),
        "camera": camera_spec,
        "input_sha256": {str(path): _hash(path) for path in input_paths},
        "episodes": [],
    }
    app = None
    env = None
    exit_code = 1
    current_step = None
    try:
        app, env = create_sim(
            scene=scene_path,
            gui=not args.headless,
            setup=args.setup,
            seed=args.seed,
            scenario_index=args.fixed_index if args.fixed_index is not None else args.index_start,
            head_camera_calibration=head_calibration,
        )
        report["backend"] = env.backend
        env.configure_joint_command_profile(args.joint_command_profile)
        if env.control_decimation % capture_interval:
            raise RuntimeError("Recording interval does not divide a policy step")
        # Isaac/Omniverse modules imported by a custom policy need Kit ready.
        if policy_path is not None:
            policy = _load_callable(policy_path, "predict" if args.trajectory_policy_script else "act")
        if args.trajectory_policy_script:
            from yubi_isaac_sim_env.policy_adapter import (
                TrajectoryChunkExecutor,
            )
            trajectory_executor = TrajectoryChunkExecutor(
                policy_hz=policy_fps,
                **trajectory_executor_profile(args.trajectory_controller_profile),
            )
        policy_specs = ({name: _camera_spec(name, head_calibration) for name in ("head", "left_wrist", "right_wrist")}
                        if args.policy_images == "all" else {})
        camera = _make_camera(camera_spec, gui=not args.headless,
                              recording_video=video_enabled or args.camera in policy_specs)
        policy_cameras = {}
        if policy_specs:
            for name, spec in policy_specs.items():
                policy_cameras[name] = camera if name == args.camera else _make_camera(
                    spec, gui=False, recording_video=True
                )
        env.world.render()
        for episode in range(args.episodes):
            index = args.fixed_index if args.fixed_index is not None else args.index_start + episode
            observation = env.reset(seed=args.seed, scenario_index=index, setup=args.setup)
            if trajectory_executor is not None:
                trajectory_executor.reset(episode)
            video_path, joint_path = _paths(args, episode)
            entry = {
                "episode": episode,
                "scenario_index": index,
                "scenario_id": observation["scenario"]["id"],
                "video": str(video_path) if video_path else None,
                "joints": str(joint_path) if joint_path else None,
                "policy_steps": 0,
                "video_frames": 0,
                "joint_samples": 0,
                "success": False,
            }
            manifest["episodes"].append(entry)
            initial = observation
            min_observed_interarm_clearance_m = observed_interarm_clearance_m(observation)
            transitions = []
            video = None
            joints = None
            try:
                video = VideoWriter(video_path, camera_width, camera_height, record_fps) if video_path else None
                joints = JointStateWriter(joint_path) if joint_path else None
                _sample(video, joints, camera, env.world, camera_spec, episode, 0, "reset", -1, observation)
                sample_index = 0
                for step in range(args.steps):
                    current_step = step
                    policy_observation = (_policy_observation(observation, policy_cameras, policy_specs, env.world)
                                          if policy_specs else observation)
                    if trajectory_executor is not None:
                        action = trajectory_executor.act(policy_observation, step, episode, policy)
                    else:
                        action = policy(policy_observation, step, episode) if policy else _builtin_action(
                            observation, step, args.policy
                        )
                    def capture(physics_substep: int) -> None:
                        nonlocal sample_index, min_observed_interarm_clearance_m
                        if physics_substep % capture_interval:
                            return
                        sample_index += 1
                        frame_observation = (env.observe() if joints is not None or trajectory_executor is not None
                                             else observation)
                        clearance = observed_interarm_clearance_m(frame_observation)
                        if clearance is not None:
                            min_observed_interarm_clearance_m = min(
                                min_observed_interarm_clearance_m, clearance
                            ) if min_observed_interarm_clearance_m is not None else clearance
                        _sample(video, joints, camera, env.world, camera_spec,
                                episode, sample_index, "step", step, frame_observation)

                    observation, reward, terminated, truncated, info = env.step(
                        action,
                        on_physics_step=capture if (video is not None or joints is not None) else None,
                        interpolate_targets=args.interpolate_targets,
                    )
                    clearance = observed_interarm_clearance_m(observation)
                    if clearance is not None:
                        min_observed_interarm_clearance_m = min(
                            min_observed_interarm_clearance_m, clearance
                        ) if min_observed_interarm_clearance_m is not None else clearance
                    transitions.append(
                        {
                            "policy_step": step,
                            "physics_time_s": observation["physics_time_s"],
                            "action": action,
                            "reward": reward,
                            "terminated": terminated,
                            "truncated": truncated,
                            "is_success": info["is_success"],
                            "spatial_alignment": _spatial_alignment_state(observation),
                            "tool_poses": {side: robot["tool_pose"] for side, robot in observation["robots"].items()},
                            "interarm_guard": (trajectory_executor.last_interarm_guard
                                               if trajectory_executor is not None else {"active": False}),
                        }
                    )
                    if (terminated and not args.continue_after_success) or truncated or not app.is_running():
                        break
            finally:
                if joints is not None:
                    joints.close()
                    entry["joint_samples"] = joints.sample_count
                if video is not None:
                    video.close()
                    entry["video_frames"] = video.frame_count
            entry["policy_steps"] = len(transitions)
            entry["success"] = bool(transitions and transitions[-1]["is_success"])
            entry["ever_success"] = any(transition["is_success"] for transition in transitions)
            entry["first_success_policy_step"] = next(
                (transition["policy_step"] for transition in transitions if transition["is_success"]), None
            )
            episode_report = {
                **entry,
                "initial_observation": initial,
                "final_observation": observation,
                "min_observed_interarm_clearance_m": min_observed_interarm_clearance_m,
                "object_displacement_m": {
                    name: math.dist(initial["objects"][name]["position_m"], observation["objects"][name]["position_m"])
                    for name in ("cup", "plate")
                },
                "transitions": transitions,
            }
            report["episodes"].append(episode_report)
            print(
                f"Episode {episode}: setup={entry['scenario_id']}, steps={entry['policy_steps']}, "
                f"success={entry['success']}, video={entry['video']}, joints={entry['joints']}",
                flush=True,
            )
        report["status"] = "completed"
        exit_code = 0
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["error_policy_step"] = current_step
        report["traceback"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        manifest["status"] = report["status"]
        if "error" in report:
            manifest["error"] = report["error"]
        if report_path:
            _write_json(report_path, report)
            print(f"Report: {report_path}", flush=True)
        if manifest_path:
            _write_json(manifest_path, manifest)
            print(f"Manifest: {manifest_path}", flush=True)
        if app is not None and env is not None and not args.headless:
            _linger(app, env.world, args.linger_seconds if args.linger_seconds is not None else 5.0, args.keep_open)
        sys.stdout.flush()
        sys.stderr.flush()
        if app is not None:
            # Isaac Sim 5.1 GPU teardown can hang or segfault on this host.
            # The video encoder, CSV, and JSON files were closed and flushed.
            os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
