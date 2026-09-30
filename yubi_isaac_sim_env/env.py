"""Isaac Sim 5.1 GPU dual-Franka/YUBI cup/plate experiment interface.

Construct this class only after ``SimulationApp`` has started and a composed
dual-Franka/YUBI USD stage is open. Both arms receive joint-position targets;
objects remain dynamic rigid bodies and are never moved by policy actions.
``reset`` teleports them only between episodes, then clears their velocities.

The motorized YUBI uses one normalized command for a mirrored revolute pair.
``q_closed_rad`` and ``q_open_rad`` must be calibrated for the motorized asset;
the operator-side YUBI Xacro is not a source for physical actuation limits.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

from .command_conditioning import JointReferenceGovernor, get_joint_reference_profile
from .setup_catalog import resolve_setup

ROOT = Path(__file__).resolve().parent


ARM_DOF_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
DRIVEN_JAW_DOF_NAME = "yubi_finger_joint"
MIMIC_JAW_DOF_NAME = "yubi_finger_mimic_joint"
ROBOT_PATHS = {
    "left": "/World/Robots/LeftMount/Panda",
    "right": "/World/Robots/RightMount/Panda",
}
OBJECT_PATHS = {"cup": "/World/Objects/Cup", "plate": "/World/Objects/Tray"}
COLLISION_POINT_RADII_M = {
    "panda_link2": 0.08,
    "panda_link3": 0.08,
    "panda_link4": 0.08,
    "panda_link5": 0.075,
    "panda_link6": 0.07,
    "panda_link7": 0.065,
    "yubi_base": 0.055,
    "yubi_leftfinger": 0.035,
    "yubi_rightfinger": 0.035,
}


def as_list(value: Any) -> list:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def rgb_from_hex(color: str) -> tuple[float, float, float]:
    if len(color) != 7 or not color.startswith("#"):
        raise ValueError(f"Expected #RRGGBB color, got {color!r}")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))


class DualFrankaYubiCupPlateEnv:
    """Minimal Gym-style reset/step API using GPU PhysX tensor views.

    Action schema::

        {"left": {"arm_joint_targets_rad": [7 floats], "gripper_open_fraction": 1.0},
         "right": {"arm_joint_targets_rad": [7 floats], "gripper_open_fraction": 1.0}}

    Omitted arms or keys keep their previous targets. The fraction maps
    linearly from calibrated ``q_closed_rad`` to ``q_open_rad`` for the driven
    jaw. The simulator sends the inverse target to the other jaw; this models
    the physical gear pair without changing the single-command hardware API.
    This is joint-space control; no Cartesian IK or grasp
    planner is implied by the API.
    """

    def __init__(
        self,
        scene_config: dict | None = None,
        task_config: dict | None = None,
        *,
        render: bool = False,
    ) -> None:
        import omni.usd
        import torch
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation, RigidPrim
        from isaacsim.core.utils.types import ArticulationActions
        from pxr import PhysxSchema, UsdPhysics

        self.torch = torch
        self.ArticulationActions = ArticulationActions
        self.scene_config = scene_config or json.loads((ROOT / "scene_config.json").read_text())
        self.task_config = task_config or json.loads((ROOT / "config.json").read_text())
        yubi_config = self.task_config.get("yubi", self.task_config)
        try:
            self.q_closed_rad = float(yubi_config["q_closed_rad"])
            self.q_open_rad = float(yubi_config["q_open_rad"])
        except KeyError as exc:
            raise ValueError("YUBI config needs calibrated q_closed_rad and q_open_rad") from exc
        self.home_open_fraction = float(yubi_config.get("home_open_fraction", 1.0))
        if not (
            math.isfinite(self.q_closed_rad)
            and math.isfinite(self.q_open_rad)
            and self.q_closed_rad != self.q_open_rad
            and math.isfinite(self.home_open_fraction)
            and 0.0 <= self.home_open_fraction <= 1.0
        ):
            raise ValueError("Invalid calibrated YUBI jaw endpoints or home_open_fraction")
        self.render = bool(render)
        self.stage = omni.usd.get_context().get_stage()
        if self.stage is None:
            raise RuntimeError("Open a dual-Franka/YUBI USD stage before constructing the environment")
        for path in [*ROBOT_PATHS.values(), *OBJECT_PATHS.values(), "/World/PhysicsScene"]:
            if not self.stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"Required prim missing: {path}")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU PhysX requested but PyTorch cannot see CUDA")
        self.world = World(
            physics_dt=1.0 / self.task_config["physics_hz"],
            rendering_dt=1.0 / self.task_config["physics_hz"],
            stage_units_in_meters=1.0,
            device="cuda:0",
        )
        self.device = torch.device("cuda:0")
        self.objects = {
            name: self.world.scene.add(RigidPrim(prim_paths_expr=path, name=f"{name}_gpu"))
            for name, path in OBJECT_PATHS.items()
        }
        self.robots = self.world.scene.add(
            Articulation(
                prim_paths_expr=list(ROBOT_PATHS.values()),
                name="dual_franka_yubi_gpu",
                reset_xform_properties=False,
            )
        )
        for path in ROBOT_PATHS.values():
            for link_name in COLLISION_POINT_RADII_M:
                if not self.stage.GetPrimAtPath(f"{path}/{link_name}").IsValid():
                    raise RuntimeError(f"Required collision link missing: {path}/{link_name}")
        self.robot_links = {
            side: {
                part: self.world.scene.add(
                    RigidPrim(prim_paths_expr=f"{path}/{prim}", name=f"{side}_{part}_gpu")
                )
                for part, prim in (
                    *((name, name) for name in COLLISION_POINT_RADII_M),
                    ("base", "yubi_base"),
                    ("left_finger", "yubi_leftfinger"),
                    ("right_finger", "yubi_rightfinger"),
                )
            }
            for side, path in ROBOT_PATHS.items()
        }
        # A fixed, massless /Panda/yubi_tool Xform is a useful camera/IK frame,
        # but it is not an articulation body and has no Jacobian row.
        for side, path in ROBOT_PATHS.items():
            tool_prim = self.stage.GetPrimAtPath(f"{path}/yubi_tool")
            if tool_prim.IsValid() and tool_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                self.robot_links[side]["tool"] = self.world.scene.add(
                    RigidPrim(prim_paths_expr=f"{path}/yubi_tool", name=f"{side}_tool_gpu")
                )
        physx = PhysxSchema.PhysxSceneAPI(self.stage.GetPrimAtPath("/World/PhysicsScene"))
        self.backend = {
            "world_device": str(self.world.device),
            "gpu_dynamics": bool(physx.GetEnableGPUDynamicsAttr().Get()),
            "broadphase": str(physx.GetBroadphaseTypeAttr().Get()),
            "cuda_name": torch.cuda.get_device_name(0),
        }
        if not (self.backend["world_device"].startswith("cuda") and self.backend["gpu_dynamics"] and self.backend["broadphase"] == "GPU"):
            raise RuntimeError(f"Isaac Sim did not enable GPU physics: {self.backend}")
        self.world.reset()
        found = [str(path) for path in self.robots.prim_paths]
        articulation_roots = {side: path + "/root_joint" for side, path in ROBOT_PATHS.items()}
        if set(found) != set(articulation_roots.values()):
            raise RuntimeError(f"Expected two Franka articulations, found: {found}")
        self.robot_index = {side: found.index(path) for side, path in articulation_roots.items()}
        self.dof_names = list(self.robots.dof_names)
        expected = set(ARM_DOF_NAMES + (DRIVEN_JAW_DOF_NAME, MIMIC_JAW_DOF_NAME))
        if not expected.issubset(self.dof_names):
            raise RuntimeError(f"Franka/YUBI joint names differ from expected: {self.dof_names}")
        self.dof_index = {name: self.dof_names.index(name) for name in expected}
        self.arm_dof_indices = [self.dof_index[name] for name in ARM_DOF_NAMES]
        self.command_dof_indices = self.arm_dof_indices + [
            self.dof_index[DRIVEN_JAW_DOF_NAME], self.dof_index[MIMIC_JAW_DOF_NAME]
        ]
        # Isaac's articulation Jacobian excludes the fixed root body. A
        # separate rigid yubi_tool body may exist; otherwise use yubi_base.
        body_names = list(self.robots.body_names)
        missing_collision_bodies = set(COLLISION_POINT_RADII_M) - set(body_names)
        if missing_collision_bodies:
            raise RuntimeError(f"Panda collision bodies missing from articulation: {missing_collision_bodies}")
        self.collision_jacobian_indices = {
            name: body_names.index(name) - 1 for name in COLLISION_POINT_RADII_M
        }
        if any(index < 0 for index in self.collision_jacobian_indices.values()):
            raise RuntimeError("A collision link cannot be the articulation root")
        tool_views_exist = all("tool" in self.robot_links[side] for side in ROBOT_PATHS)
        self.tool_body_name = "yubi_tool" if "yubi_tool" in body_names and tool_views_exist else "yubi_base"
        if self.tool_body_name not in body_names:
            raise RuntimeError(f"YUBI tool body missing from articulation: {body_names}")
        self.tool_jacobian_index = body_names.index(self.tool_body_name) - 1
        if self.tool_jacobian_index < 0:
            raise RuntimeError("YUBI tool cannot be the articulation root")
        jaw_limits = self.torch.as_tensor(self.robots.get_dof_limits(), device=self.device)
        self.arm_joint_limits_rad = {
            side: as_list(jaw_limits[row, self.arm_dof_indices, :])
            for side, row in self.robot_index.items()
        }
        driven_index = self.dof_index[DRIVEN_JAW_DOF_NAME]
        mimic_index = self.dof_index[MIMIC_JAW_DOF_NAME]
        for row in self.robot_index.values():
            driven_low, driven_high = [float(v) for v in jaw_limits[row, driven_index]]
            mimic_low, mimic_high = [float(v) for v in jaw_limits[row, mimic_index]]
            for q in (self.q_closed_rad, self.q_open_rad):
                # USD degrees -> PhysX float32 radians can round 0.7 down by
                # 1.2e-8. Allow conversion noise, not a meaningful overtravel.
                if not (driven_low - 1e-6 <= q <= driven_high + 1e-6
                        and mimic_low - 1e-6 <= -q <= mimic_high + 1e-6):
                    raise ValueError(
                        "Configured YUBI jaw endpoints exceed the USD joint limits "
                        f"(driven=[{driven_low}, {driven_high}], mimic=[{mimic_low}, {mimic_high}])"
                    )
        self.home_targets = self.torch.zeros((2, self.robots.num_dof), dtype=self.torch.float32, device=self.device)
        for side in ROBOT_PATHS:
            row = self.robot_index[side]
            for name, value in zip(ARM_DOF_NAMES, self.task_config["home_arm_joint_rad"]):
                self.home_targets[row, self.dof_index[name]] = float(value)
            jaw_q = self._jaw_target(self.home_open_fraction)
            self.home_targets[row, self.dof_index[DRIVEN_JAW_DOF_NAME]] = jaw_q
            self.home_targets[row, self.dof_index[MIMIC_JAW_DOF_NAME]] = -jaw_q
        self.current_targets = self.home_targets.clone()
        self.control_decimation = round(self.task_config["physics_hz"] / self.task_config["policy_hz"])
        if self.control_decimation < 1 or self.control_decimation * self.task_config["policy_hz"] != self.task_config["physics_hz"]:
            raise ValueError("physics_hz must be an integer multiple of policy_hz")
        self.policy_steps = 0
        self.success_streak = 0
        self.scenario: dict | None = None
        self.joint_command_profile = "direct"
        self._joint_reference_governors: dict[str, JointReferenceGovernor] = {}
        self._joint_reference_tick = 0
        self.gripper_max_velocity_rad_s = float(yubi_config["grip_drive_max_velocity_rad_s"])
        if not math.isfinite(self.gripper_max_velocity_rad_s) or self.gripper_max_velocity_rad_s <= 0:
            raise ValueError("Invalid YUBI grip_drive_max_velocity_rad_s")

    def configure_joint_command_profile(self, name: str) -> None:
        """Select servo-rate command conditioning for subsequent episodes.

        The selected profile uses that robot's official interface velocity,
        acceleration, and jerk limits. The
        NumPy governor is shared with hardware integrations and intentionally
        does not change the policy or trajectory output contract.
        """
        get_joint_reference_profile(name)  # Validate before mutating state.
        self.joint_command_profile = name
        self._joint_reference_governors.clear()

    def _jaw_target(self, open_fraction: float) -> float:
        """Map a normalized aperture command to the calibrated driven-jaw angle."""
        return self.q_closed_rad + open_fraction * (self.q_open_rad - self.q_closed_rad)

    def _set_material_colors(self, scenario: dict) -> None:
        from pxr import Gf

        self.stage.SetEditTarget(self.stage.GetSessionLayer())
        for object_name, color_key, shader_path in (
            ("cup", "cup_color", "/World/Objects/Cup/Looks/CupMaterial/PreviewSurface"),
            ("plate", "tray_color", "/World/Objects/Tray/Looks/TrayMaterial/PreviewSurface"),
        ):
            shader = self.stage.GetPrimAtPath(shader_path)
            if not shader.IsValid():
                raise RuntimeError(f"{object_name} color shader missing: {shader_path}")
            shader.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*rgb_from_hex(scenario[color_key])))

    def _position(self, name: str) -> list[float]:
        position, _ = self.objects[name].get_world_poses()
        return [float(x) for x in as_list(position[0])]

    def reset(
        self,
        seed: int = 20260924,
        scenario_index: int = 0,
        *,
        setup: str | Path | dict | None = None,
    ) -> dict:
        """Restore the arms and load a named, custom, or seeded random setup.

        ``setup`` accepts ``random:N``, a packaged setup name, a JSON path,
        or a dictionary. Private episode presets can also be loaded when
        supplied locally under ``setups/``. When omitted, ``seed`` and
        ``scenario_index`` preserve the original deterministic sampler API.
        """
        scenario = resolve_setup(
            setup,
            scene_config=self.scene_config,
            seed=int(seed),
            scenario_index=scenario_index,
        )
        reset_targets = self.home_targets.clone()
        initial_joints = scenario.get("initial_arm_joint_rad", {})
        if not isinstance(initial_joints, dict) or set(initial_joints) - set(ROBOT_PATHS):
            raise ValueError("initial_arm_joint_rad must map left/right to seven joint angles")
        for side, values in initial_joints.items():
            if not isinstance(values, list) or len(values) != 7:
                raise ValueError(f"{side} initial_arm_joint_rad needs seven radians")
            for index, (value, bounds) in enumerate(zip(values, self.arm_joint_limits_rad[side])):
                if not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                    raise ValueError(f"{side} initial joint {index + 1} is outside Panda limits")
                reset_targets[self.robot_index[side], self.arm_dof_indices[index]] = value
        initial_gripper = float(scenario.get("initial_gripper_open_fraction", self.home_open_fraction))
        if not math.isfinite(initial_gripper) or not 0.0 <= initial_gripper <= 1.0:
            raise ValueError("initial_gripper_open_fraction must be in [0, 1]")
        initial_jaw_q = self._jaw_target(initial_gripper)
        for row in self.robot_index.values():
            reset_targets[row, self.dof_index[DRIVEN_JAW_DOF_NAME]] = initial_jaw_q
            reset_targets[row, self.dof_index[MIMIC_JAW_DOF_NAME]] = -initial_jaw_q
        self.world.reset()
        zeros = self.torch.zeros_like(self.home_targets)
        self.robots.set_joint_positions(reset_targets)
        self.robots.set_joint_velocities(zeros)
        # Both simulated jaw drives receive mirrored targets. The physical
        # YUBI adapter still receives one normalized actuator command.
        self.robots.apply_action(
            self.ArticulationActions(
                joint_positions=reset_targets[:, self.command_dof_indices],
                joint_indices=self.command_dof_indices,
            )
        )
        self.current_targets = reset_targets.clone()
        profile = get_joint_reference_profile(self.joint_command_profile)
        self._joint_reference_governors.clear()
        self._joint_reference_tick = 0
        if profile is not None:
            for side, row in self.robot_index.items():
                governor = JointReferenceGovernor(profile, dt_s=profile.servo_period_s)
                governor.reset(as_list(reset_targets[row, self.arm_dof_indices]))
                self._joint_reference_governors[side] = governor
        # Advance the articulation once before installing the randomized rigid
        # bodies. Otherwise this step can move the freshly placed cup/plate.
        self.world.step(render=self.render)
        zero_velocities = self.torch.zeros((1, 6), dtype=self.torch.float32, device=self.device)
        for name, key in (("cup", "cup"), ("plate", "tray")):
            pose = self.torch.tensor([scenario[f"{key}_position_m"]], dtype=self.torch.float32, device=self.device)
            quaternion = self.torch.tensor(
                [scenario[f"{key}_quaternion_wxyz"]], dtype=self.torch.float32, device=self.device
            )
            self.objects[name].set_world_poses(positions=pose, orientations=quaternion)
            # The combined setter supports GPU PhysX rigid-body tensor views.
            self.objects[name].set_velocities(zero_velocities)
        self._set_material_colors(scenario)
        self.scenario = scenario
        self.policy_steps = 0
        self.success_streak = 0
        return self.observe()

    def observe(self) -> dict:
        objects = {}
        for name, view in self.objects.items():
            position, quaternion = view.get_world_poses()
            objects[name] = {
                "position_m": as_list(position[0]),
                "quaternion_wxyz": as_list(quaternion[0]),
                "linear_velocity_m_s": as_list(view.get_linear_velocities()[0]),
                "angular_velocity_rad_s": as_list(view.get_angular_velocities()[0]),
            }
        positions = as_list(self.robots.get_joint_positions())
        velocities = as_list(self.robots.get_joint_velocities())
        jacobians = self.robots.get_jacobians()
        robots = {}
        for side, index in self.robot_index.items():
            link_poses = {}
            for part, view in self.robot_links[side].items():
                position, quaternion = view.get_world_poses()
                link_poses[part] = {
                    "position_m": as_list(position[0]),
                    "quaternion_wxyz": as_list(quaternion[0]),
                }
            driven_q = float(positions[index][self.dof_index[DRIVEN_JAW_DOF_NAME]])
            open_fraction = (driven_q - self.q_closed_rad) / (self.q_open_rad - self.q_closed_rad)
            tool_part = "tool" if self.tool_body_name == "yubi_tool" and "tool" in link_poses else "base"
            tool_jacobian = as_list(
                jacobians[index, self.tool_jacobian_index, :6, self.arm_dof_indices]
            )
            robots[side] = {
                "joint_names": self.dof_names,
                "joint_positions": positions[index],
                "joint_velocities": velocities[index],
                "arm_joint_limits_rad": [pair[:] for pair in self.arm_joint_limits_rad[side]],
                "link_poses": link_poses,
                "tool_body_name": self.tool_body_name,
                "tool_pose": link_poses[tool_part],
                "tool_jacobian": tool_jacobian,
                "tool_translation_jacobian": tool_jacobian[:3],
                "gripper_joint_position_rad": driven_q,
                "gripper_mimic_joint_position_rad": float(
                    positions[index][self.dof_index[MIMIC_JAW_DOF_NAME]]
                ),
                "gripper_open_fraction": open_fraction,
                "collision_points": [
                    {
                        "name": name,
                        "position_m": link_poses[name]["position_m"],
                        "radius_m": radius,
                        "arm_translation_jacobian": as_list(
                            jacobians[index, self.collision_jacobian_indices[name], :3, self.arm_dof_indices]
                        ),
                    }
                    for name, radius in COLLISION_POINT_RADII_M.items()
                ],
            }
        return {
            "objects": objects,
            "robots": robots,
            "scenario": self.scenario,
            "policy_step": self.policy_steps,
            "physics_time_s": float(self.world.current_time),
        }

    def _success(self, observation: dict) -> bool:
        cup = observation["objects"]["cup"]
        plate = observation["objects"]["plate"]
        cfg = self.task_config["success"]
        cup_pos, plate_pos = cup["position_m"], plate["position_m"]
        radius = math.dist(cup_pos[:2], plate_pos[:2])
        target_height = plate_pos[2] + self.scene_config["tray"]["base_thickness"]
        w, x, y, z = cup["quaternion_wxyz"]
        z_axis_vertical = 1 - 2 * (x * x + y * y)
        speed = math.sqrt(sum(v * v for v in cup["linear_velocity_m_s"]))
        angular_speed = math.sqrt(sum(v * v for v in cup["angular_velocity_rad_s"]))
        return (
            radius <= cfg["cup_center_radius_m"]
            and abs(cup_pos[2] - target_height) <= cfg["cup_bottom_height_tolerance_m"]
            and z_axis_vertical >= math.cos(math.radians(cfg["cup_max_tilt_deg"]))
            and speed <= cfg["max_linear_speed_m_s"]
            and angular_speed <= cfg["max_angular_speed_rad_s"]
        )

    def step(
        self,
        action: dict,
        *,
        on_physics_step: Callable[[int], None] | None = None,
        interpolate_targets: bool = False,
    ) -> tuple[dict, float, bool, bool, dict]:
        """Advance one policy period and optionally observe each physics step.

        The callback receives the 1-based physics substep index after the
        simulator advances. It lets a recorder sample camera frames more often
        than the policy acts, without changing robot action timing.
        """
        if self.scenario is None:
            raise RuntimeError("Call reset before step")
        if interpolate_targets and self.joint_command_profile != "direct":
            raise ValueError("Choose either linear target interpolation or a joint command profile")
        targets = self.current_targets.clone()
        for side, command in action.items():
            if side not in self.robot_index or not isinstance(command, dict):
                raise ValueError(f"Unknown arm action {side!r}")
            row = self.robot_index[side]
            if "arm_joint_targets_rad" in command:
                values = command["arm_joint_targets_rad"]
                if len(values) != 7 or not all(math.isfinite(float(v)) for v in values):
                    raise ValueError(f"{side} arm needs seven finite joint targets")
                for name, value in zip(ARM_DOF_NAMES, values):
                    targets[row, self.dof_index[name]] = float(value)
            if "gripper_open_fraction" in command:
                open_fraction = float(command["gripper_open_fraction"])
                if not math.isfinite(open_fraction) or not 0.0 <= open_fraction <= 1.0:
                    raise ValueError(f"{side} gripper_open_fraction must be in [0, 1]")
                jaw_q = self._jaw_target(open_fraction)
                targets[row, self.dof_index[DRIVEN_JAW_DOF_NAME]] = jaw_q
                targets[row, self.dof_index[MIMIC_JAW_DOF_NAME]] = -jaw_q
        limits = self.torch.as_tensor(self.robots.get_dof_limits(), device=self.device)
        controlled = targets[:, self.command_dof_indices]
        controlled_limits = limits[:, self.command_dof_indices, :]
        controlled = self.torch.minimum(
            self.torch.maximum(controlled, controlled_limits[:, :, 0]), controlled_limits[:, :, 1]
        )
        targets[:, self.command_dof_indices] = controlled
        targets[:, self.dof_index[MIMIC_JAW_DOF_NAME]] = -targets[:, self.dof_index[DRIVEN_JAW_DOF_NAME]]
        self.current_targets = targets
        conditioned_targets = self.joint_command_profile != "direct"
        if interpolate_targets or conditioned_targets:
            start = self.torch.as_tensor(
                self.robots.get_joint_positions(), dtype=controlled.dtype, device=self.device
            )[:, self.command_dof_indices].clone()
        else:
            self.robots.apply_action(
                self.ArticulationActions(
                    joint_positions=controlled,
                    joint_indices=self.command_dof_indices,
                )
            )
        for substep in range(1, self.control_decimation + 1):
            if conditioned_targets:
                intermediate = controlled.clone()
                # The arm reference governor is identical for simulation and
                # hardware. The normalized YUBI jaw is velocity-limited by the
                # XM430-W350-R 12 V no-load speed and mirrored explicitly.
                elapsed = substep / float(self.task_config["physics_hz"])
                jaw_delta = self.torch.clamp(
                    controlled[:, -2] - start[:, -2],
                    min=-self.gripper_max_velocity_rad_s * elapsed,
                    max=self.gripper_max_velocity_rad_s * elapsed,
                )
                intermediate[:, -2] = start[:, -2] + jaw_delta
                intermediate[:, -1] = -intermediate[:, -2]
                profile = get_joint_reference_profile(self.joint_command_profile)
                assert profile is not None
                physics_step = self.policy_steps * self.control_decimation + substep
                target_tick = round(
                    physics_step / (float(self.task_config["physics_hz"]) * profile.servo_period_s)
                )
                for side, row in self.robot_index.items():
                    desired = as_list(controlled[row, : len(ARM_DOF_NAMES)])
                    governed = None
                    for _ in range(target_tick - self._joint_reference_tick):
                        governed = self._joint_reference_governors[side].update(desired)
                    if governed is None:
                        governed = self._joint_reference_governors[side].position.copy()
                    intermediate[row, : len(ARM_DOF_NAMES)] = self.torch.as_tensor(
                        governed, dtype=controlled.dtype, device=self.device
                    )
                self._joint_reference_tick = target_tick
                self.robots.apply_action(
                    self.ArticulationActions(
                        joint_positions=intermediate,
                        joint_indices=self.command_dof_indices,
                    )
                )
            elif interpolate_targets:
                alpha = substep / self.control_decimation
                intermediate = start + alpha * (controlled - start)
                self.robots.apply_action(
                    self.ArticulationActions(
                        joint_positions=intermediate,
                        joint_indices=self.command_dof_indices,
                    )
                )
            self.world.step(render=self.render)
            if on_physics_step is not None:
                on_physics_step(substep)
        self.policy_steps += 1
        observation = self.observe()
        pose_success = self._success(observation)
        self.success_streak = self.success_streak + 1 if pose_success else 0
        success = self.success_streak >= int(self.task_config["success"]["dwell_policy_steps"])
        cup_xy = observation["objects"]["cup"]["position_m"][:2]
        plate_xy = observation["objects"]["plate"]["position_m"][:2]
        reward = 1.0 if success else -math.dist(cup_xy, plate_xy)
        truncated = self.policy_steps >= self.task_config["max_policy_steps"]
        info = {
            "is_success": success,
            "pose_success": pose_success,
            "success_streak": self.success_streak,
            "gpu_backend": self.backend,
            "scenario_id": self.scenario["id"],
        }
        return observation, reward, success, truncated, info
