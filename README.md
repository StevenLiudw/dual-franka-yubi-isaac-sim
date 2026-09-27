# Dual Franka YUBI cup and plate simulation

An Isaac Sim 5.1 environment for testing dual-arm manipulation policies. The scene has two Franka Panda arms with motorized YUBI grippers, a table, a dynamic IKEA KALAS cup and plate, a head camera, and two gripper-mounted wrist cameras. The package includes deterministic object layouts, a `reset`/`step` API, policy hooks, and episode recording.

The cup, plate, table, camera poses, and YUBI dynamics contain estimates. See [the environment notes](yubi_isaac_sim_env/README.md) and [camera and mount audit](yubi_isaac_sim_env/CAMERA_MOUNT_REPORT.md) before using simulation results to predict real robot performance.

## Which assets are included?

The repository includes the cup, plate, table, tray, YUBI CAD and derived meshes, scene USDs, and the composed Franka–YUBI USD. The composed USD references one **missing runtime dependency**: NVIDIA's stock `franka_panda` directory, containing `franka_panda.usd` and four USD files under `configuration/`. Isaac Sim itself and policy checkpoints are also not included.

There is no public Drive bundle for the stock Panda files. The redistribution rights for this exact five-file NVIDIA asset have not been established, and the [Isaac Sim Additional Software and Materials License](https://docs.nvidia.com/NVIDIA-IsaacSim-Additional-Software-and-Materials-License.pdf) restricts distributing covered software and materials. Obtain a compatible copy through [NVIDIA's Isaac Sim 5.1 downloads and asset packs](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html) or a licensed installation, then provision it locally as shown below. This keeps the project assets public without republishing NVIDIA's files.

## Requirements

- Linux workstation with an NVIDIA RTX GPU and a graphical desktop for the visible GUI. The environment requires CUDA and GPU PhysX. Use `--headless` on a GPU machine without a display.
- [Isaac Sim 5.1](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html) in Python 3.11. NVIDIA's [requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) and [workstation setup](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html) give the supported drivers and GPU configurations.
- `trimesh` in that Python environment for rebuilding the composed robot asset; `ffmpeg` on `PATH` for MP4 recording.
- A local copy of the Franka Panda USD asset from your licensed Isaac Sim installation. The provisioning command below places it at the relative path expected by the scene.

For a Python environment install, follow NVIDIA's Isaac Sim 5.1 instructions. One possible Linux setup is:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install trimesh
export ISAAC_PYTHON="$PWD/.venv/bin/python"
```

Run the following commands from this repository's root. If Isaac Sim is already installed, set `ISAAC_PYTHON` to the Python executable or `python.sh` from that installation instead. Keep the variable set for the asset build scripts as well as the runner.

## Provision assets and create the scene

Point the provisioner at a **directory** containing `franka_panda.usd` and its `configuration/` files:

```bash
"$ISAAC_PYTHON" scripts/provision_franka.py --source /path/to/franka_panda
bash yubi_isaac_sim_env/rebuild_assets.sh
bash yubi_isaac_sim_env/pxr_python.sh -m yubi_isaac_sim_env.validate_assets
```

If you downloaded an NVIDIA asset pack, locate `franka_panda.usd` in its extracted files and pass its **parent directory** to `--source`. The provisioner checks the five required USD files and their references. A different Panda asset layout or release may need adaptation; do not point it at an unrelated `franka.usd` file.

The build creates `yubi_isaac_sim_env/assets/franka_yubi_panda.usdc` from the local Franka reference, the packaged YUBI meshes, and the JSON configuration. The validator checks the USD dependencies, GPU PhysX settings, both arm mounts, cameras, and asset hashes without launching the GUI. The `yubi_isaac_sim_env/scenes/` directory contains the composed stage loaded by the runner. Rebuild after changing geometry, robot configuration, or camera mounting in the asset builder. Changing only an object's reset pose or color needs no rebuild.

## Start the visible simulator

```bash
"$ISAAC_PYTHON" -m yubi_isaac_sim_env.run \
  --setup random:0 --seed 42 --camera head --policy hold \
  --steps 100 --keep-open
```

The GUI is the default. `--camera` selects `head`, `left_wrist`, `right_wrist`, or `overview` for the viewport and the single-camera recorder. The head view approximates the source center videos; the wrist views use a nominal ELP fisheye model. `--keep-open` keeps rendering after the episode until the window closes. `--headless` disables the GUI while retaining GPU physics and RTX image rendering.

The runner uses a 60 Hz physics step, a 10 Hz policy step, and 30 Hz video and joint sampling by default. `--steps 100` therefore allows up to ten simulated seconds. An episode can finish earlier if the task success condition is met.

## Using another laptop

Cloning this repository alone does **not** make the simulator runnable. The machine that executes `yubi_isaac_sim_env.run` needs Isaac Sim 5.1, a supported NVIDIA RTX GPU and driver, and the locally provisioned Panda asset. A compatible Linux laptop can install those dependencies and follow the commands above. A laptop without Isaac Sim or a suitable GPU can instead submit runs to a prepared GPU workstation and inspect the returned recordings:

```bash
ssh GPU_HOST 'cd /path/to/dual-franka-yubi-isaac-sim && /path/to/isaac-python -m yubi_isaac_sim_env.run --headless --setup random:0 --seed 42 --policy hold --steps 100 --record-run runs/remote_001'
scp -r GPU_HOST:/path/to/dual-franka-yubi-isaac-sim/runs/remote_001 ./remote_001
```

Replace the host and paths with the GPU workstation's values, and use a new run directory for each experiment. Isaac Sim and the Panda asset remain on that workstation; the laptop needs only SSH to launch the example and copy its MP4, joint CSV, and reports. Custom policy scripts currently run **inside the simulator process on the GPU host**. This repository does not yet expose a network `reset`/`step` service or turn `--headless` into a live GUI stream. For an interactive remote view, configure an Isaac Sim host and client using [NVIDIA's livestream documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/manual_livestream_clients.html); integrating that with this runner requires separate setup.

## Object setups and placement

`random:N` draws a deterministic layout for index `N` and `--seed`. The plate center is sampled inside the central tabletop region; the cup is sampled around the plate with clearance and table-edge checks. Cup and plate colors are sampled independently from [scene_config.json](yubi_isaac_sim_env/scene_config.json). The packaged `random:0` to `random:4` presets are in [setups/](yubi_isaac_sim_env/setups/).

To run a particular layout:

```bash
"$ISAAC_PYTHON" -m yubi_isaac_sim_env.run \
  --setup random:17 --seed 42 --steps 100 --policy hold
```

For five different deterministic layouts in one process, **omit `--setup`**; the runner advances the scenario index for each episode:

```bash
"$ISAAC_PYTHON" -m yubi_isaac_sim_env.run \
  --episodes 5 --index-start 0 --seed 42 --steps 100 --policy hold
```

Passing `--setup random:17` with multiple episodes repeats that same named layout. You can also pass a custom JSON file. For example, save this as `my_setup.json`:

```json
{
  "id": "my_setup",
  "plate": {"xy_m": [0.0, 0.0], "color": "#e8ce4d"},
  "cup": {"xy_m": [0.0, 0.25], "color": "#78c8e4"}
}
```

Then run `--setup my_setup.json`. Coordinates are meters in the world frame: Z is up, X points toward the far edge in the head view, and Y points toward image right after the dataset's head-image flip. Omitted Z defaults to the table top; omitted orientation defaults to the identity quaternion `[1, 0, 0, 0]` in **wxyz** order. For tilted or elevated placements, use `position_m: [x, y, z]` and `quaternion_wxyz` for each object. The loader validates finite values and table bounds and warns about low overlapping cup and plate footprints. The objects are dynamic rigid bodies after reset, so placement commands do not pin them in space.

For Python experiments, import the package and reuse one simulator process:

```python
from yubi_isaac_sim_env import create_sim, list_setups

print(list_setups())
app, env = create_sim(gui=True, seed=42)
try:
    for index in range(5):
        observation = env.reset(seed=42, scenario_index=index)
        for step in range(100):
            action = {}  # Hold both arms at their previous targets.
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
finally:
    app.close()
```

`env.reset(setup=...)` also accepts a named setup, a JSON path, or a Python dictionary with the same fields. `create_sim` returns the Isaac `SimulationApp` and the environment; close the app when finished.

## Observations and actions

`env.observe()` and `env.step()` return state in this structure:

```text
observation
  objects.cup / objects.plate
    position_m[3], quaternion_wxyz[4]
    linear_velocity_m_s[3], angular_velocity_rad_s[3]
  robots.left / robots.right
    joint_names[], joint_positions[], joint_velocities[]
    arm_joint_limits_rad[7][2]
    link_poses.{base,left_finger,right_finger,tool}
    tool_pose: {position_m[3], quaternion_wxyz[4]}
    tool_jacobian[6][7], tool_translation_jacobian[3][7]
    gripper_joint_position_rad, gripper_mimic_joint_position_rad
    gripper_open_fraction
  scenario, policy_step, physics_time_s
```

The state interface uses SI units, arm and YUBI joint positions in radians, and quaternions in **wxyz** order. The 6×7 Jacobian contains world-frame tool translation and angular rows; the 3×7 field is its translation subset. The optional policy image path adds `observation["images"]` containing copied `uint8` RGB arrays in **H×W×3** order for `head`, `left_wrist`, and `right_wrist`, plus `observation["image_metadata"]` for each camera. The default state-only observation does not include image arrays; use `--policy-images all` when running a vision policy.

The low-level `env.step(action)` contract is joint targets:

```python
action = {
    "left": {
        "arm_joint_targets_rad": [0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741],
        "gripper_open_fraction": 1.0,
    },
    "right": {"gripper_open_fraction": 1.0},
}
```

Each arm has seven `panda_joint1`–`panda_joint7` targets and one normalized gripper command. `0` is the current CAD reference jaw pose, and `1` is the modeled open limit; these endpoints still need hardware calibration. Omitted sides or keys retain their previous targets. The environment clips arm targets to articulation joint limits. A step holds the command for six physics frames, then returns `(observation, reward, terminated, truncated, info)`.

## Connect a policy

For a state-based policy, write a Python file with `act(observation, step, episode) -> action` returning the joint-target dictionary above. The runner imports the file after Isaac Sim starts:

```bash
"$ISAAC_PYTHON" -m yubi_isaac_sim_env.run \
  --setup random:0 --episodes 2 --steps 115 \
  --policy-script yubi_isaac_sim_env/policies/push_demo.py
```

The bundled `push_demo.py` is a bounded, position-only push example for episodes **0 and 1**; it does not grasp or place the cup. For a DF, ACT, π, GR00T, or other checkpoint, write an adapter that loads the checkpoint once, builds its expected observation tensor, calls inference, and returns this environment's action representation. A vision adapter can request all three images:

```bash
"$ISAAC_PYTHON" -m yubi_isaac_sim_env.run \
  --setup random:0 --steps 100 --policy-images all \
  --policy-script path/to/your_joint_policy.py
```

If a checkpoint predicts **end effector pose trajectories**, use the trajectory policy hook instead. Its `predict(observation, step, episode)` returns a chunk of world-frame YUBI tool targets:

```python
def predict(observation, step, episode):
    return {
        "action_dt_s": 0.1,
        "waypoints": [
            {
                "left": {
                    "position_m": [x_l, y_l, z_l],
                    "quaternion_wxyz": [w_l, qx_l, qy_l, qz_l],
                    "gripper_open_fraction": open_l,
                },
                "right": {
                    "position_m": [x_r, y_r, z_r],
                    "quaternion_wxyz": [w_r, qx_r, qy_r, qz_r],
                    "gripper_open_fraction": open_r,
                },
            }
        ],
    }
```

```bash
"$ISAAC_PYTHON" -m yubi_isaac_sim_env.run \
  --setup random:0 --steps 100 --policy-images all \
  --trajectory-policy-script path/to/your_trajectory_adapter.py
```

To check the trajectory interface before loading a checkpoint, run the
included two-arm hold example:

```bash
"$ISAAC_PYTHON" -m yubi_isaac_sim_env.run \
  --setup random:0 --steps 20 --camera overview \
  --trajectory-policy-script yubi_isaac_sim_env/policies/hold_trajectory.py
```

The adapter must match the checkpoint's image order, crop, normalization, history length, language prompt, action scale, timing, and tool frame. Confirm whether the checkpoint predicts absolute world poses or deltas in camera, robot-base, or tool coordinates. `action_dt_s` must match the 0.1 s policy period; resample checkpoint output in your adapter if needed. The predictor is called again when its waypoint queue is empty; an optional `execute_steps` can consume only the first part of a longer predicted chunk before replanning. The executor uses bounded 6D differential IK to convert each target to joint commands. It does not plan collision-free paths or guarantee convergence to a waypoint in one step. The model itself is not bundled here. The head image is horizontally flipped for consistency with the source center videos. Camera intrinsics and tool calibration are estimates, so validate the transform conventions before comparing a checkpoint with real-robot data.

## Record an experiment

```bash
"$ISAAC_PYTHON" -m yubi_isaac_sim_env.run \
  --setup random:0 --seed 42 --camera left_wrist \
  --policy oscillate --steps 100 --record-run runs/oscillate_001
```

For one episode, `--record-run` writes `video.mp4`, `joints.csv`, `manifest.json`, and `report.json` to a new or empty directory. Video frame `n` corresponds to joint CSV `sample_index=n`. By default, the runner records the reset frame and then three frames per 10 Hz policy step at 30 fps. `--record-fps 10` gives one sample per policy step. With multiple episodes, outputs are named `episode_NNN.mp4` and `episode_NNN_joints.csv` under the run directory. `--record-video PATH` and `--record-joints PATH` select separate outputs. Choose a new path for every run; the runner does not overwrite existing recordings.

Only the selected `--camera` view is written to the MP4. `--policy-images all` supplies three views to the policy, independent of the recorded view. The CSV contains named joint positions and velocities for both arms, including YUBI jaw angles in radians.

## Assets, provenance, and limits

The cup is modeled at about 8 cm outer diameter and 31.7 g, and the plate at 19 cm diameter and 56.7 g. These are based on the [IKEA KALAS cup](https://www.ikea.com.hk/en/products/children%27s-small-furniture-eat---study/baby-utensils-and-high-chair/kalas-art-40378670) and [plate](https://www.ikea.com.hk/en/products/children%27s-small-furniture-eat---study/baby-utensils-and-high-chair/kalas-art-20378671) listings; individual masses are estimates from six-pack net weights. The table, robot base positions, gripper friction, camera poses, and much of the YUBI mass and inertia model are provisional. The wrist cameras use nominal ELP L180 fisheye specifications at 640×480. The scene authors GPU PhysX settings, but asset validation alone does not prove stable contact or successful manipulation on your target GPU.

The project-authored simulation code is under the repository's [MIT license](LICENSE). YUBI CAD is derived from [Toyota's YUBI hardware repository](https://github.com/Toyota/yubi-hw). The source files and derived meshes retain the upstream **CERN-OHL-W-2.0** notices and provenance under `yubi_isaac_sim_env/assets/yubi/`. Keep those notices when redistributing modified hardware source or meshes. The Franka Panda USD must be provisioned from a local Isaac Sim installation and is not redistributed by this repository. The source demonstration videos, demonstration-derived placements, and policy checkpoints are not included. See [third-party notices](THIRD_PARTY_NOTICES.md) for asset details and modification notices.
