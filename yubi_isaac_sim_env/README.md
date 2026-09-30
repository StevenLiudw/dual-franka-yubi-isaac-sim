# YUBI Isaac Sim package notes

The full-pose paired replay builds on `policies/umi_pose_replay.py` and
`dual_franka_yubi_replay.usda`. It converts source hand-root axes and TCP
offset, preserves both hands in one rigid frame, and uses the estimated
`head_camera_replay.json` framing. The tuned two-episode trial layers a
fixed anchor, shared jaw bias, left-hand spacing and height, and delayed
closure onto that base policy. See the reproducible command in the
[repository README](../README.md). The older replay scripts below remain
legacy translation-only comparisons.

The [repository README](../README.md) contains installation, asset provisioning,
random and custom object placement, GUI use, policy interfaces, and recording
commands. This package contains the scene, robot asset layer, simulator API,
camera configuration, and policy adapters. It does not import or modify the
older Franka gripper simulator.

## Physical sources and assumptions

The articulated YUBI geometry comes from Toyota's
[v2.0.0 motorized hardware STEP](https://github.com/Toyota/yubi-hw/tree/v2.0.0).
The direct Franka adapter comes from the v1.1.2 `FR_FLANGE.stl`. Their source,
licenses, hashes, and conversion manifests are in [assets/yubi](assets/yubi).
The static Panda robot USD must be provisioned from the user's licensed Isaac
Sim assets; the public repository does not redistribute it. The derived
`franka_yubi_panda.usdc` only adds the YUBI geometry and joints to that
reference.

Each YUBI gripper has a mirrored revolute-jaw pair driven from one normalized
opening command. The physical device uses one XM430 motor and gears; Isaac Sim
drives `q` and `-q` explicitly because its mimic constraint locked this
opposite-limit pair at the shared zero stop. The stock
Panda hand and sliding fingers are disabled in this derived robot asset. The
adapter's central pilot enters the Franka flange recess: the local transform
from `panda_link8` to the YUBI base is (+18, 0, −2.5) mm and +90° yaw. Both
arms use the same local transform. The source hardware does not specify a
unique installed bolt-hole yaw, so the physical orientation still needs
measurement. See [CAMERA_MOUNT_REPORT.md](CAMERA_MOUNT_REPORT.md).

`gripper_open_fraction=0` commands -0.10 rad per jaw, with less than 0.5 mm
estimated distal clearance. A value of 1 commands +0.70 rad. The original
CAD reference angle of 0 rad is partly open. The revised range passed a
33-angle distal mesh clearance sweep; full-assembly and real servo limits
remain unmeasured. `gripper_aperture_calibration.json` maps already-zeroed
operator angles to the same CAD distal aperture, independently of recording
minima. The YUBI
mass, inertia, friction, drive gains, and pinch-point tool frame are
provisional; current values are in [config.json](config.json).

The cup and plate are modeled from the IKEA KALAS product dimensions and
six-pack net weights. Plate center and cup-around-plate randomization are
configured in [scene_config.json](scene_config.json). Table size, cup
profile, and several contact properties are estimates. Reset-time placements
create dynamic rigid bodies that can subsequently be moved by robot contact.

## Cameras

Both wrists use the ELP-USBFHD01M-L180/OV2710 camera model. The nominal
195° horizontal and 123° vertical fisheye fields of view come from the
[manufacturer](https://www.elpcctv.com/180-degree-fisheye-lens-full-hd-1080p-usb-camera-module-usb20-ov2710-color-sensor-mjpeg-p-206.html).
The simulator renders 640×480 RGB and samples recordings at 30 Hz by
default. [wrist_camera_model.json](wrist_camera_model.json) contains fitted
ideal fisheye intrinsics; it is not a calibration of either physical camera.
The CAD constrains the lens center and forward axis, while a 180° roll about
that axis sets image-up so the gripper enters from the bottom as it does in
the five replay video sets. This roll is an image-convention estimate, not a
measured mount angle. The replay uses human-held glove hardware; the motorized
Franka-mounted YUBI and its downward home pose still produce a different
foreground scale and table/background view. See
[CAMERA_MOUNT_REPORT.md](CAMERA_MOUNT_REPORT.md).
The head camera in [head_camera_calibration.json](head_camera_calibration.json)
is also an estimate derived from tabletop video geometry.

The default `env.observe()` returns state. With `--policy-images all`, the
runner passes head and both wrist `uint8` H×W×3 RGB arrays and camera
metadata to the policy. The selected `--camera` is independently shown in
the GUI and, when requested, written to MP4. The head image is horizontally
flipped to match the dataset convention.

## Control and validation

The simulator accepts seven joint targets in radians and normalized YUBI
opening per arm. The trajectory adapter converts absolute world-frame YUBI
tool poses into bounded joint increments with a 6×7 Jacobian. It is a local
IK controller, without a collision planner or measured hand-eye calibration.
Trajectory execution now checks padded Panda/YUBI link samples and their
Jacobian-predicted sweep before each 10 Hz command. It scales a command that
would reduce the estimated inter-arm clearance below 20 mm; this is an online
inter-arm guard, not mesh-level path planning or a real-robot safety system.
The episode report includes per-step guard scales and the smallest observed
clearance. It does not cover robot/environment contacts or a miscalibrated
link model.

The official replay uses a fixed aperture lookup rather than subtracting the
smallest angle in each recording or stretching the observed range. The
operator encoder node already applies its device zero. Motorized hardware
calibration and actual contact dynamics remain unmeasured. A jaw target can differ from its
measured angle if contact blocks the modeled torque-limited drive.
The derived Panda layer also overrides the stock acceleration drive's zero
damping with the high-bandwidth, 0.707-damping-ratio values in `config.json`.
This controls simulated joint ringing; it does not filter, retime, or slow the
10 Hz policy waypoints.
The README at repository root gives the exact `act` and `predict` policy
contracts.

After provisioning the Franka asset, validate the packaged USD without a
GPU from the repository root:

```bash
bash yubi_isaac_sim_env/pxr_python.sh -m yubi_isaac_sim_env.validate_assets
```

One-step headless GPU checks on an RTX 5080 rendered both wrist views with the
YUBI at the bottom. Full-episode contact behavior, the head view, and visual
agreement with the physical hardware still need validation. The validator
checks source hashes, relative USD dependencies, camera schemas, identical
YUBI mounts, and GPU physics authoring.
