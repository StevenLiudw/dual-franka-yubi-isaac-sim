# YUBI Isaac Sim package notes

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

Each YUBI gripper has a driven revolute jaw and inverse mimic jaw. The stock
Panda hand and sliding fingers are disabled in this derived robot asset. The
adapter's central pilot enters the Franka flange recess: the local transform
from `panda_link8` to the YUBI base is (+18, 0, −2.5) mm and +90° yaw. Both
arms use the same local transform. The source hardware does not specify a
unique installed bolt-hole yaw, so the physical orientation still needs
measurement. See [CAMERA_MOUNT_REPORT.md](CAMERA_MOUNT_REPORT.md).

`gripper_open_fraction=0` is the CAD reference pose, with an estimated
18–23 mm fingertip gap, **not** a verified fully closed pose. A value of 1
commands the conservative modeled open limit of 0.60 rad per jaw. The YUBI
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
