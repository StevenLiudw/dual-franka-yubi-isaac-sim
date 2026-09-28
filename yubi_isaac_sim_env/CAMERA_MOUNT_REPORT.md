# Wrist camera and YUBI flange audit

## Camera evidence and simulation settings

The hardware guide names two **ELP-USBFHD01M-L180** wrist cameras. The
[manufacturer listing](https://www.elpcctv.com/180-degree-fisheye-lens-full-hd-1080p-usb-camera-module-usb20-ov2710-color-sensor-mjpeg-p-206.html)
specifies an OV2710 color sensor, M12 L180 fisheye lens, approximately **195°
horizontal and 123° vertical** field of view, a 38 × 38 mm board, USB 2.0 UVC,
and MJPEG modes including 1920 × 1080 at 30 fps and 640 × 480 at up to 100 fps.
[OMNIVISION](https://www.ovt.com/products/ov2710/) gives the OV2710 native
1920 × 1080 active pixels, 3 µm pixel pitch, and rolling shutter. The five
provided wrist video sets were probed at **640 × 480, 30 fps**. That is the
simulator render size and the default video and joint recording cadence.

The simulator uses Isaac Sim's
[OpenCV fisheye camera model](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_camera.html).
The nominal 640 × 480 intrinsics are `fx=188.0477`, `fy=223.5933`,
`cx=320`, `cy=240` pixels with `k1…k4=0`. These values fit the manufacturer's
two nominal angles under an ideal equidistant model:
`FOV = 2 × image_half_extent / focal_length` (radians). They are intentionally
marked **estimated** in [wrist_camera_model.json](wrist_camera_model.json).
The native sensor's displayed active area is 5.76 × 3.24 mm by pixel pitch;
the USD focal/aperture fields are paraxial metadata. The applied fisheye schema
controls the actual ray model. The 640 × 480 crop mode and real per-unit
principal point/distortion are undocumented, so an exact image match is not
yet possible. Exposure, rolling shutter timing, and MJPEG compression are also
outside the current simulator model.

The camera position and axis are derived from the official motorized YUBI
assembly STEP (source pinned in `assets/yubi/source_manifest.json`). Both arms
reference the same robot asset and use that same local camera transform.
This establishes **identical nominal local camera mounting**, but no physical
hand-eye measurement appears in the guide.

## Replay image orientation

All five provided replay sets show the hand hardware entering from the **bottom**
of both wrist images. The original simulator camera used only `Rx(190°)` from
the motorized assembly's lens-forward axis. That placed the simulated jaws at
the **top** because the USD camera's image-up direction was opposite the replay
convention. The STEP-derived lens center and forward ray do not establish
which sensor edge is image-up, and the runner applies no wrist image rotation.

The camera now uses `Rx(190°) · Rz(180°)` in the YUBI frame. The second rotation
is around the camera's local optical axis: it moves the jaws to the bottom and
reverses left/right image placement without changing the lens center, viewing
ray, fisheye intrinsics, robot mount, or physics. One-step headless GPU renders
of **both** wrist cameras showed the jaws at the bottom at 640 × 480. The
validator checks the unchanged CAD forward ray and bottom-of-image tool sign.
This 180° roll is a replay-alignment choice, not a measured physical sensor
mount orientation.

The remaining top-down appearance has a separate cause: the simulated Franka
tools and CAD camera axes point toward the table at the home pose. The replay
footage comes from human-held YUBI glove hardware at changing wrist poses and
also includes the surrounding lab. The simulator instead uses the motorized
YUBI gripper. Matching the replay's pitch, foreground size, and background
requires measured wrist trajectories, camera extrinsics, and a closer scene;
rotating the image cannot establish those from the videos alone.

## Flange evidence and alignment

The guide links the v1.1.2 direct Franka `FR_FLANGE.stl`; this package uses
that part. The gripper assembly meshes come from Toyota's
[v2.0.0 motorized STEP](https://github.com/Toyota/yubi-hw/tree/v2.0.0),
as documented in the source manifest. The adapter has a 29 mm diameter,
2.5 mm long central pilot, and holes matching the Panda flange's 50 mm bolt
circle. The official [Franka mounting point
description](https://frankarobotics.github.io/docs/doc/franka_ros2_jazzy/franka_description_extensions/doc/mounting_points.html)
identifies `panda_link8` as the final flange frame; the
[Franka product manual](https://download.franka.de/documents/100010_Product%20Manual%20Franka%20Emika%20Robot_10.21_EN.pdf)
gives the flange recess/bolt geometry.

The corrected `panda_link8` → YUBI base transform is **(+18, 0, −2.5) mm,
Rz(+90°)** in the local link8 frame. The −2.5 mm shift seats the pilot in the
recess and places the adapter's flat mating face at the flange face. The
+90° yaw inverts the adapter's −90° placement in the YUBI CAD frame. Both
arms have the **same** local joint transform. Their robot bases point in
different world directions, so their world-facing orientations are expected
to differ.

The offline USD validator found a mount-frame coincidence error of
approximately **0.05 µm** on each arm, the same local flange transform on
both arms, and both tools pointing down toward the table. This validates the
authored scene geometry, not contact forces under GPU PhysX. The guide and
CAD do not identify a unique installed bolt-hole orientation; therefore the
physical assembly's exact yaw cannot be asserted from these sources alone.

## Measurements needed for a close hardware match

Calibrate each physical wrist camera at its dataset resolution with a
checkerboard/ChArUco pattern and OpenCV's fisheye model, then replace the
respective `K` and `k1…k4` values. A fiducial or hand-eye calibration at several
arm poses can determine each camera's pose relative to `panda_link8` and the
actual adapter yaw. Finally, run the included USD and GPU simulation checks
on the target NVIDIA workstation to inspect rendered edge coverage and
clearance under motion.
