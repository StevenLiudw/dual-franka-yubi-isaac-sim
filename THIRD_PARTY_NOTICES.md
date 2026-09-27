# Third-party materials

The repository's MIT license covers the original simulator code and project
documentation. It does not relicense the CAD, derived YUBI geometry, NVIDIA
robot assets, Isaac Sim, or any other third-party material.

## Toyota YUBI hardware

The source YUBI hardware is Copyright 2026 Toyota Motor Corporation and
licensed under CERN Open Hardware Licence Version 2, Weakly Reciprocal
(CERN-OHL-W-2.0). The original [v2.0.0 source](https://github.com/Toyota/yubi-hw/tree/v2.0.0)
and [v1.1.2 source](https://github.com/Toyota/yubi-hw/tree/v1.1.2) are preserved
in `yubi_isaac_sim_env/assets/yubi/source_v2.0.0/` and
`yubi_isaac_sim_env/assets/yubi/source_v1.1.2/` with their respective `LICENSE`
files. `yubi_isaac_sim_env/assets/yubi/source_manifest.json` records the source URLs, revisions,
and SHA-256 hashes.

**Modification notice, 2026-09-27:** For this simulator, the v2.0.0 motorized
assembly STEP was separated into fixed, left-jaw, and right-jaw meshes; the
v1.1.2 `FR_FLANGE.stl` was converted to the simulator's metric mounting frame.
The resulting STL files in `yubi_isaac_sim_env/assets/yubi/meshes/` and YUBI geometry authored in
`yubi_isaac_sim_env/assets/franka_yubi_panda.usdc` are modified material derived from the Toyota
source. The geometry was transformed for Isaac Sim joint articulation and
flange alignment. `yubi_isaac_sim_env/tools/export_yubi_assembly.py` and the manifests in
`yubi_isaac_sim_env/assets/yubi/` document the transformations. These Toyota-derived parts remain
subject to CERN-OHL-W-2.0; their complete source is included above.

## NVIDIA Isaac Sim Franka Panda asset

This public repository does **not** distribute the stock `franka_panda` USD
directory or Isaac Sim binaries. `yubi_isaac_sim_env/assets/franka_yubi_panda.usdc` references a
stock Franka Panda asset at `yubi_isaac_sim_env/assets/franka_panda/franka_panda.usd`; it does not
embed the stock robot meshes. Each user must obtain Isaac Sim assets under
their applicable NVIDIA terms and provision the Panda directory locally with
`scripts/provision_franka.py`. The destination is excluded by `.gitignore`.
See the [NVIDIA Isaac Sim Additional Software and Materials License](https://docs.nvidia.com/NVIDIA-IsaacSim-Additional-Software-and-Materials-License.pdf).
