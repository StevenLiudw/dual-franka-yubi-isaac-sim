#!/usr/bin/env bash
set -euo pipefail

# Rebuild CAD-derived USD without starting Isaac Sim or requiring a GPU.
# Use the same Python installation that contains Isaac Sim 5.1 and trimesh.
isaac_python="${ISAAC_PYTHON:-python}"
if ! command -v "$isaac_python" >/dev/null 2>&1; then
    echo "Isaac Sim Python not found: $isaac_python (set ISAAC_PYTHON)" >&2
    exit 1
fi
package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
site_dir="$($isaac_python -c 'import site; print(site.getsitepackages()[0])')"
cache_dir="$site_dir/isaacsim/extscache"
shopt -s nullglob
usd_libraries=("$cache_dir"/omni.usd.libs-*)
physx_schema=("$cache_dir"/omni.usd.schema.physx-*)
lens_schema=("$cache_dir"/omni.usd.schema.omni_lens_distortion-*)
if (( ${#usd_libraries[@]} != 1 || ${#physx_schema[@]} != 1 || ${#lens_schema[@]} != 1 )); then
    echo "Could not uniquely locate Isaac Sim USD, PhysX, and lens schemas in $cache_dir" >&2
    exit 1
fi
env_lib="$(dirname "$(dirname "$site_dir")")"
export PYTHONPATH="${physx_schema[0]}:${usd_libraries[0]}:$(dirname "$package_dir"):${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${physx_schema[0]}/bin:${usd_libraries[0]}/bin:$env_lib:${LD_LIBRARY_PATH:-}"
export PXR_PLUGINPATH_NAME="${lens_schema[0]}/usd_plugins:${physx_schema[0]}/plugins/PhysxSchema/resources:${physx_schema[0]}/plugins/OmniUsdPhysicsDeformableSchema/resources:${physx_schema[0]}/plugins/PhysxSchemaAddition/resources:${PXR_PLUGINPATH_NAME:-}"
exec "$isaac_python" "$package_dir/build_assets.py"
