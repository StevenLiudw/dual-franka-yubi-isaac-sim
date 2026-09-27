#!/usr/bin/env python3
"""Copy the stock Panda USD from a user's licensed Isaac Sim asset directory.

The public repository intentionally omits NVIDIA's asset. This script never
downloads it, and only writes to this checkout's ignored asset directory.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "yubi_isaac_sim_env" / "assets"
DESTINATION = ASSETS / "franka_panda"
REQUIRED_FILES = (
    "franka_panda.usd",
    "configuration/franka_panda_base.usd",
    "configuration/franka_panda_robot.usd",
    "configuration/franka_panda_physics.usd",
    "configuration/franka_panda_sensor.usd",
)


def _source_dir(argument: str) -> Path:
    supplied = Path(argument).expanduser()
    if supplied.is_symlink():
        raise ValueError(f"Source path is a symbolic link: {supplied}")
    if not supplied.is_dir():
        raise ValueError(f"Source directory does not exist: {supplied}")
    if not (supplied / "franka_panda.usd").is_file():
        supplied = supplied / "franka_panda"
    if supplied.is_symlink() or not supplied.is_dir():
        raise ValueError("Pass the franka_panda directory or its parent")
    return supplied.resolve(strict=True)


def _inventory(root: Path) -> dict[str, str]:
    """Reject links/special files and hash all regular files in the tree."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Not a regular directory: {root}")
    inventory: dict[str, str] = {}
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            entry = Path(current) / name
            if not stat.S_ISDIR(entry.lstat().st_mode):
                raise ValueError(f"Symbolic link or special directory is not allowed: {entry}")
        for name in files:
            entry = Path(current) / name
            if not stat.S_ISREG(entry.lstat().st_mode):
                raise ValueError(f"Symbolic link or special file is not allowed: {entry}")
            digest = hashlib.sha256()
            with entry.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            inventory[entry.relative_to(root).as_posix()] = digest.hexdigest()
    return inventory


def _validate_dependencies(root: Path, inventory: dict[str, str]) -> str:
    # These five files are the complete known USD dependency chain of the
    # Isaac Sim 5.1 Panda asset. Keep the directory together, not one USD file.
    missing = sorted(set(REQUIRED_FILES) - set(inventory))
    if missing:
        raise ValueError(f"Incomplete Franka Panda asset; missing: {', '.join(missing)}")
    for relative in REQUIRED_FILES:
        if (root / relative).stat().st_size == 0:
            raise ValueError(f"Empty USD asset: {relative}")

    # When run through pxr_python.sh, also inspect every authored USD
    # dependency instead of relying only on the known asset layout.
    try:
        from pxr import Sdf
    except ImportError:
        return "known five-file dependency chain"
    for relative in sorted(inventory):
        if Path(relative).suffix.lower() not in {".usd", ".usda", ".usdc"}:
            continue
        usd_file = root / relative
        layer = Sdf.Layer.FindOrOpen(str(usd_file))
        if layer is None:
            raise ValueError(f"Cannot read USD layer: {usd_file}")
        for reference in set(layer.GetExternalReferences()) | set(layer.subLayerPaths):
            if not reference:
                continue
            if "://" in reference or Path(reference).is_absolute():
                raise ValueError(f"External or absolute USD dependency: {reference}")
            dependency = (usd_file.parent / reference).resolve(strict=True)
            if not dependency.is_relative_to(root) or not dependency.is_file():
                raise ValueError(f"USD dependency escapes or is missing: {reference}")
    return "all USD references"


def provision(source: Path, *, dry_run: bool) -> None:
    if ASSETS.is_symlink() or not ASSETS.is_dir():
        raise ValueError(f"Expected a regular assets directory: {ASSETS}")
    if source == DESTINATION.resolve():
        raise ValueError("Source and destination are the same directory")
    source_inventory = _inventory(source)
    check = _validate_dependencies(source, source_inventory)
    print(f"Validated {len(source_inventory)} files and {check} in {source}")
    if dry_run:
        return

    if DESTINATION.is_symlink():
        raise ValueError(f"Destination is a symbolic link: {DESTINATION}")
    if DESTINATION.exists():
        if _inventory(DESTINATION) != source_inventory:
            raise ValueError(
                f"Destination already exists with different contents: {DESTINATION}. "
                "Move it aside yourself before provisioning another version."
            )
        print(f"Already provisioned: {DESTINATION}")
        return

    temporary = Path(tempfile.mkdtemp(prefix=".franka_panda.", dir=ASSETS))
    try:
        # symlinks=True ensures a newly introduced link is copied as a link,
        # never followed to an unexpected file; the final scan rejects it.
        shutil.copytree(source, temporary, dirs_exist_ok=True, symlinks=True)
        copied_inventory = _inventory(temporary)
        _validate_dependencies(temporary, copied_inventory)
        if copied_inventory != source_inventory:
            raise ValueError("Source changed while copying; no asset was installed")
        temporary.rename(DESTINATION)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Provisioned {len(source_inventory)} files in {DESTINATION}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        metavar="DIR",
        help="licensed Isaac Sim franka_panda asset directory, or its parent",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate the source without copying"
    )
    args = parser.parse_args()
    try:
        provision(_source_dir(args.source), dry_run=args.dry_run)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"provision_franka: {exc}\n")


if __name__ == "__main__":
    main()
