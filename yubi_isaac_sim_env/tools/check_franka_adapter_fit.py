"""Check v1 direct Franka adapter against the v2 motorized gripper shell.

Optional CAD check; requires VTK. The adapter transform is the same one used
for ``assets/yubi/meshes/franka_adapter_v1.stl``.
"""

from pathlib import Path

from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersModeling import vtkCollisionDetectionFilter
from vtkmodules.vtkIOGeometry import vtkSTLReader


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "assets/yubi/source_v1.1.2/STL/flange/FR_FLANGE.stl"
FIXED = ROOT / "assets/yubi/meshes/fixed.stl"


def read(path: Path):
    reader = vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader


def main() -> None:
    source = read(SOURCE)
    fixed = read(FIXED)
    matrix = vtkMatrix4x4()
    matrix.Identity()
    matrix.SetElement(0, 0, 0)
    matrix.SetElement(0, 1, 0.001)
    matrix.SetElement(1, 0, -0.001)
    matrix.SetElement(1, 1, 0)
    matrix.SetElement(2, 2, 0.001)
    matrix.SetElement(1, 3, 0.018)
    matrix.SetElement(2, 3, 0.0025)
    transform = vtkTransform()
    transform.SetMatrix(matrix)
    placed = vtkTransformPolyDataFilter()
    placed.SetTransform(transform)
    placed.SetInputConnection(source.GetOutputPort())
    placed.Update()
    identity = vtkMatrix4x4()
    identity.Identity()
    collision = vtkCollisionDetectionFilter()
    collision.SetInputData(0, placed.GetOutput())
    collision.SetInputData(1, fixed.GetOutput())
    collision.SetMatrix(0, identity)
    collision.SetMatrix(1, identity)
    collision.SetCollisionModeToFirstContact()
    collision.Update()
    contacts = collision.GetNumberOfContacts()
    print(f"Exterior adapter/fixed triangle contacts: {contacts}")
    if contacts:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
