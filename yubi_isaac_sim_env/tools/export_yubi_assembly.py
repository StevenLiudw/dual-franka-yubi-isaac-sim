"""Regenerate the official v2.0.0 motorized YUBI mesh groups with OCP.

Run from this package's source tree. The Isaac runtime uses the already
generated meshes; OCP is needed only when rebuilding from STEP.
"""

import json, pathlib, struct, numpy as np
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.collections import Sequence_TDF_Label
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.TopoDS import TopoDS_Compound
from OCP.gp import gp_Trsf, gp_Pnt, gp_Dir

package=pathlib.Path(__file__).resolve().parent.parent
repo=package/'assets/yubi/source_v2.0.0'
out=package/'assets/yubi/meshes'
out.mkdir(parents=True,exist_ok=True)
assy=repo/'STEP/gripper/YUBI Gripper Assy_Dynamixel_ver2.STEP'
origin_mm=[854.071,837.505,1225.573]  # UR5e flange robot-facing center in assembly CAD
jaw_right_ids={3,8,10,12,14,22,42,43,46,47,63}
jaw_left_ids={9,11,13,15,23,40,41,48,49,64}
excluded_ids=set(range(74,86)) # UR5e flange plus its mounting hardware replaced by Franka pair

def load_step(path):
 doc=TDocStd_Document(TCollection_ExtendedString('yubi'))
 reader=STEPCAFControl_Reader(); reader.SetNameMode(True); reader.SetColorMode(True)
 status=reader.ReadFile(str(path))
 if 'RetDone' not in str(status) or not reader.Transfer(doc): raise RuntimeError(f'STEP import failed: {path} {status}')
 st=XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
 roots=Sequence_TDF_Label(); st.GetFreeShapes(roots)
 return doc,st,roots

def label_name(l):
 a=TDataStd_Name()
 return a.Get().ToExtString() if l.FindAttribute(TDataStd_Name.GetID_s(),a) else ''

def bounds(shape):
 box=Bnd_Box(); BRepBndLib.Add_s(shape,box,False)
 return [float(getattr(box,'Get'+c)()) for c in ('XMin','YMin','ZMin','XMax','YMax','ZMax')]

def shifted_scaled(shape,offset_mm):
 # Meters with the origin translated to the robot flange face.
 t=gp_Trsf(); t.SetValues(.001,0,0,-offset_mm[0]*.001,0,.001,0,-offset_mm[1]*.001,0,0,.001,-offset_mm[2]*.001)
 return BRepBuilderAPI_Transform(shape,t,True,True).Shape()

def make_compound(shapes):
 b=BRep_Builder(); c=TopoDS_Compound(); b.MakeCompound(c)
 for s in shapes: b.Add(c,s)
 return c

def save_stl(shapes,path):
 c=make_compound(shapes)
 BRepMesh_IncrementalMesh(c,.00025,False,.25,True)
 w=StlAPI_Writer(); w.ASCIIMode=False
 if not w.Write(c,str(path)): raise RuntimeError(f'STL export failed: {path}')
 raw=path.read_bytes(); n=struct.unpack_from('<I',raw,80)[0]
 if len(raw)!=84+50*n: raise RuntimeError(f'Expected binary STL: {path}')
 tri=np.frombuffer(raw,dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')]),count=n,offset=84)['vertices'].reshape(-1,3)
 lo=tri.min(axis=0); hi=tri.max(axis=0)
 return [float(x) for x in (*lo,*hi)]

doc,st,roots=load_step(assy)
if roots.Length()!=1: raise RuntimeError(f'Expected 1 assembly root, got {roots.Length()}')
root=roots.Value(1)
children=Sequence_TDF_Label(); XCAFDoc_ShapeTool.GetComponents_s(root,children,False)
if children.Length()!=85: raise RuntimeError(f'Expected 85 components, got {children.Length()}')
# STEP assembly servo-driven spur gear axes are parallel to assembly Y.
def point_mount(p):
 return [(p.X()-origin_mm[0])*.001,(p.Y()-origin_mm[1])*.001,(p.Z()-origin_mm[2])*.001]
for side,component_id in [('right',8),('left',9)]:
 tr=XCAFDoc_ShapeTool.GetLocation_s(children.Value(component_id)).Transformation()
 manifest_pivot=point_mount(gp_Pnt(0,0,0).Transformed(tr))
 # Some part-local origins are on the gear's axis, per STEP cylindrical surface inspection.
 globals().setdefault('pivots',{})[side]=manifest_pivot
camera_tr=XCAFDoc_ShapeTool.GetLocation_s(children.Value(21)).Transformation()
camera_origin=point_mount(gp_Pnt(0,0,0).Transformed(camera_tr))
camera_front=point_mount(gp_Pnt(0,0,18.6).Transformed(camera_tr))
forward=gp_Dir(0,0,1).Transformed(camera_tr)
camera_axis=[forward.X(),forward.Y(),forward.Z()]
manifest={'source':{'repo':'https://github.com/Toyota/yubi-hw','tag':'v2.0.0','commit':'dd8bd13d2fd8e5003057243576f88be333d95fc5','assembly_step':str(assy.relative_to(repo))},'cad_units':'mm','mesh_units':'m','assembly_origin_cad_mm':origin_mm,'mount_frame_note':'Provisional frame at robot-facing center of assembly UR5e flange; use replacement Franka adapter pair pending hardware flange-frame verification.','jaw_hinge_axes_mount':[0,1,0],'jaw_hinge_pivots_mount_m':{},'components':[],'files':{}}
manifest['jaw_hinge_pivots_mount_m']=pivots
manifest['camera']={'module_origin_mount_m':camera_origin,'approx_front_lens_center_mount_m':camera_front,'optical_forward_mount':camera_axis,'pose_source':'STEP assembly NAUO21 and camera lens cylinder; lens front plane approximated as module local z=18.6 mm. Intrinsics/distortion not calibrated.'}
groups={'fixed':[],'jaw_right':[],'jaw_left':[]}
for n in range(1,children.Length()+1):
 label=children.Value(n)
 comp=XCAFDoc_ShapeTool.GetShape_s(label)
 refer=TDF_Label(); has_ref=XCAFDoc_ShapeTool.GetReferredShape_s(label,refer)
 partname=label_name(refer) if has_ref else label_name(label)
 grp='excluded' if n in excluded_ids else ('jaw_right' if n in jaw_right_ids else ('jaw_left' if n in jaw_left_ids else 'fixed'))
 rec={'nauo':n,'name':partname,'group':grp,'bounds_cad_mm':bounds(comp)}
 manifest['components'].append(rec)
 if grp!='excluded': groups[grp].append(shifted_scaled(comp,origin_mm))
for grp,shapes in groups.items():
 f=out/f'{grp}.stl'; bb=save_stl(shapes,f)
 manifest['files'][grp]={'path':f.name,'component_count':len(shapes),'bounds_mount_m':bb}
manifest['adapter_note']='Runtime uses v1.1.2 FR_FLANGE direct adapter; see franka_adapter_v1_manifest.json.'
(package/'assets/yubi/assembly_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'out':str(out),'files':manifest['files'],'jaw_pivots':manifest['jaw_hinge_pivots_mount_m']},indent=2))
