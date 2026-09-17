#!/usr/bin/env python3
"""Convert bilateral Stage-4 MANO npz sequences to merged Inspire geometry."""
from __future__ import annotations
import argparse, json, sys, tempfile, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import InspireUrdfModel, _fk_surface, _from_object_frame, _to_object_frame
from src.task.ObjectInteractionCm.research.hand_region_sampling.run import _build_inspire_pool, _sample_uniform_surface
from dex_retargeting.retargeting_config import RetargetingConfig

TIP_IDS = {"right": np.asarray([744,320,443,554,671]), "left": np.asarray([744,320,444,554,671])}
DEX = Path("/home/wbcd/workspace/dex/retarget/third_party/dex-retargeting")


def canonical_cloud_to_visual_local(
    model: InspireUrdfModel,
    points: np.ndarray,
    normals: np.ndarray,
    visual_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Undo the canonical zero-q visual poses before applying per-frame FK.

    ``_build_inspire_pool`` expresses every visual triangle in the complete
    zero-q hand frame.  ``_fk_surface`` instead requires mesh-local samples.
    Convert the sampled cloud back to each visual's local frame so that the
    current-frame link transform is applied exactly once.
    """

    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    visual_ids = np.asarray(visual_ids, dtype=np.int64)
    if points.ndim != 2 or points.shape[1:] != (3,) or normals.shape != points.shape:
        raise ValueError(f"invalid sampled cloud shape: points={points.shape}, normals={normals.shape}")
    if visual_ids.shape != (len(points),):
        raise ValueError(f"invalid visual id shape: {visual_ids.shape}")
    if np.any(visual_ids < 0) or np.any(visual_ids >= len(model.visuals)):
        raise ValueError("sampled cloud contains an out-of-range visual id")

    zero_q = np.zeros(18, dtype=np.float64)
    links = model.link_transforms(model.qpos_to_urdf_order(zero_q))
    local_points = np.empty_like(points)
    local_normals = np.empty_like(normals)
    for visual_id in np.unique(visual_ids):
        mask = visual_ids == visual_id
        visual = model.visuals[int(visual_id)]
        transform = links[visual.link] @ visual.local_transform
        rotation = transform[:3, :3].astype(np.float32)
        translation = transform[:3, 3].astype(np.float32)
        local_points[mask] = (points[mask] - translation) @ rotation
        local_normals[mask] = normals[mask] @ rotation
    local_normals /= np.clip(np.linalg.norm(local_normals, axis=1, keepdims=True), 1e-8, None)
    return local_points, local_normals


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-root",type=Path,required=True); ap.add_argument("--output-root",type=Path,required=True); ap.add_argument("--source",required=True,choices=["grab","arctic"]); ap.add_argument("--fps",type=float,default=None); ap.add_argument("--limit",type=int,default=None); ap.add_argument("--sequence",action="append",default=[]); ap.add_argument("--modification-version",default="V1.4.6")
    a=ap.parse_args(); inp=a.input_root.resolve(); out=a.output_root.resolve(); out.mkdir(parents=True,exist_ok=True)
    RetargetingConfig.set_default_urdf_dir(DEX/"assets/robots/hands")
    models={}; samplings={}; retargeters={}
    source_urdf=ROOT.parent/"dexplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf"
    tree=ET.parse(str(source_urdf)); active=[]
    for joint in tree.getroot().findall("joint"):
        if joint.get("type")!="fixed": active.append(joint.get("name"))
        if joint.get("type")=="continuous":
            joint.set("type","revolute"); lim=joint.find("limit") or ET.SubElement(joint,"limit")
            lim.set("lower","-1000000"); lim.set("upper","1000000"); lim.set("effort",lim.get("effort","1000")); lim.set("velocity",lim.get("velocity","3.14"))
    for mesh in tree.getroot().iter("mesh"):
        fn=mesh.get("filename")
        if fn and not Path(fn).is_absolute(): mesh.set("filename",str((source_urdf.parent/fn).resolve()))
    td=Path(tempfile.mkdtemp(prefix="oicm-inspire-")); patched=td/source_urdf.name; tree.write(str(patched))
    inv_native=np.argsort(np.asarray([0,1,2,3,4,5,10,11,12,13,16,17,14,15,6,7,8,9]))
    for side in ("left","right"):
        # Dexplore's production Inspire URDF contains the 18-DOF hand model
        # used by the cache FK code (the upstream retargeting URDF has only
        # the 12 revolute joints).  Keep the official objective/configuration
        # while overriding its URDF path to this production asset.
        urdf=ROOT.parent/"dexplore/dexplore/data/assets/inspire_hand_new"/f"inspire_hand_{side}.urdf"
        try:
            models[side]=InspireUrdfModel(urdf)
        except ValueError:
            # The checked-in Dexplore left visual URDF is a 12-DOF legacy
            # variant.  Its retargeting objective remains valid, while the
            # production surface FK model is the 18-DOF right asset.
            if side != "left": raise
            urdf=ROOT.parent/"dexplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf"
            models[side]=InspireUrdfModel(urdf)
        pool,_=_build_inspire_pool(urdf,0.20); cloud=_sample_uniform_surface(pool,2024,1538)
        local_points,local_normals=canonical_cloud_to_visual_local(models[side],cloud.points,cloud.normals,cloud.source_visual_ids)
        samplings[side]=(local_points,local_normals,cloud.source_visual_ids)
        retargeters[side]=RetargetingConfig.load_from_file(DEX/"dex_retargeting/configs/offline"/f"inspire_hand_{side}.yml", override={"urdf_path":str(patched),"add_dummy_free_joint":False,"target_joint_names":active,"ignore_mimic_joint":True}).build()
    if a.sequence:
        dirs=[]
        for value in a.sequence:
            seq=(inp/value).resolve()
            if inp not in seq.parents or not (seq/"shared.npz").is_file():
                raise ValueError(f"invalid sequence below {inp}: {value}")
            dirs.append(seq)
    else:
        dirs=sorted(p.parent for p in inp.glob("**/shared.npz"))
    if a.limit: dirs=dirs[:a.limit]
    done=0
    for seq in dirs:
        rel=seq.relative_to(inp); target=out/rel
        if (target/"geometry/manifest.json").is_file(): continue
        try:
            with np.load(seq/"shared.npz") as z: shared={k:np.asarray(z[k]) for k in z.files}
            obj=shared.get("obj_points_world",shared.get("obj_points_pool_world")); norms=shared.get("obj_normals_world",shared.get("obj_normals_pool_world")); pose=shared.get("obj_pose_world",shared.get("obj_root_pose_world")); raw=shared.get("raw_frame_id",shared.get("source_frame_id"))
            if obj is None or norms is None or raw is None: raise ValueError("missing shared object fields")
            if pose is None:
                pose=np.broadcast_to(np.eye(4,dtype=np.float32),(len(obj),4,4)).copy()
            T=len(obj); fps=a.fps or (30.0 if a.source=="arctic" else 30.0); points=[]; normals=[]; masks=[]
            for side in ("left","right"):
                with np.load(seq/f"{side}.npz") as z: d={k:np.asarray(z[k]) for k in z.files}
                if "hand_mesh_vertices_world" in d and d["hand_mesh_vertices_world"].ndim == 3 and d["hand_mesh_vertices_world"].shape[1] >= 778:
                    verts=d["hand_mesh_vertices_world"]
                    local=(verts - pose[:,None,:3,3]) @ pose[:,:3,:3]
                    tips=local[:,TIP_IDS[side]]
                else:
                    # ARCTIC Stage4 intentionally stores sampled MANO
                    # surfaces only.  Select one stable distal sample per
                    # finger from the canonical finger labels.
                    hp=d["hand_points_world"]; fid=np.asarray(d["hand_finger_id"]); can=np.asarray(d["hand_cano_points"])
                    center=can.mean(axis=0); tip_idx=[]
                    for finger in range(1,6):
                        cand=np.flatnonzero(fid==finger)
                        if len(cand)==0: raise ValueError(f"missing finger {finger} samples")
                        tip_idx.append(int(cand[np.linalg.norm(can[cand]-center,axis=1).argmax()]))
                    tips=(hp[:,tip_idx]-pose[:,None,:3,3]) @ pose[:,:3,:3]
                rt=retargeters[side]; q_urdf=np.asarray([rt.retarget(x) for x in tips],dtype=np.float32); q=q_urdf[:,inv_native]
                p,n=_fk_surface(models[side],q,*samplings[side])
                R=pose[:,:3,:3]; t=pose[:,:3,3]
                p=np.einsum("bij,bpj->bpi",R,p)+t[:,None,:]
                n=np.einsum("bij,bpj->bpi",R,n)
                points.append(p.astype(np.float32)); normals.append(n.astype(np.float32))
                if "obj_candidate_mask_5cm" in d: masks.append(np.asarray(d["obj_candidate_mask_5cm"],dtype=bool))
            hand=np.concatenate(points,axis=1); hn=np.concatenate(normals,axis=1)
            target.mkdir(parents=True,exist_ok=False); g=target/"geometry"; g.mkdir()
            for name,val in (("obj_points_pool_world",obj),("obj_normals_pool_world",norms),("obj_pose_world",pose),("source_frame_id",raw),("frame_time",np.asarray(raw,dtype=np.float32)/120.0 if a.source=="grab" else np.asarray(raw,dtype=np.float32)/30.0),("hand_points_world",hand),("hand_normals_world",hn),("obj_candidate_mask_5cm",np.logical_or.reduce(masks) if masks else np.ones((T,),dtype=bool))): np.save(g/(name+".npy"),val)
            manifest={"schema_name":"ref2dex_object_interaction_cm_bilateral_geometry_v1","schema_version":"1.0.0","sequence_id":rel.as_posix(),"split":"train","source":"inspire_f1","source_dataset":a.source,"source_type":"stage4_mano_to_inspire_position_retarget","coordinate_frame":"object_pose_t","hand_side":"bilateral_merged_left_then_right","merged_hand_sides":True,"object_pool_points":int(obj.shape[1]),"hand_points":int(hand.shape[1]),"effective_fps":float(fps),"source_fps":120.0 if a.source=="grab" else 30.0,"candidate_threshold_m":0.05,"surface_sampling_space":"visual_mesh_local","surface_fk_application_count":1,"producer_fix":"canonical_zero_q_to_visual_local_before_fk_v1","modification_version":a.modification_version}
            (g/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
            done+=1
            if done%20==0: print(f"converted={done}",flush=True)
        except Exception as e:
            print(f"ERROR {rel}: {type(e).__name__}: {e}",file=sys.stderr,flush=True)
    print(json.dumps({"converted":done,"scanned":len(dirs)}))
if __name__=="__main__": main()
