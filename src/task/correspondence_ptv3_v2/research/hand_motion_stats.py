from pathlib import Path
import numpy as np

ROOTS = {
    "GRAB": "/mnt/ugreen_nas/storage/Ref2Dex_storage/GRAB/processed/stage3/grab_initonly_4096_v21",
    "ContactPose": "/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/contactpose_v21_interaction5cm_allhands_20260825",
    "OakInk": "/mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk/processed/stage3_corr_oakink_true_handroot_20260824",
    "HRDex-human": "/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hrdexdb_minimal_allhands_v1/human",
    "Inspire-DFTP": "/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hrdexdb_minimal_allhands_v1/inspire_dftp",
    "Inspire-F1": "/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hrdexdb_minimal_allhands_v1/inspire_f1",
    "Allegro-V5": "/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hrdexdb_minimal_allhands_v1/allegro_v5",
}


def fmt(x: np.ndarray) -> str:
    q = np.percentile(x, [50, 75, 90, 95, 99])
    return "mean %.3f | p50 %.3f | p75 %.3f | p90 %.3f | p95 %.3f | p99 %.3f" % (x.mean(), *q)


for name, root in ROOTS.items():
    fm, fr, rw, oo, orr = [], [], [], [], []
    nedge = ngap = objfiles = 0
    files = sorted(Path(root).rglob("*.npz"))
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            hand = np.asarray(data["hand_points"], np.float32)
            if hand.shape[0] < 2:
                continue
            ids = np.asarray(data["raw_frame_id"])
            ngap += int(np.sum(np.diff(ids) != 1))
            nedge += hand.shape[0] - 1
            delta = np.linalg.norm(hand[1:] - hand[:-1], axis=-1)
            fm.append(delta.mean(1))
            fr.append(np.sqrt((delta * delta).mean(1)))
            pose = np.asarray(data["hand_root_pose"], np.float32)
            rw.append(np.linalg.norm(pose[1:, :3, 3] - pose[:-1, :3, 3], axis=1))
            if "obj_root_pose_world" in data.files:
                objfiles += 1
                obj = np.asarray(data["obj_root_pose_world"], np.float32)
                rh, th = pose[:, :3, :3], pose[:, :3, 3]
                ro, to = obj[:, :3, :3], obj[:, :3, 3]
                rel_r = np.einsum("tij,tjk->tik", ro.transpose(0, 2, 1), rh)
                rel_t = np.einsum("tij,tj->ti", ro.transpose(0, 2, 1), th - to)
                hand_obj = np.einsum("tni,tji->tnj", hand, rel_r) + rel_t[:, None, :]
                delta_obj = np.linalg.norm(hand_obj[1:] - hand_obj[:-1], axis=-1)
                oo.append(np.stack([delta_obj.mean(1), np.sqrt((delta_obj * delta_obj).mean(1))], 1))
                orr.append(np.linalg.norm(rel_t[1:] - rel_t[:-1], axis=1))
    x, y, z = np.concatenate(fm) * 1000, np.concatenate(fr) * 1000, np.concatenate(rw) * 1000
    print(f"## {name} | files={len(files)} | adjacent_edges={nedge} | raw_id_gaps={ngap} ({100*ngap/nedge:.1f}%) | obj_pose_files={objfiles}")
    print("native frame-mean point displacement (mm/frame): " + fmt(x))
    print("native frame-RMS point displacement (mm/frame): " + fmt(y))
    print("hand-root world translation (mm/frame): " + fmt(z))
    if oo:
        a, b = np.concatenate(oo) * 1000, np.concatenate(orr) * 1000
        print("object-frame frame-mean point displacement (mm/frame): " + fmt(a[:, 0]))
        print("object-frame frame-RMS point displacement (mm/frame): " + fmt(a[:, 1]))
        print("object-frame hand-root translation (mm/frame): " + fmt(b))
