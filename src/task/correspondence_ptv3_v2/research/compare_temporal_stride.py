from pathlib import Path
import numpy as np


def summarize(root: str, stride: int, object_frame: bool) -> None:
    means, rmss = [], []
    for path in sorted(Path(root).rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            hand = np.asarray(data["hand_points"], np.float32)
            if hand.shape[0] <= stride:
                continue
            if object_frame:
                pose = np.asarray(data["hand_root_pose"], np.float32)
                obj = np.asarray(data["obj_root_pose_world"], np.float32)
                rel_r = np.einsum("tij,tjk->tik", obj[:, :3, :3].transpose(0, 2, 1), pose[:, :3, :3])
                rel_t = np.einsum("tij,tj->ti", obj[:, :3, :3].transpose(0, 2, 1), pose[:, :3, 3] - obj[:, :3, 3])
                hand = np.einsum("tni,tji->tnj", hand, rel_r) + rel_t[:, None, :]
            delta = np.linalg.norm(hand[stride:] - hand[:-stride], axis=-1)
            means.append(delta.mean(1))
            rmss.append(np.sqrt((delta * delta).mean(1)))
    for label, values in (("mean", means), ("rms", rmss)):
        x = np.concatenate(values) * 1000
        print(f"{label}: mean={x.mean():.3f} p50={np.percentile(x,50):.3f} p95={np.percentile(x,95):.3f} p99={np.percentile(x,99):.3f} n={len(x)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--object-frame", action="store_true")
    args = parser.parse_args()
    summarize(args.root, args.stride, args.object_frame)
