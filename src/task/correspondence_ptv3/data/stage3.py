from __future__ import annotations

from pathlib import Path

import numpy as np

from src.task.correspondence_ptv3.contracts import Stage3Frame, Stage3FrameRef


def load_blacklist(path: str | None) -> set[str]:
    if not path:
        return set()
    blacklist_path = Path(path)
    if not blacklist_path.exists():
        raise FileNotFoundError(f"Blacklist file not found: {blacklist_path}")
    if blacklist_path.suffix.lower() == ".json":
        import json

        payload = json.loads(blacklist_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Blacklist JSON must contain a list")
        return {str(item) for item in payload}
    return {
        line.strip()
        for line in blacklist_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def is_blacklisted(path: Path, root: Path, blacklist: set[str]) -> bool:
    if not blacklist:
        return False
    keys = {str(path), path.name, path.stem, path.as_posix()}
    try:
        keys.add(path.relative_to(root).as_posix())
    except ValueError:
        pass
    return bool(keys.intersection(blacklist))


class Stage3Store:
    REQUIRED_FIELDS = {
        "raw_frame_id",
        "obj_points",
        "obj_normals",
        "obj_point_id",
        "hand_points",
        "hand_normals",
        "hand_point_id",
        "obj_to_hand_min_dist",
        "obj_candidate_mask_5cm",
        "gt_obj_to_hand_knn_idx",
    }

    def __init__(
        self,
        data_path: str | Path,
        *,
        file_list: list[str | Path] | None = None,
        blacklist_path: str | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.data_root = self.data_path if self.data_path.is_dir() else self.data_path.parent
        paths = (
            sorted(Path(path) for path in file_list)
            if file_list is not None
            else (
                sorted(self.data_path.glob("**/*.npz"))
                if self.data_path.is_dir()
                else [self.data_path]
            )
        )
        blacklist = load_blacklist(blacklist_path)
        self.file_paths = [path for path in paths if not is_blacklisted(path, self.data_root, blacklist)]
        if not self.file_paths:
            raise ValueError(f"No Stage 3 npz files found in {self.data_path}")

        self._samples: list[tuple[Path, int]] = []
        self.file_sample_ranges: list[tuple[int, int]] = []
        for path in self.file_paths:
            with np.load(path, allow_pickle=False) as data:
                missing = self.REQUIRED_FIELDS.difference(data.files)
                if missing:
                    raise KeyError(f"{path}: missing Stage 3 fields {sorted(missing)}")
                num_frames = int(data["raw_frame_id"].shape[0])
            start = len(self._samples)
            self._samples.extend((path, frame_idx) for frame_idx in range(num_frames))
            self.file_sample_ranges.append((start, len(self._samples)))

        self._cached_path: Path | None = None
        self._cached_data: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self._samples)

    def frame_ref(self, index: int) -> Stage3FrameRef:
        path, frame_idx = self._samples[index]
        return Stage3FrameRef(path=str(path), frame_idx=int(frame_idx))

    def load_frame(self, index: int) -> Stage3Frame:
        path, frame_idx = self._samples[index]
        data = self._load_file(path)
        hand_points = np.asarray(data["hand_points"][frame_idx], dtype=np.float32)
        num_hand = int(hand_points.shape[0])
        return Stage3Frame(
            seq_id=self._scalar_string(data, "seq_id", path.stem),
            side=self._scalar_string(data, "side", ""),
            raw_frame_id=int(np.asarray(data["raw_frame_id"])[frame_idx]),
            obj_points=np.asarray(data["obj_points"][frame_idx], dtype=np.float32),
            obj_normals=np.asarray(data["obj_normals"][frame_idx], dtype=np.float32),
            obj_point_id=np.asarray(data["obj_point_id"], dtype=np.int64),
            hand_points=hand_points,
            hand_normals=np.asarray(data["hand_normals"][frame_idx], dtype=np.float32),
            hand_point_id=np.asarray(data["hand_point_id"], dtype=np.int64),
            obj_to_hand_min_dist=np.asarray(data["obj_to_hand_min_dist"][frame_idx], dtype=np.float32),
            obj_candidate_mask_5cm=np.asarray(data["obj_candidate_mask_5cm"][frame_idx], dtype=bool),
            gt_obj_to_hand_knn_idx=np.asarray(data["gt_obj_to_hand_knn_idx"][frame_idx], dtype=np.int64),
            hand_cano_points=np.asarray(
                data.get("hand_cano_points", np.zeros((num_hand, 3), dtype=np.float32))
            ).astype(np.float32, copy=False),
            hand_finger_id=np.asarray(
                data.get("hand_finger_id", np.full((num_hand,), -1, dtype=np.int64))
            ).astype(np.int64, copy=False),
            hand_region_id=np.asarray(
                data.get("hand_region_id", np.full((num_hand,), -1, dtype=np.int64))
            ).astype(np.int64, copy=False),
        )

    def load_raw_file(self, path: str | Path) -> dict[str, np.ndarray]:
        return self._load_file(Path(path))

    def _load_file(self, path: Path) -> dict[str, np.ndarray]:
        if self._cached_path == path and self._cached_data is not None:
            return self._cached_data
        with np.load(path, allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}
        self._cached_path = path
        self._cached_data = payload
        return payload

    @staticmethod
    def _scalar_string(data: dict[str, np.ndarray], key: str, default: str = "") -> str:
        value = data.get(key)
        if value is None:
            return default
        array = np.asarray(value)
        return str(array.item()) if array.size == 1 else default
