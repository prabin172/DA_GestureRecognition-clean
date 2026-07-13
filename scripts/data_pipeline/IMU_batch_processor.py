import csv
import re
from pathlib import Path
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# -----------------------------
# User-configurable settings
# -----------------------------
MODE = "swing"        # "local" or "swing"
OVERWRITE = True      # if True, clears output folder .npy + index.csv before writing
IMU_HZ = 240
DS_STRIDE = 8         # 240 -> 30 Hz
SHEET_NAME = "Segment Orientation - Quat"

OFFSET_CSV_REL = Path("DataCollection/GroundTruth-Annotations-Offset.csv")
DATA_ROOT_REL = Path("DataCollection")
ANVIL_ROOT_REL = Path("DataCollection/Annotations/Omari_Annotations")
OUT_DIR_REL = Path("Data_Processed/imu_quats")


# -----------------------------
# 17 segments (match NTU 17x4 = 68)
# -----------------------------
SEGMENTS_17 = [
    "Pelvis",
    "T8",
    "Head",
    "Right Shoulder",
    "Right Upper Arm",
    "Right Forearm",
    "Right Hand",
    "Left Shoulder",
    "Left Upper Arm",
    "Left Forearm",
    "Left Hand",
    "Right Upper Leg",
    "Right Lower Leg",
    "Right Foot",
    "Left Upper Leg",
    "Left Lower Leg",
    "Left Foot",
]

# Parent hierarchy (same topology as your NTU hierarchy)
HIERARCHY = {
    0: None,   # Pelvis
    1: 0,      # T8 <- Pelvis
    2: 1,      # Head <- T8
    3: 1,      # R Shoulder <- T8
    4: 3,      # R Upper Arm <- R Shoulder
    5: 4,      # R Forearm <- R Upper Arm
    6: 5,      # R Hand <- R Forearm
    7: 1,      # L Shoulder <- T8
    8: 7,      # L Upper Arm <- L Shoulder
    9: 8,      # L Forearm <- L Upper Arm
    10: 9,     # L Hand <- L Forearm
    11: 0,     # R Upper Leg <- Pelvis
    12: 11,    # R Lower Leg <- R Upper Leg
    13: 12,    # R Foot <- R Lower Leg
    14: 0,     # L Upper Leg <- Pelvis
    15: 14,    # L Lower Leg <- L Upper Leg
    16: 15,    # L Foot <- L Lower Leg
}

# Axes used for optional twist removal (swing-only)
UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
DOWN = np.array([0.0, -1.0, 0.0], dtype=np.float32)
FORWARD = np.array([0.0, 0.0, 1.0], dtype=np.float32)
RIGHT = np.array([1.0, 0.0, 0.0], dtype=np.float32)
LEFT = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

TWIST_AXIS = [
    UP,        # Pelvis
    UP,        # T8
    UP,        # Head
    RIGHT,     # R Shoulder
    DOWN,      # R Upper Arm
    DOWN,      # R Forearm
    DOWN,      # R Hand
    LEFT,      # L Shoulder
    DOWN,      # L Upper Arm
    DOWN,      # L Forearm
    DOWN,      # L Hand
    DOWN,      # R Upper Leg
    DOWN,      # R Lower Leg
    FORWARD,   # R Foot
    DOWN,      # L Upper Leg
    DOWN,      # L Lower Leg
    FORWARD,   # L Foot
]


# -----------------------------
# Quaternion helpers (xyzw)
# -----------------------------
def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, 1e-8, None)

def quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] *= -1.0
    return out

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Hamilton product for xyzw quaternions
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return np.stack([x, y, z, w], axis=-1)

def fix_sign_continuity(q: np.ndarray) -> np.ndarray:
    # q shape (T,4) in xyzw
    out = q.copy()
    for t in range(1, out.shape[0]):
        if float(np.dot(out[t], out[t - 1])) < 0.0:
            out[t] *= -1.0
    return out

def remove_twist_xyzw(q: np.ndarray, axis_parent: np.ndarray) -> np.ndarray:
    """
    Swing-twist decomposition around a fixed axis in the parent frame.
    Returns swing-only quaternion (twist removed).
    q: (...,4) in xyzw
    """
    axis = axis_parent / np.linalg.norm(axis_parent)
    v = q[..., :3]
    w = q[..., 3:4]
    dot = np.sum(v * axis, axis=-1, keepdims=True)
    proj = dot * axis
    twist = np.concatenate([proj, w], axis=-1)
    twist = quat_normalize(twist)
    swing = quat_mul(q, quat_conj(twist))
    return quat_normalize(swing)


# -----------------------------
# Parsing helpers
# -----------------------------
def sanitize_label(label: str) -> str:
    s = label.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s if s else "unknown"

def load_offsets(offset_csv: Path) -> dict:
    offsets = {}
    with offset_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # Normalize fieldnames (strip spaces/BOM)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {offset_csv}")

        field_map = {fn.strip(): fn for fn in reader.fieldnames}

        file_col = field_map.get("File")
        off_col = field_map.get("Offset with Xsens (s)")

        if file_col is None or off_col is None:
            raise ValueError(
                f"Expected columns 'File' and 'Offset with Xsens (s)' in {offset_csv}, "
                f"but found: {reader.fieldnames}"
            )

        for row in reader:
            key = str(row[file_col]).strip()
            offsets[key] = float(row[off_col])

    return offsets

def parse_anvil(anvil_path: Path):
    """
    Returns list of (start_sec, end_sec, label_str).
    """
    text = anvil_path.read_text(encoding="utf-16")
    root = ET.fromstring(text)

    out = []
    for track in root.findall(".//track[@name='gesture']"):
        for el in track.findall("el"):
            start = float(el.attrib["start"])
            end = float(el.attrib["end"])
            attr = el.find("./attribute[@name='type']")
            label = attr.text.strip() if attr is not None and attr.text else "unknown"
            out.append((start, end, label))
    return out


# -----------------------------
# Xsens Excel -> (T,17,4) local quats in xyzw
# -----------------------------
def read_xlsx_global_quats_xyzw(xlsx_path: Path) -> np.ndarray:
    """
    Reads segment orientations from Excel.
    Xsens columns q0..q3 are assumed WXYZ and converted to XYZW.
    Returns (T,17,4) xyzw normalized.
    """
    needed_cols = ["Frame"]
    for seg in SEGMENTS_17:
        needed_cols.extend([f"{seg} q0", f"{seg} q1", f"{seg} q2", f"{seg} q3"])

    df = pd.read_excel(
        xlsx_path,
        sheet_name=SHEET_NAME,
        usecols=needed_cols,
        engine="openpyxl",
    )

    T = len(df)
    Q_wxyz = np.empty((T, 17, 4), dtype=np.float32)
    for i, seg in enumerate(SEGMENTS_17):
        cols = [f"{seg} q0", f"{seg} q1", f"{seg} q2", f"{seg} q3"]
        Q_wxyz[:, i, :] = df[cols].to_numpy(dtype=np.float32)

    # WXYZ -> XYZW
    Q = Q_wxyz[..., [1, 2, 3, 0]]
    Q = quat_normalize(Q)

    # sign continuity per segment
    for i in range(17):
        Q[:, i, :] = fix_sign_continuity(Q[:, i, :])

    return Q

def global_to_local(Q_global: np.ndarray) -> np.ndarray:
    """
    Converts global segment orientations to parent-relative local orientations.
    Q_global: (T,17,4) xyzw
    Returns Q_local: (T,17,4) xyzw
    """
    T = Q_global.shape[0]
    Q_local = np.empty_like(Q_global)

    # Root is kept as-is
    Q_local[:, 0, :] = Q_global[:, 0, :]

    for i in range(1, 17):
        p = HIERARCHY[i]
        parent_inv = quat_conj(Q_global[:, p, :])  # inverse for unit quats
        Q_local[:, i, :] = quat_mul(parent_inv, Q_global[:, i, :])

    Q_local = quat_normalize(Q_local)
    for i in range(17):
        Q_local[:, i, :] = fix_sign_continuity(Q_local[:, i, :])

    return Q_local

def slice_and_downsample(Q_local_240: np.ndarray, start_sec: float, end_sec: float, offset_sec: float):
    """
    Converts video-time interval to Xsens frame indices using offset.
    Slices at 240 Hz and downsamples to ~30 Hz.
    Returns (clip_30hz, f0_240, f1_240) or (None, f0, f1).
    """
    xs_start = start_sec - offset_sec
    xs_end = end_sec - offset_sec

    f0 = int(round(xs_start * IMU_HZ))
    f1 = int(round(xs_end * IMU_HZ))

    f0 = max(f0, 0)
    f1 = min(f1, Q_local_240.shape[0])

    if f1 <= f0 + 1:
        return None, f0, f1

    clip = Q_local_240[f0:f1]   # (Tclip,17,4) @240Hz
    clip = clip[::DS_STRIDE]    # ~30Hz

    if clip.shape[0] < 2:
        return None, f0, f1

    return clip.astype(np.float32), f0, f1


# -----------------------------
# Batch processing
# -----------------------------
def clear_output_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.npy"):
        p.unlink()
    idx = out_dir / "index.csv"
    if idx.exists():
        idx.unlink()

def main():
    # Infer repo root from file location: IROS26/src/scripts/IMU_batch_processor.py -> parents[2] = IROS26
    project_root = Path(__file__).resolve().parents[2]

    offset_csv = (project_root / OFFSET_CSV_REL).resolve()
    data_root = (project_root / DATA_ROOT_REL).resolve()
    anvil_root = (project_root / ANVIL_ROOT_REL).resolve()
    out_dir = (project_root / OUT_DIR_REL).resolve()

    if MODE not in ("local", "swing"):
        raise ValueError("MODE must be 'local' or 'swing'.")

    if OVERWRITE:
        clear_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    offsets = load_offsets(offset_csv)
    print(f"Loaded {len(offsets)} session offsets.")

    index_path = out_dir / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f_index:
        writer = csv.DictWriter(
            f_index,
            fieldnames=[
                "file", "session", "label",
                "start_sec", "end_sec", "offset_sec",
                "start_frame_240", "end_frame_240",
                "n_frames_30hz", "mode"
            ]
        )
        writer.writeheader()

        for session_id, offset_sec in offsets.items():
            sub_folder = session_id.split("-")[0]  # "sub7" from "sub7-a"
            xlsx_path = data_root / sub_folder / f"{session_id}.xlsx"
            anvil_path = anvil_root / f"{session_id}.anvil"
            print(f"\nProcessing {session_id} ...")

            if not xlsx_path.exists() or not anvil_path.exists():
                continue

            Q_global = read_xlsx_global_quats_xyzw(xlsx_path)
            Q_local = global_to_local(Q_global)

            if MODE == "swing":
                for i in range(17):
                    Q_local[:, i, :] = remove_twist_xyzw(Q_local[:, i, :], TWIST_AXIS[i])
                    Q_local[:, i, :] = fix_sign_continuity(Q_local[:, i, :])
                Q_local = quat_normalize(Q_local)

            events = parse_anvil(anvil_path)
            counter = 0

            for (start_sec, end_sec, label_raw) in events:
                label = sanitize_label(label_raw)
                clip, f0, f1 = slice_and_downsample(Q_local, start_sec, end_sec, offset_sec)
                if clip is None:
                    continue

                counter += 1
                out_name = f"{session_id}__{counter:04d}__{label}.npy"
                out_path = out_dir / out_name
                np.save(out_path, clip)  # clip is (T,17,4)

                writer.writerow({
                    "file": out_name,
                    "session": session_id,
                    "label": label,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "offset_sec": offset_sec,
                    "start_frame_240": f0,
                    "end_frame_240": f1,
                    "n_frames_30hz": int(clip.shape[0]),
                    "mode": MODE
                })
            print(f"  Saved {counter} clips from {session_id}.")

    print(f"Done. Wrote clips to: {out_dir}")
    print(f"Index: {index_path}")

if __name__ == "__main__":
    main()