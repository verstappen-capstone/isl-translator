import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LM = DATA / "landmarks"
T = 64                                  # frames per sequence after resampling
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]  # nose, shoulders, elbows, wrists


def fill_missing(arr, det):
    """Fill frames where a part was not detected with the nearest detected frame."""
    arr = arr.copy()
    idx = np.where(det)[0]
    if len(idx) == 0:
        return arr * 0
    last = idx[0]
    for t in range(arr.shape[0]):
        if det[t]:
            last = t
        else:
            arr[t] = arr[last]
    return arr


def resample(x, n):
    """x: (F, D) -> (n, D) by linear interpolation over time."""
    f = x.shape[0]
    pos = np.linspace(0, f - 1, n)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, f - 1)
    w = (pos - i0)[:, None]
    return x[i0] * (1 - w) + x[i1] * w


def video_features(path, face_stride):
    d = np.load(path)
    det = d["detected"]  # columns: pose, face, left hand, right hand
    pose = fill_missing(d["pose"], det[:, 0])
    lh = fill_missing(d["left_hand"], det[:, 2])
    rh = fill_missing(d["right_hand"], det[:, 3])

    sl, sr = pose[:, 11, :2], pose[:, 12, :2]
    mid = (sl + sr) / 2
    scale = max(float(np.median(np.linalg.norm(sl - sr, axis=1))), 1e-3)

    def norm(p):
        q = p.copy()
        q[..., :2] = (q[..., :2] - mid[:, None, :]) / scale
        q[..., 2] = q[..., 2] / scale
        return q

    parts = [norm(pose)[:, POSE_IDX]]
    lh_n, rh_n = norm(lh), norm(rh)
    if not det[:, 2].any():
        lh_n = lh_n * 0
    if not det[:, 3].any():
        rh_n = rh_n * 0
    parts += [lh_n, rh_n]
    if face_stride > 0:
        face = fill_missing(d["face"], det[:, 1])
        parts.append(norm(face)[:, ::face_stride])

    pts = np.concatenate(parts, axis=1)                     # (F, P, 3)
    feats = pts.reshape(pts.shape[0], -1)
    flags = det[:, 2:4].astype(np.float32)                  # hand present?
    feats = np.concatenate([feats, flags], axis=1)
    return resample(feats, T).astype(np.float32), det.mean(axis=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-stride", type=int, default=0,
                    help="0 = no face; N = every Nth face point")
    ap.add_argument("--out", default="features_pose_hands.npz")
    args = ap.parse_args()

    idx = pd.read_csv(LM / "index.csv")
    idx["cls"] = idx["parent"] + "/" + idx["label"]
    classes = sorted(idx["cls"].unique())
    cid = {c: i for i, c in enumerate(classes)}

    out, det_rates = {}, []
    for split in ["train", "val", "test"]:
        sub = idx[idx["split"] == split]
        X, y = [], []
        for _, r in sub.iterrows():
            f, rate = video_features(LM / r["npz"], args.face_stride)
            X.append(f)
            y.append(cid[r["cls"]])
            det_rates.append(rate)
        out[f"X_{split}"] = np.stack(X)
        out[f"y_{split}"] = np.array(y)

    np.savez_compressed(DATA / args.out, classes=np.array(classes), **out)

    print(f"classes: {len(classes)}")
    for s in ["train", "val", "test"]:
        y = out[f"y_{s}"]
        counts = np.bincount(y, minlength=len(classes))
        print(f"{s:5s} X={out[f'X_{s}'].shape}  per-class min/median/max = "
              f"{counts.min()}/{int(np.median(counts))}/{counts.max()}  "
              f"classes missing: {(counts == 0).sum()}")
    r = np.array(det_rates).mean(axis=0) * 100
    print(f"mean detection %: pose={r[0]:.0f} face={r[1]:.0f} left={r[2]:.0f} right={r[3]:.0f}")
    print(f"videos with no left hand ever: "
          f"{sum(1 for x in det_rates if x[2] == 0)}, no right hand ever: "
          f"{sum(1 for x in det_rates if x[3] == 0)}")
    print("saved", DATA / args.out)