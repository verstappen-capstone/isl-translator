import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import argparse
import re
import shutil
import time
import zipfile
from remotezip import RemoteZip
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from extract_landmarks import extract_video

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MANIFEST = DATA / "include50_manifest.csv"
OUT_DIR = DATA / "landmarks"
TMP_DIR = DATA / "tmp"
ZENODO_API = "https://zenodo.org/api/records/4010759"


def safe_name(video_path):
    stem = video_path.rsplit(".", 1)[0]
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")


def load_manifest():
    df = pd.read_csv(MANIFEST)
    df["folder_label"] = df["video_path"].str.split("/").str[1]
    bad = df["folder_label"] != df["label"]
    print(f"Manifest rows: {len(df)}; label/path mismatches dropped: {bad.sum()}")
    df = df[~bad]
    before = len(df)
    df = df.drop_duplicates(subset="video_path")
    print(f"Duplicate paths dropped: {before - len(df)}")
    print(f"Videos kept: {len(df)}; classes: {df['label'].nunique()}")
    df["split"] = df["source_file"].str.split("-").str[0]
    return df.reset_index(drop=True)


def list_zips():
    r = requests.get(ZENODO_API, timeout=60)
    r.raise_for_status()
    files = [f for f in r.json()["files"] if f["key"].lower().endswith(".zip")]
    return sorted(files, key=lambda f: f["key"])


def download(url, dest, expected_size):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == expected_size:
        print("  zip already downloaded")
        return
    for attempt in range(1, 6):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                done, last = 0, 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if done - last >= 100 * (1 << 20):
                            print(f"  downloaded {done / 1e6:.0f} / {expected_size / 1e6:.0f} MB")
                            last = done
            if dest.stat().st_size != expected_size:
                raise IOError("size mismatch after download")
            return
        except Exception as e:
            print(f"  download attempt {attempt} failed: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not download {url}")


def match_member(name, wanted):
    parts = name.replace("\\", "/").split("/")
    for k in range(len(parts)):
        cand = "/".join(parts[k:])
        if cand in wanted:
            return cand
    return None


def process_zip(zf, wanted, budget):
    found = done = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        key = match_member(info.filename, wanted)
        if key is None:
            continue
        found += 1
        out = OUT_DIR / (safe_name(key) + ".npz")
        if out.exists():
            continue
        if budget["left"] is not None and budget["left"] <= 0:
            break
        tmp = TMP_DIR / Path(key).name
        try:
            t0 = time.time()
            with zf.open(info) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            t1 = time.time()
            data = extract_video(tmp)
            t2 = time.time()
            part = out.with_name(out.stem + ".part.npz")
            np.savez_compressed(part, **data)
            part.replace(out)
            done += 1
            if budget["left"] is not None:
                budget["left"] -= 1
            print(f"  [{done}] {key}  frames={len(data['pose'])}  "
                  f"fetch={t1 - t0:.1f}s  extract={t2 - t1:.1f}s")
        except Exception as e:
            print(f"  FAILED {key}: {e}")
            with open(OUT_DIR / "failed.txt", "a") as f:
                f.write(f"{key}\t{e}\n")
        finally:
            tmp.unlink(missing_ok=True)
    return found, done

def write_index(df):
    rows = []
    for _, r in df.iterrows():
        name = safe_name(r["video_path"]) + ".npz"
        if (OUT_DIR / name).exists():
            rows.append({"npz": name, "label": r["label"], "parent": r["parent_label"],
                         "split": r["split"], "video_path": r["video_path"]})
    idx = pd.DataFrame(rows)
    idx.to_csv(OUT_DIR / "index.csv", index=False)
    print(f"\nLandmarks done: {len(idx)} / {len(df)}  (index saved to data/landmarks/index.csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list Zenodo zip files and exit")
    ap.add_argument("--only", nargs="*", help="process only these zip names")
    ap.add_argument("--limit", type=int, default=None, help="max videos to process this run")
    ap.add_argument("--keep-zips", action="store_true")
    ap.add_argument("--download-full", action="store_true",
                help="download whole zips instead of fetching only needed videos")
    args = ap.parse_args()

    zips = list_zips()
    if args.list:
        for f in zips:
            print(f"{f['key']:35s} {f['size'] / 1e6:8.0f} MB")
        print(f"\n{len(zips)} zips, {sum(f['size'] for f in zips) / 1e9:.1f} GB total")
        raise SystemExit

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    df = load_manifest()
    wanted = set(df["video_path"])
    budget = {"left": args.limit}

    for f in zips:
        if args.only and f["key"] not in args.only:
            continue
        if budget["left"] is not None and budget["left"] <= 0:
            break
        print(f"\n== {f['key']} ({f['size'] / 1e6:.0f} MB)")
        url = f["links"]["self"]
        try:
            if args.download_full:
                zp = TMP_DIR / f["key"]
                download(url, zp, f["size"])
                with zipfile.ZipFile(zp) as zf:
                    found, done = process_zip(zf, wanted, budget)
                if not args.keep_zips:
                    zp.unlink(missing_ok=True)
            else:
                with RemoteZip(url) as zf:
                    found, done = process_zip(zf, wanted, budget)
        except Exception as e:
            print(f"  zip failed: {e}")
            continue
        print(f"  INCLUDE-50 videos in this zip: {found}; newly processed: {done}")

    write_index(df)