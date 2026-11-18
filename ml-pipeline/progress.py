import re
import time
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
LM = DATA / "landmarks"


def safe_name(p):
    return re.sub(r"[^A-Za-z0-9]+", "_", p.rsplit(".", 1)[0]).strip("_")


df = pd.read_csv(DATA / "include50_manifest.csv")
df = df[df["video_path"].str.split("/").str[1] == df["label"]].drop_duplicates("video_path")
df["name"] = df["video_path"].map(lambda p: safe_name(p) + ".npz")

done_files = {f.name: f for f in LM.glob("*.npz") if not f.name.endswith(".part.npz")}
df["done"] = df["name"].isin(done_files)

print(f"Done: {df['done'].sum()} / {len(df)} ({df['done'].mean() * 100:.0f}%)")
print(df.groupby("parent_label")["done"].agg(done="sum", total="count").to_string())

now = time.time()
recent = [f for f in done_files.values() if now - f.stat().st_mtime < 600]
rate = len(recent) / 10
remaining = len(df) - df["done"].sum()
if rate > 0:
    print(f"\nRate: {rate:.1f} videos/min (last 10 min), ETA ~{remaining / rate / 60:.1f} h")
else:
    print("\nNo videos finished in the last 10 min (is the run active?)")

failed = LM / "failed.txt"
n_failed = sum(1 for _ in open(failed)) if failed.exists() else 0
print("Failed videos:", n_failed)