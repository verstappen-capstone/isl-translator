from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

local = snapshot_download(
    repo_id="ai4bharat/INCLUDE",
    repo_type="dataset",
    local_dir=DATA_DIR / "include_meta",
)

frames = []
for p in sorted(Path(local).rglob("*")):
    if p.suffix not in (".csv", ".parquet"):
        continue
    df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
    print(f"{p.name}: shape={df.shape}, columns={list(df.columns)}")
    if "include_50" in df.columns:
        df["source_file"] = p.name
        frames.append(df)

if not frames:
    raise SystemExit("No metadata file with an include_50 column found.")

meta = pd.concat(frames, ignore_index=True)
sub = meta[meta["include_50"] == True]
print("\nTotal videos:", len(meta))
print("INCLUDE-50 videos:", len(sub))
print("INCLUDE-50 classes:", sub["label"].nunique())
print(sub["parent_label"].value_counts())

sub.to_csv(DATA_DIR / "include50_manifest.csv", index=False)
print("Saved data/include50_manifest.csv")