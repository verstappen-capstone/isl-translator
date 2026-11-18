from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent / "data"
lm = root / "landmarks"

for p in lm.glob("*.part.npz"):
    p.unlink()
    print("removed partial:", p.name)

files = list(lm.glob("*.npz"))
bad = 0
for f in files:
    try:
        with np.load(f) as d:
            for k in d.files:
                _ = d[k]
    except Exception as e:
        print("corrupt, deleting:", f.name, "-", e)
        f.unlink()
        bad += 1

for v in (root / "tmp").glob("*"):
    if v.suffix.lower() in (".mov", ".mp4"):
        v.unlink()

print(f"checked {len(files)} files, deleted {bad} corrupt")