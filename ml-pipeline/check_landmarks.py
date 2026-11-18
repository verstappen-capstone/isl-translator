from pathlib import Path
import numpy as np

folder = Path(__file__).resolve().parent.parent / "data" / "landmarks"
files = sorted(folder.glob("*.npz"))
rows = []
for f in files:
    d = np.load(f)
    det = d["detected"]  # columns: pose, face, left hand, right hand
    any_hand = (det[:, 2] | det[:, 3]).mean()
    rows.append([*det.mean(axis=0), any_hand, len(det)])
a = np.array(rows)

print(f"files: {len(files)}")
print("mean detection %  pose={:.0f} face={:.0f} left={:.0f} right={:.0f} any_hand={:.0f}".format(
    *(a[:, :5].mean(axis=0) * 100)))
print("frames per video: min={:.0f} mean={:.0f} max={:.0f}".format(
    a[:, 5].min(), a[:, 5].mean(), a[:, 5].max()))
for i in np.argsort(a[:, 4])[:3]:
    print("lowest hand coverage:", files[i].name, f"{a[i, 4] * 100:.0f}%")