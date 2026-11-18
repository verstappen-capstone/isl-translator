import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "models" / "holistic_landmarker.task"
N_POSE, N_FACE, N_HAND = 33, 478, 21


def make_landmarker():
    options = vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def to_array(landmarks, n):
    """List of landmarks -> (n, 3) array. Zeros if nothing detected."""
    if landmarks and isinstance(landmarks[0], (list, tuple)):
        landmarks = landmarks[0]
    if not landmarks:
        return np.zeros((n, 3), dtype=np.float32), False
    arr = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)
    return arr, True


def extract_video(video_path, max_side=640):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    landmarker = make_landmarker()

    pose, face, lh, rh, det = [], [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(i * 1000 / fps))

        p, dp = to_array(result.pose_landmarks, N_POSE)
        f, df_ = to_array(result.face_landmarks, N_FACE)
        l, dl = to_array(result.left_hand_landmarks, N_HAND)
        r, dr = to_array(result.right_hand_landmarks, N_HAND)
        pose.append(p); face.append(f); lh.append(l); rh.append(r)
        det.append([dp, df_, dl, dr])
        i += 1

    cap.release()
    landmarker.close()
    return {
        "pose": np.stack(pose),
        "face": np.stack(face),
        "left_hand": np.stack(lh),
        "right_hand": np.stack(rh),
        "detected": np.array(det, dtype=bool),  # columns: pose, face, left, right
        "fps": np.float32(fps),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    data = extract_video(args.video)
    dt = time.time() - t0
    n = len(data["pose"])
    rates = data["detected"].mean(axis=0) * 100
    print(f"Frames: {n}, source fps: {float(data['fps']):.1f}")
    print(f"Processing time: {dt:.1f}s ({n / dt:.1f} frames/s)")
    print(f"Detection rate %  pose={rates[0]:.0f}  face={rates[1]:.0f}  "
          f"left_hand={rates[2]:.0f}  right_hand={rates[3]:.0f}")
    print("Shapes:", {k: v.shape for k, v in data.items() if k != "fps"})
    if args.out:
        np.savez_compressed(args.out, **data)
        print("Saved", args.out)