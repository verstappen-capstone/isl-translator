import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import layers, regularizers

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = Path(__file__).resolve().parent / "models"
MODELS.mkdir(exist_ok=True)


def resample(x, n):
    f = x.shape[0]
    pos = np.linspace(0, f - 1, n)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, f - 1)
    w = (pos - i0)[:, None]
    return x[i0] * (1 - w) + x[i1] * w


def augment(X, rng):
    N, T, D = X.shape
    P = (D - 2) // 3
    out = np.empty_like(X)
    for i in range(N):
        keep = int(T * rng.uniform(0.8, 1.0))
        start = rng.integers(0, T - keep + 1)
        x = resample(X[i][start:start + keep], T)
        pts = x[:, :-2].reshape(T, P, 3).copy()
        flags = (x[:, -2:] > 0.5).astype(np.float32)
        absent = np.all(pts == 0, axis=-1)            # hands that were never detected

        a = np.deg2rad(rng.uniform(-10, 10))
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32)
        pts[..., :2] = pts[..., :2] @ rot.T
        pts *= rng.uniform(0.9, 1.1)
        pts[..., :2] += rng.uniform(-0.1, 0.1, size=2)
        pts += rng.normal(0, 0.01, pts.shape)
        pts[absent] = 0

        out[i] = np.concatenate([pts.reshape(T, -1), flags], axis=1)
    return out.astype(np.float32)


def build(model_type, T, D, n_cls):
    def rnn(units, ret):
        l = layers.LSTM(units, return_sequences=ret,
                        kernel_regularizer=regularizers.l2(1e-4))
        return layers.Bidirectional(l) if model_type == "bilstm" else l

    inp = layers.Input((T, D))
    x = rnn(64, True)(inp)
    x = layers.Dropout(0.4)(x)
    x = rnn(128, False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_cls, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_pose_hands.npz")
    ap.add_argument("--model", choices=["bilstm", "lstm"], default="bilstm")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    name = args.name or f"{args.model}_{Path(args.features).stem.replace('features_', '')}"

    d = np.load(DATA / args.features, allow_pickle=True)
    classes = [str(c) for c in d["classes"]]
    Xtr, ytr = d["X_train"], d["y_train"]
    Xva, yva = d["X_val"], d["y_val"]
    Xte, yte = d["X_test"], d["y_test"]

    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = build(args.model, Xtr.shape[1], Xtr.shape[2], len(classes))
    print(f"{name}: {model.count_params():,} parameters, train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    best_loss, best_w, best_ep, wait = np.inf, None, 0, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        h = model.fit(augment(Xtr, rng), ytr, batch_size=32, epochs=1, verbose=0)
        vl, va = model.evaluate(Xva, yva, verbose=0)
        if vl < best_loss - 1e-4:
            best_loss, best_w, best_ep, wait = vl, model.get_weights(), ep, 0
        else:
            wait += 1
        if ep % 10 == 0:
            print(f"epoch {ep:3d}  train_acc={h.history['accuracy'][0]:.3f}  "
                  f"val_loss={vl:.3f}  val_acc={va:.3f}  ({time.time() - t0:.0f}s)")
        if wait >= args.patience:
            print(f"early stop at epoch {ep}")
            break

    model.set_weights(best_w)
    tr_acc = model.evaluate(Xtr, ytr, verbose=0)[1]
    va_acc = model.evaluate(Xva, yva, verbose=0)[1]
    probs = model.predict(Xte, verbose=0)
    pred = probs.argmax(1)
    top1 = (pred == yte).mean()
    top3 = np.mean([y in np.argsort(p)[-3:] for p, y in zip(probs, yte)])
    f1 = f1_score(yte, pred, labels=np.unique(yte), average="macro", zero_division=0)

    print(f"\nbest epoch {best_ep}")
    print(f"train acc {tr_acc:.3f} | val acc {va_acc:.3f}")
    print(f"TEST top-1 {top1:.3f} | top-3 {top3:.3f} | macro-F1 {f1:.3f}")

    model.save(MODELS / f"{name}.keras")
    (MODELS / f"{name}_classes.json").write_text(json.dumps(classes))
    print("saved", MODELS / f"{name}.keras")