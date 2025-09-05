# scripts/sweep/peak_param_sweep.py
import os, pickle, argparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import wandb

p = argparse.ArgumentParser()
p.add_argument("--prominence",    type=float, default=3.0)
p.add_argument("--height",        type=float, default=5.0)
p.add_argument("--distance",      type=int,   default=5)
p.add_argument("--width",         type=float, default=None)
p.add_argument("--wlen",          type=int,   default=None)
p.add_argument("--threshold",     type=float, default=None)
p.add_argument("--rel_height",    type=float, default=None)
p.add_argument("--plateau_size",  type=float, default=None)

p.add_argument("--smoothed",      action="store_true")
p.add_argument("--smooth_window", type=int,   default=21)
p.add_argument("--smooth_poly",   type=int,   default=3)

p.add_argument("--duration_threshold", type=float, default=20.0)

p.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output/data")),
)
args = p.parse_args()

wandb.init(project="rat-model-scripts_sweep")

cfg = wandb.config
for k in [
    "prominence","height","distance",
    "width","wlen","threshold","rel_height","plateau_size",
    "smoothed","smooth_window","smooth_poly",
    "duration_threshold",
]:
    if hasattr(cfg, k) and getattr(cfg, k) is not None:
        setattr(args, k, getattr(cfg, k))

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

df_ML = load_pickle(os.path.join(args.cache_dir, "df_ML.pkl"))

if "label" in df_ML.columns:
    y_all = df_ML["label"].astype(int).values
else:
    y_all = (df_ML["category"].astype(str) == "FR1").astype(int).values

def safe_savgol_1d(x, win, poly):
    n = len(x)
    if n < 3:
        return x
    w = min(win, n if n % 2 == 1 else n - 1)
    if w < 3:
        return x
    po = min(poly, w - 1)
    try:
        return savgol_filter(x, window_length=w, polyorder=po)
    except Exception:
        return x

def concat_session(cell):
    if isinstance(cell, (list, tuple)):
        parts = []
        for p in cell:
            a = np.asarray(p)
            if a.ndim == 1 and a.size > 0:
                parts.append(a)
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    a = np.asarray(cell)
    return a if (a.ndim == 1 and a.size > 0) else np.array([], dtype=float)

def max_peak_duration_from_peaks(x, peaks, thresh):
    if x.size == 0 or len(peaks) == 0:
        return 0.0
    durations = []
    n = len(x)
    for pk in peaks:
        lower = x[pk] - float(thresh)
        s = pk
        while s > 0 and x[s] >= lower:
            s -= 1
        e = pk
        while e < n - 1 and x[e] >= lower:
            e += 1
        durations.append(e - s)
    return float(max(durations)) if durations else 0.0

def peak_sharpness_mean_from_peaks(x, peaks):
    if x.size < 3 or len(peaks) == 0:
        return 0.0
    vals = []
    n = len(x)
    for pk in peaks:
        if 0 < pk < n - 1:
            left  = x[pk] - x[pk - 1]
            right = x[pk] - x[pk + 1]
            vals.append(abs(left) + abs(right))
    return float(np.mean(vals)) if vals else 0.0

def valley_sharpness_mean(x, pk_kwargs):
    if x.size < 3:
        return 0.0
    inv = -x
    valleys, _ = find_peaks(inv, **pk_kwargs)
    if len(valleys) == 0:
        return 0.0
    vals = []
    n = len(x)
    for v in valleys:
        if 0 < v < n - 1:
            left  = x[v] - x[v - 1]
            right = x[v] - x[v + 1]
            vals.append(abs(left) + abs(right))
    return float(np.mean(vals)) if vals else 0.0

pk_kwargs = {}
for name, val in [
    ("prominence", args.prominence),
    ("height",     args.height),
    ("distance",   args.distance),
    ("width",      args.width),
    ("wlen",       args.wlen),
    ("threshold",  args.threshold),
    ("rel_height", args.rel_height),
    ("plateau_size", args.plateau_size),
]:
    if val is not None:
        pk_kwargs[name] = val

num_peaks_list = []
max_dur_list   = []
pk_sharp_list  = []
val_sharp_list = []

for cell in df_ML["data"]:
    sig = concat_session(cell)
    if args.smoothed and sig.size:
        sig = safe_savgol_1d(sig, args.smooth_window, args.smooth_poly)

    if sig.size:
        peaks, _ = find_peaks(sig, **pk_kwargs)
        num_peaks = float(len(peaks))
        max_dur   = max_peak_duration_from_peaks(sig, peaks, args.duration_threshold)
        pk_sharp  = peak_sharpness_mean_from_peaks(sig, peaks)
        val_sharp = valley_sharpness_mean(sig, pk_kwargs)
    else:
        num_peaks = 0.0
        max_dur   = 0.0
        pk_sharp  = 0.0
        val_sharp = 0.0

    num_peaks_list.append(num_peaks)
    max_dur_list.append(max_dur)
    pk_sharp_list.append(pk_sharp)
    val_sharp_list.append(val_sharp)

X_all = pd.DataFrame({
    "num_of_peaks":           np.asarray(num_peaks_list, dtype=float),
    "max_peak_duration":      np.asarray(max_dur_list,    dtype=float),
    "peak_sharpness_mean":    np.asarray(pk_sharp_list,   dtype=float),
    "valley_sharpness_mean":  np.asarray(val_sharp_list,  dtype=float),
}, index=df_ML.index)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
for tr_idx, va_idx in skf.split(X_all, y_all):
    X_tr, y_tr = X_all.iloc[tr_idx], y_all[tr_idx]
    X_va, y_va = X_all.iloc[va_idx], y_all[va_idx]

    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_tr, y_tr)
    probs = gb.predict_proba(X_va)[:, 1]
    aucs.append(roc_auc_score(y_va, probs))

wandb.log({"val_auc": float(np.mean(aucs))})
