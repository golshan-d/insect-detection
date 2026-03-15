"""
Insect Activity Index Pipeline — Team 893C
==========================================
Takes a folder of WAV files recorded by the ESP32 + INMP441 node,
computes band-limited spectral energy in the insect wingbeat range,
and outputs per-clip and hourly/daily activity summaries as CSV.

Usage:
    python analyze.py --input ./recordings --output ./results

Filename convention expected from ESP32 firmware (configurable below):
    YYYYMMDD_HHMMSS.wav   e.g.  20260601_143022.wav
If your firmware names files differently, set --timestamp_fmt accordingly.

Dependencies:
    pip install numpy scipy librosa soundfile pandas matplotlib
"""

import argparse
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import soundfile as sf
from scipy.signal import butter, sosfilt
from scipy.fft import rfft, rfftfreq

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Configuration — tweak these to match your deployment
# ---------------------------------------------------------------------------

# Insect wingbeat band of interest (Hz)
# Bees: ~220–250 Hz, flies: ~100–400 Hz, broader range catches more species
BAND_LOW_HZ  = 180
BAND_HIGH_HZ = 1500

# Butterworth filter order
FILTER_ORDER = 4

# Target sample rate — resample to this if needed
TARGET_SR = 16000

# Minimum clip duration to process (seconds) — skip very short files
MIN_CLIP_DURATION_S = 0.5

# SNR estimation: noise floor estimated from the quietest X% of frames
NOISE_PERCENTILE = 20   # bottom 20% of frame energies = noise floor estimate

# Frame parameters for energy estimation (seconds)
FRAME_SIZE_S = 0.02     # 20 ms frames
HOP_SIZE_S   = 0.01     # 10 ms hop

# Activity threshold: clip is "active" if its band energy exceeds
# this many times the estimated noise floor
ACTIVITY_MULTIPLIER = 2.0

# ---------------------------------------------------------------------------


def butter_bandpass(data: np.ndarray, sr: int,
                    low: float, high: float, order: int = 4) -> np.ndarray:
    """Apply a Butterworth bandpass filter."""
    nyq = 0.5 * sr
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfilt(sos, data)


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file, convert to mono float32, return (samples, sr)."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    # Mix to mono
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    return data, sr


def resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear resample (avoids heavy librosa dep for basic use)."""
    if orig_sr == target_sr:
        return data
    try:
        import librosa
        return librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Fallback: scipy resample
        from scipy.signal import resample as sp_resample
        n_samples = int(len(data) * target_sr / orig_sr)
        return sp_resample(data, n_samples).astype(np.float32)


def compute_band_energy(data: np.ndarray, sr: int,
                        low: float, high: float) -> float:
    """
    Compute the fraction of total spectral energy in [low, high] Hz.
    Returns a value in [0, 1] — the band energy ratio.
    Also returns absolute band energy for thresholding.
    """
    N = len(data)
    if N == 0:
        return 0.0, 0.0

    spectrum = np.abs(rfft(data * np.hanning(N)))
    freqs    = rfftfreq(N, d=1.0 / sr)

    mask      = (freqs >= low) & (freqs <= high)
    band_pow  = np.sum(spectrum[mask] ** 2)
    total_pow = np.sum(spectrum ** 2)

    ratio = float(band_pow / total_pow) if total_pow > 0 else 0.0
    return ratio, float(band_pow)


def frame_energies(data: np.ndarray, sr: int,
                   frame_s: float, hop_s: float) -> np.ndarray:
    """Compute RMS energy per frame."""
    frame_len = int(frame_s * sr)
    hop_len   = int(hop_s * sr)
    frames = []
    for start in range(0, len(data) - frame_len + 1, hop_len):
        frame = data[start: start + frame_len]
        frames.append(np.sqrt(np.mean(frame ** 2)))
    return np.array(frames) if frames else np.array([0.0])


def estimate_snr(data: np.ndarray, sr: int) -> float:
    """
    Estimate SNR (dB) by comparing signal energy to noise floor.
    Noise floor = median of the quietest NOISE_PERCENTILE% of frames.
    """
    energies = frame_energies(data, sr, FRAME_SIZE_S, HOP_SIZE_S)
    noise_th  = np.percentile(energies, NOISE_PERCENTILE)
    signal_th = np.mean(energies)
    if noise_th <= 0:
        return 0.0
    snr = 20 * np.log10(signal_th / noise_th + 1e-9)
    return float(snr)


def parse_timestamp(filename: str) -> datetime | None:
    """
    Try to extract a datetime from the filename.
    Supports:  YYYYMMDD_HHMMSS   or   YYYY-MM-DD_HH-MM-SS
    Returns None if no timestamp found — file will still be processed
    but won't appear in time-series plots.
    """
    stem = Path(filename).stem
    patterns = [
        r"(\d{8})_(\d{6})",          # 20260601_143022
        r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})",  # 2026-06-01_14-30-22
    ]
    for pat in patterns:
        m = re.search(pat, stem)
        if m:
            try:
                raw = m.group(0).replace("-", "").replace("_", "")
                return datetime.strptime(raw, "%Y%m%d%H%M%S")
            except ValueError:
                continue
    return None


def activity_score(band_ratio: float, band_energy: float,
                   snr_db: float) -> float:
    """
    Combine band energy ratio, absolute band energy, and SNR into a
    single activity score in [0, 100].

    Logic:
      - band_ratio  (0–1):  how much of the signal is in insect frequencies
      - band_energy (raw):  absolute signal strength in the band
      - snr_db      (dB):   signal quality above noise floor

    Score is weighted: ratio * 50 + SNR contribution * 50, clamped to 100.
    This is intentionally simple and interpretable for gardeners.
    """
    # Normalise band energy to 0–1 using a soft sigmoid-like scale
    # (empirically tuned; adjust ENERGY_SCALE if your mic is very quiet/loud)
    ENERGY_SCALE = 1e-3
    norm_energy = float(np.tanh(band_energy * ENERGY_SCALE))

    # SNR contribution: 0 dB → 0, 20 dB → 1 (clamp)
    snr_contrib = float(np.clip(snr_db / 20.0, 0.0, 1.0))

    raw = (band_ratio * 0.4 + norm_energy * 0.3 + snr_contrib * 0.3) * 100.0
    return round(float(np.clip(raw, 0.0, 100.0)), 2)


def is_active(band_energy: float, noise_floor: float) -> bool:
    """Simple threshold: active if band energy > multiplier × noise floor."""
    return band_energy > (noise_floor * ACTIVITY_MULTIPLIER)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_folder(input_dir: str, output_dir: str) -> pd.DataFrame:
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_path.glob("*.wav")) + sorted(input_path.glob("*.WAV"))

    if not wav_files:
        print(f"[!] No WAV files found in {input_dir}")
        return pd.DataFrame()

    print(f"[+] Found {len(wav_files)} WAV file(s) in {input_dir}\n")

    records = []

    # First pass: collect all band energies to estimate a global noise floor
    print("[~] Estimating noise floor across all clips...")
    all_band_energies = []
    loaded_cache = {}

    for wav_path in wav_files:
        try:
            data, sr = load_wav(str(wav_path))
            if len(data) / sr < MIN_CLIP_DURATION_S:
                continue
            if sr != TARGET_SR:
                data = resample(data, sr, TARGET_SR)
                sr = TARGET_SR
            filtered = butter_bandpass(data, sr, BAND_LOW_HZ, BAND_HIGH_HZ,
                                       FILTER_ORDER)
            _, band_e = compute_band_energy(filtered, sr, BAND_LOW_HZ,
                                            BAND_HIGH_HZ)
            all_band_energies.append(band_e)
            loaded_cache[str(wav_path)] = (filtered, sr)
        except Exception as e:
            print(f"  [!] Could not load {wav_path.name}: {e}")

    if not all_band_energies:
        print("[!] No processable audio found.")
        return pd.DataFrame()

    # Noise floor = bottom NOISE_PERCENTILE% of band energies across all clips
    global_noise_floor = float(np.percentile(all_band_energies, NOISE_PERCENTILE))
    print(f"    Noise floor (band energy): {global_noise_floor:.4f}\n")

    # Second pass: score each clip
    print("[~] Scoring clips...")
    for wav_path in wav_files:
        cache_key = str(wav_path)
        if cache_key not in loaded_cache:
            continue

        filtered, sr = loaded_cache[cache_key]
        data_orig, _ = load_wav(cache_key)  # reload original for SNR
        if sr != TARGET_SR:
            data_orig = resample(data_orig, _, TARGET_SR)

        band_ratio, band_energy = compute_band_energy(filtered, sr,
                                                       BAND_LOW_HZ, BAND_HIGH_HZ)
        snr = estimate_snr(filtered, sr)
        score = activity_score(band_ratio, band_energy, snr)
        active = is_active(band_energy, global_noise_floor)
        timestamp = parse_timestamp(wav_path.name)

        record = {
            "filename":    wav_path.name,
            "timestamp":   timestamp,
            "duration_s":  round(len(filtered) / sr, 2),
            "band_ratio":  round(band_ratio, 4),
            "band_energy": round(band_energy, 6),
            "snr_db":      round(snr, 2),
            "activity_score": score,
            "active":      active,
        }
        records.append(record)

        status = "ACTIVE" if active else "quiet"
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "no timestamp"
        print(f"  {wav_path.name:35s}  [{ts_str}]  score={score:6.1f}  {status}")

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Aggregation + output
# ---------------------------------------------------------------------------

def save_outputs(df: pd.DataFrame, output_dir: str):
    output_path = Path(output_dir)

    # --- Per-clip CSV ---
    clip_csv = output_path / "activity_per_clip.csv"
    df.to_csv(clip_csv, index=False)
    print(f"\n[+] Per-clip results saved → {clip_csv}")

    # Only produce time-series outputs if we have timestamps
    has_ts = df["timestamp"].notna().any()

    if has_ts:
        df_ts = df[df["timestamp"].notna()].copy()
        df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"])
        df_ts = df_ts.sort_values("timestamp")

        # --- Hourly summary ---
        df_ts["hour"] = df_ts["timestamp"].dt.floor("h")
        hourly = df_ts.groupby("hour").agg(
            clip_count      = ("filename",       "count"),
            active_clips    = ("active",         "sum"),
            mean_score      = ("activity_score", "mean"),
            max_score       = ("activity_score", "max"),
            mean_snr_db     = ("snr_db",         "mean"),
        ).reset_index()
        hourly["active_ratio"] = (hourly["active_clips"] /
                                  hourly["clip_count"]).round(3)
        hourly["mean_score"]   = hourly["mean_score"].round(2)
        hourly["max_score"]    = hourly["max_score"].round(2)
        hourly["mean_snr_db"]  = hourly["mean_snr_db"].round(2)

        hourly_csv = output_path / "activity_hourly.csv"
        hourly.to_csv(hourly_csv, index=False)
        print(f"[+] Hourly summary saved   → {hourly_csv}")

        # --- Daily summary ---
        df_ts["date"] = df_ts["timestamp"].dt.date
        daily = df_ts.groupby("date").agg(
            clip_count      = ("filename",       "count"),
            active_clips    = ("active",         "sum"),
            mean_score      = ("activity_score", "mean"),
            max_score       = ("activity_score", "max"),
        ).reset_index()
        daily["active_ratio"] = (daily["active_clips"] /
                                 daily["clip_count"]).round(3)
        daily["mean_score"]   = daily["mean_score"].round(2)
        daily["max_score"]    = daily["max_score"].round(2)

        daily_csv = output_path / "activity_daily.csv"
        daily.to_csv(daily_csv, index=False)
        print(f"[+] Daily summary saved    → {daily_csv}")

        # --- Plots ---
        _plot_activity(df_ts, hourly, daily, output_path)

    else:
        print("\n[!] No timestamps found in filenames — skipping time-series "
              "outputs.\n    Rename files to YYYYMMDD_HHMMSS.wav for full output.")

        # Still save a simple bar chart by clip index
        _plot_no_timestamps(df, output_path)

    print(f"\n[✓] Done. All outputs in: {output_path.resolve()}")


def _plot_activity(df_ts, hourly, daily, output_path: Path):
    """Generate and save activity plots."""

    # -- Plot 1: Activity score over time (per clip) --
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("Oak Street Garden — Insect Activity Index", fontsize=14,
                 fontweight="bold")

    ax = axes[0]
    colors = ["#2ecc71" if a else "#bdc3c7" for a in df_ts["active"]]
    ax.bar(df_ts["timestamp"], df_ts["activity_score"], color=colors,
           width=pd.Timedelta(seconds=30), align="center")
    ax.set_ylabel("Activity Score (0–100)")
    ax.set_title("Per-clip Activity Score  (green = above threshold)")
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.autofmt_xdate(rotation=30)
    ax.grid(axis="y", alpha=0.3)

    # -- Plot 2: Hourly mean score --
    ax2 = axes[1]
    ax2.fill_between(hourly["hour"], hourly["mean_score"],
                     alpha=0.4, color="#3498db")
    ax2.plot(hourly["hour"], hourly["mean_score"],
             color="#2980b9", linewidth=2, marker="o", markersize=4)
    ax2.set_ylabel("Mean Activity Score")
    ax2.set_title("Hourly Mean Activity")
    ax2.set_ylim(0, 105)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax2.grid(alpha=0.3)

    # -- Plot 3: Daily active clip ratio --
    ax3 = axes[2]
    daily_dates = pd.to_datetime(daily["date"])
    ax3.bar(daily_dates, daily["active_ratio"] * 100,
            color="#e67e22", width=0.6)
    ax3.set_ylabel("Active Clips (%)")
    ax3.set_title("Daily Active Clip Percentage")
    ax3.set_ylim(0, 105)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / "activity_plots.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Activity plots saved   → {plot_path}")


def _plot_no_timestamps(df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#2ecc71" if a else "#bdc3c7" for a in df["active"]]
    ax.bar(range(len(df)), df["activity_score"], color=colors)
    ax.set_xlabel("Clip index")
    ax.set_ylabel("Activity Score (0–100)")
    ax.set_title("Oak Street Garden — Insect Activity Score per Clip")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plot_path = output_path / "activity_plots.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Activity plot saved    → {plot_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    global BAND_LOW_HZ, BAND_HIGH_HZ, ACTIVITY_MULTIPLIER

    parser = argparse.ArgumentParser(
        description="Insect Activity Index Pipeline — Team 893C"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to folder containing .wav files from the ESP32 node"
    )
    parser.add_argument(
        "--output", "-o", default="./results",
        help="Path to output folder for CSVs and plots (default: ./results)"
    )
    parser.add_argument(
        "--band_low", type=float, default=BAND_LOW_HZ,
        help=f"Low frequency cutoff in Hz (default: {BAND_LOW_HZ})"
    )
    parser.add_argument(
        "--band_high", type=float, default=BAND_HIGH_HZ,
        help=f"High frequency cutoff in Hz (default: {BAND_HIGH_HZ})"
    )
    parser.add_argument(
        "--threshold", type=float, default=ACTIVITY_MULTIPLIER,
        help=f"Activity threshold multiplier over noise floor (default: {ACTIVITY_MULTIPLIER})"
    )
    args = parser.parse_args()

    # Override globals with CLI args
    BAND_LOW_HZ          = args.band_low
    BAND_HIGH_HZ         = args.band_high
    ACTIVITY_MULTIPLIER  = args.threshold

    print("=" * 60)
    print("  Insect Activity Index Pipeline — Team 893C")
    print(f"  Band: {BAND_LOW_HZ}–{BAND_HIGH_HZ} Hz  |  "
          f"Threshold: {ACTIVITY_MULTIPLIER}× noise floor")
    print("=" * 60 + "\n")

    df = process_folder(args.input, args.output)

    if df.empty:
        return

    active_count = df["active"].sum()
    total        = len(df)
    mean_score   = df["activity_score"].mean()

    print(f"\n{'='*60}")
    print(f"  Summary: {active_count}/{total} clips active  |  "
          f"Mean score: {mean_score:.1f}/100")
    print(f"{'='*60}\n")

    save_outputs(df, args.output)


if __name__ == "__main__":
    main()