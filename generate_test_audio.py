"""
generate_test_audio.py — Team 893C
====================================
Generates synthetic WAV files to test analyze.py before you have
real recordings from the ESP32.

Creates two types of clips:
  - "active" clips: sine wave at bee wingbeat frequency (234 Hz) + harmonics + noise
  - "quiet" clips: background noise only

Run:
    python generate_test_audio.py
    python analyze.py --input ./test_recordings --output ./test_results
"""

import os
import numpy as np
from scipy.io import wavfile
from datetime import datetime, timedelta
from pathlib import Path

SR = 16000       # 16 kHz — matches ESP32 target rate
DURATION = 2.5   # seconds per clip
BEE_FREQ = 234   # Hz — honey bee wingbeat fundamental

OUTPUT_DIR = Path("./test_recordings")
OUTPUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

def make_bee_clip(sr, duration, snr_db=12):
    """Synthetic bee wingbeat: fundamental + harmonics at 234 Hz."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # fundamental + 2nd + 3rd harmonic (typical insect wingbeat profile)
    signal  = 0.6  * np.sin(2 * np.pi * BEE_FREQ * t)
    signal += 0.25 * np.sin(2 * np.pi * BEE_FREQ * 2 * t)
    signal += 0.10 * np.sin(2 * np.pi * BEE_FREQ * 3 * t)
    # add wind/background noise
    noise_amp = 10 ** (-snr_db / 20)
    noise = rng.normal(0, noise_amp, len(t))
    clip = signal + noise
    clip /= np.max(np.abs(clip))  # normalise
    return clip.astype(np.float32)


def make_noise_clip(sr, duration):
    """Background noise: white noise + low-frequency rumble."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = rng.normal(0, 0.1, len(t))
    # add some low-frequency content (wind, traffic) outside insect band
    noise += 0.05 * np.sin(2 * np.pi * 60 * t)
    noise /= np.max(np.abs(np.abs(noise)) + 1e-9)
    noise *= 0.15  # keep quiet
    return noise.astype(np.float32)


# Generate 12 clips spread over a morning (6am–12pm) with timestamps
base_time = datetime(2026, 6, 1, 6, 0, 0)

clips = [
    # (offset_minutes, type)
    (0,   "noise"),
    (10,  "noise"),
    (20,  "bee"),   # morning activity starts
    (30,  "bee"),
    (40,  "bee"),
    (50,  "noise"),
    (60,  "bee"),   # peak mid-morning
    (70,  "bee"),
    (80,  "bee"),
    (90,  "noise"),
    (100, "noise"),
    (110, "bee"),
]

print(f"Generating {len(clips)} test clips in {OUTPUT_DIR}/\n")
for offset_min, clip_type in clips:
    ts = base_time + timedelta(minutes=offset_min)
    fname = ts.strftime("%Y%m%d_%H%M%S") + ".wav"
    fpath = OUTPUT_DIR / fname

    if clip_type == "bee":
        audio = make_bee_clip(SR, DURATION, snr_db=rng.integers(8, 18))
    else:
        audio = make_noise_clip(SR, DURATION)

    # wavfile expects int16
    wavfile.write(str(fpath), SR, (audio * 32767).astype(np.int16))
    print(f"  {fname}  [{clip_type}]")

print(f"\nDone. Run:\n  python analyze.py --input {OUTPUT_DIR} --output ./test_results")