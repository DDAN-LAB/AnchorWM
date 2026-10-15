import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
NUM_SAMPLES = 8 * SAMPLE_RATE
DEFAULTS = {"none": 0, "noise": 20, "crop_front": 0.1,
            "crop_tail": 0.3, "stretch": 0.95, "resample": 8000}


def fit_length(wave, length=NUM_SAMPLES):
    wave = np.asarray(wave, dtype=np.float32)
    return np.pad(wave[:length], (0, max(0, length - len(wave))))


def read_audio(path):
    wave, sr = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Audio must be nonempty and finite.")
    if sr != SAMPLE_RATE:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=SAMPLE_RATE)
    return wave


def perturb(wave, kind="noise", strength=None, seed=0):
    """Noise strength is SNR in dB; crops use fractions; resample uses Hz."""
    if kind not in DEFAULTS:
        raise ValueError(f"Unknown attack: {kind}")
    value = DEFAULTS[kind] if strength is None else float(strength)
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Expected a nonempty, finite mono waveform.")
    if not np.isfinite(value):
        raise ValueError("Attack strength must be finite.")
    n = len(wave)
    if kind == "none":
        result = wave.copy()
    elif kind == "noise":
        noise = np.random.default_rng(seed).standard_normal(n)
        rms = np.sqrt(np.mean(wave.astype(np.float64) ** 2))
        noise *= rms * 10 ** (-value / 20) / np.sqrt(np.mean(noise ** 2))
        result = wave + noise
    elif kind in ("crop_front", "crop_tail"):
        if not 0 <= value < 1:
            raise ValueError("Crop fraction must be in [0, 1).")
        cut = int(n * value)
        result = wave[cut:] if kind == "crop_front" else wave[:n - cut]
    elif kind == "stretch":
        if value <= 0:
            raise ValueError("Stretch rate must be positive.")
        result = librosa.effects.time_stretch(wave, rate=value)
    else:
        if not 0 < value < SAMPLE_RATE or int(value) != value:
            raise ValueError("Resampling rate must be an integer between 1 and 15999 Hz.")
        down = librosa.resample(wave, orig_sr=SAMPLE_RATE, target_sr=int(value))
        result = librosa.resample(down, orig_sr=int(value), target_sr=SAMPLE_RATE)
    # Crops are padded at the tail; stretched audio is truncated or tail-padded.
    return fit_length(result, n)
