from dataclasses import dataclass

import numpy as np
import torch
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes

LATENT_SHAPE = (1, 8, 200, 16)
PAYLOAD_SHAPE = (1, 4, 8, 8)
REPEATS = (1, 2, 25, 2)


@dataclass
class Watermark:
    bits: np.ndarray
    key: bytes
    nonce: bytes

    def __post_init__(self):
        bits = np.asarray(self.bits)
        if bits.size != 256 or not np.isin(bits, [0, 1]).all():
            raise ValueError("The payload must contain exactly 256 binary bits.")
        if len(self.key) != 32 or len(self.nonce) != 12:
            raise ValueError("Expected a 32-byte key and a 12-byte nonce.")
        self.bits = bits.astype(np.uint8).reshape(PAYLOAD_SHAPE)

    def _crypt(self, bits):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        packed = cipher.encrypt(np.packbits(bits).tobytes())
        return np.unpackbits(np.frombuffer(packed, dtype=np.uint8))

    @classmethod
    def create(cls, seed=0, bits=None):
        """Return a private record and initial noise; key/nonce are fresh each call."""
        rng = np.random.default_rng(seed)
        if bits is None:
            bits = rng.integers(0, 2, PAYLOAD_SHAPE, dtype=np.uint8)
        wm = cls(bits, get_random_bytes(32), get_random_bytes(12))
        signs = wm._crypt(np.tile(wm.bits, REPEATS)).reshape(LATENT_SHAPE)
        # A half-normal magnitude with a prescribed sign is a truncated Gaussian.
        magnitude = np.abs(rng.standard_normal(LATENT_SHAPE)).astype(np.float32)
        magnitude = np.maximum(magnitude, np.finfo(np.float32).tiny)
        noise = magnitude * (2 * signs.astype(np.float32) - 1)
        return wm, torch.from_numpy(noise)

    def extract(self, noise):
        """Decrypt latent signs and majority-vote the 100 copies of each bit."""
        noise = noise.detach().float().cpu().numpy()
        if noise.shape != LATENT_SHAPE or not np.isfinite(noise).all():
            raise ValueError(f"Expected finite recovered noise of shape {LATENT_SHAPE}.")
        repeated = self._crypt(noise > 0).reshape(2, 4, 25, 8, 2, 8)
        votes = repeated.sum(axis=(0, 2, 4))
        return (votes > 50).astype(np.uint8).reshape(PAYLOAD_SHAPE)

    def score(self, recovered):
        recovered = np.asarray(recovered).reshape(PAYLOAD_SHAPE)
        correct = int((recovered == self.bits).sum())
        return {
            "recovered_bits": "".join(map(str, recovered.flatten().tolist())),
            "correct_bits": correct,
            "payload_bits": 256,
            "bit_accuracy": correct / 256,
            "complete_recovery": correct == 256,
        }
