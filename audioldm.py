from contextlib import nullcontext

import librosa
import numpy as np
import torch
from diffusers import AudioLDMPipeline, DDIMScheduler
from tqdm.auto import tqdm

from attacks import NUM_SAMPLES, SAMPLE_RATE, fit_length, read_audio
from soft_dtw import SoftDTW
from watermark import LATENT_SHAPE


def ddim_transition(latent, noise, alpha_from, alpha_to):
    clean = (latent - (1 - alpha_from).sqrt() * noise) / alpha_from.sqrt()
    return alpha_to.sqrt() * clean + (1 - alpha_to).sqrt() * noise


class AudioLDM:
    def __init__(self, model="cvssp/audioldm", device="cuda", steps=50):
        self.device = torch.device(device)
        self.steps = steps
        self.pipe = AudioLDMPipeline.from_pretrained(model, torch_dtype=torch.float32).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        scheduler = self.pipe.scheduler
        if not 1 <= steps < scheduler.config.num_train_timesteps:
            raise ValueError("steps must be between 1 and num_train_timesteps - 1.")
        if scheduler.config.prediction_type != "epsilon" or scheduler.config.timestep_spacing != "leading":
            raise ValueError("This example requires AudioLDM's native epsilon/leading DDIM scheduler.")
        if (self.pipe.unet.config.in_channels != 8 or self.pipe.vae_scale_factor != 4
                or self.pipe.vocoder.config.model_in_dim != 64
                or self.pipe.vocoder.config.sampling_rate != SAMPLE_RATE):
            raise ValueError("This example supports the 16 kHz AudioLDM latent layout only.")
        for module in (self.pipe.unet, self.pipe.vae, self.pipe.text_encoder, self.pipe.vocoder):
            module.eval().requires_grad_(False)

    @torch.no_grad()
    def _embedding(self, prompt, guidance=False):
        return self.pipe._encode_prompt(prompt, self.device, 1, guidance)

    @torch.no_grad()
    def _noise(self, latent, timestep, embedding, guidance_scale=1.0):
        model_input = torch.cat([latent, latent]) if guidance_scale > 1 else latent
        model_input = self.pipe.scheduler.scale_model_input(model_input, timestep)
        noise = self.pipe.unet(model_input, timestep, encoder_hidden_states=None,
                               class_labels=embedding).sample
        if guidance_scale > 1:
            empty, text = noise.chunk(2)
            noise = empty + guidance_scale * (text - empty)
        return noise

    @torch.no_grad()
    def _denoise(self, initial_noise, prompt, guidance_scale, track=False):
        if tuple(initial_noise.shape) != LATENT_SHAPE:
            raise ValueError(f"Expected initial noise of shape {LATENT_SHAPE}.")
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(self.steps, device=self.device)
        latent = initial_noise.to(self.device, torch.float32) * scheduler.init_noise_sigma
        embedding = self._embedding(prompt, guidance_scale > 1)
        states = [latent.cpu().clone()] if track else []
        for t in tqdm(scheduler.timesteps, desc="Anchor" if track else "Generate"):
            noise = self._noise(latent, t, embedding, guidance_scale)
            latent = scheduler.step(noise, t, latent, eta=0.0).prev_sample
            if track:
                states.append(latent.cpu().clone())
        return latent, states

    @torch.no_grad()
    def generate(self, initial_noise, prompt, guidance_scale=7.5):
        latent, _ = self._denoise(initial_noise, prompt, guidance_scale)
        mel = self.pipe.decode_latents(latent)
        wave = self.pipe.mel_spectrogram_to_waveform(mel)[0].cpu().numpy()
        return wave[:NUM_SAMPLES]

    def anchor_trajectory(self, initial_noise):
        """Store [z_T, ..., z_0], including the initial noise for the last reverse step."""
        _, states = self._denoise(initial_noise, "", 1.0, track=True)
        return torch.stack(states)

    @torch.no_grad()
    def encode_audio(self, path):
        wave = fit_length(read_audio(path))
        wave = wave - wave.mean()
        wave = wave / (np.max(np.abs(wave)) + 1e-8) * 0.5
        wave = torch.from_numpy(wave).to(self.device)
        spectrum = torch.stft(wave, n_fft=1024, hop_length=160, win_length=1024,
                              window=torch.hann_window(1024, device=self.device),
                              center=True, pad_mode="reflect", return_complex=True).abs()
        bank = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=1024, n_mels=64, fmin=0, fmax=8000)
        mel = torch.log((torch.from_numpy(bank).to(self.device) @ spectrum).clamp_min(1e-5))
        mel = mel.T[:800][None, None]
        # Posterior mean makes repeated extraction deterministic.
        return self.pipe.vae.encode(mel).latent_dist.mean * self.pipe.vae.config.scaling_factor

    def invert(self, latent, anchors=None, inner_steps=8, lr=0.01):
        """Empty-prompt DDIM inversion; inner_steps=0 disables anchor optimization.

        As in the source, epsilon is frozen within each reverse step. Only the
        latent is optimized, with summed channel-wise Soft-DTW and Adam decay.
        """
        if inner_steps < 0 or not np.isfinite(lr) or lr <= 0:
            raise ValueError("inner_steps must be nonnegative and lr must be positive.")
        if tuple(latent.shape) != LATENT_SHAPE:
            raise ValueError(f"Expected audio latent of shape {LATENT_SHAPE}.")
        if inner_steps and (anchors is None or tuple(anchors.shape) != (self.steps + 1, *LATENT_SHAPE)):
            raise ValueError("Expected steps + 1 anchor states, including initial noise.")
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(self.steps, device=self.device)
        embedding = self._embedding("")
        latent = latent.detach().to(self.device, torch.float32)
        sdtw = SoftDTW(use_cuda=self.device.type == "cuda", gamma=0.001)
        # Numba's kernels follow the active CUDA context, including on cuda:N.
        context = torch.cuda.device(latent.device) if latent.is_cuda else nullcontext()
        losses = []
        for i, t in enumerate(tqdm(scheduler.timesteps.flip(0), desc="Invert")):
            previous = int(t) - scheduler.config.num_train_timesteps // self.steps
            alpha_from = scheduler.alphas_cumprod[previous] if previous >= 0 else scheduler.final_alpha_cumprod
            alpha_from = alpha_from.to(self.device)
            alpha_to = scheduler.alphas_cumprod[t.cpu()].to(self.device)
            noise = self._noise(latent, t, embedding)
            if inner_steps:
                target = anchors[self.steps - i - 1].to(self.device, torch.float32).detach()
                latent = latent.detach().clone().requires_grad_(True)
                optimizer = torch.optim.Adam([latent], lr=lr * (1 - i / self.steps), weight_decay=0.01)
                step_losses = []
                for _ in range(inner_steps):
                    with context:
                        predicted = ddim_transition(latent, noise, alpha_from, alpha_to)
                        loss = sdtw(target[0], predicted[0]).sum()
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite Soft-DTW loss.")
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        step_losses.append(loss.item())
                losses.append(step_losses)
            # Recompute after the final Adam update; then release its graph.
            with torch.no_grad():
                latent = ddim_transition(latent, noise, alpha_from, alpha_to)
        if not torch.isfinite(latent).all():
            raise RuntimeError("Nonfinite recovered latent.")
        return latent.detach().cpu(), losses
