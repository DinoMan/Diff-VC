"""DiffVC: Diffusion-based voice conversion."""

import json
import os
import numpy as np
import torch
import librosa
from librosa.filters import mel as librosa_mel_fn
from huggingface_hub import hf_hub_download
from pathlib import Path

from . import params
from .model import DiffVC as DiffVCModel


mel_basis = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)


class DiffVC:
    """DiffVC inference wrapper."""

    def __init__(self, device="cuda", checkpoint_path=None, checkpoint_dir=None):
        self.device = device
        self.sr = params.sampling_rate

        if checkpoint_dir:
            checkpoint_path = checkpoint_path or os.path.join(checkpoint_dir, "vc/vc_libritts_wodyn.pt")
            hfg_config_path = os.path.join(checkpoint_dir, "vocoder/config.json")
            hfg_ckpt_path = os.path.join(checkpoint_dir, "vocoder/generator")
            spk_ckpt_path = os.path.join(checkpoint_dir, "spk_encoder/pretrained.pt")
        else:
            # Download or use provided checkpoint
            if checkpoint_path is None:
                try:
                    checkpoint_path = hf_hub_download("DinoMan/Diff-VC", "vc_libritts_wodyn.pt")
                except Exception:
                    raise FileNotFoundError(
                        "DiffVC checkpoints not found. Please provide checkpoint_dir or upload to DinoMan/Diff-VC on HuggingFace."
                    )
            hfg_config_path = hf_hub_download("DinoMan/Diff-VC", "vocoder/config.json")
            hfg_ckpt_path = hf_hub_download("DinoMan/Diff-VC", "vocoder/generator")
            spk_ckpt_path = hf_hub_download("DinoMan/Diff-VC", "spk_encoder/pretrained.pt")

        # Load DiffVC model
        self.generator = DiffVCModel(
            params.n_mels, params.channels, params.filters, params.heads,
            params.layers, params.kernel, params.dropout, params.window_size,
            params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim,
            params.beta_min, params.beta_max,
        ).to(device)
        self.generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.generator.eval()

        # Load HiFi-GAN vocoder
        from .hifigan.env import AttrDict
        from .hifigan.models import Generator as HiFiGAN

        with open(hfg_config_path) as f:
            h = AttrDict(json.load(f))
        self.hifigan = HiFiGAN(h).to(device)
        self.hifigan.load_state_dict(torch.load(hfg_ckpt_path, map_location=device)['generator'])
        self.hifigan.eval()
        self.hifigan.remove_weight_norm()

        # Load speaker encoder
        from .speaker_encoder.encoder import inference as spk_encoder
        spk_encoder.load_model(Path(spk_ckpt_path), device=device)
        self.spk_encoder = spk_encoder

    def _get_mel(self, wav):
        """Compute log mel spectrogram."""
        wav = wav[:(wav.shape[0] // 256) * 256]
        wav = np.pad(wav, 384, mode='reflect')
        stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
        stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + 1e-9)
        mel_spectrogram = np.matmul(mel_basis, stftm)
        return np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))

    def _get_embed(self, wav):
        """Get speaker embedding."""
        wav_preprocessed = self.spk_encoder.preprocess_wav(wav, source_sr=self.sr)
        return self.spk_encoder.embed_utterance(wav_preprocessed)

    def _mel_spectral_subtraction(self, mel_synth, mel_source, spectral_floor=0.02, smoothing_window=1):
        mel_len = mel_source.shape[-1]
        energy_min = 1e10
        i_min = 0
        for i in range(mel_len - 5):
            energy_cur = np.sum(np.exp(2.0 * mel_source[:, i:i + 5]))
            if energy_cur < energy_min:
                i_min = i
                energy_min = energy_cur
        estimated_noise_energy = np.min(np.exp(2.0 * mel_synth[:, i_min:i_min + 5]), axis=-1)
        mel_denoised = np.copy(mel_synth)
        for i in range(mel_len):
            signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
            estimated_signal_energy = np.maximum(signal_subtract_noise, spectral_floor * estimated_noise_energy)
            mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
        return mel_denoised

    def convert(self, source_wav, target_wav, n_timesteps=30):
        """Convert source speech to target speaker's voice.

        Args:
            source_wav: numpy array at 22050Hz
            target_wav: numpy array at 22050Hz
            n_timesteps: diffusion steps

        Returns:
            (sample_rate, converted_audio_numpy)
        """
        mel_source = torch.from_numpy(self._get_mel(source_wav)).float().unsqueeze(0).to(self.device)
        mel_source_lengths = torch.LongTensor([mel_source.shape[-1]]).to(self.device)

        mel_target = torch.from_numpy(self._get_mel(target_wav)).float().unsqueeze(0).to(self.device)
        mel_target_lengths = torch.LongTensor([mel_target.shape[-1]]).to(self.device)

        embed_target = torch.from_numpy(self._get_embed(target_wav)).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            mel_encoded, mel_ = self.generator.forward(
                mel_source, mel_source_lengths, mel_target, mel_target_lengths,
                embed_target, n_timesteps=n_timesteps, mode='ml',
            )
            mel_synth_np = mel_.cpu().squeeze().numpy()
            mel_source_np = mel_source.cpu().squeeze().numpy()
            mel = torch.from_numpy(
                self._mel_spectral_subtraction(mel_synth_np, mel_source_np)
            ).float().unsqueeze(0).to(self.device)

            audio = self.hifigan.forward(mel).cpu().squeeze().clamp(-1, 1).numpy()

        return self.sr, audio
