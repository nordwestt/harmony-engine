"""HuBERT/MERT feature extractor (vendored from sanderwood/clamp3, MIT license)."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor

from harmony.embedding.backends.clamp3_lib.mert.MusicHubert import MusicHubertModel


class AudioBERTFeature(nn.Module):
    def __init__(
        self,
        pre_trained_folder: str,
        sample_rate: int,
        *,
        force_half: bool = False,
        disable_backprop: bool = True,
        processor_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.force_half = force_half
        self.processor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=sample_rate,
            padding_value=0.0,
            return_attention_mask=True,
            do_normalize=processor_normalize,
        )

    @torch.no_grad()
    def process_wav(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.processor(
            waveform,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
            padding=True,
        ).input_values[0]

    def forward(
        self,
        input_values: torch.Tensor,
        *,
        layer: int = -1,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if not self.force_half:
            out = self.model(input_values, output_hidden_states=True).hidden_states
        else:
            out = self.model(input_values.half(), output_hidden_states=True).hidden_states
            out = [o.float() for o in out]

        if layer is not None:
            out = out[layer]
        else:
            out = torch.stack(out)

        if reduction == "mean":
            return out.mean(-2)
        if reduction == "max":
            return out.max(-2)[0]
        if reduction == "none":
            return out
        raise NotImplementedError(f"Unknown reduction: {reduction}")


class HuBERTFeature(AudioBERTFeature):
    def __init__(
        self,
        pre_trained_folder: str,
        sample_rate: int,
        *,
        force_half: bool = False,
        disable_backprop: bool = True,
        processor_normalize: bool = True,
    ) -> None:
        super().__init__(
            pre_trained_folder,
            sample_rate,
            force_half=force_half,
            disable_backprop=disable_backprop,
            processor_normalize=processor_normalize,
        )
        self.model = MusicHubertModel.from_pretrained(pre_trained_folder)
        if disable_backprop:
            self.model.eval()
        if self.force_half:
            self.model.half()
        for param in self.model.parameters():
            param.requires_grad = False
