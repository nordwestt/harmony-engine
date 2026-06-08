"""CLaMP3 model classes (vendored from sanderwood/clamp3 code/utils.py, MIT license)."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModel, BertModel, PreTrainedModel
from transformers.models.bert.modeling_bert import BertConfig

from harmony.embedding.backends.clamp3_lib.constants import (
    CLAMP3_HIDDEN_SIZE,
    CLAMP3_LOAD_M3,
    M3_HIDDEN_SIZE,
    PATCH_SIZE,
    TEXT_MODEL_NAME,
)


class M3PatchEncoder(PreTrainedModel):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.patch_embedding = torch.nn.Linear(PATCH_SIZE * 128, M3_HIDDEN_SIZE)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.base = BertModel(config=config)

    def forward(
        self,
        input_patches: torch.Tensor,
        input_masks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        input_patches = torch.nn.functional.one_hot(input_patches, num_classes=128)
        input_patches = (
            input_patches.reshape(len(input_patches), -1, PATCH_SIZE * 128).type(torch.FloatTensor)
        )
        input_patches = self.patch_embedding(input_patches.to(self.device))
        return self.base(inputs_embeds=input_patches, attention_mask=input_masks)


class CLaMP3Model(PreTrainedModel):
    def __init__(
        self,
        audio_config: BertConfig,
        symbolic_config: BertConfig,
        *,
        text_model_name: str = TEXT_MODEL_NAME,
        hidden_size: int = CLAMP3_HIDDEN_SIZE,
        load_m3: bool = CLAMP3_LOAD_M3,
    ) -> None:
        super().__init__(symbolic_config)

        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.text_proj = torch.nn.Linear(self.text_model.config.hidden_size, hidden_size)
        torch.nn.init.normal_(self.text_proj.weight, std=0.02)

        self.symbolic_model = M3PatchEncoder(symbolic_config)
        self.symbolic_proj = torch.nn.Linear(M3_HIDDEN_SIZE, hidden_size)
        torch.nn.init.normal_(self.symbolic_proj.weight, std=0.02)

        self.audio_model = BertModel(audio_config)
        self.audio_proj = torch.nn.Linear(audio_config.hidden_size, hidden_size)
        torch.nn.init.normal_(self.audio_proj.weight, std=0.02)

        _ = load_m3

    def avg_pooling(
        self,
        input_features: torch.Tensor,
        input_masks: torch.Tensor,
    ) -> torch.Tensor:
        input_masks = input_masks.unsqueeze(-1).to(self.device)
        input_features = input_features * input_masks
        return input_features.sum(dim=1) / input_masks.sum(dim=1)

    def get_text_features(
        self,
        text_inputs: torch.Tensor,
        text_masks: torch.Tensor,
        *,
        get_global: bool = False,
    ) -> torch.Tensor:
        text_features = self.text_model(
            text_inputs.to(self.device),
            attention_mask=text_masks.to(self.device),
        )["last_hidden_state"]

        if get_global:
            text_features = self.avg_pooling(text_features, text_masks)
            text_features = self.text_proj(text_features)

        return text_features

    def get_audio_features(
        self,
        audio_inputs: torch.Tensor,
        audio_masks: torch.Tensor,
        *,
        get_global: bool = False,
    ) -> torch.Tensor:
        audio_features = self.audio_model(
            inputs_embeds=audio_inputs.to(self.device),
            attention_mask=audio_masks.to(self.device),
        )["last_hidden_state"]

        if get_global:
            audio_features = self.avg_pooling(audio_features, audio_masks)
            audio_features = self.audio_proj(audio_features)

        return audio_features


def build_clamp3_model() -> CLaMP3Model:
    """Construct CLaMP3 SAAS model with default architecture."""
    from harmony.embedding.backends.clamp3_lib.constants import (
        AUDIO_HIDDEN_SIZE,
        AUDIO_NUM_LAYERS,
        MAX_AUDIO_LENGTH,
        PATCH_LENGTH,
        PATCH_NUM_LAYERS,
    )

    audio_config = BertConfig(
        vocab_size=1,
        hidden_size=AUDIO_HIDDEN_SIZE,
        num_hidden_layers=AUDIO_NUM_LAYERS,
        num_attention_heads=AUDIO_HIDDEN_SIZE // 64,
        intermediate_size=AUDIO_HIDDEN_SIZE * 4,
        max_position_embeddings=MAX_AUDIO_LENGTH,
    )
    symbolic_config = BertConfig(
        vocab_size=1,
        hidden_size=M3_HIDDEN_SIZE,
        num_hidden_layers=PATCH_NUM_LAYERS,
        num_attention_heads=M3_HIDDEN_SIZE // 64,
        intermediate_size=M3_HIDDEN_SIZE * 4,
        max_position_embeddings=PATCH_LENGTH,
    )
    return CLaMP3Model(
        audio_config=audio_config,
        symbolic_config=symbolic_config,
        load_m3=False,
    )


def load_clamp3_checkpoint(model: CLaMP3Model, checkpoint_path: str) -> dict[str, Any]:
    """Load CLaMP3 weights from a .pth checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    return checkpoint
