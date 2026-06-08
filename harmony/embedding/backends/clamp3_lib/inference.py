"""CLaMP3 global feature extraction (adapted from clamp3 code/extract_clamp3.py)."""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from harmony.embedding.backends.clamp3_lib.constants import (
    AUDIO_HIDDEN_SIZE,
    MAX_AUDIO_LENGTH,
    MAX_TEXT_LENGTH,
)
from harmony.embedding.backends.clamp3_lib.model import CLaMP3Model


def _prepare_audio_segments(mert_features: torch.Tensor) -> torch.Tensor:
    input_data = mert_features.reshape(-1, mert_features.size(-1))
    zero_vec = torch.zeros((1, input_data.size(-1)), device=input_data.device)
    return torch.cat((zero_vec, input_data, zero_vec), dim=0)


def embed_audio_features(
    model: CLaMP3Model,
    mert_features: torch.Tensor,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Encode MERT features into a single global CLaMP3 vector."""
    input_data = _prepare_audio_segments(mert_features)
    max_input_length = MAX_AUDIO_LENGTH

    segment_list: list[torch.Tensor] = []
    for i in range(0, len(input_data), max_input_length):
        segment_list.append(input_data[i : i + max_input_length])
    segment_list[-1] = input_data[-max_input_length:]

    last_hidden_states_list: list[torch.Tensor] = []
    for input_segment in segment_list:
        input_masks = torch.tensor([1] * input_segment.size(0), device=device, dtype=torch.float32)
        pad_indices = torch.ones(
            (MAX_AUDIO_LENGTH - input_segment.size(0), AUDIO_HIDDEN_SIZE),
            device=device,
            dtype=torch.float32,
        )
        input_masks = torch.cat(
            (input_masks, torch.zeros(MAX_AUDIO_LENGTH - input_segment.size(0), device=device)),
            dim=0,
        )
        input_segment = torch.cat((input_segment.to(device), pad_indices), dim=0)

        last_hidden_states = model.get_audio_features(
            input_segment.unsqueeze(0),
            input_masks.unsqueeze(0),
            get_global=True,
        )
        last_hidden_states_list.append(last_hidden_states)

    full_chunk_cnt = len(input_data) // max_input_length
    remain_chunk_len = len(input_data) % max_input_length
    if remain_chunk_len == 0:
        feature_weights = torch.tensor(
            [max_input_length] * full_chunk_cnt,
            device=device,
            dtype=torch.float32,
        ).view(-1, 1)
    else:
        feature_weights = torch.tensor(
            [max_input_length] * full_chunk_cnt + [remain_chunk_len],
            device=device,
            dtype=torch.float32,
        ).view(-1, 1)

    last_hidden_states_list = torch.concat(last_hidden_states_list, dim=0)
    last_hidden_states_list = last_hidden_states_list * feature_weights
    return last_hidden_states_list.sum(dim=0) / feature_weights.sum()


def embed_text(
    model: CLaMP3Model,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    device: torch.device | str,
    max_length: int = MAX_TEXT_LENGTH,
) -> torch.Tensor:
    """Encode text into a global CLaMP3 vector."""
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        lines = [text]
    unique_lines = list(dict.fromkeys(lines))
    item = tokenizer.sep_token.join(unique_lines)
    input_data = tokenizer(item, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = input_data["input_ids"].squeeze(0)
    max_input_length = MAX_TEXT_LENGTH

    segment_list: list[torch.Tensor] = []
    for i in range(0, len(input_ids), max_input_length):
        segment_list.append(input_ids[i : i + max_input_length])
    segment_list[-1] = input_ids[-max_input_length:]

    last_hidden_states_list: list[torch.Tensor] = []
    for input_segment in segment_list:
        input_masks = torch.tensor([1] * input_segment.size(0), device=device, dtype=torch.long)
        pad_indices = torch.ones(
            MAX_TEXT_LENGTH - input_segment.size(0),
            device=device,
            dtype=torch.long,
        ) * tokenizer.pad_token_id
        input_masks = torch.cat(
            (input_masks, torch.zeros(MAX_TEXT_LENGTH - input_segment.size(0), device=device)),
            dim=0,
        )
        input_segment = torch.cat((input_segment.to(device), pad_indices), dim=0)

        last_hidden_states = model.get_text_features(
            input_segment.unsqueeze(0),
            input_masks.unsqueeze(0),
            get_global=True,
        )
        last_hidden_states_list.append(last_hidden_states)

    full_chunk_cnt = len(input_ids) // max_input_length
    remain_chunk_len = len(input_ids) % max_input_length
    if remain_chunk_len == 0:
        feature_weights = torch.tensor(
            [max_input_length] * full_chunk_cnt,
            device=device,
            dtype=torch.float32,
        ).view(-1, 1)
    else:
        feature_weights = torch.tensor(
            [max_input_length] * full_chunk_cnt + [remain_chunk_len],
            device=device,
            dtype=torch.float32,
        ).view(-1, 1)

    last_hidden_states_list = torch.concat(last_hidden_states_list, dim=0)
    last_hidden_states_list = last_hidden_states_list * feature_weights
    return last_hidden_states_list.sum(dim=0) / feature_weights.sum()
