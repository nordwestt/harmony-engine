"""CLaMP3 SAAS inference constants (from sanderwood/clamp3 code/config.py)."""

from __future__ import annotations

CLAMP3_HIDDEN_SIZE = 768
TEXT_MODEL_NAME = "FacebookAI/xlm-roberta-base"
MAX_TEXT_LENGTH = 128

AUDIO_HIDDEN_SIZE = 768
AUDIO_NUM_LAYERS = 12
MAX_AUDIO_LENGTH = 128

PATCH_SIZE = 64
PATCH_LENGTH = 512
PATCH_NUM_LAYERS = 12
M3_HIDDEN_SIZE = 768

CLAMP3_LOAD_M3 = True

DEFAULT_WEIGHTS_FILENAME = (
    "weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base"
    "_t_length_128_a_size_768_a_layers_12_a_length_128"
    "_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
)

DEFAULT_WEIGHTS_URL = (
    "https://huggingface.co/sander-wood/clamp3/resolve/main/"
    + DEFAULT_WEIGHTS_FILENAME
)

DEFAULT_MERT_CHECKPOINT = "m-a-p/MERT-v1-95M"
MERT_SAMPLE_RATE = 24000
MERT_SLIDING_WINDOW_SEC = 5
MERT_SLIDING_OVERLAP_PERCENT = 0.0
