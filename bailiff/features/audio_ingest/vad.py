import logging

import numpy as np
import torch

logger = logging.getLogger("bailiff.audio.vad")


def _load_silero_pip():
    from silero_vad import load_silero_vad
    model = load_silero_vad()
    logger.info("Loaded Silero VAD via pip package")
    return model


def _load_silero_torch_hub():
    model, _utils = torch.hub.load(
        "snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
        onnx=False,
    )
    logger.info("Loaded Silero VAD via torch.hub (cached)")
    return model


def _load_silero_model():
    try:
        return _load_silero_pip()
    except Exception as pip_exc:
        logger.info("silero-vad pip package unavailable (%s); falling back to torch.hub", pip_exc)
        try:
            return _load_silero_torch_hub()
        except Exception as hub_exc:
            raise RuntimeError(
                "Silero VAD unavailable: pip package failed (%r) and torch.hub fallback failed (%r). "
                "Install `silero-vad` or pre-cache snakers4/silero-vad via torch.hub." % (pip_exc, hub_exc)
            ) from hub_exc


class VADEngine:
    def __init__(self, threshold: float = 0.6, sample_rate: int = 16000):
        self.model = _load_silero_model()
        self.threshold = threshold
        self.sample_rate = sample_rate

        logger.info("VAD engine ready: threshold=%.2f, sample_rate=%d", threshold, sample_rate)

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        audio_tensor = torch.from_numpy(audio_chunk.flatten())
        speech_prob = self.model(audio_tensor, self.sample_rate).item()
        return speech_prob > self.threshold
