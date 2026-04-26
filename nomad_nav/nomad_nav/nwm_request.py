"""Helpers for packing NoMaD->NWM ranking requests into ROS-friendly bytes."""

from __future__ import annotations

import io
from typing import Iterable

import numpy as np
from PIL import Image as PILImage


def pil_to_rgb_array(image: PILImage.Image) -> np.ndarray:
    """Convert a PIL image to an HWC uint8 RGB numpy array."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def serialize_ranking_request(
    request_id: int,
    context_images: Iterable[PILImage.Image],
    goal_image: PILImage.Image,
    sampled_actions: np.ndarray,
) -> bytes:
    """Serialize ranking inputs as a compressed NPZ payload."""
    context_arrays = [pil_to_rgb_array(img) for img in context_images]
    if not context_arrays:
        raise ValueError("At least one context image is required for NWM ranking.")

    actions = np.asarray(sampled_actions, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError(
            f"sampled_actions must have shape [num_samples, horizon, 2], got {actions.shape}."
        )

    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        request_id=np.asarray([request_id], dtype=np.int64),
        context_images=np.stack(context_arrays, axis=0),
        goal_image=pil_to_rgb_array(goal_image),
        sampled_actions=actions,
    )
    return buffer.getvalue()


def deserialize_ranking_request(payload: bytes) -> dict:
    """Deserialize a request produced by serialize_ranking_request."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {
            "request_id": int(np.asarray(data["request_id"]).reshape(-1)[0]),
            "context_images": np.asarray(data["context_images"], dtype=np.uint8),
            "goal_image": np.asarray(data["goal_image"], dtype=np.uint8),
            "sampled_actions": np.asarray(data["sampled_actions"], dtype=np.float32),
        }
