"""Tests for the VLM runner helpers. No model, GPU, or dataset needed."""

import numpy as np

from offroad.vlm.runner import downscale, strip_thinking


def test_downscale_keeps_aspect_ratio():
    image = np.zeros((1200, 1920, 3), dtype=np.uint8)
    assert downscale(image, 1280).shape == (800, 1280, 3)


def test_downscale_leaves_small_images_alone():
    image = np.zeros((600, 960, 3), dtype=np.uint8)
    assert downscale(image, 1280) is image


def test_strip_thinking():
    assert strip_thinking("<think>a { b }</think>\n{\"x\": 1}") == "{\"x\": 1}"
    assert strip_thinking("{\"x\": 1}") == "{\"x\": 1}"
