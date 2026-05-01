# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.model_executor.models.config import Gemma4Config
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _make_vllm_config(
    *,
    cache_dtype: str = "nvfp4",
    backend: AttentionBackendEnum | None = None,
    head_dim: int = 256,
    global_head_dim: int = 512,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                head_dim=head_dim,
                global_head_dim=global_head_dim,
            )
        ),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        attention_config=SimpleNamespace(backend=backend),
    )


@pytest.mark.parametrize(
    "cache_dtype,device_capability,expected_backend",
    [
        ("nvfp4", DeviceCapability(12, 0), AttentionBackendEnum.FLASHINFER),
        ("nvfp4", DeviceCapability(12, 1), AttentionBackendEnum.FLASHINFER),
        ("fp8_e4m3", DeviceCapability(12, 0), AttentionBackendEnum.TRITON_ATTN),
        ("nvfp4", DeviceCapability(9, 0), AttentionBackendEnum.TRITON_ATTN),
    ],
)
def test_gemma4_heterogeneous_heads_force_single_backend(
    cache_dtype,
    device_capability,
    expected_backend,
):
    vllm_config = _make_vllm_config(cache_dtype=cache_dtype)

    with patch(
        "vllm.platforms.current_platform.get_device_capability",
        return_value=device_capability,
    ):
        Gemma4Config.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend == expected_backend


def test_gemma4_heterogeneous_heads_fallback_for_unsupported_flashinfer_head_dim():
    vllm_config = _make_vllm_config(head_dim=384, global_head_dim=512)

    with patch(
        "vllm.platforms.current_platform.get_device_capability",
        return_value=DeviceCapability(12, 0),
    ):
        Gemma4Config.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend == AttentionBackendEnum.TRITON_ATTN


def test_gemma4_heterogeneous_heads_respects_explicit_backend():
    vllm_config = _make_vllm_config(backend=AttentionBackendEnum.FLASH_ATTN)

    with patch(
        "vllm.platforms.current_platform.get_device_capability",
        return_value=DeviceCapability(12, 0),
    ):
        Gemma4Config.verify_and_update_config(vllm_config)

    assert vllm_config.attention_config.backend == AttentionBackendEnum.FLASH_ATTN
