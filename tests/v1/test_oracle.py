# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.engine.arg_utils import AsyncEngineArgs

MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def test_unsupported_configs():
    try:
        with pytest.raises(ValueError):
            AsyncEngineArgs(
                model=MODEL,
                speculative_config={
                    "model": MODEL,
                },
            ).create_engine_config()
    except OSError as exc:
        msg = str(exc)
        if ("gated repo" in msg.lower() or "not a valid model identifier" in msg
                or "repository not found" in msg.lower()):
            pytest.skip(f"Model repo not accessible in this environment: {MODEL}")
        raise
