"""Tests for smithery.export modules."""

from __future__ import annotations

import pytest

from smithery.export.gguf import QUANTIZATION_LEVELS, export_gguf


class TestGGUFExport:
    def test_quantization_levels_exist(self) -> None:
        assert "q4_k_m" in QUANTIZATION_LEVELS
        assert "f16" in QUANTIZATION_LEVELS
        assert len(QUANTIZATION_LEVELS) == 16

    def test_invalid_quantization_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported quantization"):
            export_gguf("./model", quantize="invalid_quant")
