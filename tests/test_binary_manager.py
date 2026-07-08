"""Unit tests for BinaryManager."""

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from autoantibiotic.io_utils import BinaryManager, safe_run_tool


class TestBinaryManagerCheckBinary:
    """Tests for BinaryManager.check_binary()."""

    def test_known_binary_found(self) -> None:
        """Python itself should always be on PATH."""
        bm = BinaryManager()
        assert bm.check_binary("python3") or bm.check_binary("python")

    def test_nonexistent_binary_returns_false(self) -> None:
        bm = BinaryManager()
        assert bm.check_binary("this_binary_does_not_exist_xyz") is False

    def test_with_explicit_path(self) -> None:
        bm = BinaryManager()
        python_path = __import__("sys").executable
        result = bm.check_binary("python", path=python_path)
        assert result is True

    def test_with_invalid_path(self) -> None:
        bm = BinaryManager()
        result = bm.check_binary("nonexistent", path="/invalid/path/binary")
        assert result is False


class TestBinaryManagerGetVersion:
    """Tests for BinaryManager.get_version()."""

    def test_version_of_known_binary(self) -> None:
        """A binary listed in _BINARIES should return a version string."""
        bm = BinaryManager()
        with patch.dict(bm._BINARIES, {"python": "python3"}):
            version = bm.get_version("python")
        assert "unknown" not in version
        assert len(version) > 0

    def test_unknown_binary_returns_unknown(self) -> None:
        bm = BinaryManager()
        with patch.object(bm, "_BINARIES", {"nonexistent": "nonexistent"}):
            version = bm.get_version("nonexistent")
        assert version == "unknown"


class TestBinaryManagerValidateAll:
    """Tests for BinaryManager.validate_all()."""

    def test_validate_all_returns_dict(self) -> None:
        bm = BinaryManager()
        result = bm.validate_all()
        assert isinstance(result, dict)
        for name in ("vina", "gnina", "obabel", "prepare_receptor"):
            assert name in result
            assert isinstance(result[name], bool)

    def test_validate_all_caches_results(self) -> None:
        bm = BinaryManager()
        bm.validate_all()
        assert "vina" in bm._cache


class TestSafeRunTool:
    """Tests for safe_run_tool()."""

    def test_safe_run_success(self) -> None:
        """A known-good command should succeed."""
        result = safe_run_tool(["python3", "--version"], timeout=10)
        assert result.returncode == 0
        assert "Python" in (result.stdout or "")

    def test_safe_run_failure_retries_once(self) -> None:
        """A command that fails should be retried at least once."""
        with patch("autoantibiotic.io_utils.run_tool") as mock_run:
            from autoantibiotic.io_utils import AutoAntibioticError
            mock_run.side_effect = [
                AutoAntibioticError("First failure"),
                MagicMock(returncode=0, stdout="ok", stderr=""),
            ]
            result = safe_run_tool(["nonexistent_cmd"])
            assert mock_run.call_count == 2

    def test_safe_run_double_failure_raises(self) -> None:
        """If both attempts fail, the exception should propagate."""
        with patch("autoantibiotic.io_utils.run_tool") as mock_run:
            from autoantibiotic.io_utils import AutoAntibioticError
            mock_run.side_effect = AutoAntibioticError("Always fails")
            with pytest.raises(AutoAntibioticError):
                safe_run_tool(["nonexistent_cmd"])
            assert mock_run.call_count == 2
