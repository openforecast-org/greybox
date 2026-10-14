"""Tests for code quality (linting and formatting)."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def test_ruff_check():
    """Test that ruff check passes."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src/greybox/"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"ruff check failed:\n{result.stdout}\n{result.stderr}"
    )


def test_ruff_format():
    """Test that ruff format check passes."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", "src/greybox/"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"ruff format check failed:\n{result.stdout}\n{result.stderr}"
    )


def test_flake8():
    """Test that flake8 passes."""
    result = subprocess.run(
        [sys.executable, "-m", "flake8", "src/greybox/"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"flake8 check failed:\n{result.stdout}\n{result.stderr}"
    )


def test_mypy():
    """Test that mypy passes."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "src/greybox/", "--ignore-missing-imports"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"mypy check failed:\n{result.stdout}\n{result.stderr}"
    )
