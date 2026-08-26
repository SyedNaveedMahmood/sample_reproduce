"""Machine-readable runtime and source provenance capture."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision


ENVIRONMENT_SCHEMA_VERSION = "sample_fg.environment.v1"


class EnvironmentCaptureError(RuntimeError):
    """Raised when required provenance cannot be captured faithfully."""


def _git(repo: Path, *args: str, binary: bool = False):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", *args],
        cwd=repo,
        text=not binary,
    )


def capture_git_state(repo: Path) -> dict[str, object]:
    """Capture commit/dirty state and a content-sensitive working diff hash."""

    repo = Path(repo).resolve(strict=True)
    commit = _git(repo, "rev-parse", "HEAD").strip()
    status = _git(repo, "status", "--porcelain=v1", "-z")
    diff = _git(repo, "diff", "HEAD", "--binary", binary=True)
    digest = hashlib.sha256()
    digest.update(status.encode("utf-8"))
    digest.update(diff)
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    for relative in sorted(item for item in untracked.split("\0") if item):
        path = repo / relative
        digest.update(relative.encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return {
        "project_commit": commit,
        "project_dirty": bool(status),
        "diff_hash": digest.hexdigest(),
    }


def _driver_version() -> str | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = output.splitlines()[0].strip() if output.splitlines() else ""
    return value or None


def _package_freeze() -> list[str]:
    values = {
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return sorted(values, key=str.casefold)


def capture_environment(
    *,
    project_repo: Path,
    coop_upstream_commit: str,
    dassl_commit: str,
    precision_mode: str,
    clip_backbone: str,
    clip_checkpoint_identifier: str,
    clip_checkpoint_sha256: str | None,
    device_index: int = 0,
    capture_package_freeze: bool = True,
) -> dict[str, Any]:
    """Return the doc-07 environment schema without inferring unavailable facts."""

    if not coop_upstream_commit or not dassl_commit:
        raise EnvironmentCaptureError("Pinned upstream commits are required")
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise EnvironmentCaptureError("device_index must be a nonnegative integer")

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        if device_index >= torch.cuda.device_count():
            raise EnvironmentCaptureError("Requested CUDA device is not visible")
        properties = torch.cuda.get_device_properties(device_index)
        gpu: dict[str, object] = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "driver_version": _driver_version(),
            "device_index": device_index,
        }
    else:
        gpu = {
            "name": None,
            "total_memory_bytes": None,
            "driver_version": None,
            "device_index": None,
            "unavailable_reason": "torch.cuda.is_available() is false",
        }

    runtime_version = None
    runtime_reason = (
        "PyTorch does not expose a distinct CUDA runtime version in this build"
        if cuda_available
        else "CUDA unavailable"
    )
    payload: dict[str, Any] = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "git": capture_git_state(project_repo),
        "upstream": {
            "coop_commit": coop_upstream_commit,
            "dassl_commit": dassl_commit,
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "os": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            "pytorch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
        },
        "cuda": {
            "available": cuda_available,
            "runtime_version": runtime_version,
            "runtime_version_unavailable_reason": runtime_reason,
            "torch_compiled_version": torch.version.cuda,
            "visible_device_count": torch.cuda.device_count() if cuda_available else 0,
            "visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "cudnn": {
            "version": torch.backends.cudnn.version() if cuda_available else None,
        },
        "gpu": gpu,
        "precision": {"mode": precision_mode},
        "clip": {
            "backbone": clip_backbone,
            "checkpoint_identifier": clip_checkpoint_identifier,
            "checkpoint_sha256": clip_checkpoint_sha256,
        },
        "package_freeze": (
            {"captured": True, "packages": _package_freeze()}
            if capture_package_freeze
            else {"captured": False, "reason": "disabled by resolved config"}
        ),
    }
    return payload
