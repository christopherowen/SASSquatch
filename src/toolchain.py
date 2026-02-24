#!/usr/bin/env python3
"""Toolchain detection and reporting helpers."""

import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, Optional


def _safe_run(cmd, timeout=5):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def get_tool_version(tool_path: str, args=None) -> Optional[str]:
    """Return a human-readable version line for a tool."""
    args = args or ["--version"]
    result = _safe_run([tool_path] + args)
    if result is None:
        return None
    text = (result.stdout or result.stderr or "").strip()
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def detect_ptx_isa_version(ptxas_path: str = "ptxas") -> Optional[str]:
    """
    Detect max PTX ISA version supported by this ptxas.

    Uses an intentionally unsupported `.version 99.9` and parses the
    reported "current version is '<x.y>'" value.
    """
    probe = ".version 99.9\n.target sm_80\n.address_size 64\n.visible .entry k(){ret;}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(probe)
        ptx_path = f.name
    cubin_path = ptx_path + ".cubin"
    try:
        result = _safe_run([ptxas_path, ptx_path, "-arch=sm_80", "-o", cubin_path], timeout=10)
        if result is None:
            return None
        text = f"{result.stdout or ''}\n{result.stderr or ''}"
        match = re.search(r"current version is '([0-9]+\.[0-9]+)'", text)
        if match:
            return match.group(1)
        # If probe unexpectedly compiled, fall back to known-good default.
        if result.returncode == 0:
            return "9.1"
        return None
    finally:
        for p in [ptx_path, cubin_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def resolve_ptx_version(ptxas_path: str = "ptxas", default: str = "9.1") -> str:
    """Resolve PTX version via detection, falling back to default."""
    detected = detect_ptx_isa_version(ptxas_path=ptxas_path)
    return detected or default


def collect_toolchain_versions(ptxas_path: str = "ptxas",
                               nvcc_path: str = "nvcc",
                               nvdisasm_path: str = "nvdisasm") -> Dict[str, str]:
    """Collect tool and runtime versions for result artifacts."""
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
        "ptx_isa_version": resolve_ptx_version(ptxas_path=ptxas_path),
    }
    versions["ptxas"] = get_tool_version(ptxas_path) or "not found"
    versions["nvcc"] = get_tool_version(nvcc_path) or "not found"
    versions["nvdisasm"] = get_tool_version(nvdisasm_path) or "not found"
    return versions
