#!/usr/bin/env python3
"""
Shared artifact naming and path resolution helpers.
"""

from pathlib import Path
from typing import Optional


DEFAULT_ARTIFACT_DIR = "artifacts"
DEFAULT_SCAN_JSON = "scan_results.json"
DEFAULT_REPORT_MD = "scan_report.md"


def ensure_artifact_dir(path: str) -> Path:
    """Create artifact directory if needed and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def canonical_scan_json_path(artifact_dir: str = DEFAULT_ARTIFACT_DIR) -> Path:
    """Canonical scan JSON artifact path."""
    return Path(artifact_dir) / DEFAULT_SCAN_JSON


def canonical_report_md_path(artifact_dir: str = DEFAULT_ARTIFACT_DIR) -> Path:
    """Canonical report markdown artifact path."""
    return Path(artifact_dir) / DEFAULT_REPORT_MD


def resolve_scan_input_path(
    explicit_input: Optional[str] = None,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
) -> Path:
    """
    Resolve scan input JSON path, supporting legacy filenames.

    Resolution order:
      1) explicit_input (if provided)
      2) artifacts/scan_results.json
      3) results.json (legacy)
      4) results_full.json (legacy)
      5) artifacts/scan_results.json (default target)
    """
    if explicit_input:
        return Path(explicit_input)

    candidates = [
        canonical_scan_json_path(artifact_dir),
        Path("results.json"),
        Path("results_full.json"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return canonical_scan_json_path(artifact_dir)
