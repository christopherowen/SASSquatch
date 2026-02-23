#!/usr/bin/env python3
"""
Generate a Markdown report from SASSquatch JSON output.

This focuses on what the tool can *defensibly* claim:
- For each target (sm_121, sm_121a, sm_121f), a complete scan of low-12
  signature values under the Phase 3 template patchpoint.
- Decode/issue outcomes (VALID == completed under predicated-off patchpoint).
- nvdisasm mnemonic label for the patched instruction (when available).
- Comparison against the reference mnemonic list (sass_reference.py).

Usage:
  python generate_report.py
  python generate_report.py --input artifacts/scan_results.json --output artifacts/scan_report.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

from src.artifact_paths import (
    DEFAULT_ARTIFACT_DIR,
    canonical_report_md_path,
    ensure_artifact_dir,
    resolve_scan_input_path,
)
from src.sass_reference import lookup_mnemonic


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _get_phase3_by_target(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "phase3_sass_by_target" in data:
        return data["phase3_sass_by_target"]
    if "phase3_sass" in data:
        # Back-compat: treat as single unnamed target
        return {"(single)": data["phase3_sass"]}
    return {}


def _get_phase2_by_target(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "phase2_opcode_map_by_target" in data:
        return data["phase2_opcode_map_by_target"]
    if "phase2_opcode_map" in data:
        return {"(single)": data["phase2_opcode_map"]}
    return {}


def _classify_mnemonic(mnemonic: str) -> Tuple[str, str]:
    """Return (status, description) for a mnemonic string."""
    if not mnemonic or mnemonic == "???":
        return ("unknown", "")
    ref = lookup_mnemonic(mnemonic)
    if ref is None:
        return ("undocumented", "")
    return ("documented", ref.description)


def render_markdown(data: Dict[str, Any]) -> str:
    ts = data.get("timestamp") or datetime.now().isoformat()
    targets = data.get("targets") or []

    p3 = _get_phase3_by_target(data)
    p2 = _get_phase2_by_target(data)

    lines = []
    lines.append("# SASSquatch SM121x Decode Scan Report")
    lines.append("")
    lines.append(f"- **Timestamp**: `{ts}`")
    if targets:
        lines.append(f"- **Targets**: `{', '.join(targets)}`")
    lines.append("")
    lines.append("## What this report is (and is not)")
    lines.append("")
    lines.append("- **What it is**: a *complete scan of low-12 signature values* (0x000–0xFFF) under a fixed template patchpoint for each target. The patchpoint is predicated-off at runtime, so results primarily reflect **decode/issue legality**, not instruction semantics.")
    lines.append("- **What it is not**: a complete map of the full 128-bit SM121x instruction encoding space. Many instructions share major encodings and require additional bits beyond low-12 to identify uniquely.")
    lines.append("")

    if not p3:
        lines.append("## No Phase 3 data found")
        return "\n".join(lines) + "\n"

    # Per-target summaries
    lines.append("## Per-target summary")
    lines.append("")
    for t, res in p3.items():
        counts = {"VALID": 0, "ILLEGAL_INSTRUCTION": 0, "WRONG_OUTPUT": 0, "LOAD_FAILED": 0, "OTHER": 0}
        for _, v in res.items():
            r = v.get("result", "")
            if r in counts:
                counts[r] += 1
            else:
                counts["OTHER"] += 1
        lines.append(f"### `{t}`")
        lines.append("")
        lines.append(f"- **Decode-ok (VALID)**: {counts['VALID']}")
        lines.append(f"- **Illegal instruction**: {counts['ILLEGAL_INSTRUCTION']}")
        lines.append(f"- **Wrong output**: {counts['WRONG_OUTPUT']}")
        lines.append(f"- **Load failed**: {counts['LOAD_FAILED']}")
        lines.append(f"- **Other**: {counts['OTHER']}")
        lines.append("")

    # Cross-target diffs (focus on VALID vs ILLEGAL_INSTRUCTION)
    if len(p3) > 1:
        lines.append("## Cross-target differences (VALID vs ILLEGAL)")
        lines.append("")
        all_sigs = set()
        for res in p3.values():
            all_sigs.update(res.keys())

        diffs = []
        for sig in sorted(all_sigs):
            states = []
            for t in p3.keys():
                r = p3[t].get(sig, {}).get("result", "-")
                if r == "VALID":
                    states.append("V")
                elif r == "ILLEGAL_INSTRUCTION":
                    states.append("I")
                else:
                    states.append("-")
            if len(set(states)) > 1:
                diffs.append((sig, states))

        lines.append(f"- **Signature values with differing outcomes**: {len(diffs)}")
        lines.append("")
        if diffs:
            header = " | ".join(["sig"] + [t for t in p3.keys()])
            sep = " | ".join(["---"] * (1 + len(p3)))
            lines.append(header)
            lines.append(sep)
            for sig, states in diffs[:200]:
                lines.append(" | ".join([f"`{sig}`"] + states))
            if len(diffs) > 200:
                lines.append("")
                lines.append(f"(Truncated; {len(diffs) - 200} more)")
        lines.append("")

    # Undocumented mnemonics among decode-ok signatures
    lines.append("## Undocumented mnemonics (among decode-ok signatures)")
    lines.append("")
    for t, res in p3.items():
        undocumented = []
        unknown = []
        for sig, v in res.items():
            if v.get("result") != "VALID":
                continue
            m = (v.get("mnemonic") or "").strip()
            status, _ = _classify_mnemonic(m)
            if status == "undocumented":
                undocumented.append((sig, m))
            elif status == "unknown":
                unknown.append(sig)

        lines.append(f"### `{t}`")
        lines.append("")
        lines.append(f"- **Undocumented decode-ok signatures**: {len(undocumented)}")
        lines.append(f"- **Decode-ok but no mnemonic**: {len(unknown)}")
        lines.append("")
        if undocumented:
            lines.append("sig | mnemonic")
            lines.append("--- | ---")
            for sig, m in sorted(undocumented)[:200]:
                lines.append(f"`{sig}` | `{m}`")
            if len(undocumented) > 200:
                lines.append("")
                lines.append(f"(Truncated; {len(undocumented) - 200} more)")
            lines.append("")
        if unknown:
            lines.append("Decode-ok but no mnemonic (first 50):")
            lines.append("")
            lines.append(", ".join(f"`{s}`" for s in sorted(unknown)[:50]))
            lines.append("")

    # Phase2 vs Phase3: "compiler-emitted" coverage (low-12 signatures)
    if p2:
        lines.append("## Compiler-emitted vs active (Phase 2 vs Phase 3)")
        lines.append("")
        for t in p3.keys():
            p2_map = p2.get(t) or p2.get("(single)") or {}
            known = set()
            for _, info in p2_map.items():
                if isinstance(info, dict) and "bits_11_0" in info:
                    known.add(info["bits_11_0"])
            active = {sig for sig, v in p3[t].items() if v.get("result") == "VALID"}
            known_active = [sig for sig in active if sig in known]
            new_active = [sig for sig in active if sig not in known]
            lines.append(f"### `{t}`")
            lines.append("")
            lines.append(f"- **Active (decode-ok) signatures**: {len(active)}")
            lines.append(f"- **Also seen in compilation (Phase 2)**: {len(known_active)}")
            lines.append(f"- **Not seen in compilation (Phase 2)**: {len(new_active)}")
            lines.append("")

    lines.append("## Notes / limitations")
    lines.append("")
    lines.append("- A `VALID` result here means the kernel completed with the predicated-off patchpoint; it is best interpreted as **decode/issue accepted** for this template encoding family.")
    lines.append("- A `WRONG_OUTPUT` under predicated-off conditions is suspicious and should be investigated (could indicate patchpoint selection issues or unexpected side effects).")
    lines.append("- `LOAD_FAILED` can be driver/toolchain validation rejecting the modified binary, not necessarily an ISA decode property.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR, help="Artifact directory")
    ap.add_argument("--input", required=False, help="Input scan JSON path")
    ap.add_argument("--output", required=False, help="Output markdown path")
    ap.add_argument("--stdout", action="store_true", help="Print markdown to stdout")
    args = ap.parse_args()

    input_path = resolve_scan_input_path(args.input, args.artifact_dir)
    data = _load(input_path)
    md = render_markdown(data)

    if args.stdout:
        print(md, end="")
        return 0

    if args.output:
        output_path = Path(args.output)
    else:
        ensure_artifact_dir(args.artifact_dir)
        output_path = canonical_report_md_path(args.artifact_dir)

    output_path.write_text(md)
    print(f"Report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

