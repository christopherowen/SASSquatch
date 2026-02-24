#!/usr/bin/env python3
"""
SASSquatch: NVIDIA GPU ISA auditor

GPU instruction-space auditor for NVIDIA SM121.

Systematically probes the SM121 (GB10) instruction set to discover:
  - Supported vs unsupported PTX instructions
  - Architecture-specific features (SM121 vs SM100 vs SM90)
  - Undocumented instructions and type combinations
  - SASS opcode space coverage at the binary level

Phases:
  1  PTX compilation audit     -- test ptxas acceptance (no GPU execution required)
  2  SASS opcode discovery     -- analyze SASS instruction encoding
  3  SASS binary audit         -- enumerate opcodes on live GPU hardware

Usage:
    python sassquatch.py                        # Run phase 1 (PTX audit)
    python sassquatch.py --phase 1 2 3          # All phases
    python sassquatch.py --phase 3 --range 0 512  # Probe opcodes 0-511 only
    python sassquatch.py -v                      # Verbose + JSON artifact export

Run inside the vllm-dev container with CUDA toolkit available.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from src.artifact_paths import DEFAULT_ARTIFACT_DIR, canonical_scan_json_path, ensure_artifact_dir
from src.ptx_probe import (
    PTXProber, ProbeResult, ProbeOutcome,
    generate_all_probes, build_ptx_program,
)
from src.sass_probe import CubinBuilder, SASSProber, SASSProbeResult
from src.sass_reference import (
    BLACKWELL_INSTRUCTIONS, lookup_mnemonic, classify_discovered_opcode,
    get_blackwell_only_instructions, get_sm100_tmem_instructions,
    get_mxfp4_relevant_instructions, get_instruction_count,
)
from src.cuda_probe import (
    get_cuda_probe_kernels, compile_and_discover_with_hex,
)
from src.toolchain import collect_toolchain_versions


# ---------------------------------------------------------------------------
# ANSI terminal colors
# ---------------------------------------------------------------------------

class C:
    """ANSI color codes for terminal output."""
    RESET    = "\033[0m"
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    RED      = "\033[31m"
    GREEN    = "\033[32m"
    YELLOW   = "\033[33m"
    BLUE     = "\033[34m"
    MAGENTA  = "\033[35m"
    CYAN     = "\033[36m"
    WHITE    = "\033[37m"
    BG_RED   = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE  = "\033[44m"
    # 256-color
    GRAY     = "\033[38;5;245m"
    ORANGE   = "\033[38;5;208m"


# Disable colors if not a terminal
if not sys.stdout.isatty():
    for attr in dir(C):
        if not attr.startswith("_"):
            setattr(C, attr, "")


# ---------------------------------------------------------------------------
# Global state for clean shutdown
# ---------------------------------------------------------------------------

_interrupted = False

def _signal_handler(sig, frame):
    global _interrupted
    if _interrupted:
        print(f"\n{C.RED}Force quit.{C.RESET}")
        sys.exit(1)
    _interrupted = True
    print(f"\n{C.YELLOW}Interrupt received, finishing current probe...{C.RESET}")

signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

BANNER = f"""\
{C.CYAN}{C.BOLD}
  ███████╗ █████╗ ███████╗███████╗
  ██╔════╝██╔══██╗██╔════╝██╔════╝
  ███████╗███████║███████╗███████╗ {C.RESET}{C.WHITE}{C.BOLD}quatch{C.RESET}{C.CYAN}{C.BOLD}
  ╚════██║██╔══██║╚════██║╚════██║
  ███████║██║  ██║███████║███████║
  ╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝
{C.RESET}
  {C.DIM}NVIDIA GPU ISA auditing toolkit{C.RESET}
"""


def print_header(title: str):
    """Print a section header."""
    width = 72
    print(f"\n{C.CYAN}{'=' * width}{C.RESET}")
    print(f"  {C.BOLD}{title}{C.RESET}")
    print(f"{C.CYAN}{'=' * width}{C.RESET}")


def print_subheader(title: str):
    """Print a subsection header."""
    print(f"\n  {C.BLUE}{C.BOLD}{title}{C.RESET}")
    print(f"  {C.BLUE}{'-' * 60}{C.RESET}")


def progress_bar(current: int, total: int, width: int = 40, prefix: str = "") -> str:
    """Create a progress bar string."""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"{prefix}[{C.CYAN}{bar}{C.RESET}] {current}/{total} ({pct*100:.1f}%)"


# ---------------------------------------------------------------------------
# Phase 1: PTX Compilation Audit
# ---------------------------------------------------------------------------

def run_phase1(args):
    """Phase 1: Systematic PTX compilation audit."""
    print_header("PHASE 1: PTX Compilation Audit")
    print(f"  {C.DIM}Testing PTX instruction compilation across target architectures{C.RESET}")
    print(f"  {C.DIM}This discovers which instructions each GPU generation supports{C.RESET}")

    probes = generate_all_probes()
    targets = args.targets
    prober = PTXProber(targets=targets, verbose=args.verbose)

    print(f"\n  ptxas version: {C.WHITE}{prober.ptxas_version}{C.RESET}")
    print(f"  Targets: {C.WHITE}{', '.join(targets)}{C.RESET}")
    print(f"  Total probes: {C.WHITE}{len(probes)} instructions x {len(targets)} targets = {len(probes) * len(targets)}{C.RESET}")
    print()

    t_start = time.monotonic()
    last_update = 0

    # Category tracking for live display
    category_stats = {}

    def on_probe(name, target, outcome, progress, total):
        nonlocal last_update
        now = time.monotonic()

        cat = outcome.spec.category
        if cat not in category_stats:
            category_stats[cat] = {"pass": 0, "fail": 0}
        if outcome.result == ProbeResult.COMPILES:
            category_stats[cat]["pass"] += 1
        else:
            category_stats[cat]["fail"] += 1

        # Update display every 0.1s or on completion
        if now - last_update > 0.1 or progress == total:
            last_update = now
            elapsed = now - t_start
            rate = progress / elapsed if elapsed > 0 else 0

            status = f"{C.GREEN}OK{C.RESET}" if outcome.result == ProbeResult.COMPILES else f"{C.RED}FAIL{C.RESET}"
            sys.stdout.write(
                f"\r  {progress_bar(progress, total)}"
                f"  {rate:.0f}/s  "
                f"{C.DIM}{name:40s}{C.RESET} [{target}] {status}   "
            )
            sys.stdout.flush()

        if _interrupted:
            return  # Early exit handled by caller

    try:
        results = prober.run_compilation_audit(probes, callback=on_probe)
    except KeyboardInterrupt:
        pass

    elapsed = time.monotonic() - t_start
    print(f"\n\n  Completed in {elapsed:.1f}s")

    # --- Results Summary ---
    print_subheader("Compilation Results by Target")

    stats = prober.get_summary_stats()
    for target, s in stats.items():
        pct = s["compiles"] / s["total"] * 100 if s["total"] > 0 else 0
        bar_len = int(pct / 2)
        bar = f"{C.GREEN}{'█' * bar_len}{C.RESET}{C.DIM}{'░' * (50 - bar_len)}{C.RESET}"
        print(f"    {target:12s} {bar} {s['compiles']:>4d}/{s['total']:>4d} ({pct:.1f}%)")

    # --- Results by Category ---
    print_subheader("Results by Instruction Category")

    # Group results by category
    categories = {}
    for name, target_results in results.items():
        spec = next(iter(target_results.values())).spec
        cat = spec.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, target_results))

    primary_target = targets[0]

    for cat in sorted(categories.keys()):
        items = categories[cat]
        cat_pass = sum(
            1 for _, tr in items
            if primary_target in tr and tr[primary_target].result == ProbeResult.COMPILES
        )
        cat_total = len(items)
        print(f"\n    {C.BOLD}{cat.upper()}{C.RESET} ({cat_pass}/{cat_total} on {primary_target})")

        if args.verbose:
            for name, target_results in sorted(items, key=lambda x: x[0]):
                line = f"      {name:45s}"
                for t in targets:
                    if t in target_results:
                        o = target_results[t]
                        if o.result == ProbeResult.COMPILES:
                            line += f" {C.GREEN}✓{C.RESET}"
                        else:
                            line += f" {C.RED}✗{C.RESET}"
                    else:
                        line += f" {C.DIM}?{C.RESET}"
                print(line)

    # --- Anomaly Analysis ---
    print_subheader("Anomaly Analysis")
    anomalies = prober.find_anomalies()

    if anomalies["sm121_only"]:
        print(f"\n    {C.GREEN}{C.BOLD}SM121-ONLY instructions{C.RESET} (compile for SM121 but NOT SM100):")
        for name in anomalies["sm121_only"][:20]:
            print(f"      {C.GREEN}+{C.RESET} {name}")
        if len(anomalies["sm121_only"]) > 20:
            print(f"      ... and {len(anomalies['sm121_only']) - 20} more")

    if anomalies["sm121_missing_vs_sm100"]:
        print(f"\n    {C.ORANGE}{C.BOLD}SM100 instructions MISSING from SM121{C.RESET}:")
        for name in anomalies["sm121_missing_vs_sm100"][:20]:
            error_msg = ""
            if name in results and targets[0] in results[name]:
                error_msg = results[name][targets[0]].error_msg
                if error_msg:
                    error_msg = f" {C.DIM}({error_msg[:50]}){C.RESET}"
            print(f"      {C.ORANGE}-{C.RESET} {name}{error_msg}")
        if len(anomalies["sm121_missing_vs_sm100"]) > 20:
            print(f"      ... and {len(anomalies['sm121_missing_vs_sm100']) - 20} more")

    if anomalies["experimental_success"]:
        print(f"\n    {C.MAGENTA}{C.BOLD}EXPERIMENTAL instructions that COMPILED{C.RESET}:")
        for item in anomalies["experimental_success"]:
            if isinstance(item, tuple):
                name, compile_targets = item
                print(f"      {C.MAGENTA}!{C.RESET} {name} ({', '.join(compile_targets)})")
            else:
                print(f"      {C.MAGENTA}!{C.RESET} {item}")

    if anomalies["sm121_unique_vs_sm90"]:
        print(f"\n    {C.CYAN}SM121 features not in SM90{C.RESET}: {len(anomalies['sm121_unique_vs_sm90'])} instructions")

    # Summary counts
    print(f"\n    {C.BOLD}Summary:{C.RESET}")
    print(f"      Universal (all targets):    {len(anomalies['universal']):>4d}")
    print(f"      Universal fail:             {len(anomalies['universal_fail']):>4d}")
    print(f"      SM121-only:                 {len(anomalies.get('sm121_only', [])):>4d}")
    print(f"      SM100-only (missing SM121): {len(anomalies.get('sm121_missing_vs_sm100', [])):>4d}")
    print(f"      Experimental success:       {len(anomalies.get('experimental_success', [])):>4d}")

    return results, anomalies


# ---------------------------------------------------------------------------
# Phase 2: SASS Opcode Discovery
# ---------------------------------------------------------------------------

def run_phase2(args, phase1_results=None):
    """Phase 2: SASS opcode field analysis.

    If phase1_results is provided (from run_phase1), compilable probes are
    extracted directly instead of re-running compilation tests.
    """
    print_header("PHASE 2: SASS Opcode Field Discovery")
    print(f"  {C.DIM}Analyzing SASS instruction encoding to map opcode fields{C.RESET}")

    opcode_maps_by_target = {}
    for target in args.targets:
        try:
            builder = CubinBuilder(target=target)
        except RuntimeError as e:
            print(f"\n  {C.RED}Error: {e}{C.RESET}")
            print(f"  {C.DIM}Phase 2 requires nvcc and nvdisasm from the CUDA toolkit.{C.RESET}")
            return None

        print(f"\n  Target: {C.WHITE}{target}{C.RESET}")

        # Step 1: Compile template kernel
        print(f"  {C.DIM}Compiling template kernel...{C.RESET}")
        try:
            cubin_data = builder.compile_template()
            print(f"  {C.GREEN}Template compiled{C.RESET} ({len(cubin_data)} bytes)")
        except RuntimeError as e:
            print(f"  {C.RED}Compilation failed: {e}{C.RESET}")
            return None

        # Step 2: Parse ELF
        print(f"  {C.DIM}Parsing cubin ELF...{C.RESET}")
        cubin_info = builder.parse_cubin_elf(cubin_data)
        builder.extract_instructions(cubin_info)
        print(f"  {C.GREEN}Found .text section{C.RESET}: "
              f"kernel={cubin_info.kernel_name}, "
              f"offset=0x{cubin_info.text_section_offset:x}, "
              f"size={cubin_info.text_section_size} bytes, "
              f"{len(cubin_info.instructions)} instructions")

        # Step 3: Disassemble
        print(f"  {C.DIM}Disassembling...{C.RESET}")
        disasm = builder.disassemble(cubin_data)
        builder.annotate_from_disasm(cubin_info, disasm)
        if not any(i.mnemonic for i in cubin_info.instructions):
            disasm_raw = builder.disassemble_raw(cubin_data)
            builder.annotate_from_disasm(cubin_info, disasm_raw)

        # Display template instructions
        print_subheader(f"Template Kernel SASS Instructions ({target})")
        print(f"    {'Idx':>3s}  {'Offset':>6s}  {'Opcode[11:0]':>12s}  {'Mnemonic':12s}  {'Operands'}")
        print(f"    {'---':>3s}  {'------':>6s}  {'------------':>12s}  {'--------':12s}  {'--------'}")
        for i, inst in enumerate(cubin_info.instructions):
            opc = f"0x{inst.opcode_12bit:03x}"
            mnem = inst.mnemonic or "???"
            color = C.GREEN if inst.mnemonic else C.DIM
            print(f"    {i:3d}  0x{inst.offset:04x}  {opc:>12s}  {color}{mnem:12s}{C.RESET}  {inst.operands}")

        # Step 4: Discover opcode field from multiple instructions
        print(f"\n  {C.DIM}Compiling diverse instructions to map opcode field...{C.RESET}")

        # Create a SASSProber for discovery (no CUDA driver needed for this phase)
        # Give it the template cubin so it can include template opcodes
        dummy_prober = SASSProber(cuda_driver=None, cubin_builder=builder)
        dummy_prober.template_cubin = cubin_info
        opcode_map = dummy_prober.discover_opcode_field()

        # Step 5: Compile ALL successful Phase 1 probes to cubins for broader coverage

        # If Phase 1 already ran, extract compilable probes from its results.
        # Union across ALL targets so we get maximum SASS coverage.
        if phase1_results is not None:
            seen_names = set()
            compilable = []
            for name, target_results in phase1_results.items():
                for t, outcome in target_results.items():
                    if outcome.result == ProbeResult.COMPILES and name not in seen_names:
                        compilable.append(outcome.spec)
                        seen_names.add(name)
            print(f"\n  {C.DIM}Using {len(compilable)} compilable probes from Phase 1 (no re-compilation){C.RESET}")
        else:
            # Phase 1 wasn't run -- fall back to compiling from scratch
            all_probes = generate_all_probes()
            print(f"\n  {C.DIM}Phase 1 not run; compiling {len(all_probes)} probes to find compilable ones...{C.RESET}")

            compilable = []
            prober = PTXProber(targets=[target], verbose=False)
            for spec in all_probes:
                outcome = prober.probe_instruction(spec, target)
                if outcome.result == ProbeResult.COMPILES:
                    compilable.append(spec)

        print(f"  {C.WHITE}{len(compilable)}{C.RESET} compilable probes for SASS discovery")
        print(f"  {C.DIM}Compiling at -O0, -O1, -O3 to maximize opcode coverage{C.RESET}")

        t_start = time.monotonic()

        def on_probe_progress(progress, total, num_opcodes):
            now = time.monotonic()
            elapsed = now - t_start
            rate = progress / elapsed if elapsed > 0 else 0
            sys.stdout.write(
                f"\r  {progress_bar(progress, total)}"
                f"  {rate:.0f}/s  {C.GREEN}{num_opcodes} unique opcodes{C.RESET}   "
            )
            sys.stdout.flush()

        probe_opcodes = dummy_prober.discover_from_probes(
            compilable, build_ptx_program, target,
            callback=on_probe_progress,
            opt_levels=[0, 1, 3],
        )
        elapsed = time.monotonic() - t_start
        print(f"\n  Discovered {C.GREEN}{len(probe_opcodes)}{C.RESET} opcodes from PTX probes in {elapsed:.1f}s")

        # Merge into main opcode map
        for mnemonic, info in probe_opcodes.items():
            if mnemonic not in opcode_map:
                opcode_map[mnemonic] = info

        # Step 6: CUDA C++ probes for uniform datapath discovery
        print(f"\n  {C.DIM}Compiling CUDA C++ probes for uniform datapath instructions...{C.RESET}")
        cuda_kernels = get_cuda_probe_kernels()
        print(f"  {C.WHITE}{len(cuda_kernels)}{C.RESET} CUDA C++ probe kernels")

        t_cuda = time.monotonic()

        def on_cuda_progress(progress, total, num_opcodes, name, success):
            now = time.monotonic()
            elapsed = now - t_cuda
            rate = progress / elapsed if elapsed > 0 else 0
            status = f"{C.GREEN}OK{C.RESET}" if success else f"{C.RED}FAIL{C.RESET}"
            sys.stdout.write(
                f"\r  {progress_bar(progress, total)}"
                f"  {rate:.0f}/s  {C.GREEN}{num_opcodes} new opcodes{C.RESET}"
                f"  {name:30s} {status}   "
            )
            sys.stdout.flush()

        cuda_opcodes = compile_and_discover_with_hex(
            cuda_kernels, target,
            callback=on_cuda_progress,
        )
        elapsed_cuda = time.monotonic() - t_cuda
        new_from_cuda = sum(1 for m in cuda_opcodes if m not in opcode_map)
        print(f"\n  Discovered {C.GREEN}{len(cuda_opcodes)}{C.RESET} opcodes from CUDA C++ "
              f"({C.GREEN}{new_from_cuda} new{C.RESET}) in {elapsed_cuda:.1f}s")

        # Merge CUDA C++ opcodes into main map
        for mnemonic, info in cuda_opcodes.items():
            if mnemonic not in opcode_map:
                opcode_map[mnemonic] = info

        print_subheader(f"Discovered SASS Opcode Mapping ({target})")
        if opcode_map:
            print(f"    {'Mnemonic':16s}  {'bits[11:0]':>10s}  {'Documented?':12s}  {'Description'}")
            print(f"    {'--------':16s}  {'----------':>10s}  {'-----------':12s}  {'-----------'}")

            def _sort_key(item):
                try:
                    return (0, int(item[1]["bits_11_0"], 16))
                except (ValueError, TypeError):
                    return (1, item[0])  # sort unknowns alphabetically at end

            for mnemonic, info in sorted(opcode_map.items(), key=_sort_key):
                ref = lookup_mnemonic(mnemonic)
                if ref:
                    status = f"{C.GREEN}YES{C.RESET}"
                    desc = ref.description
                    if ref.notes:
                        desc += f" {C.DIM}({ref.notes}){C.RESET}"
                else:
                    status = f"{C.YELLOW}NO{C.RESET} "
                    desc = f"{C.YELLOW}Not in reference database{C.RESET}"
                print(f"    {mnemonic:16s}  {info['bits_11_0']:>10s}  {status:12s}       {desc}")
        else:
            print(f"    {C.RED}No opcodes discovered (compilation may have failed){C.RESET}")

        # Analyze opcode field
        opcodes_12 = set()
        opcodes_10 = set()
        for info in opcode_map.values():
            try:
                opcodes_12.add(int(info["bits_11_0"], 16))
            except (ValueError, TypeError):
                pass
            try:
                opcodes_10.add(int(info["bits_11_2"], 16))
            except (ValueError, TypeError):
                pass

        print(f"\n  {C.BOLD}Opcode field analysis:{C.RESET}")
        print(f"    Unique 12-bit low signatures: {len(opcodes_12)}")
        print(f"    Unique 10-bit low signatures: {len(opcodes_10)}")

        if len(opcodes_12) == len(opcode_map):
            print(f"    {C.GREEN}Observed mnemonics had unique low-12 signatures in this sample{C.RESET}")
        elif len(opcodes_10) == len(opcode_map):
            print(f"    {C.GREEN}Observed mnemonics had unique bits[11:2] signatures in this sample{C.RESET}")
        else:
            print(f"    {C.YELLOW}Collisions observed; instruction identity likely uses additional bits{C.RESET}")

        # Cross-reference with documented instruction set
        print_subheader("Reference Database Cross-Reference")
        ref_counts = get_instruction_count()
        print(f"    Documented Blackwell SASS instructions: {C.WHITE}{ref_counts.get('blackwell', 0)}{C.RESET}")

        discovered_base = set()
        for mnemonic in opcode_map:
            discovered_base.add(mnemonic.split(".")[0])

        documented_blackwell = set(
            k for k, v in BLACKWELL_INSTRUCTIONS.items()
            if "blackwell" in v.architectures
        )

        discovered_and_documented = discovered_base & documented_blackwell
        discovered_not_documented = discovered_base - documented_blackwell - {"NOP"}
        documented_not_discovered = documented_blackwell - discovered_base

        print(f"    Discovered & documented:   {C.GREEN}{len(discovered_and_documented)}{C.RESET}")
        print(f"    Discovered, NOT documented: {C.YELLOW}{len(discovered_not_documented)}{C.RESET}")
        if discovered_not_documented:
            for m in sorted(discovered_not_documented):
                print(f"      {C.YELLOW}?{C.RESET} {m}")
        print(f"    Documented, not yet discovered: {len(documented_not_discovered)}")
        print(f"    {C.DIM}(Only ~{len(opcode_map)} of ~{ref_counts.get('blackwell', 0)} instructions probed via template){C.RESET}")

        # Highlight MXFP4-relevant instructions
        mxfp4_instrs = get_mxfp4_relevant_instructions()
        sm100_tmem = get_sm100_tmem_instructions()

        print(f"\n    {C.BOLD}MXFP4-relevant instructions in reference:{C.RESET}")
        for k, v in sorted(mxfp4_instrs.items()):
            found = k in discovered_base or any(k in m for m in opcode_map)
            marker = f"{C.GREEN}FOUND{C.RESET}" if found else f"{C.DIM}not probed{C.RESET}"
            tmem = f" {C.ORANGE}[TMEM]{C.RESET}" if k in sm100_tmem else ""
            print(f"      {marker:>20s}  {k:16s} {v.description}{tmem}")

        opcode_maps_by_target[target] = opcode_map

    return opcode_maps_by_target


# ---------------------------------------------------------------------------
# Phase 3: SASS Binary Audit
# ---------------------------------------------------------------------------

def _build_known_opcodes(phase2_results):
    """Build sets of known 12-bit opcodes and mnemonics from Phase 2 data."""
    known_12bit = set()
    known_mnemonics = {}  # 12-bit opcode -> set of mnemonics

    if not phase2_results:
        return known_12bit, known_mnemonics

    for mnemonic, info in phase2_results.items():
        try:
            opc12 = int(info["bits_11_0"], 16)
            known_12bit.add(opc12)
            known_mnemonics.setdefault(opc12, set()).add(mnemonic.split(".")[0])
        except (ValueError, TypeError, KeyError):
            pass

    return known_12bit, known_mnemonics


def _disassemble_patched_cubin(builder, prober, opcode_value):
    """Disassemble a patched cubin to find the mnemonic for an opcode value."""
    import re
    patched = prober._patch_cubin(opcode_value)
    try:
        disasm = builder.disassemble(patched)
        # Parse the disassembly for the target instruction's mnemonic
        # The target instruction is at a known offset in the text section
        target = prober.template_cubin.instructions[prober.target_inst_idx]
        target_offset = target.offset

        # Look for instruction at the target offset in disassembly
        pattern = re.compile(
            r'/\*([0-9a-fA-F]+)\*/\s+'
            r'(?:@!?U?P\d+\s+)?'
            r'(\S+)'           # mnemonic
            r'(?:\s+.*?)?;'    # operands
        )
        for match in pattern.finditer(disasm):
            offset_hex = match.group(1)
            try:
                offset = int(offset_hex, 16)
            except ValueError:
                continue
            if offset == target_offset:
                return match.group(2)

        # Fallback: try nvdisasm -c format
        with tempfile.NamedTemporaryFile(
            suffix=".cubin", delete=False, prefix="squatch_id_"
        ) as f:
            f.write(patched)
            cubin_path = f.name
        try:
            result = subprocess.run(
                [builder.nvdisasm_path, "-c", cubin_path],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for match in pattern.finditer(result.stdout):
                    offset_hex = match.group(1)
                    try:
                        offset = int(offset_hex, 16)
                    except ValueError:
                        continue
                    if offset == target_offset:
                        return match.group(2)
        finally:
            os.unlink(cubin_path)

    except Exception:
        pass
    return None


def run_phase3(args, phase2_results=None):
    """Phase 3: SASS binary opcode enumeration on live GPU."""
    print_header("PHASE 3: SASS Binary Opcode Enumeration")
    print(f"  {C.DIM}Patching cubin instructions and testing execution on GPU hardware{C.RESET}")
    print(f"  {C.DIM}This is the GPU equivalent of sandsifter's x86 instruction scan{C.RESET}")

    # Import CUDA driver (needs GPU)
    try:
        from src.cuda_api import CUDADriver
    except ImportError:
        print(f"\n  {C.RED}Error: Could not import src.cuda_api module{C.RESET}")
        return None

    # Initialize CUDA
    print(f"\n  {C.DIM}Initializing CUDA...{C.RESET}")
    try:
        cuda = CUDADriver()
        gpu = cuda.gpu_info
        print(f"  {C.GREEN}GPU: {gpu}{C.RESET}")
    except Exception as e:
        print(f"  {C.RED}CUDA initialization failed: {e}{C.RESET}")
        print(f"  {C.DIM}Phase 3 requires a GPU. Run inside the Docker container.{C.RESET}")
        return None

    # Determine range
    start = args.range[0] if args.range else 0
    end = args.range[1] if args.range else 4096
    total = end - start

    phase3_by_target = {}
    for target in args.targets:
        print(f"\n  Target: {C.WHITE}{target}{C.RESET}")
        try:
            builder = CubinBuilder(target=target)
        except RuntimeError as e:
            print(f"\n  {C.RED}Error: {e}{C.RESET}")
            return None

        prober = SASSProber(cuda_driver=cuda, cubin_builder=builder)

        # Build known opcode set from Phase 2 (per-target if available)
        p2_for_target = None
        if isinstance(phase2_results, dict) and phase2_results and target in phase2_results:
            p2_for_target = phase2_results[target]
        known_12bit, known_mnemonics = _build_known_opcodes(p2_for_target)
        if known_12bit:
            print(f"  {C.DIM}Phase 2 cross-reference: {len(known_12bit)} known low-12 signatures{C.RESET}")
        else:
            print(f"  {C.YELLOW}No Phase 2 data -- all active signatures will be reported as new{C.RESET}")

        # Prepare template
        print(f"  {C.DIM}Building template cubin...{C.RESET}")
        if not prober.prepare():
            print(f"  {C.RED}Failed to prepare template cubin{C.RESET}")
            return None

        info = prober.get_template_info()
        print(f"  {C.GREEN}Template ready{C.RESET}: {info['num_instructions']} instructions, "
              f"target instruction #{info['target_instruction']}")

        # Show target instruction
        if info["instructions"]:
            target_inst = info["instructions"][info["target_instruction"]]
            print(f"  Target: {C.WHITE}{target_inst['mnemonic']} {target_inst['operands']}{C.RESET}")
            print(f"  Encoding: {target_inst['encoding_lo']} | {target_inst['encoding_hi']}")
            print(f"  Current signature: {target_inst['opcode_12bit']}")

        print(f"\n  Probing low-12 signatures: {C.WHITE}0x{start:03x} - 0x{end-1:03x} ({total} values){C.RESET}")
        print(f"  {C.YELLOW}NOTE: Template uses a predicated-off patchpoint; this is primarily a decode scan.{C.RESET}")
        print()

        t_start = time.monotonic()
        last_update = 0
        result_counts = {"valid": 0, "illegal": 0, "wrong": 0, "fail": 0,
                         "other": 0, "new": 0}

        def on_probe(opcode, outcome, progress, total_probes):
            nonlocal last_update
            now = time.monotonic()

            is_new = False

            # Update counts
            if outcome.result == SASSProbeResult.VALID:
                result_counts["valid"] += 1
                if opcode not in known_12bit:
                    result_counts["new"] += 1
                    is_new = True
            elif outcome.result == SASSProbeResult.ILLEGAL_INSTRUCTION:
                result_counts["illegal"] += 1
            elif outcome.result == SASSProbeResult.WRONG_OUTPUT:
                result_counts["wrong"] += 1
                if opcode not in known_12bit:
                    result_counts["new"] += 1
                    is_new = True
            elif outcome.result == SASSProbeResult.LOAD_FAILED:
                result_counts["fail"] += 1
            else:
                result_counts["other"] += 1

            # Update display
            if now - last_update > 0.05 or progress == total_probes:
                last_update = now
                elapsed = now - t_start
                rate = progress / elapsed if elapsed > 0 else 0
                eta = (total_probes - progress) / rate if rate > 0 else 0

                if is_new:
                    rchar = f"{C.MAGENTA}★{C.RESET}"
                elif outcome.result == SASSProbeResult.VALID:
                    rchar = f"{C.GREEN}V{C.RESET}"
                elif outcome.result == SASSProbeResult.ILLEGAL_INSTRUCTION:
                    rchar = f"{C.DIM}.{C.RESET}"
                elif outcome.result == SASSProbeResult.WRONG_OUTPUT:
                    rchar = f"{C.YELLOW}W{C.RESET}"
                else:
                    rchar = f"{C.RED}E{C.RESET}"

                new_str = (f"  {C.MAGENTA}★{result_counts['new']}{C.RESET}"
                           if result_counts['new'] > 0 else "")

                sys.stdout.write(
                    f"\r  {progress_bar(progress, total_probes)}"
                    f"  {rate:.0f}/s  ETA {eta:.0f}s"
                    f"  opc=0x{opcode:03x} {rchar}"
                    f"  {C.GREEN}V:{result_counts['valid']}{C.RESET}"
                    f" {C.DIM}I:{result_counts['illegal']}{C.RESET}"
                    f" {C.YELLOW}W:{result_counts['wrong']}{C.RESET}"
                    f" {C.RED}E:{result_counts['fail']}{C.RESET}"
                    f"{new_str}"
                    f"   "
                )
                sys.stdout.flush()

        # Run enumeration
        try:
            results = prober.enumerate_opcodes(start=start, end=end, callback=on_probe)
        except KeyboardInterrupt:
            results = prober.results

        elapsed = time.monotonic() - t_start
        print(f"\n\n  Completed in {elapsed:.1f}s ({len(results)} signatures probed)")

        # --- Identify interesting signatures (valid/wrong) and disassemble ---
        interesting = [
            opcode for opcode, outcome in sorted(results.items())
            if outcome.result in (SASSProbeResult.VALID, SASSProbeResult.WRONG_OUTPUT)
        ]

        opcode_mnemonics = {}  # opcode -> mnemonic from disassembly
        if interesting:
            print(f"\n  {C.DIM}Disassembling {len(interesting)} active signatures to identify mnemonics...{C.RESET}")
            for i, opc in enumerate(interesting):
                mnemonic = _disassemble_patched_cubin(builder, prober, opc)
                if mnemonic:
                    opcode_mnemonics[opc] = mnemonic
                if (i + 1) % 50 == 0 or (i + 1) == len(interesting):
                    sys.stdout.write(f"\r  {C.DIM}  {i+1}/{len(interesting)} disassembled{C.RESET}  ")
                    sys.stdout.flush()
            print()
            # Persist mnemonic labels into outcomes for export/reporting.
            for opc, m in opcode_mnemonics.items():
                try:
                    results[opc].mnemonic = m
                except Exception:
                    pass

        # --- Results Summary ---
        summary = prober.get_opcode_summary()
        print_subheader(f"SASS Signature Space Results ({target})")
        print(f"    Total probed:       {summary['total_probed']}")
        print(f"    {C.GREEN}Decode-ok (completed): {summary['valid_count']}{C.RESET}")
        print(f"    {C.DIM}Illegal instruction:  {summary['illegal_count']}{C.RESET}")
        print(f"    {C.YELLOW}Wrong output:          {summary['wrong_output_count']}{C.RESET}")
        print(f"    Load failed:        {len(summary['load_failed'])}")
        print(f"    Other:              {len(summary['other'])}")

        # Cross-reference with Phase 2
        new_opcodes = [opc for opc in interesting if opc not in known_12bit]
        known_opcodes = [opc for opc in interesting if opc in known_12bit]

        if known_12bit:
            print(f"\n    {C.DIM}Phase 2 known:      {len(known_opcodes)} signatures (seen in compilation){C.RESET}")
            print(f"    {C.MAGENTA}{C.BOLD}NEW discoveries:    {len(new_opcodes)} signatures{C.RESET}")
        else:
            print(f"\n    {C.DIM}(No Phase 2 data for cross-reference){C.RESET}")

        # Detail table for all active signatures
        if interesting:
            print_subheader(f"Active Signatures Detail ({target})")
            print(f"    {'Sig':>8s}  {'Status':10s}  {'Phase 2':8s}  {'Mnemonic':20s}  {'Known as'}")
            print(f"    {'------':>8s}  {'------':10s}  {'-------':8s}  {'--------':20s}  {'--------'}")
            for opc in interesting:
                outcome = results[opc]
                status = "DECODE_OK" if outcome.result == SASSProbeResult.VALID else "WRONG"
                status_color = C.GREEN if outcome.result == SASSProbeResult.VALID else C.YELLOW
                is_new = opc not in known_12bit
                phase2_str = f"{C.MAGENTA}NEW{C.RESET}    " if is_new else f"{C.DIM}known{C.RESET}  "
                mnemonic = opcode_mnemonics.get(opc, "???")
                p2_names = known_mnemonics.get(opc, set())
                p2_str = ", ".join(sorted(p2_names)) if p2_names else "-"
                if is_new:
                    ref = lookup_mnemonic(mnemonic) if mnemonic != "???" else None
                    doc_str = f"  {C.DIM}(documented){C.RESET}" if ref else ""
                    print(f"    {C.MAGENTA}0x{opc:03x}{C.RESET}     "
                          f"{status_color}{status:10s}{C.RESET}"
                          f"{phase2_str}"
                          f"{C.MAGENTA}{mnemonic:20s}{C.RESET}"
                          f"{p2_str}{doc_str}")
                else:
                    print(f"    0x{opc:03x}     "
                          f"{status_color}{status:10s}{C.RESET}"
                          f"{phase2_str}"
                          f"{mnemonic:20s}"
                          f"{p2_str}")

        # New discovery summary (documented vs undocumented)
        if new_opcodes:
            print_subheader(f"NEW Signature Discoveries ({target}) ({len(new_opcodes)})")
            documented_new = []
            undocumented_new = []
            for opc in new_opcodes:
                mnemonic = opcode_mnemonics.get(opc, "???")
                ref = lookup_mnemonic(mnemonic) if mnemonic != "???" else None
                entry = {
                    "opcode": opc,
                    "mnemonic": mnemonic,
                    "result": results[opc].result,
                    "documented": ref is not None,
                    "description": ref.description if ref else None,
                }
                (documented_new if ref else undocumented_new).append(entry)

            if documented_new:
                print(f"    {C.WHITE}Documented but compiler-unreachable ({len(documented_new)}):{C.RESET}")
                for e in documented_new:
                    print(f"      0x{e['opcode']:03x}  {e['mnemonic']:20s}  {e['description']}")
            if undocumented_new:
                print(f"\n    {C.MAGENTA}{C.BOLD}Undocumented / unknown ({len(undocumented_new)}):{C.RESET}")
                for e in undocumented_new:
                    print(f"      {C.MAGENTA}0x{e['opcode']:03x}{C.RESET}  {e['mnemonic']:20s}")

        # Heatmap
        if results:
            print_subheader(f"Signature Space Heatmap ({target})")
            _print_opcode_heatmap(results, start, end, known_12bit)

        phase3_by_target[target] = results

    # Cross-target diff (high-level)
    if len(phase3_by_target) > 1:
        print_subheader("Cross-Target Differences (Decode Scan)")
        # Build per-target sets
        per = {}
        for t, res in phase3_by_target.items():
            valid = {opc for opc, o in res.items() if o.result == SASSProbeResult.VALID}
            illegal = {opc for opc, o in res.items() if o.result == SASSProbeResult.ILLEGAL_INSTRUCTION}
            per[t] = {"valid": valid, "illegal": illegal}

        # Show signatures that are valid on some targets but illegal on others
        all_opc = set().union(*(set(res.keys()) for res in phase3_by_target.values()))
        flip = []
        for opc in sorted(all_opc):
            states = tuple(
                "V" if opc in per[t]["valid"] else ("I" if opc in per[t]["illegal"] else "-")
                for t in args.targets
                if t in per
            )
            if len(set(states)) > 1:
                flip.append((opc, states))

        print(f"    Targets: {', '.join(phase3_by_target.keys())}")
        print(f"    Signature values with differing outcomes: {len(flip)}")
        for opc, states in flip[:50]:
            st = " ".join(states)
            print(f"      0x{opc:03x}: {st}")
        if len(flip) > 50:
            print(f"      ... and {len(flip) - 50} more")

    return phase3_by_target


def _print_opcode_heatmap(results, start, end, known_12bit=None):
    """Print a visual heatmap of the opcode space."""
    if known_12bit is None:
        known_12bit = set()

    # 64 opcodes per row, 64 rows = 4096 total
    row_size = 64
    for row_start in range(start, end, row_size):
        row_end = min(row_start + row_size, end)
        line = f"    0x{row_start:03x}: "
        for opc in range(row_start, row_end):
            if opc in results:
                r = results[opc].result
                is_new = opc not in known_12bit
                if r == SASSProbeResult.VALID and is_new:
                    line += f"{C.MAGENTA}★{C.RESET}"
                elif r == SASSProbeResult.WRONG_OUTPUT and is_new:
                    line += f"{C.MAGENTA}◆{C.RESET}"
                elif r == SASSProbeResult.VALID:
                    line += f"{C.GREEN}█{C.RESET}"
                elif r == SASSProbeResult.WRONG_OUTPUT:
                    line += f"{C.YELLOW}▓{C.RESET}"
                elif r == SASSProbeResult.ILLEGAL_INSTRUCTION:
                    line += f"{C.DIM}░{C.RESET}"
                elif r == SASSProbeResult.LOAD_FAILED:
                    line += f"{C.RED}x{C.RESET}"
                else:
                    line += f"{C.RED}?{C.RESET}"
            else:
                line += " "
        print(line)

    print(f"\n    Legend: {C.GREEN}█{C.RESET}=valid(known)  "
          f"{C.MAGENTA}★{C.RESET}=valid(NEW)  "
          f"{C.YELLOW}▓{C.RESET}=wrong(known)  "
          f"{C.MAGENTA}◆{C.RESET}=wrong(NEW)  "
          f"{C.DIM}░{C.RESET}=illegal  "
          f"{C.RED}x{C.RESET}=load fail")


# ---------------------------------------------------------------------------
# Results export
# ---------------------------------------------------------------------------

def export_results(filepath: str, phase1_results=None, phase2_results=None,
                   phase3_results=None, anomalies=None, args=None):
    """Export results to a JSON file."""
    output = {
        "tool": "sassquatch",
        "version": "1.33.7",
        "timestamp": datetime.now().isoformat(),
        "targets": args.targets if args else [],
        "phases_run": args.phase if args else [],
        "toolchain_versions": collect_toolchain_versions(),
    }

    if phase1_results:
        # Serialize PTX results
        ptx_data = {}
        for name, target_results in phase1_results.items():
            ptx_data[name] = {}
            for target, outcome in target_results.items():
                ptx_data[name][target] = {
                    "compiles": outcome.result == ProbeResult.COMPILES,
                    "error": outcome.error_msg,
                    "category": outcome.spec.category,
                    "tags": outcome.spec.tags,
                }
        output["phase1_ptx"] = ptx_data

    if anomalies:
        output["anomalies"] = {
            k: v if not isinstance(v, list) or not v or not isinstance(v[0], tuple)
            else [(str(a), str(b)) for a, b in v]
            for k, v in anomalies.items()
        }

    if phase2_results:
        # Phase 2 may be either a single opcode map or a dict of maps per target.
        if isinstance(phase2_results, dict) and phase2_results and all(
            isinstance(v, dict) for v in phase2_results.values()
        ) and any(k.startswith("sm_") for k in phase2_results.keys()):
            output["phase2_opcode_map_by_target"] = phase2_results
        else:
            output["phase2_opcode_map"] = phase2_results

    if phase3_results:
        # Phase 3 may be either a single results dict or a dict of results per target.
        def _serialize_one(results_dict):
            out = {}
            for opcode, outcome in results_dict.items():
                out[f"0x{opcode:03x}"] = {
                    "result": outcome.result.name,
                    "error_code": outcome.error_code,
                    "output_value": outcome.output_value,
                    "mnemonic": getattr(outcome, "mnemonic", "") or "",
                }
            return out

        if isinstance(phase3_results, dict) and phase3_results and any(
            k.startswith("sm_") for k in phase3_results.keys()
        ):
            output["phase3_sass_by_target"] = {
                t: _serialize_one(res) for t, res in phase3_results.items()
            }
        else:
            output["phase3_sass"] = _serialize_one(phase3_results)

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  {C.GREEN}Results saved to {filepath}{C.RESET}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SASSquatch: NVIDIA GPU ISA auditor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Phases:
  1  PTX compilation audit     (uses ptxas; no GPU execution required)
  2  SASS opcode discovery     (uses nvcc + nvdisasm)
  3  SASS binary enumeration   (needs GPU, patches live cubins)

Examples:
  %(prog)s                        Run phase 1 (default)
  %(prog)s --phase 1 2 3          Run all phases
  %(prog)s --phase 3 --range 0 512  Probe first 512 SASS opcodes
  %(prog)s -v                      Verbose output, save canonical artifacts
  %(prog)s --log custom.json       Save scan JSON with custom filename
"""
    )

    parser.add_argument(
        "--phase", nargs="+", type=int, default=[1],
        help="Phases to run (1=PTX, 2=SASS discovery, 3=SASS binary). Default: 1"
    )
    parser.add_argument(
        "--targets", nargs="+", default=["sm_121", "sm_121a", "sm_121f"],
        help="PTX target architectures. Default: sm_121 sm_121a sm_121f"
    )
    parser.add_argument(
        "--range", nargs=2, type=int, metavar=("START", "END"),
        help="SASS opcode range for phase 3 (default: 0 4096)"
    )
    parser.add_argument(
        "--artifact-dir", type=str, default=DEFAULT_ARTIFACT_DIR,
        help=f"Artifact directory. Default: {DEFAULT_ARTIFACT_DIR}"
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Scan JSON filename/path. If filename only, writes under --artifact-dir."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output (show individual instruction results)"
    )
    parser.add_argument(
        "--no-banner", action="store_true",
        help="Suppress the ASCII banner"
    )

    args = parser.parse_args()

    if not args.no_banner:
        print(BANNER)

    print(f"  {C.DIM}Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")
    print(f"  {C.DIM}Phases: {', '.join(str(p) for p in args.phase)}{C.RESET}")
    print(f"  {C.DIM}Targets: {', '.join(args.targets)}{C.RESET}")

    phase1_results = None
    phase2_results = None
    phase3_results = None
    anomalies = None

    try:
        if 1 in args.phase:
            phase1_results, anomalies = run_phase1(args)

        if _interrupted:
            print(f"\n{C.YELLOW}Interrupted after Phase 1.{C.RESET}")
        elif 2 in args.phase:
            phase2_results = run_phase2(args, phase1_results=phase1_results)

        if _interrupted:
            print(f"\n{C.YELLOW}Interrupted.{C.RESET}")
        elif 3 in args.phase:
            phase3_results = run_phase3(args, phase2_results=phase2_results)

    except Exception as e:
        print(f"\n{C.RED}Error: {e}{C.RESET}")
        import traceback
        traceback.print_exc()

    # Export results (default canonical artifact path)
    ensure_artifact_dir(args.artifact_dir)
    if args.log:
        log_path = Path(args.log)
        if not log_path.is_absolute() and log_path.parent == Path("."):
            log_path = Path(args.artifact_dir) / log_path.name
    else:
        log_path = canonical_scan_json_path(args.artifact_dir)

    if any([phase1_results, phase2_results, phase3_results, anomalies]):
        export_results(
            str(log_path),
            phase1_results=phase1_results,
            phase2_results=phase2_results,
            phase3_results=phase3_results,
            anomalies=anomalies,
            args=args,
        )

    # Final summary
    print_header("Audit Complete")
    if phase1_results:
        total = len(phase1_results)
        primary = args.targets[0]
        compiles = sum(
            1 for r in phase1_results.values()
            if primary in r and r[primary].result == ProbeResult.COMPILES
        )
        print(f"  Phase 1: {compiles}/{total} PTX instructions compile for {primary}")

    if phase2_results:
        if isinstance(phase2_results, dict) and any(k.startswith("sm_") for k in phase2_results.keys()):
            total = sum(len(v) for v in phase2_results.values() if isinstance(v, dict))
            print(f"  Phase 2: {total} mnemonic mappings across {len(phase2_results)} targets")
        else:
            print(f"  Phase 2: {len(phase2_results)} mnemonic mappings")

    if phase3_results:
        if isinstance(phase3_results, dict) and any(k.startswith("sm_") for k in phase3_results.keys()):
            for t, res in phase3_results.items():
                valid = sum(1 for o in res.values() if o.result == SASSProbeResult.VALID)
                wrong = sum(1 for o in res.values() if o.result == SASSProbeResult.WRONG_OUTPUT)
                illegal = sum(1 for o in res.values() if o.result == SASSProbeResult.ILLEGAL_INSTRUCTION)
                print(f"  Phase 3 ({t}): {valid} decode-ok + {illegal} illegal + {wrong} wrong out of {len(res)}")
        else:
            valid = sum(1 for o in phase3_results.values() if o.result == SASSProbeResult.VALID)
            wrong = sum(1 for o in phase3_results.values() if o.result == SASSProbeResult.WRONG_OUTPUT)
            illegal = sum(1 for o in phase3_results.values() if o.result == SASSProbeResult.ILLEGAL_INSTRUCTION)
            print(f"  Phase 3: {valid} decode-ok + {illegal} illegal + {wrong} wrong out of {len(phase3_results)}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
