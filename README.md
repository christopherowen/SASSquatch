# SASSquatch

GPU instruction-set auditing tools for NVIDIA Blackwell (SM100/SM120/SM121).

SASSquatch systematically probes the GPU instruction set to discover what the hardware actually supports versus what is documented. It operates at both the PTX (virtual ISA) and SASS (native binary) levels.

## What It Does

### Phase 1: PTX Compilation Audit

Generates ~359 PTX instruction variants and tests compilation with `ptxas` across multiple target architectures (SM80, SM90, SM100, SM121). This reveals:

- Which instructions each GPU generation supports
- Architecture-specific instructions (SM121-only, SM100-only)
- Undocumented instruction/type combinations
- Experimental instructions accepted by `ptxas`

### Phase 2: SASS Opcode Discovery

Compiles diverse PTX programs, disassembles the resulting cubins with `nvdisasm`, and maps PTX instructions to their native SASS opcode encodings. Cross-references against the documented Blackwell instruction set to find gaps.

### Phase 3: SASS Binary Audit

This phase patches the opcode field of a compiled cubin and executes every possible opcode value on the live GPU. It classifies each result as:

- **Valid** -- instruction executed correctly
- **Illegal** -- GPU trapped (CUDA_ERROR_ILLEGAL_INSTRUCTION)
- **Wrong output** -- executed but produced incorrect results
- **Load failed** -- driver rejected the modified binary

Includes an opcode space heatmap visualization.

## Requirements

- **Python 3.8+** (stdlib only, no pip dependencies)
- **CUDA Toolkit** -- `ptxas`, `nvcc`, `nvdisasm` must be on PATH
- **NVIDIA GPU + driver** -- Phase 3 requires a live GPU
- Tested with CUDA 13.1 (PTX ISA 9.1) on SM121 (DGX Spark / GB10)

## Usage

```bash
# Phase 1 only (no GPU execution required; uses ptxas)
python sassquatch.py

# All three phases
python sassquatch.py --phase 1 2 3

# Phase 3: probe a specific opcode range
python sassquatch.py --phase 3 --range 0 512

# Verbose output + JSON export
python sassquatch.py -v

# Custom artifact directory and filename
python sassquatch.py --artifact-dir out --log custom_scan.json

# Custom target architectures
python sassquatch.py --targets sm_121a sm_100a sm_90a
```

### Running in Docker

SASSquatch was developed for use inside a CUDA container:

```bash
docker exec -it <container> python3 /path/to/sassquatch.py --phase 1 2
```

## Artifact Conventions

SASSquatch writes artifacts under `artifacts/` by default.

- `artifacts/scan_results.json` -- canonical JSON output from `sassquatch.py`
- `artifacts/scan_report.md` -- canonical Markdown report from `generate_report.py`

Legacy files such as `results.json` and `results_full.json` are still accepted
as input by analysis/reporting utilities for compatibility.

```bash
# Build/update the canonical markdown report from latest scan artifact
python generate_report.py
```

## File Structure

| File | Description |
|------|-------------|
| `sassquatch.py` | Main CLI entry point and phase orchestrator |
| `ptx_probe.py` | PTX instruction generator and compilation tester (Phase 1) |
| `sass_probe.py` | SASS binary parser, patcher, and opcode enumerator (Phase 2 & 3) |
| `cubin_utils.py` | Shared cubin ELF parsing, disassembly, and patch helpers |
| `artifact_paths.py` | Shared artifact naming and path resolution helpers |
| `sass_reference.py` | Documented Blackwell SASS instruction database |
| `cuda_api.py` | Minimal CUDA Driver API bindings via ctypes |
| `generate_report.py` | Render Markdown reports from JSON scan output |
| `investigate_unknown_opcodes.py` | Analyze unknown or unlabeled opcode signatures |
| `label_phase3_opcodes.py` | Label Phase 3 opcode signatures via `nvdisasm` |
| `setctaid_poc.py` | Focused proof-of-concept for `SETCTAID.X` behavior |
| `sm121a_opcode_map.md` | Detailed SM121a encoding analysis and findings |

## References

- [PTX ISA Specification (v9.1)](https://docs.nvidia.com/cuda/parallel-thread-execution/) -- official PTX instruction reference
- [PTX ISA Release Notes](https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes) -- new instructions added per PTX/CUDA version
- [SASS Instruction Set Reference (Blackwell)](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference) -- native GPU instruction documentation
