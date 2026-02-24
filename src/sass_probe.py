#!/usr/bin/env python3
"""
SASS binary-level instruction prober for SASSquatch.

This is the most sandsifter-like component: it operates at the native GPU
binary (SASS) level, systematically enumerating instruction opcodes and
testing them against the actual hardware.

Approach:
  1. Compile a template kernel to cubin (ELF binary for the GPU)
  2. Disassemble with nvdisasm to understand SASS instruction encoding
  3. Parse the cubin ELF to locate the .text section
  4. Identify the opcode field by analyzing known instruction encodings
  5. Enumerate all possible opcode values by patching the cubin
  6. Load each modified cubin and execute on the GPU
  7. Classify: valid instruction, illegal instruction, hang, or wrong output

This discovers instructions that exist in the hardware but may not be
exposed through PTX or documented by NVIDIA.
"""

import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from src.cubin_utils import disassemble_cubin, find_text_section
from src.toolchain import resolve_ptx_version


# ---------------------------------------------------------------------------
# ELF constants for cubin parsing
# ---------------------------------------------------------------------------

ELF_MAGIC = b"\x7fELF"

# ELF64 header offsets
EI_CLASS = 4
EI_DATA = 5
ELFCLASS64 = 2
ELFDATA2LSB = 1

# ELF64 header fields
E_SHOFF = 40      # Section header table offset
E_SHENTSIZE = 58  # Section header entry size
E_SHNUM = 60      # Number of section headers
E_SHSTRNDX = 62   # Section name string table index

# Section header fields
SH_NAME = 0
SH_TYPE = 4
SH_FLAGS = 8
SH_ADDR = 16
SH_OFFSET = 24
SH_SIZE = 32

# Section types
SHT_PROGBITS = 1
SHT_STRTAB = 3


# ---------------------------------------------------------------------------
# SASS instruction encoding
# ---------------------------------------------------------------------------

@dataclass
class SASSInstruction:
    """A decoded SASS instruction from nvdisasm output."""
    offset: int           # Byte offset in .text section
    mnemonic: str         # e.g., "MOV", "IADD3", "HMMA"
    operands: str         # Operand string
    encoding: bytes       # Raw 16-byte encoding (128-bit instruction)
    control_word: int     # Upper 64 bits (scheduling/control)
    instruction_word: int # Lower 64 bits (opcode + operands)

    @property
    def opcode_12bit(self) -> int:
        """Extract lower 12 bits as candidate opcode."""
        return self.instruction_word & 0xFFF

    @property
    def opcode_10bit(self) -> int:
        """Extract bits [11:2] as candidate opcode (some archs)."""
        return (self.instruction_word >> 2) & 0x3FF

    @property
    def opcode_field(self) -> int:
        """Extract the most likely opcode field."""
        # On SM8x/SM9x/SM10x/SM12x, opcode is typically bits [11:0]
        # of the instruction word (lower 64 bits of 128-bit instruction)
        return self.instruction_word & 0xFFF


@dataclass
class CubinInfo:
    """Parsed information about a cubin ELF binary."""
    raw_data: bytearray
    text_section_offset: int  # File offset of .text section
    text_section_size: int    # Size of .text section
    kernel_name: str          # Name of the kernel function
    instructions: List[SASSInstruction] = field(default_factory=list)


class SASSProbeResult(Enum):
    """Result of executing a patched SASS instruction."""
    VALID = auto()             # Instruction executed without error
    ILLEGAL_INSTRUCTION = auto()  # GPU trapped (CUDA_ERROR_ILLEGAL_INSTRUCTION)
    LAUNCH_FAILED = auto()     # Could not launch kernel
    LOAD_FAILED = auto()       # Could not load modified cubin
    WRONG_OUTPUT = auto()      # Executed but output differs from expected
    TIMEOUT = auto()           # Kernel appears to hang
    OTHER_ERROR = auto()       # Some other error


@dataclass
class SASSProbeOutcome:
    """Result of probing a single SASS opcode value."""
    opcode_value: int
    result: SASSProbeResult
    error_code: int = 0
    mnemonic: str = ""        # Mnemonic from nvdisasm (if available)
    output_value: int = 0     # Output value from kernel (if executed)


# ---------------------------------------------------------------------------
# Template kernel for SASS probing
# ---------------------------------------------------------------------------

TEMPLATE_KERNEL_CU = r"""\
// NOTE: Phase 3 correctness depends on the template being robust.
//
// We want to probe *decode legality* of candidate encodings without executing
// arbitrary side effects (branches, memory ops, etc.). To do that, we include
// a *predicated-off* instruction with a distinctive immediate (0x1337) and
// patch only the low-12 "signature" bits of that instruction's lower word.
//
// At runtime we always pass pred_mask=0, so predicate p is false and the
// instruction should not architecturally modify state. If the GPU traps with
// CUDA_ERROR_ILLEGAL_INSTRUCTION, that indicates a decode/issue failure even
// when predicated off.
extern "C" __global__ void squatch_kernel(unsigned int *out, unsigned int pred_mask) {
    unsigned int val = 42u;

    // Target instruction: predicated add immediate with unique marker 0x1337.
    // If pred_mask==0, p is false and the instruction should have no effect.
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.ne.u32 p, %1, 0;\n\t"
        "  @p add.u32 %0, %0, 0x1337;\n\t"
        "}\n"
        : "+r"(val)
        : "r"(pred_mask)
    );

    // Barrier to reduce reordering around the patchpoint.
    __syncwarp(0xFFFFFFFF);

    out[threadIdx.x] = val;
}
"""

PTX_VERSION = resolve_ptx_version()

TEMPLATE_KERNEL_PTX = f"""\
.version {PTX_VERSION}
.target {{target}}
.address_size 64

.visible .entry squatch_kernel(.param .u64 out_param) {{
    .reg .u64 %rd<4>;
    .reg .b32 %r<8>;
    .reg .pred %p<2>;

    // Load output pointer
    ld.param.u64 %rd1, [out_param];

    // Marker: mov immediate (we'll find and patch this in SASS)
    mov.u32 %r1, 42;

    // Warp sync
    bar.warp.sync 0xffffffff;

    // Compute address: out[tid]
    mov.u32 %r2, %tid.x;
    mul.wide.u32 %rd2, %r2, 4;
    add.u64 %rd3, %rd1, %rd2;

    // Store result
    st.global.u32 [%rd3], %r1;

    ret;
}}
"""


# ---------------------------------------------------------------------------
# Cubin builder and parser
# ---------------------------------------------------------------------------

class CubinBuilder:
    """Builds and parses cubin binaries for SASS probing."""

    def __init__(self, nvcc_path: str = "nvcc", nvdisasm_path: str = "nvdisasm",
                 cuobjdump_path: str = "cuobjdump", target: str = "sm_121"):
        self.nvcc_path = nvcc_path
        self.nvdisasm_path = nvdisasm_path
        self.cuobjdump_path = cuobjdump_path
        self.target = target
        self._verify_tools()

    def _verify_tools(self):
        """Verify required tools are available."""
        for tool in [self.nvcc_path, self.nvdisasm_path]:
            try:
                subprocess.run(
                    [tool, "--version"],
                    capture_output=True, timeout=5,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                raise RuntimeError(f"{tool} not found. Ensure CUDA toolkit is installed.")

    def compile_template(self) -> bytes:
        """Compile the template kernel to a cubin binary."""
        with tempfile.NamedTemporaryFile(
            suffix=".cu", mode="w", delete=False, prefix="squatch_"
        ) as f:
            f.write(TEMPLATE_KERNEL_CU)
            cu_path = f.name

        cubin_path = cu_path.replace(".cu", ".cubin")

        try:
            result = subprocess.run(
                [
                    self.nvcc_path,
                    "-cubin",
                    f"-arch={self.target}",
                    "-o", cubin_path,
                    cu_path,
                ],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"nvcc compilation failed:\n{result.stderr}"
                )

            with open(cubin_path, "rb") as f:
                return f.read()
        finally:
            for p in [cu_path, cubin_path]:
                if os.path.exists(p):
                    os.unlink(p)

    def disassemble(self, cubin_data: bytes) -> str:
        """Disassemble a cubin with nvdisasm (hex output)."""
        out = disassemble_cubin(
            cubin_data,
            flags=("-hex",),
            nvdisasm_path=self.nvdisasm_path,
            timeout_s=10,
        )
        if out:
            return out
        return disassemble_cubin(
            cubin_data,
            flags=(),
            nvdisasm_path=self.nvdisasm_path,
            timeout_s=10,
        )

    def disassemble_raw(self, cubin_data: bytes) -> str:
        """Disassemble with cuobjdump for raw SASS with hex encoding."""
        with tempfile.NamedTemporaryFile(
            suffix=".cubin", delete=False, prefix="squatch_"
        ) as f:
            f.write(cubin_data)
            cubin_path = f.name

        try:
            result = subprocess.run(
                [self.cuobjdump_path, "-sass", cubin_path],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout if result.returncode == 0 else ""
        except FileNotFoundError:
            return ""
        finally:
            os.unlink(cubin_path)

    def parse_cubin_elf(self, cubin_data: bytes) -> CubinInfo:
        """Parse the cubin ELF to find the .text section."""
        data = bytearray(cubin_data)
        text_offset, text_size, text_name = find_text_section(cubin_data)
        kernel_name = text_name[6:] if text_name.startswith(".text.") else "unknown"

        return CubinInfo(
            raw_data=data,
            text_section_offset=text_offset,
            text_section_size=text_size,
            kernel_name=kernel_name,
        )

    def extract_instructions(self, cubin_info: CubinInfo) -> List[SASSInstruction]:
        """Extract SASS instructions from the .text section."""
        data = cubin_info.raw_data
        offset = cubin_info.text_section_offset
        size = cubin_info.text_section_size

        instructions = []
        pos = 0
        while pos + 16 <= size:
            # 128-bit instruction: lower 64 bits + upper 64 bits
            inst_bytes = bytes(data[offset + pos:offset + pos + 16])
            word_lo = struct.unpack_from("<Q", inst_bytes, 0)[0]
            word_hi = struct.unpack_from("<Q", inst_bytes, 8)[0]

            instructions.append(SASSInstruction(
                offset=pos,
                mnemonic="",  # Will be filled from nvdisasm
                operands="",
                encoding=inst_bytes,
                instruction_word=word_lo,
                control_word=word_hi,
            ))
            pos += 16

        cubin_info.instructions = instructions
        return instructions

    def annotate_from_disasm(self, cubin_info: CubinInfo, disasm_output: str):
        """Annotate extracted instructions with nvdisasm mnemonics."""
        # Parse nvdisasm output to map offsets to mnemonics
        # Format: /*0000*/    MOV R1, c[0x0][0x28] ;    /* 0x... */
        pattern = re.compile(
            r'/\*([0-9a-fA-F]+)\*/\s+'
            r'(\S+)'          # mnemonic
            r'\s*(.*?)\s*;'   # operands
        )

        offset_to_mnemonic = {}
        for match in pattern.finditer(disasm_output):
            off = int(match.group(1), 16)
            mnemonic = match.group(2)
            operands = match.group(3).strip()
            offset_to_mnemonic[off] = (mnemonic, operands)

        for inst in cubin_info.instructions:
            if inst.offset in offset_to_mnemonic:
                inst.mnemonic, inst.operands = offset_to_mnemonic[inst.offset]


# ---------------------------------------------------------------------------
# Subprocess-isolated probe worker
# ---------------------------------------------------------------------------

# Standalone script run as a child process.  Receives template cubin on
# stdin, probes opcodes, writes JSON results to stdout, then exits.
# If a fatal GPU error occurs, the process dies -- the parent detects
# this and restarts from the next opcode.

_WORKER_SCRIPT = r'''
import ctypes, json, signal, struct, sys, threading

# Must match SASSProbeResult enum values (auto() starts at 1)
VALID = 1
ILLEGAL_INSTRUCTION = 2
LAUNCH_FAILED = 3
LOAD_FAILED = 4
WRONG_OUTPUT = 5
TIMEOUT = 6

GPU_SYNC_TIMEOUT = 5  # seconds -- kill worker if cuCtxSynchronize hangs

def emit(opcode, result, error, output):
    """Write one result as a JSON line to stdout (streaming)."""
    sys.stdout.write(json.dumps([opcode, result, error, output]) + "\n")
    sys.stdout.flush()

def _alarm_handler(signum, frame):
    """SIGALRM handler -- GPU sync timed out, exit immediately."""
    raise TimeoutError("GPU sync timeout")

def main():
    target = sys.argv[1]
    inst_file_offset = int(sys.argv[2])   # absolute file offset of target instruction (lo word)
    start = int(sys.argv[3])
    end = int(sys.argv[4])
    step = int(sys.argv[5])
    template_path = sys.argv[6]

    with open(template_path, "rb") as f:
        template_data = f.read()

    from ctypes import (byref, c_uint, c_void_p, c_size_t, c_char,
                        c_ulonglong, c_int, create_string_buffer)

    CUDA_SUCCESS = 0
    CUdevice = c_int
    CUcontext = c_void_p
    CUmodule = c_void_p
    CUfunction = c_void_p
    CUdeviceptr = c_ulonglong

    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuInit(c_uint(0))
    dev = CUdevice()
    lib.cuDeviceGet(byref(dev), c_int(0))
    ctx = CUcontext()
    lib.cuCtxCreate_v2(byref(ctx), c_uint(0), dev)

    # Install SIGALRM handler for GPU sync timeout
    signal.signal(signal.SIGALRM, _alarm_handler)

    original_word = struct.unpack_from("<Q", template_data, inst_file_offset)[0]

    for opcode in range(start, end, step):
        # Patch
        data = bytearray(template_data)
        new_word = (original_word & ~0xFFF) | (opcode & 0xFFF)
        struct.pack_into("<Q", data, inst_file_offset, new_word)

        # Load
        buf = create_string_buffer(bytes(data))
        module = CUmodule()
        r = lib.cuModuleLoadData(byref(module), buf)
        if r != CUDA_SUCCESS:
            emit(opcode, LOAD_FAILED, r, 0)
            continue

        # Get function
        func = CUfunction()
        r = lib.cuModuleGetFunction(byref(func), module, b"squatch_kernel")
        if r != CUDA_SUCCESS:
            lib.cuModuleUnload(module)
            emit(opcode, LOAD_FAILED, r, 0)
            continue

        # Alloc output
        d_out = CUdeviceptr()
        r = lib.cuMemAlloc_v2(byref(d_out), c_size_t(128))
        if r != CUDA_SUCCESS:
            lib.cuModuleUnload(module)
            emit(opcode, LOAD_FAILED, r, 0)
            continue

        lib.cuMemsetD8_v2(d_out, c_char(0), c_size_t(128))

        # Params: (out_ptr, pred_mask)
        param_out = CUdeviceptr(d_out.value)
        param_mask = c_uint(0)  # always false -> predicated instruction should not execute
        param_ptrs = (c_void_p * 2)()
        param_ptrs[0] = ctypes.cast(ctypes.pointer(param_out), c_void_p)
        param_ptrs[1] = ctypes.cast(ctypes.pointer(param_mask), c_void_p)

        # Launch
        r = lib.cuLaunchKernel(
            func,
            c_uint(1), c_uint(1), c_uint(1),
            c_uint(32), c_uint(1), c_uint(1),
            c_uint(0), c_void_p(0), param_ptrs, c_void_p(0)
        )
        if r != CUDA_SUCCESS:
            lib.cuMemFree_v2(d_out)
            lib.cuModuleUnload(module)
            # 715 = illegal instruction decode/issue
            # 700 = illegal address (not an opcode decode error; keep distinct)
            emit(opcode, ILLEGAL_INSTRUCTION if r == 715 else LAUNCH_FAILED, r, 0)
            sys.exit(0)  # GPU poisoned

        # Sync with timeout -- some opcodes cause infinite GPU loops
        try:
            signal.alarm(GPU_SYNC_TIMEOUT)
            r = lib.cuCtxSynchronize()
            signal.alarm(0)  # cancel alarm
        except TimeoutError:
            # GPU is hung -- report timeout and exit (process will be killed)
            emit(opcode, TIMEOUT, -1, 0)
            sys.exit(0)

        if r != CUDA_SUCCESS:
            lib.cuMemFree_v2(d_out)
            lib.cuModuleUnload(module)
            emit(opcode, ILLEGAL_INSTRUCTION if r == 715 else LAUNCH_FAILED, r, 0)
            sys.exit(0)  # GPU poisoned

        # Read output
        host_out = (ctypes.c_uint * 32)()
        lib.cuMemcpyDtoH_v2(host_out, d_out, c_size_t(128))
        lib.cuMemFree_v2(d_out)
        lib.cuModuleUnload(module)

        if host_out[0] == 42:
            emit(opcode, VALID, 0, 42)
        else:
            emit(opcode, WRONG_OUTPUT, 0, int(host_out[0]))

if __name__ == "__main__":
    main()
'''


def _run_probe_batch_subprocess(template_data, inst_file_offset, target, start, end, step,
                                 callback=None, results_dict=None, progress_counter=None,
                                 total=None):
    """Run a batch of opcode probes in a subprocess with streaming output.

    Streams results line-by-line from the child process so the parent can
    update progress in real-time.  Each line is a JSON array:
        [opcode, result_code, error_code, output_value]

    When the child hits a fatal GPU error, it emits the result and exits.
    Returns the last opcode processed (for the parent to know where to resume).
    """
    import json

    # Write template to a temp file for the child to read
    with tempfile.NamedTemporaryFile(
        suffix=".cubin", delete=False, prefix="sq_tmpl_"
    ) as f:
        f.write(template_data)
        tmpl_path = f.name

    # Write worker script to temp file
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, prefix="sq_worker_"
    ) as f:
        f.write(_WORKER_SCRIPT)
        script_path = f.name

    last_opcode = start - step  # nothing processed yet
    PER_OPCODE_TIMEOUT = 10  # seconds - kill worker if no output for this long

    try:
        proc = subprocess.Popen(
            [
                sys.executable, script_path,
                target, str(inst_file_offset),
                str(start), str(end), str(step),
                tmpl_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        import select
        buf = ""
        while True:
            # Wait for data with timeout
            ready, _, _ = select.select([proc.stdout], [], [], PER_OPCODE_TIMEOUT)
            if not ready:
                # Worker has been silent for PER_OPCODE_TIMEOUT seconds -- hung
                proc.kill()
                # Report current opcode as timeout
                next_opc = last_opcode + step if last_opcode >= start else start
                if next_opc < end:
                    result_enum = SASSProbeResult.TIMEOUT
                    outcome = SASSProbeOutcome(
                        opcode_value=next_opc,
                        result=result_enum,
                        error_code=-1,
                    )
                    if results_dict is not None:
                        results_dict[next_opc] = outcome
                    if progress_counter is not None:
                        progress_counter[0] += 1
                    if callback:
                        cb_total = total if total else (end - start + step - 1) // step
                        callback(next_opc, outcome, progress_counter[0] if progress_counter else 0, cb_total)
                    last_opcode = next_opc
                break

            chunk = proc.stdout.read(1)
            if not chunk:
                break  # EOF - worker exited
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    opcode, result_code, error_code, output_value = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue

                last_opcode = opcode

                result_enum = SASSProbeResult(result_code)
                outcome = SASSProbeOutcome(
                    opcode_value=opcode,
                    result=result_enum,
                    error_code=error_code,
                    output_value=output_value,
                )

                if results_dict is not None:
                    results_dict[opcode] = outcome

                if progress_counter is not None:
                    progress_counter[0] += 1

                if callback:
                    cb_total = total if total else (end - start + step - 1) // step
                    callback(opcode, outcome, progress_counter[0] if progress_counter else 0, cb_total)

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    except Exception:
        pass
    finally:
        for p in [tmpl_path, script_path]:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    return last_opcode


# ---------------------------------------------------------------------------
# SASS opcode enumerator
# ---------------------------------------------------------------------------

class SASSProber:
    """
    Enumerate SASS opcode space by patching cubin binaries.

    Strategy:
      1. Build a template cubin with a known instruction
      2. Identify a "target" instruction to patch (e.g., the MOV that loads 42)
      3. For each candidate opcode value, patch the opcode field
      4. Load and execute the modified cubin
      5. Classify the result

    This is the GPU equivalent of sandsifter's x86 instruction enumeration.
    """

    def __init__(self, cuda_driver, cubin_builder: CubinBuilder):
        self.cuda = cuda_driver
        self.builder = cubin_builder
        self.template_cubin: Optional[CubinInfo] = None
        self.target_inst_idx: int = -1
        self.results: Dict[int, SASSProbeOutcome] = {}
        self._context_poisoned: bool = False

    def prepare(self) -> bool:
        """Build and analyze the template cubin. Returns True on success."""
        # Compile template
        cubin_data = self.builder.compile_template()

        # Parse ELF
        self.template_cubin = self.builder.parse_cubin_elf(cubin_data)

        # Extract instructions
        self.builder.extract_instructions(self.template_cubin)

        # Disassemble and annotate
        disasm = self.builder.disassemble(cubin_data)
        sass_raw = self.builder.disassemble_raw(cubin_data)
        self.builder.annotate_from_disasm(self.template_cubin, disasm)
        if not any(i.mnemonic for i in self.template_cubin.instructions):
            self.builder.annotate_from_disasm(self.template_cubin, sass_raw)

        # Find target instruction: look for the distinctive immediate marker (0x1337)
        # in the disassembly operands. This is intentionally chosen to be stable
        # across minor codegen differences while remaining unlikely to occur
        # elsewhere in the template kernel.
        for idx, inst in enumerate(self.template_cubin.instructions):
            ops = (inst.operands or "").lower()
            if "0x1337" in ops or "4919" in ops:
                self.target_inst_idx = idx
                break

        # Fallback: use first non-NOP instruction that's not the first instruction
        # (first instruction is usually stack setup)
        if self.target_inst_idx < 0:
            for idx, inst in enumerate(self.template_cubin.instructions):
                if idx > 0 and inst.mnemonic and inst.mnemonic not in ("NOP", "EXIT", "BRA"):
                    self.target_inst_idx = idx
                    break

        if self.target_inst_idx < 0 and len(self.template_cubin.instructions) > 1:
            self.target_inst_idx = 1  # Last resort: second instruction

        return self.target_inst_idx >= 0

    def get_template_info(self) -> Dict:
        """Get information about the template cubin for display."""
        if not self.template_cubin:
            return {}

        inst_summary = []
        for i, inst in enumerate(self.template_cubin.instructions):
            marker = " <-- TARGET" if i == self.target_inst_idx else ""
            inst_summary.append({
                "idx": i,
                "offset": f"0x{inst.offset:04x}",
                "mnemonic": inst.mnemonic or "???",
                "operands": inst.operands,
                "opcode_12bit": f"0x{inst.opcode_12bit:03x}",
                "encoding_lo": f"0x{inst.instruction_word:016x}",
                "encoding_hi": f"0x{inst.control_word:016x}",
                "marker": marker,
            })

        return {
            "kernel_name": self.template_cubin.kernel_name,
            "text_offset": f"0x{self.template_cubin.text_section_offset:x}",
            "text_size": self.template_cubin.text_section_size,
            "num_instructions": len(self.template_cubin.instructions),
            "target_instruction": self.target_inst_idx,
            "instructions": inst_summary,
        }

    def _build_opcode_map(self) -> Dict[int, str]:
        """Build a map of known opcodes from the template disassembly."""
        opcode_map = {}
        for inst in self.template_cubin.instructions:
            if inst.mnemonic:
                opc = inst.opcode_12bit
                if opc not in opcode_map:
                    opcode_map[opc] = inst.mnemonic
        return opcode_map

    def _patch_cubin(self, opcode_value: int) -> bytes:
        """
        Create a modified cubin with the target instruction's opcode patched.

        Only modifies the lower 12 bits of the instruction word, preserving
        register encoding and control bits.
        """
        data = bytearray(self.template_cubin.raw_data)
        target = self.template_cubin.instructions[self.target_inst_idx]

        # Calculate byte offset of the target instruction in the file
        file_offset = self.template_cubin.text_section_offset + target.offset

        # Read current instruction word (lower 64 bits)
        current_word = struct.unpack_from("<Q", data, file_offset)[0]

        # Replace lower 12 bits with new opcode
        new_word = (current_word & ~0xFFF) | (opcode_value & 0xFFF)

        # Write back
        struct.pack_into("<Q", data, file_offset, new_word)

        return bytes(data)

    def probe_opcode(self, opcode_value: int) -> SASSProbeOutcome:
        """Probe a single opcode value by patching and executing.

        After any fatal kernel error (illegal instruction, launch failure),
        the CUDA context is poisoned and must be reset.  This method handles
        context recovery automatically so the next probe starts clean.
        """
        # Reset poisoned context from a previous probe
        if self._context_poisoned:
            try:
                self.cuda.reset_context()
            except Exception:
                return SASSProbeOutcome(
                    opcode_value=opcode_value,
                    result=SASSProbeResult.LOAD_FAILED,
                    error_code=-99,
                )
            self._context_poisoned = False

        # Patch the cubin
        modified_cubin = self._patch_cubin(opcode_value)
        needs_reset = False

        # Try to load
        module, error_code = self.cuda.load_cubin(modified_cubin)
        if module is None:
            return SASSProbeOutcome(
                opcode_value=opcode_value,
                result=SASSProbeResult.LOAD_FAILED,
                error_code=error_code,
            )

        try:
            # Get kernel function
            func_name = self.template_cubin.kernel_name
            try:
                func = self.cuda.get_function(module, func_name)
            except Exception:
                # Try common mangled names
                for name in ["squatch_kernel", "_Z15squatch_kernelPj"]:
                    try:
                        func = self.cuda.get_function(module, name)
                        break
                    except Exception:
                        continue
                else:
                    return SASSProbeOutcome(
                        opcode_value=opcode_value,
                        result=SASSProbeResult.LOAD_FAILED,
                        error_code=-1,
                    )

            # Allocate output buffer (32 threads * 4 bytes = 128 bytes)
            out_size = 32 * 4
            d_out = self.cuda.malloc(out_size)

            try:
                # Zero output
                self.cuda.memset(d_out, 0, out_size)

                # Launch kernel
                # Params: (out_ptr, pred_mask=0) to keep patchpoint predicated off.
                result = self.cuda.launch_and_sync(
                    func,
                    grid=(1, 1, 1),
                    block=(32, 1, 1),
                    params=[d_out, 0],
                )

                if result == 0:  # CUDA_SUCCESS
                    # Read back output
                    host_out = (ctypes.c_uint * 32)()
                    self.cuda.memcpy_dtoh(host_out, d_out, out_size)

                    # Check if output matches expected value (42)
                    if host_out[0] == 42:
                        return SASSProbeOutcome(
                            opcode_value=opcode_value,
                            result=SASSProbeResult.VALID,
                            output_value=host_out[0],
                        )
                    else:
                        return SASSProbeOutcome(
                            opcode_value=opcode_value,
                            result=SASSProbeResult.WRONG_OUTPUT,
                            output_value=host_out[0],
                        )
                else:
                    # Fatal kernel error -- context is now poisoned
                    needs_reset = True
                    if result == 715:  # CUDA_ERROR_ILLEGAL_INSTRUCTION
                        return SASSProbeOutcome(
                            opcode_value=opcode_value,
                            result=SASSProbeResult.ILLEGAL_INSTRUCTION,
                            error_code=result,
                        )
                    elif result == 700:  # CUDA_ERROR_ILLEGAL_ADDRESS
                        return SASSProbeOutcome(
                            opcode_value=opcode_value,
                            result=SASSProbeResult.ILLEGAL_INSTRUCTION,
                            error_code=result,
                        )
                    else:
                        return SASSProbeOutcome(
                            opcode_value=opcode_value,
                            result=SASSProbeResult.LAUNCH_FAILED,
                            error_code=result,
                        )
            finally:
                if not needs_reset:
                    try:
                        self.cuda.free(d_out)
                    except Exception:
                        needs_reset = True
        finally:
            if not needs_reset:
                try:
                    self.cuda.unload_module(module)
                except Exception:
                    needs_reset = True
            if needs_reset:
                self._context_poisoned = True

    def enumerate_opcodes(
        self,
        start: int = 0,
        end: int = 4096,
        step: int = 1,
        callback=None,
    ) -> Dict[int, SASSProbeOutcome]:
        """
        Enumerate SASS opcodes in [start, end) range.

        Uses subprocess isolation: after a fatal GPU error (illegal instruction),
        the CUDA device is poisoned for the entire process. We run batches in
        child processes that probe until they hit a fatal error, then restart
        from the next opcode.

        Results stream line-by-line from each subprocess so the parent can
        update progress in real-time.

        callback(opcode, outcome, progress, total) is called for each probe.
        """
        total = (end - start + step - 1) // step

        # Get the cubin template data so child processes can use it
        template_data = self.template_cubin.raw_data
        target = self.builder.target
        # Compute exact file offset of the target instruction's lower 64-bit word.
        # Passing an absolute file offset avoids ambiguity when multiple .text.*
        # sections exist in the cubin.
        target_inst = self.template_cubin.instructions[self.target_inst_idx]
        inst_file_offset = self.template_cubin.text_section_offset + target_inst.offset

        current = start
        progress_counter = [0]  # mutable so subprocess helper can update it

        while current < end:
            last_opcode = _run_probe_batch_subprocess(
                template_data, inst_file_offset, target, current, end, step,
                callback=callback,
                results_dict=self.results,
                progress_counter=progress_counter,
                total=total,
            )

            if last_opcode < current:
                # Worker crashed without reporting anything -- skip this opcode
                outcome = SASSProbeOutcome(
                    opcode_value=current,
                    result=SASSProbeResult.LAUNCH_FAILED,
                    error_code=-1,
                )
                self.results[current] = outcome
                progress_counter[0] += 1
                if callback:
                    callback(current, outcome, progress_counter[0], total)
                current += step
            else:
                current = last_opcode + step

        return self.results

    def get_opcode_summary(self) -> Dict:
        """Summarize the opcode enumeration results."""
        valid = []
        illegal = []
        wrong_output = []
        load_fail = []
        other = []

        for opcode, outcome in sorted(self.results.items()):
            if outcome.result == SASSProbeResult.VALID:
                valid.append(opcode)
            elif outcome.result == SASSProbeResult.ILLEGAL_INSTRUCTION:
                illegal.append(opcode)
            elif outcome.result == SASSProbeResult.WRONG_OUTPUT:
                wrong_output.append((opcode, outcome.output_value))
            elif outcome.result == SASSProbeResult.LOAD_FAILED:
                load_fail.append(opcode)
            else:
                other.append(opcode)

        return {
            "total_probed": len(self.results),
            "valid": valid,
            "illegal": illegal,
            "wrong_output": wrong_output,
            "load_failed": load_fail,
            "other": other,
            "valid_count": len(valid),
            "illegal_count": len(illegal),
            "wrong_output_count": len(wrong_output),
        }

    def discover_opcode_field(self) -> Dict:
        """
        Discover the SASS opcode field encoding by analyzing known instructions.

        Compiles multiple distinct PTX instructions, disassembles each, and
        compares their binary encodings to identify which bits form the opcode.
        Also includes opcodes found in the template kernel itself.
        """
        # Different PTX instructions with side effects to survive optimization.
        # Each writes its result to an output pointer so ptxas can't eliminate it.
        test_ptx_programs = {
            "mov_imm": (
                ".version 9.1\n.target {target}\n.address_size 64\n"
                ".visible .entry test_mov(.param .u64 out) {{\n"
                "    .reg .b32 %r<2>;\n.reg .u64 %rd<2>;\n"
                "    ld.param.u64 %rd0, [out];\n"
                "    mov.u32 %r0, 42;\n"
                "    st.global.u32 [%rd0], %r0;\n"
                "    ret;\n}}\n"
            ),
            "add_int": (
                ".version 9.1\n.target {target}\n.address_size 64\n"
                ".visible .entry test_add(.param .u64 out) {{\n"
                "    .reg .b32 %r<4>;\n.reg .u64 %rd<2>;\n"
                "    ld.param.u64 %rd0, [out];\n"
                "    ld.global.u32 %r1, [%rd0];\n"
                "    ld.global.u32 %r2, [%rd0+4];\n"
                "    add.s32 %r0, %r1, %r2;\n"
                "    st.global.u32 [%rd0], %r0;\n"
                "    ret;\n}}\n"
            ),
            "mul_float": (
                ".version 9.1\n.target {target}\n.address_size 64\n"
                ".visible .entry test_mul(.param .u64 out) {{\n"
                "    .reg .f32 %f<4>;\n.reg .u64 %rd<2>;\n"
                "    ld.param.u64 %rd0, [out];\n"
                "    ld.global.f32 %f1, [%rd0];\n"
                "    ld.global.f32 %f2, [%rd0+4];\n"
                "    mul.f32 %f0, %f1, %f2;\n"
                "    st.global.f32 [%rd0], %f0;\n"
                "    ret;\n}}\n"
            ),
            "fma_float": (
                ".version 9.1\n.target {target}\n.address_size 64\n"
                ".visible .entry test_fma(.param .u64 out) {{\n"
                "    .reg .f32 %f<5>;\n.reg .u64 %rd<2>;\n"
                "    ld.param.u64 %rd0, [out];\n"
                "    ld.global.f32 %f1, [%rd0];\n"
                "    ld.global.f32 %f2, [%rd0+4];\n"
                "    ld.global.f32 %f3, [%rd0+8];\n"
                "    fma.rn.f32 %f0, %f1, %f2, %f3;\n"
                "    st.global.f32 [%rd0], %f0;\n"
                "    ret;\n}}\n"
            ),
            "and_bits": (
                ".version 9.1\n.target {target}\n.address_size 64\n"
                ".visible .entry test_and(.param .u64 out) {{\n"
                "    .reg .b32 %r<4>;\n.reg .u64 %rd<2>;\n"
                "    ld.param.u64 %rd0, [out];\n"
                "    ld.global.u32 %r1, [%rd0];\n"
                "    ld.global.u32 %r2, [%rd0+4];\n"
                "    and.b32 %r0, %r1, %r2;\n"
                "    st.global.u32 [%rd0], %r0;\n"
                "    ret;\n}}\n"
            ),
            "shl_bits": (
                ".version 9.1\n.target {target}\n.address_size 64\n"
                ".visible .entry test_shl(.param .u64 out) {{\n"
                "    .reg .b32 %r<4>;\n.reg .u64 %rd<2>;\n"
                "    ld.param.u64 %rd0, [out];\n"
                "    ld.global.u32 %r1, [%rd0];\n"
                "    ld.global.u32 %r2, [%rd0+4];\n"
                "    shl.b32 %r0, %r1, %r2;\n"
                "    st.global.u32 [%rd0], %r0;\n"
                "    ret;\n}}\n"
            ),
            "setp_cmp": (
                ".version 9.1\n.target {target}\n.address_size 64\n"
                ".visible .entry test_setp(.param .u64 out) {{\n"
                "    .reg .b32 %r<4>;\n.reg .u64 %rd<2>;\n"
                "    .reg .pred %p<2>;\n"
                "    ld.param.u64 %rd0, [out];\n"
                "    ld.global.u32 %r0, [%rd0];\n"
                "    ld.global.u32 %r1, [%rd0+4];\n"
                "    setp.eq.s32 %p0, %r0, %r1;\n"
                "    selp.u32 %r2, 1, 0, %p0;\n"
                "    st.global.u32 [%rd0], %r2;\n"
                "    ret;\n}}\n"
            ),
            "cvt_f2i": (
                ".version 9.1\n.target {target}\n.address_size 64\n"
                ".visible .entry test_cvt(.param .u64 out) {{\n"
                "    .reg .b32 %r<4>;\n.reg .f32 %f<2>;\n.reg .u64 %rd<2>;\n"
                "    ld.param.u64 %rd0, [out];\n"
                "    ld.global.f32 %f0, [%rd0];\n"
                "    cvt.rni.s32.f32 %r0, %f0;\n"
                "    st.global.u32 [%rd0], %r0;\n"
                "    ret;\n}}\n"
            ),
            "div_float": (
                ".version 9.1\n.target {target}\n.address_size 64\n"
                ".visible .entry test_div(.param .u64 out) {{\n"
                "    .reg .f32 %f<4>;\n.reg .u64 %rd<2>;\n"
                "    ld.param.u64 %rd0, [out];\n"
                "    ld.global.f32 %f1, [%rd0];\n"
                "    ld.global.f32 %f2, [%rd0+4];\n"
                "    div.approx.f32 %f0, %f1, %f2;\n"
                "    st.global.f32 [%rd0], %f0;\n"
                "    ret;\n}}\n"
            ),
        }

        opcode_map = {}
        target = self.builder.target

        # First, include opcodes from the template kernel itself (if available)
        if self.template_cubin and self.template_cubin.instructions:
            for inst in self.template_cubin.instructions:
                if inst.mnemonic and inst.mnemonic not in ("NOP",):
                    base_mnem = inst.mnemonic.split(".")[0]  # e.g., "IMAD.WIDE.U32" -> "IMAD"
                    opc12 = inst.opcode_12bit
                    opc10 = inst.opcode_10bit
                    if inst.mnemonic not in opcode_map:
                        opcode_map[inst.mnemonic] = {
                            "bits_11_0": f"0x{opc12:03x}",
                            "bits_11_2": f"0x{opc10:03x}",
                            "full_word_lo": f"0x{inst.instruction_word:016x}",
                            "name": "template",
                        }

        # Then compile additional PTX programs to discover more opcodes
        for name, ptx_template in test_ptx_programs.items():
            ptx_source = ptx_template.format(target=target)
            ptx_source = ptx_source.replace(".version 9.1", f".version {PTX_VERSION}")

            with tempfile.NamedTemporaryFile(
                suffix=".ptx", mode="w", delete=False, prefix="sifter_disc_"
            ) as f:
                f.write(ptx_source)
                ptx_path = f.name

            cubin_path = ptx_path.replace(".ptx", ".cubin")

            try:
                # Use -O0 to minimize instruction rewriting
                result = subprocess.run(
                    ["ptxas", "-O0", f"-arch={target}", ptx_path, "-o", cubin_path],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode != 0:
                    continue

                with open(cubin_path, "rb") as f:
                    cubin_data = f.read()

                cubin_info = self.builder.parse_cubin_elf(cubin_data)
                self.builder.extract_instructions(cubin_info)

                disasm = self.builder.disassemble(cubin_data)
                self.builder.annotate_from_disasm(cubin_info, disasm)
                if not any(i.mnemonic for i in cubin_info.instructions):
                    disasm_raw = self.builder.disassemble_raw(cubin_data)
                    self.builder.annotate_from_disasm(cubin_info, disasm_raw)

                for inst in cubin_info.instructions:
                    if inst.mnemonic and inst.mnemonic not in ("NOP",):
                        opc12 = inst.opcode_12bit
                        opc10 = inst.opcode_10bit
                        if inst.mnemonic not in opcode_map:
                            opcode_map[inst.mnemonic] = {
                                "bits_11_0": f"0x{opc12:03x}",
                                "bits_11_2": f"0x{opc10:03x}",
                                "full_word_lo": f"0x{inst.instruction_word:016x}",
                                "name": name,
                            }

            finally:
                for p in [ptx_path, cubin_path]:
                    if os.path.exists(p):
                        os.unlink(p)

        return opcode_map

    def discover_from_probes(self, probe_specs, build_ptx_fn, target_arch: str,
                             callback=None,
                             opt_levels: Optional[List[int]] = None) -> Dict:
        """
        Discover SASS opcodes by compiling Phase 1 PTX probes to cubins.

        Compiles each probe at multiple optimization levels to maximize
        SASS opcode discovery. Higher opt levels (-O1, -O3) trigger uniform
        datapath promotions that -O0 alone misses.

        Args:
            probe_specs: list of ProbeSpec objects that compiled successfully
            build_ptx_fn: function(spec, target) -> ptx_source_str
            target_arch: ptxas target (e.g. "sm_121")
            callback: optional fn(progress, total, num_opcodes) for display
            opt_levels: list of ptxas -O levels (default: [0, 1, 3])

        Returns dict mapping mnemonic -> opcode info (same format as
        discover_opcode_field).
        """
        if opt_levels is None:
            opt_levels = [0, 1, 3]

        opcode_map = {}
        total = len(probe_specs) * len(opt_levels)
        progress = 0

        for i, spec in enumerate(probe_specs):
            ptx_source = build_ptx_fn(spec, target_arch)
            ptx_source = ptx_source.replace(".version 9.1", f".version {PTX_VERSION}")

            for opt_level in opt_levels:
                progress += 1

                with tempfile.NamedTemporaryFile(
                    suffix=".ptx", mode="w", delete=False, prefix="squatch_p2_"
                ) as f:
                    f.write(ptx_source)
                    ptx_path = f.name

                cubin_path = ptx_path.replace(".ptx", ".cubin")

                try:
                    result = subprocess.run(
                        ["ptxas", f"-O{opt_level}", f"-arch={target_arch}",
                         ptx_path, "-o", cubin_path],
                        capture_output=True, text=True, timeout=10,
                    )
                    if result.returncode != 0:
                        continue

                    with open(cubin_path, "rb") as f:
                        cubin_data = f.read()

                    cubin_info = self.builder.parse_cubin_elf(cubin_data)
                    self.builder.extract_instructions(cubin_info)

                    disasm = self.builder.disassemble(cubin_data)
                    self.builder.annotate_from_disasm(cubin_info, disasm)
                    if not any(inst.mnemonic for inst in cubin_info.instructions):
                        disasm_raw = self.builder.disassemble_raw(cubin_data)
                        self.builder.annotate_from_disasm(cubin_info, disasm_raw)

                    for inst in cubin_info.instructions:
                        if inst.mnemonic and inst.mnemonic not in ("NOP",):
                            opc12 = inst.opcode_12bit
                            opc10 = inst.opcode_10bit
                            if inst.mnemonic not in opcode_map:
                                opcode_map[inst.mnemonic] = {
                                    "bits_11_0": f"0x{opc12:03x}",
                                    "bits_11_2": f"0x{opc10:03x}",
                                    "full_word_lo": f"0x{inst.instruction_word:016x}",
                                    "name": f"{spec.name} (O{opt_level})",
                                }
                except Exception:
                    pass
                finally:
                    for p in [ptx_path, cubin_path]:
                        if os.path.exists(p):
                            os.unlink(p)

                if callback:
                    callback(progress, total, len(opcode_map))

        return opcode_map


# Need ctypes for memcpy in probe_opcode
import ctypes
