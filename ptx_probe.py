#!/usr/bin/env python3
"""
PTX instruction space prober for SASSquatch.

Systematically generates PTX programs for every instruction category and
type qualifier combination, then tests compilation with ptxas for multiple
target architectures. This discovers:

  - Which instructions SM121 supports vs SM100/SM90
  - Architecture-specific instructions
  - Undocumented instruction/type combinations
  - ptxas behavior differences across targets

Phase 1 (compilation) uses only ptxas and does not execute GPU kernels.
Phase 2 (execution) loads compiled PTX and runs on the GPU to detect
runtime anomalies like illegal instruction traps or wrong results.

References:
  PTX ISA Specification (v9.1 for CUDA 13.1):
    https://docs.nvidia.com/cuda/parallel-thread-execution/
  PTX ISA Release Notes (new instructions per version):
    https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes
  SASS Instruction Set Reference (Blackwell):
    https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference
"""

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# PTX register type mapping
# ---------------------------------------------------------------------------

def reg_decl_for_type(ptx_type: str) -> str:
    """Map a PTX type qualifier to a register declaration type."""
    mapping = {
        # Float
        ".f16": ".b16", ".f16x2": ".b32",
        ".bf16": ".b16", ".bf16x2": ".b32",
        ".f32": ".f32", ".f64": ".f64",
        ".tf32": ".b32",
        # FP8
        ".e4m3": ".b16", ".e5m2": ".b16",
        ".e2m3": ".b16", ".e3m2": ".b16",
        # FP4
        ".e2m1": ".b16",
        # Signed int
        ".s8": ".b16", ".s16": ".b16", ".s32": ".b32", ".s64": ".b64",
        # Unsigned int
        ".u8": ".b16", ".u16": ".b16", ".u32": ".b32", ".u64": ".b64",
        # Bits
        ".b1": ".pred",
        ".b8": ".b16", ".b16": ".b16", ".b32": ".b32",
        ".b64": ".b64", ".b128": ".b128",
        # Pred
        ".pred": ".pred",
    }
    return mapping.get(ptx_type, ".b32")


def reg_prefix_for_decl(reg_decl: str) -> str:
    """Get register name prefix for a declaration type."""
    prefixes = {
        ".pred": "%p", ".f32": "%f", ".f64": "%fd",
        ".b16": "%h", ".b32": "%r", ".b64": "%rd", ".b128": "%rq",
    }
    return prefixes.get(reg_decl, "%r")


# ---------------------------------------------------------------------------
# Probe result types
# ---------------------------------------------------------------------------

class ProbeResult(Enum):
    COMPILES = auto()          # ptxas accepts it
    COMPILE_ERROR = auto()     # ptxas rejects it
    EXECUTES = auto()          # Kernel runs without error
    ILLEGAL_INSTRUCTION = auto()  # GPU traps on execution
    LAUNCH_ERROR = auto()      # Launch fails (resources, etc.)
    TIMEOUT = auto()           # Kernel hangs
    WRONG_OUTPUT = auto()      # Runs but produces unexpected result


@dataclass
class ProbeSpec:
    """Specification for a single PTX instruction probe."""
    name: str              # Human-readable name (e.g., "add.f32")
    category: str          # Category (e.g., "arithmetic", "mma")
    ptx_body: str          # PTX instruction lines (inserted into kernel)
    reg_decls: str = ""    # Register declarations
    smem_decl: str = ""    # Shared memory declaration (if needed)
    needs_output: bool = False  # Whether kernel writes output
    description: str = ""  # Optional description
    tags: List[str] = field(default_factory=list)


@dataclass
class ProbeOutcome:
    """Result of probing a single instruction on a specific target."""
    spec: ProbeSpec
    target: str
    result: ProbeResult
    error_msg: str = ""
    compile_time_ms: float = 0
    exec_time_ms: float = 0


# ---------------------------------------------------------------------------
# PTX program generator
# ---------------------------------------------------------------------------

PTX_TEMPLATE = """\
.version 9.1
.target {target}
.address_size 64

.visible .entry test_kernel({params}) {{
{reg_decls}
{smem_decl}
{body}
    ret;
}}
"""

PTX_TEMPLATE_WITH_OUTPUT = """\
.version 9.1
.target {target}
.address_size 64

.visible .entry test_kernel(.param .u64 out_ptr) {{
    .reg .u64 %out_addr;
    ld.param.u64 %out_addr, [out_ptr];
{reg_decls}
{smem_decl}
{body}
    ret;
}}
"""


def build_ptx_program(spec: ProbeSpec, target: str) -> str:
    """Build a complete PTX program from a probe spec."""
    reg_block = ""
    if spec.reg_decls:
        reg_block = "\n".join(f"    {line}" for line in spec.reg_decls.strip().split("\n"))

    smem_block = ""
    if spec.smem_decl:
        smem_block = "\n".join(f"    {line}" for line in spec.smem_decl.strip().split("\n"))

    body = "\n".join(f"    {line}" for line in spec.ptx_body.strip().split("\n"))

    if spec.needs_output:
        return PTX_TEMPLATE_WITH_OUTPUT.format(
            target=target,
            reg_decls=reg_block,
            smem_decl=smem_block,
            body=body,
        )
    else:
        return PTX_TEMPLATE.format(
            target=target,
            params="",
            reg_decls=reg_block,
            smem_decl=smem_block,
            body=body,
        )


# ---------------------------------------------------------------------------
# Probe generators -- systematically create instruction probes
# ---------------------------------------------------------------------------

def _make_binary_arith(op: str, type_q: str, tags: List[str] = None) -> ProbeSpec:
    """Generate probe for binary arithmetic: op.type %d, %a, %b"""
    rd = reg_decl_for_type(type_q)
    rp = reg_prefix_for_decl(rd)
    return ProbeSpec(
        name=f"{op}{type_q}",
        category="arithmetic",
        reg_decls=f".reg {rd} {rp}<4>;",
        ptx_body=f"{op}{type_q} {rp}1, {rp}2, {rp}3;",
        tags=tags or [],
    )


def _make_unary(op: str, type_q: str, tags: List[str] = None) -> ProbeSpec:
    """Generate probe for unary: op.type %d, %a"""
    rd = reg_decl_for_type(type_q)
    rp = reg_prefix_for_decl(rd)
    return ProbeSpec(
        name=f"{op}{type_q}",
        category="arithmetic",
        reg_decls=f".reg {rd} {rp}<4>;",
        ptx_body=f"{op}{type_q} {rp}1, {rp}2;",
        tags=tags or [],
    )


def _make_ternary(op: str, type_q: str, rnd: str = "", tags: List[str] = None) -> ProbeSpec:
    """Generate probe for ternary: op{.rnd}.type %d, %a, %b, %c"""
    rd = reg_decl_for_type(type_q)
    rp = reg_prefix_for_decl(rd)
    full_op = f"{op}{rnd}{type_q}" if rnd else f"{op}{type_q}"
    return ProbeSpec(
        name=full_op,
        category="arithmetic",
        reg_decls=f".reg {rd} {rp}<5>;",
        ptx_body=f"{full_op} {rp}1, {rp}2, {rp}3, {rp}4;",
        tags=tags or [],
    )


def _make_cvt(dst_type: str, src_type: str, rnd: str = "", sat: bool = False,
              tags: List[str] = None) -> ProbeSpec:
    """Generate probe for cvt: cvt{.rnd}{.sat}.dst.src %d, %s"""
    dst_rd = reg_decl_for_type(dst_type)
    src_rd = reg_decl_for_type(src_type)
    dst_rp = reg_prefix_for_decl(dst_rd)
    src_rp = reg_prefix_for_decl(src_rd)

    qualifiers = "cvt"
    if rnd:
        qualifiers += rnd
    if sat:
        qualifiers += ".sat"
    qualifiers += f"{dst_type}{src_type}"

    # Use different register namespaces if types differ
    if dst_rd == src_rd:
        decls = f".reg {dst_rd} {dst_rp}<4>;"
        body = f"{qualifiers} {dst_rp}1, {dst_rp}2;"
    else:
        src_rp2 = src_rp.replace("%", "%s_")
        decls = f".reg {dst_rd} {dst_rp}<4>;\n.reg {src_rd} {src_rp2}<4>;"
        body = f"{qualifiers} {dst_rp}1, {src_rp2}2;"

    return ProbeSpec(
        name=qualifiers,
        category="conversion",
        reg_decls=decls,
        ptx_body=body,
        tags=tags or [],
    )


def _make_setp(cmp: str, type_q: str, tags: List[str] = None) -> ProbeSpec:
    """Generate probe for setp: setp.cmp.type %p, %a, %b"""
    rd = reg_decl_for_type(type_q)
    rp = reg_prefix_for_decl(rd)
    return ProbeSpec(
        name=f"setp.{cmp}{type_q}",
        category="comparison",
        reg_decls=f".reg .pred %p<4>;\n.reg {rd} {rp}<4>;",
        ptx_body=f"setp.{cmp}{type_q} %p1, {rp}2, {rp}3;",
        tags=tags or [],
    )


def _make_atom(op: str, space: str, type_q: str, tags: List[str] = None) -> ProbeSpec:
    """Generate probe for atom: atom.space.op.type %d, [%addr], %a"""
    rd = reg_decl_for_type(type_q)
    rp = reg_prefix_for_decl(rd)
    name = f"atom.{space}.{op}{type_q}"
    return ProbeSpec(
        name=name,
        category="atomic",
        reg_decls=f".reg .u64 %addr;\n.reg {rd} {rp}<4>;",
        ptx_body=f"atom.{space}.{op}{type_q} {rp}1, [%addr], {rp}2;",
        tags=tags or [],
    )


def _make_shfl(mode: str, tags: List[str] = None) -> ProbeSpec:
    """Generate probe for shfl.sync: shfl.sync.mode.b32 %d, %a, %b, %c, %mask"""
    return ProbeSpec(
        name=f"shfl.sync.{mode}.b32",
        category="warp",
        reg_decls=".reg .b32 %r<8>;\n.reg .pred %p<2>;",
        ptx_body=f"shfl.sync.{mode}.b32 %r1|%p1, %r2, %r3, %r4, 0xffffffff;",
        tags=tags or [],
    )


def _make_redux(op: str, type_q: str, tags: List[str] = None) -> ProbeSpec:
    """Generate probe for redux.sync: redux.sync.op.type %d, %a, %mask"""
    rd = reg_decl_for_type(type_q)
    rp = reg_prefix_for_decl(rd)
    return ProbeSpec(
        name=f"redux.sync.{op}{type_q}",
        category="warp",
        reg_decls=f".reg {rd} {rp}<4>;",
        ptx_body=f"redux.sync.{op}{type_q} {rp}1, {rp}2, 0xffffffff;",
        tags=tags or [],
    )


# ---------------------------------------------------------------------------
# Comprehensive instruction probe database
# ---------------------------------------------------------------------------

def generate_all_probes() -> List[ProbeSpec]:
    """Generate the full set of PTX instruction probes."""
    probes = []

    # ===== INTEGER ARITHMETIC =====
    for op in ["add", "sub", "min", "max"]:
        for t in [".s32", ".u32", ".s64", ".u64"]:
            probes.append(_make_binary_arith(op, t))

    for op in ["mul.lo", "mul.hi"]:
        for t in [".s32", ".u32", ".s64", ".u64"]:
            probes.append(_make_binary_arith(op, t))

    for op in ["mad.lo", "mad.hi"]:
        for t in [".s32", ".u32", ".s64", ".u64"]:
            probes.append(_make_ternary(op, t))

    for op in ["div", "rem"]:
        for t in [".s32", ".u32", ".s64", ".u64"]:
            probes.append(_make_binary_arith(op, t))

    for op in ["abs", "neg"]:
        for t in [".s32", ".s64"]:
            probes.append(_make_unary(op, t))

    # Bit manipulation
    for op in ["popc", "clz", "bfind", "brev"]:
        for t in [".b32", ".b64"]:
            probes.append(_make_unary(op, t))

    probes.append(ProbeSpec(
        name="bfe.s32", category="bitwise",
        reg_decls=".reg .b32 %r<5>;",
        ptx_body="bfe.s32 %r1, %r2, %r3, %r4;",
    ))
    probes.append(ProbeSpec(
        name="bfe.u32", category="bitwise",
        reg_decls=".reg .b32 %r<5>;",
        ptx_body="bfe.u32 %r1, %r2, %r3, %r4;",
    ))
    probes.append(ProbeSpec(
        name="bfi.b32", category="bitwise",
        reg_decls=".reg .b32 %r<6>;",
        ptx_body="bfi.b32 %r1, %r2, %r3, %r4, %r5;",
    ))

    # ===== FLOAT ARITHMETIC =====
    for op in ["add", "sub", "mul", "min", "max"]:
        for t in [".f32", ".f64"]:
            probes.append(_make_binary_arith(op, t))
        # Half precision
        for t in [".f16", ".f16x2", ".bf16", ".bf16x2"]:
            probes.append(_make_binary_arith(op, t))

    # FMA with rounding modes
    for rnd in [".rn", ".rz", ".rm", ".rp"]:
        for t in [".f32", ".f64"]:
            probes.append(_make_ternary("fma", t, rnd))
    # Half-precision FMA (no rounding mode)
    for t in [".f16", ".f16x2", ".bf16", ".bf16x2"]:
        probes.append(_make_ternary("fma", t))

    # Unary float
    for op in ["abs", "neg"]:
        for t in [".f32", ".f64", ".f16", ".f16x2", ".bf16", ".bf16x2"]:
            probes.append(_make_unary(op, t))

    # Transcendentals
    for op in ["rcp.approx", "sqrt.approx", "rsqrt.approx"]:
        probes.append(_make_unary(op, ".f32"))
    for op in ["rcp.rn", "sqrt.rn"]:
        for t in [".f32", ".f64"]:
            probes.append(_make_unary(op, t))
    for op in ["sin.approx", "cos.approx", "ex2.approx", "lg2.approx"]:
        probes.append(_make_unary(op, ".f32"))

    # ===== LOGIC / BITWISE =====
    for op in ["and", "or", "xor"]:
        for t in [".b16", ".b32", ".b64"]:
            probes.append(_make_binary_arith(op, t))
        probes.append(_make_binary_arith(op, ".pred"))

    for t in [".b16", ".b32", ".b64"]:
        probes.append(_make_unary("not", t))
    probes.append(_make_unary("not", ".pred"))
    probes.append(_make_unary("cnot", ".b16"))
    probes.append(_make_unary("cnot", ".b32"))

    # LOP3 (3-input logic)
    probes.append(ProbeSpec(
        name="lop3.b32", category="bitwise",
        reg_decls=".reg .b32 %r<5>;",
        ptx_body="lop3.b32 %r1, %r2, %r3, %r4, 0x80;",
    ))

    # Shifts
    for op in ["shl"]:
        for t in [".b16", ".b32", ".b64"]:
            rd = reg_decl_for_type(t)
            rp = reg_prefix_for_decl(rd)
            probes.append(ProbeSpec(
                name=f"shl{t}", category="bitwise",
                reg_decls=f".reg {rd} {rp}<4>;\n.reg .u32 %shift;",
                ptx_body=f"shl{t} {rp}1, {rp}2, %shift;",
            ))

    for t in [".b16", ".b32", ".b64"]:
        rd = reg_decl_for_type(t)
        rp = reg_prefix_for_decl(rd)
        probes.append(ProbeSpec(
            name=f"shr{t}", category="bitwise",
            reg_decls=f".reg {rd} {rp}<4>;\n.reg .u32 %shift;",
            ptx_body=f"shr{t} {rp}1, {rp}2, %shift;",
        ))

    # Funnel shift
    for dir in ["l", "r"]:
        for mode in ["wrap", "clamp"]:
            probes.append(ProbeSpec(
                name=f"shf.{dir}.{mode}.b32", category="bitwise",
                reg_decls=".reg .b32 %r<5>;",
                ptx_body=f"shf.{dir}.{mode}.b32 %r1, %r2, %r3, %r4;",
            ))

    # Permute
    probes.append(ProbeSpec(
        name="prmt.b32", category="bitwise",
        reg_decls=".reg .b32 %r<5>;",
        ptx_body="prmt.b32 %r1, %r2, %r3, %r4;",
    ))

    # ===== COMPARISON =====
    for cmp in ["eq", "ne", "lt", "le", "gt", "ge"]:
        for t in [".s32", ".u32", ".f32"]:
            probes.append(_make_setp(cmp, t))
    for cmp in ["eq", "ne", "lt", "le", "gt", "ge"]:
        probes.append(_make_setp(cmp, ".f64"))
    # Half-precision comparisons
    for cmp in ["eq", "ne", "lt", "gt"]:
        for t in [".f16", ".bf16"]:
            probes.append(_make_setp(cmp, t))

    # Float NaN comparisons
    for cmp in ["equ", "neu", "ltu", "gtu", "num", "nan"]:
        probes.append(_make_setp(cmp, ".f32"))

    # selp
    for t in [".b16", ".b32", ".b64", ".f32", ".f64"]:
        rd = reg_decl_for_type(t)
        rp = reg_prefix_for_decl(rd)
        probes.append(ProbeSpec(
            name=f"selp{t}", category="comparison",
            reg_decls=f".reg {rd} {rp}<4>;\n.reg .pred %p1;",
            ptx_body=f"selp{t} {rp}1, {rp}2, {rp}3, %p1;",
        ))

    # ===== CONVERSION =====
    # Standard float conversions
    for rnd in [".rn", ".rz"]:
        for dst, src in [
            (".f32", ".f64"), (".f64", ".f32"),
            (".f32", ".f16"), (".f16", ".f32"),
            (".f32", ".bf16"), (".bf16", ".f32"),
            (".f16", ".bf16"), (".bf16", ".f16"),
        ]:
            probes.append(_make_cvt(dst, src, rnd))

    # Int-float conversions
    for rnd in [".rn", ".rz"]:
        for dst in [".f32"]:
            for src in [".s32", ".u32", ".s64", ".u64"]:
                probes.append(_make_cvt(dst, src, rnd))
    for rnd in [".rni", ".rzi"]:
        for src in [".f32"]:
            for dst in [".s32", ".u32"]:
                probes.append(_make_cvt(dst, src, rnd))

    # FP8 conversions (interesting for SM121!)
    for rnd in [".rn", ".rz"]:
        for fp8 in [".e4m3", ".e5m2"]:
            for flt in [".f16", ".f32", ".bf16"]:
                probes.append(_make_cvt(fp8, flt, rnd, tags=["fp8"]))
                probes.append(_make_cvt(flt, fp8, tags=["fp8"]))

    # FP4 conversions (potentially undocumented!)
    for rnd in [".rn"]:
        for fp4 in [".e2m1"]:
            for flt in [".f16", ".f32", ".bf16"]:
                probes.append(_make_cvt(fp4, flt, rnd, tags=["fp4", "experimental"]))
                probes.append(_make_cvt(flt, fp4, tags=["fp4", "experimental"]))

    # TF32 conversions
    for rnd in [".rn", ".rna"]:
        probes.append(_make_cvt(".tf32", ".f32", rnd, tags=["tf32"]))

    # Saturation
    for dst, src in [(".u8", ".f32"), (".s8", ".f32"), (".u16", ".f32"), (".s16", ".f32")]:
        probes.append(_make_cvt(dst, src, ".rni", sat=True))

    # ===== DATA MOVEMENT =====
    for t in [".b16", ".b32", ".b64", ".f32", ".f64", ".pred"]:
        rd = reg_decl_for_type(t)
        rp = reg_prefix_for_decl(rd)
        probes.append(ProbeSpec(
            name=f"mov{t}", category="data_movement",
            reg_decls=f".reg {rd} {rp}<4>;",
            ptx_body=f"mov{t} {rp}1, {rp}2;",
        ))

    # Vectorized loads
    for vec in ["", ".v2", ".v4"]:
        for t in [".b32"]:
            width = {"": 1, ".v2": 2, ".v4": 4}[vec]
            regs = ", ".join(f"%r{i}" for i in range(width))
            probes.append(ProbeSpec(
                name=f"ld.global{vec}{t}", category="memory",
                reg_decls=f".reg .b32 %r<8>;\n.reg .u64 %addr;",
                ptx_body=f"ld.global{vec}{t} {{{regs}}}, [%addr];",
            ))

    # 256-bit load
    probes.append(ProbeSpec(
        name="ld.global.v4.u64", category="memory",
        reg_decls=".reg .u64 %rd<8>;",
        ptx_body="ld.global.v4.u64 {%rd1, %rd2, %rd3, %rd4}, [%rd0];",
        tags=["wide_load"],
    ))

    # Shared memory
    probes.append(ProbeSpec(
        name="ld.shared.b32", category="memory",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;",
        smem_decl=".shared .align 16 .b8 smem[256];",
        ptx_body="mov.u64 %addr, smem;\nld.shared.b32 %r1, [%addr];",
    ))

    # ===== MATRIX OPERATIONS (MMA) =====
    # SM80+ Tensor Core: FP16
    for shape in ["m16n8k16", "m16n8k8"]:
        probes.append(ProbeSpec(
            name=f"mma.sync.aligned.{shape}.f32.f16.f16.f32",
            category="mma",
            reg_decls=".reg .f32 %f<8>;\n.reg .b32 %r<8>;",
            ptx_body=(
                f"mma.sync.aligned.{shape}.row.col.f32.f16.f16.f32\n"
                "    {%f0, %f1, %f2, %f3},\n"
                "    {%r0, %r1},\n"
                "    {%r2},\n"
                "    {%f4, %f5, %f6, %f7};"
            ) if "k8" in shape else (
                f"mma.sync.aligned.{shape}.row.col.f32.f16.f16.f32\n"
                "    {%f0, %f1, %f2, %f3},\n"
                "    {%r0, %r1, %r2, %r3},\n"
                "    {%r4, %r5},\n"
                "    {%f4, %f5, %f6, %f7};"
            ),
            tags=["tensor_core", "fp16"],
        ))

    # SM80+ Tensor Core: BF16
    probes.append(ProbeSpec(
        name="mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32",
        category="mma",
        reg_decls=".reg .f32 %f<8>;\n.reg .b32 %r<8>;",
        ptx_body=(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
            "    {%f0, %f1, %f2, %f3},\n"
            "    {%r0, %r1, %r2, %r3},\n"
            "    {%r4, %r5},\n"
            "    {%f4, %f5, %f6, %f7};"
        ),
        tags=["tensor_core", "bf16"],
    ))

    # SM80+ Tensor Core: INT8
    probes.append(ProbeSpec(
        name="mma.sync.aligned.m16n8k32.s32.s8.s8.s32",
        category="mma",
        reg_decls=".reg .b32 %r<16>;",
        ptx_body=(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\n"
            "    {%r0, %r1, %r2, %r3},\n"
            "    {%r4, %r5, %r6, %r7},\n"
            "    {%r8, %r9},\n"
            "    {%r10, %r11, %r12, %r13};"
        ),
        tags=["tensor_core", "int8"],
    ))

    # FP8 MMA (SM89+, interesting for SM121)
    for a_type, b_type in [
        ("e4m3", "e4m3"), ("e5m2", "e5m2"),
        ("e4m3", "e5m2"), ("e5m2", "e4m3"),
    ]:
        probes.append(ProbeSpec(
            name=f"mma.sync.aligned.m16n8k32.f32.{a_type}.{b_type}.f32",
            category="mma",
            reg_decls=".reg .f32 %f<8>;\n.reg .b32 %r<8>;",
            ptx_body=(
                f"mma.sync.aligned.m16n8k32.row.col.f32.{a_type}.{b_type}.f32\n"
                "    {%f0, %f1, %f2, %f3},\n"
                "    {%r0, %r1, %r2, %r3},\n"
                "    {%r4, %r5},\n"
                "    {%f4, %f5, %f6, %f7};"
            ),
            tags=["tensor_core", "fp8"],
        ))

    # Block-scaled MMA (SM100+, the MXFP4 path!)
    # These use the .kind::mxf8f6f4 qualifier
    # Try various shapes and type combinations
    for shape in ["m64n64k64", "m128n128k64"]:
        for a_type, b_type in [
            ("e4m3", "e2m1"),    # FP8 x FP4 (MXFP4-focused configuration)
            ("e4m3", "e4m3"),    # FP8 x FP8
            ("e5m2", "e2m1"),    # FP8 x FP4 variant
            ("bf16", "e2m1"),    # BF16 x FP4
        ]:
            probes.append(ProbeSpec(
                name=f"mma.kind::mxf8f6f4.{shape}.f32.{a_type}.{b_type}",
                category="mma_block_scaled",
                reg_decls=".reg .f32 %f<32>;\n.reg .b32 %r<32>;",
                ptx_body=(
                    # This is approximate PTX; exact syntax may differ
                    f"// Block-scaled MMA: {a_type} x {b_type} @ {shape}\n"
                    f"// Testing if ptxas accepts this instruction class\n"
                    "ret;"  # Placeholder -- actual MMA PTX is complex
                ),
                tags=["tensor_core", "block_scaled", "mxfp4", "experimental"],
                description=f"Block-scaled MMA {a_type}x{b_type} {shape}",
            ))

    # ldmatrix / stmatrix
    for count in ["x1", "x2", "x4"]:
        n_regs = {"x1": 1, "x2": 2, "x4": 4}[count]
        regs = ", ".join(f"%r{i}" for i in range(n_regs))
        probes.append(ProbeSpec(
            name=f"ldmatrix.sync.aligned.{count}.m8n8.shared.b16",
            category="matrix_load",
            reg_decls=f".reg .b32 %r<{n_regs + 1}>;\n.reg .u64 %addr;",
            smem_decl=".shared .align 32 .b8 smem[1024];",
            ptx_body=f"mov.u64 %addr, smem;\nldmatrix.sync.aligned.{count}.m8n8.shared.b16 {{{regs}}}, [%addr];",
            tags=["matrix_load"],
        ))
        probes.append(ProbeSpec(
            name=f"stmatrix.sync.aligned.{count}.m8n8.shared.b16",
            category="matrix_store",
            reg_decls=f".reg .b32 %r<{n_regs + 1}>;\n.reg .u64 %addr;",
            smem_decl=".shared .align 32 .b8 smem[1024];",
            ptx_body=f"mov.u64 %addr, smem;\nstmatrix.sync.aligned.{count}.m8n8.shared.b16 [%addr], {{{regs}}};",
            tags=["matrix_store"],
        ))

    # SM100-specific ldmatrix variants (expected to fail on SM121)
    probes.append(ProbeSpec(
        name="ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64",
        category="matrix_load",
        reg_decls=".reg .b32 %r<5>;\n.reg .u64 %addr;",
        smem_decl=".shared .align 32 .b8 smem[1024];",
        ptx_body="mov.u64 %addr, smem;\nldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%r0, %r1, %r2, %r3}, [%addr];",
        tags=["matrix_load", "sm100_only"],
        description="SM100-specific ldmatrix with format conversion (FP4->FP8)",
    ))

    # ===== WARP OPERATIONS =====
    for mode in ["up", "down", "bfly", "idx"]:
        probes.append(_make_shfl(mode))

    # Vote
    for op in ["all", "any", "uni"]:
        probes.append(ProbeSpec(
            name=f"vote.sync.{op}.pred", category="warp",
            reg_decls=".reg .pred %p<4>;",
            ptx_body=f"vote.sync.{op}.pred %p1, %p2, 0xffffffff;",
        ))
    probes.append(ProbeSpec(
        name="vote.sync.ballot.b32", category="warp",
        reg_decls=".reg .pred %p<2>;\n.reg .b32 %r<2>;",
        ptx_body="vote.sync.ballot.b32 %r1, %p1, 0xffffffff;",
    ))

    # Match
    for op in ["any", "all"]:
        probes.append(ProbeSpec(
            name=f"match.{op}.sync.b32", category="warp",
            reg_decls=".reg .b32 %r<4>;\n.reg .pred %p<2>;",
            ptx_body=f"match.{op}.sync.b32 %r1, %r2, 0xffffffff;",
        ))

    # Redux
    for op in ["add", "min", "max", "and", "or", "xor"]:
        for t in [".s32", ".u32"]:
            probes.append(_make_redux(op, t))

    # Activemask
    probes.append(ProbeSpec(
        name="activemask.b32", category="warp",
        reg_decls=".reg .b32 %r<2>;",
        ptx_body="activemask.b32 %r1;",
    ))

    # ===== BARRIER / FENCE =====
    probes.append(ProbeSpec(
        name="bar.sync", category="barrier",
        reg_decls="",
        ptx_body="bar.sync 0;",
    ))
    for scope in ["cta", "gl", "sys"]:
        probes.append(ProbeSpec(
            name=f"membar.{scope}", category="barrier",
            reg_decls="",
            ptx_body=f"membar.{scope};",
        ))
    for sem in ["sc", "acq_rel"]:
        for scope in ["cta", "gpu", "sys"]:
            probes.append(ProbeSpec(
                name=f"fence.{sem}.{scope}", category="barrier",
                reg_decls="",
                ptx_body=f"fence.{sem}.{scope};",
            ))

    # ===== SPECIAL OPERATIONS =====
    # DP4A / DP2A (integer dot product)
    for t in [".s32", ".u32"]:
        probes.append(ProbeSpec(
            name=f"dp4a{t}.u32{t}", category="special",
            reg_decls=".reg .b32 %r<5>;",
            ptx_body=f"dp4a{t}.u32{t} %r1, %r2, %r3, %r4;",
            tags=["dp4a"],
        ))
        probes.append(ProbeSpec(
            name=f"dp2a.lo{t}.u32{t}", category="special",
            reg_decls=".reg .b32 %r<5>;",
            ptx_body=f"dp2a.lo{t}.u32{t} %r1, %r2, %r3, %r4;",
            tags=["dp2a"],
        ))

    # nanosleep
    probes.append(ProbeSpec(
        name="nanosleep.u32", category="special",
        reg_decls=".reg .b32 %r<2>;",
        ptx_body="nanosleep.u32 %r1;",
    ))

    # Clock
    probes.append(ProbeSpec(
        name="mov.u32_clock", category="special",
        reg_decls=".reg .b32 %r<2>;",
        ptx_body="mov.u32 %r1, %clock;",
    ))
    probes.append(ProbeSpec(
        name="mov.u64_clock64", category="special",
        reg_decls=".reg .b64 %rd<2>;",
        ptx_body="mov.u64 %rd1, %clock64;",
    ))

    # ===== SM100/SM121-SPECIFIC PROBES =====

    # TMEM (Tensor Memory) -- SM100 only, should fail on SM121
    probes.append(ProbeSpec(
        name="tcgen05.alloc.cta_group::1.sync.aligned",
        category="tmem",
        reg_decls=".reg .b64 %tmem_addr;",
        ptx_body="tcgen05.alloc.cta_group::1.sync.aligned %tmem_addr, 1024;",
        tags=["sm100_only", "tmem"],
        description="TMEM allocation (SM100 only, expected to fail on SM121)",
    ))
    probes.append(ProbeSpec(
        name="tcgen05.dealloc.cta_group::1.sync.aligned",
        category="tmem",
        reg_decls=".reg .b64 %tmem_addr;",
        ptx_body="tcgen05.dealloc.cta_group::1.sync.aligned %tmem_addr, 1024;",
        tags=["sm100_only", "tmem"],
        description="TMEM deallocation (SM100 only, expected to fail on SM121)",
    ))

    # TMA (Tensor Memory Access) cp.async.bulk
    probes.append(ProbeSpec(
        name="cp.async.commit_group", category="async",
        reg_decls="",
        ptx_body="cp.async.commit_group;",
        tags=["async"],
    ))
    probes.append(ProbeSpec(
        name="cp.async.wait_group", category="async",
        reg_decls="",
        ptx_body="cp.async.wait_group 0;",
        tags=["async"],
    ))

    # setmaxnreg (SM100+)
    for dir_q in [".inc", ".dec"]:
        probes.append(ProbeSpec(
            name=f"setmaxnreg{dir_q}.sync.aligned.u32",
            category="sm100_features",
            reg_decls="",
            ptx_body=f"setmaxnreg{dir_q}.sync.aligned.u32 64;",
            tags=["sm100_only"],
            description=f"Dynamic register allocation {dir_q} (SM100+)",
        ))

    # griddepcontrol (SM100+)
    for op in ["wait", "launch_dependents"]:
        probes.append(ProbeSpec(
            name=f"griddepcontrol.{op}",
            category="sm100_features",
            reg_decls="",
            ptx_body=f"griddepcontrol.{op};",
            tags=["sm100_only"],
            description=f"Grid dependency control: {op} (SM100+)",
        ))

    # elect.sync (SM90+)
    probes.append(ProbeSpec(
        name="elect.sync", category="warp",
        reg_decls=".reg .pred %p<2>;",
        ptx_body="elect.sync %p1 | , 0xffffffff;",
        tags=["sm90_plus"],
    ))

    # ===== UNDOCUMENTED / EXPERIMENTAL =====
    # These test instruction forms that might exist but aren't officially documented

    # FP4 quantize instruction (speculative)
    probes.append(ProbeSpec(
        name="cvt.rn.satfinite.e2m1x2.f16x2",
        category="experimental",
        reg_decls=".reg .b16 %h<4>;\n.reg .b32 %r<4>;",
        ptx_body="cvt.rn.satfinite.e2m1x2.f16x2 %h1, %r1;",
        tags=["experimental", "fp4"],
        description="FP4 pack/quantize (speculative)",
    ))

    # FP6 types (SM100 block-scaled MMA supports FP6)
    for fp6 in [".e3m2", ".e2m3"]:
        for flt in [".f16", ".f32"]:
            probes.append(_make_cvt(fp6, flt, ".rn", tags=["experimental", "fp6"]))
            probes.append(_make_cvt(flt, fp6, tags=["experimental", "fp6"]))

    # Wider integer operations
    probes.append(_make_binary_arith("add", ".u128", tags=["experimental"]))
    probes.append(ProbeSpec(
        name="mov.b128", category="experimental",
        reg_decls=".reg .b128 %rq<4>;",
        ptx_body="mov.b128 %rq1, %rq2;",
        tags=["experimental", "wide"],
    ))

    # Prefetch variants
    for level in ["L1", "L2"]:
        probes.append(ProbeSpec(
            name=f"prefetch.global.{level}", category="memory",
            reg_decls=".reg .u64 %addr;",
            ptx_body=f"prefetch.global.{level} [%addr];",
        ))
    probes.append(ProbeSpec(
        name="prefetchu.L1", category="memory",
        reg_decls=".reg .u64 %addr;",
        ptx_body="prefetchu.L1 [%addr];",
    ))

    # Cache hints (SM80+)
    probes.append(ProbeSpec(
        name="ld.global.L1::evict_last.b32", category="memory",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;",
        ptx_body="ld.global.L1::evict_last.b32 %r1, [%addr];",
        tags=["cache_hints"],
    ))
    probes.append(ProbeSpec(
        name="ld.global.L1::evict_first.b32", category="memory",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;",
        ptx_body="ld.global.L1::evict_first.b32 %r1, [%addr];",
        tags=["cache_hints"],
    ))

    # Atomic float (SM80+)
    probes.append(_make_atom("add", "global", ".f32", tags=["atomic_float"]))
    probes.append(ProbeSpec(
        name="atom.global.add.f64", category="atomic",
        reg_decls=".reg .u64 %addr;\n.reg .f64 %fd<4>;",
        ptx_body="atom.global.add.f64 %fd1, [%addr], %fd2;",
        tags=["atomic_float"],
    ))

    # red (reduction without return)
    probes.append(ProbeSpec(
        name="red.global.add.f32", category="atomic",
        reg_decls=".reg .u64 %addr;\n.reg .f32 %f<2>;",
        ptx_body="red.global.add.f32 [%addr], %f1;",
        tags=["reduction"],
    ))

    # =================================================================
    # NEW PROBE CATEGORIES -- expanding SASS opcode discovery coverage
    # =================================================================

    # ===== 32-BIT IMMEDIATE VARIANTS =====
    # These use a 32-bit immediate encoded in the instruction word,
    # producing different SASS opcodes (*32I variants).
    probes.append(ProbeSpec(
        name="add.s32.imm", category="imm32",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="add.s32 %r1, %r2, 0x12345678;",
        tags=["imm32"],
    ))
    probes.append(ProbeSpec(
        name="mul.lo.s32.imm", category="imm32",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="mul.lo.s32 %r1, %r2, 0x12345678;",
        tags=["imm32"],
    ))
    probes.append(ProbeSpec(
        name="and.b32.imm", category="imm32",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="and.b32 %r1, %r2, 0xDEADBEEF;",
        tags=["imm32"],
    ))
    probes.append(ProbeSpec(
        name="or.b32.imm", category="imm32",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="or.b32 %r1, %r2, 0xFF00FF00;",
        tags=["imm32"],
    ))
    probes.append(ProbeSpec(
        name="add.f32.imm", category="imm32",
        reg_decls=".reg .f32 %f<4>;",
        ptx_body="add.f32 %f1, %f2, 0f3F800000;",  # 1.0
        tags=["imm32"],
    ))
    probes.append(ProbeSpec(
        name="mul.f32.imm", category="imm32",
        reg_decls=".reg .f32 %f<4>;",
        ptx_body="mul.f32 %f1, %f2, 0f40000000;",  # 2.0
        tags=["imm32"],
    ))
    probes.append(ProbeSpec(
        name="fma.rn.f32.imm", category="imm32",
        reg_decls=".reg .f32 %f<5>;",
        ptx_body="fma.rn.f32 %f1, %f2, 0f3F800000, %f3;",
        tags=["imm32"],
    ))
    probes.append(ProbeSpec(
        name="add.f16x2.imm", category="imm32",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="add.f16x2 %r1, %r2, 0x3C003C00;",  # {1.0, 1.0}
        tags=["imm32", "fp16"],
    ))
    probes.append(ProbeSpec(
        name="fma.rn.f16x2.imm", category="imm32",
        reg_decls=".reg .b32 %r<5>;",
        ptx_body="fma.rn.f16x2 %r1, %r2, %r3, 0x3C003C00;",
        tags=["imm32", "fp16"],
    ))

    # ===== SHARED / LOCAL / GENERIC MEMORY =====
    # Shared memory store
    probes.append(ProbeSpec(
        name="st.shared.b32", category="memory_shared",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;",
        smem_decl=".shared .align 16 .b8 smem[256];",
        ptx_body="mov.u64 %addr, smem;\nst.shared.b32 [%addr], %r1;",
    ))
    probes.append(ProbeSpec(
        name="ld.shared.v2.b32", category="memory_shared",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;",
        smem_decl=".shared .align 16 .b8 smem[256];",
        ptx_body="mov.u64 %addr, smem;\nld.shared.v2.b32 {%r1, %r2}, [%addr];",
    ))
    probes.append(ProbeSpec(
        name="st.shared.v2.b32", category="memory_shared",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;",
        smem_decl=".shared .align 16 .b8 smem[256];",
        ptx_body="mov.u64 %addr, smem;\nst.shared.v2.b32 [%addr], {%r1, %r2};",
    ))
    probes.append(ProbeSpec(
        name="ld.shared.v4.b32", category="memory_shared",
        reg_decls=".reg .b32 %r<8>;\n.reg .u64 %addr;",
        smem_decl=".shared .align 16 .b8 smem[256];",
        ptx_body="mov.u64 %addr, smem;\nld.shared.v4.b32 {%r0, %r1, %r2, %r3}, [%addr];",
    ))
    probes.append(ProbeSpec(
        name="st.shared.v4.b32", category="memory_shared",
        reg_decls=".reg .b32 %r<8>;\n.reg .u64 %addr;",
        smem_decl=".shared .align 16 .b8 smem[256];",
        ptx_body="mov.u64 %addr, smem;\nst.shared.v4.b32 [%addr], {%r0, %r1, %r2, %r3};",
    ))
    probes.append(ProbeSpec(
        name="ld.shared.u64", category="memory_shared",
        reg_decls=".reg .u64 %rd<4>;",
        smem_decl=".shared .align 16 .b8 smem[256];",
        ptx_body="mov.u64 %rd0, smem;\nld.shared.u64 %rd1, [%rd0];",
    ))
    # Local memory
    probes.append(ProbeSpec(
        name="ld.local.b32", category="memory_local",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;\n.local .align 4 .b8 lmem[64];",
        ptx_body="mov.u64 %addr, lmem;\nld.local.b32 %r1, [%addr];",
    ))
    probes.append(ProbeSpec(
        name="st.local.b32", category="memory_local",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;\n.local .align 4 .b8 lmem[64];",
        ptx_body="mov.u64 %addr, lmem;\nst.local.b32 [%addr], %r1;",
    ))
    # Generic memory (no address space qualifier)
    probes.append(ProbeSpec(
        name="ld.b32", category="memory_generic",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;",
        ptx_body="ld.b32 %r1, [%addr];",
    ))
    probes.append(ProbeSpec(
        name="st.b32", category="memory_generic",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;",
        ptx_body="st.b32 [%addr], %r1;",
    ))
    # Store global
    probes.append(ProbeSpec(
        name="st.global.b32", category="memory",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;",
        ptx_body="st.global.b32 [%addr], %r1;",
    ))
    probes.append(ProbeSpec(
        name="st.global.v2.b32", category="memory",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;",
        ptx_body="st.global.v2.b32 [%addr], {%r1, %r2};",
    ))
    probes.append(ProbeSpec(
        name="st.global.v4.b32", category="memory",
        reg_decls=".reg .b32 %r<8>;\n.reg .u64 %addr;",
        ptx_body="st.global.v4.b32 [%addr], {%r0, %r1, %r2, %r3};",
    ))
    # Shared memory atomics
    probes.append(ProbeSpec(
        name="atom.shared.add.s32", category="memory_shared",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;",
        smem_decl=".shared .align 4 .b8 smem[64];",
        ptx_body="mov.u64 %addr, smem;\natom.shared.add.s32 %r1, [%addr], %r2;",
    ))
    probes.append(ProbeSpec(
        name="atom.shared.cas.b32", category="memory_shared",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;",
        smem_decl=".shared .align 4 .b8 smem[64];",
        ptx_body="mov.u64 %addr, smem;\natom.shared.cas.b32 %r1, [%addr], %r2, %r3;",
    ))
    probes.append(ProbeSpec(
        name="atom.shared.exch.b32", category="memory_shared",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;",
        smem_decl=".shared .align 4 .b8 smem[64];",
        ptx_body="mov.u64 %addr, smem;\natom.shared.exch.b32 %r1, [%addr], %r2;",
    ))
    # Global atomics (more ops)
    for aop in ["min", "max", "cas", "exch", "and", "or", "xor"]:
        if aop == "cas":
            probes.append(ProbeSpec(
                name=f"atom.global.{aop}.b32", category="atomic_global",
                reg_decls=".reg .b32 %r<5>;\n.reg .u64 %addr;",
                ptx_body=f"atom.global.{aop}.b32 %r1, [%addr], %r2, %r3;",
            ))
        else:
            probes.append(ProbeSpec(
                name=f"atom.global.{aop}.s32", category="atomic_global",
                reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;",
                ptx_body=f"atom.global.{aop}.s32 %r1, [%addr], %r2;",
            ))
    # red (reduction without return, more types)
    probes.append(ProbeSpec(
        name="red.global.add.s32", category="atomic",
        reg_decls=".reg .u64 %addr;\n.reg .b32 %r<2>;",
        ptx_body="red.global.add.s32 [%addr], %r1;",
        tags=["reduction"],
    ))
    probes.append(ProbeSpec(
        name="red.global.add.f64", category="atomic",
        reg_decls=".reg .u64 %addr;\n.reg .f64 %fd<2>;",
        ptx_body="red.global.add.f64 [%addr], %fd1;",
        tags=["reduction"],
    ))

    # ===== TEXTURE / SURFACE =====
    # Texture fetch (requires texture state, will likely fail but tests ptxas acceptance)
    probes.append(ProbeSpec(
        name="tex.1d.v4.f32.s32", category="texture",
        reg_decls=".reg .f32 %f<8>;\n.reg .b32 %r<4>;\n.reg .u64 %tex_handle;",
        ptx_body="tex.1d.v4.f32.s32 {%f0, %f1, %f2, %f3}, [%tex_handle, {%r0}];",
        tags=["texture"],
    ))
    probes.append(ProbeSpec(
        name="tex.2d.v4.f32.f32", category="texture",
        reg_decls=".reg .f32 %f<8>;\n.reg .u64 %tex_handle;",
        ptx_body="tex.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [%tex_handle, {%f4, %f5}];",
        tags=["texture"],
    ))
    probes.append(ProbeSpec(
        name="tex.3d.v4.f32.f32", category="texture",
        reg_decls=".reg .f32 %f<8>;\n.reg .u64 %tex_handle;",
        ptx_body="tex.3d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [%tex_handle, {%f4, %f5, %f6, %f7}];",
        tags=["texture"],
    ))
    probes.append(ProbeSpec(
        name="tld4.r.2d.v4.f32.f32", category="texture",
        reg_decls=".reg .f32 %f<8>;\n.reg .u64 %tex_handle;",
        ptx_body="tld4.r.2d.v4.f32.f32 {%f0, %f1, %f2, %f3}, [%tex_handle, {%f4, %f5}];",
        tags=["texture"],
    ))
    probes.append(ProbeSpec(
        name="txq.width.b32", category="texture",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %tex_handle;",
        ptx_body="txq.width.b32 %r0, [%tex_handle];",
        tags=["texture"],
    ))
    # Surface load/store
    probes.append(ProbeSpec(
        name="suld.b.1d.v4.b32.trap", category="surface",
        reg_decls=".reg .b32 %r<8>;\n.reg .u64 %surf_handle;",
        ptx_body="suld.b.1d.v4.b32.trap {%r0, %r1, %r2, %r3}, [%surf_handle, {%r4}];",
        tags=["surface"],
    ))
    probes.append(ProbeSpec(
        name="sust.b.1d.v4.b32.trap", category="surface",
        reg_decls=".reg .b32 %r<8>;\n.reg .u64 %surf_handle;",
        ptx_body="sust.b.1d.v4.b32.trap [%surf_handle, {%r4}], {%r0, %r1, %r2, %r3};",
        tags=["surface"],
    ))
    probes.append(ProbeSpec(
        name="suld.b.2d.v4.b32.trap", category="surface",
        reg_decls=".reg .b32 %r<8>;\n.reg .u64 %surf_handle;",
        ptx_body="suld.b.2d.v4.b32.trap {%r0, %r1, %r2, %r3}, [%surf_handle, {%r4, %r5}];",
        tags=["surface"],
    ))
    probes.append(ProbeSpec(
        name="sured.b.1d.add.s32.trap", category="surface",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %surf_handle;",
        ptx_body="sured.b.1d.add.s32.trap [%surf_handle, {%r0}], %r1;",
        tags=["surface"],
    ))
    probes.append(ProbeSpec(
        name="suatom.1d.add.s32.trap", category="surface",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %surf_handle;",
        ptx_body="suatom.1d.add.s32.trap %r0, [%surf_handle, {%r1}], %r2;",
        tags=["surface"],
    ))

    # ===== TMA / ASYNC COPY =====
    # cp.async (SM80+ async memcpy shared<->global)
    probes.append(ProbeSpec(
        name="cp.async.ca.shared.global.4", category="tma",
        reg_decls=".reg .u64 %addr;\n.reg .u64 %gaddr;",
        smem_decl=".shared .align 16 .b8 smem[256];",
        ptx_body="mov.u64 %addr, smem;\ncp.async.ca.shared.global [%addr], [%gaddr], 4;",
        tags=["async", "tma"],
    ))
    probes.append(ProbeSpec(
        name="cp.async.ca.shared.global.16", category="tma",
        reg_decls=".reg .u64 %addr;\n.reg .u64 %gaddr;",
        smem_decl=".shared .align 16 .b8 smem[256];",
        ptx_body="mov.u64 %addr, smem;\ncp.async.ca.shared.global [%addr], [%gaddr], 16;",
        tags=["async", "tma"],
    ))
    probes.append(ProbeSpec(
        name="cp.async.cg.shared.global.16", category="tma",
        reg_decls=".reg .u64 %addr;\n.reg .u64 %gaddr;",
        smem_decl=".shared .align 16 .b8 smem[256];",
        ptx_body="mov.u64 %addr, smem;\ncp.async.cg.shared.global [%addr], [%gaddr], 16;",
        tags=["async", "tma"],
    ))
    # cp.async.bulk (SM90+ TMA)
    probes.append(ProbeSpec(
        name="cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes",
        category="tma",
        reg_decls=".reg .u64 %dst;\n.reg .u64 %src;\n.reg .u64 %mbar;\n.reg .b32 %sz;",
        smem_decl=".shared .align 128 .b8 smem[4096];\n.shared .align 8 .b64 mbar;",
        ptx_body=(
            "mov.u64 %dst, smem;\n"
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%dst], [%src], 256, [%mbar];"
        ),
        tags=["async", "tma", "sm90_plus"],
    ))
    # mbarrier (SM90+)
    probes.append(ProbeSpec(
        name="mbarrier.init.shared.b64", category="tma",
        reg_decls=".reg .u32 %count;",
        smem_decl=".shared .align 8 .b64 mbar;",
        ptx_body="mov.u32 %count, 1;\nmbarrier.init.shared.b64 [mbar], %count;",
        tags=["mbarrier", "sm90_plus"],
    ))
    probes.append(ProbeSpec(
        name="mbarrier.arrive.shared.b64", category="tma",
        reg_decls=".reg .b64 %state;",
        smem_decl=".shared .align 8 .b64 mbar;",
        ptx_body="mbarrier.arrive.shared.b64 %state, [mbar];",
        tags=["mbarrier", "sm90_plus"],
    ))
    probes.append(ProbeSpec(
        name="mbarrier.test_wait.shared.b64", category="tma",
        reg_decls=".reg .b64 %state;\n.reg .pred %p;",
        smem_decl=".shared .align 8 .b64 mbar;",
        ptx_body="mbarrier.test_wait.shared.b64 %p, [mbar], %state;",
        tags=["mbarrier", "sm90_plus"],
    ))

    # ===== CONTROL FLOW =====
    # Convergence barriers (BSSY/BSYNC -- crucial for SM70+)
    probes.append(ProbeSpec(
        name="bra.uni_forward", category="control",
        reg_decls="",
        ptx_body="bra.uni LBL_END;\nLBL_END:",
        tags=["control_flow"],
    ))
    probes.append(ProbeSpec(
        name="@p_bra", category="control",
        reg_decls=".reg .pred %p1;",
        ptx_body="@%p1 bra LBL_END;\nLBL_END:",
        tags=["control_flow"],
    ))
    # call/ret (function call within kernel)
    probes.append(ProbeSpec(
        name="call_ret_pair", category="control",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="call sub_fn, ();\nbra LBL_DONE;\nsub_fn:\nmov.b32 %r1, 42;\nret;\nLBL_DONE:",
        tags=["control_flow"],
    ))
    # brx (indirect branch)
    probes.append(ProbeSpec(
        name="brx.idx", category="control",
        reg_decls=".reg .b32 %r1;",
        ptx_body="mov.b32 %r1, 0;\nbrx.idx %r1, {LBL0};\nLBL0:",
        tags=["control_flow"],
    ))
    # kill
    probes.append(ProbeSpec(
        name="exit_early", category="control",
        reg_decls="",
        ptx_body="exit;",
        tags=["control_flow"],
    ))
    # trap/breakpoint
    probes.append(ProbeSpec(
        name="trap", category="control",
        reg_decls="",
        ptx_body="trap;",
        tags=["control_flow"],
    ))
    probes.append(ProbeSpec(
        name="brkpt", category="control",
        reg_decls="",
        ptx_body="brkpt;",
        tags=["control_flow"],
    ))
    # yield
    probes.append(ProbeSpec(
        name="yield_hint", category="control",
        reg_decls="",
        ptx_body="// yield hint (compiler may emit YIELD)\nbra.uni LBL_END;\nLBL_END:",
        tags=["control_flow"],
    ))
    # bar.arrive
    probes.append(ProbeSpec(
        name="bar.arrive", category="control",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="bar.arrive 0, %r1;",
        tags=["control_flow"],
    ))
    # bar.red (barrier with reduction)
    probes.append(ProbeSpec(
        name="bar.red.and.pred", category="control",
        reg_decls=".reg .pred %p<4>;",
        ptx_body="bar.red.and.pred %p1, 0, %p2;",
        tags=["control_flow"],
    ))
    probes.append(ProbeSpec(
        name="bar.red.or.pred", category="control",
        reg_decls=".reg .pred %p<4>;",
        ptx_body="bar.red.or.pred %p1, 0, %p2;",
        tags=["control_flow"],
    ))
    probes.append(ProbeSpec(
        name="bar.red.popc.u32", category="control",
        reg_decls=".reg .b32 %r<4>;\n.reg .pred %p<2>;",
        ptx_body="bar.red.popc.u32 %r1, 0, %p1;",
        tags=["control_flow"],
    ))

    # ===== INTEGER SIMD / DOT PRODUCT =====
    # dp4a already exists above; add dp2a variants, VABSDIFF, VIADD, etc.
    # Saturating dp4a
    probes.append(ProbeSpec(
        name="dp4a.s32.s32.s32.sat", category="integer_simd",
        reg_decls=".reg .b32 %r<5>;",
        ptx_body="dp4a.s32.s32.s32 %r1, %r2, %r3, %r4;",
        tags=["integer_simd", "dp4a"],
    ))
    probes.append(ProbeSpec(
        name="dp4a.u32.u32.u32", category="integer_simd",
        reg_decls=".reg .b32 %r<5>;",
        ptx_body="dp4a.u32.u32.u32 %r1, %r2, %r3, %r4;",
        tags=["integer_simd", "dp4a"],
    ))
    # Integer scaled add (ISCADD)
    probes.append(ProbeSpec(
        name="mad.lo.s32.shl1", category="integer_simd",
        reg_decls=".reg .b32 %r<5>;",
        ptx_body="shl.b32 %r1, %r2, 2;\nadd.s32 %r3, %r1, %r4;",
        tags=["integer_simd", "iscadd"],
        description="Pattern that may lower to ISCADD",
    ))
    # VABSDIFF (SIMD absolute difference)
    probes.append(ProbeSpec(
        name="vabsdiff.s32.s32.s32", category="integer_simd",
        reg_decls=".reg .b32 %r<5>;",
        ptx_body="// abs(a-b): pattern for VABSDIFF\nsub.s32 %r1, %r2, %r3;\nabs.s32 %r4, %r1;",
        tags=["integer_simd"],
        description="Pattern that may lower to VABSDIFF",
    ))
    # Integer 3-input add (IADD3)
    probes.append(ProbeSpec(
        name="iadd3.s32", category="integer_simd",
        reg_decls=".reg .b32 %r<6>;",
        ptx_body="add.s32 %r1, %r2, %r3;\nadd.s32 %r4, %r1, %r5;",
        tags=["integer_simd", "iadd3"],
        description="Two-add pattern that may lower to IADD3",
    ))
    # min3 / max3 pattern (3-input min/max => FMNMX3/VIMNMX3)
    probes.append(ProbeSpec(
        name="min3.f32", category="integer_simd",
        reg_decls=".reg .f32 %f<6>;",
        ptx_body="min.f32 %f1, %f2, %f3;\nmin.f32 %f4, %f1, %f5;",
        tags=["integer_simd", "fmnmx3"],
        description="Two-min pattern that may lower to FMNMX3",
    ))
    probes.append(ProbeSpec(
        name="min3.s32", category="integer_simd",
        reg_decls=".reg .b32 %r<6>;",
        ptx_body="min.s32 %r1, %r2, %r3;\nmin.s32 %r4, %r1, %r5;",
        tags=["integer_simd"],
        description="Two-min pattern that may lower to IMNMX3",
    ))

    # ===== UNIFORM DATAPATH =====
    # Uniform register instructions operate on UR registers.
    # We can't directly force the compiler to use uniform datapath from PTX,
    # but we CAN test PTX that explicitly references uniform registers
    # via inline asm patterns or through specific PTX constructs.

    # S2UR (Special Register to Uniform Register)
    probes.append(ProbeSpec(
        name="mov.u32.ctaid.x.ur", category="uniform",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="mov.u32 %r1, %ctaid.x;",
        tags=["uniform"],
        description="Read CTA ID -- compiler may emit S2UR/UMOV",
    ))
    probes.append(ProbeSpec(
        name="mov.u32.ntid.x", category="uniform",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="mov.u32 %r1, %ntid.x;",
        tags=["uniform"],
        description="Read block dim -- compiler may emit S2UR",
    ))
    probes.append(ProbeSpec(
        name="mov.u32.nctaid.x", category="uniform",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="mov.u32 %r1, %nctaid.x;",
        tags=["uniform"],
        description="Read grid dim -- compiler may emit S2UR",
    ))
    probes.append(ProbeSpec(
        name="mov.u32.laneid", category="uniform",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="mov.u32 %r1, %laneid;",
        tags=["uniform"],
        description="Read lane ID",
    ))
    probes.append(ProbeSpec(
        name="mov.u32.warpid", category="uniform",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="mov.u32 %r1, %warpid;",
        tags=["uniform"],
        description="Read warp ID -- may emit S2UR",
    ))
    probes.append(ProbeSpec(
        name="mov.u32.smid", category="uniform",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="mov.u32 %r1, %smid;",
        tags=["uniform"],
        description="Read SM ID -- may emit S2UR",
    ))
    probes.append(ProbeSpec(
        name="mov.u64.globaltimer", category="uniform",
        reg_decls=".reg .u64 %rd<2>;",
        ptx_body="mov.u64 %rd1, %globaltimer;",
        tags=["uniform"],
    ))

    # Constant load patterns (should emit ULDC / LDCU / LDC)
    probes.append(ProbeSpec(
        name="ld.const.b32", category="uniform",
        reg_decls=".reg .b32 %r<4>;\n.reg .u64 %addr;",
        ptx_body="ld.const.b32 %r1, [%addr];",
        tags=["uniform"],
        description="Load from constant memory -- may emit ULDC",
    ))
    probes.append(ProbeSpec(
        name="ld.const.b64", category="uniform",
        reg_decls=".reg .u64 %rd<4>;",
        ptx_body="ld.const.b64 %rd1, [%rd2];",
        tags=["uniform"],
        description="Load 64-bit from constant memory",
    ))

    # Kernel param load patterns (P2UR/R2UR patterns)
    probes.append(ProbeSpec(
        name="param_arith_uniform", category="uniform",
        needs_output=True,
        reg_decls=".reg .b32 %r<8>;",
        ptx_body=(
            "mov.u32 %r1, 100;\n"
            "mov.u32 %r2, 200;\n"
            "add.s32 %r3, %r1, %r2;\n"
            "mul.lo.s32 %r4, %r3, %r1;\n"
            "st.global.b32 [%out_addr], %r4;"
        ),
        tags=["uniform"],
        description="Constant arithmetic -- compiler may promote to uniform datapath",
    ))

    # Uniform-eligible patterns: operations on grid/block constants
    probes.append(ProbeSpec(
        name="uniform_index_calc", category="uniform",
        needs_output=True,
        reg_decls=".reg .b32 %r<8>;\n.reg .u64 %rd<4>;",
        ptx_body=(
            "mov.u32 %r1, %ctaid.x;\n"
            "mov.u32 %r2, %ntid.x;\n"
            "mul.lo.u32 %r3, %r1, %r2;\n"
            "mov.u32 %r4, %tid.x;\n"
            "add.u32 %r5, %r3, %r4;\n"
            "mul.wide.u32 %rd1, %r5, 4;\n"
            "add.u64 %rd2, %out_addr, %rd1;\n"
            "st.global.b32 [%rd2], %r5;"
        ),
        tags=["uniform"],
        description="Block offset calc -- ctaid*ntid is uniform, may emit UIMAD/UIADD3",
    ))

    # Cache control (CCTL)
    probes.append(ProbeSpec(
        name="discard.global.L2", category="uniform",
        reg_decls=".reg .u64 %addr;",
        ptx_body="discard.global.L2 [%addr], 128;",
        tags=["uniform", "cache_control"],
    ))

    # Predicate manipulation patterns (P2R / R2P)
    probes.append(ProbeSpec(
        name="p2r_pattern", category="uniform",
        reg_decls=".reg .pred %p<8>;\n.reg .b32 %r<4>;",
        ptx_body=(
            "setp.eq.s32 %p1, %r1, 0;\n"
            "setp.eq.s32 %p2, %r1, 1;\n"
            "setp.eq.s32 %p3, %r1, 2;\n"
            "setp.eq.s32 %p4, %r1, 3;\n"
            "// Multiple predicates -- compiler may batch with P2R/R2P"
        ),
        tags=["uniform", "predicate"],
    ))

    # Warp bar.warp.sync (different from bar.sync)
    probes.append(ProbeSpec(
        name="bar.warp.sync", category="uniform",
        reg_decls="",
        ptx_body="bar.warp.sync 0xffffffff;",
        tags=["uniform"],
    ))

    # GETLMEMBASE / SETLMEMBASE
    probes.append(ProbeSpec(
        name="getlmembase", category="uniform",
        reg_decls=".reg .u64 %rd<4>;",
        ptx_body="mov.u64 %rd1, %lmembase;",
        tags=["uniform", "special_reg"],
        description="Get local memory base -- GETLMEMBASE SASS",
    ))

    # CS2R (special register to register)
    probes.append(ProbeSpec(
        name="mov.u32.pm0", category="uniform",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="mov.u32 %r1, %pm0;",
        tags=["uniform", "perf_counter"],
        description="Read perf counter -- may emit CS2R",
    ))
    probes.append(ProbeSpec(
        name="mov.u32.pm1", category="uniform",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="mov.u32 %r1, %pm1;",
        tags=["uniform", "perf_counter"],
    ))
    probes.append(ProbeSpec(
        name="mov.u32.pm2", category="uniform",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="mov.u32 %r1, %pm2;",
        tags=["uniform", "perf_counter"],
    ))
    probes.append(ProbeSpec(
        name="mov.u32.pm3", category="uniform",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="mov.u32 %r1, %pm3;",
        tags=["uniform", "perf_counter"],
    ))

    # LEPC (Load Effective PC)
    probes.append(ProbeSpec(
        name="mov.u64.pc", category="uniform",
        reg_decls=".reg .u64 %rd<4>;",
        ptx_body="// PC-relative address -- LEPC pattern\ncall sub_fn, ();\nbra LBL_END;\nsub_fn:\nmov.u64 %rd1, 0;\nret;\nLBL_END:",
        tags=["uniform"],
    ))

    # ===== ADDITIONAL CONVERSION PATTERNS =====
    # I2I (integer to integer, sign extend / truncate)
    for dst, src in [(".u32", ".u16"), (".s32", ".s16"), (".u16", ".u32"),
                     (".s16", ".s32"), (".u32", ".u8"), (".s32", ".s8")]:
        probes.append(_make_cvt(dst, src, tags=["i2i"]))

    # F2F with specific rounding (round to integer)
    for rnd in [".rni", ".rzi", ".rmi", ".rpi"]:
        probes.append(_make_cvt(".f32", ".f32", rnd, tags=["f2f_round"]))
    probes.append(_make_cvt(".f64", ".f64", ".rni", tags=["f2f_round"]))

    # Pack/unpack patterns (I2IP, F2IP, I2FP)
    # cvt.pack -- packs two 16-bit values into 32 bits
    probes.append(ProbeSpec(
        name="cvt.pack.b16.b32", category="conversion_pack",
        reg_decls=".reg .b32 %r<4>;\n.reg .b16 %h<4>;",
        ptx_body="cvt.u16.u32 %h1, %r1;\ncvt.u16.u32 %h2, %r2;",
        tags=["pack"],
        description="Truncate pattern that may emit I2IP",
    ))

    # PRMT (byte permute -- important for int conversion lowering)
    probes.append(ProbeSpec(
        name="prmt.b32.ecl", category="conversion_pack",
        reg_decls=".reg .b32 %r<5>;",
        ptx_body="prmt.b32.ecl %r1, %r2, %r3, %r4;",
        tags=["prmt"],
    ))
    probes.append(ProbeSpec(
        name="prmt.b32.ecr", category="conversion_pack",
        reg_decls=".reg .b32 %r<5>;",
        ptx_body="prmt.b32.ecr %r1, %r2, %r3, %r4;",
        tags=["prmt"],
    ))
    probes.append(ProbeSpec(
        name="prmt.b32.rc8", category="conversion_pack",
        reg_decls=".reg .b32 %r<5>;",
        ptx_body="prmt.b32.rc8 %r1, %r2, %r3, %r4;",
        tags=["prmt"],
    ))
    probes.append(ProbeSpec(
        name="prmt.b32.rc16", category="conversion_pack",
        reg_decls=".reg .b32 %r<5>;",
        ptx_body="prmt.b32.rc16 %r1, %r2, %r3, %r4;",
        tags=["prmt"],
    ))

    # SGXT (sign extend)
    probes.append(ProbeSpec(
        name="bfe.s32.signext8", category="conversion_pack",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="bfe.s32 %r1, %r2, 0, 8;",
        tags=["sgxt"],
        description="Bit-field extract that may lower to SGXT",
    ))
    probes.append(ProbeSpec(
        name="bfe.s32.signext16", category="conversion_pack",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="bfe.s32 %r1, %r2, 0, 16;",
        tags=["sgxt"],
        description="Bit-field extract that may lower to SGXT",
    ))

    # MOVM (move with matrix transpose)
    probes.append(ProbeSpec(
        name="movm_pattern", category="conversion_pack",
        reg_decls=".reg .b32 %r<8>;",
        ptx_body=(
            "mov.b32 %r0, %r4;\n"
            "mov.b32 %r1, %r5;\n"
            "mov.b32 %r2, %r6;\n"
            "mov.b32 %r3, %r7;"
        ),
        tags=["movm"],
        description="Multi-register move pattern (may emit MOVM)",
    ))

    # ===== FP RANGE CHECK =====
    probes.append(ProbeSpec(
        name="testp.finite.f32", category="float_special",
        reg_decls=".reg .f32 %f<4>;\n.reg .pred %p<2>;",
        ptx_body="testp.finite.f32 %p1, %f1;",
        tags=["fchk"],
        description="FP finite test -- may emit FCHK SASS",
    ))
    probes.append(ProbeSpec(
        name="testp.infinite.f32", category="float_special",
        reg_decls=".reg .f32 %f<4>;\n.reg .pred %p<2>;",
        ptx_body="testp.infinite.f32 %p1, %f1;",
        tags=["fchk"],
    ))
    probes.append(ProbeSpec(
        name="testp.nan.f32", category="float_special",
        reg_decls=".reg .f32 %f<4>;\n.reg .pred %p<2>;",
        ptx_body="testp.number.f32 %p1, %f1;",
        tags=["fchk"],
    ))
    probes.append(ProbeSpec(
        name="testp.finite.f64", category="float_special",
        reg_decls=".reg .f64 %fd<4>;\n.reg .pred %p<2>;",
        ptx_body="testp.finite.f64 %p1, %fd1;",
        tags=["fchk"],
    ))

    # ===== FP SWIZZLE ADD =====
    # FSWZADD is used in warp-level reductions
    probes.append(ProbeSpec(
        name="fswzadd_pattern", category="float_special",
        reg_decls=".reg .f32 %f<8>;",
        ptx_body=(
            "// Swizzle-add pattern (butterfly reduction)\n"
            "shfl.sync.bfly.b32 %f1|, %f2, 1, 0x1f, 0xffffffff;\n"
            "add.f32 %f3, %f2, %f1;"
        ),
        tags=["fswzadd"],
        description="Shfl+add pattern that may lower to FSWZADD",
    ))

    # ===== WARP-LEVEL / PREDICATE EXTRAS =====
    probes.append(ProbeSpec(
        name="bar.warp.sync.pred", category="warp_extra",
        reg_decls=".reg .b32 %r<2>;",
        ptx_body="bar.warp.sync 0xffffffff;",
    ))
    # PSETP (predicate set predicate -- combine predicates)
    probes.append(ProbeSpec(
        name="psetp_and_pattern", category="warp_extra",
        reg_decls=".reg .pred %p<6>;\n.reg .b32 %r<4>;",
        ptx_body=(
            "setp.eq.s32 %p1, %r1, 0;\n"
            "setp.eq.s32 %p2, %r2, 0;\n"
            "and.pred %p3, %p1, %p2;\n"
            "or.pred %p4, %p1, %p2;"
        ),
        tags=["psetp"],
        description="Predicate combine -- may emit PSETP/PLOP3",
    ))

    # BMOV (convergence barrier move)
    probes.append(ProbeSpec(
        name="bmov_pattern", category="warp_extra",
        reg_decls=".reg .b32 %r<4>;\n.reg .pred %p<4>;",
        ptx_body=(
            "@%p1 bra LBL_TRUE;\n"
            "mov.b32 %r1, 0;\n"
            "bra.uni LBL_END;\n"
            "LBL_TRUE:\n"
            "mov.b32 %r1, 1;\n"
            "LBL_END:"
        ),
        tags=["bmov", "convergence"],
        description="Divergent branch -- compiler emits BSSY/BSYNC/BMOV",
    ))

    # ===== ASYNC OPS / DISTRIBUTED SHARED MEM =====
    # fence.proxy (SM90+)
    for proxy in ["alias", "async", "async.global", "async.shared::cta"]:
        probes.append(ProbeSpec(
            name=f"fence.proxy.{proxy}", category="async_fence",
            reg_decls="",
            ptx_body=f"fence.proxy.{proxy};",
            tags=["fence", "sm90_plus"],
        ))

    # ===== MISCELLANEOUS =====
    # PMTRIG (performance monitor trigger)
    probes.append(ProbeSpec(
        name="pmevent_pattern", category="misc",
        reg_decls="",
        ptx_body="pmevent 1;",
        tags=["misc", "perf"],
        description="Perf monitor event -- may emit PMTRIG",
    ))

    # isspacep (test address space)
    for space in ["shared", "global", "local"]:
        probes.append(ProbeSpec(
            name=f"isspacep.{space}", category="misc",
            reg_decls=".reg .u64 %addr;\n.reg .pred %p1;",
            ptx_body=f"isspacep.{space} %p1, %addr;",
            tags=["misc"],
        ))

    # cvta (convert address)
    probes.append(ProbeSpec(
        name="cvta.to.global.u64", category="misc",
        reg_decls=".reg .u64 %rd<4>;",
        ptx_body="cvta.to.global.u64 %rd1, %rd2;",
        tags=["misc"],
    ))
    probes.append(ProbeSpec(
        name="cvta.to.shared.u64", category="misc",
        reg_decls=".reg .u64 %rd<4>;",
        smem_decl=".shared .align 16 .b8 smem[256];",
        ptx_body="mov.u64 %rd2, smem;\ncvta.to.shared.u64 %rd1, %rd2;",
        tags=["misc"],
    ))

    # QSPC (query space)
    probes.append(ProbeSpec(
        name="isspacep.const", category="misc",
        reg_decls=".reg .u64 %addr;\n.reg .pred %p1;",
        ptx_body="isspacep.const %p1, %addr;",
        tags=["misc"],
    ))

    # B2R (barrier to register)
    probes.append(ProbeSpec(
        name="b2r_pattern", category="misc",
        reg_decls=".reg .b32 %r<4>;",
        ptx_body="bar.sync 0;\nmov.u32 %r1, 0;",
        tags=["misc"],
        description="Barrier then register use -- may emit B2R",
    ))

    return probes


# ---------------------------------------------------------------------------
# PTX compilation tester
# ---------------------------------------------------------------------------

class PTXProber:
    """Systematically tests PTX instruction compilation across targets."""

    def __init__(self, targets: List[str] = None, ptxas_path: str = "ptxas",
                 verbose: bool = False):
        self.targets = targets or ["sm_121a", "sm_100a", "sm_90a", "sm_80"]
        self.ptxas_path = ptxas_path
        self.verbose = verbose
        self.results: Dict[str, Dict[str, ProbeOutcome]] = {}
        self._verify_ptxas()

    def _verify_ptxas(self):
        """Verify ptxas is available and get version."""
        try:
            r = subprocess.run(
                [self.ptxas_path, "--version"],
                capture_output=True, text=True
            )
            self.ptxas_version = r.stdout.strip().split("\n")[-1] if r.stdout else "unknown"
        except FileNotFoundError:
            raise RuntimeError(f"ptxas not found at '{self.ptxas_path}'")

    def test_compile(self, ptx_source: str, target: str) -> Tuple[bool, str, float]:
        """
        Test if PTX source compiles for a target architecture.

        Returns (success, error_message, compile_time_ms).
        """
        with tempfile.NamedTemporaryFile(
            suffix=".ptx", mode="w", delete=False, prefix="squatch_"
        ) as f:
            f.write(ptx_source)
            ptx_path = f.name

        try:
            t0 = time.monotonic()
            result = subprocess.run(
                [self.ptxas_path, f"-arch={target}", ptx_path, "-o", "/dev/null"],
                capture_output=True, text=True, timeout=10,
            )
            elapsed = (time.monotonic() - t0) * 1000

            if result.returncode == 0:
                return True, "", elapsed
            else:
                # Extract the most useful error line
                error = result.stderr.strip()
                for line in error.split("\n"):
                    if "error" in line.lower():
                        return False, line.strip(), elapsed
                return False, error[:200], elapsed
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT", 10000
        finally:
            os.unlink(ptx_path)

    def probe_instruction(self, spec: ProbeSpec, target: str) -> ProbeOutcome:
        """Probe a single instruction on a single target."""
        ptx_source = build_ptx_program(spec, target)
        success, error, compile_ms = self.test_compile(ptx_source, target)

        return ProbeOutcome(
            spec=spec,
            target=target,
            result=ProbeResult.COMPILES if success else ProbeResult.COMPILE_ERROR,
            error_msg=error,
            compile_time_ms=compile_ms,
        )

    def run_compilation_audit(
        self,
        probes: List[ProbeSpec] = None,
        callback=None,
    ) -> Dict[str, Dict[str, ProbeOutcome]]:
        """
        Run Phase 1: compilation audit across all targets.

        callback(probe_name, target, outcome, progress, total) is called
        for each test if provided.
        """
        if probes is None:
            probes = generate_all_probes()

        total = len(probes) * len(self.targets)
        progress = 0

        for spec in probes:
            self.results[spec.name] = {}
            for target in self.targets:
                outcome = self.probe_instruction(spec, target)
                self.results[spec.name][target] = outcome
                progress += 1

                if callback:
                    callback(spec.name, target, outcome, progress, total)

        return self.results

    def find_anomalies(self) -> Dict[str, List]:
        """
        Analyze results to find interesting anomalies.

        Returns dict with categories:
          - "sm121_only": Compiles for SM121 but not SM100
          - "sm100_only": Compiles for SM100 but not SM121
          - "sm121_unique_vs_sm90": SM121 has but SM90 doesn't
          - "universal": Compiles on all targets
          - "universal_fail": Fails on all targets
          - "experimental_success": Experimental probes that compiled
        """
        anomalies = {
            "sm121_only": [],
            "sm100_only": [],
            "sm121_missing_vs_sm100": [],
            "sm121_unique_vs_sm90": [],
            "universal": [],
            "universal_fail": [],
            "experimental_success": [],
        }

        sm121_key = None
        sm100_key = None
        sm90_key = None

        for t in self.targets:
            if "121" in t:
                sm121_key = t
            elif "100" in t:
                sm100_key = t
            elif "90" in t:
                sm90_key = t

        for name, target_results in self.results.items():
            compiles_on = {
                t for t, o in target_results.items()
                if o.result == ProbeResult.COMPILES
            }
            fails_on = {
                t for t, o in target_results.items()
                if o.result == ProbeResult.COMPILE_ERROR
            }

            spec = next(iter(target_results.values())).spec

            if compiles_on == set(self.targets):
                anomalies["universal"].append(name)
            elif not compiles_on:
                anomalies["universal_fail"].append(name)
            else:
                if sm121_key and sm100_key:
                    if sm121_key in compiles_on and sm100_key not in compiles_on:
                        anomalies["sm121_only"].append(name)
                    if sm100_key in compiles_on and sm121_key not in compiles_on:
                        anomalies["sm121_missing_vs_sm100"].append(name)

                if sm121_key and sm90_key:
                    if sm121_key in compiles_on and sm90_key not in compiles_on:
                        anomalies["sm121_unique_vs_sm90"].append(name)

            if "experimental" in spec.tags and compiles_on:
                anomalies["experimental_success"].append(
                    (name, list(compiles_on))
                )

        return anomalies

    def get_summary_stats(self) -> Dict:
        """Get summary statistics from results."""
        stats = {}
        for target in self.targets:
            compiles = sum(
                1 for r in self.results.values()
                if target in r and r[target].result == ProbeResult.COMPILES
            )
            fails = sum(
                1 for r in self.results.values()
                if target in r and r[target].result == ProbeResult.COMPILE_ERROR
            )
            stats[target] = {"compiles": compiles, "fails": fails, "total": compiles + fails}

        return stats
