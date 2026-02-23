#!/usr/bin/env python3
"""
SASS instruction set reference database.

Sourced from NVIDIA CUDA Binary Utilities documentation:
https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference

Section 4.4: Blackwell Instruction Set (Compute Capability 10.0 and 12.0)
This covers SM100, SM120, SM121.

Used by SASSquatch to compare discovered SASS opcodes against the
documented instruction set, identifying:
  - Documented instructions the hardware implements
  - Documented instructions the hardware does NOT implement (SM100-only?)
  - Undocumented instructions the hardware implements (hidden features)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class SASSInstructionRef:
    """Reference entry for a documented SASS instruction."""
    mnemonic: str
    description: str
    category: str
    # Which architectures document this instruction
    architectures: List[str]  # e.g., ["turing", "ampere", "hopper", "blackwell"]
    # Notes about SM121 specifics
    notes: str = ""


# ---------------------------------------------------------------------------
# Blackwell (SM100/SM120/SM121) documented SASS instruction set
# From: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#blackwell-instruction-set
# ---------------------------------------------------------------------------

BLACKWELL_INSTRUCTIONS: Dict[str, SASSInstructionRef] = {}

def _add(mnemonic: str, description: str, category: str,
         archs: List[str] = None, notes: str = ""):
    """Helper to register an instruction."""
    if archs is None:
        archs = ["blackwell"]
    BLACKWELL_INSTRUCTIONS[mnemonic] = SASSInstructionRef(
        mnemonic=mnemonic,
        description=description,
        category=category,
        architectures=archs,
        notes=notes,
    )

# ===== Floating Point Instructions =====
_add("FADD",     "FP32 Add",                              "float", ["turing", "ampere", "hopper", "blackwell"])
_add("FADD2",    "FP32 Add (paired)",                     "float", ["blackwell"], "New in Blackwell")
_add("FADD32I",  "FP32 Add (32-bit immediate)",           "float", ["turing", "ampere", "hopper", "blackwell"])
_add("FCHK",     "Floating-point Range Check",            "float", ["turing", "ampere", "hopper", "blackwell"])
_add("FFMA",     "FP32 Fused Multiply and Add",           "float", ["turing", "ampere", "hopper", "blackwell"])
_add("FFMA2",    "FP32 Fused Multiply and Add (paired)",  "float", ["blackwell"], "New in Blackwell")
_add("FFMA32I",  "FP32 Fused Multiply and Add (32-bit imm)", "float", ["turing", "ampere", "hopper", "blackwell"])
_add("FHADD",    "FP32 Addition (half-rate?)",            "float", ["blackwell"], "New in Blackwell")
_add("FHFMA",    "FP32 Fused Multiply and Add (half-rate?)", "float", ["blackwell"], "New in Blackwell")
_add("FMNMX",    "FP32 Minimum/Maximum",                  "float", ["turing", "ampere", "hopper", "blackwell"])
_add("FMNMX3",   "3-Input Floating-point Min/Max",        "float", ["blackwell"], "New in Blackwell")
_add("FMUL",     "FP32 Multiply",                         "float", ["turing", "ampere", "hopper", "blackwell"])
_add("FMUL2",    "FP32 Multiply (paired)",                "float", ["blackwell"], "New in Blackwell")
_add("FMUL32I",  "FP32 Multiply (32-bit immediate)",      "float", ["turing", "ampere", "hopper", "blackwell"])
_add("FSEL",     "Floating Point Select",                  "float", ["turing", "ampere", "hopper", "blackwell"])
_add("FSET",     "FP32 Compare And Set",                   "float", ["turing", "ampere", "hopper", "blackwell"])
_add("FSETP",    "FP32 Compare And Set Predicate",         "float", ["turing", "ampere", "hopper", "blackwell"])
_add("FSWZADD",  "FP32 Swizzle Add",                      "float", ["turing", "ampere", "hopper", "blackwell"])
_add("MUFU",     "FP32 Multi Function Operation",          "float", ["turing", "ampere", "hopper", "blackwell"])

# FP16
_add("HADD2",    "FP16 Add",                              "float16", ["turing", "ampere", "hopper", "blackwell"])
_add("HADD2_32I","FP16 Add (32-bit immediate)",           "float16", ["turing", "ampere", "hopper", "blackwell"])
_add("HFMA2",    "FP16 Fused Multiply Add",               "float16", ["turing", "ampere", "hopper", "blackwell"])
_add("HFMA2_32I","FP16 Fused Multiply Add (32-bit imm)",  "float16", ["turing", "ampere", "hopper", "blackwell"])
_add("HMMA",     "Matrix Multiply and Accumulate (FP16)",  "float16", ["turing", "ampere", "hopper", "blackwell"])
_add("HMNMX2",   "FP16 Minimum/Maximum",                  "float16", ["ampere", "hopper", "blackwell"])
_add("HMUL2",    "FP16 Multiply",                          "float16", ["turing", "ampere", "hopper", "blackwell"])
_add("HMUL2_32I","FP16 Multiply (32-bit immediate)",       "float16", ["turing", "ampere", "hopper", "blackwell"])
_add("HSET2",    "FP16 Compare And Set",                   "float16", ["turing", "ampere", "hopper", "blackwell"])
_add("HSETP2",   "FP16 Compare And Set Predicate",         "float16", ["turing", "ampere", "hopper", "blackwell"])

# FP64
_add("DADD",     "FP64 Add",                              "float64", ["turing", "ampere", "hopper", "blackwell"])
_add("DFMA",     "FP64 Fused Multiply Add",               "float64", ["turing", "ampere", "hopper", "blackwell"])
_add("DMMA",     "Matrix Multiply and Accumulate (FP64)",  "float64", ["ampere", "hopper", "blackwell"])
_add("DMUL",     "FP64 Multiply",                          "float64", ["turing", "ampere", "hopper", "blackwell"])
_add("DSETP",    "FP64 Compare And Set Predicate",         "float64", ["turing", "ampere", "hopper", "blackwell"])

# Tensor Core MMA (Blackwell-specific)
_add("OMMA",     "FP4 Matrix Multiply and Accumulate",     "tensor_core", ["blackwell"],
     "MXFP4-related instruction: FP4 MMA across a warp")
_add("QMMA",     "FP8 Matrix Multiply and Accumulate",     "tensor_core", ["blackwell"],
     "FP8 MMA across a warp. Hopper had QGMMA (warpgroup); Blackwell has warp-level QMMA")

# ===== Integer Instructions =====
_add("BMSK",     "Bitfield Mask",                          "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("BREV",     "Bit Reverse",                            "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("FLO",      "Find Leading One",                       "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("IABS",     "Integer Absolute Value",                 "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("IADD",     "Integer Addition",                       "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("IADD3",    "3-input Integer Addition",               "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("IADD32I",  "Integer Addition (32-bit immediate)",    "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("IDP",      "Integer Dot Product and Accumulate",     "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("IDP4A",    "Integer Dot Product and Accumulate",     "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("IMAD",     "Integer Multiply And Add",               "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("IMMA",     "Integer Matrix Multiply and Accumulate", "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("IMNMX",    "Integer Minimum/Maximum",                "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("IMUL",     "Integer Multiply",                       "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("IMUL32I",  "Integer Multiply (32-bit immediate)",    "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("ISCADD",   "Scaled Integer Addition",                "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("ISCADD32I","Scaled Integer Addition (32-bit imm)",   "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("ISETP",    "Integer Compare And Set Predicate",      "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("LEA",      "Load Effective Address",                 "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("LOP",      "Logic Operation",                        "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("LOP3",     "Logic Operation (3-input)",              "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("LOP32I",   "Logic Operation (32-bit immediate)",     "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("POPC",     "Population Count",                       "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("SHF",      "Funnel Shift",                           "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("SHL",      "Shift Left",                             "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("SHR",      "Shift Right",                            "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("VABSDIFF", "Absolute Difference",                    "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("VABSDIFF4","Absolute Difference (4-way)",            "integer", ["turing", "ampere", "hopper", "blackwell"])
_add("VHMNMX",   "SIMD FP16 3-Input Min/Max",             "integer", ["hopper", "blackwell"])
_add("VIADD",    "SIMD Integer Addition",                  "integer", ["hopper", "blackwell"])
_add("VIADDMNMX","SIMD Integer Add and Fused Min/Max",    "integer", ["hopper", "blackwell"])
_add("VIMNMX",   "SIMD Integer Minimum/Maximum",          "integer", ["hopper", "blackwell"])
_add("VIMNMX3",  "SIMD Integer 3-Input Min/Max",          "integer", ["hopper", "blackwell"])

# Note: BMMA (Bit MMA) is in Turing/Ampere/Hopper but NOT in Blackwell table
_add("BMMA",     "Bit Matrix Multiply and Accumulate",     "integer", ["turing", "ampere", "hopper"],
     "Present in earlier archs but ABSENT from Blackwell docs")

# ===== Conversion Instructions =====
_add("F2F",      "Float to Float Conversion",              "conversion", ["turing", "ampere", "hopper", "blackwell"])
_add("F2I",      "Float to Integer Conversion",            "conversion", ["turing", "ampere", "hopper", "blackwell"])
_add("I2F",      "Integer to Float Conversion",            "conversion", ["turing", "ampere", "hopper", "blackwell"])
_add("I2I",      "Integer to Integer Conversion",          "conversion", ["turing", "ampere", "hopper", "blackwell"])
_add("I2IP",     "Integer to Integer Conversion and Pack", "conversion", ["turing", "ampere", "hopper", "blackwell"])
_add("I2FP",     "Integer to FP32 Convert and Pack",       "conversion", ["ampere", "hopper", "blackwell"])
_add("F2IP",     "FP32 Down-Convert to Integer and Pack",  "conversion", ["ampere", "hopper", "blackwell"])
_add("FRND",     "Round to Integer",                       "conversion", ["turing", "ampere", "hopper", "blackwell"])

# ===== Movement Instructions =====
_add("MOV",      "Move",                                   "movement", ["turing", "ampere", "hopper", "blackwell"])
_add("MOV32I",   "Move (32-bit immediate)",                "movement", ["turing", "ampere", "hopper", "blackwell"])
_add("MOVM",     "Move Matrix with Transposition",         "movement", ["turing", "ampere", "hopper", "blackwell"])
_add("PRMT",     "Permute Register Pair",                  "movement", ["turing", "ampere", "hopper", "blackwell"])
_add("SEL",      "Select with Predicate",                  "movement", ["turing", "ampere", "hopper", "blackwell"])
_add("SGXT",     "Sign Extend",                            "movement", ["turing", "ampere", "hopper", "blackwell"])
_add("SHFL",     "Warp Wide Register Shuffle",             "movement", ["turing", "ampere", "hopper", "blackwell"])

# ===== Predicate Instructions =====
_add("PLOP3",    "Predicate Logic Operation",              "predicate", ["turing", "ampere", "hopper", "blackwell"])
_add("PSETP",    "Combine Predicates and Set Predicate",   "predicate", ["turing", "ampere", "hopper", "blackwell"])
_add("P2R",      "Predicate Register to Register",         "predicate", ["turing", "ampere", "hopper", "blackwell"])
_add("R2P",      "Register to Predicate Register",         "predicate", ["turing", "ampere", "hopper", "blackwell"])

# ===== Load/Store Instructions =====
_add("FENCE",    "Memory Visibility Guarantee",            "memory", ["hopper", "blackwell"])
_add("LD",       "Load from Generic Memory",               "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("LDC",      "Load Constant",                          "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("LDG",      "Load from Global Memory",                "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("LDGDEPBAR","Global Load Dependency Barrier",         "memory", ["ampere", "hopper", "blackwell"])
_add("LDGMC",    "Reducing Load",                          "memory", ["hopper", "blackwell"])
_add("LDGSTS",   "Async Global to Shared Memcopy",         "memory", ["ampere", "hopper", "blackwell"])
_add("LDL",      "Load within Local Memory",               "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("LDS",      "Load within Shared Memory",              "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("LDSM",     "Load Matrix from Shared Memory",         "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("STSM",     "Store Matrix to Shared Memory",          "memory", ["hopper", "blackwell"])
_add("ST",       "Store to Generic Memory",                "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("STG",      "Store to Global Memory",                 "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("STL",      "Store to Local Memory",                  "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("STS",      "Store to Shared Memory",                 "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("STAS",     "Async Store to Distributed Shared Mem",  "memory", ["hopper", "blackwell"])
_add("SYNCS",    "Sync Unit",                              "memory", ["hopper", "blackwell"])
_add("MATCH",    "Match Register Values Across Thread Group", "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("QSPC",     "Query Space",                            "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("ATOM",     "Atomic Operation on Generic Memory",     "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("ATOMS",    "Atomic Operation on Shared Memory",      "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("ATOMG",    "Atomic Operation on Global Memory",      "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("REDAS",    "Async Reduction on Distributed Shared Mem", "memory", ["hopper", "blackwell"])
_add("REDG",     "Reduction on Generic Memory",            "memory", ["hopper", "blackwell"])
_add("RED",      "Reduction on Generic Memory",            "memory", ["turing", "ampere"],
     "Renamed to REDG in Hopper+")
_add("CCTL",     "Cache Control",                          "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("CCTLL",    "Cache Control (L2)",                     "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("ERRBAR",   "Error Barrier",                          "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("MEMBAR",   "Memory Barrier",                         "memory", ["turing", "ampere", "hopper", "blackwell"])
_add("CCTLT",    "Texture Cache Control",                  "memory", ["turing", "ampere", "hopper", "blackwell"])

# ===== Uniform Datapath Instructions =====
_add("CREDUX",   "Coupled Reduction (Vector->Uniform)",    "uniform", ["blackwell"], "New in Blackwell")
_add("CS2UR",    "Load Constant Memory into Uniform Reg",  "uniform", ["blackwell"], "New in Blackwell")
_add("LDCU",     "Load Constant Memory into Uniform Reg",  "uniform", ["blackwell"], "New in Blackwell")
_add("R2UR",     "Vector Register to Uniform Register",    "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("REDUX",    "Reduction (Vector->Uniform Register)",   "uniform", ["ampere", "hopper", "blackwell"])
_add("S2UR",     "Special Register to Uniform Register",   "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("UBMSK",    "Uniform Bitfield Mask",                  "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("UBREV",    "Uniform Bit Reverse",                    "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("UCGABAR_ARV",  "CGA Barrier Synchronization (Arrive)", "uniform", ["hopper", "blackwell"])
_add("UCGABAR_WAIT", "CGA Barrier Synchronization (Wait)",   "uniform", ["hopper", "blackwell"])
_add("UCLEA",    "Load Effective Address for Constant",    "uniform", ["turing", "ampere", "hopper", "blackwell"])

# New Blackwell uniform float/int ops
_add("UFADD",    "Uniform FP32 Addition",                  "uniform_float", ["blackwell"], "New in Blackwell")
_add("UF2F",     "Uniform Float-to-Float Conversion",      "uniform_float", ["blackwell"], "New in Blackwell")
_add("UF2FP",    "Uniform FP32 Down-convert and Pack",     "uniform_float", ["ampere", "hopper", "blackwell"])
_add("UF2I",     "Uniform Float-to-Integer Conversion",    "uniform_float", ["blackwell"], "New in Blackwell")
_add("UF2IP",    "Uniform FP32 Down-Convert to Int Pack",  "uniform_float", ["blackwell"], "New in Blackwell")
_add("UFFMA",    "Uniform FP32 Fused Multiply-Add",        "uniform_float", ["blackwell"], "New in Blackwell")
_add("UFLO",     "Uniform Find Leading One",               "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("UFMNMX",   "Uniform Floating-point Min/Max",         "uniform_float", ["blackwell"], "New in Blackwell")
_add("UFMUL",    "Uniform FP32 Multiply",                  "uniform_float", ["blackwell"], "New in Blackwell")
_add("UFRND",    "Uniform Round to Integer",               "uniform_float", ["blackwell"], "New in Blackwell")
_add("UFSEL",    "Uniform Floating-Point Select",          "uniform_float", ["blackwell"], "New in Blackwell")
_add("UFSET",    "Uniform FP Compare and Set",             "uniform_float", ["blackwell"], "New in Blackwell")
_add("UFSETP",   "Uniform FP Compare and Set Predicate",   "uniform_float", ["blackwell"], "New in Blackwell")
_add("UI2F",     "Uniform Integer to Float Conversion",    "uniform_float", ["blackwell"], "New in Blackwell")
_add("UI2FP",    "Uniform Int to FP32 Convert and Pack",   "uniform_float", ["blackwell"], "New in Blackwell")
_add("UI2I",     "Uniform Int-to-Int Conversion",          "uniform_float", ["blackwell"], "New in Blackwell")
_add("UI2IP",    "Uniform Dual Int-to-Int Conv and Pack",  "uniform_float", ["blackwell"], "New in Blackwell")
_add("UIABS",    "Uniform Integer Absolute Value",         "uniform", ["blackwell"], "New in Blackwell")
_add("UIMNMX",   "Uniform Integer Minimum/Maximum",        "uniform", ["blackwell"], "New in Blackwell")

_add("UIADD3",   "Uniform Integer Addition",               "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("UIADD3.64","Uniform Integer Addition (64-bit)",      "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("UIMAD",    "Uniform Integer Multiplication",         "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("UISETP",   "Uniform Integer Compare and Set Pred",   "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("ULEA",     "Uniform Load Effective Address",         "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("ULEPC",    "Uniform Load Effective PC",              "uniform", ["hopper", "blackwell"])
_add("ULOP",     "Uniform Logic Operation",                "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("ULOP3",    "Uniform Logic Operation (3-input)",      "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("ULOP32I",  "Uniform Logic Operation (32-bit imm)",   "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("UMOV",     "Uniform Move",                           "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("UP2UR",    "Uniform Predicate to Uniform Register",  "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("UPLOP3",   "Uniform Predicate Logic Operation",      "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("UPOPC",    "Uniform Population Count",               "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("UPRMT",    "Uniform Byte Permute",                   "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("UPSETP",   "Uniform Predicate Logic Operation",      "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("UR2UP",    "Uniform Register to Uniform Predicate",  "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("USEL",     "Uniform Select",                         "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("USETMAXREG","Release, Deallocate and Allocate Regs", "uniform", ["hopper", "blackwell"])
_add("USGXT",    "Uniform Sign Extend",                    "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("USHF",     "Uniform Funnel Shift",                   "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("USHL",     "Uniform Left Shift",                     "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("USHR",     "Uniform Right Shift",                    "uniform", ["turing", "ampere", "hopper", "blackwell"])
_add("UGETNEXTWORKID", "Uniform Get Next Work ID",         "uniform", ["blackwell"], "New in Blackwell")
_add("UMEMSETS",  "Initialize Shared Memory",              "uniform", ["blackwell"], "New in Blackwell")
_add("UREDGR",   "Uniform Reduction on Global Mem w/ Release", "uniform", ["blackwell"], "New in Blackwell")
_add("USTGR",    "Uniform Store to Global Mem w/ Release", "uniform", ["blackwell"], "New in Blackwell")
_add("UVIADD",   "Uniform SIMD Integer Addition",          "uniform", ["blackwell"], "New in Blackwell")
_add("UVIMNMX",  "Uniform SIMD Integer Min/Max",           "uniform", ["blackwell"], "New in Blackwell")
_add("UVIRTCOUNT","Virtual Resource Management",            "uniform", ["blackwell"], "New in Blackwell")
_add("VOTEU",    "Vote (Uniform Destination)",              "uniform", ["turing", "ampere", "hopper", "blackwell"])
# ULDC is in older archs
_add("ULDC",     "Load Constant into Uniform Register",    "uniform", ["turing", "ampere", "hopper", "blackwell"])

# ===== Tensor Memory Access Instructions =====
_add("UBLKCP",   "Bulk Data Copy",                         "tma", ["hopper", "blackwell"])
_add("UBLKPF",   "Bulk Data Prefetch",                     "tma", ["hopper", "blackwell"])
_add("UBLKRED",  "Bulk Data Copy from Shared w/ Reduction","tma", ["hopper", "blackwell"])
_add("UTMACCTL", "TMA Cache Control",                      "tma", ["hopper", "blackwell"])
_add("UTMACMDFLUSH","TMA Command Flush",                   "tma", ["hopper", "blackwell"])
_add("UTMALDG",  "Tensor Load from Global to Shared Mem",  "tma", ["hopper", "blackwell"])
_add("UTMAPF",   "Tensor Prefetch",                        "tma", ["hopper", "blackwell"])
_add("UTMAREDG", "Tensor Store Shared->Global w/ Reduction","tma", ["hopper", "blackwell"])
_add("UTMASTG",  "Tensor Store from Shared to Global Mem", "tma", ["hopper", "blackwell"])

# ===== Tensor Core Memory Instructions (Blackwell-only: TMEM) =====
_add("LDT",      "Load Matrix from Tensor Memory to RF",   "tensor_memory", ["blackwell"],
     "SM100 TMEM instruction - may NOT exist on SM121")
_add("LDTM",     "Load Matrix from Tensor Memory to RF",   "tensor_memory", ["blackwell"],
     "SM100 TMEM instruction - may NOT exist on SM121")
_add("STT",      "Store Matrix to Tensor Memory from RF",  "tensor_memory", ["blackwell"],
     "SM100 TMEM instruction - may NOT exist on SM121")
_add("STTM",     "Store Matrix to Tensor Memory from RF",  "tensor_memory", ["blackwell"],
     "SM100 TMEM instruction - may NOT exist on SM121")
_add("UTCATOMSWS","Atomic on SW State Register (TC)",      "tensor_memory", ["blackwell"],
     "SM100 tensor core instruction")
_add("UTCBAR",   "Tensor Core Barrier",                    "tensor_memory", ["blackwell"],
     "SM100 tensor core barrier")
_add("UTCCP",    "Async copy Shared->Tensor Memory",       "tensor_memory", ["blackwell"],
     "SM100 TMEM instruction")
_add("UTCHMMA",  "Uniform Matrix Multiply and Accumulate (FP16)", "tensor_memory", ["blackwell"],
     "SM100 warpgroup MMA via tensor core")
_add("UTCIMMA",  "Uniform Matrix Multiply and Accumulate (INT)", "tensor_memory", ["blackwell"],
     "SM100 warpgroup MMA via tensor core")
_add("UTCOMMA",  "Uniform Matrix Multiply and Accumulate (FP4)", "tensor_memory", ["blackwell"],
     "SM100 FP4 MMA via tensor core - MXFP4 warpgroup path")
_add("UTCQMMA",  "Uniform Matrix Multiply and Accumulate (FP8)", "tensor_memory", ["blackwell"],
     "SM100 FP8 MMA via tensor core")
_add("UTCSHIFT", "Shift elements in Tensor Memory",        "tensor_memory", ["blackwell"],
     "SM100 TMEM instruction")

# ===== Texture Instructions =====
_add("TEX",      "Texture Fetch",                          "texture", ["turing", "ampere", "hopper", "blackwell"])
_add("TLD",      "Texture Load",                           "texture", ["turing", "ampere", "hopper", "blackwell"])
_add("TLD4",     "Texture Load 4",                         "texture", ["turing", "ampere", "hopper", "blackwell"])
_add("TMML",     "Texture MipMap Level",                   "texture", ["turing", "ampere", "hopper", "blackwell"])
_add("TXD",      "Texture Fetch With Derivatives",         "texture", ["turing", "ampere", "hopper", "blackwell"])
_add("TXQ",      "Texture Query",                          "texture", ["turing", "ampere", "hopper", "blackwell"])

# ===== Surface Instructions =====
_add("SUATOM",   "Atomic Op on Surface Memory",            "surface", ["turing", "ampere", "hopper", "blackwell"])
_add("SULD",     "Surface Load",                           "surface", ["turing", "ampere", "hopper", "blackwell"])
_add("SURED",    "Reduction Op on Surface Memory",         "surface", ["turing", "ampere", "hopper", "blackwell"])
_add("SUST",     "Surface Store",                          "surface", ["turing", "ampere", "hopper", "blackwell"])

# ===== Control Instructions =====
_add("ACQBULK",  "Wait for Bulk Release Warp State",       "control", ["hopper", "blackwell"])
_add("ACQSHMINIT","Wait for Shared Mem Init Release",      "control", ["blackwell"], "New in Blackwell")
_add("BMOV",     "Move Convergence Barrier State",         "control", ["turing", "ampere", "hopper", "blackwell"])
_add("BPT",      "BreakPoint/Trap",                        "control", ["turing", "ampere", "hopper", "blackwell"])
_add("BRA",      "Relative Branch",                        "control", ["turing", "ampere", "hopper", "blackwell"])
_add("BREAK",    "Break out of Convergence Barrier",       "control", ["turing", "ampere", "hopper", "blackwell"])
_add("BRX",      "Relative Branch Indirect",               "control", ["turing", "ampere", "hopper", "blackwell"])
_add("BRXU",     "Relative Branch (Uniform Offset)",       "control", ["turing", "ampere", "hopper", "blackwell"])
_add("BSSY",     "Barrier Set Convergence Sync Point",     "control", ["turing", "ampere", "hopper", "blackwell"])
_add("BSYNC",    "Synchronize on Convergence Barrier",     "control", ["turing", "ampere", "hopper", "blackwell"])
_add("CALL",     "Call Function",                          "control", ["turing", "ampere", "hopper", "blackwell"])
_add("CGAERRBAR","CGA Error Barrier",                      "control", ["hopper", "blackwell"])
_add("ELECT",    "Elect a Leader Thread",                  "control", ["hopper", "blackwell"])
_add("ENDCOLLECTIVE","Reset MCOLLECTIVE mask",             "control", ["hopper", "blackwell"])
_add("EXIT",     "Exit Program",                           "control", ["turing", "ampere", "hopper", "blackwell"])
_add("JMP",      "Absolute Jump",                          "control", ["turing", "ampere", "hopper", "blackwell"])
_add("JMX",      "Absolute Jump Indirect",                 "control", ["turing", "ampere", "hopper", "blackwell"])
_add("JMXU",     "Absolute Jump (Uniform Offset)",         "control", ["turing", "ampere", "hopper", "blackwell"])
_add("KILL",     "Kill Thread",                            "control", ["turing", "ampere", "hopper", "blackwell"])
_add("NANOSLEEP","Suspend Execution",                      "control", ["turing", "ampere", "hopper", "blackwell"])
_add("PREEXIT",  "Dependent Task Launch Hint",             "control", ["hopper", "blackwell"])
_add("RET",      "Return From Subroutine",                 "control", ["turing", "ampere", "hopper", "blackwell"])
_add("RPCMOV",   "PC Register Move",                       "control", ["turing", "ampere", "hopper", "blackwell"])
_add("WARPSYNC", "Synchronize Threads in Warp",            "control", ["turing", "ampere", "hopper", "blackwell"])
_add("YIELD",    "Yield Control",                          "control", ["turing", "ampere", "hopper", "blackwell"])

# ===== Miscellaneous Instructions =====
_add("B2R",      "Move Barrier to Register",               "misc", ["turing", "ampere", "hopper", "blackwell"])
_add("BAR",      "Barrier Synchronization",                "misc", ["turing", "ampere", "hopper", "blackwell"])
_add("CS2R",     "Move Special Register to Register",      "misc", ["turing", "ampere", "hopper", "blackwell"])
_add("DEPBAR",   "Dependency Barrier",                     "misc", ["turing", "ampere", "hopper", "blackwell"])
_add("GETLMEMBASE","Get Local Memory Base Address",        "misc", ["turing", "ampere", "hopper", "blackwell"])
_add("LEPC",     "Load Effective PC",                      "misc", ["turing", "ampere", "hopper", "blackwell"])
_add("NOP",      "No Operation",                           "misc", ["turing", "ampere", "hopper", "blackwell"])
_add("PMTRIG",   "Performance Monitor Trigger",            "misc", ["turing", "ampere", "hopper", "blackwell"])
_add("S2R",      "Move Special Register to Register",      "misc", ["turing", "ampere", "hopper", "blackwell"])
_add("SETCTAID", "Set CTA ID",                             "misc", ["turing", "ampere", "hopper", "blackwell"])
_add("SETLMEMBASE","Set Local Memory Base Address",        "misc", ["turing", "ampere", "hopper", "blackwell"])
_add("VOTE",     "Vote Across SIMT Thread Group",          "misc", ["turing", "ampere", "hopper", "blackwell"])

# ===== Hopper-specific warpgroup instructions (NOT in Blackwell docs) =====
# These may still exist on SM121 or may have been replaced by OMMA/QMMA
_add("HGMMA",    "FP16 MMA Across Warpgroup",             "warpgroup", ["hopper"],
     "Hopper warpgroup instruction. Replaced by HMMA/UTCHMMA on Blackwell?")
_add("IGMMA",    "Integer MMA Across Warpgroup",          "warpgroup", ["hopper"],
     "Hopper warpgroup instruction. Replaced by IMMA/UTCIMMA on Blackwell?")
_add("QGMMA",    "FP8 MMA Across Warpgroup",              "warpgroup", ["hopper"],
     "Hopper warpgroup instruction. Replaced by QMMA/UTCQMMA on Blackwell?")
_add("BGMMA",    "Bit MMA Across Warpgroup",              "warpgroup", ["hopper"],
     "Hopper warpgroup instruction")
_add("WARPGROUP", "Warpgroup Synchronization",             "warpgroup", ["hopper"],
     "Hopper warpgroup instruction")
_add("WARPGROUPSET","Set Warpgroup Counters",              "warpgroup", ["hopper"],
     "Hopper warpgroup instruction")


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def get_blackwell_only_instructions() -> Dict[str, SASSInstructionRef]:
    """Get instructions that are new in Blackwell (not in earlier architectures)."""
    return {
        k: v for k, v in BLACKWELL_INSTRUCTIONS.items()
        if v.architectures == ["blackwell"]
    }


def get_sm100_tmem_instructions() -> Dict[str, SASSInstructionRef]:
    """Get instructions likely SM100-only (TMEM, tensor core memory)."""
    return {
        k: v for k, v in BLACKWELL_INSTRUCTIONS.items()
        if v.category in ("tensor_memory",)
    }


def get_mxfp4_relevant_instructions() -> Dict[str, SASSInstructionRef]:
    """Get instructions relevant to MXFP4 inference workload."""
    relevant = {}
    for k, v in BLACKWELL_INSTRUCTIONS.items():
        if v.category in ("tensor_core", "tensor_memory"):
            relevant[k] = v
        elif "MMA" in k or "mma" in v.description.lower():
            relevant[k] = v
        elif "FP4" in v.description or "FP8" in v.description:
            relevant[k] = v
    return relevant


def lookup_mnemonic(mnemonic: str) -> Optional[SASSInstructionRef]:
    """Look up a SASS mnemonic in the reference database.

    Handles suffixed forms like "IMAD.WIDE.U32" by stripping suffixes.
    """
    # Try exact match first
    if mnemonic in BLACKWELL_INSTRUCTIONS:
        return BLACKWELL_INSTRUCTIONS[mnemonic]

    # Try with just the base mnemonic (e.g., "IMAD.WIDE.U32" -> "IMAD")
    base = mnemonic.split(".")[0]
    if base in BLACKWELL_INSTRUCTIONS:
        return BLACKWELL_INSTRUCTIONS[base]

    # Try progressively longer prefixes (e.g., "LDC.64" -> "LDC")
    parts = mnemonic.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in BLACKWELL_INSTRUCTIONS:
            return BLACKWELL_INSTRUCTIONS[candidate]

    return None


def classify_discovered_opcode(mnemonic: str) -> str:
    """
    Classify a discovered SASS opcode against the reference database.

    Returns one of:
      "documented"       - Known instruction, documented for Blackwell
      "documented_other" - Known instruction, documented for other arch only
      "undocumented"     - Not in the reference database at all
      "nop"              - NOP instruction
    """
    if mnemonic == "NOP":
        return "nop"

    ref = lookup_mnemonic(mnemonic)
    if ref is None:
        return "undocumented"
    elif "blackwell" in ref.architectures:
        return "documented"
    else:
        return "documented_other"


def get_all_categories() -> Dict[str, List[str]]:
    """Get all instructions organized by category."""
    cats: Dict[str, List[str]] = {}
    for k, v in BLACKWELL_INSTRUCTIONS.items():
        if v.category not in cats:
            cats[v.category] = []
        cats[v.category].append(k)
    return cats


def get_instruction_count() -> Dict[str, int]:
    """Get instruction counts by architecture."""
    counts: Dict[str, int] = {}
    for v in BLACKWELL_INSTRUCTIONS.values():
        for arch in v.architectures:
            counts[arch] = counts.get(arch, 0) + 1
    return counts
