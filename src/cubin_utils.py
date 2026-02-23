#!/usr/bin/env python3
"""
Shared helpers for cubin ELF parsing, patching, and disassembly.

These utilities centralize common low-level operations used by multiple
opcode-discovery scripts.
"""

import os
import re
import struct
import subprocess
import tempfile
from typing import Optional, Tuple


# ELF64 header offsets
E_SHOFF = 40
E_SHENTSIZE = 58
E_SHNUM = 60
E_SHSTRNDX = 62

# Section header offsets
SH_NAME = 0
SH_TYPE = 4
SH_OFFSET = 24
SH_SIZE = 32

# Section type
SHT_PROGBITS = 1


def find_text_section(cubin_data: bytes) -> Tuple[int, int, str]:
    """
    Find a .text section in a cubin ELF and return (offset, size, name).
    """
    db = bytearray(cubin_data)

    if db[:4] != b"\x7fELF":
        raise ValueError("Not a valid ELF file")

    e_shoff = struct.unpack_from("<Q", db, E_SHOFF)[0]
    e_shentsize = struct.unpack_from("<H", db, E_SHENTSIZE)[0]
    e_shnum = struct.unpack_from("<H", db, E_SHNUM)[0]
    e_shstrndx = struct.unpack_from("<H", db, E_SHSTRNDX)[0]

    str_sh_offset = e_shoff + e_shstrndx * e_shentsize
    str_tab_offset = struct.unpack_from("<Q", db, str_sh_offset + SH_OFFSET)[0]

    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        name_idx = struct.unpack_from("<I", db, sh + SH_NAME)[0]
        sh_type = struct.unpack_from("<I", db, sh + SH_TYPE)[0]
        if sh_type != SHT_PROGBITS:
            continue

        name_end = db.index(0, str_tab_offset + name_idx)
        name = db[str_tab_offset + name_idx:name_end].decode("ascii", errors="replace")
        if name.startswith(".text"):
            offset = struct.unpack_from("<Q", db, sh + SH_OFFSET)[0]
            size = struct.unpack_from("<Q", db, sh + SH_SIZE)[0]
            return offset, size, name

    raise RuntimeError("No .text section found in cubin")


def instruction_file_offset(text_offset: int, inst_idx: int) -> int:
    """Return file offset for a 128-bit instruction index in .text."""
    return text_offset + inst_idx * 16


def patch_instruction_words(
    cubin_data: bytes,
    file_offset: int,
    lo_word: Optional[int] = None,
    hi_word: Optional[int] = None,
) -> bytearray:
    """Patch instruction words at a given file offset in cubin bytes."""
    patched = bytearray(cubin_data)
    if lo_word is not None:
        struct.pack_into("<Q", patched, file_offset, lo_word)
    if hi_word is not None:
        struct.pack_into("<Q", patched, file_offset + 8, hi_word)
    return patched


def disassemble_cubin(
    cubin_data: bytes,
    flags=("-c",),
    nvdisasm_path: str = "nvdisasm",
    timeout_s: int = 10,
) -> str:
    """Disassemble cubin bytes with nvdisasm and return stdout."""
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin_data)
        path = f.name
    try:
        result = subprocess.run(
            [nvdisasm_path] + list(flags) + [path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return result.stdout
    finally:
        if os.path.exists(path):
            os.unlink(path)


def extract_disasm_statement_at_offset(disasm_output: str, offset: int) -> str:
    """
    Extract the disassembly statement for a specific instruction byte offset.
    Returns '???' when no line is found.
    """
    # Example prefix: /*0010*/ ...
    prefix = "/*%04x*/" % offset
    for line in disasm_output.splitlines():
        if prefix in line:
            parts = line.split("*/", 1)
            if len(parts) > 1:
                return parts[1].strip().rstrip(";").strip() or "???"
    return "???"


def parse_mnemonic_from_disasm_line(disasm_stmt: str) -> str:
    """Return mnemonic token from a disassembly statement."""
    m = re.match(r"\s*([A-Za-z0-9_.]+)", disasm_stmt or "")
    return m.group(1) if m else "???"
