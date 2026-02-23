import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifact_paths import DEFAULT_ARTIFACT_DIR, resolve_scan_input_path
from src.cubin_utils import (
    disassemble_cubin,
    extract_disasm_statement_at_offset,
    find_text_section,
    instruction_file_offset,
    patch_instruction_words,
)

ap = argparse.ArgumentParser()
ap.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR, help="Artifact directory")
ap.add_argument("--input", required=False, help="Input scan JSON path")
args = ap.parse_args()

input_path = resolve_scan_input_path(args.input, args.artifact_dir)
with open(input_path) as f:
    data = json.load(f)

# Build known opcodes from Phase 2 (support by-target and single-target schemas)
p2 = data.get("phase2_opcode_map", {})
if not p2 and "phase2_opcode_map_by_target" in data:
    per_target = data.get("phase2_opcode_map_by_target", {})
    first_target = next(iter(per_target.values()), {})
    p2 = first_target if isinstance(first_target, dict) else {}
known_opcodes = set()
for mnemonic, info in p2.items():
    if isinstance(info, dict) and "bits_11_0" in info:
        try:
            opc = int(info["bits_11_0"], 16) & 0xFFF
            known_opcodes.add(opc)
        except ValueError:
            pass

# Get interesting Phase 3 opcodes (support by-target and single-target schemas)
p3 = data.get("phase3_sass", {})
if not p3 and "phase3_sass_by_target" in data:
    per_target = data.get("phase3_sass_by_target", {})
    first_target = next(iter(per_target.values()), {})
    p3 = first_target if isinstance(first_target, dict) else {}
interesting = []
for opc_hex, info in p3.items():
    if isinstance(info, dict):
        opc = int(opc_hex, 16) & 0xFFF
        r = info.get("result")
        if r in ("VALID", "WRONG_OUTPUT", "TIMEOUT") and opc not in known_opcodes:
            interesting.append((opc, r, info.get("output_value", 0)))

interesting.sort()

# Build template cubin
from src.sass_probe import CubinBuilder
builder = CubinBuilder(target="sm_121a")
template_data = builder.compile_template()

text_offset, _, _ = find_text_section(template_data)

target_idx = 1
inst_file_offset = instruction_file_offset(text_offset, target_idx)
original_word = struct.unpack_from("<Q", template_data, inst_file_offset)[0]

print("NEW opcodes: %d (accepted in probe run; not present in current Phase 2 map)" % len(interesting))
print()
print("  %-8s %-12s %s" % ("Opcode", "Status", "SASS Mnemonic"))
print("  %-8s %-12s %s" % ("------", "------", "-------------"))

for opc, result, out_val in interesting:
    new_word = (original_word & 0xFFFFFFFFFFFFF000) | (opc & 0xFFF)
    patched = patch_instruction_words(template_data, inst_file_offset, lo_word=new_word)

    try:
        out = disassemble_cubin(bytes(patched), flags=("-c",), timeout_s=5)
        mnemonic = extract_disasm_statement_at_offset(out, target_idx * 16)
    except Exception as e:
        mnemonic = "ERR:%s" % e

    status = result
    if result == "WRONG_OUTPUT":
        status = "WRONG(%s)" % out_val
    elif result == "TIMEOUT":
        status = "HANG"

    print("  0x%03x    %-12s %s" % (opc, status, mnemonic))
