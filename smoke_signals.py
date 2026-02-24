import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import sassquatch
from src.ptx_probe import ProbeResult


class _FakeInstruction:
    def __init__(self):
        self.offset = 0
        self.mnemonic = "MOV"
        self.operands = "R0, 0x2a"
        self.opcode_12bit = 0x123


class _FakeCubinInfo:
    def __init__(self):
        self.text_section_offset = 0x700
        self.text_section_size = 16
        self.kernel_name = "squatch_kernel"
        self.instructions = [_FakeInstruction()]


class _FakeBuilder:
    def __init__(self, target):
        self.target = target

    def compile_template(self):
        return b"\x7fELFfake"

    def parse_cubin_elf(self, _cubin_data):
        return _FakeCubinInfo()

    def extract_instructions(self, _cubin_info):
        return None

    def disassemble(self, _cubin_data):
        return ""

    def annotate_from_disasm(self, _cubin_info, _disasm):
        return None

    def disassemble_raw(self, _cubin_data):
        return ""


class _FakeSASSProber:
    def __init__(self, cuda_driver, cubin_builder):
        self.cuda_driver = cuda_driver
        self.cubin_builder = cubin_builder
        self.template_cubin = None

    def discover_opcode_field(self):
        return {
            "MOV": {
                "bits_11_0": "0x123",
                "bits_11_2": "0x048",
                "full_word_lo": "0x0000000000000123",
                "name": "template",
            }
        }

    def discover_from_probes(self, _compilable, _build_ptx_program, _target, callback=None, opt_levels=None):
        if callback:
            callback(1, 1, 1)
        return {}


class SmokeSignalTests(unittest.TestCase):
    def test_core_modules_import(self):
        modules = [
            "sassquatch",
            "src.artifact_paths",
            "src.cubin_utils",
            "src.cuda_api",
            "src.cuda_probe",
            "src.ptx_probe",
            "src.sass_probe",
            "src.sass_reference",
        ]
        for module_name in modules:
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))

    def test_cli_help_runs(self):
        result = subprocess.run(
            [sys.executable, "sassquatch.py", "--help"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout.lower())

    def test_phase2_returns_target_mapping(self):
        args = SimpleNamespace(targets=["sm_121a"])
        fake_phase1 = {
            "add.s32": {
                "sm_121a": SimpleNamespace(
                    result=ProbeResult.COMPILES,
                    spec=SimpleNamespace(name="add.s32"),
                )
            }
        }

        with mock.patch.object(sassquatch, "CubinBuilder", side_effect=lambda target: _FakeBuilder(target)), \
             mock.patch.object(sassquatch, "SASSProber", _FakeSASSProber), \
             mock.patch.object(sassquatch, "get_cuda_probe_kernels", return_value=[]), \
             mock.patch.object(sassquatch, "compile_and_discover_with_hex", return_value={}):
            result = sassquatch.run_phase2(args, phase1_results=fake_phase1)

        self.assertIn("sm_121a", result)
        self.assertIsInstance(result["sm_121a"], dict)
        self.assertGreaterEqual(len(result["sm_121a"]), 1)

    def test_export_results_writes_expected_top_level_keys(self):
        args = SimpleNamespace(targets=["sm_121a"], phase=[1, 2], verbose=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_path = tmp.name
        try:
            sassquatch.export_results(output_path, args=args)
            with open(output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for key in ["tool", "version", "timestamp", "targets", "phases_run"]:
                self.assertIn(key, payload)
        finally:
            try:
                import os
                os.unlink(output_path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
