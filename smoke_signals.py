import unittest
from types import SimpleNamespace
from unittest import mock

import sassquatch
from src.ptx_probe import ProbeResult
from src.sass_probe import TEMPLATE_KERNEL_CU


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
            callback(1, 1, 2)
        return {
            "IADD3": {
                "bits_11_0": "0x321",
                "bits_11_2": "0x0c8",
                "full_word_lo": "0x0000000000000321",
                "name": "probe",
            }
        }


class SmokeSignalTests(unittest.TestCase):
    def test_template_keeps_escaped_newlines_in_inline_ptx(self):
        self.assertIn("\"{\\n\\t\"", TEMPLATE_KERNEL_CU)
        self.assertIn("\"  .reg .pred p;\\n\\t\"", TEMPLATE_KERNEL_CU)
        self.assertIn("\"}\\n\"", TEMPLATE_KERNEL_CU)

    def test_run_phase2_orchestration_smoke(self):
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
        self.assertIn("MOV", result["sm_121a"])
        self.assertIn("IADD3", result["sm_121a"])


if __name__ == "__main__":
    unittest.main()
