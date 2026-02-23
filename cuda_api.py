#!/usr/bin/env python3
"""
CUDA Driver API bindings via ctypes for SASSquatch GPU instruction probing.

Provides low-level access to:
- GPU device information
- PTX/cubin module loading and JIT compilation
- Kernel launch and synchronization
- Device memory management

No external dependencies beyond ctypes and the CUDA driver library.
"""

import ctypes
import ctypes.util
from ctypes import (
    c_int, c_uint, c_char_p, c_void_p, c_size_t, c_ulonglong, c_char,
    byref, POINTER, create_string_buffer, cast, Structure
)
from dataclasses import dataclass
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# CUDA constants
# ---------------------------------------------------------------------------

CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1
CUDA_ERROR_OUT_OF_MEMORY = 2
CUDA_ERROR_NOT_INITIALIZED = 3
CUDA_ERROR_INVALID_IMAGE = 200
CUDA_ERROR_INVALID_CONTEXT = 201
CUDA_ERROR_NO_BINARY_FOR_GPU = 209
CUDA_ERROR_INVALID_SOURCE = 300
CUDA_ERROR_INVALID_PTX = 218
CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222
CUDA_ERROR_LAUNCH_FAILED = 719
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
CUDA_ERROR_ILLEGAL_ADDRESS = 700
CUDA_ERROR_LAUNCH_TIMEOUT = 702

# JIT options
CU_JIT_INFO_LOG_BUFFER = 0
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 1
CU_JIT_ERROR_LOG_BUFFER = 5
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
CU_JIT_LOG_VERBOSE = 12
CU_JIT_TARGET = 19
CU_JIT_OPTIMIZATION_LEVEL = 7

# Device attributes
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37
CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9

# Type aliases
CUdevice = c_int
CUcontext = c_void_p
CUmodule = c_void_p
CUfunction = c_void_p
CUdeviceptr = c_ulonglong
CUresult = c_int
CUjit_option = c_uint

# Error name mapping for common errors
CUDA_ERROR_NAMES = {
    CUDA_SUCCESS: "CUDA_SUCCESS",
    CUDA_ERROR_INVALID_VALUE: "CUDA_ERROR_INVALID_VALUE",
    CUDA_ERROR_OUT_OF_MEMORY: "CUDA_ERROR_OUT_OF_MEMORY",
    CUDA_ERROR_NOT_INITIALIZED: "CUDA_ERROR_NOT_INITIALIZED",
    CUDA_ERROR_INVALID_IMAGE: "CUDA_ERROR_INVALID_IMAGE",
    CUDA_ERROR_NO_BINARY_FOR_GPU: "CUDA_ERROR_NO_BINARY_FOR_GPU",
    CUDA_ERROR_INVALID_SOURCE: "CUDA_ERROR_INVALID_SOURCE",
    CUDA_ERROR_INVALID_PTX: "CUDA_ERROR_INVALID_PTX",
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION: "CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
    CUDA_ERROR_LAUNCH_FAILED: "CUDA_ERROR_LAUNCH_FAILED",
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
    CUDA_ERROR_ILLEGAL_INSTRUCTION: "CUDA_ERROR_ILLEGAL_INSTRUCTION",
    CUDA_ERROR_ILLEGAL_ADDRESS: "CUDA_ERROR_ILLEGAL_ADDRESS",
    CUDA_ERROR_LAUNCH_TIMEOUT: "CUDA_ERROR_LAUNCH_TIMEOUT",
}


class CUDAError(Exception):
    """CUDA driver API error."""
    def __init__(self, code: int, msg: str = ""):
        self.code = code
        name = CUDA_ERROR_NAMES.get(code, f"UNKNOWN({code})")
        super().__init__(f"{name}: {msg}" if msg else name)


@dataclass
class GPUInfo:
    """GPU device information."""
    name: str
    compute_major: int
    compute_minor: int
    sm_count: int
    warp_size: int
    max_threads_per_block: int
    max_shared_memory: int
    max_registers: int
    clock_mhz: int
    memory_clock_mhz: int
    memory_bus_width: int
    l2_cache_kb: int

    @property
    def sm_version(self) -> str:
        return f"sm_{self.compute_major}{self.compute_minor}"

    @property
    def sm_version_a(self) -> str:
        return f"sm_{self.compute_major}{self.compute_minor}a"

    def __str__(self) -> str:
        return (
            f"{self.name} | SM {self.compute_major}.{self.compute_minor} | "
            f"{self.sm_count} SMs | Warp {self.warp_size}"
        )


class CUDADriver:
    """Minimal CUDA driver API wrapper via ctypes."""

    def __init__(self, device_ordinal: int = 0):
        self.lib = self._load_library()
        self._setup_prototypes()
        self._check(self.lib.cuInit(c_uint(0)))

        # Get device
        self.device = CUdevice()
        self._check(self.lib.cuDeviceGet(byref(self.device), device_ordinal))

        # Create context
        self.context = CUcontext()
        self._check(self.lib.cuCtxCreate_v2(
            byref(self.context), c_uint(0), self.device
        ))

        self.gpu_info = self._query_gpu_info()

    def _load_library(self):
        """Find and load libcuda.so."""
        # Try ctypes.util first
        path = ctypes.util.find_library("cuda")
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                pass

        # Try common paths
        candidates = [
            "libcuda.so.1",
            "libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
            "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
            "/usr/lib/aarch64-linux-gnu/libcuda.so",
            "/usr/local/cuda/lib64/stubs/libcuda.so",
        ]
        for p in candidates:
            try:
                return ctypes.CDLL(p)
            except OSError:
                continue

        raise RuntimeError(
            "Could not find libcuda.so. "
            "Ensure NVIDIA driver is installed."
        )

    def _setup_prototypes(self):
        """Set up ctypes function prototypes."""
        L = self.lib
        protos = [
            ("cuInit", [c_uint]),
            ("cuDeviceGet", [POINTER(CUdevice), c_int]),
            ("cuDeviceGetName", [c_char_p, c_int, CUdevice]),
            ("cuDeviceGetAttribute", [POINTER(c_int), c_int, CUdevice]),
            ("cuDeviceGetCount", [POINTER(c_int)]),
            ("cuCtxCreate_v2", [POINTER(CUcontext), c_uint, CUdevice]),
            ("cuCtxDestroy_v2", [CUcontext]),
            ("cuCtxPopCurrent_v2", [POINTER(CUcontext)]),
            ("cuCtxSynchronize", []),
            ("cuDevicePrimaryCtxReset_v2", [CUdevice]),
            ("cuDevicePrimaryCtxRelease_v2", [CUdevice]),
            ("cuModuleLoadDataEx", [
                POINTER(CUmodule), c_void_p, c_uint,
                POINTER(CUjit_option), POINTER(c_void_p)
            ]),
            ("cuModuleLoadData", [POINTER(CUmodule), c_void_p]),
            ("cuModuleUnload", [CUmodule]),
            ("cuModuleGetFunction", [
                POINTER(CUfunction), CUmodule, c_char_p
            ]),
            ("cuLaunchKernel", [
                CUfunction,
                c_uint, c_uint, c_uint,  # grid
                c_uint, c_uint, c_uint,  # block
                c_uint,                  # shared mem
                c_void_p,                # stream
                POINTER(c_void_p),       # params
                POINTER(c_void_p),       # extra
            ]),
            ("cuMemAlloc_v2", [POINTER(CUdeviceptr), c_size_t]),
            ("cuMemFree_v2", [CUdeviceptr]),
            ("cuMemcpyDtoH_v2", [c_void_p, CUdeviceptr, c_size_t]),
            ("cuMemcpyHtoD_v2", [CUdeviceptr, c_void_p, c_size_t]),
            ("cuMemsetD8_v2", [CUdeviceptr, c_char, c_size_t]),
            ("cuGetErrorString", [CUresult, POINTER(c_char_p)]),
            ("cuGetErrorName", [CUresult, POINTER(c_char_p)]),
        ]
        for name, argtypes in protos:
            fn = getattr(L, name, None)
            if fn:
                fn.restype = CUresult
                fn.argtypes = argtypes

    def _check(self, result: int, context: str = ""):
        """Check CUDA result, raise CUDAError on failure."""
        if result != CUDA_SUCCESS:
            err_str = c_char_p()
            self.lib.cuGetErrorString(CUresult(result), byref(err_str))
            msg = err_str.value.decode() if err_str.value else ""
            if context:
                msg = f"{context}: {msg}"
            raise CUDAError(result, msg)

    def _get_attribute(self, attr: int) -> int:
        """Query a device attribute."""
        val = c_int()
        self._check(
            self.lib.cuDeviceGetAttribute(byref(val), attr, self.device)
        )
        return val.value

    def _query_gpu_info(self) -> GPUInfo:
        """Query GPU device information."""
        name_buf = create_string_buffer(256)
        self._check(self.lib.cuDeviceGetName(name_buf, 256, self.device))

        return GPUInfo(
            name=name_buf.value.decode(),
            compute_major=self._get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
            compute_minor=self._get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
            sm_count=self._get_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
            warp_size=self._get_attribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE),
            max_threads_per_block=self._get_attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
            max_shared_memory=self._get_attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK),
            max_registers=self._get_attribute(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK),
            clock_mhz=self._get_attribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE) // 1000,
            memory_clock_mhz=self._get_attribute(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) // 1000,
            memory_bus_width=self._get_attribute(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH),
            l2_cache_kb=self._get_attribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE) // 1024,
        )

    def load_ptx(self, ptx_source: str) -> Tuple[Optional[CUmodule], str]:
        """
        JIT-compile PTX source and load as a module.

        Returns (module, error_log). Module is None on failure.
        """
        module = CUmodule()

        # Set up JIT options for error logging
        info_buf = create_string_buffer(4096)
        error_buf = create_string_buffer(4096)
        info_size = c_void_p(4096)
        error_size = c_void_p(4096)

        options = (CUjit_option * 4)(
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        )
        values = (c_void_p * 4)(
            ctypes.cast(info_buf, c_void_p),
            info_size,
            ctypes.cast(error_buf, c_void_p),
            error_size,
        )

        ptx_bytes = ptx_source.encode("utf-8") + b"\0"
        result = self.lib.cuModuleLoadDataEx(
            byref(module),
            ptx_bytes,
            c_uint(4),
            options,
            values,
        )

        error_log = error_buf.value.decode(errors="replace").strip()

        if result != CUDA_SUCCESS:
            return None, error_log or CUDA_ERROR_NAMES.get(result, f"error {result}")

        return module, ""

    def load_cubin(self, cubin_data: bytes) -> Tuple[Optional[CUmodule], int]:
        """
        Load a cubin binary as a module.

        Returns (module, error_code). Module is None on failure.
        """
        module = CUmodule()
        buf = ctypes.create_string_buffer(cubin_data)
        result = self.lib.cuModuleLoadData(byref(module), buf)

        if result != CUDA_SUCCESS:
            return None, result

        return module, CUDA_SUCCESS

    def get_function(self, module: CUmodule, name: str) -> CUfunction:
        """Get a kernel function handle from a module."""
        func = CUfunction()
        self._check(
            self.lib.cuModuleGetFunction(
                byref(func), module, name.encode()
            ),
            f"get_function({name})"
        )
        return func

    def unload_module(self, module: CUmodule):
        """Unload a CUDA module."""
        if module:
            self.lib.cuModuleUnload(module)

    def malloc(self, size: int) -> CUdeviceptr:
        """Allocate device memory."""
        ptr = CUdeviceptr()
        self._check(
            self.lib.cuMemAlloc_v2(byref(ptr), c_size_t(size)),
            f"malloc({size})"
        )
        return ptr

    def free(self, ptr: CUdeviceptr):
        """Free device memory."""
        if ptr:
            self.lib.cuMemFree_v2(ptr)

    def memset(self, ptr: CUdeviceptr, value: int, size: int):
        """Set device memory to a byte value."""
        self._check(
            self.lib.cuMemsetD8_v2(ptr, c_char(value), c_size_t(size))
        )

    def memcpy_dtoh(self, dst: ctypes.Array, src: CUdeviceptr, size: int):
        """Copy from device to host."""
        self._check(
            self.lib.cuMemcpyDtoH_v2(dst, src, c_size_t(size)),
            "memcpy_dtoh"
        )

    def memcpy_htod(self, dst: CUdeviceptr, src, size: int):
        """Copy from host to device."""
        self._check(
            self.lib.cuMemcpyHtoD_v2(dst, src, c_size_t(size)),
            "memcpy_htod"
        )

    def launch_kernel(
        self,
        func: CUfunction,
        grid: Tuple[int, int, int] = (1, 1, 1),
        block: Tuple[int, int, int] = (1, 1, 1),
        shared_mem: int = 0,
        params: list = None,
    ) -> int:
        """
        Launch a kernel. Returns CUDA error code (0 = success).

        Does NOT raise on launch failure -- caller decides how to handle.
        """
        if params:
            # Build parameter array for cuLaunchKernel.
            # Each element of param_ptrs is a void* pointing to the
            # parameter value (which itself is a device pointer).
            param_ptrs = (c_void_p * len(params))()
            param_storage = []
            for i, p in enumerate(params):
                if isinstance(p, int):
                    storage = CUdeviceptr(p)
                elif isinstance(p, CUdeviceptr):
                    storage = CUdeviceptr(p.value)
                elif hasattr(p, "value"):
                    storage = CUdeviceptr(p.value)
                else:
                    storage = p
                param_storage.append(storage)
                param_ptrs[i] = ctypes.cast(
                    ctypes.pointer(storage), c_void_p
                )
        else:
            param_ptrs = None

        result = self.lib.cuLaunchKernel(
            func,
            c_uint(grid[0]), c_uint(grid[1]), c_uint(grid[2]),
            c_uint(block[0]), c_uint(block[1]), c_uint(block[2]),
            c_uint(shared_mem),
            c_void_p(0),  # default stream
            param_ptrs,
            c_void_p(0),  # extra
        )
        return result

    def synchronize(self) -> int:
        """Synchronize context. Returns error code."""
        return self.lib.cuCtxSynchronize()

    def launch_and_sync(
        self,
        func: CUfunction,
        grid: Tuple[int, int, int] = (1, 1, 1),
        block: Tuple[int, int, int] = (1, 1, 1),
        shared_mem: int = 0,
        params: list = None,
    ) -> int:
        """Launch kernel and synchronize. Returns final error code."""
        result = self.launch_kernel(func, grid, block, shared_mem, params)
        if result != CUDA_SUCCESS:
            return result
        return self.synchronize()

    def reset_context(self):
        """Destroy and recreate the CUDA context.

        Required after fatal kernel errors (illegal instruction, illegal
        address, launch failed) which poison the CUDA context.  All device
        pointers allocated on the old context become invalid.

        After a fatal kernel error, the GPU device itself enters an error
        state that persists even after context destruction.  We must reset
        the device's primary context to clear this state before creating
        a new context.
        """
        import time

        # 1. Destroy the poisoned context (ignore errors)
        if self.context:
            self.lib.cuCtxDestroy_v2(self.context)
            self.context = None

        # 2. Pop any remaining contexts from the stack
        dummy = CUcontext()
        while True:
            r = self.lib.cuCtxPopCurrent_v2(byref(dummy))
            if r != CUDA_SUCCESS:
                break

        # 3. Reset the device's primary context to clear device-level error
        self.lib.cuDevicePrimaryCtxReset_v2(self.device)

        # Small delay to let the GPU settle after reset
        time.sleep(0.001)

        # 4. Create a fresh context
        self.context = CUcontext()
        result = self.lib.cuCtxCreate_v2(
            byref(self.context), c_uint(0), self.device
        )
        if result != CUDA_SUCCESS:
            raise CUDAError(result, "Failed to recreate context after reset")

    def destroy(self):
        """Destroy CUDA context."""
        if self.context:
            self.lib.cuCtxDestroy_v2(self.context)
            self.context = None

    def __del__(self):
        self.destroy()
