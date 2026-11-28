"""A strict numpy stand-in for CuPy, to run the GPU code paths on machines without a GPU.

Device arrays are an ndarray subclass. Like CuPy, mixing a device array with a host (numpy) array in an
operation raises, and host<->device moves need ``asarray`` / ``asnumpy``. ``cuda.Device`` records which device
each block of work ran on.
"""

from __future__ import annotations

import contextlib
import sys
import types

import numpy as np


class DeviceArray(np.ndarray):
    def __getitem__(self, key):  # fancy indexing allocates on the device as well
        out = super().__getitem__(key)
        if isinstance(out, np.ndarray) and not np.shares_memory(out, self):
            _budget(out.nbytes)
        return out

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        for x in inputs + tuple(kwargs.get("out", ()) or ()):
            if isinstance(x, np.ndarray) and not isinstance(x, DeviceArray) and x.ndim > 0:
                raise TypeError(f"host array mixed with a device array in {ufunc.__name__}")
        args = [x.view(np.ndarray) if isinstance(x, DeviceArray) else x for x in inputs]
        if "out" in kwargs:
            kwargs["out"] = tuple(o.view(np.ndarray) if isinstance(o, DeviceArray) else o for o in kwargs["out"])
        out = getattr(ufunc, method)(*args, **kwargs)
        return _wrap(out)

    def __array__(self, dtype=None, copy=None):
        if _STATE["strict_host"]:
            raise TypeError("implicit device -> host conversion (use asnumpy)")
        base = self.view(np.ndarray)
        return base if dtype is None else base.astype(dtype)

    def __float__(self):  # cupy allows float() of a 0-d / size-1 array (synchronises)
        return float(self.view(np.ndarray))


_STATE = {"strict_host": False, "devices_used": [], "current": 0, "limits": {}, "max_bytes": None}


class OutOfMemoryError(MemoryError):
    """``cupy.cuda.memory.OutOfMemoryError``."""


def _budget(nbytes: int) -> None:
    """A device with little memory: ``_STATE["max_bytes"]`` is the largest allocation that succeeds."""
    limit = _STATE.get("max_bytes")
    if limit is not None and nbytes > limit:
        _STATE["oom"] = _STATE.get("oom", 0) + 1
        raise OutOfMemoryError(f"Out of memory allocating {nbytes:,} bytes (limit set to: {limit:,})")


def _wrap(out):
    if isinstance(out, tuple):
        return tuple(_wrap(o) for o in out)
    if isinstance(out, np.ndarray) and not isinstance(out, DeviceArray):
        _budget(out.nbytes)
        return out.view(DeviceArray)
    return out


def _unwrap(x):
    if isinstance(x, DeviceArray):
        return x.view(np.ndarray)
    if isinstance(x, (list, tuple)):
        return type(x)(_unwrap(v) for v in x)
    return x


def _check_inputs(name, args):
    has_dev = any(isinstance(a, DeviceArray) for a in args)
    host = [a for a in args if isinstance(a, np.ndarray) and not isinstance(a, DeviceArray) and a.ndim > 0]
    if has_dev and host:
        raise TypeError(f"host array mixed with a device array in {name}")


class _Module(types.ModuleType):
    """``cupy``: numpy functions on device arrays, plus the CUDA bits the backend uses."""

    def __init__(self, n_devices: int = 2, mem=(8 << 30, 11 << 30)):
        super().__init__("cupy")
        self.n_devices = n_devices
        self.float32, self.float64, self.int64, self.bool_ = np.float32, np.float64, np.int64, np.bool_
        state = _STATE
        mod = self

        class Device:
            def __init__(self, dev=0):
                self.id = int(dev)

            def __enter__(self):
                if not 0 <= self.id < mod.n_devices:
                    raise RuntimeError(f"invalid device {self.id}")
                self._prev = state["current"]
                state["current"] = self.id
                state["devices_used"].append(self.id)
                return self

            def __exit__(self, *exc):
                state["current"] = self._prev
                return False

        class MemoryPool:
            def set_limit(self, size=None, fraction=None):
                state["limits"][state["current"]] = fraction if fraction is not None else size

            def free_all_blocks(self):
                state["freed"] = state.get("freed", 0) + 1

        runtime = types.SimpleNamespace(
            getDeviceCount=lambda: mod.n_devices,
            memGetInfo=lambda: mem,
            getDeviceProperties=lambda dev: {"name": b"NVIDIA GeForce RTX 2080 Ti", "totalGlobalMem": 11 << 30},
        )
        self.cuda = types.SimpleNamespace(Device=Device, runtime=runtime,
                                          memory=types.SimpleNamespace(OutOfMemoryError=OutOfMemoryError))
        self._pool = MemoryPool()
        self.linalg = types.SimpleNamespace(norm=self._fn(np.linalg.norm, "linalg.norm"))

    def get_default_memory_pool(self):
        return self._pool

    def RawKernel(self, code, name, options=()):  # noqa: N802 - CuPy's name
        """A CUDA kernel run as its C++ build (sources written for ``gpu_search``'s CPU emulation)."""
        if _STATE.get("raw_kernel_error"):
            raise RuntimeError(_STATE["raw_kernel_error"])
        import ctypes

        from cspaps.gpu_search import compile_cpu

        f = compile_cpu(code, name)

        def launch(grid, block, args, shared_mem=0):
            _STATE.setdefault("launches", []).append({"grid": grid, "block": block, "shared_mem": shared_mem,
                                                      "options": options, "device": _STATE["current"]})
            assert shared_mem <= 48 * 1024, "dynamic shared memory over 48 KB"
            c_args = []
            for a in args:
                if isinstance(a, DeviceArray):
                    assert a.flags.c_contiguous
                    c_args.append(ctypes.c_void_p(a.ctypes.data))
                elif isinstance(a, np.ndarray):
                    raise TypeError("host array passed to a kernel")
                elif isinstance(a, np.int64):
                    c_args.append(ctypes.c_longlong(int(a)))
                elif isinstance(a, np.int32):
                    c_args.append(ctypes.c_int(int(a)))
                elif isinstance(a, np.float64):
                    c_args.append(ctypes.c_double(float(a)))
                else:
                    raise TypeError(f"kernel argument of type {type(a)}: pass a numpy scalar of the C type")
            f(*c_args)

        class Kernel:
            def __call__(self, grid, block, args, shared_mem=0):
                launch(grid, block, args, shared_mem)

        return Kernel()

    @staticmethod
    def asarray(a, dtype=None):
        if isinstance(a, DeviceArray):
            return a if dtype is None else a.astype(dtype)
        out = np.array(a, dtype=dtype)
        _budget(out.nbytes)
        return out.view(DeviceArray)

    @staticmethod
    def asnumpy(a):
        if isinstance(a, DeviceArray):
            return np.array(a.view(np.ndarray))
        if isinstance(a, np.ndarray):
            raise TypeError("asnumpy of a host array")
        return np.asarray(a)

    @staticmethod
    def _fn(f, name):
        def call(*args, **kwargs):
            _check_inputs(name, list(args) + list(kwargs.values()))
            return _wrap(f(*_unwrap(args), **{k: _unwrap(v) for k, v in kwargs.items()}))

        return call

    def __getattr__(self, name):
        f = getattr(np, name)
        if isinstance(f, type) and issubclass(f, np.generic):  # dtypes (cupy.int32 is numpy.int32)
            return f
        if callable(f):
            return self._fn(f, name)
        return f


def install(monkeypatch, n_devices: int = 2):
    """Make ``cspaps.backend`` see this fake as CuPy with ``n_devices`` devices (and ``cupyx.scipy.special``)."""
    from scipy.special import ndtr

    from cspaps import backend

    fake = _Module(n_devices)
    special = types.ModuleType("cupyx.scipy.special")
    special.ndtr = fake._fn(ndtr, "ndtr")
    cupyx = types.ModuleType("cupyx")

    def scatter_add(a, slices, value):
        _check_inputs("scatter_add", [a, slices, value])
        np.add.at(a.view(np.ndarray), _unwrap(slices), _unwrap(value))

    cupyx.scatter_add = scatter_add
    cupyx.scipy = types.ModuleType("cupyx.scipy")
    cupyx.scipy.special = special
    for name, mod in (("cupy", fake), ("cupyx", cupyx), ("cupyx.scipy", cupyx.scipy), ("cupyx.scipy.special", special)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("CSPAPS_GPU", "1")
    monkeypatch.setattr(backend, "_cupy", lambda: fake)
    monkeypatch.setattr(backend, "_CONFIGURED", set())
    _STATE.update(strict_host=True, devices_used=[], current=0, limits={}, max_bytes=None, oom=0, freed=0)
    monkeypatch.setattr(sys.modules[__name__], "_STATE", _STATE)
    return fake, _STATE


@contextlib.contextmanager
def host_ok():
    """Allow implicit device -> host conversion (for assertions)."""
    old = _STATE["strict_host"]
    _STATE["strict_host"] = False
    try:
        yield
    finally:
        _STATE["strict_host"] = old
