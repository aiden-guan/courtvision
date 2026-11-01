"""Scheduling for dedicated live-analysis threads; numerical behaviour is unaffected."""

from __future__ import annotations

import ctypes
import sys
import threading

_state = threading.local()
_opencv_lock = threading.Lock()
_opencv_threads: int | None = None


def configure_opencv(threads: int) -> None:
    """Set the app's native pool before the first frame, avoiding repeated process-wide reconfiguration."""
    global _opencv_threads
    with _opencv_lock:
        if _opencv_threads != threads:
            import cv2

            cv2.setNumThreads(threads)
            _opencv_threads = threads


def configure_live_thread() -> None:
    """Request macOS's user-initiated QoS for work the viewer is waiting for. Best effort, current thread only;
    other platforms retain their scheduler. Avoid the interactive/UI priority and any process-wide nice change."""
    if getattr(_state, "configured", False):
        return
    _state.configured = True
    if sys.platform != "darwin":
        return
    try:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.pthread_self.restype = ctypes.c_void_p
        get_qos = lib.pthread_get_qos_class_np
        get_qos.argtypes, get_qos.restype = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint),
                                           ctypes.POINTER(ctypes.c_int)], ctypes.c_int
        current, relative = ctypes.c_uint(), ctypes.c_int()
        if get_qos(lib.pthread_self(), ctypes.byref(current), ctypes.byref(relative)) == 0 and current.value >= 0x19:
            return  # preserve an existing higher priority
        set_qos = lib.pthread_set_qos_class_self_np
        set_qos.argtypes, set_qos.restype = [ctypes.c_uint, ctypes.c_int], ctypes.c_int
        set_qos(0x19, 0)  # QOS_CLASS_USER_INITIATED, normal relative priority (sys/qos.h)
    except (OSError, AttributeError):
        pass
