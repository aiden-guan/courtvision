"""The OCR helper cache is shared by live and background analysis."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

from courtvision.perception import vision_ocr


def test_two_first_users_can_compile_the_same_helper(tmp_path, monkeypatch):
    monkeypatch.setenv("COURTVISION_CACHE", str(tmp_path))
    compiling = threading.Barrier(2)
    outputs = []

    def compile_helper(args, **kwargs):
        path = Path(args[-1])
        outputs.append(path)
        path.write_bytes(b"compiled helper")
        compiling.wait(timeout=5)

    monkeypatch.setattr(vision_ocr.subprocess, "run", compile_helper)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(vision_ocr._binary) for _ in range(2)]
        paths = [future.result(timeout=10) for future in futures]
    assert paths[0] == paths[1] and paths[0].read_bytes() == b"compiled helper"
    assert len(set(outputs)) == 2
    assert list(tmp_path.iterdir()) == [paths[0]]
