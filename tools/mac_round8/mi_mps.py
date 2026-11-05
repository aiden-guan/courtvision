"""tools/student_finetune.py mark-ignore with the labeller on the GPU (mps_student.MpsNet, the export's
student_from_onnx copy beside it as .pt): the same tool, arguments as its CLI's (src out [--promote] ...)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from mps_student import MpsNet  # noqa: E402
import student_finetune as sf  # noqa: E402

orig = sf.mark_ignore


def on_gpu(src, out, model_path, *args, **kw):
    kw["model"] = MpsNet(Path(model_path).with_suffix(".pt"))
    return orig(src, out, model_path, *args, **kw)


sf.mark_ignore = on_gpu
sys.exit(sf.main(["mark-ignore", *sys.argv[1:]]))
