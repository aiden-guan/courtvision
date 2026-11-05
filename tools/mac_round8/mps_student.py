"""The student on the Mac's GPU (PyTorch MPS, fp32) for the round-8 tools: the CPU is oversubscribed (load ~50 on 11
cores), where ONNX Runtime takes 13-19 s a frame. ``MpsNet`` loads a checkpoint (a ``student_from_onnx`` copy of an
export, or a training run's last.pt) and returns the five raw outputs; batched or one input at a time."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import torch  # noqa: E402
from courtvision.perception.student import OUTPUTS, build_net  # noqa: E402


class MpsNet:
    def __init__(self, path, threads=None, device="mps"):
        self.path = Path(path)
        net = build_net(3, pretrained=False)
        state = torch.load(self.path, map_location="cpu", weights_only=True)
        net.load_state_dict(state.get("model", state))
        self.net = net.eval().to(device)
        self.device = device

    def batch(self, x: np.ndarray) -> list[dict[str, np.ndarray]]:
        """(n, 9, H, W) float32 -> n dicts of raw outputs (logit heatmaps), batch dim dropped; one transfer back."""
        with torch.inference_mode():
            o = self.net(torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(self.device))
            sizes = [o[k].shape[1] for k in OUTPUTS]
            cat = torch.cat([o[k] for k in OUTPUTS], dim=1).float().cpu().numpy()
        out, c = [], 0
        parts = {}
        for k, s in zip(OUTPUTS, sizes):
            parts[k] = cat[:, c:c + s]; c += s
        return [{k: parts[k][i] for k in OUTPUTS} for i in range(len(x))]

    def __call__(self, x: np.ndarray) -> dict[str, np.ndarray]:  # detector_bench.TorchModel's interface
        return self.batch(x)[0]


def patch_bench():
    """detector_bench's .pt models on MPS (same raw outputs as TorchModel on the CPU, to float rounding)."""
    import detector_bench as db
    db.TorchModel = MpsNet
    return db
