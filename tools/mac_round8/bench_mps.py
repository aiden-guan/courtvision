"""tools/detector_bench.py with its .pt models on the GPU (mps_student.MpsNet)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import mps_student
db = mps_student.patch_bench()
sys.exit(db.main(sys.argv[1:]))
