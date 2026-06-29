from pathlib import Path
import runpy
import sys


SCRIPT = Path(__file__).resolve().parent / "dreamer4-src" / "dreamer4" / "train_dynamics.py"

SCRIPT_DIR = SCRIPT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")