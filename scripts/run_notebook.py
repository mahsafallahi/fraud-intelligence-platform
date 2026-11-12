"""Run a project notebook end to end, refusing to start while PyCharm is open.

Usage (from the project root):
    .venv/Scripts/python.exe scripts/run_notebook.py notebooks/05_causal.ipynb
    .venv/Scripts/python.exe scripts/run_notebook.py notebooks/05_causal.ipynb --check-only

Hard stops (non-zero exit, nothing is executed):
    2  PyCharm is running (it can rewrite an open notebook during a run; on 2026-11-06
       it made the git_dirty tag of two MLflow runs ambiguous)
    3  the interpreter is not the project's .venv
    4  the notebook does not exist

Otherwise runs nbconvert in place with no timeout and returns its exit code.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def running_processes() -> list:
    """Names of the running processes."""
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, check=True).stdout
        return [line.split('","')[0].strip('"') for line in out.splitlines() if line.strip()]
    out = subprocess.run(["ps", "-eo", "comm="], capture_output=True, text=True, check=True).stdout
    return out.split()


def pycharm_processes(names) -> list:
    """The PyCharm process names among `names`."""
    return sorted({n for n in names if "pycharm" in n.lower()})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebook", help="path of the notebook, e.g. notebooks/05_causal.ipynb")
    parser.add_argument("--check-only", action="store_true", help="run the checks without executing")
    args = parser.parse_args(argv)

    found = pycharm_processes(running_processes())
    if found:
        print(f"STOP: PyCharm is running ({', '.join(found)}). Close it, then run again.", file=sys.stderr)
        return 2
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        print(f"STOP: not the project's .venv ({sys.executable}).", file=sys.stderr)
        return 3
    notebook = Path(args.notebook)
    if not notebook.exists():
        print(f"STOP: notebook not found: {notebook}", file=sys.stderr)
        return 4
    print(f"checks passed: PyCharm closed, interpreter {sys.executable}")
    if args.check_only:
        return 0
    cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
           "--ExecutePreprocessor.timeout=-1", str(notebook)]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
