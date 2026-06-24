"""第三周环境检查脚本。

用途：
1. 检查当前 Python 和操作系统环境。
2. 判断是否可能处于 WSL 环境。
3. 检查 PyTorch / Jupyter 是否可导入。

注意：本脚本不会安装任何软件包。
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        output = completed.stdout.strip() or completed.stderr.strip()
        return output if output else "(no output)"
    except Exception as exc:
        return f"failed: {exc!r}"


def detect_wsl() -> bool:
    if platform.system().lower() != "linux":
        return False

    os_release = Path("/proc/version")
    if os_release.exists():
        text = os_release.read_text(encoding="utf-8", errors="ignore").lower()
        return "microsoft" in text or "wsl" in text
    return False


def main() -> None:
    print("=" * 60)
    print("Basic Environment")
    print("=" * 60)
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("Machine:", platform.machine())
    print("Current directory:", Path.cwd())
    print("Conda prefix:", os.environ.get("CONDA_PREFIX", "(not in conda env)"))
    print("Possible WSL:", detect_wsl())

    print("\n" + "=" * 60)
    print("Command Availability")
    print("=" * 60)
    for command in ["python", "python3", "conda", "jupyter"]:
        print(f"{command:>8}:", shutil.which(command) or "(not found in PATH)")

    if shutil.which("conda"):
        print("\nconda info:")
        print(run_command(["conda", "info", "--envs"]))

    print("\n" + "=" * 60)
    print("Python Packages")
    print("=" * 60)
    for module_name in ["numpy", "pandas", "matplotlib", "torch", "notebook", "jupyterlab", "paddle"]:
        print(f"{module_name:>10}:", "available" if has_module(module_name) else "not installed")

    if has_module("torch"):
        import torch

        print("\nPyTorch detail:")
        print("torch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device count:", torch.cuda.device_count())
            print("CUDA device name:", torch.cuda.get_device_name(0))
        else:
            print("Using CPU-only mode is OK for this week's homework.")

    print("\n检查完成。本脚本没有安装或修改任何环境。")


if __name__ == "__main__":
    main()
