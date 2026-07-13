import subprocess
import sys
from pathlib import Path

REQUIREMENTS = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "opencv-python",
    "torch",
    "torchvision",
    "ultralytics",
    "mediapipe",
    "pytest",
    "openai",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def install_packages() -> int:
    command = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
    result = subprocess.run(command)
    if result.returncode != 0:
        return result.returncode

    command = [sys.executable, "-m", "pip", "install"] + REQUIREMENTS
    return subprocess.run(command).returncode


def main() -> int:
    print("Installing required Python packages...")
    return install_packages()


if __name__ == "__main__":
    raise SystemExit(main())
