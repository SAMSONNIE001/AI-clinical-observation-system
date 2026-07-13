import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = PROJECT_ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from app.core.setup_checks import gather_startup_warnings


def main() -> int:
    warnings = gather_startup_warnings()
    if not warnings:
        print("Setup check passed. All required artifacts and optional dependencies appear available.")
        return 0

    print("Setup check found warnings:")
    for warning in warnings:
        print(f"- {warning}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
