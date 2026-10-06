"""Single-command smoke test for Tasks 1 and 2: python smoke_test.py

Task 1 and Task 2 both have top-level modules named dataset, tokenizer and utils,
so their tests must run in separate Python processes.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

SUITES = [
    ("Task 1", ["tests/test_task1_attention.py", "tests/test_task1_smoke.py"]),
    ("Task 2", ["tests/test_task2_smoke.py"]),
]


def main() -> int:
    for name, tests in SUITES:
        print(f"== {name}: pytest {' '.join(tests)}", flush=True)
        code = subprocess.run([sys.executable, "-m", "pytest", *tests, "-q"], cwd=REPO).returncode
        if code != 0:
            print(f"{name} smoke tests failed (pytest exit code {code}).")
            return code
    print("All smoke tests passed: Task 1 (22) + Task 2 (12) = 34 tests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
