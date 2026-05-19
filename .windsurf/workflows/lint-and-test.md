---
description: Lint and validate the Trend Following strategy codebase
---

1. Verify the package imports cleanly from `c:\Users\ukard\OneDrive\Desktop\trading`:
   `trend_following\venv\Scripts\python.exe -c "import trend_following; from trend_following.config import TrendFollowingConfig; from trend_following.signals.generator import generate_signal; print('All imports OK')"`

2. Confirm the CLI entry point is reachable:
   `trend_following\venv\Scripts\python.exe -m trend_following.cli --help`

3. Run flake8 linting if installed (skip otherwise):
   `trend_following\venv\Scripts\python.exe -m flake8 trend_following --select=E,W --max-line-length=120 --statistics --count`
   If flake8 is not installed: `trend_following\venv\Scripts\python.exe -m pip install flake8 --quiet` then re-run.

4. There are no formal pytest test files in this project. If the user wants to add tests, suggest creating `trend_following/tests/` and running:
   `trend_following\venv\Scripts\python.exe -m pytest trend_following/tests/ -v`

5. Report any import errors, CLI failures, or lint violations found.
