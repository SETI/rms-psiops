---
name: feedback-no-ruff-format
description: Never run ruff format on this project — format code manually
metadata:
  type: feedback
---

Never run `ruff format` on this project. Format code manually to match the project's style.

**Why:** User explicitly prohibited it.

**How to apply:** Do not invoke `ruff format` or `scripts/run-all-checks.sh --ruff-format` as an action. Do not suggest format-on-edit hooks using ruff format. When checking style, use `ruff check` only.
