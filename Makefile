run:
	uv run python3.13 ./main.py

build:
ruff_check:
	uv run ruff check .

ruff_fix:
	uv run ruff check --fix .

.PHONY: run ruff_check ruff_fix