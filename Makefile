PY := uv run python

.PHONY: data analysis report test lint

data:
	$(PY) analysis/00_download_data.py

analysis:
	for s in analysis/0[1-8]_*.py; do $(PY) $$s || exit 1; done
	$(PY) analysis/09_summary_table.py

report:
	$(PY) report/build.py

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check . && uv run mypy
