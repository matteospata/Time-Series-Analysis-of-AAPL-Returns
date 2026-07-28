.PHONY: install test lint run demo clean

install:
	python -m pip install -e ".[full]"

test:
	python -m pytest -q

lint:
	ruff check src tests

run:
	python -m financial_time_series.cli run --ticker AAPL --years 10 --output artifacts/aapl-10y --lstm-epochs 30

demo:
	python -m financial_time_series.cli run --data data/raw/aapl_smoke_fixture.csv --output artifacts/demo --lstm-epochs 5 --lstm-window 10

clean:
	rm -rf artifacts .pytest_cache .ruff_cache
