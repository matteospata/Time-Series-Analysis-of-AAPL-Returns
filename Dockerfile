FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir .[full]

ENTRYPOINT ["python", "-m", "financial_time_series.cli"]
CMD ["run", "--ticker", "AAPL", "--years", "10", "--output", "artifacts/aapl-10y", "--lstm-epochs", "30", "--lstm-window", "20"]
