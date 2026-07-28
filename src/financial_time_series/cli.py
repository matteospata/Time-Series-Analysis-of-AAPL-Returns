from __future__ import annotations

import argparse
import json

from financial_time_series.pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research-grade financial time-series analysis pipeline.")
    run = parser.add_subparsers(dest="command", required=True).add_parser("run", help="Run the complete analysis pipeline.")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="Local CSV containing Date and Close/Adj Close columns.")
    source.add_argument("--ticker", help="Ticker to download through yfinance, e.g. AAPL.")
    run.add_argument("--start", help="Inclusive start date. If omitted, calculated from --years.")
    run.add_argument("--end")
    run.add_argument("--years", type=int, default=10, help="Lookback window for live ticker mode.")
    run.add_argument("--output", default="artifacts")
    run.add_argument("--test-size", type=float, default=0.2)
    run.add_argument("--confidence", type=float, default=0.99)
    run.add_argument("--var-window", type=int, default=252)
    run.add_argument("--lstm-window", type=int, default=20)
    run.add_argument("--lstm-epochs", type=int, default=30)
    run.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(ticker=args.ticker or "AAPL", start=args.start, end=args.end, years=args.years, data_path=args.data, output_dir=args.output, test_size=args.test_size, confidence=args.confidence, var_window=args.var_window, lstm_window=args.lstm_window, lstm_epochs=args.lstm_epochs, seed=args.seed)
    result = run_pipeline(config)
    print(json.dumps({"artifact_dir": result["artifact_dir"], "forecast_metrics": result["report"]["forecast_metrics"], "var_backtests": result["report"]["var_backtests"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
