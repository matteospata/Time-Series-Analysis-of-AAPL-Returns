from pathlib import Path

from financial_time_series.pipeline import PipelineConfig, resolve_start_date, run_pipeline


def test_ten_year_window_is_resolved_deterministically():
    assert resolve_start_date(None, "2026-07-28", 10) == "2016-07-28"


def test_demo_pipeline_writes_reports_and_plots(tmp_path):
    data_path = Path(__file__).parents[1] / "data" / "raw" / "aapl_smoke_fixture.csv"
    result = run_pipeline(PipelineConfig(data_path=str(data_path), output_dir=str(tmp_path), lstm_window=10, lstm_epochs=1, test_size=0.2))
    artifact_dir = Path(result["artifact_dir"])
    assert (artifact_dir / "report.json").exists()
    assert (artifact_dir / "acf_pacf.png").exists()
    assert (artifact_dir / "var_backtest.png").exists()
    assert result["report"]["models"]["garch"]["backend"] in {"arch", "scipy-fallback"}
