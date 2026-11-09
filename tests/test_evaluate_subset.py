import importlib.util
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "next_purchase.py"
SPEC = importlib.util.spec_from_file_location("next_purchase", MODULE_PATH)
next_purchase = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(next_purchase)


def test_cmd_evaluate_filters_by_user_and_date_window(monkeypatch, capsys):
    df = pd.DataFrame(
        [
            {"user_id": "u1", "timestamp": "2024-01-01", "parent_asin": "p1", "asin": "a1"},
            {"user_id": "u1", "timestamp": "2024-01-05", "parent_asin": "p2", "asin": "a2"},
            {"user_id": "u1", "timestamp": "2024-01-10", "parent_asin": "p3", "asin": "a3"},
            {"user_id": "u2", "timestamp": "2024-02-01", "parent_asin": "q1", "asin": "b1"},
            {"user_id": "u2", "timestamp": "2024-02-05", "parent_asin": "q2", "asin": "b2"},
            {"user_id": "u2", "timestamp": "2024-02-10", "parent_asin": "q3", "asin": "b3"},
            {"user_id": "u3", "timestamp": "2024-03-01", "parent_asin": "r1", "asin": "c1"},
            {"user_id": "u3", "timestamp": "2024-03-05", "parent_asin": "r2", "asin": "c2"},
            {"user_id": "u3", "timestamp": "2024-03-10", "parent_asin": "r3", "asin": "c3"},
        ]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.date

    seen = []

    def fake_predict_for(past):
        seen.append(list(past))
        return [SimpleNamespace(id="q3")]

    monkeypatch.setattr(next_purchase, "predict_for", fake_predict_for)

    next_purchase.cmd_evaluate(
        df,
        user_id="u2",
        start_date=date(2024, 2, 1),
        end_date=date(2024, 2, 28),
    )

    out = capsys.readouterr().out
    assert "Tested 1 users" in out
    assert seen == [["q1", "q2"]]
