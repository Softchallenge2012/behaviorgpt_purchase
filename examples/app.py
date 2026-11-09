from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, render_template, request

from next_purchase import _apply_date_window, _parse_date_arg, cmd_evaluate, cmd_predict, load

APP_DIR = Path(__file__).parent
PREDICTED_JSON = APP_DIR / "predicted_parent_asin_details.json"
ACTUAL_JSON = APP_DIR / "actual_parent_asin_after_end_date.json"

app = Flask(__name__)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_text(payload) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except TypeError:
        return str(payload)


@app.get("/")
def index():
    predicted_payload = {}
    actual_payload = {}
    predicted_text = ""
    actual_text = ""

    user_id = request.args.get("user_id", "")
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    submitted = bool(user_id and start_date and end_date)

    if submitted:
        start = _parse_date_arg(start_date, "start_date")
        end = _parse_date_arg(end_date, "end_date")
        all_data = load()
        data = _apply_date_window(all_data, start_date=start, end_date=end)
        result_details = cmd_evaluate(data, user_id=user_id, start_date=start, end_date=end)
        if result_details:
            predicted_payload = result_details[0]["result_details"].get("predicted", [])
            actual_payload = result_details[0]["result_details"].get("actual", [])
        predicted_text = _to_text(predicted_payload)
        actual_text = _to_text(actual_payload)

    return render_template(
        "results.html",
        submitted=submitted,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        actual_text=actual_text,
        predicted_text=predicted_text,
        actual_path=ACTUAL_JSON.name,
        predicted_path=PREDICTED_JSON.name,
        actual_exists=bool(actual_text),
        predicted_exists=bool(predicted_text),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
