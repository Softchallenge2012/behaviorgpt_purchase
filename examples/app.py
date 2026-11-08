from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, render_template, request

from next_purchase import _apply_date_window, _parse_date_arg, cmd_predict, load

APP_DIR = Path(__file__).parent
PREDICTED_JSON = APP_DIR / "predicted_parent_asin_details.json"
ACTUAL_JSON = APP_DIR / "actual_parent_asin_after_end_date.json"

app = Flask(__name__)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _titles(payload: dict) -> list[str]:
    details = payload.get("details") or []
    out = []
    for row in details:
        title = (row.get("title") or "").strip()
        parent_asin = (row.get("parent_asin") or "").strip()
        if title:
            out.append(title)
        elif parent_asin:
            out.append(f"(no title) {parent_asin}")
        else:
            out.append("(missing title)")
    return out


@app.get("/")
def index():
    predicted_payload = {}
    actual_payload = {}

    user_id = request.args.get("user_id", "")
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    submitted = bool(user_id and start_date and end_date)

    if submitted:
        start = _parse_date_arg(start_date, "start_date")
        end = _parse_date_arg(end_date, "end_date")
        all_data = load()
        data = _apply_date_window(all_data, start_date=start, end_date=end)
        cmd_predict(data, user_id, all_df=all_data, end_date=end)
        predicted_payload = _read_json(PREDICTED_JSON)
        actual_payload = _read_json(ACTUAL_JSON)

    return render_template(
        "results.html",
        submitted=submitted,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        actual_titles=_titles(actual_payload),
        predicted_titles=_titles(predicted_payload),
        actual_path=ACTUAL_JSON.name,
        predicted_path=PREDICTED_JSON.name,
        actual_exists=ACTUAL_JSON.exists(),
        predicted_exists=PREDICTED_JSON.exists(),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
