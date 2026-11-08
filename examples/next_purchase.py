"""Next-purchase prediction with BehaviorGPT (Unbox AI Python SDK).

Usage (the API key is read from the UNBOXAI_API_KEY environment variable):
    python next_purchase.py check [CATALOG_ID]    # does the catalog recognise our ids?
    python next_purchase.py evaluate [CATALOG_ID] # hide each user's last purchase, see if we find it
    python next_purchase.py predict [USER_ID]     # next purchases for every user -> predictions.csv
    python next_purchase.py predict USER_ID       # next purchases for one user
    python next_purchase.py predict USER_ID START_DATE END_DATE
    python next_purchase.py predict --start-date YYYY-MM-DD --end-date YYYY-MM-DD

Catalog selection priority:
1) CATALOG_ID argument (for check/evaluate)
2) UNBOXAI_CATALOG_ID environment variable
3) retail_catalog
"""
import sys
import os
import json
import pandas as pd
from behaviorgpt import Order, UnboxAIClient
from behaviorgpt._exceptions import UnboxAIError

# ---- Settings you may need to change after running `check` -------------------
CSV_PATH = "groceries_user.csv"
META_CSV_PATH = "meta_groceries.csv"
CATALOG_ID = os.environ.get("UNBOXAI_CATALOG_ID", "cat_701a6b1727de3ad6936af4d4")
ID_COLUMN = "parent_asin"   # which column holds the catalog's item id: "parent_asin" or "asin"
ID_PREFIX = ""              # some catalogs prefix ids, e.g. "pa" + parent_asin. Set it if `check` shows so.
TOP_K = 10                  # how many next purchases to return per user
MAX_HISTORY = 50            # use only each user's most recent purchases (keeps calls small)
ONLY_VERIFIED = True        # count only verified purchases as purchases
EVAL_USERS = 200            # how many users to test in `evaluate`
# -------------------------------------------------------------------------------

if len(sys.argv) > 2 and sys.argv[2].startswith("cat_"):
    CATALOG_ID = sys.argv[2]

client = UnboxAIClient(market="us", default_catalog_id=CATALOG_ID)  # reads UNBOXAI_API_KEY itself


def explain_unknown_catalog(err):
    msg = str(err)
    if "unknown catalog" not in msg.lower():
        return False

    print(f"Catalog not found: {CATALOG_ID}")
    print("Set a valid catalog id via UNBOXAI_CATALOG_ID or pass one after the command, e.g.:")
    print("  python next_purchase.py check cat_xxx")
    print("\nCatalogs visible to this API key:")
    try:
        cats = client.list_catalogs()
        if not cats:
            print("  (none)")
        for c in cats:
            print(f"  {c.catalog_id} | status={c.status} | owned={c.owned}")
    except Exception as list_err:  # noqa: BLE001
        print("  Could not list catalogs:", type(list_err).__name__, list_err)
    return True


def load():
    df = pd.read_csv(CSV_PATH)
    if ONLY_VERIFIED:
        df = df[df["verified_purchase"]]
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.date
    df = df.sort_values(["user_id", "timestamp"])
    return df


def _parse_date_arg(raw, name):
    try:
        return pd.to_datetime(raw).date()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Invalid {name} value {raw!r}. Use YYYY-MM-DD.") from exc


def _parse_window_args(argv):
    args = []
    start_date = None
    end_date = None
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--start-date":
            if i + 1 >= len(argv):
                raise SystemExit("Missing value for --start-date")
            start_date = _parse_date_arg(argv[i + 1], "--start-date")
            i += 2
            continue
        if token == "--end-date":
            if i + 1 >= len(argv):
                raise SystemExit("Missing value for --end-date")
            end_date = _parse_date_arg(argv[i + 1], "--end-date")
            i += 2
            continue
        if token.startswith("--start-date="):
            start_date = _parse_date_arg(token.split("=", 1)[1], "--start-date")
            i += 1
            continue
        if token.startswith("--end-date="):
            end_date = _parse_date_arg(token.split("=", 1)[1], "--end-date")
            i += 1
            continue
        args.append(token)
        i += 1

    if start_date and end_date and start_date > end_date:
        raise SystemExit("--start-date must be <= --end-date")

    return args, start_date, end_date


def _apply_date_window(df, start_date=None, end_date=None):
    if start_date is not None:
        df = df[df["timestamp"] >= start_date]
    if end_date is not None:
        df = df[df["timestamp"] <= end_date]
    return df


def histories(df):
    """user_id -> list of catalog item ids, oldest first, each item once (first time bought)."""
    out = {}
    for uid, g in df.groupby("user_id"):
        ids = list(dict.fromkeys(ID_PREFIX + str(x) for x in g[ID_COLUMN]))
        out[uid] = ids
    return out


def lookup(df):
    """catalog id -> (parent_asin, asin), built from our own data file."""
    m = {}
    for p, a, col in zip(df["parent_asin"], df["asin"], df[ID_COLUMN]):
        m[ID_PREFIX + str(col)] = (p, a)
    return m


def load_meta_lookup(path=META_CSV_PATH):
    df = pd.read_csv(path, dtype=str).fillna("")
    if "parent_asin" not in df.columns:
        return {}
    df = df.drop_duplicates(subset=["parent_asin"], keep="first")
    return df.set_index("parent_asin").to_dict(orient="index")


def user_parent_asins_after(df, user_id, end_date):
    if end_date is None:
        return []
    mask = (df["user_id"] == user_id) & (df["timestamp"] > end_date)
    future = df.loc[mask, "parent_asin"].astype(str).tolist()
    return list(dict.fromkeys(future))


def print_parent_asin_details(header, parent_asins, meta_lookup):
    print(f"\n{header}")
    if not parent_asins:
        print("  (none)")
        return

    detail_fields = [
        "title",
        "main_category",
        "store",
        "price",
        "average_rating",
        "rating_number",
        "categories",
        "features",
        "description",
        "catalog_id",
    ]

    for asin in parent_asins:
        details = meta_lookup.get(asin)
        if not details:
            print(f"- {asin}: metadata not found in {META_CSV_PATH}")
            continue
        compact = {"parent_asin": asin}
        for field in detail_fields:
            value = details.get(field, "")
            if value:
                compact[field] = value
        print(json.dumps(compact, ensure_ascii=True))


def build_parent_asin_details(parent_asins, meta_lookup):
    detail_fields = [
        "title",
        "main_category",
        "store",
        "price",
        "average_rating",
        "rating_number",
        "categories",
        "features",
        "description",
        "catalog_id",
    ]
    rows = []
    for asin in parent_asins:
        details = meta_lookup.get(asin)
        if not details:
            rows.append({"parent_asin": asin, "metadata_found": False})
            continue
        compact = {"parent_asin": asin, "metadata_found": True}
        for field in detail_fields:
            value = details.get(field, "")
            if value:
                compact[field] = value
        rows.append(compact)
    return rows


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def predict_for(seen, k=TOP_K):
    """Rank what comes after the user's purchases, dropping anything already bought."""
    history = [Order([i]) for i in seen]
    res = client.complete(history, catalog_id=CATALOG_ID, limit=k * 3)
    seen_set = set(seen)
    return [it for it in res.items if it.id not in seen_set][:k]


def to_parent_asin_and_asin(item, lk):
    """Prefer the catalog row; fall back to our own data file."""
    d = item.data or {}
    if d.get("parent_asin") and d.get("asin"):
        return d["parent_asin"], d["asin"]
    if item.id in lk:
        return lk[item.id]
    raw = item.id[len(ID_PREFIX):] if ID_PREFIX and item.id.startswith(ID_PREFIX) else item.id
    return (raw, "") if ID_COLUMN == "parent_asin" else ("", raw)


def cmd_check(df):
    hist = histories(df)
    sample = [i for ids in list(hist.values())[:5] for i in ids[:1]]
    print("Our ids look like:", sample)
    print("\nWhat the catalog returns with no history (cold start):")
    for it in client.complete([], catalog_id=CATALOG_ID, limit=5).items:
        print("  id:", it.id, "| row:", it.data)
    print("\nDoes the catalog know our ids?")
    for i in sample:
        try:
            print("  ", i, "->", client.shop.item(i).name)
        except Exception as e:  # noqa: BLE001
            print("  ", i, "-> NOT FOUND:", type(e).__name__)


def cmd_evaluate(df):
    hist = {u: ids for u, ids in histories(df).items() if len(ids) >= 3}
    users = list(hist)[:EVAL_USERS]
    hits = tried = 0
    for u in users:
        ids = hist[u]
        past, target = ids[:-1], ids[-1]
        try:
            preds = [it.id for it in predict_for(past)]
        except Exception as e:  # noqa: BLE001
            print("skip", u, type(e).__name__, e)
            continue
        tried += 1
        hits += target in preds
    print(f"Tested {tried} users with 3+ purchases. Last purchase found in top {TOP_K}: "
          f"{hits}/{tried} = {hits / max(tried, 1):.1%}")


def cmd_predict(df, only_user=None, all_df=None, end_date=None):
    lk = lookup(df)
    meta_lookup = load_meta_lookup(META_CSV_PATH)
    hist = histories(df)
    users = [only_user] if only_user else list(hist)
    rows = []
    for u in users:
        if u not in hist:
            print("No purchases found for", u)
            continue
        for rank, it in enumerate(predict_for(hist[u]), 1):
            p, a = to_parent_asin_and_asin(it, lk)
            rows.append({"user_id": u, "rank": rank, "parent_asin": p, "asin": a, "score": it.score})
    out = pd.DataFrame(rows)
    print(out.head(20).to_string(index=False))

    if only_user:
        predicted_parent_asins = []
        if not out.empty:
            predicted_parent_asins = list(dict.fromkeys(out["parent_asin"].astype(str).tolist()))
        print_parent_asin_details(
            f"Detailed metadata for predicted parent_asin values (user_id={only_user}):",
            predicted_parent_asins,
            meta_lookup,
        )

        future_parent_asins = []
        if all_df is not None and end_date is not None:
            future_parent_asins = user_parent_asins_after(all_df, only_user, end_date)
        print_parent_asin_details(
            f"Detailed metadata for purchased parent_asin values after end date (user_id={only_user}, end_date={end_date}):",
            future_parent_asins,
            meta_lookup,
        )

        predicted_details = build_parent_asin_details(predicted_parent_asins, meta_lookup)
        future_details = build_parent_asin_details(future_parent_asins, meta_lookup)

        predicted_json_path = f"predicted_parent_asin_details_{only_user}.json"
        if end_date is None:
            future_json_path = f"actual_parent_asin_after_end_date_{only_user}.json"
        else:
            future_json_path = (
                f"actual_parent_asin_after_{end_date.isoformat()}_{only_user}.json"
            )

        save_json(
            predicted_json_path,
            {
                "user_id": only_user,
                "end_date": end_date.isoformat() if end_date is not None else None,
                "parent_asins": predicted_parent_asins,
                "details": predicted_details,
            },
        )
        save_json(
            future_json_path,
            {
                "user_id": only_user,
                "end_date": end_date.isoformat() if end_date is not None else None,
                "parent_asins": future_parent_asins,
                "details": future_details,
            },
        )
        save_json(
            "predicted_parent_asin_details.json",
            {
                "user_id": only_user,
                "end_date": end_date.isoformat() if end_date is not None else None,
                "parent_asins": predicted_parent_asins,
                "details": predicted_details,
            },
        )
        save_json(
            "actual_parent_asin_after_end_date.json",
            {
                "user_id": only_user,
                "end_date": end_date.isoformat() if end_date is not None else None,
                "parent_asins": future_parent_asins,
                "details": future_details,
            },
        )
        print(f"\nSaved {predicted_json_path}")
        print(f"Saved {future_json_path}")
        print("Saved predicted_parent_asin_details.json")
        print("Saved actual_parent_asin_after_end_date.json")

    if not only_user:
        out.to_csv("predictions.csv", index=False)
        print("\nSaved predictions.csv")


if __name__ == "__main__":
    cli, start_date, end_date = _parse_window_args(sys.argv[1:])
    cmd = cli[0] if cli else "check"

    predict_user = None
    if cmd == "predict":
        predict_args = cli[1:]
        if len(predict_args) >= 3 and start_date is None and end_date is None:
            predict_user = predict_args[0]
            start_date = _parse_date_arg(predict_args[1], "START_DATE")
            end_date = _parse_date_arg(predict_args[2], "END_DATE")
            if start_date > end_date:
                raise SystemExit("START_DATE must be <= END_DATE")
        elif len(predict_args) == 2 and start_date is None and end_date is None:
            # Common accidental form: predict START_DATE END_DATE (missing user_id).
            _parse_date_arg(predict_args[0], "START_DATE")
            _parse_date_arg(predict_args[1], "END_DATE")
            raise SystemExit(
                "Missing USER_ID. Use: predict USER_ID START_DATE END_DATE"
            )
        elif len(predict_args) >= 1 and not predict_args[0].startswith("cat_"):
            predict_user = predict_args[0]

    all_data = load()
    data = _apply_date_window(all_data, start_date=start_date, end_date=end_date)
    try:
        if cmd == "check":
            cmd_check(data)
        elif cmd == "evaluate":
            cmd_evaluate(data)
        elif cmd == "predict":
            cmd_predict(data, predict_user, all_df=all_data, end_date=end_date)
        else:
            print(__doc__)
    except UnboxAIError as e:
        if explain_unknown_catalog(e):
            raise SystemExit(2)
        raise
