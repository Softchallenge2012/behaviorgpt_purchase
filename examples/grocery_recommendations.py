"""Recommend grocery ASINs from a user's timestamped reviews.

Run with an API key and a catalog embedded with the CSV's ASINs as item IDs:

    export UNBOXAI_API_KEY=...
    python examples/grocery_recommendations.py USER_ID --catalog-id CATALOG_ID

The metadata CSV's catalog_id is a source category code, not an API catalog ID.
The API catalog must contain ASINs from that source category. The default source
category is 17 (Grocery). The default retail catalog is not this dataset.
Reviews are treated as views, not confirmed purchases. Naive CSV timestamps
are assumed to be UTC unless --timezone specifies their actual timezone.
"""

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from behaviorgpt import UnboxAIClient, View
from behaviorgpt.types import SessionEvent

DEFAULT_CSV = Path(__file__).with_name("groceries_small.csv")
DEFAULT_METADATA = Path(__file__).with_name("meta_groceries.csv")


def load_reviews(path, user_id, timezone, as_of):
    history = []
    parents = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            asin, parent = row["asin"].strip(), row["parent_asin"].strip()
            if asin and parent:
                parents.setdefault(asin, set()).add(parent)
            if row["user_id"] != user_id:
                continue
            if not asin or not row["timestamp"].strip():
                raise ValueError("a selected review is missing its ASIN or timestamp")
            timestamp = datetime.fromisoformat(row["timestamp"].strip())
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone)
            timestamp = timestamp.astimezone(UTC)
            if timestamp < as_of:
                history.append((timestamp, asin, parent))
    history.sort(key=lambda review: review[0])
    if not history:
        raise ValueError(f"no reviews before the prediction time for user {user_id!r}")
    return history, parents


def load_catalog_parents(path, wanted, source_catalog_id):
    parents = {}
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            parent = (row.get("parent_asin") or "").strip()
            if parent not in wanted:
                continue
            category = (row.get("catalog_id") or "").strip()
            if parent in parents and parents[parent] != category:
                raise ValueError(f"conflicting catalog IDs for parent ASIN {parent!r}")
            parents[parent] = category
    selected = {
        parent for parent, category in parents.items() if category == source_catalog_id
    }
    return selected, parents


def recommend(
    client,
    path,
    user_id,
    catalog_id,
    timezone,
    as_of,
    limit,
    metadata_path=DEFAULT_METADATA,
    source_catalog_id="17",
):
    reviews, parents = load_reviews(path, user_id, timezone, as_of)
    selected, known = load_catalog_parents(
        metadata_path,
        {parent for matches in parents.values() for parent in matches},
        source_catalog_id,
    )
    missing = sum(parent not in known for _, _, parent in reviews)
    if missing:
        print(f"Skipped {missing} reviews without metadata.", file=sys.stderr)
    reviews = [review for review in reviews if review[2] in selected]
    if not reviews:
        raise ValueError(f"no reviews match source catalog ID {source_catalog_id!r}")
    history = [
        SessionEvent(
            store_id=catalog_id,
            domains=client.domains,
            timezone=timezone.key,
            timestamp=timestamp,
            event=View(asin),
            session_id=f"review-{index}",
        )
        for index, (timestamp, asin, _) in enumerate(reviews)
    ]
    request = client.requester.make(
        catalog_id, history=history, timestamp=as_of, limit=limit
    )
    response = client.recommendations.create(
        request, headers={"x-catalog-id": catalog_id}
    )
    results = []
    for item in response.items:
        matches = parents.get(item.id, set())
        if len(matches) != 1:
            raise ValueError(
                f"cannot identify a unique parent ASIN for {item.id!r} in the CSV"
            )
        if not matches <= selected:
            raise ValueError(
                f"item {item.id!r} is not in source catalog ID {source_catalog_id!r}"
            )
        results.append(
            {"asin": item.id, "parent_asin": next(iter(matches)), "score": item.score}
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("user_id")
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--source-catalog-id", default="17")
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--as-of",
        help="prediction time as an ISO 8601 timestamp with timezone; default: now",
    )
    args = parser.parse_args()
    if args.catalog_id.isdigit():
        parser.error(
            "--catalog-id expects an Unbox catalog id (for example cat_...). "
            "You passed a numeric source category code. "
            "Use --source-catalog-id for values like 17."
        )
    timezone = ZoneInfo(args.timezone)
    as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(UTC)
    if as_of.tzinfo is None:
        parser.error("--as-of must include a timezone, e.g. 2020-05-01T00:00:00Z")
    as_of = as_of.astimezone(UTC)
    client = UnboxAIClient(market="US", timezone=args.timezone)
    results = recommend(
        client,
        args.csv,
        args.user_id,
        args.catalog_id,
        timezone,
        as_of,
        args.limit,
        args.metadata,
        args.source_catalog_id,
    )
    print(json.dumps({"user_id": args.user_id, "recommendations": results}, indent=2))


if __name__ == "__main__":
    main()