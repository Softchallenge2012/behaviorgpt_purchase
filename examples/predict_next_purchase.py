#!/opt/anaconda3/envs/crewai/bin/python
"""Predict each user's next purchase from Amazon Grocery sample JSONL files.

This script:
1) Reads review events from examples/Grocery_and_Gourmet_Food.jsonl
2) Reads product metadata from examples/meta_Grocery_and_Gourmet_Food.jsonl
3) Joins both files on parent_asin
4) Builds per-user purchase sequences in time order
5) Learns a lightweight next-item model from temporal transitions
6) Evaluates top-k accuracy and prints sample predictions

Features used from the prompt:
- user_id
- date from timestamp
- title
- text
- rating
- price
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

WORD_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "great",
    "i",
    "in",
    "is",
    "it",
    "its",
    "my",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
    "you",
    "yummy",
}


@dataclass(slots=True)
class Event:
    user_id: str
    parent_asin: str
    timestamp_ms: int
    title: str
    text: str
    rating: float
    price: float | None


def parse_price(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    cleaned = value.strip().replace(",", "")
    if not cleaned:
        return None

    # Keep only the first numeric token (for values like "$12.99 - $15.99").
    match = re.search(r"[0-9]+(?:\.[0-9]+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def tokenize(text: str, max_tokens: int = 8) -> list[str]:
    tokens = [t for t in WORD_RE.findall(text.lower()) if t not in STOP_WORDS and len(t) >= 3]
    return tokens[:max_tokens]


def iter_jsonl(path: Path, max_rows: int | None = None) -> Iterable[dict]:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_rows is not None and count >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                count += 1
            except json.JSONDecodeError:
                continue


def load_events(reviews_path: Path, max_reviews: int | None = None) -> tuple[list[Event], set[str]]:
    events: list[Event] = []
    needed_asins: set[str] = set()

    for row in iter_jsonl(reviews_path, max_rows=max_reviews):
        parent_asin = row.get("parent_asin")
        user_id = row.get("user_id")
        timestamp = row.get("timestamp")

        if not parent_asin or not user_id or timestamp is None:
            continue

        try:
            timestamp_ms = int(timestamp)
            rating = float(row.get("rating", 0.0))
        except (TypeError, ValueError):
            continue

        ev = Event(
            user_id=str(user_id),
            parent_asin=str(parent_asin),
            timestamp_ms=timestamp_ms,
            title=str(row.get("title", "")),
            text=str(row.get("text", "")),
            rating=rating,
            price=None,
        )
        events.append(ev)
        needed_asins.add(ev.parent_asin)

    return events, needed_asins


def load_prices(meta_path: Path, needed_asins: set[str], max_meta_rows: int | None = None) -> dict[str, float | None]:
    prices: dict[str, float | None] = {}

    for row in iter_jsonl(meta_path, max_rows=max_meta_rows):
        parent_asin = row.get("parent_asin")
        if parent_asin not in needed_asins:
            continue
        prices[str(parent_asin)] = parse_price(row.get("price"))

    return prices


def attach_prices(events: list[Event], prices: dict[str, float | None]) -> None:
    for ev in events:
        ev.price = prices.get(ev.parent_asin)


def price_bucket(price: float | None) -> str:
    if price is None or not math.isfinite(price):
        return "price_unknown"
    if price < 5:
        return "price_<5"
    if price < 10:
        return "price_5_10"
    if price < 20:
        return "price_10_20"
    if price < 40:
        return "price_20_40"
    return "price_40_plus"


def rating_bucket(rating: float) -> str:
    if rating < 2.5:
        return "rating_low"
    if rating < 4.0:
        return "rating_mid"
    return "rating_high"


def event_context_features(ev: Event) -> tuple[list[str], datetime]:
    dt = datetime.fromtimestamp(ev.timestamp_ms / 1000, tz=timezone.utc)
    tokens = tokenize(f"{ev.title} {ev.text}")
    features = [
        f"dow:{dt.weekday()}",
        f"month:{dt.month}",
        f"hour:{dt.hour}",
        rating_bucket(ev.rating),
        price_bucket(ev.price),
    ]
    features.extend(f"kw:{tok}" for tok in tokens)
    return features, dt


def build_user_sequences(events: list[Event]) -> dict[str, list[Event]]:
    by_user: dict[str, list[Event]] = defaultdict(list)
    for ev in events:
        by_user[ev.user_id].append(ev)

    for seq in by_user.values():
        seq.sort(key=lambda x: x.timestamp_ms)

    return by_user


def build_train_test_pairs(
    by_user: dict[str, list[Event]],
) -> tuple[list[tuple[Event, str]], list[tuple[Event, str]], dict[str, Event]]:
    train_pairs: list[tuple[Event, str]] = []
    test_pairs: list[tuple[Event, str]] = []
    latest_event_by_user: dict[str, Event] = {}

    for user_id, seq in by_user.items():
        if len(seq) < 2:
            latest_event_by_user[user_id] = seq[-1]
            continue

        transitions = [(seq[i], seq[i + 1].parent_asin) for i in range(len(seq) - 1)]
        *train_part, test_part = transitions

        train_pairs.extend(train_part)
        test_pairs.append(test_part)
        latest_event_by_user[user_id] = seq[-1]

    return train_pairs, test_pairs, latest_event_by_user


class NextPurchaseModel:
    def __init__(self) -> None:
        self.global_next: Counter[str] = Counter()
        self.user_next: dict[str, Counter[str]] = defaultdict(Counter)
        self.feature_next: dict[str, Counter[str]] = defaultdict(Counter)

        self.weights = {
            "user": 4.0,
            "feature": 1.0,
            "global": 0.2,
        }

    def fit(self, train_pairs: list[tuple[Event, str]]) -> None:
        for ev, next_asin in train_pairs:
            self.global_next[next_asin] += 1
            self.user_next[ev.user_id][next_asin] += 1

            features, _ = event_context_features(ev)
            for feat in features:
                self.feature_next[feat][next_asin] += 1

    def predict_topk(self, ev: Event, k: int = 5) -> list[tuple[str, float]]:
        scores: defaultdict[str, float] = defaultdict(float)

        for asin, count in self.global_next.items():
            scores[asin] += self.weights["global"] * count

        for asin, count in self.user_next.get(ev.user_id, {}).items():
            scores[asin] += self.weights["user"] * count

        features, _ = event_context_features(ev)
        for feat in features:
            for asin, count in self.feature_next.get(feat, {}).items():
                scores[asin] += self.weights["feature"] * count

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]


def evaluate(model: NextPurchaseModel, test_pairs: list[tuple[Event, str]], topk: int = 5) -> tuple[float, float]:
    if not test_pairs:
        return 0.0, 0.0

    top1_hits = 0
    topk_hits = 0

    for ev, y_true in test_pairs:
        preds = model.predict_topk(ev, k=topk)
        pred_items = [asin for asin, _ in preds]

        if pred_items:
            if pred_items[0] == y_true:
                top1_hits += 1
            if y_true in pred_items:
                topk_hits += 1

    n = len(test_pairs)
    return top1_hits / n, topk_hits / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next purchase from Grocery sample JSONL files")
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("examples/Grocery_and_Gourmet_Food.jsonl"),
        help="Path to review JSONL",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("examples/meta_Grocery_and_Gourmet_Food.jsonl"),
        help="Path to metadata JSONL",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=300000,
        help="Max review rows to read (for speed/memory)",
    )
    parser.add_argument(
        "--max-meta-rows",
        type=int,
        default=None,
        help="Optional cap for metadata rows",
    )
    parser.add_argument(
        "--show-users",
        type=int,
        default=10,
        help="How many users to print example predictions for",
    )
    args = parser.parse_args()

    print("Loading review events...")
    events, needed_asins = load_events(args.reviews, max_reviews=args.max_reviews)
    print(f"Loaded {len(events):,} events from {args.reviews}")
    print(f"Unique parent_asin in sample: {len(needed_asins):,}")

    print("Loading metadata prices and joining by parent_asin...")
    prices = load_prices(args.meta, needed_asins=needed_asins, max_meta_rows=args.max_meta_rows)
    attach_prices(events, prices)
    print(f"Matched prices for {sum(1 for e in events if e.price is not None):,} events")

    by_user = build_user_sequences(events)
    train_pairs, test_pairs, latest_event_by_user = build_train_test_pairs(by_user)

    print(f"Users in sample: {len(by_user):,}")
    print(f"Train transitions: {len(train_pairs):,}")
    print(f"Test transitions: {len(test_pairs):,}")

    model = NextPurchaseModel()
    model.fit(train_pairs)

    top1, top5 = evaluate(model, test_pairs, topk=5)
    print(f"Top-1 accuracy: {top1:.4f}")
    print(f"Top-5 accuracy: {top5:.4f}")

    print("\nSample next-purchase predictions:")
    shown = 0
    for user_id, last_event in latest_event_by_user.items():
        preds = model.predict_topk(last_event, k=5)
        if not preds:
            continue
        pred_text = ", ".join(f"{asin}:{score:.1f}" for asin, score in preds)
        print(f"user_id={user_id} -> {pred_text}")
        shown += 1
        if shown >= args.show_users:
            break


if __name__ == "__main__":
    main()
