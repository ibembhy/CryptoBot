from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from kalshi_btc_bot.types import MarketSnapshot
from kalshi_btc_bot.utils.math import clamp
from kalshi_btc_bot.utils.time import ensure_utc


def _scaled_probability(value: float | int | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if number > 1.0:
        number = number / 100.0
    return clamp(number, 0.0, 1.0)


def _coalesce(mapping: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _settlement_price(mapping: Mapping[str, object]) -> float | None:
    direct = _coalesce(
        mapping,
        "settlement_price",
        "final_price",
        "settlement_value_dollars",
        "yes_settlement_value_dollars",
    )
    if direct is not None:
        return float(direct)
    result = str(_coalesce(mapping, "result", "market_result") or "").lower()
    if result == "yes":
        return 1.0
    if result == "no":
        return 0.0
    return None


def _infer_direction(
    raw_market: Mapping[str, object],
    *,
    contract_type: str,
    fallback_spot_price: float | None = None,
    threshold: float | None = None,
) -> str | None:
    direction = _coalesce(raw_market, "direction", "settlement_rule")
    if isinstance(direction, str):
        normalized = direction.lower()
        if "below" in normalized or "down" in normalized:
            return "below" if contract_type == "threshold" else "down"
        if "up" in normalized or "above" in normalized:
            return "above" if contract_type == "threshold" else "up"

    text_fields = [
        _coalesce(raw_market, "yes_sub_title", "yes_subtitle"),
        _coalesce(raw_market, "subtitle", "title", "question"),
    ]
    text = " ".join(str(value).lower() for value in text_fields if value)
    if contract_type == "threshold":
        if "or above" in text or "above" in text or "higher" in text:
            return "above"
        if "or below" in text or "below" in text or "lower" in text:
            return "below"
        if threshold is not None and fallback_spot_price is not None:
            return "above" if threshold >= fallback_spot_price else "below"

    if contract_type == "direction":
        if "down" in text or "below" in text:
            return "down"
        if text:
            return "up"
    return None


def normalize_market(
    raw_market: Mapping[str, object],
    *,
    spot_price: float,
    observed_at: datetime,
    underlying_symbol: str = "BTC-USD",
    source: str = "kalshi",
    series_ticker_override: str | None = None,
) -> MarketSnapshot:
    contract_type = str(_coalesce(raw_market, "contract_type", "type", "market_type", "subcategory") or "threshold").lower()
    if contract_type not in {"threshold", "range", "direction"}:
        if "range" in contract_type:
            contract_type = "range"
        elif "direction" in contract_type or "updown" in contract_type:
            contract_type = "direction"
        else:
            contract_type = "threshold"

    yes_bid = _scaled_probability(_coalesce(raw_market, "yes_bid", "yes_bid_price", "yes_bid_dollars"))
    yes_ask = _scaled_probability(_coalesce(raw_market, "yes_ask", "yes_ask_price", "yes_ask_dollars"))
    no_bid = _scaled_probability(_coalesce(raw_market, "no_bid", "no_bid_price", "no_bid_dollars"))
    no_ask = _scaled_probability(_coalesce(raw_market, "no_ask", "no_ask_price", "no_ask_dollars"))

    mid_components = [value for value in (yes_bid, yes_ask) if value is not None]
    mid_price = sum(mid_components) / len(mid_components) if mid_components else None
    implied_probability = yes_ask if yes_ask is not None else mid_price

    # Kalshi's expiration_time can reflect a later archival/settlement deadline.
    # For trading and replay we care about when the market stops trading, which is close_time.
    expiry_raw = _coalesce(
        raw_market,
        "close_time",
        "expected_expiration_time",
        "expiry",
        "settlement_time",
        "expiration_time",
    )
    if expiry_raw is None:
        raise ValueError("Market missing expiry")
    expiry = ensure_utc(expiry_raw).to_pydatetime()

    threshold = _coalesce(raw_market, "threshold", "strike", "floor_strike")
    range_low = _coalesce(raw_market, "range_low", "floor", "lower_bound")
    range_high = _coalesce(raw_market, "range_high", "cap", "upper_bound")
    direction = _infer_direction(
        raw_market,
        contract_type=contract_type,
        fallback_spot_price=spot_price,
        threshold=None if threshold is None else float(threshold),
    )

    metadata = dict(raw_market)
    return MarketSnapshot(
        source=source,
        series_ticker=str(series_ticker_override or _coalesce(raw_market, "series_ticker", "seriesTicker", "series") or "CRYPTO"),
        market_ticker=str(_coalesce(raw_market, "ticker", "market_ticker", "event_ticker") or ""),
        contract_type=contract_type,  # type: ignore[arg-type]
        underlying_symbol=underlying_symbol,
        observed_at=ensure_utc(observed_at).to_pydatetime(),
        expiry=expiry,
        spot_price=float(spot_price),
        threshold=None if threshold is None else float(threshold),
        range_low=None if range_low is None else float(range_low),
        range_high=None if range_high is None else float(range_high),
        direction=direction if isinstance(direction, str) else None,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        mid_price=mid_price,
        implied_probability=implied_probability,
        volume=None if _coalesce(raw_market, "volume") is None else float(_coalesce(raw_market, "volume")),
        open_interest=None if _coalesce(raw_market, "open_interest", "openInterest") is None else float(_coalesce(raw_market, "open_interest", "openInterest")),
        settlement_price=_settlement_price(raw_market),
        metadata=metadata,
    )
