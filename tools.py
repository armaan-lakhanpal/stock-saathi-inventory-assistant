"""
tools.py
Deterministic inventory functions for the Inventory / Stock Query Assistant.

Gemini never calculates or "knows" stock. It only decides WHICH of these functions to call.
Every number in the chatbot's answer comes from here, read from data/store.db.

Each public function has a clear docstring, because Gemini reads these docstrings
to decide when to use the function.
"""
import os
import re
import sqlite3
from functools import lru_cache

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "store.db")

# Filler words people use in questions (English + Hinglish). Removed before matching.
STOPWORDS = {
    "the", "a", "an", "of", "for", "in", "do", "we", "have", "has", "is", "are", "any", "our",
    "stock", "qty", "quantity", "how", "much", "many", "left", "available", "units", "unit",
    "kitna", "kitne", "kitni", "bacha", "bache", "bachi", "hai", "hain", "ka", "ki", "ke",
    "packet", "packets", "pc", "pcs", "piece", "pieces", "bottle", "bottles", "please", "check",
    "patti", "wala", "wali", "wale",
}

# Hindi/Hinglish word -> (English main noun, categories it applies to or None for any).
# A synonym only matches products where that word is the MAIN noun, so "doodh" finds the two milks
# but not "Milk Bread" or "Dairy Milk" chocolate.
SYNONYMS = {
    "doodh": ("milk", {"Dairy & Bakery"}), "dudh": ("milk", {"Dairy & Bakery"}),
    "chai": ("tea", None), "namak": ("salt", None), "cheeni": ("sugar", None), "chini": ("sugar", None),
    "shakkar": ("sugar", None), "anda": ("eggs", None), "ande": ("eggs", None), "makhan": ("butter", None),
    "tel": ("oil", None), "chawal": ("rice", None), "sabun": ("soap", None), "paani": ("water", None),
    "pani": ("water", None), "dahi": ("curd", None), "biskut": ("biscuits", None),
    "machhar": ("mosquito", None), "kapde": ("detergent", None),
}

CATEGORY_ALIASES = {
    "Staples": ["staples", "grocery", "kirana", "atta dal rice", "spices", "masala", "oil"],
    "Packaged Foods": ["packaged foods", "noodles", "instant food", "sauces", "breakfast"],
    "Dairy & Bakery": ["dairy", "bakery", "milk", "bread", "dairy and bakery", "fridge"],
    "Beverages": ["beverages", "drinks", "cold drinks", "juice", "soft drinks"],
    "Biscuits & Snacks": ["biscuits", "snacks", "chips", "namkeen", "chocolates", "biscuits and snacks"],
    "Personal Care": ["personal care", "soap", "shampoo", "toothpaste", "cosmetics", "toiletries"],
    "Home Care": ["home care", "cleaning", "detergent", "household", "mosquito"],
}


# --------------------------------------------------------------------------------------
# Data loading and metric calculation (same logic as generate_data.py)
# --------------------------------------------------------------------------------------
def _normalise(text: str) -> str:
    """Lowercase, unify units (250 gram -> 250g, 1 litre -> 1l), drop punctuation."""
    t = str(text).lower().replace("-", " ").replace("&", " and ")
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(kgs?|kilo|kilos)\b", r"\1kg", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(grams?|gms?|gm|g)\b", r"\1g", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(ml|millilitre|milliliter)\b", r"\1ml", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(litres?|liters?|ltrs?|ltr|l)\b", r"\1l", t)
    t = re.sub(r"[^a-z0-9. ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@lru_cache(maxsize=1)
def _load():
    """Load tables from SQLite and compute reorder metrics once (cached)."""
    with sqlite3.connect(DB_PATH) as con:
        products = pd.read_sql("SELECT * FROM products", con)
        suppliers = pd.read_sql("SELECT * FROM suppliers", con)
        inventory = pd.read_sql("SELECT * FROM inventory", con)
        sales = pd.read_sql("SELECT * FROM sales", con, parse_dates=["date"])
        meta = dict(pd.read_sql("SELECT * FROM store_meta", con).values)

    cfg = {
        "store_name": meta["store_name"],
        "as_of": pd.Timestamp(meta["as_of_date"]),
        "z": float(meta["z_service"]),
        "cover_days": int(meta["target_cover_days"]),
        "window": int(meta["velocity_window"]),
        "dead_days": int(meta["dead_stock_days"]),
        "over_days": int(meta["overstock_days"]),
        "new_days": int(meta["new_item_days"]),
    }
    metrics = _compute_metrics(products, inventory, sales, suppliers, cfg)

    # search vocabulary per SKU: words from name, brand, pack size and aliases
    vocab = {}
    for r in products.itertuples(index=False):
        text = " ".join([r.product_name, r.brand, r.pack_size, r.aliases.replace("|", " ")])
        vocab[r.sku] = set(_normalise(text).split())
    names = dict(zip(products["sku"], products["product_name"]))
    heads = {r.sku: ({_head_word(r.product_name)} | {_head_word(a) for a in r.aliases.split("|")}, r.category)
             for r in products.itertuples(index=False)}
    return products, suppliers, sales, metrics, cfg, vocab, names, heads


def _compute_metrics(products, inventory, sales, suppliers, cfg):
    as_of, window = cfg["as_of"], cfg["window"]
    window_dates = pd.date_range(as_of - pd.Timedelta(days=window - 1), as_of)
    recent = (sales[sales["date"].isin(window_dates)]
              .pivot_table(index="date", columns="sku", values="qty_sold", aggfunc="sum")
              .reindex(index=window_dates, columns=products["sku"], fill_value=0)
              .fillna(0))

    df = (products.merge(inventory, on="sku")
                  .merge(suppliers[["supplier_id", "supplier_name", "lead_time_days", "min_order_value_inr"]],
                         on="supplier_id"))
    df["first_sale_date"] = df["sku"].map(sales.groupby("sku")["date"].min())
    df["last_sale_date"] = df["sku"].map(sales.groupby("sku")["date"].max())
    df["days_since_last_sale"] = (as_of - df["last_sale_date"]).dt.days
    days_live = (as_of - df["first_sale_date"]).dt.days + 1
    df["history_days"] = np.minimum(days_live.fillna(0), window).astype(int)

    vel, vel7, sd = [], [], []
    for sku, h in zip(df["sku"], df["history_days"]):
        s = recent[sku].iloc[-h:] if h > 0 else pd.Series([0.0])
        vel.append(s.mean())
        sd.append(s.std(ddof=1) if len(s) > 1 else 0.0)
        vel7.append(recent[sku].iloc[-7:].mean())
    df["velocity_30d"] = np.round(vel, 2)
    df["velocity_7d"] = np.round(vel7, 2)
    df["std_daily"] = np.round(sd, 2)

    df["safety_stock"] = np.round(cfg["z"] * df["std_daily"] * np.sqrt(df["lead_time_days"]), 1)
    df["reorder_point"] = np.round(df["velocity_30d"] * df["lead_time_days"] + df["safety_stock"], 1)
    df["days_of_cover"] = np.where(df["velocity_30d"] > 0,
                                   np.round(df["on_hand_qty"] / df["velocity_30d"].replace(0, np.nan), 1),
                                   np.inf)
    need = df["velocity_30d"] * cfg["cover_days"] + df["reorder_point"] - df["on_hand_qty"] - df["on_order_qty"]
    df["needs_reorder"] = (df["on_hand_qty"] + df["on_order_qty"]) <= df["reorder_point"]
    df["suggested_order_qty"] = [int(np.ceil(n / cp) * cp) if (t and n > 0) else 0
                                 for n, cp, t in zip(need, df["case_pack"], df["needs_reorder"])]

    def status(r):
        if r.on_hand_qty == 0:
            return "OUT_OF_STOCK"
        if pd.isna(r.last_sale_date) or r.days_since_last_sale >= cfg["dead_days"]:
            return "DEAD_STOCK"
        if r.history_days < cfg["new_days"]:
            return "NEW_ITEM"
        if r.needs_reorder:
            return "LOW_STOCK"
        if r.days_of_cover > cfg["over_days"]:
            return "OVERSTOCK"
        return "OK"

    df["status"] = df.apply(status, axis=1)
    df.loc[df["status"] == "DEAD_STOCK", ["suggested_order_qty", "needs_reorder"]] = [0, False]
    df["trend"] = np.where(df["velocity_7d"] > 1.5 * df["velocity_30d"].clip(lower=0.1), "UP",
                  np.where(df["velocity_7d"] < 0.5 * df["velocity_30d"], "DOWN", "STABLE"))
    df["order_value_at_cost"] = np.round(df["suggested_order_qty"] * df["cost_price"], 2)
    return df


def refresh_data():
    """Clear the cache so the next call re-reads the database (use after stock changes)."""
    _load.cache_clear()


# --------------------------------------------------------------------------------------
# Product matching
# --------------------------------------------------------------------------------------
def _head_word(text: str) -> str:
    """Last real word of a name/alias, ignoring sizes: 'Amul Taaza Toned Milk 500ml' -> 'milk'."""
    words = [w for w in _normalise(text).split() if not any(c.isdigit() for c in w) and w not in {"pcs", "pc", "pack"}]
    return words[-1] if words else ""


def _token_matches(token: str, words: set, sku: str = None) -> bool:
    if token in words:
        return True
    if token in SYNONYMS and sku is not None:
        eng, cats = SYNONYMS[token]
        heads, category = _load()[7][sku]
        return eng in heads and (cats is None or category in cats)
    if len(token) <= 3 or any(ch.isdigit() for ch in token):
        return False                                   # short words and sizes must match exactly
    return any(fuzz.ratio(token, w) >= 85 for w in words if len(w) > 3)


def _resolve(query: str):
    """Return (kind, skus): kind is 'found', 'ambiguous' or 'not_found'."""
    products, _, _, _, _, vocab, names, _ = _load()
    q = str(query).strip()
    if re.fullmatch(r"(?i)sku\d+", q):
        sku = q.upper()
        return ("found", [sku]) if sku in vocab else ("not_found", [])

    tokens = [t for t in _normalise(q).split() if t not in STOPWORDS]
    if not tokens:
        return "not_found", []

    hits = [sku for sku, words in vocab.items() if all(_token_matches(t, words, sku) for t in tokens)]
    if len(hits) == 1:
        return "found", hits
    if len(hits) > 1:
        # exact full-name match wins (e.g. user typed the complete product name)
        exact = [s for s in hits if _normalise(names[s]) == _normalise(q)]
        return ("found", exact) if len(exact) == 1 else ("ambiguous", hits)

    # nothing matched every word: offer the closest product names as suggestions
    best = process.extract(_normalise(q), {s: _normalise(n) for s, n in names.items()},
                           scorer=fuzz.WRatio, limit=3)
    return "not_found", [sku for _, score, sku in best if score >= 60]


# --------------------------------------------------------------------------------------
# Helpers to build clean, JSON-friendly answers
# --------------------------------------------------------------------------------------
def _num(x, nd=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), nd) if isinstance(x, (float, np.floating)) else int(x)


def _row(sku):
    m = _load()[3]
    return m[m["sku"] == sku].iloc[0]


def _brief(r):
    return {"sku": r.sku, "product": r.product_name, "on_hand": int(r.on_hand_qty),
            "on_order": int(r.on_order_qty), "status": r.status}


def _reorder_reason(r):
    cover = "no recent sales" if np.isinf(r.days_of_cover) else f"about {r.days_of_cover} days of cover"
    return (f"Sells about {r.velocity_30d}/day, supplier lead time {r.lead_time_days} days. "
            f"{int(r.on_hand_qty)} on hand + {int(r.on_order_qty)} on order ({cover}) is at or below the "
            f"reorder point of {r.reorder_point}. Order {int(r.suggested_order_qty)} "
            f"(rounded up to case pack of {int(r.case_pack)}).")


def _match_category(category: str):
    if not category:
        return None
    c = _normalise(category)
    best_cat, best_score = None, 0
    for cat, al in CATEGORY_ALIASES.items():
        score = max(fuzz.ratio(c, _normalise(a)) for a in [cat] + al)
        if score > best_score:
            best_cat, best_score = cat, score
    return best_cat if best_score >= 80 else None


def _category_note(category, cat):
    if category and not cat:
        return {"category_note": f"Category '{category}' not recognised, so all categories are shown. "
                                 f"Valid: {', '.join(CATEGORY_ALIASES)}"}
    return {}


# --------------------------------------------------------------------------------------
# PUBLIC TOOLS (these are given to Gemini)
# --------------------------------------------------------------------------------------
def get_stock(product: str) -> dict:
    """Look up current stock and details of ONE product.

    Use this for any question about a specific product: how many are in stock, whether it is
    available, its price (MRP), shelf location, supplier, or reorder status.

    Args:
        product: Only the product name or description as the user said it, without filler words.
            Hindi/Hinglish names and misspellings are fine, e.g. "tata namak", "maggie 70g",
            "aashirvaad atta 10kg", "dettol soap".

    Returns:
        A dict with "result" = "found", "ambiguous" or "not_found".
        found: full stock details. ambiguous: a list of matching products with their stock, so you
        can show them and ask which one the user meant. not_found: the product is not sold in this
        store (never invent a number), plus any close name suggestions.
    """
    products, _, _, _, cfg, _, _, _ = _load()
    kind, skus = _resolve(product)

    if kind == "not_found":
        return {"result": "not_found", "query": product,
                "suggestions": [_brief(_row(s)) for s in skus],
                "note": "This product is not in the store catalogue. Do not guess a stock number."}
    if kind == "ambiguous":
        return {"result": "ambiguous", "query": product, "match_count": len(skus),
                "options": [_brief(_row(s)) for s in skus[:8]],
                "note": "Several products match. Show these options and ask which one the user means."}

    r = _row(skus[0])
    p = products.set_index("sku").loc[r.sku]
    return {
        "result": "found",
        "sku": r.sku,
        "product": r.product_name,
        "category": r.category,
        "pack_size": p.pack_size,
        "on_hand": int(r.on_hand_qty),
        "on_order": int(r.on_order_qty),
        "status": r.status,
        "days_of_cover": _num(r.days_of_cover, 1),
        "avg_daily_sales_30d": _num(r.velocity_30d),
        "reorder_point": _num(r.reorder_point, 1),
        "suggested_order_qty": int(r.suggested_order_qty),
        "mrp_inr": _num(p.mrp),
        "shelf_location": p.shelf_location,
        "supplier": r.supplier_name,
        "supplier_lead_time_days": int(r.lead_time_days),
        "data_as_of": str(r.last_updated),
    }


def get_low_stock_items(category: str = "") -> dict:
    """List products that are out of stock or running low (at or below their reorder point).

    Use for questions like "what is running low?", "kya khatam ho raha hai?", "any stock alerts?",
    "what's out of stock?". Items already covered by a pending supplier order are marked.

    Args:
        category: Optional category filter, e.g. "dairy", "snacks", "home care". Leave empty for all.
    """
    m = _load()[3]
    cat = _match_category(category)
    df = m[(m["status"].isin(["OUT_OF_STOCK", "LOW_STOCK"])) |
           ((m["status"] == "NEW_ITEM") & m["needs_reorder"])]
    if cat:
        df = df[df["category"] == cat]
    df = df.sort_values(["on_hand_qty", "days_of_cover"])
    items = []
    for r in df.itertuples(index=False):
        d = _brief(r)
        d["days_of_cover"] = _num(r.days_of_cover, 1)
        d["covered_by_pending_order"] = bool(r.on_order_qty > 0 and not r.needs_reorder)
        items.append(d)
    return {**_category_note(category, cat), "category_filter": cat or "all", "count": len(items),
            "out_of_stock_count": int((df["status"] == "OUT_OF_STOCK").sum()), "items": items}


def get_reorder_suggestions(category: str = "") -> dict:
    """Recommend what to reorder today and how much, grouped by supplier, with the reasoning.

    The quantity is based on sales velocity (average daily sales over the last 30 days), supplier lead
    time and a safety stock, rounded up to the case pack. Items already covered by pending orders,
    and dead stock, are never suggested.

    Args:
        category: Optional category filter, e.g. "staples", "personal care". Leave empty for all.
    """
    m = _load()[3]
    cat = _match_category(category)
    df = m[m["suggested_order_qty"] > 0]
    if cat:
        df = df[df["category"] == cat]
    groups = []
    for (sid, sname), g in df.groupby(["supplier_id", "supplier_name"]):
        value = float(g["order_value_at_cost"].sum())
        mov = int(g["min_order_value_inr"].iloc[0])
        groups.append({
            "supplier": sname,
            "lead_time_days": int(g["lead_time_days"].iloc[0]),
            "order_value_at_cost_inr": round(value, 2),
            "meets_min_order_value": value >= mov,
            "min_order_value_inr": mov,
            "items": [{"product": r.product_name, "order_qty": int(r.suggested_order_qty),
                       "on_hand": int(r.on_hand_qty), "trend": r.trend, "reason": _reorder_reason(r)}
                      for r in g.itertuples(index=False)],
        })
    return {**_category_note(category, cat), "category_filter": cat or "all", "items_to_order": int(len(df)),
            "total_order_value_at_cost_inr": round(float(df["order_value_at_cost"].sum()), 2),
            "suppliers": groups,
            "method": "velocity x lead time + safety stock (95% service level) = reorder point; "
                      "order enough for 14 days beyond the reorder point."}


def get_sales_summary(product: str, days: int = 30) -> dict:
    """Show how fast ONE product is selling: units sold, average per day, recent trend.

    Use for "how is X selling?", "X ki sale kaisi hai?", "is X a fast mover?".

    Args:
        product: Product name as the user said it (same rules as get_stock).
        days: Look-back period in days (default 30, max 180).
    """
    _, _, sales, _, cfg, _, _, _ = _load()
    kind, skus = _resolve(product)
    if kind != "found":
        return get_stock(product)                       # reuse ambiguous / not_found reply
    r = _row(skus[0])
    days = int(max(1, min(int(days), 180)))
    start = cfg["as_of"] - pd.Timedelta(days=days - 1)
    s = sales[(sales["sku"] == r.sku) & (sales["date"] >= start)]
    out = {"result": "found", "product": r.product_name, "period_days": days,
           "units_sold": int(s["qty_sold"].sum()),
           "revenue_inr": round(float(s["qty_sold"].sum() * r.selling_price), 2),
           "avg_daily_30d": _num(r.velocity_30d), "avg_daily_7d": _num(r.velocity_7d),
           "trend": r.trend, "status": r.status, "history_days_used": int(r.history_days)}
    if r.status == "NEW_ITEM":
        out["warning"] = f"New product with only {int(r.history_days)} days of sales history, so the estimate is less reliable."
    if r.trend == "UP":
        out["note"] = "Demand in the last 7 days is well above the 30-day average."
    return out


def get_top_sellers(days: int = 30, category: str = "", limit: int = 5) -> dict:
    """List the best-selling products by units over a recent period.

    Args:
        days: Look-back period in days (default 30, max 180).
        category: Optional category filter. Leave empty for all.
        limit: How many products to return (default 5, max 15).
    """
    products, _, sales, _, cfg, _, _, _ = _load()
    days = int(max(1, min(int(days), 180)))
    limit = int(max(1, min(int(limit), 15)))
    cat = _match_category(category)
    s = sales[sales["date"] >= cfg["as_of"] - pd.Timedelta(days=days - 1)]
    s = s.merge(products[["sku", "product_name", "category", "selling_price"]], on="sku")
    if cat:
        s = s[s["category"] == cat]
    s = s.assign(revenue=s["qty_sold"] * s["selling_price"])
    top = (s.groupby("product_name")[["qty_sold", "revenue"]].sum()
             .sort_values("qty_sold", ascending=False).head(limit))
    return {**_category_note(category, cat), "period_days": days, "category_filter": cat or "all",
            "top_products": [{"product": n, "units_sold": int(q), "revenue_inr": round(float(v), 2)}
                             for n, (q, v) in top.iterrows()]}


def get_slow_moving_items() -> dict:
    """List dead stock (no sale in 60+ days) and overstocked items (more than 60 days of cover).

    Use for "what is not selling?", "which stock is stuck?", "kya nahi bik raha?". Includes the money
    tied up in this stock at cost price.
    """
    m = _load()[3]
    df = m[m["status"].isin(["DEAD_STOCK", "OVERSTOCK"])].copy()
    df["value"] = df["on_hand_qty"] * df["cost_price"]
    items = [{"product": r.product_name, "status": r.status, "on_hand": int(r.on_hand_qty),
              "days_of_cover": _num(r.days_of_cover, 1),
              "days_since_last_sale": _num(r.days_since_last_sale, 0),
              "stock_value_at_cost_inr": round(float(r.value), 2)} for r in df.itertuples(index=False)]
    return {"count": len(items), "total_value_tied_up_inr": round(float(df["value"].sum()), 2),
            "items": items,
            "note": "Dead stock should not be reordered; consider discounts or combo offers to clear it."}


def get_store_overview() -> dict:
    """Give a one-shot health summary of the store: number of products, stock value, how many items
    are out of stock, low, overstocked or dead, the categories sold, and the date the data is as of.

    Use for greetings like "how is the store doing?" or "give me a summary".
    """
    products, _, _, m, cfg, _, _, _ = _load()
    return {"store_name": cfg["store_name"], "data_as_of": str(cfg["as_of"].date()),
            "total_products": int(len(m)),
            "stock_value_at_cost_inr": round(float((m["on_hand_qty"] * m["cost_price"]).sum()), 2),
            "status_counts": {k: int(v) for k, v in m["status"].value_counts().items()},
            "categories": sorted(products["category"].unique().tolist())}


# The list of tools handed to Gemini in app.py
TOOLS = [get_stock, get_low_stock_items, get_reorder_suggestions, get_sales_summary,
         get_top_sellers, get_slow_moving_items, get_store_overview]
