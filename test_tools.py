"""
test_tools.py
Tests tools.py WITHOUT any AI. Run:  python test_tools.py
Checks product matching (Hinglish, misspellings, ambiguity, unknown products)
and that every reorder number matches data/reorder_metrics.csv from generate_data.py.
"""
import json
import os
import pandas as pd
import tools

passed = failed = 0

def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}  {detail}")

print("\n1. Product matching")
cases = [
    # (query, expected result, expected product or None)
    ("aashirvaad atta 5kg", "found", "Aashirvaad Whole Wheat Atta 5kg"),
    ("atta 10 kilo", "found", "Aashirvaad Whole Wheat Atta 10kg"),
    ("tata namak", "found", "Tata Salt 1kg"),
    ("parleji 250 gram", "found", "Parle-G Biscuits 250g"),
    ("Parle-G 250g", "found", "Parle-G Biscuits 250g"),
    ("maggie masala 70g", "found", "Maggi 2-Minute Masala Noodles 70g"),
    ("korean maggi", "found", "Maggi Korean Spicy Noodles 75g"),
    ("amul butter", "found", "Amul Butter 100g"),
    ("dove soap", "found", "Dove Cream Beauty Bar 100g"),
    ("doodh", "ambiguous", None),           # toned + full cream milk only
    ("amul doodh", "ambiguous", None),
    ("chai", "ambiguous", None),
    ("anda", "found", "Farm Fresh Eggs 6 pcs"),
    ("makhan", "found", "Amul Butter 100g"),           # toned milk and full cream milk
    ("surf", "ambiguous", None),
    ("dettol", "ambiguous", None),
    ("maggi", "ambiguous", None),
    ("chai patti", "ambiguous", None),      # Tata Tea and Red Label
    ("tel", "ambiguous", None),             # all oils
    ("sarson ka tel", "found", "Fortune Kachi Ghani Mustard Oil 1L"),
    ("cheeni", "found", "Madhur Sugar 1kg"),
    ("pepsi", "not_found", None),
    ("SKU1001", "found", "Aashirvaad Whole Wheat Atta 5kg"),
]
for q, exp_kind, exp_name in cases:
    r = tools.get_stock(q)
    ok = r["result"] == exp_kind and (exp_name is None or r.get("product") == exp_name)
    shown = r.get("product") or [o["product"] for o in r.get("options", r.get("suggestions", []))]
    check(f"{q!r:24} -> {r['result']:9} {shown}", ok, f"(expected {exp_kind} {exp_name or ''})")

print("\n2. Numbers match generate_data.py (reorder_metrics.csv)")
ref = pd.read_csv(os.path.join(os.path.dirname(tools.DB_PATH), "reorder_metrics.csv"))
m = tools._load()[3]
cmp = ref.merge(m, on="sku", suffixes=("_ref", "_tool"))
for col in ["on_hand_qty", "velocity_30d", "reorder_point", "suggested_order_qty", "status"]:
    same = (cmp[f"{col}_ref"] == cmp[f"{col}_tool"]).all()
    check(f"{col:22} identical for all {len(cmp)} SKUs", same)

print("\n3. Planted scenarios")
check("Amul Butter out of stock", tools.get_stock("amul butter")["status"] == "OUT_OF_STOCK")
check("Fortune oil NOT reordered (12 on order)", tools.get_stock("fortune sunflower oil")["suggested_order_qty"] == 0)
check("Ferrero is dead stock", tools.get_stock("ferrero")["status"] == "DEAD_STOCK")
check("Korean Maggi has new-item warning", "warning" in tools.get_sales_summary("korean maggi"))
check("Red Bull trend is UP", tools.get_sales_summary("red bull")["trend"] == "UP")

print("\n4. Sample outputs (what Gemini will receive)")
print("\nget_stock('atta 5kg'):\n", json.dumps(tools.get_stock("atta 5kg"), indent=2))
low = tools.get_low_stock_items()
print(f"\nget_low_stock_items(): {low['count']} items ->", [i["product"] for i in low["items"]])
reo = tools.get_reorder_suggestions()
print(f"\nget_reorder_suggestions(): {reo['items_to_order']} items, "
      f"total Rs {reo['total_order_value_at_cost_inr']:,} across {len(reo['suppliers'])} suppliers")
print("   first reason:", reo["suppliers"][0]["items"][0]["reason"])
print("\nget_low_stock_items('dairy'):", [i["product"] for i in tools.get_low_stock_items("dairy")["items"]])
print("get_reorder_suggestions('personal care'):", [i["product"] for g in tools.get_reorder_suggestions("personal care")["suppliers"] for i in g["items"]])
print("get_low_stock_items('xyz'):", tools.get_low_stock_items("xyz").get("category_note"))
print("get_top_sellers():", [t["product"] for t in tools.get_top_sellers()["top_products"]])
print("get_slow_moving_items():", [i["product"] for i in tools.get_slow_moving_items()["items"]])
print("get_store_overview():", tools.get_store_overview()["status_counts"])

print(f"\nRESULT: {passed} passed, {failed} failed")
