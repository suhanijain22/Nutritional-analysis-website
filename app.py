from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
import re
import os

app = Flask(__name__)

# ──────────────────────────────────────────────
# Load & clean dataset once at startup
# ──────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "nutrition.csv")

def parse_num(val):
    """Extract first float from strings like '91.27 g', '9.00 mg', etc."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    s = str(val).strip()
    if s in ("", "nan", "0"):
        return 0.0
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else 0.0

def load_data():
    df = pd.read_csv(CSV_PATH)
    df["name"] = df["name"].fillna("Unknown")

    records = []
    for _, row in df.iterrows():
        fat_total = parse_num(row.get("fat"))
        sat_fat   = parse_num(row.get("saturated_fat")) or parse_num(row.get("saturated_fatty_acids"))

        rec = {
            "name":        str(row["name"]),
            "serving":     str(row.get("serving_size", "100 g")),
            "calories":    int(parse_num(row.get("calories", 0))),
            "protein":     parse_num(row.get("protein")),
            "fat":         fat_total,
            "carbs":       parse_num(row.get("carbohydrate")),
            "fiber":       parse_num(row.get("fiber")),
            "sugars":      parse_num(row.get("sugars")),
            "sodium":      parse_num(row.get("sodium")),
            "calcium":     parse_num(row.get("calcium")),
            "potassium":   parse_num(row.get("potassium")),
            "iron":        parse_num(row.get("irom")),
            "magnesium":   parse_num(row.get("magnesium")),
            "cholesterol": parse_num(row.get("cholesterol")),
            "sat_fat":     sat_fat,
            "mono_fat":    parse_num(row.get("monounsaturated_fatty_acids")),
            "poly_fat":    parse_num(row.get("polyunsaturated_fatty_acids")),
            "vit_a":       parse_num(row.get("vitamin_a")),
            "vit_c":       parse_num(row.get("vitamin_c")),
            "vit_d":       parse_num(row.get("vitamin_d")),
            "vit_e":       parse_num(row.get("vitamin_e")),
            "vit_b12":     parse_num(row.get("vitamin_b12")),
            "vit_b6":      parse_num(row.get("vitamin_b6")),
            "vit_k":       parse_num(row.get("vitamin_k")),
            "water":       parse_num(row.get("water")),
            "caffeine":    parse_num(row.get("caffeine")),
            "alcohol":     parse_num(row.get("alcohol")),
            "choline":     parse_num(row.get("choline")),
            "folate":      parse_num(row.get("folate")),
            "niacin":      parse_num(row.get("niacin")),
            "thiamin":     parse_num(row.get("thiamin")),
            "riboflavin":  parse_num(row.get("riboflavin")),
            "copper":      parse_num(row.get("copper")),
            "manganese":   parse_num(row.get("manganese")),
            "phosphorous": parse_num(row.get("phosphorous")),
            "selenium":    parse_num(row.get("selenium")),
            "zinc":        parse_num(row.get("zink")),
        }
        records.append(rec)
    return records

print("Loading nutrition dataset...")
RECORDS = load_data()
NAMES   = [r["name"] for r in RECORDS]
print(f"Loaded {len(RECORDS)} food items.")


# ──────────────────────────────────────────────
# Helper: compute analysis fields
# ──────────────────────────────────────────────
DAILY_VALUES = {
    "protein":     50,   "fiber":    28,    "fat":       78,
    "carbs":       300,  "sugars":   50,    "sodium":    2300,
    "calcium":     1300, "potassium":4700,  "iron":      18,
    "magnesium":   420,  "cholesterol": 300,
    "vit_a":       5000, "vit_c":    90,    "vit_d":     800,
    "vit_e":       15,   "vit_b12":  2.4,   "vit_b6":    1.7,
    "vit_k":       120,  "choline":  550,   "folate":    400,
    "niacin":      16,   "thiamin":  1.2,   "riboflavin":1.3,
    "phosphorous": 1250, "selenium": 55,    "zinc":      11,
    "copper":      0.9,  "manganese":2.3,
}

def letter_score(val, dv):
    if dv == 0:
        return "N/A"
    pct = val / dv
    if pct >= 0.4:  return "A"
    if pct >= 0.2:  return "B"
    if pct >= 0.1:  return "C"
    if pct >= 0.05: return "D"
    return "F"

def pct_dv(val, dv):
    if dv == 0: return 0
    return round(min(val / dv * 100, 999), 1)

def generate_badges(r):
    badges = []
    if r["protein"] >= 20:        badges.append({"label": "High Protein",      "type": "good"})
    if r["fiber"]   >= 5:         badges.append({"label": "High Fiber",         "type": "good"})
    if r["vit_c"]   >= 45:        badges.append({"label": "Rich in Vitamin C",  "type": "good"})
    if r["calcium"] >= 300:       badges.append({"label": "Good Calcium Source","type": "good"})
    if r["iron"]    >= 4:         badges.append({"label": "Iron Rich",          "type": "good"})
    if r["fat"]     <= 3 and r["calories"] > 0: badges.append({"label": "Low Fat",  "type": "info"})
    if r["calories"] <= 50:       badges.append({"label": "Low Calorie",        "type": "info"})
    if r["carbs"]   < 10 and r["calories"] > 0: badges.append({"label": "Low Carb", "type": "info"})
    if r["water"]   >= 70:        badges.append({"label": "High Water Content", "type": "info"})
    if r["sodium"]  > 600:        badges.append({"label": "High Sodium ⚠️",     "type": "warn"})
    if r["sat_fat"] > 10:         badges.append({"label": "High Sat. Fat ⚠️",   "type": "warn"})
    if r["sugars"]  > 20:         badges.append({"label": "High Sugar ⚠️",      "type": "warn"})
    if r["cholesterol"] > 150:    badges.append({"label": "High Cholesterol ⚠️","type": "warn"})
    if r["caffeine"] > 50:        badges.append({"label": "Contains Caffeine",  "type": "neutral"})
    if r["alcohol"]  > 0:         badges.append({"label": "Contains Alcohol",   "type": "neutral"})
    return badges

def macro_calories(r):
    """Calorie contribution from each macro."""
    return {
        "protein": round(r["protein"] * 4, 1),
        "fat":     round(r["fat"]     * 9, 1),
        "carbs":   round(r["carbs"]   * 4, 1),
    }

def analyze(r):
    mc = macro_calories(r)
    total_macro_cal = sum(mc.values()) or 1
    return {
        **r,
        "badges":  generate_badges(r),
        "scores": {
            "protein":  letter_score(r["protein"],  DAILY_VALUES["protein"]),
            "fiber":    letter_score(r["fiber"],    DAILY_VALUES["fiber"]),
            "vitamins": letter_score(
                (r["vit_c"]/DAILY_VALUES["vit_c"] + r["vit_a"]/DAILY_VALUES["vit_a"] +
                 r["vit_d"]/DAILY_VALUES["vit_d"] + r["vit_e"]/DAILY_VALUES["vit_e"] +
                 r["vit_b12"]/DAILY_VALUES["vit_b12"]) / 5 * DAILY_VALUES["protein"],
                DAILY_VALUES["protein"]),
            "minerals": letter_score(
                (r["calcium"]/DAILY_VALUES["calcium"] + r["iron"]/DAILY_VALUES["iron"] +
                 r["potassium"]/DAILY_VALUES["potassium"] + r["magnesium"]/DAILY_VALUES["magnesium"]) / 4 * DAILY_VALUES["protein"],
                DAILY_VALUES["protein"]),
        },
        "pct_dv": {k: pct_dv(r.get(k, 0), v) for k, v in DAILY_VALUES.items()},
        "macro_cal": mc,
        "macro_cal_pct": {k: round(v / total_macro_cal * 100, 1) for k, v in mc.items()},
    }


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/foods")
def api_foods():
    """Return all food names with their index."""
    return jsonify([{"id": i, "name": n} for i, n in enumerate(NAMES)])


@app.route("/api/search")
def api_search():
    """Search food items by name."""
    q = request.args.get("q", "").strip().lower()
    limit = int(request.args.get("limit", 20))
    if not q:
        return jsonify([])
    results = [
        {"id": i, "name": NAMES[i]}
        for i in range(len(NAMES))
        if q in NAMES[i].lower()
    ]
    return jsonify(results[:limit])


@app.route("/api/food/<int:food_id>")
def api_food(food_id):
    """Return full nutrition analysis for a food item."""
    if food_id < 0 or food_id >= len(RECORDS):
        return jsonify({"error": "Food not found"}), 404
    return jsonify(analyze(RECORDS[food_id]))


@app.route("/api/compare")
def api_compare():
    """Compare two food items side by side."""
    id1 = request.args.get("id1", type=int)
    id2 = request.args.get("id2", type=int)
    if id1 is None or id2 is None:
        return jsonify({"error": "Provide id1 and id2"}), 400
    if not (0 <= id1 < len(RECORDS)) or not (0 <= id2 < len(RECORDS)):
        return jsonify({"error": "Invalid food id"}), 404
    return jsonify({
        "food1": analyze(RECORDS[id1]),
        "food2": analyze(RECORDS[id2]),
    })


@app.route("/api/stats")
def api_stats():
    """Dataset-level stats."""
    return jsonify({
        "total_items": len(RECORDS),
        "avg_calories": round(sum(r["calories"] for r in RECORDS) / len(RECORDS), 1),
        "max_protein_food": max(RECORDS, key=lambda r: r["protein"])["name"],
        "max_fiber_food":   max(RECORDS, key=lambda r: r["fiber"])["name"],
        "max_vit_c_food":   max(RECORDS, key=lambda r: r["vit_c"])["name"],
    })


if __name__ == "__main__":

    app.run(debug=True, port=5001)
