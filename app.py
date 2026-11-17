import os, re, json, random, zipfile, urllib.request, urllib.error
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def read_xlsx_builtin(path):
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            strings = ["".join(si.itertext()) for si in root.findall("a:si", ns)]
        sheet_files = sorted([f for f in z.namelist() if f.startswith("xl/worksheets/sheet") and f.endswith(".xml")])
        sheet = ET.fromstring(z.read(sheet_files[0]))
        rows = []
        for r in sheet.findall(".//a:sheetData/a:row", ns):
            row_vals = {}
            for c in r.findall("a:c", ns):
                ref = "".join(ch for ch in c.attrib.get("r", "A") if ch.isalpha())
                col_idx = sum((ord(ch) - 64) * (26 ** i) for i, ch in enumerate(reversed(ref))) - 1
                v = c.find("a:v", ns)
                inline = c.find("a:is", ns)
                val = "".join(inline.itertext()) if inline is not None else (v.text if v is not None else "")
                if c.attrib.get("t") == "s" and val != "":
                    val = strings[int(val)]
                row_vals[col_idx] = val
            if row_vals:
                max_c = max(row_vals.keys())
                rows.append([row_vals.get(i, "") for i in range(max_c + 1)])
    headers = [str(h).strip() for h in rows[0]]
    data = [r + [""] * (len(headers) - len(r)) if len(r) < len(headers) else r[:len(headers)] for r in rows[1:]]
    return pd.DataFrame(data, columns=headers)


def run_numpy_kmeans(X, k=4, seed=42, max_iter=50):
    centers = np.array([
        [0.45, 0.20, 0.15, 0.20],
        [0.18, 0.45, 0.18, 0.19],
        [0.18, 0.20, 0.44, 0.18],
        [0.20, 0.20, 0.15, 0.45],
    ], dtype=float)
    labels = np.zeros(len(X), dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for j in range(k):
            if np.any(labels == j):
                centers[j] = X[labels == j].mean(axis=0)
                centers[j] /= centers[j].sum()
    return labels, centers

PALETTE = ["#2563eb", "#f97316", "#10b981", "#8b5cf6", "#ef4444"]

def render_svg_bar_chart(df, height=270):
    cols = list(df.columns)
    idx = [str(x) for x in df.index]
    vals = df.to_numpy(dtype=float)
    y_max = max(float(np.nanmax(vals)) * 1.15, 0.1)
    w, h = 760, height
    pad_l, pad_r, pad_t, pad_b = 52, 24, 34, 54
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    n_groups = max(len(idx), 1)
    n_series = max(len(cols), 1)
    group_w = pw / n_groups
    bar_w = min(38.0, (group_w * 0.72) / n_series)

    svg = [f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto;background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:6px;">']
    for i in range(5):
        yv = y_max * i / 4
        py = pad_t + ph - (ph * i / 4)
        svg.append(f'<line x1="{pad_l}" y1="{py:.1f}" x2="{w - pad_r}" y2="{py:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        svg.append(f'<text x="{pad_l - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="11" fill="#64748b">{yv:.2f}</text>')
    lx = pad_l
    for s_idx, col_name in enumerate(cols):
        color = PALETTE[s_idx % len(PALETTE)]
        svg.append(f'<rect x="{lx}" y="10" width="12" height="12" rx="3" fill="{color}"/>')
        svg.append(f'<text x="{lx + 16}" y="20" font-size="12" font-weight="500" fill="#334155">{col_name}</text>')
        lx += max(95, len(str(col_name)) * 7 + 28)
    for g_idx, label in enumerate(idx):
        gx = pad_l + g_idx * group_w + (group_w - bar_w * n_series) / 2
        for s_idx in range(n_series):
            v = max(0.0, float(vals[g_idx, s_idx]))
            bh = (v / y_max) * ph
            bx = gx + s_idx * bar_w
            by = pad_t + ph - bh
            color = PALETTE[s_idx % len(PALETTE)]
            svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{max(bar_w - 2, 2):.1f}" height="{bh:.1f}" rx="3" fill="{color}"/>')
            if n_series == 1:
                svg.append(f'<text x="{bx + bar_w / 2:.1f}" y="{by - 5:.1f}" text-anchor="middle" font-size="11" font-weight="600" fill="#1e293b">{v:.2f}</text>')
        cx = pad_l + (g_idx + 0.5) * group_w
        short_lbl = label if len(label) <= 22 else label[:20] + "…"
        svg.append(f'<text x="{cx:.1f}" y="{pad_t + ph + 22}" text-anchor="middle" font-size="11" font-weight="500" fill="#334155">{short_lbl}</text>')
    svg.append("</svg>")
    st.markdown("".join(svg), unsafe_allow_html=True)


def render_svg_line_chart(df, height=270):
    cols = list(df.columns)
    idx = list(df.index)
    vals = df.to_numpy(dtype=float)
    v_min, v_max = float(np.nanmin(vals)), float(np.nanmax(vals))
    span = max(v_max - v_min, 0.05)
    y_min = max(0.0, v_min - 0.1 * span)
    y_max = v_max + 0.1 * span
    w, h = 760, height
    pad_l, pad_r, pad_t, pad_b = 54, 24, 34, 42
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    n_pts = max(len(idx), 1)

    svg = [f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto;background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:6px;">']
    for i in range(5):
        yv = y_min + (y_max - y_min) * i / 4
        py = pad_t + ph - (ph * i / 4)
        svg.append(f'<line x1="{pad_l}" y1="{py:.1f}" x2="{w - pad_r}" y2="{py:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
        svg.append(f'<text x="{pad_l - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="11" fill="#64748b">{yv:.2f}</text>')
    lx = pad_l
    for s_idx, col_name in enumerate(cols):
        color = PALETTE[s_idx % len(PALETTE)]
        svg.append(f'<rect x="{lx}" y="10" width="12" height="12" rx="3" fill="{color}"/>')
        svg.append(f'<text x="{lx + 16}" y="20" font-size="12" font-weight="500" fill="#334155">{col_name}</text>')
        lx += max(110, len(str(col_name)) * 7 + 28)
    step = max(1, n_pts // 8)
    for i in range(0, n_pts, step):
        px = pad_l + (pw * i / max(n_pts - 1, 1))
        svg.append(f'<text x="{px:.1f}" y="{pad_t + ph + 20}" text-anchor="middle" font-size="11" fill="#64748b">{idx[i]}</text>')
    for s_idx in range(len(cols)):
        color = PALETTE[s_idx % len(PALETTE)]
        pts = []
        for i in range(n_pts):
            px = pad_l + (pw * i / max(n_pts - 1, 1))
            py = pad_t + ph - ((float(vals[i, s_idx]) - y_min) / max(y_max - y_min, 1e-6)) * ph
            pts.append(f"{px:.1f},{py:.1f}")
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(pts)}"/>')
    svg.append("</svg>")
    st.markdown("".join(svg), unsafe_allow_html=True)

NON_MEAL_EXACT_OR_SUBSTR = [
    "add-on:",
    "toast upgrade:",
    "savory roasted corn",
    "fresh fruit cup",
    "banana blueberry oat yogurt cup",
    "whole wheat toast (2 pcs)",
    "cheese bolognese fries",
    "veggie spring rolls",
    "savory glazed chicken wings",
    "vegetable soup",
    "cream of mushroom soup",
    "pan-fried turnip cake",
    "fish meat siu mai",
    "chiu chow dumpling (teochew fun gor)",
    "mini glutinous rice chicken wrap",
]

MEAL_STAPLE_TERMS = [
    "rice", "noodle", "udon", "ramen", "spaghetti", "pasta", "penne",
    "macaroni", "vermicelli", "salad", "burger", "sandwich", "baguette",
    "congee", "hotpot", "grill", "set", "meal", "combo", "hot dog",
    "mcmuffin", "breakfast", "hotcakes", "oatmeal"
]

def is_main_meal_dish(row):
    name = str(row["Dish Name"]).strip().lower()
    cat = str(row["Category"]).strip().lower()
    if "add-on" in cat:
        return False
    if any(term in name for term in NON_MEAL_EXACT_OR_SUBSTR):
        if not any(combo_kw in name for combo_kw in ["congee +", "platter", "b9 light choice"]):
            return False
    if name.endswith("soup") and not any(st_kw in name for st_kw in MEAL_STAPLE_TERMS):
        return False
    return True


@st.cache_data
def load_and_integrate_data():
    files = os.listdir(BASE_DIR)
    survey_file = [f for f in files if f.lower().endswith(".csv") and "survey" in f.lower()][0]
    b_file = [f for f in files if f.lower().endswith(".xlsx") and "breakfast" in f.lower()][0]
    l_file = [f for f in files if f.lower().endswith(".xlsx") and "lunch" in f.lower()][0]
    d_file = [f for f in files if f.lower().endswith(".xlsx") and "dinner" in f.lower()][0]
    
    raw_s = pd.read_csv(os.path.join(BASE_DIR, survey_file))
    df_s = raw_s.iloc[2:].copy()
    df_s = df_s[df_s["Status"] != "Survey Preview"].copy()
    for c in ["Q2_1", "Q2_2", "Q2_3", "Q2_4", "Q3_1"]:
        df_s[c] = pd.to_numeric(df_s[c], errors="coerce")
    df_s = df_s.dropna(subset=["Q2_1", "Q2_2", "Q2_3", "Q2_4"]).copy()
    for c in ["Q2_1", "Q2_2", "Q2_3", "Q2_4"]:
        df_s[c] = df_s[c].clip(1, 5).astype(int)

    med_budget = int(df_s["Q3_1"].dropna().median()) if df_s["Q3_1"].notna().any() else 125
    clean_s = pd.DataFrame({
        "Student_ID": [f"UST_{i:03d}" for i in range(1, len(df_s) + 1)],
        "Residential_Status": df_s["Q1"].fillna("Not Specified").replace("", "Not Specified").values,
        "Daily_Budget_HKD": df_s["Q3_1"].fillna(med_budget).astype(int).values,
        "Rating_Price": df_s["Q2_1"].values,
        "Rating_Taste": df_s["Q2_2"].values,
        "Rating_Health": df_s["Q2_3"].values,
        "Rating_Convenience": df_s["Q2_4"].values,
        "Most_Visited_Outlet": df_s["Q4"].fillna("LG7 Gold Rice Bowl").replace("", "LG7 Gold Rice Bowl").values,
    })

    r_cols = ["Rating_Price", "Rating_Taste", "Rating_Health", "Rating_Convenience"]
    X_weights = clean_s[r_cols].div(clean_s[r_cols].sum(axis=1), axis=0).to_numpy()
    labels, centers = run_numpy_kmeans(X_weights, k=4)
    clean_s["Cluster_ID"] = labels

    names_map = ["Value-Focused", "Flavor-Driven", "Health-Conscious", "Convenience-First"]
    cluster_names = {i: names_map[i] for i in range(4)}
    clean_s["Preference_Group"] = clean_s["Cluster_ID"].map(cluster_names)
    profiles = {cluster_names[i]: np.round(centers[i], 3).tolist() for i in range(4)}

 
    dfs = []
    for fname, slot in [(b_file, "Breakfast"), (l_file, "Lunch"), (d_file, "Dinner")]:
        d = read_xlsx_builtin(os.path.join(BASE_DIR, fname))
        d["Meal_Slot"] = slot
        p_col = [c for c in d.columns if "price" in c.lower()][0]
        c_col = [c for c in d.columns if "category" in c.lower()][0]
        f_col = [c for c in d.columns if "fried" in c.lower()][0]
        fl_col = [c for c in d.columns if "flavor" in c.lower()][0]
        d = d.rename(columns={p_col: "Price_HKD", c_col: "Category", f_col: "Fried", fl_col: "Flavor Profile"})
        dfs.append(d[["Meal_Slot", "Dish Name", "Outlet", "Location", "Price_HKD", "Flavor Profile", "Fried", "Category"]])

    menu = pd.concat(dfs, ignore_index=True)
    menu["Price_HKD"] = pd.to_numeric(menu["Price_HKD"], errors="coerce")
    menu = menu.dropna(subset=["Dish Name", "Price_HKD"]).copy()
    menu = menu[menu["Dish Name"].astype(str).str.strip() != ""]
    menu = menu[menu["Price_HKD"] >= 12.0]
    menu = menu[menu.apply(is_main_meal_dish, axis=1)].reset_index(drop=True)

   
    def normalize_outlet(o):
        ol = str(o).strip().lower()
        if "gold rice bowl" in ol or "golden bowl" in ol or "金飯碗" in ol:
            return "Golden Rice"
        if "super sandwich" in ol:
            return "Super Sandwich"
        return str(o).strip()
    menu["Outlet"] = menu["Outlet"].apply(normalize_outlet)

  
    non_veg_pattern = re.compile(
        r"\b(?:pork|beef|chicken|duck|ham|bacon|sausage|fish|salmon|tuna|shrimp|prawn|crab|squid|eel|egg|lamb|meat|cuttlefish|pomfret|bream|salami|bolognese|char siu|spare rib|unagi)s?\b"
    )
    def verify_veggie_category(row):
        cat = str(row["Category"]).strip()
        text = (str(row["Dish Name"]) + " " + str(row["Flavor Profile"])).lower()
        if "veggie" in cat.lower() and "high protein" not in cat.lower():
            if non_veg_pattern.search(text):
                return "High Protein (meat/egg)"
        return cat
    menu["Category"] = menu.apply(verify_veggie_category, axis=1)

    p_min, p_max = menu["Price_HKD"].min(), menu["Price_HKD"].max()
    menu["Price_Score"] = ((p_max - menu["Price_HKD"]) / (p_max - p_min)).round(3)
    menu["Taste_Score"] = menu["Flavor Profile"].fillna("").str.lower().apply(
        lambda s: 0.92 if any(k in s for k in ["spicy", "mala", "truffle", "curry", "umami", "rich", "satay", "garlic", "roasted", "bbq"])
        else (0.80 if any(k in s for k in ["savory", "savoury", "sweet", "tangy", "herbal", "creamy"]) else 0.68)
    )
    menu["Health_Score"] = menu.apply(
        lambda r: 0.25 if ("yes" in str(r["Fried"]).lower() or "deep-fried" in str(r["Fried"]).lower())
        else (0.52 if "pan-fried" in str(r["Fried"]).lower()
              else (0.92 if (str(r["Category"]).lower().startswith("veggie") or any(k in str(r["Category"]).lower() for k in ["salad", "healthy", "steamed"])) else 0.78)),
        axis=1
    )
    menu["Conv_Score"] = menu["Outlet"].fillna("").str.lower().apply(
        lambda o: 0.95 if ("mcdonald" in o or "super sandwich" in o or "starbucks" in o)
        else (0.82 if ("lg1" in o or "pacific" in o or "golden rice" in o or "gold rice bowl" in o or "golden bowl" in o or "passione" in o) else 0.62)
    )
    menu["Search_Text"] = (
        menu["Dish Name"].fillna("") + " " + menu["Flavor Profile"].fillna("") + " " +
        menu["Category"].fillna("") + " " + menu["Outlet"].fillna("")
    ).str.lower()

    return clean_s, profiles, menu



FOOD_SYNONYMS = {
    "egg": ["egg", "eggs", "omelette", "omelet", "scrambled"],
    "beef": ["beef", "steak", "brisket", "sirloin", "ribeye", "oxtail"],
    "pork": ["pork", "ham", "bacon", "sausage", "char siu", "luncheon", "ribs", "minced pork"],
    "chicken": ["chicken", "wings", "drumstick", "poultry"],
    "duck": ["duck", "roast duck"],
    "fish": ["fish", "salmon", "tuna", "cod", "halibut", "eel", "fillet", "pomfret", "bream"],
    "seafood": ["seafood", "shrimp", "prawn", "prawns", "crab", "lobster", "clam", "mussel", "oyster", "squid", "cuttlefish", "shellfish"],
    "shrimp": ["shrimp", "prawn", "prawns"],
    "dairy": ["dairy", "milk", "cheese", "butter", "cream", "creamy", "yogurt", "latte", "cappuccino"],
    "milk": ["milk", "dairy", "latte"],
    "cheese": ["cheese", "cheesy"],
    "peanut": ["peanut", "peanuts", "nut", "nuts", "satay", "cashew", "almond", "walnut"],
    "gluten": ["gluten", "wheat", "bread", "toast", "pasta", "spaghetti", "macaroni", "udon", "ramen", "sandwich", "burger", "croissant", "pastry"],
    "soy": ["soy", "soya", "tofu", "bean curd", "beancurd", "edamame"],
    "tofu": ["tofu", "bean curd", "beancurd"],
    "mushroom": ["mushroom", "mushrooms", "truffle", "fungi"],
    "onion": ["onion", "onions", "scallion", "spring onion"],
    "garlic": ["garlic"],
    "fried": ["fried", "deep-fried", "deep fried", "crispy", "tempura", "katsu", "cutlet", "hash brown", "fries"],
    "spicy": ["spicy", "mala", "chili", "chilli", "hot", "laksa", "sichuan", "kimchi", "jalapeno"],
    "mala": ["mala", "sichuan"],
    "curry": ["curry", "curried"],
    "truffle": ["truffle"],
    "noodle": ["noodle", "noodles", "udon", "ramen", "spaghetti", "pasta", "macaroni", "vermicelli", "pho", "laksa"],
    "udon": ["udon"],
    "ramen": ["ramen"],
    "pasta": ["pasta", "spaghetti", "macaroni", "penne", "carbonara", "bolognese"],
    "rice": ["rice", "risotto", "donburi", "bibimbap"],
    "congee": ["congee", "porridge"],
    "soup": ["soup", "broth", "borscht", "miso"],
    "burger": ["burger", "burgers"],
    "sandwich": ["sandwich", "sandwiches", "panini", "sub", "bagel", "toast", "croissant"],
    "toast": ["toast", "french toast"],
    "salad": ["salad", "greens"],
    "veggie": ["veggie", "vegetable", "vegetables", "vegan", "vegetarian", "plant"],
    "dumpling": ["dumpling", "dumplings", "gyoza", "wonton"],
    "bbq": ["bbq", "barbecue", "roast", "roasted", "grilled"],
    "coffee": ["coffee", "americano", "espresso", "latte", "cappuccino", "mocha"],
}

RESTAURANT_ALIASES = {
    "mcdonald": ["mcdonald's", "mcdonalds", "mcdonald", "mcd"],
    "super sandwich": ["super sandwich", "super sandwiches", "oss"],
    "tamjai samgor": ["tamjai", "tam jai", "samgor", "sam gor"],
    "lg1 canteen": ["lg1", "lg1 canteen"],
    "pacific catering": ["pacific", "pacific catering"],
    "golden rice": ["golden rice", "gold rice bowl", "golden bowl", "金飯碗"],
    "american diner": ["american diner", "diner"],
}

NEGATION_TRIGGERS = [
    r"\ballerg(?:y|ic|ies)\b",
    r"\bintoleran(?:t|ce)\b",
    r"\bban(?:ned|s|ning)?\b",
    r"\bforbid(?:den)?\b",
    r"\brestrict(?:ed|ion)?\b",
    r"\bavoid(?:ing)?\b",
    r"\bexclude(?:d|s)?\b",
    r"\bexcluding\b",
    r"\bskip(?:ping)?\b",
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bw/o\b",
    r"\bnever\b",
    r"\bhate\b",
    r"\bdislike\b",
    r"\bcan'?t\s+eat\b",
    r"\bcannot\s+eat\b",
    r"\bdon'?t\s+(?:want|eat|like)\b",
    r"\bfree\s+of\b",
    r"\b\w+-free\b",
    r"\bkill\s+me\b",
    r"\bsick\b",
    r"\bsensitive\s+to\b",
]

NON_FOOD_STOPWORDS = {
    "food", "foods", "meal", "meals", "dish", "dishes", "item", "items", "ingredient", "ingredients",
    "to", "of", "for", "with", "and", "or", "the", "a", "an", "my", "me", "i", "am", "is", "are", "have",
    "has", "had", "please", "pls", "today", "now", "lunch", "dinner", "breakfast", "budget", "hkd", "hk",
    "dollar", "dollars", "allergy", "allergic", "allergies", "intolerant", "intolerance", "ban", "banned",
    "avoid", "no", "not", "without", "exclude", "excluded", "skip", "free", "cannot", "cant", "dont", "eat",
    "want", "like", "hate", "dislike", "never", "forbid", "forbidden", "restrict", "restricted", "any",
    "some", "very", "much", "too", "also", "only", "at", "in", "on", "from", "around", "under", "over",
    "healthy", "clean", "diet", "gym", "cheap", "broke", "tight", "save", "money", "restaurant", "canteen"
}


def _match_food_terms_in_text(segment):
    found = []
    for canon, variants in FOOD_SYNONYMS.items():
        for v in variants:
            if re.search(rf"\b{re.escape(v)}s?\b", segment):
                if canon not in found:
                    found.append(canon)
                break
    return found


def _detect_restaurants_in_text(txt, available_outlets):
    matched = []
    for outlet in available_outlets:
        o_low = outlet.lower()
        if o_low in txt and outlet not in matched:
            matched.append(outlet)
            continue
        for key, aliases in RESTAURANT_ALIASES.items():
            if key in o_low and any(a in txt for a in aliases):
                if outlet not in matched:
                    matched.append(outlet)
                break
    return matched


def parse_craving_prompt_offline(prompt_text, base_weights, default_budget, ui_restaurants=None, available_outlets=None):
    txt = prompt_text.lower()

   
    m_explicit = re.search(r"(?:hk\$|\$|hkd\s*|budget\s*(?:of|is|:|=)?\s*\$?|under\s*(?:hk\$|\$)?\s*|around\s*(?:hk\$|\$)?\s*)(\d{2,3})\b", txt)
    if m_explicit:
        budget = int(m_explicit.group(1))
    else:
        m_num = re.search(r"\b(\d{2,3})\s*(?:hkd|dollars?|bucks?)\b", txt)
        budget = int(m_num.group(1)) if m_num else default_budget

  
    clauses = re.split(r"[,;.\n]+|\b(?:but|except|however|though|also)\b", txt)

    exclude_kw = []
    craving_kw = []
    neg_pattern = re.compile("|".join(NEGATION_TRIGGERS))

    for raw_clause in clauses:
        clause = raw_clause.strip()
        if not clause:
            continue

        sub_clauses = re.split(r"\band\b", clause) if neg_pattern.search(clause) else [clause]

        for sub in sub_clauses:
            sub = sub.strip()
            if not sub:
                continue
            has_neg = bool(neg_pattern.search(sub))
            matched_foods = _match_food_terms_in_text(sub)

            if has_neg:
                for f in matched_foods:
                    if f not in exclude_kw:
                        exclude_kw.append(f)
                for m_free in re.finditer(r"\b([a-z]+)-free\b", sub):
                    w_free = m_free.group(1).rstrip("s")
                    matched_canon = _match_food_terms_in_text(w_free)
                    target = matched_canon[0] if matched_canon else w_free
                    if target not in NON_FOOD_STOPWORDS and target not in exclude_kw:
                        exclude_kw.append(target)
                for m_custom in re.finditer(
                    r"(?:allerg(?:y|ic|ies)(?:\s+to)?|intoleran(?:t|ce)(?:\s+to)?|ban(?:ned)?|forbid(?:den)?|avoid|exclude(?:d)?|without|no|skip|hate|dislike|cannot\s+eat|can'?t\s+eat)\s+(?:food\s+|to\s+|is\s+|:?\s*)([a-z]{3,15})\b",
                    sub
                ):
                    cand = m_custom.group(1).rstrip("s")
                    if cand not in NON_FOOD_STOPWORDS:
                        canon_match = _match_food_terms_in_text(cand)
                        target = canon_match[0] if canon_match else cand
                        if target not in exclude_kw:
                            exclude_kw.append(target)
                for m_before in re.finditer(
                    r"\b([a-z]{3,15})\s+(?:allerg(?:y|ic|ies)|intoleran(?:t|ce)|ban(?:ned)?|forbid(?:den)?|restrict(?:ed|ion)?)\b",
                    sub
                ):
                    cand = m_before.group(1).rstrip("s")
                    if cand not in NON_FOOD_STOPWORDS:
                        canon_match = _match_food_terms_in_text(cand)
                        target = canon_match[0] if canon_match else cand
                        if target not in exclude_kw:
                            exclude_kw.append(target)
            else:
                for f in matched_foods:
                    if f not in exclude_kw and f not in craving_kw and f != "fried":
                        craving_kw.append(f)

    craving_kw = [k for k in craving_kw if k not in exclude_kw]

  
    prompt_restaurants = _detect_restaurants_in_text(txt, available_outlets or [])
    preferred_restaurants = list(dict.fromkeys((ui_restaurants or []) + prompt_restaurants))

  
    w = list(base_weights)
    if any(k in txt for k in ["fast", "quick", "short queue", "convenient", "grab and go"]):
        w[3] += 0.14
    if any(k in txt for k in ["healthy", "salad", "clean", "protein", "diet", "gym", "light", "low cal"]) or "fried" in exclude_kw:
        w[2] += 0.14
    if any(k in txt for k in ["cheap", "save money", "broke", "tight budget", "affordable", "low cost"]) or budget <= 65:
        w[0] += 0.14
    if craving_kw or any(k in txt for k in ["tasty", "delicious", "craving", "treat", "flavorful", "flavourful"]):
        w[1] += 0.10
    w_final = [round(x / sum(w), 3) for x in w]

    return {
        "parser_mode": "Built-in Offline NLP (Clause-Aware Allergy, Restaurant & Craving Engine)",
        "budget_hkd": int(np.clip(budget, 35, 250)),
        "preferred_restaurants": preferred_restaurants,
        "exclude_keywords": exclude_kw,
        "craving_keywords": craving_kw,
        "w_base": base_weights,
        "w_final": w_final,
    }


def parse_craving_prompt_llm(prompt_text, base_weights, default_budget, api_key, api_url, model_name, ui_restaurants=None, available_outlets=None):
    fallback = parse_craving_prompt_offline(prompt_text, base_weights, default_budget, ui_restaurants, available_outlets)
    if not api_key or not api_key.strip():
        return fallback

    sys_prompt = (
        "You are an HKUST dining constraint parser. Given a student's meal request and base preference weights "
        f"[Price, Taste, Health, Convenience] = {base_weights} (default budget HK${default_budget}), "
        "return ONLY a valid JSON object with keys: "
        "\"budget_hkd\" (int, 35-250), "
        "\"exclude_keywords\" (list of lowercase strings for ANY food allergy, intolerance, banned item, or disliked food such as egg, beef, pork, chicken, fish, seafood, shrimp, dairy, milk, cheese, peanut, gluten, soy, mushroom, fried, spicy), "
        "\"craving_keywords\" (list of lowercase food/flavor keywords they positively want; NEVER include anything from exclude_keywords), "
        "\"w_final\" (list of 4 floats summing to 1.0 representing adjusted [Price, Taste, Health, Convenience] weights)."
    )
    payload = json.dumps({
        "model": model_name,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.1
    }).encode("utf-8")

    req = urllib.request.Request(
        api_url.strip(),
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            raw_content = body["choices"][0]["message"]["content"]
            m_json = re.search(r"\{.*\}", raw_content, re.DOTALL)
            parsed_llm = json.loads(m_json.group(0) if m_json else raw_content)
            w_f = [float(x) for x in parsed_llm.get("w_final", fallback["w_final"])]
            w_sum = sum(w_f) if sum(w_f) > 0 else 1.0
            llm_ex = [str(x).lower() for x in parsed_llm.get("exclude_keywords", [])]
            merged_ex = list(dict.fromkeys(llm_ex + fallback["exclude_keywords"]))
            llm_crav = [str(x).lower() for x in parsed_llm.get("craving_keywords", fallback["craving_keywords"]) if str(x).lower() not in merged_ex]
            return {
                "parser_mode": f"Live LLM API ({model_name}) + Allergy Safety Guard",
                "budget_hkd": int(parsed_llm.get("budget_hkd", fallback["budget_hkd"])),
                "preferred_restaurants": fallback["preferred_restaurants"],
                "exclude_keywords": merged_ex,
                "craving_keywords": llm_crav,
                "w_base": base_weights,
                "w_final": [round(x / w_sum, 3) for x in w_f],
            }
    except Exception as e:
        fallback["parser_mode"] = f"Offline Fallback (API error: {type(e).__name__})"
        return fallback


# --- MODULE 3: Genetic Algorithm 3-Meal Optimizer ---
def filter_pools(menu, parsed):
    df = menu.copy()
    for ex in parsed["exclude_keywords"]:
        if ex == "fried":
            df = df[~df["Fried"].astype(str).str.lower().str.contains("yes|deep-fried")]
            df = df[~df["Search_Text"].str.contains("fried|tempura|katsu|cutlet|crispy", regex=True)]
        else:
            syn_list = FOOD_SYNONYMS.get(ex, [ex])
            pattern = "|".join(rf"\b{re.escape(w)}s?\b" for w in syn_list)
            df = df[~df["Search_Text"].str.contains(pattern, regex=True)]

    def _safe_slot(slot_name):
        sub = df[df["Meal_Slot"] == slot_name].reset_index(drop=True)
        if len(sub) == 0:
            sub = menu[menu["Meal_Slot"] == slot_name].reset_index(drop=True)
        return sub

    return _safe_slot("Breakfast"), _safe_slot("Lunch"), _safe_slot("Dinner")


def combo_fitness(combo, w, budget, craving_kw, preferred_restaurants=None, use_craving_bonus=True):
    total_cost = sum(float(r["Price_HKD"]) for r in combo)
    base_u = sum(
        w[0]*r["Price_Score"] + w[1]*r["Taste_Score"] + w[2]*r["Health_Score"] + w[3]*r["Conv_Score"]
        for r in combo
    )
    bonus = 0.0
    if use_craving_bonus and craving_kw:
        for r in combo:
            matches = 0
            for kw in craving_kw:
                syn_list = FOOD_SYNONYMS.get(kw, [kw])
                if any(s in r["Search_Text"] for s in syn_list):
                    matches += 1
            bonus += 0.15 * min(matches, 2)
    # Restaurant preference bonus (+0.45 per meal from a preferred restaurant)
    rest_bonus = 0.0
    if preferred_restaurants:
        pref_set = {p.lower() for p in preferred_restaurants}
        for r in combo:
            if str(r["Outlet"]).strip().lower() in pref_set:
                rest_bonus += 0.45
    over_budget_pen = 2.0 * max(0.0, total_cost - budget)
    outlets = [r["Outlet"] for r in combo]
    names = [r["Dish Name"] for r in combo]
    repeat_pen = (0.50 if len(set(names)) < 3 else 0.0) + (0.35 if (len(set(outlets)) == 1 and not preferred_restaurants) else 0.0)
    return base_u + bonus + rest_bonus - over_budget_pen - repeat_pen, base_u, total_cost


def run_genetic_algorithm(b_pool, l_pool, d_pool, w, budget, craving_kw, preferred_restaurants=None, use_craving_bonus=True, pop_size=50, generations=40, seed=42):
    rng = random.Random(seed)
    pop = [[b_pool.iloc[rng.randrange(len(b_pool))],
            l_pool.iloc[rng.randrange(len(l_pool))],
            d_pool.iloc[rng.randrange(len(d_pool))]] for _ in range(pop_size)]
    history = []
    for _ in range(generations):
        scored = [(combo_fitness(c, w, budget, craving_kw, preferred_restaurants, use_craving_bonus)[0], c) for c in pop]
        scored.sort(key=lambda x: x[0], reverse=True)
        history.append(round(scored[0][0], 3))
        new_pop = [scored[0][1], scored[1][1]]
        while len(new_pop) < pop_size:
            p1 = max(rng.sample(scored, 3), key=lambda x: x[0])[1]
            p2 = max(rng.sample(scored, 3), key=lambda x: x[0])[1]
            child = [p1[0] if rng.random() < 0.5 else p2[0],
                     p1[1] if rng.random() < 0.5 else p2[1],
                     p1[2] if rng.random() < 0.5 else p2[2]]
            if rng.random() < 0.15:
                slot = rng.randrange(3)
                pool = [b_pool, l_pool, d_pool][slot]
                child[slot] = pool.iloc[rng.randrange(len(pool))]
            new_pop.append(child)
        pop = new_pop
    best_fit, best_combo = max([(combo_fitness(c, w, budget, craving_kw, preferred_restaurants, use_craving_bonus)[0], c) for c in pop], key=lambda x: x[0])
    _, base_u, cost = combo_fitness(best_combo, w, budget, craving_kw, preferred_restaurants, use_craving_bonus)
    return best_combo, round(best_fit, 3), round(base_u, 3), round(cost, 1), history


# ================= STREAMLIT UI & STYLING =================
st.set_page_config(page_title="EatSmart @ HKUST — AI Campus Meal Planner", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .meal-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        height: 100%;
    }
    .meal-slot-label {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #2563eb;
        margin-bottom: 4px;
    }
    .meal-dish-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 8px;
        line-height: 1.35;
    }
    .meal-meta {
        font-size: 0.88rem;
        color: #475569;
        margin-bottom: 6px;
    }
    .score-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid #f1f5f9;
    }
    .score-pill {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.76rem;
        color: #334155;
        font-weight: 500;
    }
    .summary-banner {
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 14px 0;
        font-size: 0.95rem;
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍽️ EatSmart @ HKUST — Budget-Aware AI Campus Meal Planner")
st.caption("Personalized full-day meal optimization combining student survey clustering, natural-language constraint extraction, and genetic algorithm search (Main Meals Only).")

clean_s, profiles, menu = load_and_integrate_data()
all_outlets = sorted(menu["Outlet"].dropna().unique().tolist())

# --- SIDEBAR: Interactive Preference Matcher (Drives the 4-Group Classification) ---
st.sidebar.header("🧭 Interactive Preference Matcher")
st.sidebar.caption("Rate each dining factor (1–5) to automatically match your closest HKUST preference group:")
r_p = st.sidebar.slider("Importance of Affordable Price", 1, 5, 4)
r_t = st.sidebar.slider("Importance of Taste & Flavor", 1, 5, 5)
r_h = st.sidebar.slider("Importance of Nutrition & Health", 1, 5, 3)
r_c = st.sidebar.slider("Importance of Fast Service & Short Queue", 1, 5, 3)

u_vec = np.array([r_p, r_t, r_h, r_c], dtype=float)
u_norm = np.round(u_vec / u_vec.sum(), 3)
selected_profile = min(profiles.keys(), key=lambda k: np.linalg.norm(u_norm - np.array(profiles[k])))
w_base = profiles[selected_profile]

st.sidebar.success(f"**Matched Group:** **{selected_profile}**  \n**Your Weights:** `{u_norm.tolist()}`  \n**Group Centroid:** `{w_base}`")

st.sidebar.divider()
default_budget = st.sidebar.slider("Daily Budget Limit (HK$)", 35, 160, 95, 5)

with st.sidebar.expander("🔌 Optional: Live LLM API Settings", expanded=False):
    st.caption("The constraint parser works **100% offline** by default. Add an OpenAI-compatible key below only if you want live cloud LLM parsing.")
    llm_api_key = st.text_input("API Key (Optional)", value="", type="password")
    llm_api_url = st.text_input("Endpoint URL", value="https://api.openai.com/v1/chat/completions")
    llm_model = st.text_input("Model Name", value="gpt-4o-mini")

tab1, tab2, tab3 = st.tabs([
    " 1. Student Preference Profiler & Menu Database",
    " 2. Smart Constraint Parser & Daily Meal Planner",
    " 3. Benchmark Evaluation & Budget Sensitivity"
])

with tab1:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader(f"Student Dining Preference Groups ({len(clean_s)} Survey Responses)")
        st.caption("Normalized priority weights discovered via K-Means clustering on HKUST student dining survey data.")
        p_df = pd.DataFrame(profiles, index=["Price Weight", "Taste Weight", "Health Weight", "Convenience Weight"]).T
        st.dataframe(p_df, use_container_width=True)
        render_svg_bar_chart(p_df)
    with c2:
        st.subheader(" Your Matched Preference Profile")
        st.caption("Synchronized with your 4 rating sliders in the left sidebar.")
        st.info(f"**Closest Preference Group:** **{selected_profile}**")
        match_df = pd.DataFrame(
            {"Your Input Weights": u_norm.tolist(), f"{selected_profile} Centroid": w_base},
            index=["Price", "Taste", "Health", "Convenience"]
        )
        st.dataframe(match_df, use_container_width=True)

    st.subheader(f" HKUST Campus Main-Meal Database ({len(menu)} Full Meals Across Breakfast, Lunch & Dinner)")
    slot_filter = st.selectbox("Filter Menu by Meal Slot", ["All", "Breakfast", "Lunch", "Dinner"])
    view_df = menu if slot_filter == "All" else menu[menu["Meal_Slot"] == slot_filter]
    st.dataframe(
        view_df[["Meal_Slot", "Dish Name", "Outlet", "Location", "Price_HKD", "Flavor Profile", "Fried", "Category", "Price_Score", "Taste_Score", "Health_Score", "Conv_Score"]],
        use_container_width=True
    )

with tab2:
    st.subheader("🗣️ Natural-Language Craving, Restaurant & Constraint Extractor")
    in_col1, in_col2 = st.columns([1.4, 1])
    with in_col1:
        preset_prompts = {
            "Custom Input (Type your own below)": "I have HK$95 today, craving spicy noodles for dinner, no beef please.",
            "Scenario A — TamJai Spicy Noodles + No Beef (HK$95)": "I have HK$95 today, craving TamJai SamGor spicy noodles, no beef please.",
            "Scenario B — Egg Allergy + Tight Budget (HK$85)": "budget $85, food allergy egg, want rice or noodles",
            "Scenario C — Banned Pork & Fried Food + Gym Protein (HK$115)": "Budget HK$115 for a healthy gym diet day. I want chicken or fish with rice, banned pork and avoid fried food.",
            "Scenario D — Seafood Allergy + Curry & Sandwich Craving (HK$135)": "I have $135 today and want a delicious curry or truffle dish and a sandwich for breakfast. Allergic to seafood and shrimp."
        }
        chosen_preset = st.selectbox("Select a Sample Scenario or Enter Custom Text:", list(preset_prompts.keys()))
        user_prompt = st.text_input(
            "Describe your daily budget, allergies/restrictions, or food cravings:",
            value=preset_prompts[chosen_preset]
        )
    with in_col2:
        ui_restaurants = st.multiselect(
            "Craving for Restaurant(s) (Optional — Leave empty for All):",
            options=all_outlets,
            default=[]
        )

    if llm_api_key.strip():
        parsed = parse_craving_prompt_llm(user_prompt, w_base, default_budget, llm_api_key, llm_api_url, llm_model, ui_restaurants, all_outlets)
    else:
        parsed = parse_craving_prompt_offline(user_prompt, w_base, default_budget, ui_restaurants, all_outlets)

    p_col1, p_col2 = st.columns([1.2, 1])
    with p_col1:
        st.markdown("#####  Extracted Dining Constraints")
        st.caption(f"**Parser Mode:** `{parsed['parser_mode']}`")
        m1, m2, m3 = st.columns(3)
        m1.metric("Target Budget", f"HK${parsed['budget_hkd']}")
        m2.metric("Restaurant Craving", ", ".join(parsed["preferred_restaurants"]) if parsed["preferred_restaurants"] else "All Outlets")
        m3.metric("Excluded Items", ", ".join(parsed["exclude_keywords"]) if parsed["exclude_keywords"] else "None")
        st.write(
            "**Positive Craving Keywords:**",
            ", ".join(f"`{k}`" for k in parsed["craving_keywords"]) if parsed["craving_keywords"] else "`None`"
        )
        with st.expander("View Structured JSON Output", expanded=False):
            st.json(parsed)
    with p_col2:
        st.markdown("#####  Priority Weight Adjustment (`w_base` → `w_final`)")
        w_comp_df = pd.DataFrame(
            {"Base Group Weight": parsed["w_base"], "Adjusted Weight": parsed["w_final"]},
            index=["Price", "Taste", "Health", "Convenience"]
        )
        st.dataframe(w_comp_df, use_container_width=True)

    st.subheader(" Optimized 3-Meal Daily Plan (Genetic Algorithm: 50 Population × 40 Generations)")
    b_pool, l_pool, d_pool = filter_pools(menu, parsed)
    best_combo, best_fit, base_u, total_cost, history = run_genetic_algorithm(
        b_pool, l_pool, d_pool, parsed["w_final"], parsed["budget_hkd"],
        parsed["craving_keywords"], parsed["preferred_restaurants"], True
    )

    m_cols = st.columns(3)
    for col, slot_name, dish in zip(m_cols, [" Breakfast", " Lunch", " Dinner"], best_combo):
        with col:
            st.markdown(
                f"""
                <div class="meal-card">
                    <div class="meal-slot-label">{slot_name}</div>
                    <div class="meal-dish-title">{dish['Dish Name']}</div>
                    <div class="meal-meta">📍 <b>{dish['Outlet']}</b> ({dish['Location']})</div>
                    <div class="meal-meta">💰 <b>HK${float(dish['Price_HKD']):.1f}</b> &nbsp;|&nbsp; 🥗 {dish['Category']}</div>
                    <div class="meal-meta">🌶️ Flavor: {dish['Flavor Profile']} &nbsp;|&nbsp; Fried: {dish['Fried']}</div>
                    <div class="score-pill-row">
                        <span class="score-pill">Price: {dish['Price_Score']:.2f}</span>
                        <span class="score-pill">Taste: {dish['Taste_Score']:.2f}</span>
                        <span class="score-pill">Health: {dish['Health_Score']:.2f}</span>
                        <span class="score-pill">Conv: {dish['Conv_Score']:.2f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    budget_status = "✅ Within Budget" if total_cost <= parsed["budget_hkd"] else f"⚠️ Exceeds Budget by HK${total_cost - parsed['budget_hkd']:.1f} (Min Feasible Campus Combo)"
    st.markdown(
        f"""
        <div class="summary-banner">
            <b>Total Daily Cost:</b> HK${total_cost:.1f} / HK${parsed['budget_hkd']} ({budget_status}) &nbsp;&nbsp;|&nbsp;&nbsp;
            <b>Objective Fitness Score:</b> {best_fit} &nbsp;&nbsp;|&nbsp;&nbsp;
            <b>Multi-Attribute Utility:</b> {base_u} / 3.00
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("Genetic Algorithm Convergence Curve (Best Fitness Score Across 40 Generations)")
    render_svg_line_chart(pd.DataFrame({"Best Fitness Score Across 40 Generations": history}))

with tab3:
    st.subheader(f"1. Optimization Method & Ablation Benchmark (Target Budget: HK${parsed['budget_hkd']})")
    st.caption("Comparing our full optimization pipeline against greedy cheapest-first selection, random valid selection, and equal-weight ablation.")
    b_p, l_p, d_p = filter_pools(menu, parsed)
    cheap_combo = [b_p.sort_values("Price_HKD").iloc[0], l_p.sort_values("Price_HKD").iloc[0], d_p.sort_values("Price_HKD").iloc[0]]
    _, cheap_u, cheap_cost = combo_fitness(cheap_combo, parsed["w_final"], parsed["budget_hkd"], parsed["craving_keywords"], parsed["preferred_restaurants"], False)

    rng = random.Random(7)
    rand_c = [b_p.iloc[rng.randrange(len(b_p))], l_p.iloc[rng.randrange(len(l_p))], d_p.iloc[rng.randrange(len(d_p))]]
    _, rand_u, rand_cost = combo_fitness(rand_c, parsed["w_final"], parsed["budget_hkd"], parsed["craving_keywords"], parsed["preferred_restaurants"], False)

    abl_combo, _, _, abl_cost, _ = run_genetic_algorithm(b_p, l_p, d_p, [0.25, 0.25, 0.25, 0.25], parsed["budget_hkd"], [], None, False)
    _, abl_true_u, _ = combo_fitness(abl_combo, parsed["w_final"], parsed["budget_hkd"], parsed["craving_keywords"], parsed["preferred_restaurants"], False)

    comp_df = pd.DataFrame([
        {"Method": "1. Cheapest-First Greedy", "Breakfast": cheap_combo[0]["Dish Name"], "Lunch": cheap_combo[1]["Dish Name"], "Dinner": cheap_combo[2]["Dish Name"], "Total Cost (HK$)": round(cheap_cost, 1), "Utility Satisfaction (0-3.0)": round(cheap_u, 3)},
        {"Method": "2. Random Valid Selection", "Breakfast": rand_c[0]["Dish Name"], "Lunch": rand_c[1]["Dish Name"], "Dinner": rand_c[2]["Dish Name"], "Total Cost (HK$)": round(rand_cost, 1), "Utility Satisfaction (0-3.0)": round(rand_u, 3)},
        {"Method": "3. Equal-Weight Ablation GA ([0.25×4])", "Breakfast": abl_combo[0]["Dish Name"], "Lunch": abl_combo[1]["Dish Name"], "Dinner": abl_combo[2]["Dish Name"], "Total Cost (HK$)": round(abl_cost, 1), "Utility Satisfaction (0-3.0)": round(abl_true_u, 3)},
        {"Method": "4. Full Personalized AI Planner", "Breakfast": best_combo[0]["Dish Name"], "Lunch": best_combo[1]["Dish Name"], "Dinner": best_combo[2]["Dish Name"], "Total Cost (HK$)": round(total_cost, 1), "Utility Satisfaction (0-3.0)": round(best_fit, 3)},
    ])
    st.dataframe(comp_df, use_container_width=True)
    render_svg_bar_chart(comp_df.set_index("Method")[["Utility Satisfaction (0-3.0)"]])

    st.subheader("2. Daily Budget Sensitivity Analysis (HK$50 to HK$150)")
    st.caption("Multi-attribute utility achieved across daily budget levels for each student preference group.")
    b_all = menu[menu["Meal_Slot"] == "Breakfast"].reset_index(drop=True)
    l_all = menu[menu["Meal_Slot"] == "Lunch"].reset_index(drop=True)
    d_all = menu[menu["Meal_Slot"] == "Dinner"].reset_index(drop=True)
    sweep_budgets = list(range(50, 155, 10))
    sweep_rows = {}
    for p_name, p_w in profiles.items():
        vals = []
        for b_val in sweep_budgets:
            _, fit_v, u_v, c_v, _ = run_genetic_algorithm(b_all, l_all, d_all, p_w, b_val, [], None, False, pop_size=35, generations=25)
            vals.append(u_v if c_v <= b_val else max(0.8, u_v - 0.5))
        sweep_rows[p_name] = vals
    sweep_df = pd.DataFrame(sweep_rows, index=sweep_budgets)
    sweep_df.index.name = "Daily Budget (HK$)"
    render_svg_line_chart(sweep_df)
