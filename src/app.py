import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from rapidfuzz import process as fz
import re
from datetime import date, timedelta

st.set_page_config(page_title="NL Healthcare Analytics", layout="wide")
st.title("🧠 Natural Language Healthcare Data Assistant")

# =============================================================================
# FILE UPLOAD
# =============================================================================

uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file)

# Convert numeric-suspicious columns
for c in df.columns:
    clean = df[c].astype(str).str.replace(r"[,$ ]", "", regex=True)
    if clean.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.7:
        df[c] = pd.to_numeric(clean, errors="coerce")

conn = sqlite3.connect(":memory:")
df.to_sql("data", conn, index=False, if_exists="replace")

st.success(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
st.dataframe(df.head())

# =============================================================================
# LOAD INTENT MODEL
# =============================================================================

try:
    clf = pipeline("text-classification", model="intent_model", tokenizer="intent_model")
except Exception as e:
    st.error("❌ Could not load intent_model folder. Place it next to app.py.")
    st.stop()


# =============================================================================
# COLUMN EXTRACTION
# =============================================================================

def extract_columns(q):
    q = q.lower()
    cols = df.columns.tolist()

    # 1. word match
    words = re.findall(r"[a-zA-Z]+", q)
    exact = [c for c in cols if c.lower() in words]
    if exact:
        return exact

    # 2. keyword map
    keyword_map = {
        "Insurance Provider": ["insurance", "provider"],
        "Medical Condition": ["condition", "asthma", "flu", "covid", "diabetes", "arthritis", "hypertension"],
        "Billing Amount": ["billing", "amount", "charge", "cost", "payment", "price"],
        "Gender": ["male", "female", "gender"],
        "Age": ["age", "older", "younger"],
        "Admission Type": ["emergency", "routine", "referral"],
        "Blood Type": ["blood", "blood type"],
        "Doctor": ["doctor", "dr"],
        "Medication": ["medication", "drug", "treatment"],
        "Test Result": ["positive", "negative", "test"],
        "Hospital": ["hospital", "clinic"],
        "Room Number": ["room"]
    }

    for col, keys in keyword_map.items():
        if any(k in q for k in keys):
            return [col]

    # 3. fuzzy fallback
    names = [c.lower() for c in cols]
    best = fz.extractOne(q, names)
    if best and best[1] >= 80:
        return [cols[best[2]]]

    return []


# =============================================================================
# MULTIPLE CATEGORICAL MATCHES + OR LOGIC
# =============================================================================

def detect_value_matches(q):
    """
    Returns dict like:
    {
       "Medical Condition": ["asthma", "flu"],
       "Test Result": ["positive"]
    }
    Supports OR-logic.
    """
    ql = q.lower()
    matches = {}

    # Special blood type detection
    bloods = ["a+", "a-", "b+", "b-", "o+", "o-", "ab+", "ab-"]
    merged = ql.replace(" ", "")
    for b in bloods:
        if b in merged:
            if "Blood Type" in df.columns:
                matches.setdefault("Blood Type", []).append(b)

    # Explicit test result detection
    if "positive" in ql:
        matches.setdefault("Test Result", []).append("positive")
    if "negative" in ql:
        matches.setdefault("Test Result", []).append("negative")

    # OR-split logic
    parts = re.split(r"\bor\b", ql)

    for chunk in parts:
        chunk = chunk.strip()
        for col in df.columns:
            if df[col].dtype == object and df[col].nunique() <= 300:
                vals = [str(v).lower() for v in df[col].dropna().unique()]
                m = fz.extractOne(chunk, vals, score_cutoff=90)
                if m:
                    matches.setdefault(col, []).append(m[0])

    return matches



# =============================================================================
# NUMERIC FILTERS (NO K-PARSING)
# =============================================================================

def extract_numeric_filters(q):
    """
    Handles:
        age under 80
        billing less than 12000
        age over 60
        billing above 20000
    """
    ql = q.lower().replace("$", "").replace(",", "")
    conds = []

    # Age
    if "age" in ql or "years old" in ql:
        m_under = re.search(r"(under|less than|below)\s+(\d+)", ql)
        m_over = re.search(r"(over|greater than|above)\s+(\d+)", ql)
        if m_under:
            conds.append(f"`Age` < {int(m_under.group(2))}")
        if m_over:
            conds.append(f"`Age` > {int(m_over.group(2))}")

    # Billing
    if any(x in ql for x in ["billing", "charge", "payment", "amount", "price", "cost"]):
        m_under = re.search(r"(under|less than|below)\s+(\d+)", ql)
        m_over = re.search(r"(over|greater than|above)\s+(\d+)", ql)
        if m_under:
            conds.append(f"`Billing Amount` < {int(m_under.group(2))}")
        if m_over:
            conds.append(f"`Billing Amount` > {int(m_over.group(2))}")

    return conds



# =============================================================================
# DATE RANGE DETECTION
# =============================================================================

def parse_dates(q):
    q = q.lower()
    today = date.today()

    if "last year" in q:
        return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)

    if "last 30 days" in q:
        return today - timedelta(days=30), today

    if "last month" in q:
        end = today.replace(day=1) - timedelta(days=1)
        start = end.replace(day=1)
        return start, end

    return None, None


# =============================================================================
# COMPARISON TARGET DETECTION
# =============================================================================

def detect_comparison_target(q):
    q = q.lower()
    compare_map = {
        "Gender": ["gender", "male", "female"],
        "Blood Type": ["blood", "blood type", "a+", "o+", "ab+", "b-"],
        "Medical Condition": ["condition", "asthma", "flu", "covid", "diabetes", "arthritis", "hypertension"],
        "Admission Type": ["admission", "emergency", "routine", "referral"],
        "Hospital": ["hospital", "clinic"],
        "Insurance Provider": ["aetna", "cigna", "insurance", "provider"],
        "Doctor": ["doctor", "dr"],
        "Medication": ["medication", "drug", "treatment"],
        "Test Result": ["positive", "negative", "test"],
    }
    for col, keys in compare_map.items():
        if any(k in q for k in keys):
            return col
    return None


# =============================================================================
# INTENT
# =============================================================================

def get_intent(q):
    return clf(q)[0]["label"]


# =============================================================================
# SQL BUILDER WITH OR-LOGIC + NUMERIC FIX
# =============================================================================

def build_sql(query):
    q = query.lower()
    intent = get_intent(query)

    value_matches = detect_value_matches(query)
    numeric_filters = extract_numeric_filters(query)
    start, end = parse_dates(query)
    cols = extract_columns(query)

    # Build WHERE
    where = []

    # OR logic for categorical filters
    for col, vals in value_matches.items():
        if len(vals) == 1:
            where.append(f"LOWER(`{col}`) = LOWER('{vals[0]}')")
        else:
            ors = " OR ".join([f"LOWER(`{col}`)=LOWER('{v}')" for v in vals])
            where.append("(" + ors + ")")

    # Numeric
    for cond in numeric_filters:
        where.append(cond)

    # Date
    if start and end:
        for col in df.columns:
            if "date" in col.lower():
                where.append(f"date(`{col}`) BETWEEN date('{start}') AND date('{end}')")

    where_sql = "WHERE " + " AND ".join(where) if where else ""

    # INTENT HANDLING
    if intent == "filter":
        return intent, f"SELECT * FROM data {where_sql} LIMIT 200;"

    if intent == "count":
        return intent, f"SELECT COUNT(*) AS total FROM data {where_sql};"

    if intent == "aggregate":
        group_col = cols[0] if cols else "Insurance Provider"
        return (
            intent,
            f"""
            SELECT `{group_col}`, AVG(`Billing Amount`) AS value
            FROM data
            {where_sql}
            GROUP BY `{group_col}`
            ORDER BY value DESC;
            """
        )

    if intent == "compare":
        group_col = detect_comparison_target(query) or (cols[0] if cols else "Gender")
        metric = "AVG(`Billing Amount`)" if any(x in q for x in ["billing", "amount", "cost", "price"]) else "COUNT(*)"
        return (
            intent,
            f"""
            SELECT `{group_col}`, {metric} AS value
            FROM data
            {where_sql}
            GROUP BY `{group_col}`
            ORDER BY value DESC;
            """
        )

    if intent == "trend":
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        return (
            intent,
            f"""
            SELECT `{date_col}` AS dt, COUNT(*) AS value
            FROM data
            GROUP BY dt
            ORDER BY dt;
            """
        )

    return intent, "SELECT * FROM data LIMIT 100;"


# =============================================================================
# DRILL-DOWN FILTERS
# =============================================================================

def apply_drill_filters(result):
    st.subheader("🔍 Drill-Down Filters")
    filtered = result.copy()

    for col in result.columns:
        if col not in ["value", "dt"]:
            vals = sorted(list(filtered[col].dropna().unique()))
            if len(vals) > 1:
                choose = st.multiselect(f"Filter by {col}", vals)
                if choose:
                    filtered = filtered[filtered[col].isin(choose)]
    return filtered


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize(intent, result):
    if result.empty:
        st.warning("⚠ No results found")
        return

    # DRILL DOWN
    result = apply_drill_filters(result)

    group_col = next((c for c in result.columns if c not in ["value", "dt"]), None)

    st.subheader("📊 Visualization Options")
    chart = st.selectbox("Choose a chart type:", ["Bar Chart", "Pie Chart", "Line Chart", "Table Only"])

    # LINE CHART
    if chart == "Line Chart":
        if "dt" not in result.columns:
            st.error("❌ No date column available for trend plot.")
        else:
            result["dt"] = pd.to_datetime(result["dt"], errors="coerce")
            fig, ax = plt.subplots(figsize=(10,4))
            sns.lineplot(data=result, x="dt", y="value", marker="o")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # BAR CHART
    elif chart == "Bar Chart" and group_col:
        fig, ax = plt.subplots(figsize=(10,4))
        sns.barplot(data=result, x=group_col, y="value", palette="Blues_d")
        plt.xticks(rotation=45)
        # Labels
        for i, v in enumerate(result["value"]):
            ax.text(i, v + max(result["value"])*0.02, str(v), ha="center")
        plt.tight_layout()
        st.pyplot(fig)

    # PIE CHART
    elif chart == "Pie Chart" and group_col:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.pie(result["value"], labels=result[group_col], autopct="%1.1f%%")
        st.pyplot(fig)

    # TABLE
    st.subheader("📋 Table Result")
    st.dataframe(result)

    # INSIGHTS
    st.subheader("🧠 Insights")
    try:
        if intent in ["aggregate", "compare"] and group_col:
            idx_hi = result["value"].idxmax()
            idx_lo = result["value"].idxmin()
            hi_cat = result.loc[idx_hi, group_col]
            lo_cat = result.loc[idx_lo, group_col]
            st.write(f"• **Highest:** {hi_cat} → {result['value'][idx_hi]}")
            st.write(f"• **Lowest:** {lo_cat} → {result['value'][idx_lo]}")
        elif intent == "trend":
            if result["value"].iloc[-1] > result["value"].iloc[0]:
                st.write("📈 Upward trend detected.")
            else:
                st.write("📉 Downward trend detected.")
    except:
        st.write("ℹ No insights available.")


# =============================================================================
# MAIN QUERY BOX
# =============================================================================

query = st.text_input("💬 Ask a question about your data:")

if query:
    intent, sql = build_sql(query)
    st.info(f"🎯 Intent detected: **{intent}**")
    st.code(sql, language="sql")

    try:
        result_df = pd.read_sql_query(sql, conn)
        visualize(intent, result_df)
    except Exception as e:
        st.error(f"SQL Error: {e}")
