import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import re
from datetime import date, timedelta

st.set_page_config(page_title="Natural Language Healthcare Analytics", layout="wide")
st.title("Natural Language Healthcare Data Assistant")

# =====================================================================
# DATA LOADING
# =====================================================================

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file)

# convert likely numeric columns
for c in df.columns:
    clean = df[c].astype(str).str.replace(r"[,$ ]", "", regex=True)
    if clean.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.7:
        df[c] = pd.to_numeric(clean, errors="coerce")

# in-memory SQL
conn = sqlite3.connect(":memory:")
df.to_sql("data", conn, index=False, if_exists="replace")

st.write(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
st.dataframe(df.head())

# =====================================================================
# INTENT MODEL
# =====================================================================

try:
    clf = pipeline("text-classification", model="intent_model", tokenizer="intent_model")
except Exception as e:
    st.error(f"Could not load intent_model. Error: {e}")
    st.stop()


def get_intent(text: str) -> str:
    return clf(text)[0]["label"]


# =====================================================================
# RULE DICTIONARIES
# =====================================================================

# mapping from (column, canonical value) -> keywords to detect it
SIMPLE_VALUE_RULES = [
    # Gender
    {"col": "Gender", "val": "female", "kw": ["female", "woman", "women", "lady", "ladies"]},
    {"col": "Gender", "val": "male", "kw": ["male", "man", "men", "gentleman", "gentlemen"]},

    # Test result
    {"col": "Test Result", "val": "positive", "kw": ["tested positive", "test positive", "positive"]},
    {"col": "Test Result", "val": "negative", "kw": ["tested negative", "test negative", "negative"]},

    # Admission type
    {"col": "Admission Type", "val": "emergency", "kw": ["emergency", "er"]},
    {"col": "Admission Type", "val": "routine", "kw": ["routine", "scheduled"]},
    {"col": "Admission Type", "val": "referral", "kw": ["referral", "referred"]},

    # Medical conditions (add more if your data has them)
    {"col": "Medical Condition", "val": "asthma", "kw": ["asthma"]},
    {"col": "Medical Condition", "val": "flu", "kw": ["flu", "influenza"]},
    {"col": "Medical Condition", "val": "covid", "kw": ["covid", "covid-19", "corona"]},
    {"col": "Medical Condition", "val": "diabetes", "kw": ["diabetes", "diabetic"]},
    {"col": "Medical Condition", "val": "hypertension", "kw": ["hypertension", "high blood pressure"]},
    {"col": "Medical Condition", "val": "arthritis", "kw": ["arthritis"]},

    # Blood type
    {"col": "Blood Type", "val": "o+", "kw": ["o+", "o positive"]},
    {"col": "Blood Type", "val": "o-", "kw": ["o-", "o negative"]},
    {"col": "Blood Type", "val": "a+", "kw": ["a+", "a positive"]},
    {"col": "Blood Type", "val": "a-", "kw": ["a-", "a negative"]},
    {"col": "Blood Type", "val": "b+", "kw": ["b+", "b positive"]},
    {"col": "Blood Type", "val": "b-", "kw": ["b-", "b negative"]},
    {"col": "Blood Type", "val": "ab+", "kw": ["ab+", "ab positive"]},
    {"col": "Blood Type", "val": "ab-", "kw": ["ab-", "ab negative"]},

    # Insurance provider (example names)
    {"col": "Insurance Provider", "val": "aetna", "kw": ["aetna"]},
    {"col": "Insurance Provider", "val": "cigna", "kw": ["cigna"]},
    {"col": "Insurance Provider", "val": "blue cross", "kw": ["blue cross", "bluecross"]},
    {"col": "Insurance Provider", "val": "united", "kw": ["united", "uhc", "united healthcare"]},
]

# group-by synonym map for aggregate / compare
GROUP_SYNONYMS = {
    "Insurance Provider": ["insurance provider", "insurer", "provider"],
    "Admission Type": ["admission type", "admission", "type of admission"],
    "Gender": ["gender", "sex"],
    "Blood Type": ["blood type", "blood group"],
    "Medical Condition": ["medical condition", "condition", "disease", "diagnosis"],
    "Hospital": ["hospital", "clinic"],
    "Test Result": ["test result", "result"],
    "Medication": ["medication", "drug", "treatment"],
}


# =====================================================================
# CATEGORICAL FILTERS
# =====================================================================

def extract_simple_categorical_filters(query: str):
    """
    Returns dict: col -> set(values)
    based purely on keyword rules above.
    """
    ql = query.lower()
    filters = {}

    for rule in SIMPLE_VALUE_RULES:
        col, val, kws = rule["col"], rule["val"], rule["kw"]

        if col not in df.columns:
            continue

        if any(k in ql for k in kws):
            filters.setdefault(col, set()).add(val)

    return filters


# =====================================================================
# NUMERIC FILTERS (AGE, BILLING)
# =====================================================================

def extract_numeric_filters(query: str):
    """
    Build numeric conditions for Age and Billing Amount.

    Handles patterns like:
      - over 40
      - over age 60
      - older than 65
      - under 30
      - younger than 25
      - billing over 20000
      - billing amount less than 12000
    """
    ql = query.lower().replace("$", "").replace(",", "")
    conds = []

    has_age_col = "Age" in df.columns
    has_bill_col = "Billing Amount" in df.columns

    # Helper: generic window around match to decide if it is billing-related
    billing_words = ["billing", "amount", "cost", "charge", "payment", "price"]

    # 1. explicit "older than/younger than"
    if has_age_col:
        for m in re.finditer(r"(older than|greater than)\s+(\d+)", ql):
            num = int(m.group(2))
            conds.append(f"`Age` > {num}")
        for m in re.finditer(r"(younger than|less than|under)\s+(\d+)", ql):
            num = int(m.group(2))
            conds.append(f"`Age` < {num}")

        # patterns with "age" word
        for m in re.finditer(r"(over|above)\s+age\s+(\d+)", ql):
            num = int(m.group(2))
            conds.append(f"`Age` > {num}")
        for m in re.finditer(r"(under|below)\s+age\s+(\d+)", ql):
            num = int(m.group(2))
            conds.append(f"`Age` < {num}")

    # 2. generic "over 40", "under 30"
    for m in re.finditer(r"(over|above|greater than|under|below|less than)\s+(\d+)", ql):
        op_word, num_str = m.groups()
        num = int(num_str)
        op = ">" if op_word in ["over", "above", "greater than"] else "<"

        start, end = m.span()
        window_start = max(0, start - 25)
        window_end = min(len(ql), end + 25)
        window = ql[window_start:window_end]

        # If billing words appear near, treat as billing
        if any(w in window for w in billing_words) and has_bill_col:
            conds.append(f"`Billing Amount` {op} {num}")
        # otherwise, if we have Age column, treat as age
        elif has_age_col:
            conds.append(f"`Age` {op} {num}")

    # 3. explicit billing patterns
    if has_bill_col:
        for m in re.finditer(
            r"(billing amount|billing|amount|cost|charge|payment)\s+(over|above|greater than|under|below|less than)\s+(\d+)",
            ql,
        ):
            _, op_word, num_str = m.groups()
            num = int(num_str)
            op = ">" if op_word in ["over", "above", "greater than"] else "<"
            conds.append(f"`Billing Amount` {op} {num}")

    # deduplicate
    conds = list(dict.fromkeys(conds))
    return conds


# =====================================================================
# DATE FILTERS (simple)
# =====================================================================

def parse_date_range(query: str):
    q = query.lower()
    today = date.today()

    if "last year" in q:
        return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)

    if "last month" in q:
        end = today.replace(day=1) - timedelta(days=1)
        start = end.replace(day=1)
        return start, end

    if "last 30 days" in q:
        return today - timedelta(days=30), today

    return None, None


def pick_date_column():
    for c in df.columns:
        if "date" in c.lower():
            return c
    return None


# =====================================================================
# GROUP BY COLUMN DETECTION (for aggregate / compare)
# =====================================================================

def detect_group_column(query: str):
    ql = query.lower()

    # Prefer phrase after "by ...".
    m = re.search(r"\bby ([a-z ]+)", ql)
    phrase = None
    if m:
        phrase = m.group(1)
        # cut at "for", "with", "who", "that"
        phrase = re.split(r"\b(for|with|who|that|where)\b", phrase)[0].strip()

    # If we have a phrase, use it; otherwise use full query
    candidate_texts = [phrase] if phrase else []
    candidate_texts.append(ql)

    for text in candidate_texts:
        for col, variants in GROUP_SYNONYMS.items():
            if col in df.columns and any(v in text for v in variants):
                return col

    # fallback heuristics
    if "Insurance Provider" in df.columns:
        return "Insurance Provider"
    if "Medical Condition" in df.columns:
        return "Medical Condition"
    if "Admission Type" in df.columns:
        return "Admission Type"
    if "Gender" in df.columns:
        return "Gender"

    # last fallback: first non-numeric column
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[0]


# =====================================================================
# WHERE CLAUSE BUILDER
# =====================================================================

def build_where_clauses(query: str):
    clauses = []

    # categorical
    cat_filters = extract_simple_categorical_filters(query)
    for col, vals in cat_filters.items():
        if len(vals) == 1:
            val = list(vals)[0]
            clauses.append(f"LOWER(`{col}`) = LOWER('{val}')")
        else:
            parts = [f"LOWER(`{col}`) = LOWER('{v}')" for v in vals]
            clauses.append("(" + " OR ".join(parts) + ")")

    # numeric
    clauses.extend(extract_numeric_filters(query))

    # date ranges
    start, end = parse_date_range(query)
    if start and end:
        date_col = pick_date_column()
        if date_col:
            clauses.append(f"date(`{date_col}`) BETWEEN date('{start}') AND date('{end}')")

    return clauses


# =====================================================================
# SQL BUILDER
# =====================================================================

def build_sql(query: str):
    intent = get_intent(query)
    ql = query.lower()

    where_clauses = build_where_clauses(query)
    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # FILTER
    if intent == "filter":
        sql = f"SELECT * FROM data {where_sql} LIMIT 200;"
        return intent, sql

    # COUNT
    if intent == "count":
        sql = f"SELECT COUNT(*) AS value FROM data {where_sql};"
        return intent, sql

    # AGGREGATE / COMPARE (both return category + value)
    if intent in ["aggregate", "compare"]:
        group_col = detect_group_column(query)

        # decide metric
        if any(w in ql for w in ["number of", "count of", "how many", "count"]):
            metric = "COUNT(*)"
        elif "Billing Amount" in df.columns and any(w in ql for w in ["billing", "amount", "cost", "price", "charge"]):
            metric = "AVG(`Billing Amount`)"
        elif "Billing Amount" in df.columns:
            metric = "AVG(`Billing Amount`)"
        else:
            metric = "COUNT(*)"

        sql = f"""
        SELECT `{group_col}` AS category,
               {metric} AS value
        FROM data
        {where_sql}
        GROUP BY `{group_col}`
        ORDER BY value DESC;
        """
        return intent, sql

    # TREND
    if intent == "trend":
        date_col = pick_date_column() or df.columns[0]
        sql = f"""
        SELECT `{date_col}` AS dt,
               COUNT(*) AS value
        FROM data
        {where_sql}
        GROUP BY dt
        ORDER BY dt;
        """
        return intent, sql

    # fallback
    sql = "SELECT * FROM data LIMIT 100;"
    return intent, sql


# =====================================================================
# DRILLDOWN FILTERING
# =====================================================================

def apply_drilldown_filters(result: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Interactive filters")
    filtered = result.copy()

    for col in result.columns:
        if col not in ["value", "dt"]:
            unique_vals = sorted(list(filtered[col].dropna().unique()))
            if 1 < len(unique_vals) <= 50:
                selected = st.multiselect(f"Filter by {col}", unique_vals)
                if selected:
                    filtered = filtered[filtered[col].isin(selected)]
    return filtered


# =====================================================================
# VISUALIZATION
# =====================================================================

def visualize(intent: str, result: pd.DataFrame):
    if result.empty:
        st.warning("No data returned for this query.")
        return

    has_value_col = "value" in result.columns

    # Apply drilldown filters first
    result = apply_drilldown_filters(result)

    # FILTER / COUNT: table only
    if intent == "filter" or not has_value_col:
        st.subheader("Table result")
        st.dataframe(result)
        return

    st.subheader("Chart options")
    chart_type = st.selectbox("Choose chart type", ["Bar", "Pie", "Line", "Table only"])

    # category / dt detection
    category_col = "category" if "category" in result.columns else None
    date_col = "dt" if "dt" in result.columns else None

    if chart_type == "Line":
        if date_col not in result.columns:
            st.error("No date column available for a line chart.")
        else:
            result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=result, x=date_col, y="value", marker="o")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

    elif chart_type == "Bar" and category_col:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=result, x=category_col, y="value", ax=ax)
        plt.xticks(rotation=45)
        # bar labels
        for i, v in enumerate(result["value"]):
            ax.text(i, v, str(v), ha="center", va="bottom")
        plt.tight_layout()
        st.pyplot(fig)

    elif chart_type == "Pie" and category_col:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(result["value"], labels=result[category_col], autopct="%1.1f%%")
        st.pyplot(fig)

    # table always
    st.subheader("Table")
    st.dataframe(result)

    # simple insights
    st.subheader("Insights")
    try:
        idx_max = result["value"].idxmax()
        idx_min = result["value"].idxmin()
        if category_col:
            st.write(f"Highest: {result.loc[idx_max, category_col]} = {result.loc[idx_max, 'value']}")
            st.write(f"Lowest: {result.loc[idx_min, category_col]} = {result.loc[idx_min, 'value']}")
        else:
            st.write("Summary statistics:")
            st.write(result["value"].describe())
    except Exception:
        st.write("No insights available.")


# =====================================================================
# MAIN UI
# =====================================================================

query = st.text_input("Ask a question about your data:")

if query:
    intent, sql = build_sql(query)
    st.write(f"Detected intent: {intent}")
    st.code(sql, language="sql")

    try:
        result_df = pd.read_sql_query(sql, conn)
        visualize(intent, result_df)
    except Exception as e:
        st.error(f"SQL execution error: {e}")
