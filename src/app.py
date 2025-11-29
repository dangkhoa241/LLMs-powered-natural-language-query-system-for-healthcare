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

# DATA LOADING
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file)

for c in df.columns:
    clean = df[c].astype(str).str.replace(r"[,$ ]", "", regex=True)
    if clean.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.7:
        df[c] = pd.to_numeric(clean, errors="coerce")

for c in df.columns:
    if "date" in c.lower():
        df[c] = pd.to_datetime(df[c], errors="coerce")
        df[c] = df[c].dt.strftime("%Y-%m-%d")

conn = sqlite3.connect(":memory:")
df.to_sql("data", conn, index=False, if_exists="replace")

st.write(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
st.dataframe(df.head())

# INTENT MODEL
try:
    clf = pipeline("text-classification", model="intent_model", tokenizer="intent_model")
    st.success("Loaded intent classification model.")
except Exception as e:
    st.error(f"Could not load intent_model. Error: {e}")
    st.stop()

def get_intent(text: str) -> str:
    return clf(text)[0]["label"]


# RULE DICTIONARIES
# mapping from (column, canonical value) -> keywords to detect it
SIMPLE_VALUE_RULES = [
    # Gender
    {"col": "Gender", "val": "female", "kw": ["female", "woman", "women", "lady", "ladies"]},
    {"col": "Gender", "val": "male", "kw": ["male", "man", "men", "gentleman", "gentlemen"]},

    # Test result
    {"col": "Test Result", "val": "positive", "kw": ["tested positive", "test positive", "positive"]},
    {"col": "Test Result", "val": "negative", "kw": ["tested negative", "test negative", "negative"]},

    # Admission type
    {"col": "Admission Type", "val": "emergency", "kw": [" emergency ", " emergency,", " emergency.", " emergency?"]},
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

COLUMN_KEYWORDS = {
    "gender": ["gender", "male", "female"],
    "age": ["age", "older", "younger", "over", "under"],
    "medical condition": ["asthma", "flu", "covid", "hypertension", "diabetes"],
    "admission type": ["emergency", "urgent", "elective"],
    "insurance provider": ["insurance"],
    "billing amount": ["billing", "amount", "payment", "cost", "charge", "expenses"],
    "blood type": ["blood", "o+", "o-", "a+", "a-", "b+", "b-", "ab+", "ab-"],
}

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

# CATEGORICAL FILTERS
def extract_simple_categorical_filters(query: str):
    ql = query.lower()
    filters = {}

    for rule in SIMPLE_VALUE_RULES:
        col, val, kws = rule["col"], rule["val"], rule["kw"]

        if col not in df.columns:
            continue

        for k in kws:
            if re.search(rf"\b{k.lower()}\b", ql):
                filters.setdefault(col, set()).add(val)

    return filters

# NUMERIC FILTERS (AGE, BILLING)
def extract_numeric_filters(query: str):

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

# DATE FILTERS (simple)
def parse_date_range(query: str):
    q = query.lower()
    today = date.today()

    # Specific year detection (e.g., "in 2023")
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        y = int(m.group(1))
        return date(y, 1, 1), date(y, 12, 31)

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

# GROUP BY COLUMN DETECTION (for aggregate / compare)
def detect_group_column(query: str):
    ql = query.lower()

    if any(k in query.lower() for k in ["male", "female", "gender"]):
        if "Gender" in df.columns:
            return "Gender"

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

# WHERE CLAUSE BUILDER
def build_where_clauses(query: str):
    clauses = []

    # 1. Strict categorical filters
    cat_filters = extract_simple_categorical_filters(query)

    for col, vals in cat_filters.items():
        if len(vals) == 1:
            val = list(vals)[0]
            clauses.append(f"LOWER(`{col}`) = LOWER('{val}')")
        else:
            parts = [f"LOWER(`{col}`) = LOWER('{v}')" for v in vals]
            clauses.append("(" + " OR ".join(parts) + ")")

    # 2. Numeric filters (already safe)
    clauses.extend(extract_numeric_filters(query))

    # 3. Date range support
    start, end = parse_date_range(query)
    if start and end:
        date_col = pick_date_column()
        if date_col:
            clauses.append(
                f"date(`{date_col}`) BETWEEN date('{start}') AND date('{end}')"
            )

    return clauses

# SQL BUILDER
def build_sql(query: str):
    intent = get_intent(query)
    ql = query.lower()

    # 1. Build WHERE clause from clean filters only
    where_clauses = build_where_clauses(query)
    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # 2. FILTER → simple return
    if intent == "filter":
        return intent, f"SELECT * FROM data {where_sql} LIMIT 200;"

    # 3. COUNT
    if intent == "count":
        return intent, f"SELECT COUNT(*) AS value FROM data {where_sql};"

    # 4. AGGREGATE / COMPARE
    if intent in ["aggregate", "compare"]:

        group_col = detect_group_column(query)

        # CLEAN METRIC DETECTION (QUESTION-DRIVEN, NOT DATA-DRIVEN)
        # 1. Explicit COUNT intent
        if any(w in ql for w in ["how many", "count", "number of"]):
            agg_func = "COUNT"
            metric_expr = "COUNT(*)"
            metric_label = "Count"

        # 2. Billing / Cost intent → AVG
        elif any(w in ql for w in ["billing", "amount", "cost", "price", "charge"]):
            if "Billing Amount" in df.columns:
                agg_func = "AVG"
                metric_expr = "AVG(`Billing Amount`)"
                metric_label = "Average Billing Amount"
            else:
                agg_func = "COUNT"
                metric_expr = "COUNT(*)"
                metric_label = "Count"

        # 3. Default fallback → COUNT
        else:
            agg_func = "COUNT"
            metric_expr = "COUNT(*)"
            metric_label = "Count"

        # FINAL SQL
        sql = f"""
        SELECT `{group_col}` AS category,
            {metric_expr} AS value
        FROM data
        {where_sql}
        GROUP BY `{group_col}`
        ORDER BY value DESC;
        """

        return intent, sql

    # 5. TREND (time series)
    if intent == "trend":
        date_col = pick_date_column() or df.columns[0]
        ql = query.lower()

        # GROUP BY YEAR
        if "by year" in ql:
            sql = f"""
            SELECT strftime('%Y', `{date_col}`) AS year,
                COUNT(*) AS value
            FROM data
            {where_sql}
            GROUP BY year
            ORDER BY year;
            """

        # GROUP BY MONTH
        elif "by month" in ql:
            sql = f"""
            SELECT strftime('%Y-%m', `{date_col}`) AS month,
                COUNT(*) AS value
            FROM data
            {where_sql}
            GROUP BY month
            ORDER BY month;
            """

        # DEFAULT: GROUP BY DAY
        else:
            sql = f"""
            SELECT `{date_col}` AS dt,
                COUNT(*) AS value
            FROM data
            {where_sql}
            GROUP BY dt
            ORDER BY dt;
            """

        return intent, sql

    # 6. Fallback
    return intent, "SELECT * FROM data LIMIT 100;"

# VISUALIZATION
def visualize(intent: str, result: pd.DataFrame):
    if result.empty:
        st.warning("No data returned.")
        return

    # Show table first
    st.subheader("Table Result")
    st.dataframe(result)

    # BLOCK charts for FILTER
    if intent == "filter":
        st.info("Charts are only generated for aggregated or comparison queries.")
        return

    # Detect proper plotting columns
    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    text_cols = result.select_dtypes(exclude="number").columns.tolist()

    if not numeric_cols or not text_cols:
        st.subheader("Table Result")
        st.dataframe(result)
        return

    y_col = numeric_cols[0]
    x_col = text_cols[0]

    result[x_col] = result[x_col].astype(str)
    result = result.sort_values(by=x_col)

    # STRICT Small Chart Size
    fig_w, fig_h = (4, 2.5)

    # Chart Type Selector
    chart_type = st.selectbox(
        "Chart Type", ["Bar", "Pie", "Line", "Table only"], index=0
    )

    st.subheader("Visualization")

    # Bar Chart
    if chart_type == "Bar":
        fig, ax = plt.subplots(figsize=(fig_w * 0.8, fig_h * 0.8))

        sns.barplot(data=result, x=x_col, y=y_col, ax=ax)


        # Smart bar labels (small, non-overlapping)
        max_val = result[y_col].max()
        offset = max_val * 0.02

        for i, v in enumerate(result[y_col]):
            ax.text(
                i,
                v + offset,
                f"{v:,.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="normal"
            )

        # Semantic label maps
        X_LABEL_MAP = {
            "gender": "Gender",
            "insurance": "Insurance Provider",
            "admission": "Admission Type",
            "blood": "Blood Type",
            "condition": "Medical Condition",
            "hospital": "Hospital",
            "medication": "Medication",
        }

        Y_LABEL_MAP = {
            "billing": "Cost",
            "amount":  "Cost",
            "cost":    "Cost",
            "charge":  "Cost",
            "payment": "Cost",
            "count":   "Count",
            "number":  "Count",
            "total":   "Count",
        }

        query_tokens = set(query.lower().split())

        # X label resolution
        clean_x = (
            next(
                (label for k, label in X_LABEL_MAP.items() if k in query.lower()),
                x_col.replace("_", " ").title()
            )
        )

        # Y label resolution
        clean_y = (
            next(
                (label for k, label in Y_LABEL_MAP.items() if k in query.lower()),
                y_col.replace("_", " ").title()
            )
        )

        ax.set_xlabel(clean_x, fontsize=10)
        ax.set_ylabel(clean_y, fontsize=10)


        # Clean readable title
        ax.set_title(query.capitalize(), fontsize=11, fontweight="semibold", pad=12)

        # Reduce clutter
        plt.xticks(rotation=15, fontsize=4)
        plt.yticks(fontsize=4)
        sns.despine()

        plt.tight_layout()
        st.pyplot(fig)

    # Pie Chart
    elif chart_type == "Pie":
        fig, ax = plt.subplots(figsize=(fig_w * 0.7, fig_h * 0.7))

        ax.pie(
            result[y_col],
            labels=result[x_col],
            autopct="%1.1f%%",
            textprops={"fontsize": 8},
            startangle=90
        )

        ax.set_title(query.capitalize(), fontsize=11, pad=8)
        plt.tight_layout()
        st.pyplot(fig)


    # Line Chart
    elif chart_type == "Line":
        fig, ax = plt.subplots(figsize=(3.2, 1.8))

        sns.lineplot(
            data=result,
            x=x_col,
            y=y_col,
            marker="o",
            linewidth=1.2,
            markersize=4,
            ax=ax
        )

        # Clean labels
        ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=8)
        ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=8)

        # Small title
        ax.set_title(query.capitalize(), fontsize=9, pad=5)

        # Prevent zoom exaggeration
        ymin, ymax = result[y_col].min(), result[y_col].max()
        pad = (ymax - ymin) * 0.6 if ymax != ymin else 1
        ax.set_ylim(ymin - pad, ymax + pad)

        plt.xticks(rotation=18, fontsize=7)
        plt.yticks(fontsize=7)
        sns.despine()

        plt.tight_layout(pad=0.3)

        st.pyplot(fig, use_container_width=False)

    # Insights
    st.subheader("Insights")
    try:
        idx_max = result[y_col].idxmax()
        idx_min = result[y_col].idxmin()

        st.write(
            f"Highest {y_col}: "
            f"{result.loc[idx_max, x_col]} → {result.loc[idx_max, y_col]}"
        )
        st.write(
            f"Lowest {y_col}: "
            f"{result.loc[idx_min, x_col]} → {result.loc[idx_min, y_col]}"
        )
    except Exception:
        st.write("No insights available.")

# MAIN UI
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
