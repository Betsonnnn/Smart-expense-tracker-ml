from datetime import timedelta, date
from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import uuid

st.set_page_config(page_title="Smart Expense Tracker", layout="wide")

col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    st.write("")

with col2:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0;'>💸 Smart Expense Tracker</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; margin-top: 0; color: gray;'>Analyze your expenses with filters, charts, and ML insights</p>",
        unsafe_allow_html=True
    )

with col3:
    add_clicked = st.button("➕ Add Expense", use_container_width=True)


# Load Data
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "expenses.csv"

if not DATA_PATH.exists():
    st.error("❌ data/expenses.csv not found. Please add it first.")
    st.stop()

df = pd.read_csv(DATA_PATH)
if "ID" not in df.columns:
    df["ID"] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
    df.to_csv(DATA_PATH, index=False)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])


df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])


if "open_add_modal" not in st.session_state:
    st.session_state.open_add_modal = False


def save_expense(date_value, category_value, amount_value):
    new_row = pd.DataFrame([{
        "ID": str(uuid.uuid4())[:8],
        "Date": date_value.strftime("%Y-%m-%d"),
        "Category": category_value,
        "Amount": int(amount_value),
    }])


def delete_expense(expense_id):
    df_all = pd.read_csv(DATA_PATH)

    # make sure both are string
    df_all["ID"] = df_all["ID"].astype(str)
    expense_id = str(expense_id)

    # keep only rows that are NOT the selected ID
    df_all = df_all[df_all["ID"] != expense_id]

    df_all.to_csv(DATA_PATH, index=False)


if add_clicked:
    st.session_state.open_add_modal = True

if st.session_state.open_add_modal:

    @st.dialog("Add New Expense")
    def add_expense_dialog():
        new_date = st.date_input("Date")
        new_category = st.selectbox(
            "Category",
            ["Food", "Transport", "Entertainment",
                "Bills", "Shopping", "Health", "Others"]
        )
        new_amount = st.number_input("Amount (₹)", min_value=1, step=10)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Save"):
                save_expense(new_date, new_category, new_amount)
                st.cache_data.clear()
                st.success("✅ Expense added!")
                st.session_state.open_add_modal = False
                st.rerun()

        with col2:
            if st.button("Cancel"):
                st.session_state.open_add_modal = False
                st.rerun()

    add_expense_dialog()


# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🔍 Filters")
currency_symbol = "₹"

# Ensure date column is valid
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

# Data date limits
min_date_in_data = df["Date"].min().date()
max_date_in_data = df["Date"].max().date()

# Default: last 30 days (or whatever exists)
default_end = max_date_in_data
default_start = max(min_date_in_data, default_end - timedelta(days=30))

# Allow navigation beyond dataset (IMPORTANT)
min_value = min_date_in_data - timedelta(days=365)
max_value = max_date_in_data + timedelta(days=365)

date_range = st.sidebar.date_input(
    "Select date range",
    value=(default_start, default_end),
    min_value=min_value,
    max_value=max_value
)

# Normalize if user selects only one date
if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    date_range = (default_start, default_end)

start_date, end_date = date_range


if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

categories = ["All"] + sorted(df["Category"].unique().tolist())
selected_category = st.sidebar.selectbox("Category", categories)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

if selected_category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]

st.sidebar.header("💰 Budget")

monthly_budget = st.sidebar.number_input(
    "Set Monthly Budget (₹)",
    min_value=0,
    value=10000,
    step=500
)


# ----------------------------
# KPIs
# ----------------------------
total_expense = filtered_df["Amount"].sum()
avg_expense = filtered_df["Amount"].mean()

cat_totals = filtered_df.groupby(
    "Category")["Amount"].sum().sort_values(ascending=False)
top_category = cat_totals.index[0] if len(cat_totals) > 0 else "N/A"

col1, col2, col3 = st.columns(3)
col1.metric("Total Expense", f"{currency_symbol} {total_expense:,.0f}")
col2.metric("Average Expense", f"{currency_symbol} {avg_expense:,.0f}")
col3.metric("Top Category", top_category)

st.divider()
st.subheader("📌 Budget Status")

current_month = pd.Timestamp.today().to_period("M")
df["Month"] = df["Date"].dt.to_period("M")

month_df = df[df["Month"] == current_month]
month_total = month_df["Amount"].sum()

if monthly_budget > 0:
    progress = min(month_total / monthly_budget, 1.0)
    st.progress(progress)

    st.write(
        f"Spent this month: ₹ {month_total:,.0f} / ₹ {monthly_budget:,.0f}")

    if month_total >= monthly_budget:
        st.error("❌ You exceeded your monthly budget!")
    elif month_total >= 0.8 * monthly_budget:
        st.warning("⚠️ You have used 80% of your budget.")
    else:
        st.success("✅ Budget is under control.")
else:
    st.info("Set a monthly budget in the sidebar.")

# ----------------------------
# Data Preview
# ----------------------------
st.subheader("Data Preview")
st.dataframe(filtered_df.head(25), use_container_width=True)
st.subheader("🗑 Delete an Expense")

if len(filtered_df) > 0:
    delete_id = st.selectbox(
        "Select Expense ID to delete",
        filtered_df.head(25)["ID"].tolist()
    )

    if st.button("Delete Selected Expense"):
        delete_expense(delete_id)
        st.success(f"Deleted expense ID: {delete_id}")
        st.rerun()
else:
    st.info("No expenses available to delete.")

st.divider()

# ----------------------------
# Charts
# ----------------------------
st.subheader("Charts")

colA, colB = st.columns(2)

# Category bar chart
with colA:
    st.write("### Category-wise Expenses")

    cat_totals = (
        filtered_df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(cat_totals)


# Daily trend line chart
with colB:
    st.write("### Daily Expense Trend")

    daily_totals = filtered_df.groupby(
        filtered_df["Date"].dt.date)["Amount"].sum()
    daily_totals = daily_totals.sort_index()

    st.line_chart(daily_totals)

st.divider()
st.subheader(" Next Month Expense Prediction")

df_pred = df.copy()
df_pred["Month"] = df_pred["Date"].dt.to_period("M").astype(str)

monthly = df_pred.groupby("Month")["Amount"].sum().reset_index()
monthly = monthly.sort_values("Month")

if len(monthly) >= 2:
    monthly["MonthIndex"] = np.arange(len(monthly))

    X = monthly[["MonthIndex"]]
    y = monthly["Amount"]

    model = LinearRegression()
    model.fit(X, y)

    next_month_df = pd.DataFrame({"MonthIndex": [int(len(monthly))]})
    prediction = model.predict(next_month_df)[0]

    st.success(f"Predicted next month expense: ₹ {prediction:,.0f}")
    st.write("📌 Monthly totals used for prediction:")
    st.line_chart(monthly.set_index("Month")["Amount"])
else:
    st.warning(
        "Not enough monthly data. Add at least 2 months of expenses for prediction.")

st.divider()
st.subheader("Overspending / Anomaly Detection")

if len(filtered_df) >= 10:
    iso = IsolationForest(contamination=0.05, random_state=42)
    filtered_df.loc[:, "Anomaly"] = iso.fit_predict(filtered_df[["Amount"]])

    anomalies = filtered_df[filtered_df["Anomaly"]
                            == -1].sort_values("Amount", ascending=False)

    if not anomalies.empty:
        st.error("⚠️ Unusual high expenses detected!")

        st.dataframe(anomalies[["Date", "Category", "Amount"]].head(
            15), use_container_width=True)
    else:
        st.success("✅ No unusual spending detected in the selected range.")

else:
    st.info("Add more expenses (at least 10) to enable anomaly detection.")
