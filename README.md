# 💸 Smart Expense Tracker with ML Insights

A Streamlit-based expense tracker that allows users to record expenses, filter by date/category, visualize spending trends, and generate ML-based insights like next-month expense prediction and anomaly detection.

---

## 🚀 Features
- Add expenses using a modal dialog
- Date range filter (fully functional)
- Category-wise bar chart
- Daily trend line chart
- Monthly budget progress + alerts
- ML Prediction: Next month expense forecast (Linear Regression)
- ML Detection: Overspending/anomaly detection (Isolation Forest)

---

## 🧠 ML Logic Used
### 1) Expense Forecasting
- Monthly totals are extracted from the dataset
- Linear Regression is trained on month index vs total spending
- Next month total expense is predicted

### 2) Overspending Detection
- Isolation Forest detects unusual high expenses
- Flagged transactions are displayed in a table

---

## 🛠 Tech Stack
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn

---

## 📂 Project Structure
