import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('data/expenses.csv')

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# K-Means clustering on Amount
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Amount']])

# Get centroids and sort to assign labels
centroids = kmeans.cluster_centers_.flatten()
sorted_indices = np.argsort(centroids)
label_map = {sorted_indices[0]: 'Low', sorted_indices[1]: 'Medium', sorted_indices[2]: 'High'}
df['Spending_Level'] = df['Cluster'].map(label_map)

# Group by month and calculate monthly totals
df['Month'] = df['Date'].dt.to_period('M')
monthly_totals = df.groupby('Month')['Amount'].sum().reset_index()
monthly_totals['Month_Num'] = monthly_totals['Month'].dt.month

# For linear regression: predict cumulative expense by day
df = df.sort_values('Date')
df['Day'] = df['Date'].dt.day
df['Cumulative_Amount'] = df.groupby('Month')['Amount'].cumsum()

# Train linear regression on day vs cumulative
X = df[['Day']]
y = df['Cumulative_Amount']
reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)

# Print key insights
print("K-Means Clustering Insights:")
print(f"Cluster Centers: {centroids}")
print(f"Low Spending Range: Amount < {centroids[sorted_indices[0]]:.2f}")
print(f"Medium Spending Range: {centroids[sorted_indices[0]]:.2f} - {centroids[sorted_indices[1]]:.2f}")
print(f"High Spending Range: Amount > {centroids[sorted_indices[1]]:.2f}")
print(f"Number of Low Spending Expenses: {df['Spending_Level'].value_counts()['Low']}")
print(f"Number of Medium Spending Expenses: {df['Spending_Level'].value_counts()['Medium']}")
print(f"Number of High Spending Expenses: {df['Spending_Level'].value_counts()['High']}")

print("\nLinear Regression Insights:")
print(f"Coefficient: {reg.coef_[0]:.2f}")
print(f"Intercept: {reg.intercept_:.2f}")
print(f"R^2 Score: {r2_score(y, y_pred):.2f}")
print(f"Predicted Total Monthly Expense (for day 31): {reg.predict([[31]])[0]:.2f}")
print(f"Actual Total Monthly Expense: {df['Amount'].sum():.2f}")
