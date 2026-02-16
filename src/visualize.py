import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/plots", exist_ok=True)

def visualize_data():

    # Load data/expenses.csv using pandas
    df = pd.read_csv('data/expenses.csv')

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Create a bar chart of total expense by Category
    category_totals = df.groupby('Category')['Amount'].sum()
    plt.figure()
    plt.bar(category_totals.index, category_totals.values)
    plt.title('Total Expense by Category')
    plt.xlabel('Category')
    plt.ylabel('Total Amount')
    plt.savefig('results/plots/category_expense.png')
    plt.close()

    # Create a line chart showing daily expense trend
    daily_totals = df.groupby('Date')['Amount'].sum()
    plt.figure()
    plt.plot(daily_totals.index, daily_totals.values)
    plt.title('Daily Expense Trend')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.savefig('results/plots/daily_trend.png')
    plt.close()


if __name__ == "__main__":
    visualize_data()
