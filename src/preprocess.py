import pandas as pd


def preprocess_data():
    # Load the CSV file
    df = pd.read_csv('data/expenses.csv')

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Add Month and Day columns
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # Remove missing values
    df.dropna(inplace=True)

    # Print dataset info
    print(df.info())

    # Print total expense amount
    total_expense = df['Amount'].sum()
    print(f"\nTotal Expense Amount: {total_expense}")

    # Print category-wise total expenses
    category_totals = df.groupby('Category')['Amount'].sum()
    print("\nCategory-wise Total Expenses:")
    print(category_totals)

    # Print the highest spending category
    highest_category = category_totals.idxmax()
    print(f"\nHighest Spending Category: {highest_category}")

    return df


if __name__ == "__main__":
    preprocess_data()
