import pandas as pd
import random
from datetime import datetime, timedelta

# Categories and payment modes
categories = ["Food", "Transport", "Entertainment", "Bills", "Shopping", "Health", "Others"]
payment_modes = ["Cash", "Card", "UPI"]

data = []

start_date = datetime(2025, 1, 1)

for i in range(120):
    date = start_date + timedelta(days=i // 4)
    category = random.choice(categories)
    amount = random.randint(50, 1500)
    payment = random.choice(payment_modes)

    data.append([date.strftime("%Y-%m-%d"), category, amount, payment])

df = pd.DataFrame(data, columns=["Date", "Category", "Amount", "Payment_Mode"])

# Save CSV inside data folder
df.to_csv("data/expenses.csv", index=False)

print("expenses.csv created successfully inside data folder")
