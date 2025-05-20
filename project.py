# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random

# Simulated historical sales data
data = {
    'month': list(range(1, 13)),
    'sales': [random.randint(80, 120) for _ in range(12)]
}
df = pd.DataFrame(data)

# Prepare data for ML model
X = df[['month']]
y = df['sales']
model = LinearRegression()
model.fit(X, y)

# Predict next 3 months
future_months = pd.DataFrame({'month': [13, 14, 15]})
predicted_sales = model.predict(future_months)

# Output predictions
for month, sale in zip(future_months['month'], predicted_sales):
    print(f"Predicted sales for month {month}: {int(sale)} units")

# Plotting the data
plt.plot(df['month'], df['sales'], label='Historical Sales')
plt.plot(future_months['month'], predicted_sales, label='Predicted Sales', linestyle='--')
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Forecasting")
plt.legend()
plt.show()
