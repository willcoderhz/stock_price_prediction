import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate some sample training data
X_train = np.array([[i] for i in range(1, 101)])  # Days from 1 to 100
y_train = np.array([i * 2 for i in range(1, 101)])  # Assume stock price is twice the day number

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the model to a .pkl file
with open('ml_models/stock_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("The model has been successfully saved to stock_price_model.pkl")
