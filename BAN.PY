import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Simulate some network data
np.random.seed(0)  # For reproducibility
num_samples = 100

# Features
bandwidth = np.random.normal(100, 20, num_samples)  # Bandwidth in Mbps
packet_loss = np.random.uniform(0, 0.1, num_samples)  # Packet loss percentage
jitter = np.random.normal(5, 1, num_samples)  # Jitter in ms
congestion_level = np.random.uniform(0, 1, num_samples)  # Network congestion (0 to 1)
distance_to_server = np.random.normal(50, 15, num_samples)  # Distance in km

# Latency calculation including new features
latency = (
    100 - bandwidth * 0.5 +
    (packet_loss * 200) +
    (jitter * 10) +
    (congestion_level * 50) +
    (distance_to_server * 0.5) +
    np.random.normal(0, 10, num_samples)  # Random noise
)

# Create a DataFrame
data = pd.DataFrame({
    'Bandwidth': bandwidth,
    'PacketLoss': packet_loss,
    'Jitter': jitter,
    'CongestionLevel': congestion_level,
    'DistanceToServer': distance_to_server,
    'Latency': latency
})

# Split data into features and target
X = data[['Bandwidth', 'PacketLoss', 'Jitter', 'CongestionLevel', 'DistanceToServer']]
y = data['Latency']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions)
plt.xlabel('Actual Latency (ms)')
plt.ylabel('Predicted Latency (ms)')
plt.title('Actual vs Predicted Latency')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line
plt.xlim([0, max(y_test.max(), predictions.max())])
plt.ylim([0, max(y_test.max(), predictions.max())])
plt.grid()
plt.show()

# Print model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
