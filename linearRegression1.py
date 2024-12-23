import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit app title
st.title("Linear Regression with Streamlit")

# Description of the app
st.write("""
This app allows you to input a dataset with independent and dependent variables. 
It will perform a linear regression and display the results, including a regression line.
""")

# Sidebar for user input
st.sidebar.header("Input Data")

# Number of data points the user wants to input
num_points = st.sidebar.slider("Select number of data points", min_value=10, max_value=100, value=30)

# Initialize empty lists for data
x_data = []
y_data = []

# Collect user input for each data point
for i in range(num_points):
    x = st.sidebar.number_input(f"Enter X value for data point {i + 1}", key=f"x_{i}", step=0.1)
    y = st.sidebar.number_input(f"Enter Y value for data point {i + 1}", key=f"y_{i}", step=0.1)
    x_data.append(x)
    y_data.append(y)

# Convert the lists into numpy arrays for use with scikit-learn
X = np.array(x_data).reshape(-1, 1)  # Reshape X to be a 2D array
Y = np.array(y_data)

# Split data into training and test sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, Y_train)

# Make predictions using the test data
y_pred = model.predict(X_test)

# Show model coefficients and performance metrics
st.subheader("Model Evaluation:")
st.write(f"**Intercept**: {model.intercept_}")
st.write(f"**Coefficient (Slope)**: {model.coef_[0]}")

# Calculate Mean Squared Error (MSE) and R² score
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

st.write(f"**Mean Squared Error (MSE)**: {mse}")
st.write(f"**R² Score**: {r2}")

# Plot the regression line along with the original data points
st.subheader("Regression Line Plot")

# Scatter plot of the original data
plt.scatter(X, Y, color="blue", label="Data Points")

# Plot the regression line
plt.plot(X, model.predict(X), color="red", label="Regression Line")

# Add labels and legend
plt.xlabel("X (Independent Variable)")
plt.ylabel("Y (Dependent Variable)")
plt.legend()

# Display the plot
st.pyplot(plt)
