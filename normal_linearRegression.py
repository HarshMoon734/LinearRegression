import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def linear_regression(data):
    # Extract feature (x) and target (y)
    x = data['pageSpeeds']
    y = data['purchaseAmount']
    
    # Calculate means of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate the terms needed for the numerator and denominator of the slope
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    # Calculate the slope (m) and intercept (b)
    m = numerator / denominator
    b = y_mean - m * x_mean
    
    # Calculate R-squared
    y_pred = m * x + b
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    
    # Plotting the data and the regression line
    plt.scatter(x, y, label='Data')
    plt.plot(x, y_pred, color='red', label='Regression Line')
    plt.xlabel('Page Speeds')
    plt.ylabel('Purchase Amount')
    plt.legend()
    plt.show()
    
    return m, b, r_squared

# Generate synthetic data
np.random.seed(0)
data = pd.DataFrame({
    'pageSpeeds': np.random.normal(3.0, 1.0, 1000),
    'purchaseAmount': 100 - (np.random.normal(3.0, 1.0, 1000) + np.random.normal(0, 0.1, 1000)) * 3
})

# Perform linear regression
slope, intercept, r_squared = linear_regression(data)
print(f'Slope: {slope}')
print(f'Intercept: {intercept}')
print(f'R-squared: {r_squared}')
