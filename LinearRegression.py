import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 2: Generate synthetic data
np.random.seed(0)  # for reproducibility
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 0.1, 1000)) * 3

# Reshape data for sklearn
pageSpeeds = pageSpeeds.reshape(-1, 1)
purchaseAmount = purchaseAmount.reshape(-1, 1)

# Step 3: Perform linear regression using scikit-learn
model = LinearRegression()
model.fit(pageSpeeds, purchaseAmount)

# Get the slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

# Step 4: Calculate R-squared
predicted = model.predict(pageSpeeds)
r_squared = r2_score(purchaseAmount, predicted)

# Step 5: Plot the original data and the regression line
plt.scatter(pageSpeeds, purchaseAmount, label='Data')
plt.plot(pageSpeeds, predicted, color='red', label='Regression Line')
plt.xlabel('Page Speeds')
plt.ylabel('Purchase Amount')
plt.legend()
plt.show()

print(f'Slope: {slope[0]}')
print(f'Intercept: {intercept[0]}')
print(f'R-squared: {r_squared}')
