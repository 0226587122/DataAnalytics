import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.random.rand(50)  # Random values for x-axis
y = np.random.rand(50)  # Random values for y-axis

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', marker='o')

# Adding titles and labels
plt.title('Simple Scatter Plot')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')

# Show plot
plt.grid(True)
plt.show()
