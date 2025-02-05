import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = np.random.randn(1000)  # Generate 1000 random numbers from a normal distribution

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='skyblue', edgecolor='black')

# Adding titles and labels
plt.title('Simple Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show plot
plt.grid(axis='y')
plt.show()
