import matplotlib.pyplot as plt

# Create a figure object
fig = plt.figure()

# Add the first axis to the figure canvas at [0, 0, 1, 1]
ax1 = fig.add_axes([0, 0, 1, 1])

# Add the second axis to the figure canvas at [0.2, 0.5, 0.2, 0.2]
ax2 = fig.add_axes([0.2, 0.5, 0.2, 0.2])

# Display the figure with the two axes
plt.show()