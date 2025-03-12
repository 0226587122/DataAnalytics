import matplotlib.pyplot as plt

# Create a figure object
fig = plt.figure()

# Add an axis to the figure canvas at [0, 0, 1, 1]
ax = fig.add_axes([0, 0, 1, 1])

# Data for plotting
x = [0, 1, 2, 3, 4]
y = [0, 1, 4, 9, 16]

# Plot (x, y) on the axis
ax.plot(x, y)

# Set labels and title
ax.set_title('Title of the Plot')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')

# Display the plot
plt.show()