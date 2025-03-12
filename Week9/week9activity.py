import matplotlib.pyplot as plt

# Data
labels = ['Apples', 'Bananas', 'Grapes', 'Oranges']
sizes = [30, 25, 20, 25]  # Percentages
colors = ['red', 'yellow', 'purple', 'orange']
explode = (0, 0, 0, 0.1)  # Highlight Apples

# Create pie chart
plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, shadow=True, startangle=140)
plt.title("Favorite Fruits of 100 Students\n")
plt.show()