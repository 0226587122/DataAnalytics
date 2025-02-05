import matplotlib.pyplot as plt
categories = ['N', 'H', 'G', 'A']
values = [3, 7, 1, 8]
plt.bar(categories, values, color='blue')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('BarChart')
plt.show()

