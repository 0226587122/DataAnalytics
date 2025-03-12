import matplotlib.pyplot as plt

# Correct data
study_hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Study hours of 10 students
test_scores = [50, 55, 65, 70, 72, 78, 85, 88, 92, 95]  # Corresponding test scores

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(study_hours, test_scores, color='blue', marker='o')

# Adding titles and labels
plt.title('Scatter Plot of Study Hours vs Test Scores')
plt.xlabel('Study Hours')
plt.ylabel('Test Scores')

# Show plot
plt.grid(True)
plt.show()
