import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score, r2_score

# Load the dataset from a file path
file_path = input("Enter the path to your dataset file (e.g., '/path/to/dataset.csv'): ")
dataset = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Overview:")
print(dataset.info())
print("\nFirst 5 rows of the dataset:")
print(dataset.head())

# Handle missing values (if any)
print("\nHandling Missing Values...")
dataset.fillna(method='ffill', inplace=True)  # Forward fill as an example
print("Missing values handled.")

# Exploratory Data Analysis (EDA)
print("\nPerforming Exploratory Data Analysis...")
print("Summary Statistics:")
print(dataset.describe())

# Visualize correlations between numeric features only
plt.figure(figsize=(10, 8))

# Select only numeric columns for correlation
numeric_data = dataset.select_dtypes(include=[np.number])  # Select numeric columns
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Distribution of App Sessions
sns.histplot(dataset['App Sessions'], kde=True)
plt.title("Distribution of App Sessions")
plt.xlabel("App Sessions")
plt.ylabel("Frequency")
plt.show()

# Scatter plot for Distance Travelled vs Calories Burned
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Distance Travelled (km)', y='Calories Burned', data=dataset, hue='Activity Level')
plt.title("Distance Travelled vs Calories Burned")
plt.xlabel("Distance Travelled (km)")
plt.ylabel("Calories Burned")
plt.show()

# Feature Engineering
print("\nCreating new features...")
dataset['Calories_per_Session'] = dataset['Calories Burned'] / dataset['App Sessions']
dataset['Distance_per_Session'] = dataset['Distance Travelled (km)'] / dataset['App Sessions']
print("New features created.")

# Select features and target for regression
features = ['Age', 'App Sessions', 'Distance Travelled (km)', 'Calories Burned', 'Calories_per_Session', 'Distance_per_Session']
target = 'Calories Burned'

X = dataset[features]
y = dataset[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regression Model: Random Forest Regressor
print("\nBuilding Regression Model...")
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Evaluate Regression Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Clustering: KMeans to identify user groups
print("\nPerforming Clustering...")
clustering_features = ['Age', 'App Sessions', 'Distance Travelled (km)', 'Calories Burned']
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(dataset[clustering_features])

# Add cluster labels to the dataset
dataset['Cluster'] = clusters

# Evaluate Clustering with Silhouette Score
silhouette_avg = silhouette_score(dataset[clustering_features], clusters)
print(f"Silhouette Score: {silhouette_avg}")

# Visualize Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Distance Travelled (km)', y='Calories Burned', hue=dataset['Cluster'], palette='viridis')
plt.title("Clustering of Users")
plt.xlabel("Distance Travelled (km)")
plt.ylabel("Calories Burned")
plt.legend(title="Cluster")
plt.show()

# Save the modified dataset with clusters
dataset.to_csv('clustered_dataset.csv', index=False)
print("Clustered dataset saved as 'clustered_dataset.csv'.")

# Implications for Software Engineering Decision-Making
print("\nImplications:")
print("1. Regression analysis helps predict user calorie burn based on app usage.")
print("2. Clustering identifies distinct user groups for targeted feature development.")
print("3. Insights from these analyses can guide personalized app experiences, improve user retention, and prioritize development efforts.")