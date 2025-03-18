# Import necessary libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For creating visualizations
import seaborn as sns  # For advanced visualizations
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.ensemble import RandomForestRegressor  # For regression modeling
from sklearn.cluster import KMeans  # For clustering analysis
from sklearn.metrics import mean_squared_error, silhouette_score, r2_score  # For evaluation metrics

# Load the dataset from a file path
file_path = input("Enter the path to your dataset file (e.g., '/path/to/dataset.csv'): ")
dataset = pd.read_csv(file_path)  # Load the dataset into a DataFrame

# Display basic information about the dataset
print("Dataset Overview:")
print(dataset.info())  # Information about columns, data types, and null values
print("\nFirst 5 rows of the dataset:")
print(dataset.head())  # Display the first few rows of the dataset

# Handle missing values (if any)
print("\nHandling Missing Values...")
dataset.fillna(method='ffill', inplace=True)  # Fill missing values using forward fill
print("Missing values handled.")

# Exploratory Data Analysis (EDA)
print("\nPerforming Exploratory Data Analysis...")
print("Summary Statistics:")
print(dataset.describe())  # Summary statistics for numerical columns

# Feature Engineering
print("\nCreating new features...")
dataset['Calories_per_Session'] = dataset['Calories Burned'] / dataset['App Sessions']  # Calories per session
dataset['Distance_per_Session'] = dataset['Distance Travelled (km)'] / dataset['App Sessions']  # Distance per session
print("New features created.")

# Define features for regression and clustering
features = ['Age', 'App Sessions', 'Distance Travelled (km)', 'Calories Burned', 
            'Calories_per_Session', 'Distance_per_Session']  # Features for the model

# Identify Outliers in the Dataset
print("\nIdentifying Outliers...")
Q1 = dataset[features].quantile(0.25)  # First quartile (25th percentile)
Q3 = dataset[features].quantile(0.75)  # Third quartile (75th percentile)
IQR = Q3 - Q1  # Interquartile range
# Identify outliers beyond 1.5 times the IQR
outliers = dataset[((dataset[features] < (Q1 - 1.5 * IQR)) | (dataset[features] > (Q3 + 1.5 * IQR))).any(axis=1)]
print(f"Number of Outliers: {outliers.shape[0]}")
print("Outliers:")
print(outliers)  # Display the outliers

# Save outliers to a CSV file for further analysis
outliers.to_csv('outliers_dataset.csv', index=False)
print("Outliers saved as 'outliers_dataset.csv'.")

# Visualize Outliers with a Box Plot
plt.figure(figsize=(8, 4))  # Set figure size
sns.boxplot(data=dataset[features], palette="Set3")  # Box plot for all features
plt.title("Box Plot to Identify Outliers")  # Title for the plot
plt.xticks(rotation=45)  # Rotate feature names for better readability
plt.show()  # Display the box plot

# Filter Correlation Map Based on Numeric Columns
print("\nFiltering Correlation Map...")
# Select only numeric columns for the correlation map
numeric_columns = dataset.select_dtypes(include=[np.number])  # Filter numeric columns
plt.figure(figsize=(8, 6))  # Set figure size
sns.heatmap(numeric_columns.corr(), annot=True, cmap="coolwarm", fmt=".2f")  # Correlation heatmap
plt.title("Filtered Correlation Heatmap (Numeric Columns Only)")  # Title for the heatmap
plt.show()  # Display the heatmap

# Select features and target for regression
target = 'Calories Burned'  # Target variable

X = dataset[features]  # Feature matrix
y = dataset[target]  # Target vector

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80-20 split

# Regression Model: Random Forest Regressor
print("\nBuilding Regression Model...")
regressor = RandomForestRegressor(random_state=42)  # Initialize the regressor
regressor.fit(X_train, y_train)  # Train the model on the training data
y_pred = regressor.predict(X_test)  # Predict on the test data

# Evaluate Regression Model
mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
r2 = r2_score(y_test, y_pred)  # Calculate R-squared
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Clustering: KMeans to identify user groups
print("\nPerforming Clustering...")
clustering_features = ['Age', 'App Sessions', 'Distance Travelled (km)', 'Calories Burned']  # Features for clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Initialize KMeans with 3 clusters
clusters = kmeans.fit_predict(dataset[clustering_features])  # Fit and predict clusters

# Add cluster labels to the dataset
dataset['Cluster'] = clusters  # Add the cluster label to the dataset

# Evaluate Clustering with Silhouette Score
silhouette_avg = silhouette_score(dataset[clustering_features], clusters)  # Calculate silhouette score
print(f"Silhouette Score: {silhouette_avg}")

# Visualize Clusters with Histogram and Legend
plt.figure(figsize=(8, 6))  # Set figure size
sns.histplot(data=dataset, x='Distance Travelled (km)', hue='Cluster', kde=True, palette='viridis', bins=30)  # Histogram for clusters
plt.title("Distribution of Distance Travelled by Cluster")  # Title for the histogram
plt.xlabel("Distance Travelled (km)")  # X-axis label
plt.ylabel("Frequency")  # Y-axis label
# Add legend with descriptions for each cluster
cluster_labels = ["Low Activity", "Moderate Activity", "High Activity"]  # Example labels
plt.legend(title="Cluster", labels=cluster_labels)  # Add legend with cluster descriptions
plt.show()  # Display the histogram

# Save the modified dataset with clusters
dataset.to_csv('clustered_dataset.csv', index=False)  # Save the dataset to a CSV file
print("Clustered dataset saved as 'clustered_dataset.csv'.")

# Box Plot: Distribution of Calories Burned by Activity Level
plt.figure(figsize=(8, 6))  # Set figure size
sns.boxplot(x='Cluster', y='Calories Burned', data=dataset, palette="Set3")  # Box plot for Calories Burned
plt.title("Calories Burned by Cluster")  # Title for the box plot
plt.xlabel("Cluster")  # X-axis label
plt.ylabel("Calories Burned")  # Y-axis label

# Add a legend to explain the clusters
cluster_labels = ["Low Activity", "Moderate Activity", "High Activity"]  # Example labels
plt.legend(title="Cluster", labels=cluster_labels)  # Add legend with cluster descriptions

plt.show()  # Display the box plot


# Scatter Plot: Actual vs Predicted Calories Burned
plt.figure(figsize=(8, 6))  # Set figure size
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)  # Scatter plot for actual vs predicted
plt.title("Actual vs Predicted Calories Burned")  # Title for the scatter plot
plt.xlabel("Actual Calories Burned")  # X-axis label
plt.ylabel("Predicted Calories Burned")  # Y-axis label
plt.axline([0, 0], [1, 1], color='red', linestyle='--', label='Perfect Prediction')  # Reference line
plt.legend()  # Add legend
plt.show()  # Display the scatter plot

# Regression Model Evaluation: Residual Plot
residuals = y_test - y_pred  # Calculate residuals
plt.figure(figsize=(8, 6))  # Set figure size
sns.histplot(residuals, kde=True, bins=30, color='blue')  # Histogram for residuals
plt.title("Residuals Distribution")  # Title for the histogram
plt.xlabel("Residuals (Actual - Predicted)")  # X-axis label
plt.ylabel("Frequency")  # Y-axis label
plt.axvline(0, color='red', linestyle='--', label='Zero Residual')  # Reference line
plt.legend()  # Add legend
plt.show()  # Display the histogram

# Implications for Software Engineering Decision-Making
print("\nImplications:")
print("1. Outliers identified in the dataset provide insights into extreme user behaviors.")
print("2. The correlation map filtered irrelevant features, improving model interpretability.")
print("3. Regression analysis predicts user calorie burn based on app usage.")
print("4. Clustering identifies distinct user groups for targeted feature development.")
print("5. These insights guide personalized app experiences, improve user retention, and prioritize feature development.")
