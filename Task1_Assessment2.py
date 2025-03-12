import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, 
                             classification_report, silhouette_score)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Constants
DATASET_URL = "D:\Programs\VSCode-GitHub\DataAnalytics\dataset for assignment 2.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data(url):
    """Load the dataset from the given URL."""
    try:
        df = pd.read_csv(url)
        print("Dataset loaded successfully.")
        print("Columns in the dataset:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

def preprocess_data(df):
    """Preprocess the data: clean column names, encode categorical variables, and handle missing values."""
    # Clean up column names
    df.columns = df.columns.str.strip()

    # Convert categorical variables into numeric
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["Activity Level"] = df["Activity Level"].map({"Sedentary": 0, "Moderate": 1, "Active": 2})
    df["Location"] = df["Location"].map({"Urban": 0, "Suburban": 1, "Rural": 2})

    # Check for missing values and handle them
    if df.isnull().sum().any():
        df.fillna(df.mean(), inplace=True)  # Fill missing values with mean
        print("Missing values found and filled with mean.")

    return df

def perform_regression_analysis(X, y):
    """Perform regression analysis and print results."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    y_pred = reg_model.predict(X_test)

    print("\nRegression Analysis:")
    print("Coefficients:", reg_model.coef_)
    print("Intercept:", reg_model.intercept_)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R-squared:", r2_score(y_test, y_pred))

def perform_clustering(X, df):
    """Perform clustering analysis and visualize clusters."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    print("\nClustering Analysis:")
    print("Cluster Centers:\n", kmeans.cluster_centers_)
    print("Silhouette Score:", silhouette_score(X_scaled, kmeans.labels_))

    # Visualize Clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df["Cluster"], cmap="viridis", marker='o')
    plt.title("User Clusters")
    plt.xlabel("Feature 1: Gender (0: Male, 1: Female)")
    plt.ylabel("Feature 2: Age (Standardized)")
    plt.colorbar(label='Cluster')
    plt.show()

def perform_predictive_modeling(X, y):
    """Perform predictive modeling and print results."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\nPredictive Modeling:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def main():
    # Step 1: Load the dataset
    df = load_data(DATASET_URL)

    # Step 2: Preprocessing
    df = preprocess_data(df)

    # Step 3: Regression Analysis
    X_reg = df[["Gender", "Age", "Activity Level", "Location", "Distance Travelled (km)", "Calories Burned"]]  # Corrected column name
    y_reg = df["App Sessions"]
    perform_regression_analysis(X_reg, y_reg)

    # Step 4: Clustering
    perform_clustering(X_reg, df)  # Pass df as an argument

    # Step 5: Predictive Modeling
    df["High Engagement"] = (df["App Sessions"] > 100).astype(int)
    X_clf = df[["Gender", "Age", "Activity Level", "Location", "Distance Travelled (km)", "Calories Burned"]]  # Corrected column name
    y_clf = df["High Engagement"]
    perform_predictive_modeling(X_clf, y_clf)

if __name__ == "__main__":
    main()