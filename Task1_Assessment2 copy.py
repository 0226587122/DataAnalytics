# fitness_app_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.preprocessing import StandardScaler
import joblib
from matplotlib.backends.backend_pdf import PdfPages

# Step 1: Load the dataset
def load_data(url):
    print("Loading dataset...")
    df = pd.read_csv(url)
    print("Dataset loaded successfully.")
    return df

# Step 2: Explore the dataset
def explore_data(df):
    print("\nDataset Overview:")
    print(df.head())

    print("\nDataset Information:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

# Step 3: Preprocess the data
def preprocess_data(df):
    print("\nHandling missing values...")
    df.fillna(df.mean(), inplace=True)

    print("Converting categorical variables to numerical...")
    df = pd.get_dummies(df, drop_first=True)

    print("Normalizing/Standardizing the data...")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df, df_scaled, scaler

# Step 4: Feature selection and engineering
def select_features(df, target):
    print("\nSelecting relevant features...")
    corr_matrix = df.corr()
    return corr_matrix, corr_matrix[target].sort_values(ascending=False).index[1:6]

# Step 5: Build and evaluate regression model
def build_regression_model(X_train, X_test, y_train, y_test):
    print("\nBuilding Linear Regression Model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    return lr_model, mse, y_test, y_pred

# Step 6: Build and evaluate clustering model
def build_clustering_model(df_scaled, n_clusters=3):
    print("\nBuilding KMeans Clustering Model...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)

    silhouette_avg = silhouette_score(df_scaled, clusters)
    print(f"Silhouette Score: {silhouette_avg}")
    return kmeans, silhouette_avg, clusters

# Step 7: Save models
def save_models(lr_model, kmeans):
    print("\nSaving models...")
    joblib.dump(lr_model, 'linear_regression_model.pkl')
    joblib.dump(kmeans, 'kmeans_clustering_model.pkl')
    print("Models saved successfully.")

# Step 8: Create a PDF with all outputs
def create_pdf_report(df, corr_matrix, features, mse, silhouette_avg, y_test, y_pred, clusters):
    print("\nCreating PDF report...")
    with PdfPages('fitness_app_analysis_report.pdf') as pdf:
        # Page 1: Dataset Overview
        plt.figure(figsize=(10, 6))
        plt.text(0.1, 0.5, str(df.head()), fontsize=10, va='center')
        plt.title("Dataset Overview")
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # Page 2: Correlation Matrix Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        pdf.savefig()
        plt.close()

        # Page 3: Regression Results
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Regression Results (MSE: {mse:.2f})")
        pdf.savefig()
        plt.close()

        # Page 4: Clustering Visualization
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df_scaled[:, 0], y=df_scaled[:, 1], hue=clusters, palette='viridis')
        plt.title(f"Clustering Results (Silhouette Score: {silhouette_avg:.2f})")
        pdf.savefig()
        plt.close()

        # Page 5: Insights
        plt.figure(figsize=(10, 6))
        insights = (
            f"Insights:\n"
            f"1. Regression Model MSE: {mse:.2f}\n"
            f"2. Clustering Silhouette Score: {silhouette_avg:.2f}\n"
            f"3. Selected Features: {features}"
        )
        plt.text(0.1, 0.5, insights, fontsize=12, va='center')
        plt.title("Final Insights")
        plt.axis('off')
        pdf.savefig()
        plt.close()

    print("PDF report saved as 'fitness_app_analysis_report.pdf'.")

# Step 9: Main function
def main():
    # Dataset URL (replace with actual URL)
    url = "https://statics.teams.cdn.office.net/evergreen-assets/safelinks/1/atp-safelinks.html"

    # Load data
    df = load_data(url)

    # Explore data
    explore_data(df)

    # Preprocess data
    df, df_scaled, scaler = preprocess_data(df)

    # Select features (assuming 'user_engagement' is the target variable)
    target = 'user_engagement'
    corr_matrix, features = select_features(df, target)

    # Prepare data for regression
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and evaluate regression model
    lr_model, mse, y_test, y_pred = build_regression_model(X_train, X_test, y_train, y_test)

    # Build and evaluate clustering model
    kmeans, silhouette_avg, clusters = build_clustering_model(df_scaled)

    # Save models
    save_models(lr_model, kmeans)

    # Create PDF report
    create_pdf_report(df, corr_matrix, features, mse, silhouette_avg, y_test, y_pred, clusters)

# Run the script
if __name__ == "__main__":
    main()