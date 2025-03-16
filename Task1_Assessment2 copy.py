# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For advanced visualizations
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  # For regression and classification models
from sklearn.metrics import (mean_squared_error, r2_score,  # For regression evaluation
                             accuracy_score, precision_score,  # For classification evaluation
                             recall_score, confusion_matrix, 
                             ConfusionMatrixDisplay)

# Function to load and validate the dataset
def load_dataset(file_path):
    try:
        dataset = pd.read_csv(file_path)  # Load the dataset
        print("Dataset loaded successfully.")
        print(f"Dataset shape: {dataset.shape}")
        return dataset
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
        return None

# Load the dataset
dataset_path = '/Users/nikkialonzo/Documents/GitHub/DataAnalytics/DataAnalytics/dataset for assignment 2.csv'
dataset = load_dataset(dataset_path)

# Check for missing values
if dataset is not None:
    print("Checking for missing values...")
    print(dataset.isnull().sum())  # Display count of missing values per column

# Exploratory Data Analysis (EDA)
if dataset is not None:
    print("Dataset Summary:")
    print(dataset.describe(include='all'))  # Display summary statistics for all columns

    # Visualizing distributions with Histograms
    print("Creating histograms...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(dataset['App Sessions'], bins=30, ax=axes[0])
    axes[0].set_title('Histogram of App Sessions')
    axes[0].set_xlabel('App Sessions')

    sns.histplot(dataset['Distance Travelled (km)'], bins=30, ax=axes[1])
    axes[1].set_title('Histogram of Distance Travelled (km)')
    axes[1].set_xlabel('Distance Travelled (km)')

    sns.histplot(dataset['Calories Burned'], bins=30, ax=axes[2])
    axes[2].set_title('Histogram of Calories Burned')
    axes[2].set_xlabel('Calories Burned')

    plt.tight_layout()
    plt.show()

    # Boxplot for App Sessions
    print("Creating boxplot for App Sessions...")
    plt.figure(figsize=(6, 5))
    sns.boxplot(y=dataset['App Sessions'])  # Boxplot for detecting outliers
    plt.title('Boxplot of App Sessions')
    plt.ylabel('App Sessions')
    plt.show()

# Prepare data for modeling
if dataset is not None:
    print("Preparing data for modeling...")
    # Convert categorical variables into dummy/indicator variables
    X = pd.get_dummies(dataset[['Age', 'Gender', 'Activity Level', 'Location', 'Distance Travelled (km)', 'Calories Burned']],
                       columns=['Gender', 'Activity Level', 'Location'])
    y_regression = dataset['App Sessions']  # Target variable for regression

    # Split data into training and testing sets
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

    # Train a Regression Model
    print("Training regression model...")
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train_reg, y_train_reg)  # Fit the model to training data
    y_pred_reg = regressor.predict(X_test_reg)  # Predict on test data

    # Evaluate Regression Model
    mse = mean_squared_error(y_test_reg, y_pred_reg)  # Mean Squared Error
    r2 = r2_score(y_test_reg, y_pred_reg)  # R-squared score
    print(f'Regression Model Evaluation:\nMSE: {mse:.2f}, R²: {r2:.2f}')

    # Visualize Feature Importances
    print("Visualizing feature importances...")
    feature_importances = pd.Series(regressor.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('Feature Importances in Regression Model')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()

    # Classification Model (categorizing app sessions)
    print("Preparing data for classification model...")
    bins = [0, 90, 150, dataset['App Sessions'].max()]  # Define bins for categorizing app sessions
    labels = ['Low', 'Medium', 'High']  # Define labels for each bin
    dataset['App Usage Category'] = pd.cut(dataset['App Sessions'], bins=bins, labels=labels, include_lowest=True)

    y_classification = dataset['App Usage Category']  # Target variable for classification
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_classification, test_size=0.2, random_state=42)

    # Train a Classification Model
    print("Training classification model...")
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train_cls, y_train_cls)  # Fit the model to training data
    y_pred_cls = classifier.predict(X_test_cls)  # Predict on test data

    # Evaluate Classification Model
    accuracy = accuracy_score(y_test_cls, y_pred_cls)  # Accuracy score
    precision = precision_score(y_test_cls, y_pred_cls, average='weighted')  # Precision score
    recall = recall_score(y_test_cls, y_pred_cls, average='weighted')  # Recall score
    print(f'Classification Model Evaluation:\nAccuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')

    # Confusion Matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test_cls, y_pred_cls, labels=labels)  # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix (Classification Model)')
    plt.show()
