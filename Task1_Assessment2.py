# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, precision_score,
                             recall_score, confusion_matrix,
                             ConfusionMatrixDisplay)

# Load the dataset
dataset = pd.read_csv('/Users/nikkialonzo/Documents/GitHub/DataAnalytics/DataAnalytics/dataset for assignment 2.csv')

# Exploratory Data Analysis (EDA)
print(dataset.describe(include='all'))

# Visualizing distributions with Histograms
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(dataset['App Sessions'], bins=30, ax=axes[0])
axes[0].set_title('Histogram of App Sessions')

sns.histplot(dataset['Distance Travelled (km)'], bins=30, ax=axes[1])
axes[1].set_title('Histogram of Distance Travelled (km)')

sns.histplot(dataset['Calories Burned'], bins=30, ax=axes[2])
axes[2].set_title('Histogram of Calories Burned')

plt.tight_layout()
plt.show()

# Boxplot for App Sessions
plt.figure(figsize=(6, 5))
sns.boxplot(y=dataset['App Sessions'])
plt.title('Boxplot of App Sessions')
plt.show()

# Prepare data for modeling
X = pd.get_dummies(dataset[['Age', 'Gender', 'Activity Level', 'Location', 'Distance Travelled (km)', 'Calories Burned']],
                   columns=['Gender', 'Activity Level', 'Location'])
y_regression = dataset['App Sessions']

# Regression Model
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = regressor.predict(X_test_reg)

# Regression Model Evaluation
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f'MSE: {mse}, R²: {r2}')

# Feature importance visualization
feature_importances = pd.Series(regressor.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Feature Importances in Regression Model')
plt.show()

# Classification Model (categorizing app sessions)
bins = [0, 90, 150, dataset['App Sessions'].max()]
labels = ['Low', 'Medium', 'High']
dataset['App Usage Category'] = pd.cut(dataset['App Sessions'], bins=bins, labels=labels, include_lowest=True)

y_classification = dataset['App Usage Category']
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_classification, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_cls, y_train_cls)
y_pred_cls = classifier.predict(X_test_cls)

# Classification Model Evaluation
accuracy = accuracy_score(y_test_cls, y_pred_cls)
precision = precision_score(y_test_cls, y_pred_cls, average='weighted')
recall = recall_score(y_test_cls, y_pred_cls, average='weighted')
print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')

# Confusion Matrix
cm = confusion_matrix(y_test_cls, y_pred_cls, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Classification Model)')
plt.show()
