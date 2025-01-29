import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('House_Data.csv')

# 1. Handle Missing Values
# Fill missing values with the mean for numerical columns and mode for categorical columns
data.fillna({
    'price': data['price'].mean(),  # Example for numerical
    'society': data['society'].mode()[0]  # Example for categorical
}, inplace=True)

# 2. Detecting Outliers
# Using IQR method to detect outliers
Q1 = data['price'].quantile(0.25)
Q3 = data['price'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 3. Delete Duplicate Data
data.drop_duplicates(inplace=True)

# 4. Delete Null
data.dropna(inplace=True)

# 5. Delete Outliers
data = data[(data['price'] >= lower_bound) & (data['price'] <= upper_bound)]

# Save the cleaned data
data.to_csv('Cleaned_House_Data.csv', index=False)

print("Data cleaning completed. Cleaned data saved as 'Cleaned_House_Data.csv'.")