# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Datasets
## Load CSV file into a DataFrame
file_path = "/Users/marie/Documents/GitHub/YB-2407-DA/house//House_Data.csv"
df = pd.read_csv(file_path)

# Display the first few rows
print(df.head())

# Visualize before data cleaning
plt.figure(figsize=(10, 5))
sns.boxplot(data=df)
plt.show()

# Function to process total_sqft values
def convert_sqft(value):
    try:
        if '-' in str(value):  # Handling range values like "1200 - 1500"
            vals = value.split('-')
            return (float(vals[0]) + float(vals[1])) / 2  # Convert to average
        return float(value)  # Convert if it's already a number
    except:
        return np.nan  # Return NaN for invalid values

# Handling total_sqft conversion
df['total_sqft'] = df['total_sqft'].apply(convert_sqft)

# Data Cleaning
df['size'] = df['size'].str.extract('(\\d+)').astype(float)  # Extract numeric values from size
df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')  # Convert total_sqft to numeric

# Filling missing numeric values with zero value
df['bath'] = df['bath'].fillna(0)
df['balcony'] = df['balcony'].fillna(0)

# Filling missing categorical values with mode
df['size'] = df['size'].fillna(df['size'].mode()[0]) # Use mode for 'size'

df['site_location'] = df['site_location'].fillna("")

# Drop the 'society' column if it has too many missing values
#df.drop(columns=['society'], inplace=True)  # Optional

# Verify missing values are handled
#print(df.isnull().sum())




# Handling Missing Values
## Checking for missing values
#df.isnull().sum()

## Imputing missing values with the median
#df.fillna(df.median(), inplace=True)
#df.isnull().sum()  # Verifying if missing values are handled

# Detecting Outliers
# Using Boxplot to visually identify outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=df)
plt.show()

# Detecting outliers using the IQR method
#Q1 = df.quantile(0.25)
#Q3 = df.quantile(0.75)
##IQR = Q3 - Q1
#outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
#df[outliers].dropna()


# Summary Statistics
#print(df.describe())



