import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('House_Data.csv')

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values before imputation:\n", missing_values)

# Impute missing values with the median of numeric columns only
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Verify that missing values have been handled
missing_values_after = df.isnull().sum()
print("Missing values after imputation:\n", missing_values_after)


# Visualize the data to identify outliers using a boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df)
plt.title('Boxplot of Original Data')
plt.show()

# Step 1: Remove duplicate entries
df.drop_duplicates(inplace=True)

# Visualize the data to identify outliers using a boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df)
plt.title('Boxplot of Data after removal of duplicated entries')
plt.show()

# Step 3: Remove outliers using the Interquartile Range (IQR) method
def remove_outliers(dataframe, column):
    """Remove outliers from a specified column using the IQR method."""
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

# Visualize the cleaned data to confirm outlier removal
plt.figure(figsize=(10, 5))
sns.boxplot(data=df)
plt.title('Boxplot of Data after removal of outliers using IQR method')
plt.show()

# Step 4: Apply outlier removal to all numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns
for column in numerical_columns:
    df = remove_outliers(df, column)

# Visualize the cleaned data to confirm outlier removal
plt.figure(figsize=(10, 5))
sns.boxplot(data=df)
plt.title('Boxplot of Cleaned Data')
plt.show()

# Save the cleaned data to a new CSV file
df.to_csv('Cleaned_House_Data.csv', index=False)

print("Data cleaning completed. Cleaned data saved to 'Cleaned_House_Data.csv'")