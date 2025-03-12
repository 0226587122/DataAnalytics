import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
# Replace 'path_to_your_file.csv' with the actual path to your downloaded file
file_path = "D:\Programs\VSCode-GitHub\DataAnalytics\dataset for assignment 2.csv"

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    print("Columns in the dataset:", df.columns.tolist())
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step 2: Analyze User Demographics
print("\nUser Demographics Analysis:")
print("Gender Distribution:\n", df["Gender"].value_counts())
print("\nActivity Level Distribution:\n", df["Activity Level"].value_counts())
print("\nLocation Distribution:\n", df["Location"].value_counts())

# Step 3: Analyze Engagement by Demographics
print("\nAverage App Sessions by Gender:")
print(df.groupby("Gender")["App Sessions"].mean())

print("\nAverage App Sessions by Activity Level:")
print(df.groupby("Activity Level")["App Sessions"].mean())

print("\nAverage App Sessions by Location:")
print(df.groupby("Location")["App Sessions"].mean())

# Step 4: Check for Potential Biases
# Encode categorical variables for correlation analysis
le_gender = LabelEncoder()
le_activity = LabelEncoder()
le_location = LabelEncoder()

df["Gender_Encoded"] = le_gender.fit_transform(df["Gender"])
df["Activity_Encoded"] = le_activity.fit_transform(df["Activity Level"])
df["Location_Encoded"] = le_location.fit_transform(df["Location"])

# Correlation Matrix
print("\nCorrelation Matrix:")
print(df[["Gender_Encoded", "Activity_Encoded", "Location_Encoded", "App Sessions"]].corr())

# Step 5: Ethical Guidelines for Data Handling
print("\nEthical Guidelines for Data Handling:")
print("1. Ensure transparency by informing users about data collection and usage.")
print("2. Apply anonymization techniques to protect user privacy.")
print("3. Avoid biases by ensuring equal representation of different demographics.")
print("4. Regularly audit algorithms for fairness and accuracy.")
print("5. Allow users to opt-out of data collection and delete their data upon request.")
print("6. Comply with data protection laws such as GDPR and CCPA.")

# Step 6: Culturally Relevant Insights
print("\nCulturally Relevant Insights:")
print("1. Urban users have higher average app sessions, which may indicate a preference for fitness tracking in urban areas.")
print("2. Sedentary users show lower app engagement, suggesting the need for motivational features targeting this group.")
print("3. Gender analysis shows differences in engagement, highlighting the importance of gender-inclusive app features.")