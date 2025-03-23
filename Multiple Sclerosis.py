import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set basic style parameters
plt.style.use('default')
sns.set_palette("husl")

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. Regional Prevalence Data
regions = ['North America', 'Europe', 'Australia/NZ', 'Latin America', 'Asia', 'Africa']
prevalence = [164.6, 133.0, 125.2, 12.5, 5.2, 5.0]

ax1 = plt.subplot(2, 2, 1)
bars = sns.barplot(x=regions, y=prevalence)
plt.title('MS Prevalence by Region (per 100,000 population)', pad=20)
plt.xticks(rotation=45)
plt.ylabel('Cases per 100,000 population')
for i, v in enumerate(prevalence):
    plt.text(i, v, f'{v}', ha='center', va='bottom')

# 2. Gender Distribution in Australia
ax2 = plt.subplot(2, 2, 2)
gender_data = [75, 25]  # 3:1 ratio (75% female, 25% male)
plt.pie(gender_data, labels=['Female', 'Male'], autopct='%1.1f%%', colors=['#FF9999', '#66B2FF'])
plt.title('Gender Distribution of MS in Australia')

# 3. Genetic Risk Factors
ax3 = plt.subplot(2, 2, 3)
risk_categories = ['General\nPopulation', 'Sibling\nRisk', 'Parent-Child\nRisk', 'Monozygotic\nTwin']
risk_percentages = [0.1, 4, 2, 25]  # Approximate values
bars = sns.barplot(x=risk_categories, y=risk_percentages)
plt.title('Genetic Risk Factors for MS')
plt.ylabel('Risk Percentage (%)')
for i, v in enumerate(risk_percentages):
    plt.text(i, v, f'{v}%', ha='center', va='bottom')

# 4. Age Distribution
ax4 = plt.subplot(2, 2, 4)
age_ranges = ['20-29', '30-39', '40-49', '50-59', '60+']
age_distribution = [25, 35, 25, 10, 5]  # Approximate distribution
plt.plot(age_ranges, age_distribution, marker='o', linewidth=2, markersize=10)
plt.title('Age Distribution at MS Diagnosis')
plt.ylabel('Percentage of Cases (%)')
plt.xlabel('Age Range')
for i, v in enumerate(age_distribution):
    plt.text(i, v, f'{v}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Create tables with additional data
ms_data = {
    'Category': ['Global MS Population', 'Australian MS Population', 'Average Age of Diagnosis', 
                'Annual Cost (AUD)', 'EBV Positive Rate'],
    'Value': ['2.8 million', '25,600+', '32 years', '$1.75 billion', '99.5%']
}

risk_factors = {
    'Risk Factor': ['HLA-DRB1*15:01 (Heterozygous)', 'HLA-DRB1*15:01 (Homozygous)', 
                   'Vitamin D Deficiency', 'EBV Infection'],
    'Risk Increase': ['3-fold', '6-fold', '2-fold', 'Significant association']
}

# Convert to DataFrames
ms_df = pd.DataFrame(ms_data)
risk_df = pd.DataFrame(risk_factors)

# Display tables
print("\nMS Key Statistics:")
print(ms_df.to_string(index=False))
print("\nRisk Factor Analysis:")
print(risk_df.to_string(index=False))
