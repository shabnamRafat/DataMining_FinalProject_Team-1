#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Load the dataset
# Note: Replace the path with the appropriate path to your dataset.
df = pd.read_csv("C:\\Users\\uemaa\\Documents\\Data_Mining\\Final_project_VR\\Virtual_Reality_in_Education_Impact.csv")

# Display basic information about the dataset
print("Dataset Information:\n")
df.info()

print("\nBasic Statistical Description:\n")
print(df.describe())


#%%
# Feature Engineering
# Creating new features based on existing data

# Feature 1: Total VR Usage Hours
# Assuming there are columns like 'daily_usage_hours' or similar that can be summed to get total usage
if 'daily_usage_hours' in df.columns:
    df['total_vr_usage_hours'] = df['daily_usage_hours'] * 7  # Assuming the data represents weekly engagement
    print("\nFeature 'total_vr_usage_hours' added.")

# Feature 2: Engagement Level Categorization
# Binning VR usage hours into categories: Low, Medium, High
if 'total_vr_usage_hours' in df.columns:
    bins = [0, 10, 30, np.inf]
    labels = ['Low', 'Medium', 'High']
    df['engagement_level'] = pd.cut(df['total_vr_usage_hours'], bins=bins, labels=labels)
    print("\nFeature 'engagement_level' added.")

# Feature 3: Stress Level Normalization
# Assuming there is a 'stress_level' column, we can normalize it between 0 and 1
if 'stress_level' in df.columns:
    df['normalized_stress_level'] = (df['stress_level'] - df['stress_level'].min()) / (df['stress_level'].max() - df['stress_level'].min())
    print("\nFeature 'normalized_stress_level' added.")

# Feature 4: Creativity Index
# Creating a combined index based on creativity scores, assuming multiple columns contribute to creativity
creativity_columns = [col for col in df.columns if 'creativity' in col.lower()]
if len(creativity_columns) > 0:
    df['creativity_index'] = df[creativity_columns].mean(axis=1)
    print("\nFeature 'creativity_index' added.")

# Feature 5: Age Group Categorization
# Categorizing age into groups: Teen, Young Adult, Adult, Senior
if 'age' in df.columns:
    bins = [0, 18, 25, 65, np.inf]
    labels = ['Teen', 'Young Adult', 'Adult', 'Senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    print("\nFeature 'age_group' added.")

# Verifying new features
print("\nFirst 5 Rows with New Features:\n")
print(df.head())

# Save the dataset with new features to a new CSV file
df.to_csv("C:\\Users\\uemaa\\Documents\\Data_Mining\\Final_project_VR\\Virtual_Reality_in_Education_Features.csv", index=False)
print("\nDataset with new features saved to 'Virtual_Reality_in_Education_Impact_Features.csv'")

#%%
# Plotting Histograms for New Features
def plot_new_feature_histograms(df, new_features):
    df[new_features].hist(figsize=(15, 10), bins=20, color='orange', edgecolor='black')
    plt.suptitle('Distribution of New Features', size=16)
    plt.savefig(r'C:\\Users\\uemaa\\Documents\\Data_Mining\\Final_project_VR\\Figures\\histograms_new_features.png')
    plt.show()

new_features = ['total_vr_usage_hours', 'normalized_stress_level', 'creativity_index']
plot_new_feature_histograms(df, [feature for feature in new_features if feature in df.columns])

# %%
