#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
#%%

data = pd.read_csv('VR.csv')

# Understanding the structure of the dataset

data.head()


#%%

# Checking for missing values

data.isnull().sum()

#%%

# Summary statistics of the numerical columns

data.describe()



#%%

# Checking the data types and number of unique values

data.dtypes
data.nunique()


#%%

# Box plots to detect outliers

numerical_cols = ['Hours_of_VR_Usage_Per_Week', 'Improvement_in_Learning_Outcomes', 
                  'Instructor_VR_Proficiency', 'Perceived_Effectiveness_of_VR', 
                  'Impact_on_Creativity']

plt.figure(figsize=(10, 6))
sns.boxplot(data=data[numerical_cols], palette='Set2')
plt.title('Box plots for Numerical Features')
plt.show()

#%%

numerical_columns = ['Age', 'Hours_of_VR_Usage_Per_Week', 'Engagement_Level', 
                     'Perceived_Effectiveness_of_VR', 'Impact_on_Creativity', 'Stress_Level_with_VR_Usage']

# Histograms for numerical columns

plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[col], kde=True, bins=15)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


#%%

categorical_columns = ['Gender', 'Grade_Level', 'Field_of_Study', 'Usage_of_VR_in_Education', 
                       'Improvement_in_Learning_Outcomes', 'Instructor_VR_Proficiency', 
                       'Access_to_VR_Equipment', 'Stress_Level_with_VR_Usage', 
                       'Collaboration_with_Peers_via_VR', 'Feedback_from_Educators_on_VR', 
                       'Interest_in_Continuing_VR_Based_Learning', 'Region', 
                       'School_Support_for_VR_in_Curriculum']

# Count plots for categorical columns

plt.figure(figsize=(12, 12))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(4, 4, i)
    sns.countplot(y=data[col], palette="viridis")
    plt.title(f'Count of {col}')
plt.tight_layout()
plt.show()




#%%

# Boxplots for numerical vs categorical relationships

plt.figure(figsize=(14, 10))
for i, col in enumerate(categorical_columns, 1):
      
        plt.subplot(4, 4, i)
        sns.boxplot(x=data[col], y=data['Engagement_Level'], palette="Set2")
        plt.title(f'Engagement Level vs {col}')
plt.tight_layout()
plt.show()



#%%

#SMART Question 2:

#How does the instructor's VR proficiency affect students' improvement in learning outcomes?



# Applying Label Encoding

label_encoder = LabelEncoder()
data['Instructor_VR_Proficiency'] = label_encoder.fit_transform(data['Instructor_VR_Proficiency'])


print(data[['Instructor_VR_Proficiency']].head())



#%%


# Boxplot to visualize Improvement in Learning Outcomes across Instructor's VR Proficiency

plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Instructor_VR_Proficiency', y='Improvement_in_Learning_Outcomes', palette='Set2')
plt.xticks([0, 1, 2], ['Beginner', 'Intermediate', 'Advanced'])

plt.title('Improvement in Learning Outcomes by Instructor\'s VR Proficiency')
plt.xlabel('Instructor\'s VR Proficiency')
plt.ylabel('Improvement in Learning Outcomes')
plt.show()


# %%

# Converting 'Improvement_in_Learning_Outcomes' from 'Yes'/'No' to 1/0

data['Improvement_in_Learning_Outcomes'] = data['Improvement_in_Learning_Outcomes'].map({'Yes': 1, 'No': 0})



# Grouping the data by Instructor VR Proficiency and extracting the 'Improvement_in_Learning_Outcomes' and performing One-Way ANOVA

grouped = [data[data['Instructor_VR_Proficiency'] == proficiency]['Improvement_in_Learning_Outcomes'] 
           for proficiency in data['Instructor_VR_Proficiency'].unique()]


f_statistic, p_value = stats.f_oneway(*grouped)

print(f"F-statistic: {f_statistic}, p-value: {p_value}")


# Based on the F-statistic and p-value from the One-Way ANOVA, it seems that Instructor VR Proficiency does not have a statistically significant effect on Improvement in Learning Outcomes.

# %%






