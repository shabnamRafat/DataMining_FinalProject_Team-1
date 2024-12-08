
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import scipy.stats as stats


#%%
vr_in_education = pd.read_csv('Virtual_Reality_in_Education_Impact.csv')
variables = vr_in_education.columns


#%%
print(f"\033[4mFirst 10 Rows from the Data Frame:\033[0m:\n")
print(tabulate(vr_in_education.head(10), headers='keys', tablefmt='pretty'))
print(f"\nNumber of total observations: {len(vr_in_education)}\n")
print(f"\nThe {len(variables)} variables are: \n{variables.values}\n")
print(f"\033[4mThe types of the variables:\033[0m:\n{vr_in_education.dtypes}")



#%%
# Display basic information about the dataset
print(f"\033[4mDataset Information:\033[0m\n")
print(vr_in_education.info())
print(f"\033[4m\nBasic Statistical Description:\033[0m\n")
print(tabulate(vr_in_education.describe(), headers='keys', tablefmt='pretty'))



#%%
# Checking for missing values
def check_missing_values(df):
    missing = df.isnull().sum()
    print(f"\nTotal Number of Missing Values: {len(missing[missing > 0])}")
    


#%%
# Removing Duplicates
# Drop duplicate rows if any

print("\nBefore removing duplicates, dataset shape:", vr_in_education.shape)
vr_in_education.drop_duplicates(inplace=True)
print("\nAfter removing duplicates, dataset shape:", vr_in_education.shape)



#%%
# Outlier Detection and Handling
# Using the IQR method to remove outliers from numerical features
def remove_outliers(df, numerical_features):
    for feature in numerical_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df


numerical_features = ['Age', 'Hours_of_VR_Usage_Per_Week']
vr_in_education = remove_outliers(vr_in_education, numerical_features)
print("\nAfter removing outliers, dataset shape:", vr_in_education.shape)



#%%
# Converting the variables to required types

vr_in_education_copy = vr_in_education.copy()

vr_in_education['Engagement_Level'] = vr_in_education['Engagement_Level'].astype('object')
vr_in_education['Perceived_Effectiveness_of_VR'] = vr_in_education['Perceived_Effectiveness_of_VR'].astype('object')
vr_in_education['Impact_on_Creativity'] = vr_in_education['Impact_on_Creativity'].astype('object')

variables = vr_in_education.columns
print(f"\033[4mThe types of the variables:\033[0m:\n{vr_in_education.dtypes}")



#%%

# Count the occurrences of each category in the "Usage_of_VR_in_Education" column
usage_counts = vr_in_education['Usage_of_VR_in_Education'].value_counts()
# Create the pie chart
plt.pie(usage_counts, labels=usage_counts.index, autopct='%1.1f%%', startangle=90, colors=['blue', 'lightblue'])
plt.title('Usage of VR in Education')
plt.show()


#%%

# Distribution of VR Usage Hours
sns.histplot(vr_in_education['Hours_of_VR_Usage_Per_Week'], kde=False, bins=10, color='seagreen')
plt.xlabel('Hours of VR Usage per Week')
plt.ylabel('Frequency')
plt.title('Histogram of VR Usage per Week')
plt.show()

#%%

# Create a Q-Q plot
stats.probplot(vr_in_education['Hours_of_VR_Usage_Per_Week'], dist="norm", plot=plt)
plt.title('Q-Q Plot for VR Usage per Week')
plt.show()


#%%

# Declare class for diagram

class RelationshipPlots:
    def __init__(self, dataframe, outlier_design=None):
        """Initialize with a fixed figure size and dataframe."""
        self.data = dataframe

    def plot_count(self, x, hue, title, xlabel, ylabel, palette='Set2'):
        """Generate a countplot with flexible x, hue, title, xlabel, ylabel."""
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.data, x=x, hue=hue, palette=palette)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.legend(title=hue)
        plt.tight_layout()
        plt.show()

    def plot_box(self, x, y, title, xlabel, ylabel, palette='Set2', outlier_design=None):
        """Generate a boxplot with flexible x, y, title, xlabel, ylabel."""
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.data, x=x, y=y, palette=palette, flierprops=outlier_design)
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_violin(self, x, y, title, xlabel, ylabel, palette='Set2'):
        """Generate a violin plot with flexible x, y, title, xlabel, ylabel."""
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=self.data, x=x, y=y, palette=palette)
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.tight_layout()
        plt.show()






#%%
outlier_design = dict(marker='*', markerfacecolor='red', markersize=8, linestyle='none', markeredgecolor='black', markeredgewidth=.5)


#%%

vr_usage = RelationshipPlots(vr_in_education, outlier_design=outlier_design)

vr_usage.plot_count(x='Gender', hue='Usage_of_VR_in_Education', title='Gender vs VR Usage', 
                    xlabel='Gender', ylabel='VR Usage', palette='Set2')
vr_usage.plot_box(x='Gender', y='Hours_of_VR_Usage_Per_Week', title='Gender vs Hours of VR Usage', 
                    xlabel='Gender', ylabel='Hours of VR Usage per Week', palette='Set2')



#%%

vr_usage.plot_count(x='Subject', hue='Usage_of_VR_in_Education', title='Subject vs VR Usage', 
                    xlabel='Subject', ylabel='VR Usage', palette='Set2')
vr_usage.plot_violin(x='Subject', y='Hours_of_VR_Usage_Per_Week', title='Subject vs Hours of VR Usage', 
                    xlabel='Subject', ylabel='Hours of VR Usage per Week', palette='Set2')



#%%

vr_usage.plot_count(x='Field_of_Study', hue='Usage_of_VR_in_Education', title='Field_of_Study vs VR Usage', 
                    xlabel='Field_of_Study', ylabel='VR Usage', palette='Set2')
vr_usage.plot_violin(x='Field_of_Study', y='Hours_of_VR_Usage_Per_Week', title='Field_of_Study vs Hours of VR Usage', 
                    xlabel='Field_of_Study', ylabel='Hours of VR Usage per Week', palette='Set2')



#%%

vr_usage.plot_count(x='Grade_Level', hue='Usage_of_VR_in_Education', title='Grade_Level vs VR Usage', 
                    xlabel='Grade_Level', ylabel='VR Usage', palette='Set2')
vr_usage.plot_violin(x='Grade_Level', y='Hours_of_VR_Usage_Per_Week', title='Grade_Level vs Hours of VR Usage', 
                    xlabel='Grade_Level', ylabel='Hours of VR Usage per Week', palette='Set2')



#######################################################################################################################




#%%

# Categorical Feature Analysis (Count Plots)
def plot_categorical_counts(df, categorical_features):
    for feature in categorical_features:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=feature, palette='viridis', order=df[feature].value_counts().index)
        plt.xticks(rotation=45)
        plt.title(f'Count Plot of {feature}')
        plt.show()


#%%
categorical_features = ['Gender', 'Grade_Level', 'Field_of_Study', 'Usage_of_VR_in_Education', 'Engagement_Level',
                         'Improvement_in_Learning_Outcomes', 'Subject', 'Instructor_VR_Proficiency', 
                         'Perceived_Effectiveness_of_VR', 'Access_to_VR_Equipment', 'Impact_on_Creativity',
                         'Stress_Level_with_VR_Usage', 'Collaboration_with_Peers_via_VR','Feedback_from_Educators_on_VR', 
                         'Interest_in_Continuing_VR_Based_Learning', 'Region', 'School_Support_for_VR_in_Curriculum']

plot_categorical_counts(vr_in_education, categorical_features)

# %%
for feature in categorical_features:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=vr_in_education, x=feature, palette='viridis', order=vr_in_education[feature].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f'Count Plot of {feature}')
    plt.show()
# %%
