
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df = pd.read_csv("C:\\Users\\uemaa\\Documents\\Data_Mining\\Final_project_VR\\Virtual_Reality_in_Education_Impact.csv")


# Display basic information about the dataset
print("Dataset Information:\n")
df.info()

print("\nBasic Statistical Description:\n")
print(df.describe())


#%%
# Checking for missing values
def check_missing_values(df):
    missing = df.isnull().sum()
    print("\nMissing Values Summary:\n")
    print(missing[missing > 0])
check_missing_values(df)

# Displaying the first few rows of the dataset
print("\nFirst 5 Rows of the Dataset:\n")
print(df.head())

# Handling Missing Values
# Fill missing numerical values with the median of each column
numerical_features = [feature for feature in df.columns if df[feature].dtype != 'object']
for feature in numerical_features:
    if df[feature].isnull().sum() > 0:
        df[feature].fillna(df[feature].median(), inplace=True)

# Fill missing categorical values with the mode of each column
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'object']
for feature in categorical_features:
    if df[feature].isnull().sum() > 0:
        df[feature].fillna(df[feature].mode()[0], inplace=True)

# Verifying if there are still any missing values
check_missing_values(df)

#%%
# Removing Duplicates
# Drop duplicate rows if any
df.drop_duplicates(inplace=True)
print("\nAfter removing duplicates, dataset shape:", df.shape)

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

df = remove_outliers(df, numerical_features)
print("\nAfter removing outliers, dataset shape:", df.shape)

# Displaying the first few rows of the cleaned dataset
print("\nFirst 5 Rows of the Cleaned Dataset:\n")
print(df.head())

# Save the cleaned dataset to a new CSV file
df.to_csv("C:\\Users\\uemaa\\Documents\\Data_Mining\\Final_project_VR\\Virtual_Reality_in_Education_Impact.csv", index=False)
print("\nCleaned dataset saved to 'Virtual_Reality_in_Education_Impact_Cleaned.csv'")

#%%
# Exploring categorical variables
print("\nCategorical Variables:\n", categorical_features)

# Exploring numerical variables
print("\nNumerical Variables:\n", numerical_features)

# Plotting Histograms for Numerical Features
def plot_histograms(df, numerical_features):
    df[numerical_features].hist(figsize=(15, 10), bins=20, color='skyblue', edgecolor='black')
    plt.suptitle('Distribution of Numerical Features', size=16)
    plt.savefig(r'C:\Users\uemaa\Documents\Data_Mining\Final_project_VR\Figures\histograms_numerical_features.png')
    plt.show()

plot_histograms(df, numerical_features)

# Boxplots for Numerical Features to Detect Outliers
def plot_boxplots(df, numerical_features):
    for feature in numerical_features:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x=feature, color='lightblue')
        plt.title(f'Boxplot of {feature}')
        plt.savefig(rf'C:\Users\uemaa\Documents\Data_Mining\Final_project_VR\Figures\boxplot_{feature}.png')
        plt.show()

plot_boxplots(df, numerical_features)

# Pair Plot to understand relationships between numerical features
sns.pairplot(df[numerical_features])
plt.suptitle('Pair Plot of Numerical Features', y=1.02)
plt.savefig(r'C:\Users\uemaa\Documents\Data_Mining\Final_project_VR\Figures\pairplot_numerical_features.png')
plt.show()

# Correlation Heatmap
def correlation_heatmap(df, numerical_features):
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.savefig(r'C:\Users\uemaa\Documents\Data_Mining\Final_project_VR\Figures\correlation_heatmap.png')
    plt.show()

correlation_heatmap(df, numerical_features)

# Categorical Feature Analysis (Count Plots)
def plot_categorical_counts(df, categorical_features):
    for feature in categorical_features:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=feature, palette='viridis', order=df[feature].value_counts().index)
        plt.xticks(rotation=45)
        plt.title(f'Count Plot of {feature}')
        plt.savefig(rf'C:\Users\uemaa\Documents\Data_Mining\Final_project_VR\Figures\countplot_{feature}.png')
        plt.show()

plot_categorical_counts(df, categorical_features)

# %%
