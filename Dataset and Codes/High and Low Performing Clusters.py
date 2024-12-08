#%%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import statsmodels.api as sm
from statsmodels.formula.api import ols
from tabulate import tabulate

import EDA_VR

import warnings

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    # This will only run if my_module.py is run directly
    print("All the relevant files have been imported!")

#%%
print(EDA_VR.variables.values)




#%%
vr_in_education = EDA_VR.vr_in_education_copy


#%%

# %%[markdown]

##### **Q3: What are the key distinguishing features between high-performing and low-performing clusters?**

##### **Answer:**
# <u>Performance Indicators for clustering: </u></br>
# [ 'Engagement_Level' 'Improvement_in_Learning_Outcomes' 'Impact_on_Creativity'] 
# 
# <u>Distinguishing features:</u></br>
# ['Usage_of_VR_in_Education' 'Hours_of_VR_Usage_Per_Week' 'Instructor_VR_Proficiency' 'Access_to_VR_Equipment' 'School_Support_for_VR_in_Curriculum' 'Collaboration_with_Peers_via_VR'  'Stress_Level_with_VR_Usage' 'Feedback_from_Educators_on_VR' 'Interest_in_Continuing_VR_Based_Learning' 'Perceived_Effectiveness_of_VR']
#




#%%

# Check the values in performance features

print(f"Engagement Level Values: {np.unique(vr_in_education['Engagement_Level'])}")
print(f"Improvement in Learning Outcome: {np.unique(vr_in_education['Improvement_in_Learning_Outcomes'])}")
print(f"Impact on Creativity: {np.unique(vr_in_education['Impact_on_Creativity'])}")



#%%[markdown]

# Creating High-Performing and Low-Performing Clusters:

#%%

# Select performance indicators for clustering
performance_indicators = ['Engagement_Level', 'Improvement_in_Learning_Outcomes',  'Impact_on_Creativity']

# Encode 'Improvement_in_Learning_Outcomes'
vr_in_education['Improvement_in_Learning_Outcomes'] = vr_in_education['Improvement_in_Learning_Outcomes'].map({'No': 0, 'Yes': 1})

# Normalize the performance indicators
scaler = StandardScaler()
performance_indicators_scaled = scaler.fit_transform(vr_in_education[performance_indicators])

# Apply k-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
vr_in_education['Cluster'] = kmeans.fit_predict(performance_indicators_scaled)

# Evaluate the clustering using the silhouette score
silhouette_avg = silhouette_score(performance_indicators_scaled, vr_in_education['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')


# Compare performance indicators between the two clusters
cluster_0 = vr_in_education[vr_in_education['Cluster'] == 0]
cluster_1 = vr_in_education[vr_in_education['Cluster'] == 1]

# Calculate mean values for each performance indicator
cluster_0_means = cluster_0[performance_indicators].mean()
cluster_1_means = cluster_1[performance_indicators].mean()

# Display the results
performance_comparison_vr_in_education = pd.DataFrame({
    'Cluster_0_Mean': cluster_0_means,
    'Cluster_1_Mean': cluster_1_means
})

print(performance_comparison_vr_in_education)


# Determine which cluster is high-performing based on the mean values of performance indicators
high_performing_cluster = 'Cluster_1' if cluster_1_means.mean() > cluster_0_means.mean() else 'Cluster_0'
print(f'High-performing cluster: {high_performing_cluster}')


#%%
from sklearn.decomposition import PCA

# Identify the clusters and noise points
clusters = vr_in_education['Cluster'].unique()
print(f'Clusters identified: {clusters}')

# Filter out noise points (cluster label -1)
vr_in_education_filtered = vr_in_education[vr_in_education['Cluster'] != -1]

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(performance_indicators_scaled)
vr_in_education_filtered['PCA1'] = principal_components[:, 0]
vr_in_education_filtered['PCA2'] = principal_components[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 7))
for cluster in clusters:
    if cluster != -1:  # Exclude noise points
        cluster_data = vr_in_education_filtered[vr_in_education_filtered['Cluster'] == cluster]
        plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}')

plt.title('K-Means Clustering Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()



#%%
# Select distinguishing features
distinguishing_features = ['Usage_of_VR_in_Education', 'Hours_of_VR_Usage_Per_Week', 'Instructor_VR_Proficiency', 'Access_to_VR_Equipment',  'Collaboration_with_Peers_via_VR', 'Stress_Level_with_VR_Usage', 'Feedback_from_Educators_on_VR', 'Interest_in_Continuing_VR_Based_Learning', 'Perceived_Effectiveness_of_VR']

# Encode categorical variables
label_encoders = {}
for column in distinguishing_features:
    if vr_in_education[column].dtype == 'object':
        le = LabelEncoder()
        vr_in_education[column] = le.fit_transform(vr_in_education[column])
        label_encoders[column] = le

# Compare distinguishing features between the two clusters
high_performing = vr_in_education[vr_in_education['Cluster'] == (1 if high_performing_cluster == 'Cluster_1' else 0)]
low_performing = vr_in_education[vr_in_education['Cluster'] == (0 if high_performing_cluster == 'Cluster_1' else 1)]

# Calculate mean values for each distinguishing feature
high_performing_means = high_performing[distinguishing_features].mean()
low_performing_means = low_performing[distinguishing_features].mean()

# Display the results
comparison_vr_in_education = pd.DataFrame({
    'High-performing_Mean': high_performing_means,
    'Low-performing_Mean': low_performing_means
})

print(comparison_vr_in_education)

#%%

# Dictionary to store t-test results
ttest_results = {}

# Perform t-test for each feature
for feature in distinguishing_features:
    # Perform independent two-sample t-test (assuming unequal variance by default)
    t_stat, p_value = stats.ttest_ind(high_performing[feature], low_performing[feature], equal_var=False)
    
    # Store the results
    ttest_results[feature] = {
        't-statistic': t_stat,
        'p-value': p_value
    }

# Convert t-test results into a DataFrame for easy visualization
ttest_results_df = pd.DataFrame.from_dict(ttest_results, orient='index')

# Display the results
print(ttest_results_df)






#############################################################################


#%%

# Define the columns and values for student attributes(group-1)
student_attributes_group_1 = ['Gender','Grade_Level', 'Field_of_Study', 'Region', 'School_Support_for_VR_in_Curriculum']

# Define a function to perform ANOVA test for different factors
def perform_anova_test(df, distinguishing_features, factor_col):
    anova_results = {}
    for feature in distinguishing_features:
        model = ols(f'{feature} ~ Cluster + C({factor_col}) + C(Usage_of_VR_in_Education)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_results[feature] = anova_table
    return anova_results

# Perform ANOVA test for each different factor
anova_results = {}
for factor in student_attributes_group_1:
    anova_results[factor] = perform_anova_test(vr_in_education, distinguishing_features, factor)

# Display the results with p-values and interpretation
for factor, results in anova_results.items():
    print(f"ANOVA results for {factor}:")
    print("\n")
    for feature, result in results.items():
        print(f"ANOVA results for {feature}:")
        #print(result)
        p_value = result.iloc[0, -1]  # Extract the p-value
        print(f"P-value: {p_value:.2f}")
        if p_value < 0.05:
            print(f"Interpretation: There is a significant difference in {feature} across {factor} based on VR usage between high and low performing clusters.\n")
        else:
            print(f"Interpretation: There is no significant difference in {feature} across {factor} based on VR usage between high and low performing clusters.\n")


# %%[markdown]

##### **Q4: How do cluster characteristics vary across different regional and support system contexts?**


#%%



# Define the columns and values for student attributes(group-1)
student_attributes_group_2 = ['Region', 'School_Support_for_VR_in_Curriculum']

# Define a function to perform ANOVA test for different factors
def perform_anova_test(df, distinguishing_features, factor_col):
    anova_results = {}
    for feature in distinguishing_features:
        model = ols(f'{feature} ~ Cluster + C({factor_col}) + C(Usage_of_VR_in_Education)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_results[feature] = anova_table
    return anova_results

# Perform ANOVA test for each different factor
anova_results = {}
for factor in student_attributes_group_2:
    anova_results[factor] = perform_anova_test(vr_in_education, distinguishing_features, factor)

# Display the results with p-values and interpretation
for factor, results in anova_results.items():
    print(f"ANOVA results for {factor}:")
    print("\n")
    for feature, result in results.items():
        print(f"ANOVA results for {feature}:")
        #print(result)
        p_value = result.iloc[0, -1]  # Extract the p-value
        print(f"P-value: {p_value:.2f}")
        if p_value < 0.05:
            print(f"Interpretation: There is a significant difference in {feature} across {factor} based on VR usage between high and low performing clusters.\n")
        else:
            print(f"Interpretation: There is no significant difference in {feature} across {factor} based on VR usage between high and low performing clusters.\n")



#%%

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster




# %%

from sklearn.cluster import DBSCAN
# Select the region column for clustering
region_column = ['Region']

# Encode the 'Region' column
label_encoder = LabelEncoder()
vr_in_education['Region_Encoded'] = label_encoder.fit_transform(vr_in_education['Region'])

# Normalize the 'Region_Encoded' column (optional, depending on algorithm used)
# Since it's already encoded numerically, normalization isn't strictly necessary here, but for DBSCAN, it's often helpful.
scaler = StandardScaler()
region_scaled = scaler.fit_transform(vr_in_education[['Region_Encoded']])

# Apply DBSCAN clustering (or Agglomerative Clustering, as appropriate)
dbscan = DBSCAN(eps=0.5, min_samples=2)
vr_in_education['Cluster_Region'] = dbscan.fit_predict(region_scaled)

# Identify the clusters and noise points
clusters = vr_in_education['Cluster_Region'].unique()
print(f'Clusters identified: {clusters}')

# Filter out noise points (cluster label -1)
vr_in_education_filtered = vr_in_education[vr_in_education['Cluster_Region'] != -1]

# Evaluate the clustering using the silhouette score
silhouette_avg = silhouette_score(region_scaled, vr_in_education['Cluster_Region'])
print(f'Silhouette Score: {silhouette_avg}')





#%%



# Compare distinguishing features between the regions (clusters)
region_clusters = vr_in_education['Region_Encoded'].unique()

# Initialize a dictionary to store the mean values for each region
region_means = {}

# Loop through each region and calculate the mean values for the distinguishing features
for region in region_clusters:
    region_data = vr_in_education[vr_in_education['Region_Encoded'] == region]
    region_means[region] = region_data[distinguishing_features].mean()

# Convert the region means dictionary to a DataFrame for better readability
region_comparison = pd.DataFrame(region_means)

# Display the results
print(region_comparison)


#%%

# Identify the unique clusters (for example, based on the 'Region' or any other clustering method)
clusters_region = vr_in_education['Cluster_Region'].unique()

# Dictionary to store the results of ANOVA
anova_results = {}

# Loop over each distinguishing feature and perform ANOVA for each cluster
for feature in distinguishing_features:
    # List to store the data for each cluster for the current feature
    cluster_data = [vr_in_education[vr_in_education['Cluster_Region'] == cluster][feature] for cluster in clusters_region]
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*cluster_data)
    
    # Store the results
    anova_results[feature] = {
        'F-statistic': f_stat,
        'p-value': p_value
    }

# Convert the ANOVA results into a DataFrame for easy visualization
anova_results_df = pd.DataFrame.from_dict(anova_results, orient='index')

# Display the results
print(anova_results_df)





#%%

# Encode 'School_Support_for_VR_in_Curriculum' (assuming it's 'Yes'/'No')
vr_in_education['School_Support_for_VR_in_Curriculum_Encoded'] = vr_in_education['School_Support_for_VR_in_Curriculum'].map({'No': 0, 'Yes': 1})

# Select only the encoded 'School_Support_for_VR_in_Curriculum' for clustering
support_column = vr_in_education[['School_Support_for_VR_in_Curriculum_Encoded']]

# Apply KMeans clustering (with 2 clusters as requested)
kmeans = KMeans(n_clusters=2, random_state=42)
vr_in_education['Cluster_School_Support_System'] = kmeans.fit_predict(support_column)

# Calculate the silhouette score
silhouette_avg = silhouette_score(support_column, vr_in_education['Cluster_School_Support_System'])
print(f'Silhouette Score: {silhouette_avg}')

# Encode categorical variables if needed
label_encoders = {}
for column in distinguishing_features:
    if vr_in_education[column].dtype == 'object':
        le = LabelEncoder()
        vr_in_education[column] = le.fit_transform(vr_in_education[column])
        label_encoders[column] = le

# Compare distinguishing features between the two clusters
cluster_0 = vr_in_education[vr_in_education['Cluster_School_Support_System'] == 0]
cluster_1 = vr_in_education[vr_in_education['Cluster_School_Support_System'] == 1]

# Calculate mean values for each distinguishing feature
cluster_0_means = cluster_0[distinguishing_features].mean()
cluster_1_means = cluster_1[distinguishing_features].mean()

# Display the results
performance_comparison_vr_in_education = pd.DataFrame({
    'Cluster_School_Support_System_0_Mean': cluster_0_means,
    'Cluster_School_Support_System_1_Mean': cluster_1_means
})

print(performance_comparison_vr_in_education)



#%%
# Results dictionary to store p-values and test results
test_results = {}

# Loop over each distinguishing feature and perform t-test
for feature in distinguishing_features:
    cluster_0 = vr_in_education[vr_in_education['Cluster_School_Support_System'] == 0][feature]
    cluster_1 = vr_in_education[vr_in_education['Cluster_School_Support_System'] == 1][feature]
    
    # Perform t-test between the two clusters for the current feature
    t_stat, p_value = stats.ttest_ind(cluster_0, cluster_1)
    
    # Store the results
    test_results[feature] = {
        't_statistic': t_stat,
        'p_value': p_value
    }

# Convert results into a DataFrame for easy visualization
test_results_df = pd.DataFrame(test_results).T
print(test_results_df)



# %%


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming your data is loaded into a pandas DataFrame 'df'
# Preprocessing: Convert categorical variables to numeric (e.g., One-Hot Encoding)

df_without_region = vr_in_education_copy.drop('Region', axis=1)

df_encoded = pd.get_dummies(vr_in_education_copy, drop_first=True)

X = df_encoded  # all features, after one-hot encoding
y = vr_in_education_copy['Region']  # target variable


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# %%
print(df_encoded.columns)

# %%
import numpy as np
import pandas as pd
from scipy import stats

# Assuming you have the clusters assigned as 'Cluster' column in the dataframe
# Features to test
features = [
    'Usage_of_VR_in_Education', 'Hours_of_VR_Usage_Per_Week', 'Instructor_VR_Proficiency',
    'Access_to_VR_Equipment', 'School_Support_for_VR_in_Curriculum', 'Collaboration_with_Peers_via_VR',
    'Stress_Level_with_VR_Usage', 'Feedback_from_Educators_on_VR', 'Interest_in_Continuing_VR_Based_Learning',
    'Perceived_Effectiveness_of_VR'
]

# Loop through each feature and perform ANOVA
for feature in features:
    # Group the data by clusters and extract the feature values
    groups = [vr_in_education_copy[vr_in_education_copy['Region'] == i][feature].dropna() for i in vr_in_education_copy['Region'].unique()]
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    print(f"Feature: {feature}")
    print(f"F-statistic: {f_stat}, P-value: {p_value}")
    
    # Check if the result is significant
    if p_value < 0.05:
        print("There is a significant difference in means among the clusters.\n")
    else:
        print("There is no significant difference in means among the clusters.\n")

# %%
vr_in_education_copy.columns
# %%
