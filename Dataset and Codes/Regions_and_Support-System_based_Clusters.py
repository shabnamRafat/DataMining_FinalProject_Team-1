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
from sklearn.decomposition import PCA
from scipy import stats 
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import DBSCAN
import warnings

# Import other py files
import EDA_VR
import Hypothesis_Testing
import High_and_Low_Performing_Clusters

if __name__ == "__main__":
    # This will only run if my_module.py is run directly
    print("All the relevant files have been imported!")

warnings.filterwarnings('ignore')

#%%

vr_in_education = EDA_VR.vr_in_education_copy
vr_in_education_copy = vr_in_education.copy()
distinguishing_features = High_and_Low_Performing_Clusters.distinguishing_features


#%%



#%%


# %%[markdown]

##### **Q4: How do cluster characteristics vary across different regional and support system contexts?**

##### **Answer:**
# <u>Cluster characteristics across regions: </u></br>


#%%
# Select the region column for clustering
region_column = ['Region']
print(f"Regions: {vr_in_education['Region'].unique()}")

#%%

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

#%%[markdown]
# <u>Cluster characteristics based on support system of school: </u></br>


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

