#%%
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

import warnings

warnings.filterwarnings('ignore')

#%%
vr_in_education = pd.read_csv('Virtual_Reality_in_Education_Impact.csv')
variables = vr_in_education.columns

#%%
from tabulate import tabulate
print(tabulate(vr_in_education.head(), headers='keys', tablefmt='pretty'))
print(variables.values)
print(f"Number of total observations: {len(vr_in_education)}")


# %%[markdown]

##### **Q3: What are the key distinguishing features between high-performing and low-performing clusters?**

##### **Answer:**
# <u>Performance Indicators for clustering: </u></br>
# [ 'Engagement_Level' 'Improvement_in_Learning_Outcomes' 'Perceived_Effectiveness_of_VR' 'Impact_on_Creativity'] 
# 
# <u>Distinguishing features:</u></br> 
# ['Usage_of_VR_in_Education' 'Hours_of_VR_Usage_Per_Week' 'Instructor_VR_Proficiency' 'Access_to_VR_Equipment' 'School_Support_for_VR_in_Curriculum' 'Collaboration_with_Peers_via_VR'  'Stress_Level_with_VR_Usage' 'Feedback_from_Educators_on_VR' 'Interest_in_Continuing_VR_Based_Learning']
#


#%%

print(f"Engagement Level Values: {np.unique(vr_in_education['Engagement_Level'])}")
print(f"Improvement in Learning Outcome: {np.unique(vr_in_education['Improvement_in_Learning_Outcomes'])}")
print(f"Perceived Effectiveness of VR: {np.unique(vr_in_education['Perceived_Effectiveness_of_VR'])}")
print(f"Impact on Creativity: {np.unique(vr_in_education['Impact_on_Creativity'])}")


#%%
print(f"Usage_of_VR_in_Education: {np.unique(vr_in_education['Usage_of_VR_in_Education'])}")
print(f"Hours_of_VR_Usage_Per_Week: {np.unique(vr_in_education['Interest_in_Continuing_VR_Based_Learning'])}")
print(f"Instructor_VR_Proficiency: {np.unique(vr_in_education['Stress_Level_with_VR_Usage'])}")
print(f"Access_to_VR_Equipment: {np.unique(vr_in_education['Interest_in_Continuing_VR_Based_Learning'])}")
print(f"School_Support_for_VR_in_Curriculum: {np.unique(vr_in_education['Stress_Level_with_VR_Usage'])}")
print(f"Collaboration with Peers via VR: {np.unique(vr_in_education['Collaboration_with_Peers_via_VR'])}")
print(f"Stress Level with VR Usage: {np.unique(vr_in_education['Stress_Level_with_VR_Usage'])}")
print(f"Feedback from Educators on VR: {np.unique(vr_in_education['Feedback_from_Educators_on_VR'])}")
print(f"Interest in Continuing VR Based Learning: {np.unique(vr_in_education['Interest_in_Continuing_VR_Based_Learning'])}")
print(f"Gender: {np.unique(vr_in_education['Region'])}")


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

#############################################################################

#%%
import statsmodels.api as sm
from statsmodels.formula.api import ols


#%%

# Define the columns and values for additional factors
student_attributes = ['Gender','Grade_Level', 'Field_of_Study', 'Region', 'School_Support_for_VR_in_Curriculum']

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
for factor in student_attributes:
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
