#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import EDA_VR

import warnings
warnings.filterwarnings('ignore')



#%% Load Data
data = EDA_VR.vr_in_education



#%% Data Preparation: Encode categorical variables
categorical_features = ['Gender', 'Grade_Level', 'Field_of_Study', 'Subject', 'Region', 'School_Support_for_VR_in_Curriculum']
# Using OneHotEncoder to transform categorical data into binary features
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder='passthrough')
data_encoded = transformer.fit_transform(data)
data_encoded = pd.DataFrame(data_encoded, columns=transformer.get_feature_names_out())



#%% Correlation Heatmap
numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.show()



#%% Clustering Analysis
binary_cols = ['Usage_of_VR_in_Education', 'Access_to_VR_Equipment', 'Collaboration_with_Peers_via_VR', 
               'Feedback_from_Educators_on_VR', 'Interest_in_Continuing_VR_Based_Learning', 
               'School_Support_for_VR_in_Curriculum']
for col in binary_cols:
    data[col] = data[col].map({'Yes': 1, 'No': 0})
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = ['Gender', 'Grade_Level', 'Field_of_Study', 'Subject', 'Region', 'School_Support_for_VR_in_Curriculum']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  
    ('scaler', StandardScaler()) 
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
cluster_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('cluster', KMeans(n_clusters=5, random_state=42))  
])
cluster_pipeline.fit(data)
data['Cluster'] = cluster_pipeline['cluster'].labels_

# Visualizing the clusters based on VR engagement and usage hours
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Hours_of_VR_Usage_Per_Week', y='Engagement_Level', hue='Cluster', data=data, palette='viridis')
plt.title('Cluster Distribution by VR Engagement and Hours of Usage')
plt.xlabel('Hours of VR Usage Per Week')
plt.ylabel('Engagement Level')
plt.legend(title='Cluster')
plt.show()



#%% Predictive Modeling
data.replace({'Yes': 1, 'No': 0}, inplace=True)
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = pd.to_numeric(data[column], errors='coerce')
imputer = SimpleImputer(strategy='mean')
X = data.drop('Improvement_in_Learning_Outcomes', axis=1) 
y = data['Improvement_in_Learning_Outcomes']
X_imputed = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}, R2: {r2}')



#%% Prediction Details per Cluster
binary_cols = ['Usage_of_VR_in_Education', 'Access_to_VR_Equipment', 'Collaboration_with_Peers_via_VR', 
               'Feedback_from_Educators_on_VR', 'Interest_in_Continuing_VR_Based_Learning', 
               'School_Support_for_VR_in_Curriculum']
for col in binary_cols:
    data[col] = data[col].map({'Yes': 1, 'No': 0}).fillna(data[col])
print("Check for 'Yes'/'No' values:", data[binary_cols].apply(lambda x: x.unique()))
categorical_features = [col for col in data.columns if data[col].dtype == 'object' and col not in ['Improvement_in_Learning_Outcomes', 'Cluster']]
numerical_features = [col for col in data.columns if data[col].dtype in ['int64', 'float64'] and col not in ['Improvement_in_Learning_Outcomes', 'Cluster']]
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]
    X_c = cluster_data.drop(['Improvement_in_Learning_Outcomes', 'Cluster'], axis=1)
    y_c = cluster_data['Improvement_in_Learning_Outcomes']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train_c, y_train_c)
    y_pred_c = model_pipeline.predict(X_test_c)
    mse_c = mean_squared_error(y_test_c, y_pred_c)
    r2_c = r2_score(y_test_c, y_pred_c)
    print(f"Cluster {cluster} - MSE: {mse_c:.2f}, R2: {r2_c:.2f}")



# %%
#Model Evaluation with Random Forest
pca = PCA(n_components=0.95)  
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('pca', pca)
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]
    X_c = cluster_data.drop(['Improvement_in_Learning_Outcomes', 'Cluster'], axis=1)
    y_c = cluster_data['Improvement_in_Learning_Outcomes']
    scores = cross_val_score(model_pipeline, X_c, y_c, cv=5, scoring='neg_mean_squared_error')
    mse_c = -scores.mean()  
    r2_scores = cross_val_score(model_pipeline, X_c, y_c, cv=5, scoring='r2')
    r2_c = r2_scores.mean()
    print(f"Cluster {cluster} - MSE: {mse_c:.2f}, R2: {r2_c:.2f}")


    
#%%
# Conclusion
# Analyzing the correlation directly
for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]
    correlation = cluster_data[['Engagement_Level', 'Improvement_in_Learning_Outcomes']].corr()
    print(f"Cluster {cluster} - Correlation between Engagement Level and Academic Outcomes:\n{correlation}")
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='Engagement_Level', y='Improvement_in_Learning_Outcomes', data=cluster_data)
    plt.title(f'Cluster {cluster}: VR Engagement vs Academic Outcomes')
    plt.xlabel('VR Engagement Level')
    plt.ylabel('Improvement in Learning Outcomes')
    plt.show()
# %%
