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

#%% Load Data
# Reading the dataset into a pandas DataFrame
data = pd.read_csv('Virtual_Reality_in_Education_Features.csv')

#%% Data Preparation: Encode categorical variables
# Identifying categorical variables to be encoded
categorical_features = ['Gender', 'Grade_Level', 'Field_of_Study', 'Subject', 'Region', 'School_Support_for_VR_in_Curriculum']
# Using OneHotEncoder to transform categorical data into binary features
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder='passthrough')
data_encoded = transformer.fit_transform(data)
data_encoded = pd.DataFrame(data_encoded, columns=transformer.get_feature_names_out())

#%% Exploratory Data Analysis (EDA)
# Visualizing the distribution of student ages using a histogram
plt.figure(figsize=(12, 8))
sns.histplot(data['Age'], kde=True, element='step', color='teal')
plt.title('Distribution of Student Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

#%% Correlation Heatmap
# Extracting only numeric data for correlation analysis
numeric_data = data.select_dtypes(include=[np.number])

# Calculating correlation matrix
correlation_matrix = numeric_data.corr()

# Visualizing correlations using heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.show()

#%% Clustering Analysis
# Preprocess binary columns by converting 'Yes'/'No' to 1/0
binary_cols = ['Usage_of_VR_in_Education', 'Access_to_VR_Equipment', 'Collaboration_with_Peers_via_VR', 
               'Feedback_from_Educators_on_VR', 'Interest_in_Continuing_VR_Based_Learning', 
               'School_Support_for_VR_in_Curriculum']
for col in binary_cols:
    data[col] = data[col].map({'Yes': 1, 'No': 0})

# Define preprocessing for numeric columns: impute missing values then scale them
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Imputing missing values with the mean
    ('scaler', StandardScaler())  # Scaling features to standardize data
])

# Define preprocessing for categorical columns: impute missing values then encode them
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Imputing missing categories with 'missing'
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encoding categories
])

# Combining preprocessing steps into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Creating a pipeline that includes preprocessing and clustering
cluster_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('cluster', KMeans(n_clusters=5, random_state=42))  # Using KMeans for clustering
])

# Fitting the clustering model
cluster_pipeline.fit(data)

# Adding cluster labels to the dataset
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
# Replacing categorical 'Yes'/'No' values with numerical 1/0 for all columns
data.replace({'Yes': 1, 'No': 0}, inplace=True)

# Ensuring all data is numeric for model compatibility
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = pd.to_numeric(data[column], errors='coerce')

# Imputing any remaining missing values
imputer = SimpleImputer(strategy='mean')
X = data.drop('Improvement_in_Learning_Outcomes', axis=1)  # Dropping the target variable to isolate features
y = data['Improvement_in_Learning_Outcomes']

X_imputed = imputer.fit_transform(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Training a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}, R2: {r2}')

#%% Prediction Details per Cluster
categorical_features = [col for col in data.columns if data[col].dtype == 'object' and col not in ['Improvement_in_Learning_Outcomes', 'Cluster']]
numerical_features = [col for col in data.columns if data[col].dtype in ['int64', 'float64'] and col not in ['Improvement_in_Learning_Outcomes', 'Cluster']]

# Create transformers for numerical and categorical data
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a Column Transformer to apply transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Prepare the full preprocessor and model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Now loop over each cluster to fit and evaluate the model
for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]
    X_c = cluster_data.drop(['Improvement_in_Learning_Outcomes', 'Cluster'], axis=1)
    y_c = cluster_data['Improvement_in_Learning_Outcomes']
    
    # Split the data into training and test sets
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
    
    # Train the model
    model_pipeline.fit(X_train_c, y_train_c)
    
    # Predict and evaluate the model
    y_pred_c = model_pipeline.predict(X_test_c)
    mse_c = mean_squared_error(y_test_c, y_pred_c)
    r2_c = r2_score(y_test_c, y_pred_c)
    
    print(f"Cluster {cluster} - MSE: {mse_c:.2f}, R2: {r2_c:.2f}")
