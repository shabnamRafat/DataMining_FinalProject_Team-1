# Virtual Reality in Education Impact Analysis

#%% Import Libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

#%% ## Step 1: Load Data
# Load the dataset and ensure all necessary columns are present.
data = pd.read_csv('/mnt/data/Virtual_Reality_in_Education_Impact.csv')

# Check required columns
required_columns = ['VR_Engagement', 'Hours_of_Usage', 'Perceived_Creativity', 'Stress_Levels', 'Academic_Outcome']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Dataset must contain the following columns: {required_columns}")

# ## Step 2: Preprocess Data
# Standardize the data to prepare it for clustering.
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['VR_Engagement', 'Hours_of_Usage', 'Perceived_Creativity', 'Stress_Levels']])

# ## Step 3: Perform Clustering
# Use KMeans to group students into clusters based on VR usage patterns.
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# ## Step 4: Analyze Clusters
# Summarize and visualize the identified clusters.
cluster_summary = data.groupby('Cluster').mean()
print("Cluster Summary:")
print(cluster_summary)

# Visualize Clusters
sns.pairplot(data, hue='Cluster', vars=['VR_Engagement', 'Hours_of_Usage', 'Perceived_Creativity', 'Stress_Levels'])
plt.title("Pairwise Distribution of Clusters")
plt.show()

# ## Cluster Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Cluster', palette='viridis')
plt.title("Cluster Distribution")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()

# ## Step 5: Investigate Academic Outcomes per Cluster
# Fit regression models within each cluster to investigate how VR engagement correlates with academic outcomes.
results = {}
for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]
    X = cluster_data[['VR_Engagement', 'Hours_of_Usage', 'Perceived_Creativity', 'Stress_Levels']]
    y = cluster_data['Academic_Outcome']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    results[cluster] = {
        'MSE': mse,
        'R2': r2,
        'Coefficients': dict(zip(X.columns, model.coef_))
    }

    print(f"Cluster {cluster}:
MSE: {mse:.2f}, R2: {r2:.2f}")
    print(f"Coefficients: {results[cluster]['Coefficients']}\n")

    # Visualization of Actual vs Predicted Academic Outcomes
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.7, label=f"Cluster {cluster}")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label="Ideal Fit")
    plt.xlabel("Actual Academic Outcome")
    plt.ylabel("Predicted Academic Outcome")
    plt.title(f"Actual vs Predicted Academic Outcome for Cluster {cluster}")
    plt.legend()
    plt.show()

# ## Step 6: Predict Academic Outcomes for New Data
# Define a function to predict academic outcomes for new student data.
def predict_academic_outcome(new_data):
    new_data_scaled = scaler.transform(new_data[['VR_Engagement', 'Hours_of_Usage', 'Perceived_Creativity', 'Stress_Levels']])
    cluster_labels = kmeans.predict(new_data_scaled)
    predictions = []

    for i, cluster in enumerate(cluster_labels):
        model = LinearRegression()
        cluster_data = data[data['Cluster'] == cluster]
        X = cluster_data[['VR_Engagement', 'Hours_of_Usage', 'Perceived_Creativity', 'Stress_Levels']]
        y = cluster_data['Academic_Outcome']
        model.fit(X, y)
        prediction = model.predict([new_data.iloc[i, :-1].values])[0]
        predictions.append(prediction)

    new_data['Predicted_Academic_Outcome'] = predictions
    return new_data

# Example New Data
new_students = pd.DataFrame({
    'VR_Engagement': [7, 4],
    'Hours_of_Usage': [30, 10],
    'Perceived_Creativity': [8, 5],
    'Stress_Levels': [5, 7]
})

predictions = predict_academic_outcome(new_students)
print("Predicted Academic Outcomes for New Data:")
print(predictions)

# ## Step 7: Additional Exploration
# Explore correlation between VR engagement levels and academic outcomes.
sns.lmplot(x='VR_Engagement', y='Academic_Outcome', hue='Cluster', data=data, aspect=2, height=6)
plt.title("VR Engagement vs Academic Outcome")
plt.show()

# Relationship Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data[['VR_Engagement', 'Hours_of_Usage', 'Perceived_Creativity', 'Stress_Levels', 'Academic_Outcome']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Save results to file
results_df = pd.DataFrame(results).T
results_df.to_csv("cluster_analysis_results.csv", index_label="Cluster")
