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
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # This will only run if my_module.py is run directly
    print("All the relevant files have been imported!")


# SMART Question 1 : How does the instructor's VR proficiency affect students' improvement in learning outcomes?

# Converting 'Improvement_in_Learning_Outcomes' from 'Yes'/'No' to 1/0

vr_in_education = EDA_VR.vr_in_education


vr_in_education['Improvement_in_Learning_Outcomes'] = vr_in_education['Improvement_in_Learning_Outcomes'].map({'Yes': 1, 'No': 0})

# Grouping the data by Instructor VR Proficiency and extracting the 'Improvement_in_Learning_Outcomes' and performing One-Way ANOVA

grouped = [vr_in_education[vr_in_education['Instructor_VR_Proficiency'] == proficiency]['Improvement_in_Learning_Outcomes'] 
           for proficiency in vr_in_education['Instructor_VR_Proficiency'].unique()]


f_statistic, p_value = f_oneway(*grouped)

print(f"F-statistic: {f_statistic}, p-value: {p_value}")


# Based on the F-statistic and p-value from the One-Way ANOVA, it seems that Instructor VR Proficiency does not have a statistically significant effect on Improvement in Learning Outcomes.

# %%
