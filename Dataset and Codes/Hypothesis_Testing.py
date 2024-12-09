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

# %%

vr_in_education = EDA_VR.vr_in_education_copy

print(tabulate(vr_in_education.head(5), headers='keys', tablefmt='pretty'))


#%%

# Declare another class for hypothesis testing

class StatisticalTests:
    def __init__(self, dataframe):
        """Initialize with the dataframe."""
        self.data = dataframe

    def chi_square_test(self, var1, var2):
        """Performs a Chi-Square test of independence between two categorical variables."""
        h0 = "There is no association between " + var1 + " and " + var2
        h1 = "There is association between " + var1 + " and " + var2
        print(f"Null Hypothesis(H0): {h0}")
        print(f"Alternative Hypothesis(H1): {h1}")
        contingency_table = pd.crosstab(self.data[var1], self.data[var2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        results = {
            'chi2': chi2,
            'p-value': p,
            'dof': dof,
            'expected': expected
        }
        if results['p-value'] < 0.05:
            test_result = test_result = f"Since, p-value {results['p-value']} is less than 0.05, we reject Null Hypothesis(H0). There is an association between {var1} and {var2}."
            
        else:
            test_result = test_result = f"Since, p-value {results['p-value']} is greater than 0.05, we fail to reject Null Hypothesis(H0). There is no association between {var1} and {var2}."
        
        return test_result
    

    def anova_test(self, var, groups):
        """Performs an ANOVA test to compare the means of a numerical variable across different groups."""
        h0 = f"There is no significant difference in the mean of {var} across the groups defined by {groups}."
        h1 = f"There is a significant difference in the mean of {var} across the groups defined by {groups}."
        print(f"Null Hypothesis(H0): {h0}")
        print(f"Alternative Hypothesis(H1): {h1}")

        # Ensure the variable is numerical
        if not pd.api.types.is_numeric_dtype(self.data[var]):
            raise ValueError(f"The variable '{var}' is not numerical.")

        # Ensure the groups variable is categorical
        if not pd.api.types.is_categorical_dtype(self.data[groups]) and not pd.api.types.is_object_dtype(self.data[groups]):
            raise ValueError(f"The variable '{groups}' is not categorical.")

        grouped_data = [self.data[var][self.data[groups] == group] for group in self.data[groups].unique()]
        f_stat, p_value = f_oneway(*grouped_data)

        results = {
            'F-statistic': f_stat,
            'p-value': p_value
        }

        if results['p-value'] < 0.05:
            test_result = f"Since, p-value {results['p-value']} is less than 0.05, we reject Null Hypothesis(H0). There is a significant difference in the mean of {var} across the groups defined by {groups}."
        else:
            test_result = f"Since, p-value {results['p-value']} is greater than 0.05, we fail to reject Null Hypothesis(H0). There is no significant difference in the mean of {var} across the groups defined by {groups}."

        return test_result


    def logistic_regression(self, independent_var, dependent_var, test_size=0.3, random_state=42):
        """
        Perform logistic regression using statsmodels and sklearn, 
        and return model evaluation based on p-value and accuracy.
        """
        # Convert dependent variable to binary if needed
        if self.data[dependent_var].dtype == 'object':
            self.data[dependent_var] = self.data[dependent_var].map({'Yes': 1, 'No': 0})

        # Independent and dependent variables
        X = self.data[[independent_var]]
        y = self.data[dependent_var]

        # Add constant for intercept
        X = sm.add_constant(X)

        # Logistic regression using statsmodels
        model = sm.Logit(y, X)
        result = model.fit(disp=0)

        # Evaluate based on p-value
        p_value = result.pvalues[independent_var]
        if p_value < 0.05:
            sm_result = f"Statistical significance: p-value ({p_value}) < 0.05. {independent_var} significantly impacts {dependent_var}."
        else:
            sm_result = f"Not statistically significant: p-value ({p_value}) ≥ 0.05. {independent_var} does not significantly impact {dependent_var}."

        # Logistic regression using sklearn
        X_train, X_test, y_train, y_test = train_test_split(
            self.data[[independent_var]], y, test_size=test_size, random_state=random_state
        )
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        sklearn_result = f"Logistic Regression accuracy: {accuracy:.2f}."

        return {"statsmodels_result": sm_result, "sklearn_result": sklearn_result}

    def linear_regression(self, independent_var, dependent_var, test_size=0.3, random_state=42):
        """
        Perform linear regression using statsmodels and sklearn, 
        and return model evaluation based on p-value, R^2, and MSE.
        """
        # Independent and dependent variables
        X = self.data[[independent_var]]
        y = self.data[dependent_var]

        # Add constant for intercept
        X = sm.add_constant(X)

        # Linear regression using statsmodels
        model = sm.OLS(y, X)
        result = model.fit()

        # Evaluate based on p-value
        p_value = result.pvalues[independent_var]
        if p_value < 0.05:
            sm_result = f"Statistical significance: p-value ({p_value:.4f}) < 0.05.{independent_var} significantly impacts {dependent_var}."
        else:
            sm_result = f"Not statistically significant: p-value ({p_value:.4f}) ≥ 0.05.{independent_var} does not significantly impact {dependent_var}."

        # Linear regression using sklearn
        X_train, X_test, y_train, y_test = train_test_split(
            self.data[[independent_var]], y, test_size=test_size, random_state=random_state
        )
        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        y_pred = linreg.predict(X_test)

        # Evaluate performance
        r_squared = linreg.score(X_test, y_test)  # R^2 score
        mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error

        sklearn_result = (
            f"Linear Regression performance:\n"
            f"  - R^2 score: {r_squared:.4f}\n"
            f"  - Mean Squared Error (MSE): {mse:.4f}"
        )

        return {"statsmodels_result": sm_result, "sklearn_result": sklearn_result}
    


# %%
vr_in_education = pd.read_csv('Virtual_Reality_in_Education_Impact.csv')
stats_vr_in_education = StatisticalTests(vr_in_education)


#%%

print(f"\n\033[4mChi-Square Test for Gender vs. Usage_of_VR_in_Education:\033[0m\n")
chi_square_results_vr_usage_vs_gender = stats_vr_in_education.chi_square_test('Gender', 'Usage_of_VR_in_Education')
print(f"\nResult:\n{chi_square_results_vr_usage_vs_gender}")


print(f"\n\033[4mAnova Test for Gender vs. Hours_of_VR_Usage_Per_Week:\033[0m\n")
anova_results_gender_vs_vr_hours = stats_vr_in_education.anova_test('Hours_of_VR_Usage_Per_Week', 'Gender')
print(f"\nResult:\n{anova_results_gender_vs_vr_hours}")



# %%

#Relationship between VR Usage and Subject

print(f"\n\033[4mChi-Square Test for Subject vs. Usage_of_VR_in_Education:\033[0m\n")
chi_square_results_vr_usage_vs_subject = stats_vr_in_education.chi_square_test('Subject', 'Usage_of_VR_in_Education')
print(f"\nResult:\n{chi_square_results_vr_usage_vs_subject}")


print(f"\n\033[4mAnova Test for Subject vs. Hours_of_VR_Usage_Per_Week:\033[0m\n")
anova_results_subject_vs_vr_hours = stats_vr_in_education.anova_test('Hours_of_VR_Usage_Per_Week', 'Subject')
print(f"\nResult:\n{anova_results_subject_vs_vr_hours}")


# %%
# Relationship between VR Usage and Academic Outcome

print(f"\n\033[4mChi-Square Test for Outcome vs. Usage_of_VR_in_Education:\033[0m\n")
chi_square_results_vr_usage_vs_outcome = stats_vr_in_education.chi_square_test('Improvement_in_Learning_Outcomes', 'Usage_of_VR_in_Education')
print(f"\nResult:\n{chi_square_results_vr_usage_vs_outcome}")


print(f"\n\033[4mLogistic Regression Test for Outcome vs. Hours_of_VR_Usage_Per_Week:\033[0m\n")
anova_results_vr_hours_vs_outcocme = stats_vr_in_education.logistic_regression('Hours_of_VR_Usage_Per_Week', 'Improvement_in_Learning_Outcomes')
print(f"\nResult:\n{anova_results_vr_hours_vs_outcocme}")

# %%
# Relationship between VR Usage and Engagement Level

print(f"\n\033[4mChi-Square Test for Engagement Level vs. Usage_of_VR_in_Education:\033[0m\n")
chi_square_results_vr_usage_vs_engagement = stats_vr_in_education.chi_square_test('Engagement_Level', 'Usage_of_VR_in_Education')
print(f"\nResult:\n{chi_square_results_vr_usage_vs_engagement}")



print(f"\n\033[4mLogistic Regression Test for Engagement Level vs. Hours_of_VR_Usage_Per_Week:\033[0m\n")
anova_results_vr_hours_vs_engagement = stats_vr_in_education.linear_regression('Hours_of_VR_Usage_Per_Week', 'Engagement_Level')
print(f"\nResult:\n{anova_results_vr_hours_vs_engagement}")

# %%
# Relationship between Instructor VR Efficiency and VR Usage

print(f"\n\033[4mChi-Square Test for Intructor VR Efficiency vs. Perceived Efficiency:\033[0m\n")
chi_square_results_intructor_vs_efficiency = stats_vr_in_education.chi_square_test('Instructor_VR_Proficiency', 'Perceived_Effectiveness_of_VR')
print(f"\nResult:\n{chi_square_results_intructor_vs_efficiency}")


print(f"\n\033[4mChi-Square Test for Intructor VR Efficiency vs. VR-based Learning Continuation:\033[0m\n")
chi_square_results_intructor_vs_continuation = stats_vr_in_education.chi_square_test('Instructor_VR_Proficiency', 'Interest_in_Continuing_VR_Based_Learning')
print(f"\nResult:\n{chi_square_results_intructor_vs_continuation}")

# %%