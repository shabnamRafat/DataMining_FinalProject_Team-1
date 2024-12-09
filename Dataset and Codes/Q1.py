
# SMART Q1 : How does the instructor's VR proficiency affect students' improvement in learning outcomes?

# Converting 'Improvement_in_Learning_Outcomes' from 'Yes'/'No' to 1/0

data['Improvement_in_Learning_Outcomes'] = data['Improvement_in_Learning_Outcomes'].map({'Yes': 1, 'No': 0})



# Grouping the data by Instructor VR Proficiency and extracting the 'Improvement_in_Learning_Outcomes' and performing One-Way ANOVA

grouped = [data[data['Instructor_VR_Proficiency'] == proficiency]['Improvement_in_Learning_Outcomes'] 
           for proficiency in data['Instructor_VR_Proficiency'].unique()]


f_statistic, p_value = stats.f_oneway(*grouped)

print(f"F-statistic: {f_statistic}, p-value: {p_value}")


# Based on the F-statistic and p-value from the One-Way ANOVA, it seems that Instructor VR Proficiency does not have a statistically significant effect on Improvement in Learning Outcomes.
