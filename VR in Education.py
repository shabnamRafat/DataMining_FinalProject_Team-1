#%%
# Import required libraries
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

#%%
vr_in_education = pd.read_csv('Virtual_Reality_in_Education_Impact.csv')
variables = vr_in_education.columns

#%%
print(vr_in_education.head())
print(variables.values)


# %%[markdown]

##### **Q3: What are the key distinguishing features between high-performing and low-performing clusters?**

##### **Answer:**
# <u>Performance Indicators for clustering: </u></br>
# [ 'Engagement_Level' 'Improvement_in_Learning_Outcomes' 'Perceived_Effectiveness_of_VR' 'Impact_on_Creativity' 'Stress_Level_with_VR_Usage'
# 'Collaboration_with_Peers_via_VR' 'Feedback_from_Educators_on_VR' 'Interest_in_Continuing_VR_Based_Learning'] 
# 
# <u>Distinguishing features:</u></br> 
# ['Age' 'Gender' 'Grade_Level' 'Field_of_Study'] </br>
# ['Usage_of_VR_in_Education' 'Hours_of_VR_Usage_Per_Week' 'Instructor_VR_Proficiency' 
# 'Access_to_VR_Equipment' 'School_Support_for_VR_in_Curriculum']



# %%
