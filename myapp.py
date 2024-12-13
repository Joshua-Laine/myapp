


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression  # Corrected import
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(page_title="LinkedIn Usage Prediction", page_icon=":guardsman:", layout="centered")

# Read the data into a DataFrame
s = pd.read_csv('social_media_usage.csv')

def clean_sm(x):
    return np.where(x == 1, 1, 0)


s['sm_li'] = clean_sm(s['web1h'])


s['income'] = s['income'].apply(lambda x: x if x <= 9 else np.nan)


s['education'] = s['educ2'].apply(lambda x: x if x <= 8 else np.nan)


s['parent'] = s['par'].apply(lambda x: x if x in [1, 2] else np.nan)
s['married'] = s['marital'].apply(lambda x: x if x in [1, 2, 3, 4, 5, 6] else np.nan)
s['female'] = s['gender'].apply(lambda x: 1 if x == 2 else (0 if x == 1 else np.nan))

# Clean 'age': Values 97 and 98 should be treated as missing
s['age'] = s['age'].apply(lambda x: x if x not in [97, 98] else np.nan)
ss = s[['sm_li', 'income', 'education', 'parent', 'married', 'female', 'age']].dropna()

y = ss['sm_li']
X = ss[['income', 'education', 'parent', 'married', 'female', 'age']]


# Step 1: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Instantiate the Logistic Regression model with class_weight='balanced'
model = LogisticRegression(class_weight='balanced', random_state=42)

# Step 3: Fit the model with the training data
model.fit(X_train, y_train)

# Step 4: Make predictions on the test data
y_pred = model.predict(X_test)

import numpy as np

# Features for the first person
new_data_1 = np.array([[8, 7, 0, 1, 1, 42]])

# Features for the second person
new_data_2 = np.array([[8, 7, 0, 1, 1, 82]])

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define a function to clean and process the input
def clean_sm(x):
    return np.where(x == 1, 1, 0)


st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f9;
    }
    .stButton>button {
        background-color: #FF6347;
        color: white;
        font-size: 16px;
    }
    .stTitle {
        color: #32CD32;
    }
    .stMarkdown {
        color: #8A2BE2;
    }
    .stWrite {
        color: #FF4500;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("LinkedIn Usage Prediction")
st.write("Please enter the following details:")

# User inputs with detailed definitions for each variable

# Income (using the provided definitions)
income = st.selectbox('Income (1-9)', 
    options=[
        1, 2, 3, 4, 5, 6, 7, 8, 9, 98, 99
    ], 
    format_func=lambda x: {
        1: 'Less than $10,000',
        2: '10 to under $20,000',
        3: '20 to under $30,000',
        4: '30 to under $40,000',
        5: '40 to under $50,000',
        6: '50 to under $75,000',
        7: '75 to under $100,000',
        8: '100 to under $150,000',
        9: '$150,000 or more',
        98: "Don't know",
        99: "Refused"
    }[x], index=4)

# Education Level (using the provided definitions)
education = st.selectbox('Education Level (1-8)', 
    options=[1, 2, 3, 4, 5, 6, 7, 8, 98, 99], 
    format_func=lambda x: {
        1: 'Less than high school',
        2: 'High school incomplete',
        3: 'High school graduate',
        4: 'Some college, no degree',
        5: 'Two-year associate degree',
        6: 'Four-year degree/Bachelor’s degree',
        7: 'Some postgraduate schooling',
        8: 'Postgraduate degree (e.g., PhD)',
        98: 'Don’t know',
        99: 'Refused'
    }[x], index=5)

# Parent Status (1 = Yes, 2 = No)
parent = st.selectbox('Are you a parent?', options=[1, 2, 8, 9], 
    format_func=lambda x: {
        1: 'Yes',
        2: 'No',
        8: "Don't know",
        9: "Refused"
    }[x], index=0)

# Marital Status (1-6 with definitions)
married = st.selectbox('Marital Status (1-6)', 
    options=[1, 2, 3, 4, 5, 6, 8, 9], 
    format_func=lambda x: {
        1: 'Married',
        2: 'Living with a partner',
        3: 'Divorced',
        4: 'Separated',
        5: 'Widowed',
        6: 'Never been married',
        8: 'Don’t know',
        9: 'Refused'
    }[x], index=0)

# Gender (1 = male, 2 = female, 3 = other)
female = st.selectbox('Gender', options=[1, 2, 3, 98, 99], 
    format_func=lambda x: {
        1: 'Male',
        2: 'Female',
        3: 'Other',
        98: "Don't know",
        99: "Refused"
    }[x], index=1)

# Age (numerical input, cleaning values like 97+)
age = st.slider('Age', min_value=18, max_value=100, value=30)

# Data preprocessing (same as earlier, matching input types)
user_data = np.array([[income, education, parent, married, female, age]])

# Make predictions
probability = model.predict_proba(user_data)[0][1]  # Probability of using LinkedIn (1)
prediction = model.predict(user_data)[0]  # 1 if LinkedIn user, 0 otherwise

# Show results
st.write(f"### Prediction: **{'LinkedIn User' if prediction == 1 else 'Non-User'}**", unsafe_allow_html=True)
st.write(f"### Probability of LinkedIn Usage: **{probability:.2f}**", unsafe_allow_html=True)

prediction_data = ['LinkedIn User', 'Non-User']
prediction_count = [probability, 1 - probability]

fig, ax = plt.subplots()
ax.bar(prediction_data, prediction_count, color=['#32CD32', '#FF6347'])
ax.set_title('LinkedIn Usage Prediction Distribution', fontsize=14)
ax.set_ylabel('Probability')

# Show the plot in the Streamlit app
st.pyplot(fig)