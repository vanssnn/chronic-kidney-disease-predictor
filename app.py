# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle, json

# Set page config
st.set_page_config(
    page_title="Chronic Kidney Disease Predictor",
    page_icon="👨‍⚕️",
    layout="wide"
)

# Set title and description
st.title('👨‍⚕️Chronic Kidney Disease Predictor')
st.markdown("**Chronic Kidney Disease (CKD)** is a longstanding disease of the kidneys that can lead to renal failure. Symptoms may develop slowly or not at all and aren't specific to the disease. This web app predicts whether a patient has **Chronic Kidney Disease (CKD)** based on their data.")
st.markdown('Check out the [GitHub repository](%s) for the report, exploratory data analysis (EDA), model analysis, and the code for this web app.' % 'https://github.com/vanssnn/chronic-kidney-disease-predictor')
total_features = 24

# Initialize the omitted features
if 'omit_feat' not in st.session_state:
    st.session_state.omit_feat = []
    st.session_state.omit_feat_mat = np.zeros(total_features, dtype=bool)

# Disable widgets if the feature is omitted function
def disable_widgets():
    st.session_state.omit_feat_mat = np.zeros(total_features, dtype=bool)
    indices = [labels.index(item) for item in st.session_state.omit_feat if item in labels]
    st.session_state.omit_feat_mat[indices] = True

# Get the column information
column_info = {}
with open('./assets/column_info.json', 'r') as file:
    column_info = json.load(file)

labels = column_info['full']

# Initialize the DataFrame
X = pd.DataFrame(np.empty((1, total_features)), columns=labels)

st.header("👇Input the Patient's Data")

# Omit features widgets
omit_feat = st.multiselect(
    "Select the features you don't know", labels, 
    placeholder="Ommited Features ex. Potassium (i don't know the potassium level).",
    key="omit_feat", on_change=disable_widgets
)

# Info message if the user omits features
with st.empty():
    if len(st.session_state.omit_feat) > 0:
        st.info(f"The model can predict omitted features, bearing in mind that the accuracy may vary.", icon='📖')

# Input widgets to get the patient's data
with st.form("my_form"):
    cols = st.columns(4)
    with cols[0]:
        X[labels[0]] = st.slider(labels[0], min_value=0, max_value=100, value=20, disabled=st.session_state.omit_feat_mat[0])
        X[labels[1]] = st.slider(labels[1], min_value=0, max_value=200, value=80, disabled=st.session_state.omit_feat_mat[1])
        X[labels[2]] = st.select_slider(labels[2], options=[1.005, 1.010, 1.015, 1.020, 1.025], value=1.015, disabled=st.session_state.omit_feat_mat[2])
        X[labels[3]] = st.select_slider(labels[3], options=[0, 1, 2, 3, 4, 5], value=0, disabled=st.session_state.omit_feat_mat[3])
        X[labels[4]] = st.select_slider(labels[4], options=[0, 1, 2, 3, 4, 5], value=0, disabled=st.session_state.omit_feat_mat[4])

    with cols[1]:
        X[labels[5]] = st.selectbox(labels[5], ('Normal', 'Abnormal'), disabled=st.session_state.omit_feat_mat[5])
        X[labels[6]] = st.selectbox(labels[6], ('Normal', 'Abnormal'), disabled=st.session_state.omit_feat_mat[6])
        X[labels[7]] = st.selectbox(labels[7], ('Not Present', 'Present'), disabled=st.session_state.omit_feat_mat[7])
        X[labels[8]] = st.selectbox(labels[8], ('Not Present', 'Present'), disabled=st.session_state.omit_feat_mat[8])
        X[labels[9]] = st.number_input(labels[9], min_value=0, max_value=500, value=70, disabled=st.session_state.omit_feat_mat[9])
        X[labels[10]] = st.number_input(labels[10], min_value=0, max_value=400, value=36, disabled=st.session_state.omit_feat_mat[10])
        X[labels[11]] = st.number_input(labels[11], min_value=0.0, max_value=80.0, value=1.0, step=0.1, disabled=st.session_state.omit_feat_mat[11])

    with cols[2]:
        X[labels[12]] = st.number_input(labels[12], min_value=0.0, max_value=180.0, value=150.0, step=0.5, disabled=st.session_state.omit_feat_mat[12])
        X[labels[13]] = st.number_input(labels[13], min_value=0.0, max_value=50.0, value=4.6, step=0.1, disabled=st.session_state.omit_feat_mat[13])
        X[labels[14]] = st.number_input(labels[14], min_value=0.0, max_value=20.0, value=17.0, step=0.1, disabled=st.session_state.omit_feat_mat[14])
        X[labels[15]] = st.slider(labels[15], min_value=0, max_value=60, value=52, disabled=st.session_state.omit_feat_mat[15])
        X[labels[16]] = st.number_input(labels[16], min_value=2000, max_value=26400, value=9800, step=100, disabled=st.session_state.omit_feat_mat[16])
        X[labels[17]] = st.slider(labels[17], min_value=2.0, max_value=10.0, value=5.0, step=0.1, disabled=st.session_state.omit_feat_mat[17])

    with cols[3]:
        X[labels[18]] = st.selectbox(labels[18], ('No', 'Yes'), disabled=st.session_state.omit_feat_mat[18])
        X[labels[19]] = st.selectbox(labels[19], ('No', 'Yes'), disabled=st.session_state.omit_feat_mat[19])
        X[labels[20]] = st.selectbox(labels[20], ('No', 'Yes'), disabled=st.session_state.omit_feat_mat[20])
        X[labels[21]] = st.selectbox(labels[21], ('Good', 'Poor'), disabled=st.session_state.omit_feat_mat[21])
        X[labels[22]] = st.selectbox(labels[22], ('No', 'Yes'), disabled=st.session_state.omit_feat_mat[22])
        X[labels[23]] = st.selectbox(labels[23], ('No', 'Yes'), disabled=st.session_state.omit_feat_mat[23])
    
    predict_btn = st.form_submit_button("Predict")


# Set the omitted features to null
X[st.session_state.omit_feat] = np.nan

# Initialize the processed DataFrame
X_proc = X.copy()

# Rename the columns to the abbreviated names so it can be processed
cols = column_info['abbrev']
rename_dict = {labels[i]: cols[i] for i in range(len(labels))}
X_proc.rename(columns=rename_dict, inplace=True)

# Lower all the input strings and remove spaces
X_proc = X_proc.map(lambda s: s.lower().replace(' ', '') if type(s) == str else s)

# Load the assets for the model
with open('./assets/cat_imputer.pickle', 'rb') as file:
    cat_imputer = pickle.load(file)

with open('./assets/encoder.pickle', 'rb') as file:
    encoder = pickle.load(file)

with open('./assets/cont_imputer.pickle', 'rb') as file:
    cont_imputer = pickle.load(file)

with open('./assets/scaler.pickle', 'rb') as file:
    scaler = pickle.load(file)

with open('./assets/feat_extraction.pickle', 'rb') as file:
    feat_extraction = pickle.load(file)

with open('./assets/model.pickle', 'rb') as file:
    model = pickle.load(file)

# Process the input data
X_proc[column_info['cat_imputer']] = cat_imputer.transform(X_proc[column_info['cat_imputer']])
X_proc[column_info['encoder']] = encoder.transform(X_proc[column_info['encoder']])
X_proc = cont_imputer.transform(X_proc)
X_proc = pd.DataFrame(X_proc, columns=column_info['abbrev'])

X_proc[column_info['scaler']] = scaler.transform(X_proc[column_info['scaler']])
X_proc = feat_extraction.transform(X_proc)

# Predict the output
[y_pred] = model.predict(X_proc)

# Display the prediction
if predict_btn:
      
    st.header("🎯Prediction")

    if y_pred == 1:
        st.error("The patient has Chronic Kidney Disease (CKD).", icon='🩺')
    else:
        st.balloons()
        st.success("The patient does **NOT** have Chronic Kidney Disease (CKD).", icon='🩺')