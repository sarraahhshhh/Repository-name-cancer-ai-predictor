import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

st.set_page_config(
    page_title="AI Cancer Predictor",
    page_icon="🧬",
    layout="wide"
)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Prediction", "Dataset Explorer", "About"]
)

# -------------------
# ABOUT PAGE
# -------------------

if page == "About":

    st.title("🧬 AI Cancer Prediction System")

    st.write("""
This application uses machine learning to predict whether a tumor
is **benign or malignant** using biomedical features.

Technologies used:

• Python  
• Scikit-learn  
• Streamlit  
• Machine Learning
""")

# -------------------
# DATASET PAGE
# -------------------

elif page == "Dataset Explorer":

    st.title("Cancer Dataset Explorer")

    st.write("Preview of the dataset used for training the model.")

    st.dataframe(df)

    st.subheader("Feature Distribution")

    st.bar_chart(df.iloc[:, :10])

# -------------------
# PREDICTION PAGE
# -------------------

else:

    st.title("🧬 Cancer Prediction Dashboard")

    st.write("Enter tumor measurements to predict cancer type.")

    col1, col2 = st.columns(2)

    with col1:
        radius = st.number_input("Mean Radius")
        texture = st.number_input("Mean Texture")

    with col2:
        perimeter = st.number_input("Mean Perimeter")
        area = st.number_input("Mean Area")

    if st.button("Predict"):

        features = np.array([[radius, texture, perimeter, area]])

        prediction = model.predict(features)
        probability = model.predict_proba(features)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.success("Tumor likely BENIGN ✅")
        else:
            st.error("Tumor likely MALIGNANT ⚠️")

        st.subheader("Prediction Confidence")

        prob_df = pd.DataFrame({
            "Class": ["Malignant", "Benign"],
            "Probability": probability[0]
        })

        st.bar_chart(prob_df.set_index("Class"))