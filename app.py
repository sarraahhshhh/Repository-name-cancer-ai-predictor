import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Cancer Prediction System",
    page_icon="🧬",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("accuracy.pkl", "rb") as f:
    accuracy = pickle.load(f)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# ---------------- SESSION STATE ----------------

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------- SIDEBAR ----------------

st.sidebar.title("🧬 Cancer AI Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Prediction Tool", "Dataset Explorer", "Model Insights", "About Project"]
)

# ---------------- HOME PAGE ----------------

if page == "Home":

    st.title("🧬 AI Powered Cancer Prediction System")

    col1, col2 = st.columns([2,1])

    with col1:

        st.write("""
### Early Detection with Artificial Intelligence

Cancer diagnosis often requires analyzing complex medical data.
Machine learning models can assist doctors by identifying patterns
in biomedical measurements and predicting whether a tumor
is **benign or malignant**.

This dashboard demonstrates how artificial intelligence
can support medical decision making and improve early detection.
""")

        st.metric("Model Accuracy", f"{round(accuracy*100,2)} %")

    with col2:

        st.image(
            "https://images.unsplash.com/photo-1582719478250-c89cae4dc85b",
            caption="AI in Healthcare",
            use_column_width=True
        )

    st.divider()

    st.subheader("Key Features")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("### 🔬 Machine Learning Model")

        st.write("Understand the AI algorithm used in this system.")

        if st.button("Explore Model"):

            st.session_state.page = "ml"

    with c2:

        st.markdown("### 📊 Interactive Data Visualization")

        st.write("Explore how medical data is visualized.")

        if st.button("Explore Visualization"):

            st.session_state.page = "viz"

    with c3:

        st.markdown("### ⚡ Real-time Tumor Prediction")

        st.write("See how predictions are generated.")

        if st.button("Explore Prediction"):

            st.session_state.page = "pred"

# ---------------- FEATURE PAGE : ML MODEL ----------------

if st.session_state.page == "ml":

    st.title("🔬 Machine Learning Model")

    st.write("""
This system uses a **Random Forest Classifier**, a powerful
machine learning algorithm commonly used in medical data analysis.

### What is Random Forest?

Random Forest is an ensemble learning technique that builds
multiple decision trees and combines their predictions
to improve accuracy.

### Why It Works Well for Medical Data

• Handles complex relationships between variables  
• Reduces overfitting  
• Provides feature importance insights  
• Works well with structured biomedical datasets

### Training Process

1. The model is trained on the Breast Cancer Wisconsin dataset.
2. The dataset contains measurements of tumor cell nuclei.
3. The algorithm learns patterns associated with malignant tumors.
4. The trained model predicts tumor type based on new data.
""")

    st.image(
        "https://images.unsplash.com/photo-1555949963-aa79dcee981c",
        caption="Machine Learning Process"
    )

    if st.button("⬅ Back to Home"):

        st.session_state.page = "Home"

# ---------------- FEATURE PAGE : VISUALIZATION ----------------

if st.session_state.page == "viz":

    st.title("📊 Interactive Data Visualization")

    st.write("""
Data visualization helps researchers understand complex datasets.

In this dashboard we visualize:

• Feature distributions  
• Prediction probabilities  
• Model feature importance  

These charts help explain how the machine learning model
interprets tumor characteristics.
""")

    st.bar_chart(df.iloc[:, :10])

    st.image(
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71",
        caption="Healthcare Data Analytics"
    )

    if st.button("⬅ Back to Home"):

        st.session_state.page = "Home"

# ---------------- FEATURE PAGE : PREDICTION ----------------

if st.session_state.page == "pred":

    st.title("⚡ Real-time Tumor Prediction")

    st.write("""
The prediction system analyzes tumor measurements
entered by the user.

### Prediction Workflow

1. User enters tumor measurements
2. Data is converted into numerical features
3. Machine learning model processes the data
4. The system predicts tumor type
5. Probability scores show prediction confidence

This allows quick decision support for medical analysis.
""")

    st.image(
        "https://images.unsplash.com/photo-1581091870627-3c5c6c9f68a1",
        caption="AI Medical Prediction"
    )

    if st.button("⬅ Back to Home"):

        st.session_state.page = "Home"

# ---------------- PREDICTION TOOL ----------------

if page == "Prediction Tool":

    st.title("🔬 Tumor Prediction Tool")

    st.write("Enter tumor measurements to generate prediction.")

    col1, col2 = st.columns(2)

    with col1:

        radius = st.slider("Mean Radius", 5.0, 30.0, 14.0)

        texture = st.slider("Mean Texture", 5.0, 40.0, 19.0)

    with col2:

        perimeter = st.slider("Mean Perimeter", 40.0, 200.0, 90.0)

        area = st.slider("Mean Area", 200.0, 2500.0, 700.0)

    if st.button("Predict Tumor Type"):

        features = np.array([[radius, texture, perimeter, area]])

        prediction = model.predict(features)

        probability = model.predict_proba(features)

        if prediction[0] == 1:

            st.success("Tumor likely BENIGN")

        else:

            st.error("Tumor likely MALIGNANT")

        prob_df = pd.DataFrame({

            "Class": ["Malignant", "Benign"],

            "Probability": probability[0]

        })

        st.bar_chart(prob_df.set_index("Class"))

# ---------------- DATASET EXPLORER ----------------

if page == "Dataset Explorer":

    st.title("📊 Dataset Explorer")

    st.write("""
The model was trained using the **Breast Cancer Wisconsin Dataset**.

The dataset contains measurements of cell nuclei extracted
from breast mass images.
""")

    st.dataframe(df)

    st.subheader("Feature Distribution")

    st.bar_chart(df.iloc[:, :10])

# ---------------- MODEL INSIGHTS ----------------

if page == "Model Insights":

    st.title("📈 Model Insights")

    st.metric("Model Accuracy", f"{round(accuracy*100,2)} %")

    importance = pd.DataFrame({

        "Feature": [
            "Mean Radius",
            "Mean Texture",
            "Mean Perimeter",
            "Mean Area"
        ],

        "Importance": model.feature_importances_
    })

    st.bar_chart(importance.set_index("Feature"))

# ---------------- ABOUT PROJECT ----------------

if page == "About Project":

    st.title("📚 About This Project")

    st.write("""
### Overview

This project demonstrates how machine learning can be used
to support early cancer detection.

### Technologies Used

• Python  
• Streamlit  
• Scikit-learn  
• Pandas  
• NumPy  

### Future Improvements

• Explainable AI  
• Larger medical datasets  
• Clinical decision support systems
""")