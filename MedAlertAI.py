import streamlit as st
import pickle
import pandas as pd

# Load model
# model = pickle.load(open('model.pkl', 'rb'))

# Giao diện Streamlit
st.title("Machine Learning Prediction App")

# Upload file CSV đầu vào
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    # Dự đoán
    #predictions = model.predict(data)
    #st.write("Predictions:")
    #st.write(predictions)
