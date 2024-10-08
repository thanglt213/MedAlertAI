import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

# Giao diện Streamlit
st.title("Machine Learning Prediction App with Isolation Forest")

# Upload file CSV cho dữ liệu huấn luyện
train_file = st.file_uploader("Upload your training data CSV file", type=["csv"], key='train')

# Upload file CSV cho dữ liệu dự đoán
uploaded_file = st.file_uploader("Upload your input data CSV file", type=["csv"], key='data')

if train_file and uploaded_file:
    # Đọc dữ liệu huấn luyện
    train_data = pd.read_csv(train_file)
    st.write("Dữ liệu huấn luyện:", train_data.shape)
    # Xóa dữ liệu NaN
    train_data = train_data.dropna()
    st.write(train_data.head())

    # Lưu số dòng của dữ liệu huấn luyện
    num_train_rows = train_data.shape[0]

    # Đọc dữ liệu dự đoán
    data = pd.read_csv(uploaded_file)
    st.write("Dữ liệu dự đoán:", data.shape)
    # Xóa dữ liệu NaN
    data = data.dropna()
    st.write(data.head())

    # Thêm cột 'is_train' để đánh dấu tập dữ liệu huấn luyện và dự đoán
    train_data['is_train'] = 1
    data['is_train'] = 0

    # Gộp train_data và data
    combined_data = pd.concat([train_data, data], ignore_index=True)

    # Chuyển tất cả các cột thành kiểu chuỗi
    for col in combined_data.columns:
        combined_data[col] = combined_data[col].astype(str)

    # Chuyển đổi các trường cụ thể thành kiểu số
    numeric_columns = ['days_to_report', 'requested_amount_per_day']
    combined_data[numeric_columns] = combined_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Mã hóa các cột phân loại trong combined_data
    label_encoders = {}
    for column in combined_data.columns:
        if combined_data[column].dtype == 'object':
            le = LabelEncoder()
            try:
                combined_data[column] = le.fit_transform(combined_data[column])
                label_encoders[column] = le
            except TypeError as e:
                st.error(f"Lỗi khi mã hóa cột {column}: {e}")

    # Tách lại dữ liệu huấn luyện và dữ liệu dự đoán dựa trên số dòng đã lưu
    train_data_encoded = combined_data.iloc[:num_train_rows].drop(columns=['is_train'])
    data_encoded = combined_data.iloc[num_train_rows:].drop(columns=['is_train'])

    # Hiển thị dữ liệu sau mã hóa và chuẩn hóa
    st.write("Dữ liệu sau mã hóa và chuẩn hóa:",data_encoded.head())

    # Khởi tạo mô hình Isolation Forest
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

    # Lấy dữ liệu để huấn luyện
    train_data_encoded = train_data_encoded.select_dtypes(include=[np.number])
    # Huấn luyện mô hình
    model.fit(train_data_encoded)
    st.write("Mô hình đã được huấn luyện thành công.")

    # Dự đoán với dữ liệu mới
    st.write("Dự đoán:")
    predictions = model.predict(data_encoded)

    # Update kết quả dự đoán vào dữ liệu gốc
    result_df = pd.DataFrame({'Prediction': predictions})
    result_df['Prediction'] = result_df['Prediction'].replace({1: 'Bình thường', -1: 'Bất thường'})
    data['Prediction'] = result_df['Prediction']
    
    # Hiển thị DataFrame 
    st.dataframe(data)
    
    # Nút tải xuống file CSV kết quả
    csv = data.to_csv(index=False)
    st.download_button("Tải xuống kết quả dự đoán", csv, "predictions.csv", "text/csv", key="download")

else:
    st.warning("Vui lòng tải lên cả hai tệp dữ liệu huấn luyện và dữ liệu dự đoán.")

