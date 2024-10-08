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
    st.write("Dữ liệu huấn luyện:")
    st.write(train_data.head())
    st.write("Shape của dữ liệu huấn luyện:", train_data.shape)

    # Lưu số dòng của dữ liệu huấn luyện
    num_train_rows = train_data.shape[0]

    # Đọc dữ liệu dự đoán
    data = pd.read_csv(uploaded_file)
    st.write("Dữ liệu dự đoán:")
    st.write(data.head())
    st.write("Shape của dữ liệu dự đoán:", data.shape)

    # Thêm cột 'is_train' để đánh dấu tập dữ liệu huấn luyện và dự đoán
    train_data['is_train'] = 1
    data['is_train'] = 0

    # Gộp train_data và data
    combined_data = pd.concat([train_data, data], ignore_index=True)
    st.write("Shape của combined_data sau khi gộp:", combined_data.shape)

    # Hiển thị dữ liệu sau khi gộp
    st.write("Dữ liệu sau khi gộp:")
    st.write(combined_data.head())

    # Kiểm tra các cột rỗng trước khi xử lý NaN
    st.write("Số lượng giá trị NaN trong từng cột trước khi xử lý:")
    nan_counts = combined_data.isnull().sum()
    st.write(nan_counts[nan_counts > 0])  # Hiển thị các cột có giá trị NaN

    # Chuyển tất cả các cột thành kiểu chuỗi
    for col in combined_data.columns:
        combined_data[col] = combined_data[col].astype(str)

    # Chuyển đổi các trường cụ thể thành kiểu số
    numeric_columns = ['group', 'days_to_report', 'requested_amount_per_day']
    combined_data[numeric_columns] = combined_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Hiển thị dữ liệu sau chuyển đổi kiểu
    st.write("Dữ liệu sau chuyển đổi kiểu:")
    st.write(combined_data.head())


    
    '''
    combined_data = combined_data.dropna()
    st.write("Shape của combined_data sau khi xử lý NaN - dòng 70:", combined_data.shape)

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

    # Kiểm tra hình dạng dữ liệu huấn luyện
    st.write("Shape của train_data_encoded - dòng 86:", train_data_encoded.shape)

    if train_data_encoded.shape[0] == 0:
        st.error("Dữ liệu huấn luyện trống. Vui lòng kiểm tra lại dữ liệu đầu vào. - dòng 89")
    else:
        # Khởi tạo mô hình Isolation Forest
        model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

        # Kiểm tra dữ liệu huấn luyện
        if train_data_encoded.isnull().values.any():
            st.error("Dữ liệu huấn luyện chứa giá trị NaN. Vui lòng xử lý trước khi huấn luyện mô hình. - dòng 96")
        else:
            # Ensure all data is numeric
            train_data_encoded = train_data_encoded.select_dtypes(include=[np.number])

            # Attempt to fit the model
            try:
                model.fit(train_data_encoded)
                st.write("Mô hình đã được huấn luyện thành công.")
            except ValueError as e:
                st.error(f"Lỗi khi huấn luyện mô hình: {e}")

    # Dự đoán với dữ liệu mới
    predictions = model.predict(data_encoded)

    # Hiển thị kết quả dự đoán
    st.write("Kết quả dự đoán:")
    result_df = pd.DataFrame({'Prediction': predictions})
    result_df['Prediction'] = result_df['Prediction'].replace({1: 'Normal', -1: 'Anomaly'})
    
    # Hiển thị bảng kết quả
    st.write(result_df)

    # Nút tải xuống file CSV kết quả
    csv = result_df.to_csv(index=False)
    st.download_button("Tải xuống kết quả dự đoán", csv, "predictions.csv", "text/csv", key="download")

else:
    st.warning("Vui lòng tải lên cả hai tệp dữ liệu huấn luyện và dữ liệu dự đoán.")
'''
