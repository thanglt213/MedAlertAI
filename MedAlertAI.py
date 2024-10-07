import streamlit as st
import pickle
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import os

# Giao diện Streamlit
st.title("Machine Learning Prediction App with Isolation Forest")

# Upload file CSV cho dữ liệu huấn luyện
train_file = st.file_uploader("Upload your training data CSV file", type=["csv"], key='train')

# Upload file CSV cho dữ liệu dự đoán
uploaded_file = st.file_uploader("Upload your input data CSV file", type=["csv"], key='data')

# Nút để xóa file mô hình
if st.button("Xóa file mô hình"):
    model_file = 'model.pkl'
    if os.path.exists(model_file):
        os.remove(model_file)
        st.success(f"File {model_file} đã được xóa.")
    else:
        st.warning(f"File {model_file} không tồn tại.")

if train_file and uploaded_file:
    # Đọc dữ liệu huấn luyện
    train_data = pd.read_csv(train_file)
    st.write("Dữ liệu huấn luyện:")
    st.write(train_data)

    # Đọc dữ liệu dự đoán
    data = pd.read_csv(uploaded_file)
    st.write("Dữ liệu dự đoán:")
    st.write(data)

    # Thêm cột 'is_train' để đánh dấu tập dữ liệu huấn luyện và dự đoán
    train_data['is_train'] = 1
    data['is_train'] = 0

    # Gộp train_data và data
    combined_data = pd.concat([train_data, data], ignore_index=True)

    # Kiểm tra và xử lý giá trị NaN
    combined_data.fillna('missing', inplace=True)

    # Chuyển tất cả các cột thành kiểu chuỗi
    for column in combined_data.columns:
        combined_data[column] = combined_data[column].astype(str)

    # Chuyển đổi các trường cụ thể thành kiểu số
    numeric_columns = ['group', 'days_to_report', 'requested_amount_per_day']
    combined_data[numeric_columns] = combined_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Xóa các cột có NaN sau khi chuyển đổi (nếu có)
    combined_data = combined_data.dropna()

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

    # Tách lại dữ liệu huấn luyện và dữ liệu dự đoán
    train_data_encoded = combined_data[combined_data['is_train'] == 1].drop(columns=['is_train'])
    data_encoded = combined_data[combined_data['is_train'] == 0].drop(columns=['is_train'])

    # Khởi tạo mô hình Isolation Forest
    model_file = 'model.pkl'

    if os.path.exists(model_file):
        # Nếu mô hình đã tồn tại, load mô hình từ file
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
        st.write("Mô hình đã được tải từ file.")
    else:
        # Nếu mô hình chưa tồn tại, huấn luyện mô hình
        model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        
        # Huấn luyện mô hình chỉ với dữ liệu đầu vào
        model.fit(train_data_encoded)

        # Lưu mô hình lại
        with open(model_file, 'wb') as file:
            pickle.dump(model, file)
        st.write("Mô hình đã được huấn luyện và lưu vào file.")

    # Dự đoán với dữ liệu mới
    predictions = model.predict(data_encoded)

    # Hiển thị kết quả dự đoán
    st.write("Kết quả dự đoán:")
    # Đối với Isolation Forest, 1 là bình thường, -1 là bất thường
    result_df = pd.DataFrame({'Prediction': predictions})
    result_df['Prediction'] = result_df['Prediction'].replace({1: 'Normal', -1: 'Anomaly'})
    st.write(result_df)

else:
    st.warning("Vui lòng tải lên cả hai tệp dữ liệu huấn luyện và dữ liệu dự đoán.")



