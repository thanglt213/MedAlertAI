import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

# Giao diện Streamlit
st.title("Phát hiện bất thường trong bồi thường bảo hiểm sức khỏe")

st.info("Bất thường không có nghĩa là gian lận, nhưng gian lận là bất thường", icon="ℹ️")

# Expander cho upload và hiển thị dữ liệu huấn luyện
with st.expander("Tải và xem file dữ liệu huấn luyện - CSV file"):
    train_file = st.file_uploader("Chọn file CSV huấn luyện", type=["csv"], key='train')
    if train_file is not None:
        train_data = pd.read_csv(train_file)
        train_data = train_data.dropna()
        st.write("Dữ liệu huấn luyện:", train_data.head())

# Expander cho upload và hiển thị dữ liệu dự đoán
with st.expander("Tải và xem file dữ liệu cần tìm bất thường - CSV file"):
    uploaded_file = st.file_uploader("Chọn file CSV dự đoán", type=["csv"], key='data')
    if uploaded_file is not None:
        predict_data = pd.read_csv(uploaded_file)
        predict_data = predict_data.dropna()
        st.write("Dữ liệu cần tìm bất thường:", predict_data.head())

# Hàm highlight các dòng
def highlight_rows(df, column, value, color):
    def highlight_condition(row):
        return [f'background-color: {color}' if row[column] == value else '' for _ in row]

    return df.style.apply(highlight_condition, axis=1)

if train_file and uploaded_file:
    # Lưu số dòng của dữ liệu huấn luyện
    num_train_rows = train_data.shape[0]

    # Thêm cột 'is_train' để đánh dấu tập dữ liệu huấn luyện và dự đoán
    train_data['is_train'] = 1
    predict_data['is_train'] = 0

    # Gộp train_data và predict_data
    combined_data = pd.concat([train_data, predict_data], ignore_index=True)

    # Chuyển một số trường về kiểu numeric còn lại về dạng object
    combined_data = combined_data.astype('str')
  
    # Chọn các cột kiểu numeric
    cols_to_numeric =  ['days_to_report', 'requested_amount_per_day']

    # Chuyển các cột trong danh sách về kiểu numeric
    for col in cols_to_numeric:
        combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')

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
                
    # Chuẩn hóa dữ liệu bằng Standard Scaler
    # Khởi tạo StandardScaler
    scaler = StandardScaler()

    # Lựa chọn các cột cần chuẩn hóa trong combined_data (ví dụ, tất cả trừ cột không phải số)
    numeric_columns = combined_data.select_dtypes(include=['float64', 'int64']).columns

    # Chuẩn hóa dữ liệu
    combined_data[numeric_columns] = scaler.fit_transform(combined_data[numeric_columns])
        
    # Tách lại dữ liệu huấn luyện và dữ liệu dự đoán dựa trên số dòng đã lưu
    train_data_encoded = combined_data.iloc[:num_train_rows].drop(columns=['is_train'])
    predict_data_encoded = combined_data.iloc[num_train_rows:].drop(columns=['is_train'])

    # Hiển thị dữ liệu sau mã hóa và chuẩn hóa
    #st.write("Dữ liệu sau mã hóa và chuẩn hóa:",predict_data_encoded.head())

    # Khởi tạo mô hình Isolation Forest
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

    # Lấy dữ liệu để huấn luyện
    train_data_encoded = train_data_encoded.select_dtypes(include=[np.number])
    # Huấn luyện mô hình
    model.fit(train_data_encoded)
    st.write("Mô hình đã được huấn luyện thành công.")

    # Dự đoán với dữ liệu mới
    st.write("Dự đoán:")
    predictions = model.predict(predict_data_encoded)

    # Update kết quả dự đoán vào dữ liệu gốc
    result_df = pd.DataFrame({'Prediction': predictions})
    result_df['Prediction'] = result_df['Prediction'].replace({1: 'Bình thường', -1: 'Bất thường'})
    predict_data['Prediction'] = result_df['Prediction']
    
    # Hiển thị DataFrame 
    st.write("Kết quả dự đoán:", predict_data)
    
    # Thực hiện highlight
    #highlighted_data = highlight_rows(predict_data, 'Prediction', 'Bất thường', 'lightblue')
    # Hiển thị dataframe với highlight
    #st.dataframe(highlighted_data)
    
    # Nút tải xuống file CSV kết quả
    csv = predict_data.to_csv(index=False)
    st.download_button("Tải xuống kết quả dự đoán", csv, "predictions.csv", "text/csv", key="download")
else:
    st.warning("Vui lòng tải lên cả hai tệp dữ liệu huấn luyện và dữ liệu dự đoán.")

