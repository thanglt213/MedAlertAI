import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import plotly.express as px
from PIL import Image

# Giao diện Streamlit
st.title("Phát hiện bất thường trong bồi thường bảo hiểm sức khỏe")

# Đường dẫn tới file ảnh
image_path = "ica.jpg"

# Mở file ảnh
image = Image.open(image_path)

# Hiển thị ảnh trong Streamlit
st.image(image, caption=" ", width=200, height=100)

st.info("Bất thường không có nghĩa là gian lận, nhưng gian lận là bất thường!", icon="ℹ️")

st.markdown("## 1. Tải dữ liệu huấn luyện và dự đoán")
# Expander cho upload và hiển thị dữ liệu huấn luyện
with st.expander("Tải và xem file dữ liệu huấn luyện - CSV file"):
    train_file = st.file_uploader("Chọn file CSV huấn luyện", type=["csv"], key='train')
    if train_file is not None:
        train_data = pd.read_csv(train_file)
        train_data = train_data.dropna()
        train_data = train_data.astype('str')
        st.write("Dữ liệu huấn luyện:", train_data.head())

# Expander cho upload và hiển thị dữ liệu dự đoán
with st.expander("Tải và xem file dữ liệu cần tìm bất thường - CSV file"):
    uploaded_file = st.file_uploader("Chọn file CSV dự đoán", type=["csv"], key='data')
    if uploaded_file is not None:
        predict_data = pd.read_csv(uploaded_file)
        predict_data = predict_data.dropna()
        predict_data = predict_data.astype('str')
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

    # Chuyển một số trường về kiểu numeric còn lại về dạng string
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
    st.info("Mô hình đã được huấn luyện thành công.")

    st.markdown("## 2. Dự đoán")
    # Dự đoán với dữ liệu mới
    predictions = model.predict(predict_data_encoded)

    # Update kết quả dự đoán vào dữ liệu gốc
    result_df = pd.DataFrame({'Prediction': predictions})
    result_df['Prediction'] = result_df['Prediction'].replace({1: 'Bình thường', -1: 'Bất thường'})
    predict_data['Prediction'] = result_df['Prediction']

    # Đếm số dòng có 'Prediction' == 'Bất thường'
    count_bat_thuong = len(predict_data[predict_data['Prediction'] == 'Bất thường'])

    # Đếm tổng số dòng
    total_count = len(predict_data)

    # Tạo dòng kết quả
    #result_line = f"{count_bat_thuong}/{total_count}"

    # Hiển thị kết quả trên Streamlit
    st.write("Bất thường: ", f"{count_bat_thuong}/{total_count}")

    # Hiển thị DataFrame 
    # Sắp xếp lại DataFrame theo thứ tự cột mong muốn
    # Thứ tự cột mong muốn
    desired_order = ['Prediction', 'branch', 'claim_no','distribution_channel','hospital']
    st.dataframe(predict_data[desired_order], use_container_width=True)
    
    # Thực hiện highlight
    #highlighted_data = highlight_rows(predict_data, 'Prediction', 'Bất thường', 'lightblue')
    # Hiển thị dataframe với highlight
    #st.dataframe(highlighted_data)
    
    # Nút tải xuống file CSV kết quả
    csv = predict_data.to_csv(index=False)
    st.download_button("Tải xuống kết quả dự đoán", csv, "predictions.csv", "text/csv", key="download")

    st.markdown("## 3. Trực quan hóa kết quả dự đoán")
    # ------------------------------------------------------- Biểu đồ thể hiện số lượng hồ sơ bồi thường có dấu hiệu bất thường qua kênh khai thác
    st.markdown("#### **Kênh khai thác:**")
    
    # Lấy dữ liệu liên quan đến distribution_channel và Prediction
    chart_data = predict_data[predict_data['Prediction'] == 'Bất thường'][['distribution_channel', 'Prediction']]
    
    # Đếm số lượng prediction theo distribution_channel
    prediction_counts = chart_data.groupby(['distribution_channel', 'Prediction']).size().unstack(fill_value=0)
    
    # Tính tổng số lượng hồ sơ cho từng distribution_channel và sắp xếp giảm dần
    prediction_counts['Total'] = prediction_counts.sum(axis=1)
    prediction_counts = prediction_counts.sort_values(by='Total', ascending=False)
    
    # Xóa cột 'Total' để không hiển thị trong biểu đồ
    prediction_counts = prediction_counts.drop(columns='Total')
    
    # Hiển thị dữ liệu cho biểu đồ
    #st.write(prediction_counts)
    st.dataframe(prediction_counts, use_container_width=True)
    
    # Tạo biểu đồ cột sử dụng Plotly
    fig = px.bar(prediction_counts.reset_index(), 
                 x='distribution_channel', 
                 y=prediction_counts.columns,  # Các cột tương ứng với giá trị Prediction
                 title='Số lượng hồ sơ bồi thường theo kênh khai thác (sắp xếp giảm dần)',
                 labels={'value': '', 'distribution_channel': 'Kênh khai thác'},
                 text_auto=True,  # Thêm nhãn số lượng trên mỗi thanh
                 barmode='stack')  # Biểu đồ stack bar
    
    # Hiển thị biểu đồ trong Streamlit
    st.plotly_chart(fig)

    # ------------------------------------------------------- Biểu đồ thể hiện tỷ lệ % số hồ sơ bất thường qua kênh khai thác
    # Biểu đồ thể hiện tỷ lệ % số hồ sơ bất thường qua kênh khai thác
    st.markdown("#### **Kênh khai thác (Tỷ lệ % bất thường):**")
    
    # Lấy dữ liệu liên quan đến distribution_channel và Prediction
    chart_data = predict_data[['distribution_channel', 'Prediction']]
    
    # Đếm số lượng prediction theo distribution_channel
    prediction_counts = chart_data.groupby(['distribution_channel', 'Prediction']).size().unstack(fill_value=0)
    
    # Tính tổng số lượng hồ sơ cho mỗi distribution_channel
    prediction_counts['Total'] = prediction_counts.sum(axis=1)
    
    # Tính tỷ lệ % số hồ sơ 'Bất thường' so với tổng số hồ sơ
    prediction_counts['Bất thường %'] = (prediction_counts.get('Bất thường', 0) / prediction_counts['Total']) * 100
    
    # Sắp xếp theo tỷ lệ % bất thường giảm dần
    prediction_counts = prediction_counts.sort_values(by='Bất thường %', ascending=False)
    
    # Hiển thị dữ liệu cho biểu đồ
    #st.write(prediction_counts[['Bất thường %']])
    st.dataframe(prediction_counts, use_container_width=True)
    
    # Định dạng nhãn tỷ lệ % với dấu phần trăm
    prediction_counts['Bất thường % Text'] = prediction_counts['Bất thường %'].map('{:.2f}%'.format)
    
    # Tạo biểu đồ cột sử dụng Plotly
    fig = px.bar(prediction_counts.reset_index(), 
                 x='distribution_channel', 
                 y='Bất thường %',  # Biểu diễn cột tỷ lệ % bất thường
                 title='Tỷ lệ % hồ sơ bất thường theo kênh khai thác',
                 labels={'Bất thường %': 'Tỷ lệ % Bất thường', 'distribution_channel': 'Kênh khai thác'},
                 text='Bất thường % Text',  # Thay đổi tham số text để hiển thị tỷ lệ %
                 text_auto=False)  # Tắt text_auto để sử dụng nhãn tùy chỉnh
    
    # Hiển thị biểu đồ trong Streamlit
    st.plotly_chart(fig)

    # ------------------------------------------------------- Biểu đồ thể hiện số lượng hồ sơ bồi thường có dấu hiệu bất thường qua bệnh viện 
    st.markdown("#### **Theo bệnh viện:**")
    chart_data = predict_data[['hospital', 'Prediction']]

    # Đếm số lượng prediction theo hospital
    prediction_counts = chart_data.groupby(['hospital', 'Prediction']).size().unstack(fill_value=0)

    # Hiển thị dữ liệu để kiểm tra
    #st.write(prediction_counts)
    st.dataframe(prediction_counts, use_container_width=True)


    # Tạo biểu đồ cột ngang với plotly
    fig = px.bar(
        prediction_counts, 
        orientation='h',  # 'h' để tạo biểu đồ cột ngang
        title="Số lượng hồ sơ dự đoán theo bệnh viện",
        labels={"value": "", "hospital": ""},
        text_auto=True  # Thêm nhãn số lượng trên mỗi thanh
    )

    # Hiển thị biểu đồ với Streamlit
    st.plotly_chart(fig)

    # Biểu đồ thể hiện số lượng hồ sơ bồi thường có dấu hiệu bất thường qua chi nhánh -------------------------------------------------------
    st.markdown("#### **Theo chi nhánh:**")
    chart_data = predict_data[['branch', 'Prediction']]

    # Đếm số lượng prediction theo branch
    prediction_counts = chart_data.groupby(['branch', 'Prediction']).size().unstack(fill_value=0)
    
    # Thêm cột tổng số lượng prediction theo từng chi nhánh để tính tỷ lệ phần trăm
    prediction_counts['Total'] = prediction_counts.sum(axis=1)
    
    # Tính tỷ lệ phần trăm cho mỗi loại prediction
    prediction_percentage = prediction_counts.div(prediction_counts['Total'], axis=0) * 100
    
    # Bỏ cột Total (vì không cần hiển thị tỷ lệ tổng)
    prediction_percentage = prediction_percentage.drop(columns='Total')
    
    # Sắp xếp theo cột 'Bất thường' giảm dần (hoặc 'Bình thường' nếu muốn)
    prediction_percentage = prediction_percentage.sort_values('Bất thường', ascending=True)
    
    # Hiển thị dữ liệu để kiểm tra
    #st.write(prediction_percentage)
    st.dataframe(prediction_percentage, use_container_width=True)
    
    # Tạo biểu đồ cột ngang với plotly và hiển thị tỷ lệ phần trăm
    fig = px.bar(
        prediction_percentage,  # Sử dụng tỷ lệ phần trăm
        x=prediction_percentage.columns,  # Trục x là các loại Prediction
        y=prediction_percentage.index,  # Trục y là chi nhánh
        #orientation='h',  # 'h' để tạo biểu đồ cột ngang
        title="Tỷ lệ hồ sơ dự đoán theo chi nhánh",
        labels={"value": "Tỷ lệ %", "branch": "Chi nhánh"},
        text_auto='.2f'  # Thêm nhãn tỷ lệ phần trăm với 2 chữ số thập phân
    )
    
    # Hiển thị biểu đồ trong Streamlit
    st.plotly_chart(fig)


else:
    st.warning("Vui lòng tải lên cả hai tệp dữ liệu huấn luyện và dữ liệu dự đoán.")

