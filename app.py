import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- TẢI MÔ HÌNH VÀ SCALER ĐÃ LƯU ---
# Dùng cache để không phải tải lại mô hình mỗi khi người dùng tương tác
@st.cache_resource
def load_artifacts():
    """Tải mô hình và scaler đã được huấn luyện."""
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# --- MAP DỮ LIỆU CHUỖI SANG SỐ (DỰA TRÊN LABEL ENCODER ĐÃ DÙNG) ---
# Streamlit sẽ trả về giá trị dạng chữ, ta cần chuyển nó thành số mà mô hình hiểu được
home_ownership_map = {'RENT': 2, 'OWN': 1, 'MORTGAGE': 0, 'OTHER': 3}
loan_intent_map = {'PERSONAL': 4, 'EDUCATION': 1, 'MEDICAL': 3, 'VENTURE': 5, 'HOMEIMPROVEMENT': 2, 'DEBTCONSOLIDATION': 0}
gender_map = {'MALE': 1, 'FEMALE': 0}
education_map = {'HIGH SCHOOL': 1, 'BACHELOR': 0, 'MASTER': 2, 'PHD': 3}
defaults_on_file_map = {'NO': 0, 'YES': 1}

# --- XÂY DỰNG GIAO DIỆN WEB ---
st.set_page_config(page_title="Dự đoán Khoản vay", layout="wide")

st.title("Dự đoán Khả năng Phê duyệt Khoản vay 🏦")
st.write("Nhập thông tin của người vay để mô hình dự đoán.")

# Tạo 2 cột để giao diện gọn gàng hơn
col1, col2 = st.columns(2)

with col1:
    st.subheader("Thông tin cá nhân")
    person_age = st.number_input("Tuổi", min_value=18, max_value=100, value=25)
    person_income = st.number_input("Thu nhập hàng năm ($)", min_value=0, value=50000)
    person_emp_exp = st.number_input("Kinh nghiệm làm việc (năm)", min_value=0, max_value=50, value=5)
    
    # Dùng selectbox cho các lựa chọn có sẵn
    home_ownership = st.selectbox("Tình trạng sở hữu nhà", options=list(home_ownership_map.keys()))
    gender = st.selectbox("Giới tính", options=list(gender_map.keys()))
    education = st.selectbox("Trình độ học vấn", options=list(education_map.keys()))


with col2:
    st.subheader("Thông tin khoản vay")
    loan_amnt = st.number_input("Số tiền vay ($)", min_value=0, value=10000)
    loan_int_rate = st.number_input("Lãi suất (%)", min_value=0.0, max_value=50.0, value=10.0, format="%.2f")
    loan_percent_income = loan_amnt / person_income if person_income > 0 else 0
    
    loan_intent = st.selectbox("Mục đích vay", options=list(loan_intent_map.keys()))
    cb_person_cred_hist_length = st.number_input("Thời gian lịch sử tín dụng (năm)", min_value=0, max_value=30, value=2)
    credit_score = st.number_input("Điểm tín dụng", min_value=300, max_value=850, value=650)
    defaults_on_file = st.selectbox("Đã từng vỡ nợ chưa?", options=list(defaults_on_file_map.keys()))

# --- XỬ LÝ DỮ LIỆU VÀ DỰ ĐOÁN ---
if st.button("Dự đoán", type="primary"):
    # 1. Thu thập dữ liệu từ người dùng
    input_data = {
        'person_age': person_age,
        'person_gender': gender_map[gender],
        'person_education': education_map[education],
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': home_ownership_map[home_ownership],
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent_map[loan_intent],
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': defaults_on_file_map[defaults_on_file]
    }
    
    # Tạo DataFrame từ input
    input_df = pd.DataFrame([input_data])
    
    # 2. Chuẩn hóa dữ liệu bằng scaler đã lưu
    # Cần đảm bảo thứ tự cột giống hệt lúc huấn luyện
    # Đây là thứ tự cột gốc trước khi chia train/test trong file preprocess.ipynb
    original_cols_order = ['person_age', 'person_gender', 'person_education', 'person_income',
                           'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
                           'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                           'credit_score', 'previous_loan_defaults_on_file']
    
    input_df = input_df[original_cols_order]
    
    input_scaled = scaler.transform(input_df)
    
    # 3. Thực hiện dự đoán
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    # 4. Hiển thị kết quả
    st.subheader("Kết quả dự đoán")
    if prediction[0] == 1:
        st.success("✅ Khoản vay có khả năng được CHẤP THUẬN.")
        st.write(f"Độ tin cậy: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.error("❌ Khoản vay có khả năng bị TỪ CHỐI.")
        st.write(f"Độ tin cậy (bị từ chối): {prediction_proba[0][0]*100:.2f}%")