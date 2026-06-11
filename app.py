import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# 1. CẤU HÌNH TRANG WEB STREAMLIT (Lệnh đầu tiên của app)
st.set_page_config(
    layout="wide",
    page_title="Hệ Thống Phát Hiện Giao Dịch Gian Lận",
    page_icon="🛡️"
)

# 2. IMPORT & CÁC HÀM CACHE DÙNG CHUNG
@st.cache_data
def load_data(file_bytes, file_name):
    """
    Nạp dữ liệu từ bytes và đồng bộ định dạng với notebook.
    Hỗ trợ cả file .csv và .xlsx
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return None

# Định nghĩa tập biến đặc trưng đầu vào X dựa trên cấu trúc dữ liệu phát hiện gian lận
FEATURES = [f"X_{i}" for i in range(1, 15)]
TARGET = "default"

# 3. SIDEBAR — VÙNG CẤU HÌNH
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải file dữ liệu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn tệp dữ liệu có cấu trúc chứa các cột từ X_1 đến X_14 và cột mục tiêu 'default'."
    )
    
    st.markdown("---")
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán: Random Forest Classifier")
    
    # Các siêu tham số trích xuất dựa trên cấu trúc thiết lập mô hình học máy
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, 
        max_value=300, 
        value=100, 
        step=10,
        help="Số lượng cây quyết định trong rừng."
    )
    
    criterion = st.selectbox(
        "Tiêu chí đánh giá (criterion)", 
        options=["gini", "entropy", "log_loss"], 
        index=0,
        help="Hàm đo lường chất lượng phân tách của các nút."
    )
    
    max_depth = st.number_input(
        "Độ sâu tối đa (max_depth)", 
        min_value=1, 
        max_value=50, 
        value=10,
        help="Độ sâu tối đa cho phép của các cây quyết định (Để trống nếu không giới hạn)."
    )
    
    # Các tham số nâng cao gom vào expander như quy định
    with st.expander("Tham số nâng cao"):
        min_samples_split = st.slider("Min samples split", min_value=2, max_value=20, value=2)
        random_state = st.number_input("Random State", min_value=0, value=42, step=1)
        test_size = st.slider("Tỷ lệ tập kiểm tra (Test size)", min_value=0.1, max_value=0.5, value=0.3, step=0.05)

    st.markdown("---")
    # Nút bấm hành động duy nhất để kích hoạt huấn luyện mô hình
    train_clicked = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)


# 4. HEADER — VÙNG ĐỊNH HƯỚNG
st.title("🛡️ Hệ Thống Phát Hiện Giao Dịch Gian Lận & Rủi Ro")
st.caption("Ứng dụng phân tích dữ liệu giao dịch và dự báo khả năng gian lận của khách hàng dựa trên nền tảng học máy.")

if uploaded_file is None:
    st.info("👋 Vui lòng tải lên file dữ liệu (.csv hoặc .xlsx) từ thanh Sidebar bên trái để bắt đầu.")
    st.stop()
else:
    # Đọc dữ liệu từ file đã tải lên
    file_bytes = uploaded_file.read()
    df_raw = load_data(file_bytes, uploaded_file.name)
    
    if df_raw is None:
        st.error("Không thể xử lý dữ liệu. Vui lòng kiểm tra lại định dạng tệp.")
        st.stop()
        
    st.caption(f"📁 Đang dùng tệp dữ liệu: `{uploaded_file.name}` ({df_raw.shape[0]} dòng, {df_raw.shape[1]} cột)")

st.divider()


# 5. KHỐI HUẤN LUYỆN (Chạy khi bấm nút và lưu kết quả vào session_state)
if train_clicked:
    with st.spinner("🔄 Đang xử lý dữ liệu và huấn luyện mô hình... Vui lòng đợi."):
        # Kiểm tra sự tồn tại của các cột đặc trưng bắt buộc
        missing_cols = [col for col in FEATURES + [TARGET] if col not in df_raw.columns]
        if missing_cols:
            st.error(f"Thiếu các cột bắt buộc trong dữ liệu: {missing_cols}")
        else:
            # Tách đặc trưng và biến mục tiêu
            X = df_raw[FEATURES]
            y = df_raw[TARGET]
            
            # Phân tách tập dữ liệu Train/Test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state), stratify=y
            )
            
            # Chuẩn hóa dữ liệu bằng StandardScaler giống trong notebook
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Khởi tạo và huấn luyện mô hình Random Forest
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=int(max_depth) if max_depth else None,
                min_samples_split=min_samples_split,
                random_state=int(random_state),
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Dự báo trên tập kiểm tra
            y_pred = model.predict(X_test_scaled)
            y_probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Lưu trữ vào session_state để chia sẻ giữa các tab mà không lo train lại khi chuyển tab
            st.session_state['trained'] = True
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
            st.session_state['y_probs'] = y_probs
            st.session_state['features_importance'] = pd.DataFrame({
                '
