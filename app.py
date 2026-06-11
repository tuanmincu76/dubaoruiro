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
                'Đặc trưng': FEATURES,
                'Độ quan trọng': model.feature_importances_
            }).sort_values(by='Độ quan trọng', ascending=False)
            
            st.success("🎉 Huấn luyện mô hình thành công! Hãy chuyển sang các Tab bên dưới để xem chi tiết kết quả.")


# 6. KHỐI GIAO DIỆN CHÍNH CHIA TAB
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "⚙️ Kết quả huấn luyện & Kiểm định", 
    "🔮 Sử dụng mô hình dự báo"
])

# --- TAB 1: TỔNG QUAN DỮ LIỆU ---
with tab1:
    st.subheader("Phân tích cấu trúc dữ liệu")
    
    # 1. Kích thước dữ liệu hiển thị bằng Metric
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Số lượng bản ghi (Dòng)", f"{df_raw.shape[0]:,}")
    col_m2.metric("Số lượng trường (Cột)", f"{df_raw.shape[1]}")
    file_size_mb = len(file_bytes) / (1024 * 1024)
    col_m3.metric("Dung lượng tệp", f"{file_size_mb:.2f} MB")
    
    # 2. Xem dữ liệu thô
    st.write("📂 **5 hàng dữ liệu đầu tiên:**")
    st.dataframe(df_raw.head(), use_container_width=True)
    
    # 3. Thống kê mô tả (Chỉ các biến đưa vào mô hình)
    st.write("📊 **Thống kê mô tả các biến đặc trưng đưa vào mô hình:**")
    available_features = [c for c in FEATURES + [TARGET] if c in df_raw.columns]
    st.dataframe(df_raw[available_features].describe().T, use_container_width=True)


# --- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ---
with tab2:
    st.subheader("Trực quan phân phối và phân loại dữ liệu")
    
    # Trực quan hóa biến mục tiêu trước (nếu có)
    if TARGET in df_raw.columns:
        fig_target = px.histogram(
            df_raw, x=TARGET, 
            title="Biểu đồ phân phối của Biến mục tiêu (Gian lận/Default)",
            color=TARGET, color_discrete_sequence=px.colors.qualitative.Set2,
            labels={TARGET: "Trạng thái rủi ro"}
        )
        fig_target.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_target, use_container_width=True)
    
    st.markdown("---")
    st.write("📸 **Biểu đồ phân phối các biến số độc lập (Chọn tối đa 4 biến để tối ưu hiển thị):**")
    
    # Thanh đa chọn cho biến trực quan
    selected_vars = st.multiselect(
        "Chọn các trường dữ liệu cần vẽ biểu đồ phân phối:", 
        options=FEATURES, 
        default=FEATURES[:4]
    )
    
    if selected_vars:
        # Tạo lưới biểu đồ 2x2 bằng st.columns
        cols = st.columns(2)
        for idx, var in enumerate(selected_vars):
            if var in df_raw.columns:
                current_col = cols[idx % 2]
                with current_col:
                    fig_var = px.histogram(
                        df_raw, x=var, 
                        title=f"Phân phối của biến {var}",
                        marginal="box", # Thêm biểu đồ hộp để quan sát điểm ngoại lai
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_var.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_var, use_container_width=True)


# --- TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH ---
with tab3:
    st.subheader("Đánh giá độ chính xác của Mô hình Học máy")
    
    # Kiểm tra trạng thái huấn luyện
    if 'trained' not in st.session_state:
        st.info("💡 Chưa có dữ liệu mô hình. Vui lòng bấm vào nút **'Huấn luyện mô hình'** ở thanh Sidebar bên trái để xem kết quả đánh giá.")
    else:
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        y_probs = st.session_state['y_probs']
        fi_df = st.session_state['features_importance']
        
        # Chỉ tiêu vô hướng
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probs) if y_probs is not None else "N/A"
        
        c_m1, c_m2 = st.columns(2)
        c_m1.metric("Độ chính xác tổng thể (Accuracy Score)", f"{acc:.4%}")
        if isinstance(auc, float):
            c_m2.metric("Chỉ số ROC-AUC Score", f"{auc:.4f}")
        else:
            c_m2.metric("Chỉ số ROC-AUC Score", auc)
            
        st.markdown("---")
        
        # Bố cục chia hai cột trái phải
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.write("📊 **Báo cáo chi tiết phân loại (Classification Report):**")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)
            
            st.write("🧱 **Ma trận nhầm lẫn (Confusion Matrix):**")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=['Thực tế: Không gian lận (0)', 'Thực tế: Gian lận (1)'], 
                                 columns=['Dự báo: Không (0)', 'Dự báo: Có (1)'])
            st.dataframe(cm_df, use_container_width=True)
            
        with col_right:
            st.write("🎯 **Biểu đồ mức độ quan trọng của các biến đặc trưng (Feature Importance):**")
            fig_fi = px.bar(
                fi_df, x='Độ quan trọng', y='Đặc trưng', 
                orientation='h',
                title="Mức độ ảnh hưởng của các biến đầu vào tới dự báo rủi ro",
                color='Độ quan trọng', color_continuous_scale='Blues'
            )
            fig_fi.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_fi, use_container_width=True)


# --- TAB 4: SỬ DỤNG MÔ HÌNH ---
with tab4:
    st.subheader("Dự báo dữ liệu rủi ro gian lận mới")
    
    if 'trained' not in st.session_state:
        st.info("💡 Chức năng này yêu cầu mô hình đã được huấn luyện thành công. Vui lòng thiết lập tham số và bấm nút tại Sidebar trước.")
    else:
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        
        # Chế độ lựa chọn dự báo theo hướng dẫn thiết kế giao diện
        mode = st.radio(
            "Chọn hình thức nhập dữ liệu đầu vào để dự báo:",
            options=["Chế độ 1: Nhập trực tiếp thông số", "Chế độ 2: Tải tệp danh sách hàng loạt (Excel/CSV)"],
            horizontal=True
        )
        
        # CHẾ ĐỘ 1 — NHẬP TRỰC TIẾP QUA FORM
        if mode == "Chế độ 1: Nhập trực tiếp thông số":
            st.write("📝 **Nhập thông số của khách hàng/giao dịch:**")
            
            with st.form("prediction_form"):
                form_cols = st.columns(3)
                input_data = {}
                
                # Tạo tự động các ô nhập liệu tương ứng từ X_1 đến X_14 lấy giá trị trung vị mặc định
                for i, col_name in enumerate(FEATURES):
                    col_idx = i % 3
                    # Lấy giá trị trung vị từ tập dữ liệu tải lên ban đầu để gán làm giá trị mặc định cho form
                    default_val = float(df_raw[col_name].median()) if col_name in df_raw.columns else 0.0
                    min_val = float(df_raw[col_name].min()) if col_name in df_raw.columns else -100.0
                    max_val = float(df_raw[col_name].max()) if col_name in df_raw.columns else 100.0
                    
                    with form_cols[col_idx]:
                        input_data[col_name] = st.number_input(
                            f"Thông số {col_name}",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            format="%.4f"
                        )
                
                submit_pred = st.form_submit_button("🔮 Thực hiện dự báo rủi ro", type="primary")
                
            if submit_pred:
                # Chuyển đổi dữ liệu nhập vào thành DataFrame đúng định dạng để tiền xử lý
                single_df = pd.DataFrame([input_data])
                single_scaled = scaler.transform(single_df[FEATURES])
                
                # Dự báo kết quả rủi ro
                pred_class = model.predict(single_scaled)[0]
                pred_prob = model.predict_proba(single_scaled)[0][1]
                
                st.markdown("### Kết quả phân tích:")
                col_res1, col_res2 = st.columns(2)
                
                if pred_class == 1:
                    col_res1.error("🚨 CẢNH BÁO: PHÁT HIỆN RỦI RO GIAN LẬN")
                    col_res2.metric("Xác suất rủi ro", f"{pred_prob:.2%}")
                else:
                    col_res1.success("✅ AN TOÀN: GIAO DỊCH KHÔNG GIAN LẬN")
                    col_res2.metric("Xác suất rủi ro", f"{pred_prob:.2%}")

        # CHẾ ĐỘ 2 — TẢI FILE DANH SÁCH DỰ BÁO HÀNG LOẠT
        elif mode == "Chế độ 2: Tải tệp danh sách hàng loạt (Excel/CSV)":
            st.write("📂 **Tải lên tập dữ liệu mới cần quét rủi ro (Yêu cầu chứa đầy đủ các cột từ X_1 đến X_14):**")
            
            predict_file = st.file_uploader(
                "Chọn tệp dữ liệu kiểm định mới:", 
                type=["csv", "xlsx"],
                key="predict_file_uploader"
            )
            
            if predict_file:
                df_predict_raw = load_data(predict_file.read(), predict_file.name)
                
                if df_predict_raw is not None:
                    # Kiểm tra cấu trúc Schema cột của file mới
                    missing_features = [col for col in FEATURES if col not in df_predict_raw.columns]
                    
                    if missing_features:
                        st.error(f"Tệp tải lên không hợp lệ. Thiếu các cột biến đặc trưng sau: {missing_features}")
                    else:
                        st.success("Cấu trúc file hợp lệ! Đang xử lý dự báo hàng loạt...")
                        
                        # Thực hiện chuẩn hóa và dự báo
                        X_new = df_predict_raw[FEATURES]
                        X_new_scaled = scaler.transform(X_new)
                        
                        predictions = model.predict(X_new_scaled)
                        probabilities = model.predict_proba(X_new_scaled)[:, 1]
                        
                        # Tạo bảng kết quả kết hợp dữ liệu gốc
                        df_results = df_predict_raw.copy()
                        df_results['Dự_Báo_Default'] = predictions
                        df_results['Xác_Suất_Rủi_Ro'] = probabilities
                        
                        # Thống kê tổng quan kết quả dự báo
                        total_cnt = len(df_results)
                        fraud_cnt = int(np.sum(predictions == 1))
                        
                        c_p1, c_p2 = st.columns(2)
                        c_p1.metric("Tổng số lượng giao dịch đã quét", f"{total_cnt:,}")
                        c_p2.metric("Số giao dịch phát hiện nghi vấn gian lận", f"{fraud_cnt:,}", 
                                  delta=f"{(fraud_cnt/total_cnt):.2%} tổng số", delta_color="inverse")
                        
                        st.write("📋 **Bảng kết quả dự báo chi tiết:**")
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Xuất file kết quả dự báo ra định dạng CSV hỗ trợ download trực tiếp
                        csv_buffer = io.StringIO()
                        df_results.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Tải xuống bảng kết quả dự báo (.CSV)",
                            data=csv_data,
                            file_name="ket_qua_du_bao_gian_lan.csv",
                            mime="text/csv"
                        )
