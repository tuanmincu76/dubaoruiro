# 🛡️ Ứng Dụng Web Phát Hiện Giao Dịch Gian Lận & Đánh Giá Rủi Ro

Ứng dụng web được xây dựng bằng framework **Streamlit**, chuyển đổi trực tiếp từ quy trình phân tích và huấn luyện mô hình học máy (Machine Learning) trong Notebook. Ứng dụng tích hợp thuật toán phân loại rừng ngẫu nhiên (**Random Forest Classifier**) nhằm tự động hóa nhận dạng các giao dịch tài chính hoặc hành vi khách hàng có dấu hiệu rủi ro cao (gian lận/bùng nợ).

---

## ✨ Tính Năng Chính Của Ứng Dụng
Ứng dụng được thiết kế phân khu giao diện khoa học theo cấu trúc 4 tab làm việc độc lập tại phân vùng chính:

1. **📊 Tab Tổng quan dữ liệu:** Xem nhanh cấu trúc kích thước dòng/cột, dung lượng tệp, xem trước các bản ghi dữ liệu thô và thống kê phân phối toán học (Describe) của các biến.
2. **📈 Tab Trực quan hóa dữ liệu:** Sử dụng thư viện động trực quan hóa phân phối tần suất của biến mục tiêu `default` và tùy biến lựa chọn vẽ đồ thị Histogram kết hợp Boxplot cho các trường đặc trưng từ `X_1` tới `X_14`.
3. **⚙️ Tab Kết quả huấn luyện & Kiểm định:** Thống kê độ chính xác tổng thể (Accuracy Score), ROC-AUC Score, bảng báo cáo phân loại (Classification Report), ma trận nhầm lẫn (Confusion Matrix) kèm biểu đồ xếp hạng độ quan trọng của các đặc trưng ảnh hưởng trực tiếp tới mô hình.
4. **🔮 Tab Sử dụng mô hình dự báo:**
   - **Chế độ nhập đơn lẻ:** Cung cấp Form điều khiển nhập thủ công các chỉ số (Giá trị khởi tạo mặc định bằng Trung vị (Median) dữ liệu gốc), hiển thị nhãn kết quả An toàn hay Cảnh báo tức thì.
   - **Chế độ dự báo hàng loạt:** Cho phép tải lên tệp danh sách mới chứa cấu trúc các cột đặc trưng, trả ra dữ liệu kết quả chấm điểm xác suất kèm nút tải xuống (.CSV) định dạng `utf-8-sig`.

---

## 📁 Yêu Cầu Cấu Trúc File Dữ Liệu Đầu Vào
Tệp tải lên huấn luyện hoặc dự báo phải ở định dạng `.csv` hoặc `.xlsx` và bắt buộc tuân thủ đúng tên các trường dữ liệu sau:
- **Biến đặc trưng (14 cột độc lập):** `X_1`, `X_2`, `X_3`, `X_4`, `X_5`, `X_6`, `X_7`, `X_8`, `X_9`, `X_10`, `X_11`, `X_12`, `X_13`, `X_14` (Kiểu dữ liệu số liên tục hoặc rời rạc).
- **Biến mục tiêu (Chỉ dùng khi huấn luyện):** Cột mang tên `default` chứa dữ liệu nhãn phân loại nhị phân (`0`: Giao dịch bình thường / `1`: Giao dịch/Khách hàng gian lận).

---

## ⚙️ Hướng Dẫn Cài Đặt Và Chạy Ứng Dụng

### Bước 1: Chuẩn bị môi trường máy tính
Khuyến nghị sử dụng Python phiên bản từ **3.9 đến 3.12**.

### Bước 2: Cài đặt các thư viện cần thiết
Mở Terminal / Command Prompt tại thư mục dự án chứa các file này và chạy lệnh:
```bash
pip install -r requirements.txt
