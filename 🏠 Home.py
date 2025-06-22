import streamlit as st

st.set_page_config(page_title="Dự đoán rời bỏ", layout="wide", initial_sidebar_state="expanded")

st.title("👋 Chào mừng Anh Bùi Ngọc Tùng đến với ứng dụng Dự đoán khách bàng Rời bỏ Khách hàng")
st.markdown("""
Ứng dụng này giúp ngân hàng dự đoán **khách hàng có khả năng rời bỏ (churn)** dựa trên hành vi gần đây của họ như:
- Số giao dịch,
- Số dư tài khoản,
- Số lần khiếu nại,
- Sự hoạt động gần đây...

### 🔍 Tính năng:
- Dữ đoán khả năng rời bỏ của khách hàng
- Hiển thị dữ liệu, biểu đồ minh họa
- Demo với dữ liệu giả lập (không cần file tải lên)

👉 Vào mục **"Dự báo khách hàng Rời Bỏ"** ở menu bên trái để bắt đầu!
""")



