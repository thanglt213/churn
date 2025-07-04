import streamlit as st

st.set_page_config(page_title="Dự đoán rời bỏ", layout="wide", initial_sidebar_state="expanded")

st.title("Một ứng dụng AI: Dự đoán khách hàng rời bỏ Ngân hàng")
st.markdown("""
Ứng dụng này giúp ngân hàng dự đoán **khách hàng có khả năng rời bỏ (churn)** dựa trên hành vi gần đây của họ như:
- Số giao dịch,
- Số dư tài khoản,
- Số lần khiếu nại,
- Sự hoạt động gần đây,
- Thời gian gắn bó...

### 🔍 Tính năng:
- Dữ đoán khả năng rời bỏ của khách hàng
- Hiển thị dữ liệu, biểu đồ minh họa
- Demo với dữ liệu giả lập (không cần file tải lên)

👉 Vào mục **"Dự báo khách hàng Rời Bỏ"** ở menu bên trái để bắt đầu!
""")



