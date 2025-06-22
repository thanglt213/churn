import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import plotly.express as px

# ===== Cấu hình giao diện =====
st.set_page_config(
    page_title="Dự đoán khách hàng rời bỏ",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.sidebar.title("📂 Menu")

st.title("🔍 Dự đoán khách hàng rời bỏ ngân hàng")

# ===== Giải thích dữ liệu =====
with st.expander("📖 Giải thích các trường dữ liệu"):
    st.markdown("""
    | Trường | Mô tả |
    |--------|-------|
    | `age` | Tuổi |
    | `gender` | Giới tính (`Male`, `Female`) |
    | `tenure` | Số năm gắn bó |
    | `balance` | Số dư tài khoản |
    | `num_txn_30d` | Giao dịch 30 ngày gần nhất |
    | `avg_txn_amt` | Số tiền trung bình mỗi giao dịch |
    | `has_credit_card` | Có thẻ tín dụng (1/0) |
    | `num_complaints` | Số lần khiếu nại |
    | `is_active` | (tự tính): 1 nếu có giao dịch |
    | `churned` | 1 = rời bỏ, 0 = giữ lại |
    """)

# ===== Dữ liệu mẫu =====
@st.cache_data
def load_train_data():
    return pd.DataFrame({
        'customer_id': ['C1001', 'C1002', 'C1003', 'C1004', 'C1005', 'C1006'],
        'age': [45, 29, 34, 60, 40, 50],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Female', 'Male'],
        'tenure': [6, 2, 4, 7, 3, 10],
        'balance': [12000, 200, 5000, 30000, 0, 20000],
        'num_txn_30d': [15, 5, 12, 2, 0, 20],
        'avg_txn_amt': [80, 25, 45, 300, 10, 100],
        'has_credit_card': [1, 1, 0, 1, 1, 1],
        'num_complaints': [0, 1, 0, 3, 0, 0],
        'churned': [0, 1, 0, 1, 1, 0]
    })

@st.cache_data
def load_predict_data():
    return pd.DataFrame({
        'customer_id': ['C2001', 'C2002', 'C2003'],
        'age': [38, 62, 41],
        'gender': ['Male', 'Female', 'Female'],
        'tenure': [3, 5, 1],
        'balance': [3000, 100, 0],
        'num_txn_30d': [8, 2, 0],
        'avg_txn_amt': [40, 15, 8],
        'has_credit_card': [1, 1, 0],
        'num_complaints': [0, 1, 0]
    })

# ===== Upload dữ liệu huấn luyện =====
use_custom_train = st.checkbox("🛠️ Dùng dữ liệu huấn luyện từ CSV")

if use_custom_train:
    train_file = st.file_uploader("Tải file CSV huấn luyện", type="csv", key="train_csv")
    if train_file is not None:
        df_train_raw = pd.read_csv(train_file)
        st.success("✅ Đã tải dữ liệu huấn luyện.")
    else:
        st.warning("⚠️ Chưa tải file. Dùng dữ liệu mặc định.")
        df_train_raw = load_train_data()
else:
    df_train_raw = load_train_data()

st.subheader("📚 Dữ liệu huấn luyện (gốc)")
st.dataframe(df_train_raw)

# ===== Tiền xử lý bản sao =====
df_train = df_train_raw.copy()
if 'gender' in df_train.columns:
    df_train['gender'] = df_train['gender'].map({'Male': 0, 'Female': 1})
if 'num_txn_30d' in df_train.columns:
    df_train['is_active'] = df_train['num_txn_30d'].apply(lambda x: 1 if x > 0 else 0)

# ===== Huấn luyện mô hình =====
X = df_train.drop(columns=["customer_id", "churned"])
y = df_train["churned"]
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X, y)

# ===== Đánh giá mô hình trên tập huấn luyện =====
st.subheader("📋 Đánh giá mô hình trên dữ liệu huấn luyện")
y_pred = model.predict(X)
st.text(classification_report(y, y_pred, target_names=["Giữ lại", "Rời bỏ"]))

# ===== Upload dữ liệu dự đoán =====
st.subheader("📥 Tải dữ liệu CSV để dự đoán (tuỳ chọn)")
uploaded_file = st.file_uploader("Chọn file CSV", type="csv", key="csv_upload")

if uploaded_file is not None:
    df_predict_raw = pd.read_csv(uploaded_file)
    st.success("✅ Đã tải dữ liệu dự đoán.")
else:
    df_predict_raw = load_predict_data()
    st.info("🧪 Đang dùng dữ liệu mẫu.")

st.subheader("📄 Dữ liệu cần dự đoán (gốc)")
st.dataframe(df_predict_raw)

# ===== Tiền xử lý bản sao =====
df_predict = df_predict_raw.copy()
if 'gender' in df_predict.columns:
    df_predict['gender'] = df_predict['gender'].map({'Male': 0, 'Female': 1})
if 'num_txn_30d' in df_predict.columns:
    df_predict['is_active'] = df_predict['num_txn_30d'].apply(lambda x: 1 if x > 0 else 0)

# ===== Dự đoán & xác suất =====
X_new = df_predict.drop(columns=["customer_id"])
df_predict_raw["Churn Dự đoán"] = model.predict(X_new)
probs = model.predict_proba(X_new)
df_predict_raw["Xác suất rời bỏ (%)"] = (probs[:, 1] * 100).round(2)

# ===== Hiển thị kết quả =====
st.subheader("📊 Kết quả dự đoán")
df_show = df_predict_raw[["customer_id", "Churn Dự đoán", "Xác suất rời bỏ (%)"]].sort_values(by="Xác suất rời bỏ (%)", ascending=False)
st.dataframe(df_show)

# ===== Biểu đồ tròn tổng thể =====
pie_data = df_predict_raw["Churn Dự đoán"].value_counts().rename(index={0: "Giữ lại", 1: "Rời bỏ"}).reset_index()
pie_data.columns = ["Trạng thái", "Số lượng"]

fig = px.pie(
    pie_data,
    names="Trạng thái",
    values="Số lượng",
    title="📈 Tỷ lệ khách hàng dự đoán rời bỏ"
)
st.plotly_chart(fig)

# ===== Biểu đồ thanh Top 5 nguy cơ cao =====
st.subheader("🔥 Top 5 khách hàng nguy cơ rời bỏ cao nhất")
top5 = df_show.head(5)
fig_bar = px.bar(
    top5,
    x="customer_id",
    y="Xác suất rời bỏ (%)",
    color="Xác suất rời bỏ (%)",
    color_continuous_scale="Reds",
    title="Top 5 khách hàng có xác suất rời bỏ cao nhất"
)
st.plotly_chart(fig_bar)
