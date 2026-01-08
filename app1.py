# app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Bank Deposit Predictor",
    page_icon="💳",
    layout="wide"
)

# -------------------------------------------------
# ADVANCED UI (CSS + GLASSMORPHISM)
# -------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.main {
    background: linear-gradient(135deg, #020617, #020617);
}

.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(18px);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

.header {
    font-size: 40px;
    font-weight: 700;
    color: #f8fafc;
}

.sub {
    color: #cbd5f5;
    font-size: 16px;
}

.metric {
    background: linear-gradient(135deg,#1e293b,#020617);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    color: white;
}

.success-box {
    background: linear-gradient(135deg,#22c55e,#15803d);
    padding: 25px;
    border-radius: 18px;
    font-size: 22px;
    color: white;
    font-weight: 600;
}

.fail-box {
    background: linear-gradient(135deg,#ef4444,#7f1d1d);
    padding: 25px;
    border-radius: 18px;
    font-size: 22px;
    color: white;
    font-weight: 600;
}

.stButton>button {
    background: linear-gradient(135deg,#6366f1,#22d3ee);
    color: black;
    font-weight: 700;
    border-radius: 30px;
    padding: 12px 30px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown('<div class="header">💳 AI Term Deposit Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Smart customer classification using Machine Learning</div><br>', unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("bank_marketing_dataset.csv")

df = load_data()

# -------------------------------------------------
# OUTLIER REMOVAL (RANGE BASED)
# -------------------------------------------------
df = df[(df["age"] >= 18) & (df["age"] <= 90)]
df = df[(df["balance"] >= -2000) & (df["balance"] <= 100000)]

# -------------------------------------------------
# FEATURES & TARGET
# -------------------------------------------------
X = df[["age", "job", "balance", "loan", "contact"]]
y = df["deposit"].str.strip().str.lower().map({"yes": 1, "no": 0})

# -------------------------------------------------
# HANDLE MISSING VALUES
# -------------------------------------------------
X["age"] = X["age"].fillna(X["age"].median())
X["balance"] = X["balance"].fillna(X["balance"].median())
X["job"] = X["job"].fillna(X["job"].mode()[0])
X["loan"] = X["loan"].fillna(X["loan"].mode()[0])
X["contact"] = X["contact"].fillna(X["contact"].mode()[0])

# -------------------------------------------------
# LABEL ENCODING
# -------------------------------------------------
le_job = LabelEncoder()
le_loan = LabelEncoder()
le_contact = LabelEncoder()

X["job"] = le_job.fit_transform(X["job"])
X["loan"] = le_loan.fit_transform(X["loan"])
X["contact"] = le_contact.fit_transform(X["contact"])

# -------------------------------------------------
# SPLIT & SCALE
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
num_cols = ["age", "balance"]

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# -------------------------------------------------
# MODEL (DECISION TREE – GINI)
# -------------------------------------------------
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    min_samples_split=8,
    random_state=42
)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# -------------------------------------------------
# METRICS DASHBOARD
# -------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric">Accuracy<br><h2>{accuracy:.2f}</h2></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric">Model<br><h2>Decision Tree</h2></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric">Criterion<br><h2>GINI</h2></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------
# INPUT PANEL
# -------------------------------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("🧾 Customer Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 90, 35)
    job = st.selectbox("Occupation", le_job.classes_)
    contact = st.selectbox("Contact Method", le_contact.classes_)

with col2:
    balance = st.number_input("Account Balance", value=2500)
    loan = st.selectbox("Personal Loan", le_loan.classes_)

st.markdown("</div><br>", unsafe_allow_html=True)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("🚀 Predict Customer Behavior"):
    input_df = pd.DataFrame([{
        "age": age,
        "job": le_job.transform([job])[0],
        "balance": balance,
        "loan": le_loan.transform([loan])[0],
        "contact": le_contact.transform([contact])[0]
    }])

    input_df[num_cols] = scaler.transform(input_df[num_cols])
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.markdown('<div class="success-box">✅ Customer is LIKELY to Subscribe</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="fail-box">❌ Customer is NOT Likely to Subscribe</div>', unsafe_allow_html=True)
