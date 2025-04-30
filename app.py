import streamlit as st
import pandas as pd
import pickle

# Sayfa ayarları
st.set_page_config(page_title="Müşteri Analiz Platformu", layout="centered")

# Başlık
st.markdown("""
    <h1 style='text-align: center; color: #ff4b4b;'>📊 MÜŞTERİ ANALİZ PLATFORMU</h1>
    <p style='text-align: center;'>Bu uygulama, belirli müşteri bilgilerine göre şirketten ayrılma olasılığını tahmin eder.</p>
""", unsafe_allow_html=True)

# Modeli yükle
with open("customer_churn_model.pkl", "rb") as file:
    model_data = pickle.load(file)
    model = model_data["model"]
    expected_features = model_data["features_names"]

# Yan panelde girişler
st.sidebar.header("📌 Müşteri Bilgileri")

user_input = {
    "tenure": st.sidebar.number_input(
        "Şirkette kaldığı süre (ay)", 0, 100, 12,
        help="Müşterinin şirkette kaldığı toplam ay sayısı"
    ),
    "MonthlyCharges": st.sidebar.number_input(
        "Aylık Ücret", 0.0, 1000.0, 75.5,
        help="Müşterinin aylık ödediği hizmet bedeli"
    ),
    "TotalCharges": st.sidebar.number_input(
        "Toplam Ücret", 0.0, 10000.0, 3000.0,
        help="Müşterinin bugüne kadar ödediği toplam tutar"
    ),
    "Contract_Two year": st.sidebar.selectbox(
        "2 Yıllık Sözleşmesi Var mı?", [0, 1],
        help="Müşteri iki yıllık sözleşmeye sahip mi? (1: Evet, 0: Hayır)"
    ),
    "InternetService_Fiber optic": st.sidebar.selectbox(
        "Fiber İnternet Kullanıyor mu?", [0, 1],
        help="Müşteri fiber internet kullanıyor mu? (1: Evet, 0: Hayır)"
    ),
    "PaymentMethod_Electronic check": st.sidebar.selectbox(
        "Elektronik Çek ile Ödeme?", [0, 1],
        help="Ödeme yöntemi olarak elektronik çek kullanıyor mu? (1: Evet, 0: Hayır)"
    )
}

input_df = pd.DataFrame([user_input])

# Eksik kolonları tamamlama
for col in expected_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[expected_features]

# Renk belirleme fonksiyonu
def get_color(prob):
    if prob > 0.75:
        return "🔴", "red"
    elif prob >= 0.5:
        return "🟠", "orange"
    else:
        return "🟢", "green"

# Tahmin butonu
if st.button("Tahmini Gör"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    emoji, color = get_color(probability)

    st.markdown(f"""
        <div style="text-align:center">
            <h3 style="color:{color}">{emoji} Tahmin Skoru: %{probability*100:.2f}</h3>
            <progress value="{probability*100:.2f}" max="100" style="width: 100%; height: 20px; accent-color: {color};"></progress>
        </div>
    """, unsafe_allow_html=True)

    if prediction == 1:
        st.error("🚨 Müşteri kaybedecek gibi görünüyor!")
        st.info("✉ Bu müşteri için sadakat programları, özel teklifler veya iletişime geçilmesi düşünülebilir.")
    else:
        st.success("✅ Müşteri kalacak gibi görünüyor.")
        st.info("✨ Mevcut müşteri memnuniyeti stratejileri etkili olabilir. Bu stratejiler korunmalı.")

# Alt bilgi
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 13px;'>
        Bu uygulama bir iş analitiği projesi kapsamında geliştirilmiştir.
    </div>
""", unsafe_allow_html=True)
