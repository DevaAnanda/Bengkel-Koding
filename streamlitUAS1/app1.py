import streamlit as st
import pandas as pd
import joblib
import os

if not os.path.exists("randomforest_obesity_model.pkl"):
    st.error("Model belum ditemukan! Pastikan file .pkl tersedia.")
else:
    model = joblib.load("randomforest_obesity_model.pkl")

st.title("🩺 Prediksi Tipe Obesitas")
st.write("Masukkan data gaya hidup kamu di bawah ini:")

# -------------------------------
# Form input manual tanpa input_columns.pkl
# -------------------------------
# Kolom input sesuai dengan training
user_input = {
    'Age': st.number_input("Umur", min_value=0, max_value=120, value=25),
    'Height': st.number_input("Tinggi Badan (m)", min_value=1.0, max_value=2.5, value=1.70),
    'Weight': st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0),
    'FCVC': st.slider("Frekuensi konsumsi sayur (1–3)", 1.0, 3.0, 2.0),
    'NCP': st.slider("Jumlah makan per hari (1–4)", 1.0, 4.0, 3.0),
    'CH2O': st.slider("Minum air (liter/hari)", 1.0, 5.0, 2.0),
    'FAF': st.slider("Aktivitas fisik (jam/minggu)", 0.0, 10.0, 1.0),
    'TUE': st.slider("Waktu di layar/gadget (jam/hari)", 0.0, 10.0, 1.0),

    'Gender': st.selectbox("Jenis Kelamin", ["Male", "Female"]),
    'family_history_with_overweight': st.selectbox("Riwayat keluarga overweight", ["yes", "no"]),
    'FAVC': st.selectbox("Sering makan makanan berkalori tinggi?", ["yes", "no"]),
    'CAEC': st.selectbox("Kebiasaan ngemil", ["no", "Sometimes", "Frequently", "Always"]),
    'SMOKE': st.selectbox("Merokok?", ["yes", "no"]),
    'SCC': st.selectbox("Monitoring kalori?", ["yes", "no"]),
    'CALC': st.selectbox("Konsumsi alkohol", ["no", "Sometimes", "Frequently", "Always"]),
    'MTRANS': st.selectbox("Transportasi harian", ["Walking", "Bike", "Motorbike", "Automobile", "Public_Transportation"])
}

# Ubah ke DataFrame
df_user = pd.DataFrame([user_input])

# -------------------------------
# Prediksi
# -------------------------------
if st.button("🔍 Prediksi"):
    prediction = model.predict(df_user)[0]
    st.success(f"**Hasil Prediksi:** {prediction}")

    # Deskripsi hasil
    obesity_desc = {
        "Insufficient_Weight": "Berat badan kurang. Cek asupan nutrisi.",
        "Normal_Weight": "Berat badan ideal. Pertahankan gaya hidup sehat!",
        "Overweight_Level_I": "Kelebihan berat badan ringan. Perlu perhatian.",
        "Overweight_Level_II": "Kelebihan berat badan sedang. Mulai waspada.",
        "Obesity_Type_I": "Obesitas tingkat 1. Perlu penyesuaian pola hidup.",
        "Obesity_Type_II": "Obesitas tingkat 2. Perlu pemantauan medis.",
        "Obesity_Type_III": "Obesitas tingkat 3. Harus segera konsultasi medis."
    }
    st.info(obesity_desc.get(prediction, "Deskripsi tidak tersedia."))
