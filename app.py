import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime

# ==========================================
# KONFIGURASI HALAMAN UTAMA DASHBOARD
# ==========================================
st.set_page_config(page_title="Dashboard Logistik Real-Time", page_icon="📦", layout="wide")

# Menyembunyikan komponen tombol plus/minus bawaan agar form terlihat bersih
st.markdown("""
    <style>
    button[data-testid="stNumberInputStepUp"], 
    button[data-testid="stNumberInputStepDown"] { display: none !important; }
    input::-webkit-outer-spin-button,
    input::-webkit-inner-spin-button { -webkit-appearance: none !important; margin: 0 !important; }
    input[type=number] { -moz-appearance: textfield !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. MEMUAT ARTEFAK MODEL SEPERTI PADA DATASET ASLI
# ==========================================
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load("xgb_model_ecom_fix.pkl")
        fitur_names = joblib.load("fitur_model_fix.pkl")
        encoders = joblib.load("encoders.pkl")
        
        # Mengambil urutan 5 fitur paling berpengaruh langsung dari model
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_5_fitur = [fitur_names[i] for i in indices[:5]]
        
        return model, fitur_names, top_5_fitur, encoders
    except Exception as e:
        st.error(f"⚠️ Gagal memuat file model atau komponen dataset: {e}")
        return None, [], [], {}

model, fitur_names, top_5_fitur, encoders = load_model_artifacts()

# Nilai Evaluasi Model Final dari Data Uji (Notebook)
METRIK_EVALUASI = {
    "Accuracy": 0.852,  
    "Precision": 0.834,
    "Recall": 0.871,   
    "F1_Score": 0.852,
    "ROC_AUC": 0.891
}

# ==========================================
# 2. ANTARMUKA PENGGUNA (UI LAYOUT)
# ==========================================
st.title("📦 Sistem Analisis & Prediksi Logistik Real-Time")
st.markdown("**Kelompok 9** — Model Klasifikasi Keandalan Pengiriman (XGBoost)")
st.divider()

col_input, col_metrik = st.columns([1.2, 1])

# --- KOLOM KIRI: FORM INPUT INPUT DINAMIS ---
with col_input:
    st.subheader("📝 Input Parameter Pengiriman Baru")
    
    data_input_user = {}
    DAFTAR_PROVINSI = [
        "Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Kepulauan Riau",
        "Jambi", "Bengkulu", "Sumatera Selatan", "Kepulauan Bangka Belitung", "Lampung",
        "DKI Jakarta", "Banten", "Jawa Barat", "Jawa Tengah", "DI Yogyakarta", "Jawa Timur",
        "Bali", "Nusa Tenggara Barat", "Nusa Tenggara Timur",
        "Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan", "Kalimantan Timur", "Kalimantan Utara",
        "Sulawesi Utara", "Gorontalo", "Sulawesi Tengah", "Sulawesi Barat", "Sulawesi Selatan", "Sulawesi Tenggara",
        "Maluku", "Maluku Utara", "Papua", "Papua Barat", "Papua Selatan", "Papua Tengah", "Papua Pegunungan", "Papua Barat Daya"
    ]
    
    # Perulangan pembuatan form secara dinamis hanya untuk fitur utama
    for fitur in top_5_fitur:
        label = fitur.replace("_", " ").title()
        
        # Kondisi jika variabel bertipe wilayah/provinsi
        if "provinsi" in fitur.lower() or "province" in fitur.lower():
            pilihan_teks = st.selectbox(f"📍 {label}", options=DAFTAR_PROVINSI)
            if "provinsi" in encoders:
                try:
                    data_input_user[fitur] = float(encoders["provinsi"].transform([pilihan_teks])[0])
                except:
                    data_input_user[fitur] = float(DAFTAR_PROVINSI.index(pilihan_teks))
            else:
                data_input_user[fitur] = float(DAFTAR_PROVINSI.index(pilihan_teks))
                
        # Kondisi jika variabel bertipe penanggalan/hari pembuatan pesanan
        elif "pesanan" in fitur.lower() or "order" in fitur.lower() or "buat" in fitur.lower():
            tanggal_buat = st.date_input(f"📅 {label}", datetime.date.today())
            data_input_user[fitur] = float(tanggal_buat.day)
            
        # Kondisi variabel numerik logistik dasar lainnya (berbentuk Integer)
        else:
            data_input_user[fitur] = st.number_input(f"🔢 {label}", value=0, step=1)
            
    st.markdown("<br>", unsafe_allow_html=True)
    tanggal_kirim = st.date_input("🚚 Tanggal Paket Mulai Dikirim", datetime.date.today())
    
    submit_button = st.button("🚀 Proses Prediksi & Analisis Real-Time", type="primary", use_container_width=True)

# --- KOLOM KANAN: GRAFIK DAN PERFORMA MODEL ---
with col_metrik:
    st.subheader("📊 Metrik Evaluasi Model Final")
    
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Akurasi", value=f"{METRIK_EVALUASI['Accuracy']*100:.2f}%")
    m2.metric(label="Recall (Utama)", value=METRIK_EVALUASI['Recall'])
    m3.metric(label="ROC-AUC", value=METRIK_EVALUASI['ROC_AUC'])
    
    st.markdown("<br>**Grafik Performa Model Klasifikasi**", unsafe_allow_html=True)
    df_chart = pd.DataFrame(
        list(METRIK_EVALUASI.values()), 
        index=list(METRIK_EVALUASI.keys()), 
        columns=["Skor Metrik"]
    )
    st.bar_chart(df_chart, height=240)

# ==========================================
# 3. LOGIKA PREDIKSI KELAS & KALKULASI ETA DINAMIS
# ==========================================
if submit_button:
    if model is None:
        st.error("Gagal mengeksekusi. Model tidak termuat dengan sempurna.")
    else:
        # Penanganan otomatis dimensi array data (1, 24) agar tidak memicu error shape
        full_input_data = {}
        for fitur in fitur_names:
            if fitur in top_5_fitur:
                full_input_data[fitur] = data_input_user[fitur]
            else:
                full_input_data[fitur] = 0.0
                
        df_input = pd.DataFrame([full_input_data])[fitur_names]
        
        # Eksekusi kalkulasi model matematika
        prediksi = model.predict(df_input)[0]
        probabilitas = model.predict_proba(df_input)[0]
        
        # Estimasi durasi murni dari probabilitas statistik real-time model
        prob_terlambat = probabilitas[1] 
        waktu_dasar_hari = 2 
        maksimal_penundaan = 8 
        
        kalkulasi_hari = waktu_dasar_hari + (prob_terlambat * maksimal_penundaan)
        durasi_hari = int(round(kalkulasi_hari)) 
        
        st.divider()
        st.subheader("🎯 Hasil Prediksi & Validasi Logistik")
        
        res_col1, res_col2 = st.columns(2)
        
        if prediksi == 0:
            status = "Tepat Waktu (Kelas 0)"
            conf = probabilitas[0] * 100
            res_col1.success(f"### ✅ {status}\n**Tingkat Keyakinan: {conf:.1f}%**")
            res_col1.write("Berdasarkan kombinasi parameter matriks data input, kondisi pengantaran logistik diprediksi berjalan stabil.")
        else:
            status = "Terlambat (Kelas 1)"
            conf = probabilitas[1] * 100
            res_col1.error(f"### ⚠️ {status}\n**Tingkat Keyakinan: {conf:.1f}%**")
            res_col1.write("Sistem mendeteksi adanya risiko penundaan operasional pengiriman berdasarkan pola matriks data input.")
            
        # Kalkulasi kalender real-time kedatangan barang (ETA)
        tanggal_estimasi_sampai = tanggal_kirim + datetime.timedelta(days=durasi_hari)
        
        with res_col2:
            st.info(f"""
            **📅 Rencana Garis Waktu Pengiriman (ETA):**
            - Tanggal Mulai Dikirim: **{tanggal_kirim.strftime('%d %B %Y')}**
            - Prediksi Tiba di Tujuan: **{tanggal_estimasi_sampai.strftime('%d %B %Y')}**
            - *Estimasi Waktu Perjalanan Dinamis: {durasi_hari} Hari.*
            """)
            
        # Tampilan Ringkasan Data Input Objektif (Murni dari Dataset)
        st.markdown("#### 🔍 Ringkasan Karakteristik Input Operasional")
        for tf in top_5_fitur:
            label_fitur = tf.replace('_', ' ').title()
            if "provinsi" in tf.lower() or "province" in tf.lower():
                st.markdown(f"- **{label_fitur}**: {DAFTAR_PROVINSI[int(data_input_user[tf])]}")
            elif "pesanan" in tf.lower() or "order" in tf.lower() or "buat" in tf.lower():
                st.markdown(f"- **{label_fitur} (Ekstraksi Hari)**: Nilai {int(data_input_user[tf])}")
            else:
                st.markdown(f"- **{label_fitur}**: {int(data_input_user[tf])}")