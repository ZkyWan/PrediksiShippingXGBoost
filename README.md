# 📦 Sistem Analisis & Prediksi Logistik Real-Time (Kelompok 9)

Aplikasi berbasis web interaktif yang dirancang untuk memprediksi risiko penundaan (*shipping delay*) pada operasional pengiriman e-commerce secara real-time. Proyek ini mengintegrasikan model machine learning mutakhir ke dalam antarmuka web yang siap pakai untuk membantu pengambilan keputusan logistik yang lebih tangkas.

🔗 **Link Aplikasi:** [https://prediksi-shipping-logistik-xgboost.streamlit.app/]

## 🚀 Fitur Utama
* **Ekstraksi Fitur Dinamis:** Sistem secara otomatis memuat top 5 fitur paling berpengaruh (*feature importance*) langsung dari arsitektur model untuk efisiensi input pengguna.
* **Kalkulasi ETA Real-Time:** Menggunakan probabilitas statistik dari model untuk menghitung estimasi waktu perjalanan secara dinamis.
* **Dashboard Interaktif:** Visualisasi metrik evaluasi final dan grafik performa model klasifikasi langsung dalam satu halaman.

## 📊 Performa Model (XGBoost Classifier)
Model dilatih menggunakan algoritma **XGBoost** dan menghasilkan performa optimal pada data uji dengan metrik sebagai berikut:
* **Akurasi:** 85.20% (Kemampuan umum model memprediksi dengan benar)
* **Recall (Metrik Utama):** 0.871 (Sangat kuat dalam meminimalkan risiko lolosnya paket yang berpotensi terlambat)
* **ROC-AUC:** 0.891 (Kemampuan diskriminasi kelas yang sangat tinggi)

## 🛠️ Tech Stack
* **Language:** Python
* **Framework UI:** Streamlit
* **Libraries:** XGBoost, Scikit-Learn, Pandas, NumPy, Joblib
* **Deployment:** Streamlit Community Cloud

💡 Highlight Teknis Proyek:
1. Menggunakan algoritma XGBoost Classifier yang berhasil mencapai nilai Akurasi 85.20% dan ROC-AUC 0.891.
2. Fokus pada optimasi metrik Recall (0.871) untuk memastikan sistem sangat sensitif dalam mendeteksi paket yang berisiko terlambat (meminimalkan False Negatives).
3. Implementasi penanganan dimensi array dinamis secara langsung di production, sehingga mencegah terjadinya error shape saat deployment di Streamlit Cloud.

Proyek ini menjadi langkah penting bagi saya dalam memahami siklus end-to-end Data Science, mulai dari data preparation, modeling, hingga model deployment.

Silakan coba aplikasinya secara langsung di sini: [Masukkan Link Streamlit Kamu]
Kritik dan saran sangat saya harapkan untuk pengembangan model ini ke depan! 💻✨

#DataScience #MachineLearning #XGBoost #Streamlit #Logistics #Python #Portfolio
