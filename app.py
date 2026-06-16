import html
from datetime import date, time as dtime

import joblib
import pandas as pd
import streamlit as st

# ==========================================================================
# 1. KONFIGURASI HALAMAN & TEMA VISUAL
# ==========================================================================
st.set_page_config(
    page_title="Prediksi Keterlambatan Barang",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    :root {
        --navy: #101826;
        --navy-light: #1c2a3f;
        --amber: #f5a524;
        --danger: #ef4444;
        --safe: #10b981;
        --slate: #475569;
        --bg: #f4f6f9;
    }

    .stApp { background-color: var(--bg); }

    /* ---------- HERO HEADER ---------- */
    .hero {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
        background-image:
            radial-gradient(circle at 1px 1px, rgba(255,255,255,0.06) 1px, transparent 0),
            linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
        background-size: 22px 22px, cover;
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(16, 24, 38, 0.25);
    }
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff;
        font-size: 1.9rem;
        margin: 0 0 6px 0;
        letter-spacing: -0.01em;
    }
    .hero p {
        color: #b9c3d4;
        margin: 0;
        font-size: 0.95rem;
    }
    .hero .accent {
        display: inline-block;
        width: 38px;
        height: 4px;
        background: var(--amber);
        border-radius: 4px;
        margin: 10px 0 0 0;
    }

    /* ---------- CARD ---------- */
    .card {
        background: #ffffff;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 1px 3px rgba(16,24,38,0.08);
        border: 1px solid #e9ecf1;
        margin-bottom: 16px;
    }
    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        color: var(--navy);
        font-size: 1.02rem;
        margin-bottom: 4px;
    }
    .section-sub { color: var(--slate); font-size: 0.85rem; margin-bottom: 14px; }

    /* ---------- RISK BADGE & METER ---------- */
    .risk-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.02em;
    }
    .meter-track {
        position: relative;
        height: 14px;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--safe) 0%, var(--amber) 55%, var(--danger) 100%);
        margin: 14px 0 6px 0;
    }
    .meter-marker {
        position: absolute;
        top: -7px;
        width: 4px;
        height: 28px;
        background: var(--navy);
        border-radius: 2px;
        transform: translateX(-2px);
    }
    .imp-row { display: flex; align-items: center; margin-bottom: 9px; font-size: 0.85rem; }
    .imp-label { width: 230px; color: var(--navy); flex-shrink: 0; }
    .imp-bar-bg { flex: 1; background: #eef1f5; border-radius: 6px; height: 10px; overflow: hidden; margin-right: 10px; }
    .imp-bar-fill { height: 10px; background: linear-gradient(90deg, var(--navy-light), var(--amber)); border-radius: 6px; }
    .imp-pct { width: 48px; text-align: right; color: var(--slate); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>📦 Sistem Cerdas Prediksi Keterlambatan Barang</h1>
        <p>Prediksi risiko keterlambatan pengiriman berdasarkan parameter operasional pesanan — didukung XGBoost Classifier.</p>
        <div class="accent"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# 2. URUTAN FITUR MODEL (HARUS SAMA DENGAN SAAT TRAINING)
# ==========================================================================
FEATURE_ORDER = [
    "product_category",
    "Opsi Pengiriman",
    "Metode Pembayaran",
    "Jumlah",
    "Total Diskon",
    "Total Berat",
    "Ongkos Kirim Dibayar oleh Pembeli",
    "Estimasi Potongan Biaya Pengiriman",
    "Total Pembayaran",
    "Perkiraan Ongkos Kirim",
    "Kota/Kabupaten",
    "Provinsi",
    "source_file",
    "Waktu Pesanan Dibuat_month",
    "Waktu Pesanan Dibuat_day",
    "Waktu Pesanan Dibuat_dow",
    "Waktu Pesanan Dibuat_hour",
]

# Kolom yang tidak terlalu berpengaruh terhadap prediksi (lihat tab "Tentang
# Model") tetap wajib diisi oleh model, tapi tidak perlu membebani pengguna —
# disembunyikan di balik nilai default yang wajar atau dipindah ke bagian opsional.
LOW_IMPACT_DEFAULTS = {
    "Jumlah": 1,
    "Total Berat": 1000,
}


# ==========================================================================
# 3. LOAD MODEL & ENCODER
# ==========================================================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model_xgboost_smote.pkl")
        encoders = joblib.load("encoders.pkl")
        return model, encoders
    except Exception as e:
        st.error(f"❌ Gagal memuat file model/encoder: {e}")
        return None, None


model, encoders = load_assets()


def get_options(col_name):
    if encoders and col_name in encoders:
        return list(encoders[col_name].classes_)
    return ["-"]


def encode_categorical(col_name, value):
    """Encode nilai kategorikal yang dipilih dari dropdown (selalu valid karena
    opsinya memang ditarik dari kelas yang dikenal encoder)."""
    if encoders and col_name in encoders:
        try:
            return int(encoders[col_name].transform([value])[0])
        except Exception:
            return 0
    return value


def encode_numeric_via_label(col_name, numeric_value):
    """'Total Pembayaran' tersimpan di data latih sebagai teks berformat ribuan
    Indonesia (mis. '60.000'), bukan angka murni — sehingga ikut ter-label-encode
    seperti kolom kategorikal. Fungsi ini mencocokkan nilai numerik real yang
    diinput pengguna ke format string yang sama sebelum di-encode, supaya angka
    yang benar-benar dimasukkan pengguna ikut memengaruhi hasil prediksi
    (bukan otomatis jatuh ke nilai default)."""
    if not encoders or col_name not in encoders:
        return numeric_value

    le = encoders[col_name]
    candidates = [
        f"{int(round(numeric_value)):,}".replace(",", "."),  # 60000 -> "60.000"
        str(int(round(numeric_value))),                       # "60000"
        f"{float(numeric_value):.1f}",
        f"{float(numeric_value):.3f}",
    ]
    for cand in candidates:
        try:
            return int(le.transform([cand])[0])
        except Exception:
            continue
    return 0


def default_source_file():
    """source_file hanyalah artefak nama berkas Excel saat penggabungan data
    (bukan parameter operasional nyata), jadi tidak ditampilkan ke pengguna —
    diisi otomatis dengan kelas yang dikenal model."""
    opts = get_options("source_file")
    return opts[0] if opts and opts[0] != "-" else "data.xlsx"


# ==========================================================================
# 4. PEMETAAN WILAYAH (CASCADING PROVINSI -> KOTA/KABUPATEN)
# ==========================================================================
pemetaan_wilayah = {
    "DKI JAKARTA": ["KOTA JAKARTA SELATAN", "KOTA JAKARTA TIMUR", "KOTA JAKARTA PUSAT", "KOTA JAKARTA BARAT", "KOTA JAKARTA UTARA"],
    "JAWA BARAT": ["KAB. KARAWANG", "KOTA BANDUNG", "KAB. BOGOR", "KAB. BEKASI", "KOTA DEPOK", "KAB. BANDUNG BARAT", "KOTA BOGOR", "KOTA BEKASI", "KOTA CIMAHI", "KOTA TASIKMALAYA", "KAB. GARUT"],
    "JAWA TENGAH": ["KOTA SEMARANG", "KAB. BANYUMAS", "KOTA SURAKARTA", "KAB. CILACAP", "KAB. BREBES", "KAB. MAGELANG", "KAB. KENDAL"],
    "BANTEN": ["KAB. TANGERANG", "KOTA TANGERANG SELATAN", "KOTA TANGERANG", "KAB. SERANG", "KOTA CILEGON"],
    "JAWA TIMUR": ["KOTA SURABAYA", "KOTA MALANG", "KAB. SIDOARJO", "KAB. GRESIK", "KAB. JEMBER", "KAB. BANYUWANGI", "KOTA KEDIRI"],
    "DI YOGYAKARTA": ["KOTA YOGYAKARTA", "KAB. SLEMAN", "KAB. BANTUL", "KAB. GUNUNGKIDUL", "KAB. KULON PROGO"],
    "BALI": ["KOTA DENPASAR", "KAB. BADUNG", "KAB. GIANYAR", "KAB. BULELENG"],
    "SUMATERA UTARA": ["KOTA MEDAN", "KAB. DELI SERDANG", "KOTA BINJAI"],
    "SULAWESI SELATAN": ["KOTA MAKASSAR", "KAB. GOWA", "KAB. MAROS"],
}


# ==========================================================================
# 5. SIDEBAR — STATUS, LEGENDA, INFO RINGKAS
# ==========================================================================
with st.sidebar:
    st.markdown("### 📦 Status Sistem")
    if model and encoders:
        st.success("Model & encoder siap digunakan")
    else:
        st.error("Model/encoder belum termuat")

    st.markdown("### 🎯 Legenda Tingkat Risiko")
    st.markdown(
        """
        <div style="font-size:0.85rem; line-height:1.9;">
        🟢 <b>Rendah</b> (&lt; 30%) — kemungkinan terlambat kecil<br>
        🟡 <b>Sedang</b> (30–60%) — perlu dipantau<br>
        🔴 <b>Tinggi</b> (&gt; 60%) — risiko keterlambatan besar
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption(
        "Dibangun dengan XGBoost Classifier + SMOTE untuk menyeimbangkan kelas. "
        "Kelompok 9 · Program Studi S1 Sistem Informasi."
    )


# ==========================================================================
# 6. TABS UTAMA
# ==========================================================================
tab_prediksi, tab_model = st.tabs(["🔮 Cek Risiko Pengiriman", "📊 Tentang Model"])

# --------------------------------------------------------------------------
# TAB 1 — FORM PREDIKSI
# --------------------------------------------------------------------------
with tab_prediksi:
    if not (model and encoders):
        st.warning("Lengkapi file `model_xgboost_smote.pkl` dan `encoders.pkl` di folder yang sama dengan app.py.")
    else:
        st.markdown(
            '<div class="card"><div class="section-label">📝 Parameter Pengiriman</div>'
            '<div class="section-sub">Diurutkan dari yang paling berpengaruh terhadap risiko keterlambatan.</div></div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            # --- Waktu pesanan (fitur paling berpengaruh) -> dihitung dari tanggal & jam REAL, bukan konstanta ---
            c1, c2 = st.columns(2)
            with c1:
                tanggal_input = st.date_input("📅 Tanggal Pesanan", value=date.today())
            with c2:
                jam_input = st.time_input("🕒 Jam Pesanan", value=dtime(14, 0))

            total_pembayaran = st.number_input("💰 Total Pembayaran (IDR)", min_value=0, value=60000, step=1000)
            estimasi_potongan = st.number_input("🎟️ Estimasi Potongan Biaya Pengiriman (IDR)", min_value=0, value=0, step=500)

            provinsi_input = st.selectbox("📍 Provinsi Tujuan", get_options("Provinsi"))

            semua_kota_valid = get_options("Kota/Kabupaten")
            if provinsi_input in pemetaan_wilayah:
                pilihan_kota = [k for k in pemetaan_wilayah[provinsi_input] if k in semua_kota_valid]
                if not pilihan_kota:
                    pilihan_kota = semua_kota_valid
            else:
                pilihan_kota = semua_kota_valid
            kota_input = st.selectbox("🏢 Kota/Kabupaten Tujuan", pilihan_kota)

        with col2:
            perkiraan_ongkir = st.number_input("🧾 Perkiraan Ongkos Kirim (IDR)", min_value=0, value=15000, step=500)
            total_diskon = st.number_input("📉 Total Diskon (IDR)", min_value=0, value=0, step=500)
            ongkir_dibayar = st.number_input("💳 Ongkos Kirim Dibayar Pembeli (IDR)", min_value=0, value=15000, step=500)
            opsi_pengiriman_input = st.selectbox("🚚 Opsi Pengiriman", get_options("Opsi Pengiriman"))

        with st.expander("⚙️ Detail tambahan (opsional — pengaruh terhadap risiko relatif kecil)"):
            d1, d2 = st.columns(2)
            with d1:
                kategori_input = st.selectbox("📦 Kategori Produk", get_options("product_category"))
                jumlah_input = st.number_input("📦 Jumlah Barang", min_value=1, value=LOW_IMPACT_DEFAULTS["Jumlah"])
            with d2:
                metode_bayar_input = st.selectbox("💳 Metode Pembayaran", get_options("Metode Pembayaran"))
                berat_input = st.number_input("⚖️ Total Berat (Gram)", min_value=0, value=LOW_IMPACT_DEFAULTS["Total Berat"])

        st.markdown("")
        run_predict = st.button("🔮 Eksekusi Prediksi Risiko", type="primary", use_container_width=True)

        if run_predict:
            feature_values = {
                "product_category": encode_categorical("product_category", kategori_input),
                "Opsi Pengiriman": encode_categorical("Opsi Pengiriman", opsi_pengiriman_input),
                "Metode Pembayaran": encode_categorical("Metode Pembayaran", metode_bayar_input),
                "Jumlah": jumlah_input,
                "Total Diskon": total_diskon,
                "Total Berat": berat_input,
                "Ongkos Kirim Dibayar oleh Pembeli": ongkir_dibayar,
                "Estimasi Potongan Biaya Pengiriman": estimasi_potongan,
                "Total Pembayaran": encode_numeric_via_label("Total Pembayaran", total_pembayaran),
                "Perkiraan Ongkos Kirim": perkiraan_ongkir,
                "Kota/Kabupaten": encode_categorical("Kota/Kabupaten", kota_input),
                "Provinsi": encode_categorical("Provinsi", provinsi_input),
                "source_file": encode_categorical("source_file", default_source_file()),
                "Waktu Pesanan Dibuat_month": tanggal_input.month,
                "Waktu Pesanan Dibuat_day": tanggal_input.day,
                "Waktu Pesanan Dibuat_dow": tanggal_input.weekday(),
                "Waktu Pesanan Dibuat_hour": jam_input.hour,
            }
            input_data = pd.DataFrame([feature_values])[FEATURE_ORDER]

            try:
                pred = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                p_late = float(proba[1])

                if p_late < 0.30:
                    tier, color, note = "RENDAH", "#10b981", "Kemungkinan keterlambatan kecil — pesanan dapat diproses normal."
                elif p_late < 0.60:
                    tier, color, note = "SEDANG", "#f5a524", "Perlu dipantau — pertimbangkan konfirmasi ketersediaan kurir."
                else:
                    tier, color, note = "TINGGI", "#ef4444", "Risiko keterlambatan besar — pertimbangkan opsi pengiriman ekspres atau koordinasi ulang dengan kurir."

                label_text = "TERLAMBAT / BERMASALAH" if pred == 1 else "TEPAT WAKTU"

                st.markdown(
                    f"""
                    <div class="card">
                        <div class="section-label">Hasil Analisis</div>
                        <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-top:10px;">
                            <div>
                                <span class="risk-badge" style="background:{color}1a; color:{color};">RISIKO {tier}</span>
                                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.3rem; color:var(--navy); margin-top:8px;">
                                    Prediksi: {label_text}
                                </div>
                            </div>
                            <div style="text-align:right;">
                                <div style="color:var(--slate); font-size:0.8rem;">Probabilitas Terlambat</div>
                                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.8rem; color:{color};">{p_late*100:.1f}%</div>
                            </div>
                        </div>
                        <div class="meter-track">
                            <div class="meter-marker" style="left:{p_late*100:.1f}%;"></div>
                        </div>
                        <div style="color:var(--slate); font-size:0.85rem; margin-top:10px;">{note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"⚠️ Terjadi kesalahan saat memprediksi: {e}")

# --------------------------------------------------------------------------
# TAB 2 — INFORMASI MODEL (dihitung langsung dari model yang dimuat)
# --------------------------------------------------------------------------
with tab_model:
    if not (model and encoders):
        st.info("Model belum termuat — informasi tidak tersedia.")
    else:
        st.markdown(
            '<div class="card"><div class="section-label">🧠 Ringkasan Model</div>'
            '<div class="section-sub">Algoritma <b>XGBoost Classifier</b>, dilatih dengan teknik <b>SMOTE</b> '
            "untuk menyeimbangkan kelas data minoritas (pesanan terlambat/bermasalah lebih sedikit dari yang "
            "tepat waktu). Target label biner: 0 = Tepat Waktu, 1 = Terlambat/Batal.</div></div>",
            unsafe_allow_html=True,
        )

        try:
            importances = pd.Series(model.feature_importances_, index=FEATURE_ORDER)
            importances = importances.sort_values(ascending=False)
            total_imp = importances.sum() or 1.0

            rows_html = ""
            for feat, val in importances.head(10).items():
                pct = val / total_imp * 100
                rows_html += (
                    '<div class="imp-row">'
                    f'<div class="imp-label">{html.escape(feat)}</div>'
                    f'<div class="imp-bar-bg"><div class="imp-bar-fill" style="width:{pct:.1f}%;"></div></div>'
                    f'<div class="imp-pct">{pct:.1f}%</div>'
                    "</div>"
                )

            st.markdown(
                '<div class="card"><div class="section-label">📈 Feature Importance (dihitung langsung dari model aktif)</div>'
                f'<div style="margin-top:14px;">{rows_html}</div></div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.caption(f"Feature importance tidak dapat dihitung: {e}")

        st.markdown(
            '<div class="card"><div class="section-label">ℹ️ Catatan</div>'
            '<div class="section-sub" style="margin-bottom:0;">Model ini dilatih pada data historis e-commerce '
            "logistik dalam jumlah terbatas, sehingga sebaiknya digunakan sebagai alat bantu pendukung keputusan, "
            "bukan satu-satunya dasar keputusan operasional.</div></div>",
            unsafe_allow_html=True,
        )