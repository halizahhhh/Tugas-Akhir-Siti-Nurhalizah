import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import re
import string
import gc
import plotly.express as px
from transformers import AutoTokenizer, AutoModel
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Analisis Sentimen EDOM", layout="wide")

# =========================
# SESSION STATE
# =========================
if 'mulai_proses' not in st.session_state:
    st.session_state.mulai_proses = False
if 'df_hasil' not in st.session_state:
    st.session_state.df_hasil = None
if 'last_file' not in st.session_state:
    st.session_state.last_file = None
if 'jumlah_awal' not in st.session_state:
    st.session_state.jumlah_awal = 0
if 'jumlah_bersih' not in st.session_state:
    st.session_state.jumlah_bersih = 0
if 'jumlah_hapus' not in st.session_state:
    st.session_state.jumlah_hapus = 0

# =========================
# LOAD MODEL & TOOLS
# =========================
@st.cache_resource
def load_models():
    # Load model KNN
    model_knn = joblib.load('model/knn_indobert.pkl')

    # Load IndoBERT
    model_name = "indobenchmark/indobert-base-p2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    bert_model.eval()

    # Load Sastrawi Tools
    stemmer = StemmerFactory().create_stemmer()
    stop_factory = StopWordRemoverFactory()
    stopwords = set(stop_factory.get_stop_words())

    return model_knn, tokenizer, bert_model, stemmer, stopwords

knn_model, tokenizer, bert_model, stemmer, stopwords = load_models()

# =========================
# KAMUS NORMALISASI
# =========================
normalisasi = {
    "gk": "tidak", "ga": "tidak", "nggak": "tidak", "bgt": "banget", "bngt": "banget",
    "bener": "benar", "dosenny": "dosen nya", "mahasiswanya": "mahasiswa nya",
    "bgs": "bagus", "bagusss": "bagus", "baguss": "bagus", "bkin": "bikin", "bgin": "bikin",
    "bbrp": "beberapa", "dosenx": "dosen", "dosenya": "dosen", "mhs": "mahasiswa",
    "mhsw": "mahasiswa", "bnyk": "banyak", "bnk": "banyak", "banyakkk": "banyak",
    "krn": "karena", "krena": "karena", "karna": "karena", "jg": "juga", "jga": "juga",
    "jugaaa": "juga", "sbnrnya": "sebenarnya", "sbnr": "sebenarnya", "brrti": "berarti",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah", "blm": "belum", "belomm": "belum",
    "kl": "kalau", "klo": "kalau", "klu": "kalau", "kalo": "kalau", "aj": "saja",
    "aja": "saja", "sj": "saja", "mksd": "maksud", "mksud": "maksud",
    "ngajar": "mengajar", "ngasih": "memberikan", "ngambil": "mengambil",
    "njelasin": "menjelaskan", "jelasin": "menjelaskan", "mantep": "mantap", "mantab": "mantap",
    "keren banget": "sangat baik", "baik banget": "sangat baik", "ok": "oke",
    "oke banget": "sangat baik", "lumayan kok": "cukup baik", "cukup baik lah": "cukup baik",
    "td": "tadi", "skr": "sekarang", "nnti": "nanti", "materinya": "materi",
    "penjelasannya": "penjelasan", "pembelajarannya": "pembelajaran", "tp": "tapi", "tpi": "tapi"
}

# =========================
# PREPROCESSING
# =========================

def remove_undefined_text(text):
    if pd.isna(text):
        return None

    t = str(text).strip().lower()

    invalid = [
        '-', '.', ',', '..', '...', '', ' ', '"."', '_', '__',
        'tidak ada', 'cukup', 'tidak', 'kosong', 'tidaj ada'
    ]

    if len(t) <= 3:
        return None

    if len(t.split()) <= 3 and t in invalid:
        return None

    return text

def normalize_text(text):
    text = text.lower()
    for key, value in normalisasi.items():
        text = re.sub(rf"\b{re.escape(key)}\b", value, text)
    return text

def clean_text(text):
    # Case folding
    text = str(text).lower()
    # Hilangkan angka
    text = re.sub(r'\d+', ' ', text)
    # Hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Hapus simbol non-alfanumerik
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalisasi kata (setelah hapus tanda baca, sesuai notebook)
    text = normalize_text(text)
    # Trim spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_pipeline(text):
    """
    Pipeline preprocessing lengkap sesuai urutan notebook:
    remove_undefined → clean_text → tokenisasi → stopword removal → stemming per token
    """
    # STEP 1: Validasi teks tidak bermakna
    text = remove_undefined_text(text)
    if text is None:
        return None

    # STEP 2: Whitespace normalisasi awal
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)

    # STEP 3: Clean text (lowercase, hapus angka, tanda baca, normalisasi)
    text = clean_text(text)

    # STEP 4: Tokenisasi sederhana
    tokens = text.split()

    if not tokens:
        return None

    # STEP 5: Stopword removal
    tokens_filtered = [w for w in tokens if w not in stopwords]

    if not tokens_filtered:
        return None

    tokens_stemmed = [stemmer.stem(w) for w in tokens_filtered]

    return " ".join(tokens_stemmed)


# =========================
# EMBEDDING (BERT)
# =========================
def get_embeddings(texts, batch_size=8):
    if len(texts) == 0:
        return np.array([])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            out = bert_model(**inputs)
            # Mean pooling dari last_hidden_state (sesuai notebook)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(emb)

        del inputs, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.vstack(all_embeddings)


# =========================
# UI & MAIN LOGIC
# =========================
st.title("📊 Sistem Analisis Sentimen Dosen (IndoBERT + KNN)")
st.markdown("Aplikasi ini menggunakan model yang telah dilatih untuk mengklasifikasi komentar mahasiswa.")

st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    if st.session_state.last_file != uploaded_file.name:
        st.session_state.mulai_proses = False
        st.session_state.df_hasil = None
        st.session_state.last_file = uploaded_file.name
        st.session_state.jumlah_awal = 0
        st.session_state.jumlah_bersih = 0
        st.session_state.jumlah_hapus = 0

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if 'Dosen' not in df.columns or 'Komentar' not in df.columns:
        st.error("Kolom harus ada: 'Dosen' & 'Komentar'")
        st.stop()

    st.success(f"Berhasil memuat {len(df)} data")

    if st.button("🚀 Jalankan Analisis Sentimen"):
        st.session_state.mulai_proses = True

    if st.session_state.mulai_proses:
        with st.spinner("Memproses data sesuai standar preprocessing notebook..."):

            # STEP 1: Simpan jumlah awal
            st.session_state.jumlah_awal = len(df)

            # STEP 2: Hapus duplikat berdasarkan kolom Komentar
            df_proc = df.drop_duplicates(subset='Komentar').reset_index(drop=True).copy()

            # STEP 3: Anonimisasi Dosen 
            unique_dosen = df_proc['Dosen'].unique()
            anon_map = {d: f"Dosen_{i+1:02d}" for i, d in enumerate(unique_dosen)}
            df_proc['Dosen_Anonim'] = df_proc['Dosen'].map(anon_map)

            # STEP 4: Jalankan full preprocessing pipeline sesuai notebook
            df_proc['clean_text'] = df_proc['Komentar'].apply(preprocess_pipeline)

            # STEP 5: Hitung statistik
            st.session_state.jumlah_bersih = df_proc['clean_text'].notnull().sum()
            st.session_state.jumlah_hapus = st.session_state.jumlah_awal - st.session_state.jumlah_bersih

            # STEP 6: Filter data valid untuk prediksi
            df_valid = df_proc[df_proc['clean_text'].notnull()].copy()

            if not df_valid.empty:
                # STEP 7: Embedding dengan IndoBERT
                vectors = get_embeddings(df_valid['clean_text'].tolist())

                # STEP 8: Prediksi dengan KNN
                preds = knn_model.predict(vectors)

                # STEP 9: Mapping label
                df_valid['sentimen'] = pd.Series(
                    preds, index=df_valid.index
                ).map({1: "Positif", 0: "Negatif"})

                # Gabungkan kembali ke dataframe utama
                df_proc['sentimen'] = None
                df_proc.loc[df_valid.index, 'sentimen'] = df_valid['sentimen']
            else:
                st.warning("Tidak ada data valid setelah preprocessing.")

            st.session_state.df_hasil = df_proc

    # =========================
    # OUTPUT DISPLAY
    # =========================
    if st.session_state.df_hasil is not None:
        df_res = st.session_state.df_hasil

        # METRIC SECTION
        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Data Awal", st.session_state.jumlah_awal)
        col2.metric("Data Valid (Bersih)", st.session_state.jumlah_bersih)
        col3.metric("Data Tereliminasi", st.session_state.jumlah_hapus)

        st.divider()

        # RINGKASAN & CHART
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Ringkasan Data")
            st.dataframe(
                df_res[['Dosen_Anonim', 'Komentar', 'clean_text', 'sentimen']],
                use_container_width=True
            )

        with c2:
            st.subheader("Distribusi Sentimen Keseluruhan")
            df_chart = df_res[df_res['sentimen'].notnull()]
            if not df_chart.empty:
                fig = px.pie(
                    df_chart, names='sentimen', hole=0.4, color='sentimen',
                    color_discrete_map={'Positif': '#2ca02c', 'Negatif': '#d62728'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Tidak ada data valid untuk ditampilkan di chart.")

        # FILTER PER DOSEN
        st.divider()
        st.subheader("Filter per Dosen")
        selected = st.selectbox(
            "Pilih Dosen",
            sorted(df_res['Dosen_Anonim'].dropna().unique())
        )

        df_filter = df_res[df_res['Dosen_Anonim'] == selected]
        fc1, fc2 = st.columns([2, 1])
        with fc1:
            st.dataframe(
                df_filter[['Komentar', 'clean_text', 'sentimen']],
                use_container_width=True
            )
        with fc2:
            df_chart2 = df_filter[df_filter['sentimen'].notnull()]
            if not df_chart2.empty:
                fig2 = px.pie(
                    df_chart2, names='sentimen', hole=0.4, color='sentimen',
                    color_discrete_map={'Positif': '#2ca02c', 'Negatif': '#d62728'}
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Tidak ada prediksi untuk dosen ini.")

        # DOWNLOAD BUTTON
        st.divider()
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Analisis (.csv)",
            data=csv,
            file_name="hasil_sentimen.csv",
            mime="text/csv"
        )

else:
    st.info("Silakan upload file Excel (.xlsx) terlebih dahulu.")