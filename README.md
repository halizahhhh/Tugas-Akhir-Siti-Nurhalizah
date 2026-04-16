**Perbandingan Kinerja IndoBERT dan TF-IDF dalam Klasifikasi Sentimen EDOM Menggunakan KNN**

**Deskripsi Proyek**
Repository ini berisi implementasi penelitian mengenai perbandingan kinerja dua metode representasi teks, yaitu IndoBERT dan TF-IDF, dalam tugas klasifikasi sentimen pada data Evaluasi Dosen oleh Mahasiswa (EDOM).
Klasifikasi dilakukan menggunakan algoritma K-Nearest Neighbor (KNN) untuk mengetahui kombinasi metode yang paling optimal dalam mengolah data teks berbahasa Indonesia.

**Tujuan Penelitian**
Penelitian ini bertujuan untuk:
1. Membandingkan kinerja representasi teks IndoBERT dan TF-IDF dalam mengklasifikasikan sentimen pada data EDOM.
2. Menentukan model klasifikasi sentimen yang paling optimal dengan membandingkan:
- IndoBERT + KNN
- TF-IDF + KNN

**Metode yang Digunakan**
1. Representasi Teks
- TF-IDF (*Term Frequency - Inverse Document Frequency*) : Mengubah teks menjadi fitur numerik berdasarkan frekuensi kata.
IndoBERT
- Model berbasis Transformer yang menghasilkan representasi teks berbasis konteks.
2. Algoritma Klasifikasi
*K-Nearest Neighbor* (KNN) : Digunakan untuk mengklasifikasikan sentimen berdasarkan kedekatan antar data.

**Alur Penelitian**
Proses dalam notebook mengikuti tahapan berikut:
1. Data Collection
2. Data Cleaning & Preprocessing
3. Pelabelan Sentimen
4. Split Dataset (Training & Testing)
5. Representasi Teks:
- TF-IDF
- IndoBERT
6. Validasi dengan *Cross-Validation*
7. Klasifikasi menggunakan KNN
8. Evaluasi Model
9. Perbandingan Hasil
