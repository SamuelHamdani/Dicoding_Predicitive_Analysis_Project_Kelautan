# Laporan Proyek Machine Learning - Kelautan - Samuel Christian Hamdani

---

# Project Overview

---

### Latar Belakang
Kualitas air kolam merupakan faktor krusial dalam keberhasilan budidaya ikan. Parameter seperti pH, suhu, dan tingkat kekeruhan air sangat mempengaruhi kesehatan dan pertumbuhan ikan. Kondisi air yang tidak sesuai dapat menyebabkan stres, penyakit, bahkan kematian pada ikan. Oleh karena itu, pemantauan kualitas air secara berkala sangat diperlukan untuk menjaga keberlangsungan ekosistem kolam dan meningkatkan produktivitas budidaya.

Namun, pengawasan kualitas air secara manual membutuhkan waktu, tenaga, dan keterampilan teknis yang tidak sedikit. Di sisi lain, perkembangan teknologi analisis data dan machine learning saat ini memungkinkan pemanfaatan data historis untuk membangun model prediktif yang dapat membantu dalam pengambilan keputusan secara cepat dan akurat.

Melalui proyek ini, dikembangkan sebuah model klasifikasi prediktif untuk menilai tingkat kualitas air kolam (Baik, Cukup, atau Buruk) berdasarkan data parameter fisik-kimia air dan jenis ikan yang dibudidayakan. Model ini dibangun menggunakan beberapa algoritma machine learning, seperti K-Nearest Neighbor (KNN), Random Forest, dan AdaBoost, untuk membandingkan performa masing-masing dan menentukan algoritma yang paling tepat digunakan dalam konteks ini.

Hasil dari proyek ini diharapkan dapat memberikan solusi berbasis data untuk mempermudah pemilik kolam dalam memantau kualitas air dan mengambil tindakan preventif secara efisien, sehingga mendukung praktik budidaya ikan yang lebih sehat, produktif, dan berkelanjutan.

### Problem Statements


1.   Kualitas air kolam yang buruk dapat menyebabkan pertumbuhan ikan terganggu, stres, bahkan kematian, yang berdampak pada kerugian ekonomi bagi pembudidaya.
2.   Pemantauan kualitas air secara manual membutuhkan waktu, tenaga, dan keahlian yang tidak semua pemilik kolam miliki.

### Goals

1. Mengembangkan sebuah model prediksi kualitas air kolam berdasarkan parameter pH, suhu, kekeruhan, dan jenis ikan untuk membantu mendeteksi potensi penurunan kualitas air yang dapat memengaruhi kesehatan dan pertumbuhan ikan.
2. Memberikan solusi otomatis dan efisien dalam pemantauan kualitas air, sehingga pemilik kolam tidak perlu melakukan pemeriksaan manual yang memakan waktu dan memerlukan keahlian khusus.


### Solution Statements
1. Mengembangkan model **klasifikasi** prediktif dengan pendekatan supervised learning untuk mengkategorikan kualitas air kolam menjadi Baik, Cukup, dan Buruk.
2. Melatih dan menguji tiga algoritma machine learning (KNN, Random Forest, dan AdaBoost) dengan dataset kualitas air untuk memperoleh performa terbaik.


# Data Understanding

---

Pada tahap ini dilakukan eksplorasi awal terhadap data yang digunakan untuk membangun model prediktif. Dataset yang digunakan berisi beberapa fitur penting yang memengaruhi kualitas air kolam dan jenis ikan yang dibudidayakan.

## Sumber Data
Sumber Dataset : https://www.kaggle.com/datasets/monirmukul/realtime-pond-water-dataset-for-fish-farming?select=realfishdataset.csv

Dataset yang diambil berasal dari sumber open source yaitu Kaggle yang menyimpan hanya 1 dataset bernama 'realfishdataset.csv'. Data diambil melalui proses monitoring kualitas air pada tempat budidaya ikan secara realtime.

## Jumlah Data

Jumlah Kolom : Pada dataset terdapat jumlah kolom sebanyak 4 kolom yang menyimpan detail kualitas air kolam seperti:
1. pH
2. Temperature (Suhu)
3. Turbidity (Kekeruhan)
4. Fish (Jenis Ikan)

Jumlah Baris : Sebanyak 591 baris yang menyimpan data pencatatan kualitas air kolam pada dataset ini.

Kondisi Data: 

1. Nilai Missing Value : Pada dataset tidak terdapat missing value (nilai kosong) setelah dilakukan pemeriksaan nilai missing value.
2. Nilai Duplikat : Pada dataset, terdapat data duplikat sebanyak 304 data. Data yang terduplikat akan dihapus dari dataset dan akan menyisakan 287 data untuk diolah.
3. Nilai Outlier : Pada dataset, terdapat beberapa nilai yang dikategorikan sebagai outlier pada data pH, suhu, dan kekeruhan. Dengan adanya nilai outlier akan dilakukan penanganan dengan metode IQR (Index Quartile Range) yang berguna untuk menghapus data yang tersimpang terlalu jauh dari nilai rata-rata.

## Deskripsi Fitur
Berikut adalah penjelasan masing-masing fitur dalam dataset:
1. Jenis_Ikan (categorical): Menyatakan jenis ikan yang dibudidayakan di kolam.
2. pH (numerical): Mengukur tingkat keasaman air kolam. Nilai pH yang ideal biasanya berada pada kisaran 6,5–8,5.
3. Suhu (numerical): Suhu air kolam yang berpengaruh terhadap metabolisme ikan. Suhu optimal umumnya sekitar 20–30°C, tergantung jenis ikan.
4. Kekeruhan (numerical): Menunjukkan kejernihan air kolam. Air yang terlalu keruh dapat menghambat fotosintesis dan membahayakan ikan.

## Proses Eksplorasi Awal
Pada bagian ini, dilakukan beberapa tahapan awal yang dilakukan sebelum pengembangan model, diantaranya:
1. Load Dataset

    Dataset diupload dari google drive yang dimana sudah didownload dari sumber (Kaggle) agar dapat digunakan untuk identifikasi tingkat kualitas air yang menyimpan data seperti jenis       ikan, tingkat ph, suhu, dan tingkat kekeruhan

2. Pemeriksaan Nilai Kosong

    Dataset diperiksa untuk mengetahui apakah terdapat nilai yang hilang atau tidak valid. Jika ditemukan, dilakukan penanganan seperti imputasi atau penghapusan baris.

3. Pemeriksaan Nilai Duplikat

    Dataset diperiksa untuk mengetahui apakah terdapat nilai yang Duplikat. Jika ditemukan, dilakukan penanganan seperti penghapusan baris.

4. Distribusi Data

    Dilakukan visualisasi distribusi nilai untuk fitur numerik (pH, suhu, kekeruhan) untuk memahami pola dan outlier.

5. Analisis Korelasi (Univariate Analysis)

    Matriks korelasi dibuat untuk melihat hubungan antar variabel numerik serta potensi pengaruhnya terhadap kualitas air.

6. Analisis Fitur Kategorikal (Multivariate Analysis)

    Frekuensi jenis ikan dan distribusinya terhadap label kualitas air dianalisis untuk mengetahui pola yang mungkin muncul.

# Data Preparation

---

Tahap selanjutnya setelah EDA ialah mempersiapkan data untuk proses modelling yaitu Data Preparation. Data Preparation digunakan untuk memastikan data yang digunakan bersih, lengkap, dan sesuai format untuk proses pemodelan. Dengan menyiapkan data secara tepat, model yang dibuat dapat bekerja lebih efisien, akurat, dan mengurangi risiko kesalahan seperti bias atau overfitting.

Tahapan yang dilakukan dalam data preparation yang dilakukan, diantaranya:
1. Penanganan Outlier dan Data Duplikat : Langkah pertama dalam data preparation adalah membersihkan data dari nilai yang duplikat dan outlier .Tujuannya adalah memastikan kualitas data yang baik sehingga tidak mengganggu analisis atau model machine learning.
2. Transformasi Data: Tahap ini dilakukan untuk memperkaya informasi dan menyederhanakan analisis, diantaranya:
    *  Menghitung rata-rata dan standar deviasi parameter air (ph, temperature, turbidity) berdasarkan masing-masing jenis ikan.
    *  Menentukan kisaran ideal untuk tiap parameter berdasarkan nilai rata-rata ± standar deviasi.
    *  Membuat fungsi untuk mengevaluasi apakah setiap baris data sesuai dengan kisaran ideal dari jenis ikan terkait.
    *  Menambahkan kolom baru yang menunjukkan kondisi kualitas air (misalnya: “ideal” atau “tidak ideal”).
3. Encode Data : Data kategorikal seperti jenis ikan diubah menjadi format numerik agar bisa diproses oleh algoritma machine learning menggunakan Label Encoding jika jenis ikan akan diperlakukan sebagai kelas.
5. Pembagian Data : Setelah data sudah disiapkan sesuai yang diinginkan, Data dibagi menjadi dua bagian:
    - Training set: Digunakan untuk melatih model (biasanya 70–80% dari data).
    - Testing set: Digunakan untuk mengevaluasi performa model terhadap data baru (20–30%).


# Modelling

---


Dalam tahap pengembangan model ini, dilakukan pembandingan performa dari tiga algoritma klasifikasi yang berbeda, yaitu K-Nearest Neighbors (KNN), Random Forest, dan AdaBoost. Setiap model dilatih menggunakan data latih (X_train, y_train) dan dievaluasi menggunakan data uji (X_test, y_test) berdasarkan akurasi dan classification report.

1. K-Nearest Neighbors (KNN)

    Cara Kerja: KNN mengklasifikasikan data baru berdasarkan kemiripan (jarak) dengan data-data lain di sekitarnya. Model ini menghitung jarak ke k tetangga terdekat dan menentukan label berdasarkan mayoritas tetangga.

  Parameter Utama:
  - n_neighbors: jumlah tetangga yang digunakan untuk klasifikasi.

  - metric: jenis jarak yang digunakan (default: euclidean).

  Kelebihan:
  - Sederhana dan mudah diimplementasikan.

  - Tidak memerlukan pelatihan secara eksplisit (lazy learner).

  Kekurangan:
  - Sensitif terhadap skala data dan outlier.

  - Kurang efisien untuk dataset besar karena menghitung jarak ke semua titik.

2. Random Forest
  
    Cara Kerja: Random Forest merupakan model ensemble yang terdiri dari banyak pohon keputusan (decision tree). Model ini menghasilkan prediksi berdasarkan voting mayoritas dari seluruh pohon yang dibentuk dari subset data dan fitur yang berbeda.

  Parameter Utama:
  - n_estimators: jumlah pohon yang digunakan (default: 100).

  - max_depth: kedalaman maksimum tiap pohon.

  - random_state: penetapan seed untuk replikasi hasil.

  Kelebihan:
  - Memiliki akurasi tinggi dan tahan terhadap overfitting.
  
  - Dapat menangani data kategorikal dan numerik.

  - Memberikan informasi pentingnya fitur (feature importance).

  Kekurangan:
  - Cenderung lebih lambat dalam prediksi dibanding model sederhana.

  - Kurang interpretatif dibanding model tunggal seperti decision tree.

3. AdaBoost

    Cara Kerja: AdaBoost (Adaptive Boosting) adalah algoritma boosting yang membentuk model kuat dari beberapa model lemah (weak learners), biasanya decision tree sederhana. Model ini menekankan pada kesalahan dari prediksi sebelumnya dan memperbaikinya pada iterasi berikutnya.

  Parameter Utama:
  - n_estimators: jumlah model lemah yang akan digabungkan.

  - learning_rate: menentukan kontribusi masing-masing model.

  - random_state: agar hasil dapat direproduksi.

  Kelebihan:
  - Meningkatkan akurasi model sederhana.

  - Cukup efektif untuk dataset kecil sampai menengah.

  Kekurangan:
  - Sensitif terhadap outlier dan noise dalam data.

  - Bisa overfitting jika jumlah estimasi terlalu tinggi.

# Evaluation

---

Pada tahap ini dilakukan evaluasi terhadap model yang telah dikembangkan untuk mengklasifikasi tingkat kualitas air kolam dengan menambahkan databaru kedalam dataset. Data tersebut kemudian akan dianalisa oleh model dan ditentukan tingkat kualitasnya beserta hasil metrik evaluasinya.

Untuk mengevaluasi performa dari model yang dikembangkan, digunakan beberapa metrik klasifikasi, yaitu:

- Accuracy: Mengukur seberapa sering model memprediksi dengan benar dari seluruh data uji.

- Precision, Recall, dan F1-score (melalui classification_report):

- Precision: Mengukur ketepatan prediksi positif.

- Recall: Mengukur kemampuan model dalam menemukan semua kasus positif.

- F1-score: Rata-rata harmonis dari precision dan recall, terutama berguna saat terdapat ketidakseimbangan kelas.

Pemilihan metrik-metrik ini bertujuan untuk memberikan gambaran menyeluruh tentang performa model terhadap berbagai jenis kesalahan klasifikasi, bukan hanya sekadar akurasi.

## Hasil Evaluasi
Dari kode yang membuat contoh data baru berupa jenis ikan, tingkat ph, suhu, dan kekeruhan air. Data yang diuji oleh model untuk mendapatkan kualitas air kolam dengan menganalisa tingkat ph, suhu, dan kekeruhan. Hasil yang didapatkan dari kualitas kolam tersebut adalah 'good' yang berarti kualitas air layak digunakan.

#Hubungan Business Understanding

| **Aspek**                                                         | **Evaluasi**                                                                                                      |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Problem 1:** Kualitas air buruk berdampak pada ikan dan ekonomi | *Model mampu mengidentifikasi kualitas air, sehingga pembudidaya bisa melakukan tindakan preventif lebih awal.* |
| **Problem 2:** Pemantauan manual memakan waktu dan tenaga         | *Solusi model otomatis ini mengurangi kebutuhan pemantauan manual dengan bantuan sistem prediksi.*              |
| **Goal 1:** Mengembangkan model prediksi berbasis parameter air   | *Model klasifikasi menggunakan parameter pH, suhu, kekeruhan, dan jenis ikan telah dibuat dan berjalan baik.*   |
| **Goal 2:** Solusi efisien tanpa keahlian teknis tinggi           | *Model dapat digunakan hanya dengan input data, tanpa analisis manual, sehingga memudahkan pemilik kolam.*      |

## Dampak Solusi
Solusi yang dirancang, berupa model klasifikasi kualitas air berbasis machine learning, terbukti mampu memberikan rekomendasi instan dan akurat terhadap kualitas air berdasarkan data input yang sederhana. Hal ini tidak hanya memberikan efisiensi operasional bagi pembudidaya, tetapi juga mengurangi potensi kerugian akibat keterlambatan penanganan kualitas air.

Dengan demikian, tahap evaluasi menunjukkan bahwa model yang dikembangkan berdampak positif secara langsung terhadap tujuan bisnis dan memberikan nilai tambah nyata dalam konteks monitoring kualitas air kolam ikan.

