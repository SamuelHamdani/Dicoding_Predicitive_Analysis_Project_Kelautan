# Laporan Proyek Machine Learning - Kelautan - Samuel Christian Hamdani

---

# Project Overview

---

Kualitas air kolam merupakan faktor krusial dalam keberhasilan budidaya ikan. Parameter seperti pH, suhu, dan tingkat kekeruhan air sangat mempengaruhi kesehatan dan pertumbuhan ikan. Kondisi air yang tidak sesuai dapat menyebabkan stres, penyakit, bahkan kematian pada ikan. Oleh karena itu, pemantauan kualitas air secara berkala sangat diperlukan untuk menjaga keberlangsungan ekosistem kolam dan meningkatkan produktivitas budidaya.

Namun, pengawasan kualitas air secara manual membutuhkan waktu, tenaga, dan keterampilan teknis yang tidak sedikit. Di sisi lain, perkembangan teknologi analisis data dan machine learning saat ini memungkinkan pemanfaatan data historis untuk membangun model prediktif yang dapat membantu dalam pengambilan keputusan secara cepat dan akurat.

Melalui proyek ini, dikembangkan sebuah model klasifikasi prediktif untuk menilai tingkat kualitas air kolam (Baik, Cukup, atau Buruk) berdasarkan data parameter fisik-kimia air dan jenis ikan yang dibudidayakan. Model ini dibangun menggunakan beberapa algoritma machine learning, seperti K-Nearest Neighbor (KNN), Random Forest, dan AdaBoost, untuk membandingkan performa masing-masing dan menentukan algoritma yang paling tepat digunakan dalam konteks ini.

Hasil dari proyek ini diharapkan dapat memberikan solusi berbasis data untuk mempermudah pemilik kolam dalam memantau kualitas air dan mengambil tindakan preventif secara efisien, sehingga mendukung praktik budidaya ikan yang lebih sehat, produktif, dan berkelanjutan.

Sumber Dataset : 
https://www.kaggle.com/datasets/monirmukul/realtime-pond-water-dataset-for-fish-farming?select=realfishdataset.csv

Referensi:
https://www.sciencedirect.com/science/article/pii/S2352340923008302

# Business Understanding

## Problem Statements
1.   Kualitas air kolam yang buruk dapat menyebabkan pertumbuhan ikan terganggu, stres, bahkan kematian, yang berdampak pada kerugian ekonomi bagi pembudidaya.
2.   Pemantauan kualitas air secara manual membutuhkan waktu, tenaga, dan keahlian yang tidak semua pemilik kolam miliki.

## Goals

Mengembangkan sebuah model dalam memprediksi tingkat kualitas air kolam berdasarkan parameter pH, suhu, kekeruhan, dan jenis ikan. Dengan adanya model ini, pemilik kolam dapat memperoleh informasi prediktif secara langsung dari data pemantauan, sehingga dapat mengambil tindakan lebih cepat dan efisien dalam menjaga kualitas air kolam dan kesehatan ikan.

##Solution Statements
1. Mengembangkan model **klasifikasi** prediktif dengan pendekatan supervised learning untuk mengkategorikan kualitas air kolam menjadi Baik, Cukup, dan Buruk.
2. Melatih dan menguji tiga algoritma machine learning (KNN, Random Forest, dan AdaBoost) dengan dataset kualitas air untuk memperoleh performa terbaik.


# Data Understanding

---

#Data Understanding


---

Pada tahap ini dilakukan eksplorasi awal terhadap data yang digunakan untuk membangun model prediktif menggunakan EDA(Exploratory Data Analysis). Dataset yang digunakan berisi beberapa fitur penting yang memengaruhi kualitas air kolam dan jenis ikan yang dibudidayakan.

## Sumber Data
Data diperoleh dari pengukuran parameter air kolam yang meliputi:
1. Jenis ikan yang dibudidayakan
2. Nilai pH air
3. Suhu air (dalam derajat Celsius)
4. Tingkat kekeruhan air (dalam NTU atau satuan sejenis)
5. Label kualitas air (Baik, Cukup, Buruk)

## Deskripsi Fitur
Berikut adalah penjelasan masing-masing fitur dalam dataset:
1. Jenis_Ikan (categorical): Menyatakan jenis ikan yang dibudidayakan di kolam.
2. pH (numerical): Mengukur tingkat keasaman air kolam. Nilai pH yang ideal biasanya berada pada kisaran 6,5–8,5.
3. Suhu (numerical): Suhu air kolam yang berpengaruh terhadap metabolisme ikan. Suhu optimal umumnya sekitar 20–30°C, tergantung jenis ikan.
4. Kekeruhan (numerical): Menunjukkan kejernihan air kolam. Air yang terlalu keruh dapat menghambat fotosintesis dan membahayakan ikan.
5. Kualitas_Air (categorical – target): Label hasil pengukuran kualitas air kolam, diklasifikasikan menjadi tiga kelas: Baik, Cukup, dan Buruk.

## Proses EDA
Pada bagian ini, dilakukan beberapa tahapan EDA yang dilakukan sebelum pengembangan model, diantaranya:
1. Load Dataset

  Dataset diupload dari google drive yang dimana sudah didownload dari sumber (Kaggle) agar dapat digunakan untuk identifikasi tingkat kualitas air yang menyimpan data seperti jenis ikan, tingkat ph, suhu, dan tingkat kekeruhan

2. Pemeriksaan Nilai Kosong

  Dataset diperiksa untuk mengetahui apakah terdapat nilai yang hilang atau tidak valid. Jika ditemukan, dilakukan penanganan seperti imputasi atau penghapusan baris.

3. Distribusi Data

  Dilakukan visualisasi distribusi nilai untuk fitur numerik (pH, suhu, kekeruhan) untuk memahami pola dan outlier.

4. Analisis Korelasi (Univariate Analysis)

  Matriks korelasi dibuat untuk melihat hubungan antar variabel numerik serta potensi pengaruhnya terhadap kualitas air.

5. Analisis Fitur Kategorikal (Multivariate Analysis)

  Frekuensi jenis ikan dan distribusinya terhadap label kualitas air dianalisis untuk mengetahui pola yang mungkin muncul.


# Data Preparation

---

Tahap selanjutnya setelah EDA ialah mempersiapkan data untuk proses modelling yaitu Data Preparation. Data Preparation digunakan untuk memastikan data yang digunakan bersih, lengkap, dan sesuai format untuk proses pemodelan. Dengan menyiapkan data secara tepat, model yang dibuat dapat bekerja lebih efisien, akurat, dan mengurangi risiko kesalahan seperti bias atau overfitting.

Tahapan yang dilakukan dalam data preparation yang dilakukan, diantaranya:
Menghitung Rata-rata & Standar Deviasi per Jenis Ikan

1. Mendefinisikan Kisaran Ideal untuk Parameter Air
2. Membuat Fungsi untuk Evaluasi Kualitas Air
3. Menerapkan Fungsi Evaluasi ke Setiap Baris Data
4. Visualisasi Distribusi Kualitas Air
5. Encoding Kategori Jenis Ikan
6. Membagi Data Jadi Training dan Testing Set


# Modelling

---

Setelah data yang dibutuhkan telah siap, selanjutnya ialah proses membangun model prediktif menggunakan teknik machine learning. Data yang telah dipersiapkan digunakan untuk melatih dan menguji berbagai algoritma seperti KNN, RandomForest, atau Boosting Algorithm untuk menemukan model terbaik. Aktivitas utama meliputi :
1. Pemilihan metode modeling yang sesuai
2. Pelatihan model dengan data training
3. Pengujian model menggunakan data testing
4. Evaluasi performa model menggunakan metrik seperti akurasi.

# Evaluation

---

Setelah model telah dikembangkan, tahap selanjutnya adalah melakukan evaluasi model dengan memasukkan data baru kedalam dataset untuk mengklasifikasi kategori kualitas air kolam untuk budidaya ikan. Pada tahap ini, dilakukan evaluasi dengan menambahkan data baru kedalam dataset dan meminta model untuk melakukan klasifikasi kualitas air kolam (Good, Okay, Bad).

