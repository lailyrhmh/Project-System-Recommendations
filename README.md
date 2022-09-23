# Laporan Proyek Machine Learning - Laily Rachmah
---

## Project Overview
---
Salah satu industri budaya yang secara global memiliki pasar yang luas adalah industry perfilman. Pertumbuhan industri perfilman yang menurun akibat terjadinya pandemi terutama pada sektor distribusi melalui bioskop. Hal ini menyebabkan beralihnya sector distribusi yang kemudian beralih untuk menggunakan layanan film _streaming_. 

Berkembangnya layanan film _streaming_ menyebabkan persaingan antara layanan penyedia _streaming_ video _online_ yang semakin ketat. Berkembangnya layanan ini selaras dengan banyaknya jumlah produksi film, mulai dari berbagai genre, alur cerita maupun tema film. Hal ini dapat dimanfaatkan untuk pengoptimalan distribusi film kepada calon pengguna berdasarkan preferensi yaitu dengan dibuatnya sistem rekomendasi. 



## Business Understanding
---

### Problem Statement
* Bagaimana cara membangun model _machine learning_ untuk merekomendasikan konten netflix yang mungkin serupa dengan konten yang dicari berdasarkan tipe?
* Bagaimana cara membangun model _machine learning content-based filtering_ menggunakan genre dan diskripsi konten?


### Goal
* Membangun model _machine learning_ untuk merekomendasikan konten berdasarkan tipe Movie atau TV Show.
* Membangun model _machine learning content-based filtering_ menggunakan genre dan deskripsi konten.


### Solution
* Melakukan pembagian data berdasarkan tipe konten sebelum melakukan depelopment model _machine learning_.
* Mem-preproses data dengan baik kemudian membangun model _content-based filtering_ menggunakan tfidf vectorizer dan cosine similarity.



## Data Understanding
---
- **Informasi dataset**
  <br>Dataset yang digunakan pada proyek ini adalah dataset Netflix Movie and TV Show yang dapat diunduh melalui tautan [berikut ini.](https://www.kaggle.com/datasets/shivamb/netflix-shows?datasetId=434238&searchQuery=recom)
  <br>Dataset berjumlah 8807 baris data dan memiliki 12 _feature_ yaitu:
    *   `show_id` : id konten netflix
    *   `type` : tipe konten
    *   `title` : judul konten
    *   `director` : nama _director_ konten
    *   `cast` : nama pemain atau _cast_
    *   `country` : asal negara dari konten
    *   `date_added` : tanggal ditambahkan ke-netflix
    *   `release_year` : tanggal rilis
    *   `rating` : rating konten
    *   `duration_ms` : dursi konten
    *   `listed_in` : genre konten
    *   `description` : deskripsi konten

- **_Feature_ yang digunakan**
  <br>Dalam sistem rekomendasi ini, fitur yang digunakan untuk pengembangan model _machine learning_ yaitu `type`, `title`, `listed_in` dan `description`

### Exploratory Data Analysis
<br> Pada tahapan ini dilakukan pengecekan informasi data kemudian _missing value_ dan duplikat data. Data yang digunakan todak memiliki _missing value_ dan juga duplikat data sehingga dilanjutkan ketahap analisis distribusi data.
- **Distribusi data `type` Movie dan TV Show**
  <br>Data menunjukan Movie dengan jumlah 69.6% dan TV Show dengan jumlah 30.4% dari seluruh data
  ![image](https://user-images.githubusercontent.com/91611703/191995037-7c0470f4-ba47-434b-9401-3193642c9b6a.png)

- **Distribusi data berdasarkan genre konten**
  <br>Data menunjukan genre terbanyak yaitu ada pada genre International Movies kemudian yang ke-dua yaitu Dramas dst.
  ![image](https://user-images.githubusercontent.com/91611703/191995085-1fd287d6-2769-42a7-8b90-7d91d5af9e54.png)

- **Menenmukan kata populer pada `description`**
  <br>Data menunjukan Kata populer teratas yaitu life dan find dst.
  ![image](https://user-images.githubusercontent.com/91611703/191995113-e4534a22-4d67-48de-a11c-c2f4f8dcb87a.png)


## Data Preparation
---
- **Membagi data berdasarkan `type`**
  <br>Dataframe baru dibuat berdasarkan dengan dua `type` yaitu Movie dan TV Show untuk melakukan permodelam _machine learning_ berdasarkan tipe.
  
- **Stop Word dan Lemmatization**
  <br>Pada _feature_ `description` terdapat banyak kata seperti “the”, “a”, “an”, “in” yang dapat dihilangkan atau diabaikan menggunakan _stop word_ kemudian penggunaan _lemmatization_ pada preproses data untuk mengelompokkan kata kedalam bentuk dasar ataupun kata yang memiliki arti yang sama sehigga dapat dianalisis menjadi satu iten kata.
  
- **Kolom baru untuk proses modelling**
  <br>Setelah dilakukan preproses pada kata yang ada pada `description`, tahap selanjutnya yaitu menggabungkan kata yang ada pada `listed_in` dan `description` kedalam satu kolom baru yaitu `text` untuk selanjutnya dipakai dalam perhitungan model _content-based filtering_.


## Modeling
---
### Content-Based Filtering

- **Metode Rekayasa Fitur**
  <br>Pada model _content-based filteirng_ ini, akan dilakukan menggunakan TF-IDF untuk menemukan representasi fitur penting dari setiap konten video netflix. Fungsi yang digunakan yaitu TfidfVectorizer dari library scikit-learn. Dalam sistem rekomendasi ini juga digunakan parameter `min_df=0.01` untuk menghilangkan term/istilah yang terlalu jarang muncul dengan hitungan minimal 1% term muncul yang akan diproses sebagai fitur.
  
- **Fit dan Tranform ke Dalam Matrix**
  <br>Setelah menemukan representasi fitur, dilakuakn fit dan transform pada data tipe Movie dan TV show dengan hasil matrx sebagai berikut:
  * Matrix Movie
    <br>`(6131, 285)`
    <br>Artinya data movie sebanyak 6131 baris data dengan 285 fitur penting dari `genre` dan `description`
    
  * Matrix TV Show
    <br>`(2676, 302)`
    <br>Artinya data movie sebanyak 2676 baris data dengan 302 fitur penting dari `genre` dan `description`
    
- **Cosine Similarity**
  <br>Untuk menghitung tingkat kesamaan dari fitur digunakan teknik cosine similarity dengan fungsi cosine_similarity() dari library sklearn. Setelah dihitung, maka akan dibuat dataframe baru berdasarkan cosine similarity untuk selanjutnya dapat digunakan dalam sistem rekomendasi.


## Evaluation
---


# Referensi (APA7)  :
---
[1] Laily, F. T., & Purbantina, A. P. (2021). Digitalisasi Industri Perfilman Korea Selatan Melalui Netflix Sebagai Alternatif Pasar _Ekspor Film. Expose: Jurnal Ilmu Komunikasi, 4(2)_, 141-155. [Tautan](http://e-journal.president.ac.id/presunivojs/index.php/EXPOSE/article/view/1494)

[2] Fajriansyah, M., Adikara, P. P., & Widodo, A. W. (2021). Sistem Rekomendasi Film Menggunakan Content Based Filtering. _Jurnal Pengembangan Teknologi Informasi dan Ilmu Komputer e-ISSN, 2548_, 964X. [Tautan](http://j-ptiik.ub.ac.id/index.php/j-ptiik/article/download/9163/4159)

[3] FIRDAUS, D. (2020). _EVALUATION OF NETFLIX RECOMMENDER SYSTEM BY USING DELONE AND MCLEAN INFORMATION SYSTEM SUCCESS MODEL_ (Doctoral dissertation, Universitas Gadjah Mada). [Tautan](http://etd.repository.ugm.ac.id/penelitian/detail/196994)

[4] Zayyad, M. R. A., & Kurniawardhani, A. (2021). Penerapan Metode Deep Learning pada Sistem Rekomendasi Film. _AUTOMATA, 2_(1). [Tautan](https://journal.uii.ac.id/AUTOMATA/article/view/17426)
