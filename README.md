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
<br> Pada tahapan ini dilakukan pengecekan informasi data kemudian _missing value_ dan duplikat data. Data yang digunakan todak memiliki _misiing value_ dan juga duplikat data sehingga dilanjutkan ketahap analisis distribusi data.
- **Distribusi data _type_ Movie dan TV Show**
  <br>Data menunjukan Movie dengan jumlah 69.6% dan TV Show dengan jumlah 30.4% dari seluruh data
  ![image](https://user-images.githubusercontent.com/91611703/191995037-7c0470f4-ba47-434b-9401-3193642c9b6a.png)

- **Distribusi data berdasarkan genre konten**
  <br>Data menunjukan genre terbanyak yaitu ada pada genre International Movies kemudian yang ke-dua yaitu Dramas dst.
  ![image](https://user-images.githubusercontent.com/91611703/191995085-1fd287d6-2769-42a7-8b90-7d91d5af9e54.png)

- **Menenmukan kata populer pada _description_**
  <br>Data menunjukan Kata populer teratas yaitu life dan find dst.
  ![image](https://user-images.githubusercontent.com/91611703/191995113-e4534a22-4d67-48de-a11c-c2f4f8dcb87a.png)


## Data Preparation
---



## Modeling
---


## Evaluation
---


# Referensi (APA7)  :
---
[1] Laily, F. T., & Purbantina, A. P. (2021). Digitalisasi Industri Perfilman Korea Selatan Melalui Netflix Sebagai Alternatif Pasar _Ekspor Film. Expose: Jurnal Ilmu Komunikasi, 4(2)_, 141-155. [Tautan](http://e-journal.president.ac.id/presunivojs/index.php/EXPOSE/article/view/1494)

[2] Fajriansyah, M., Adikara, P. P., & Widodo, A. W. (2021). Sistem Rekomendasi Film Menggunakan Content Based Filtering. _Jurnal Pengembangan Teknologi Informasi dan Ilmu Komputer e-ISSN, 2548_, 964X. [Tautan](http://j-ptiik.ub.ac.id/index.php/j-ptiik/article/download/9163/4159)

[3] FIRDAUS, D. (2020). _EVALUATION OF NETFLIX RECOMMENDER SYSTEM BY USING DELONE AND MCLEAN INFORMATION SYSTEM SUCCESS MODEL_ (Doctoral dissertation, Universitas Gadjah Mada). [Tautan](http://etd.repository.ugm.ac.id/penelitian/detail/196994)

[4] Zayyad, M. R. A., & Kurniawardhani, A. (2021). Penerapan Metode Deep Learning pada Sistem Rekomendasi Film. _AUTOMATA, 2_(1). [Tautan](https://journal.uii.ac.id/AUTOMATA/article/view/17426)
