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
  * Cosine Similarity Movie
    ```html
    |                 title | Ghost Lab | This Is the Life | Lizzie Borden Took an Ax | 7 Khoon Maaf | We Are Family |
    |                 title |           |                  |                          |              |               |
    |----------------------:|----------:|-----------------:|-------------------------:|-------------:|--------------:|
    |      Ekşi Elmalar     |  0.131873 |         0.133046 |                 0.143831 |     0.158522 |      0.180019 |
    |       New Money       |  0.115808 |         0.056238 |                 0.058831 |     0.117822 |      0.099081 |
    | The Edge of Democracy |  0.179247 |         0.174660 |                 0.119083 |     0.105048 |      0.117550 |
    |      Dolphin Kick     |  0.102474 |         0.045805 |                 0.056889 |     0.026914 |      0.034067 |
    |  Garfield's Pet Force |  0.021678 |         0.117604 |                 0.104707 |     0.074934 |      0.082060 |
    |        Chaahat        |  0.107094 |         0.049598 |                 0.101311 |     0.211924 |      0.054877 |
    |     Organize Isler    |  0.081284 |         0.040430 |                 0.033914 |     0.153904 |      0.025511 |
    |          Gigi         |  0.026737 |         0.387760 |                 0.032757 |     0.038089 |      0.100149 |
    |        Banyuki        |  0.125369 |         0.067827 |                 0.358346 |     0.092482 |      0.057023 |
    |          Zion         |  0.061115 |         0.103701 |                 0.149083 |     0.086541 |      0.035696 |
    ```
    <br>Nilai cosine similarity yang semakin besar, menunjukkan bahwa kemiripan atau kesamaan antar movie semakin besar. Seperti movie berjudul This Is the Life dan Gigi yang memiliki nilai kesamaan 0.387760.

  * Cosine Similarity TV Show
    ```html
    |                           title | Trio and a Bed | Korean Pork Belly Rhapsody |            Ken Burns Presents: College Behind Bars:  |  Hasmukh | Seven and Me |
    |                                 |                |                            | A Film by Lynn Novick and Produced by Sarah Botstein |          |              |
    |                           title |                |                            |                                                      |          |              |
    |--------------------------------:|---------------:|---------------------------:|-----------------------------------------------------:|---------:|-------------:|
    |          The Last O.G.          |       0.081342 |                   0.035724 |                                             0.000000 | 0.136928 |     0.028505 |
    |          Meteor Garden          |       0.192441 |                   0.230628 |                                             0.000000 | 0.111326 |     0.224565 |
    |           Brotherhood           |       0.186350 |                   0.155229 |                                             0.023458 | 0.122889 |     0.213032 |
    |           Saint Seiya           |       0.078478 |                   0.119882 |                                             0.021670 | 0.041787 |     0.150097 |
    |        Revolutionary Love       |       0.249150 |                   0.239510 |                                             0.021028 | 0.130760 |     0.067510 |
    |         Turn Up Charlie         |       0.099665 |                   0.069958 |                                             0.036709 | 0.123601 |     0.054028 |
    |          Civilizations          |       0.136259 |                   0.108695 |                                             0.166118 | 0.043338 |     0.041353 |
    |            Cocomelon            |       0.056926 |                   0.089915 |                                             0.050711 | 0.019352 |     0.051701 |
    | Power Rangers Dino Super Charge |       0.179629 |                   0.135612 |                                             0.023081 | 0.072706 |     0.134800 |
    |                H                |       0.330259 |                   0.137654 |                                             0.000000 | 0.126051 |     0.120530 |
    ```
    <br>Nilai cosine similarity yang semakin besar, menunjukkan bahwa kemiripan atau kesamaan antar TV Show semakin besar. Seperti TV Show berjudul Trio and a Bed dan Revolitionary Love yang memiliki nilai kesamaan 0.387760.

- **Uji Coba Sistem Rekomendasi**
  <br>Sistem menggunakan dataframe baru yang berisi cosine similarity antar konten video untuk kemudian dihitung kemiripiannya dan dicetak hasil _top 10_ dari konten video netflix dengan kemiripan paling tinggi. Berikut hasil dari uji coba sistem:
  * Rekomendasi Movie
    <br>Cek data movie
    ```html
    | title |              listed_in              |                        description                        |                                                                                         text                                                                                        |
    |:-----:|:-----------------------------------:|:---------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
    | Bugs  | Documentaries, International Movies | A willing team of chef and researcher go on a gastronomic | Documentaries, International Movies A willing team of chef and researcher go on a gastronomic adventure around the globe to weigh the benefit of using bug as a future food source. |
    |       |                                     |                                                           |                                                                                                                                                                                     |
    
    ```
    
    <br>Hasil Rekomendasi Movie
    ```html
    | index |         title         |                        listed_in                        |                                                                       description                                                                      |
    |:-----:|:---------------------:|:-------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------:|
    |     0 | She Did That          | Documentaries                                           | Go inside the life of extraordinary, black female entrepreneur as they discus building legacy and pioneering a new future for the next generation.     |
    |     1 | Chasing Coral         | Documentaries                                           | Divers, scientist and photographer around the world mount an epic underwater campaign to document the disappearance of coral reefs.                    |
    |     2 | The Ivory Game        | Documentaries                                           | Filmmakers infiltrate the corrupt global network of ivory trafficking, exposing poacher and dealer as African elephant edge closer to extinction.      |
    |     3 | Operation Chromite    | Action & Adventure, Dramas, International Movies        | To pave the way for a major amphibious invasion, a team of South Korean spy go behind enemy line to steal a map of North Korean coastal defenses.      |
    |     4 | The Tour              | Comedies, Documentaries, International Movies           | Miloš Knor brings comedian Lukáš Pavlásek, Tomáš Matonoha, Ester Kočičková, Michal Kavalčík and Richard Nedvěd on a tour around the Czech Republic.    |
    |     5 | Bitcoin Heist         | Action & Adventure, Comedies, International Movies      | A unconventional, efficient Interpol special agent go rogue and assembles a team of thief to catch a shadowy hacker called "The Ghost."                |
    |     6 | La Gran Ilusión       | International Movies                                    | Known as "El Mago Pop," illusionist Antonio Díaz shock and awe celebrity and bystander around the world with his mind-blowing performances.            |
    |     7 | Addicted to Life      | Action & Adventure, Documentaries, International Movies | Chasing extreme challenges, athletic daredevil test their limit in various environment from giant wave to snowy slope around the world.                |
    |     8 | Chappaquiddick        | Dramas                                                  | Senator Ted Kennedy watch his future unravel in the wake of an infamous car crash as family and ally vie to protect his reputation.                    |
    |     9 | Madness in the Desert | Documentaries, International Movies                     | The story of making "Lagaan," one of the millennium's seminal Indian films, is told from the point of view of production team member Satyajit Bhatkal. |
    ```
    
  * Rekomendasi TV Show
    <br>Cek data TV Show
    ```html
    |   title  |         listed_in         |                                                           description                                                          |                                                                           text                                                                           |
    |:--------:|:-------------------------:|:------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------:|
    | Cocomong | Kids' TV, Korean TV Shows | What's in your fridge? In sunny Refrigerator Land, everyday ingredient transform into animal friend who love a good adventure. | Kids' TV, Korean TV Shows What's in your fridge? In sunny Refrigerator Land, everyday ingredient transform into animal friend who love a good adventure. |
    |          |                           |                                                                                                                                |                                                                                                                                                          |
    ```
    
    <br>Hasil Rekomendasi TV Show
    ```html
    | index |                 title                |                            listed_in                           |                                                                      description                                                                      |
    |:-----:|:------------------------------------:|:--------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------:|
    |     0 | Resurrection: Ertugrul               | International TV Shows, TV Action & Adventure, TV Dramas       | When a good deed unwittingly endangers his clan, a 13th-century Turkish warrior agrees to fight a sultan's enemy in exchange for new tribal land.     |
    |     1 | Miniforce                            | Kids' TV, Korean TV Shows                                      | Four animal superheroes called the Miniforce transform into robot to protect small and defenseless creature from the hand of scheming villains.       |
    |     2 | Twirlywoos                           | British TV Shows, Kids' TV                                     | The colorful and curious family of Twirlywoos bounce around their boat and visit new places, turning learning into adventure wherever they land.      |
    |     3 | Vagabond                             | International TV Shows, Korean TV Shows, TV Action & Adventure | When his nephew dy in a plane crash, stunt man Cha Dal-geon resolve to find out what happened, with the help of covert operative Go Hae-ri.           |
    |     4 | Crash Landing on You                 | International TV Shows, Korean TV Shows, Romantic TV Shows     | A paragliding mishap drop a South Korean heiress in North Korea – and into the life of an army officer, who decides he will help her hide.            |
    |     5 | Arthdal Chronicles                   | International TV Shows, Korean TV Shows, TV Action & Adventure | In a mythical land called Arth, the inhabitant of the ancient city of Arthdal and its surrounding region vie for power as they build a new society.   |
    |     6 | The Epic Tales of Captain Underpants | Kids' TV, TV Comedies                                          | Fourth-grade friend George and Harold have a shared love of prank and comic book – and turning their principal into an undies-wearing superhero.      |
    |     7 | Buddy Thunderstruck                  | Kids' TV, TV Comedies                                          | Follow the outrageous, high-octane adventure of Buddy Thunderstruck, a truck-racing dog who brings gut and good time to the town of Greasepit.        |
    |     8 | YooHoo to the Rescue                 | Kids' TV, Korean TV Shows                                      | In a series of magical missions, quick-witted YooHoo and his can-do crew travel the globe to help animal in need.                                     |
    |     9 | Octonauts: Above & Beyond            | British TV Shows, Kids' TV                                     | The Octonauts expand their exploration beyond the sea — and onto land! With new ride and new friends, they'll protect any habitat and animal at risk. |
    ```

## Evaluation
---
Evaluasi untuk model _machine learning content-based filtering_ adalah menggunakan metrik _precision_ untuk menghitung jumlah presisi dari model sistem rekomendasi yang telah dibuat yaitu dengan formula sebagai berikut:

<br>![image](https://user-images.githubusercontent.com/91611703/192021837-2fac44e9-3fdf-49a7-abf2-cfb3a4a05263.png)

Berikut rincian dari hasil analisis:
- **Rekomendasi Movie**
  <br>_Feature_ movie
  <br>```Documentaries, International Movies A willing team of chef and researcher go on a gastronomic adventure around the globe to weigh the benefit of using bug as a future food source. ```
  
  dengan hasil
  
  ```html
    | index |         title         |                        listed_in                        |                                                                       description                                                                      |
    |:-----:|:---------------------:|:-------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------:|
    |     0 | She Did That          | Documentaries                                           | Go inside the life of extraordinary, black female entrepreneur as they discus building legacy and pioneering a new future for the next generation.     |
    |     1 | Chasing Coral         | Documentaries                                           | Divers, scientist and photographer around the world mount an epic underwater campaign to document the disappearance of coral reefs.                    |
    |     2 | The Ivory Game        | Documentaries                                           | Filmmakers infiltrate the corrupt global network of ivory trafficking, exposing poacher and dealer as African elephant edge closer to extinction.      |
    |     3 | Operation Chromite    | Action & Adventure, Dramas, International Movies        | To pave the way for a major amphibious invasion, a team of South Korean spy go behind enemy line to steal a map of North Korean coastal defenses.      |
    |     4 | The Tour              | Comedies, Documentaries, International Movies           | Miloš Knor brings comedian Lukáš Pavlásek, Tomáš Matonoha, Ester Kočičková, Michal Kavalčík and Richard Nedvěd on a tour around the Czech Republic.    |
    |     5 | Bitcoin Heist         | Action & Adventure, Comedies, International Movies      | A unconventional, efficient Interpol special agent go rogue and assembles a team of thief to catch a shadowy hacker called "The Ghost."                |
    |     6 | La Gran Ilusión       | International Movies                                    | Known as "El Mago Pop," illusionist Antonio Díaz shock and awe celebrity and bystander around the world with his mind-blowing performances.            |
    |     7 | Addicted to Life      | Action & Adventure, Documentaries, International Movies | Chasing extreme challenges, athletic daredevil test their limit in various environment from giant wave to snowy slope around the world.                |
    |     8 | Chappaquiddick        | Dramas                                                  | Senator Ted Kennedy watch his future unravel in the wake of an infamous car crash as family and ally vie to protect his reputation.                    |
    |     9 | Madness in the Desert | Documentaries, International Movies                     | The story of making "Lagaan," one of the millennium's seminal Indian films, is told from the point of view of production team member Satyajit Bhatkal. |
    ```
    <br>Menggunakan _feature_ kita bisa membandingkan dengan hasil rekomendasi movie mana yang tidak relevan baik dengan `genre` ataupun `description`. Dari 10 hasil rekomendasi, dapat dilihat 1 dari 9 movie yang direkomendasikan memiliki `genre` Dramas dengan `description` yang tidak sesuai atau tidak relevan dengan salah satu _feature_ pada movie yang menjadi acuan. Dihitung menggunakan rumus _precision_ maka 
    ```html
    P = 9/10 = 90% presicion
    ```
    untuk model rekomendasi movie.
  
- **Rekomendasi TV Show**
  <br>_Feature_ tv show
  <br>```Kids' TV, Korean TV Shows What's in your fridge? In sunny Refrigerator Land, everyday ingredient transform into animal friend who love a good adventure. ```
  
  dengan hasil
  ```html
    | index |                 title                |                            listed_in                           |                                                                      description                                                                      |
    |:-----:|:------------------------------------:|:--------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------:|
    |     0 | Resurrection: Ertugrul               | International TV Shows, TV Action & Adventure, TV Dramas       | When a good deed unwittingly endangers his clan, a 13th-century Turkish warrior agrees to fight a sultan's enemy in exchange for new tribal land.     |
    |     1 | Miniforce                            | Kids' TV, Korean TV Shows                                      | Four animal superheroes called the Miniforce transform into robot to protect small and defenseless creature from the hand of scheming villains.       |
    |     2 | Twirlywoos                           | British TV Shows, Kids' TV                                     | The colorful and curious family of Twirlywoos bounce around their boat and visit new places, turning learning into adventure wherever they land.      |
    |     3 | Vagabond                             | International TV Shows, Korean TV Shows, TV Action & Adventure | When his nephew dy in a plane crash, stunt man Cha Dal-geon resolve to find out what happened, with the help of covert operative Go Hae-ri.           |
    |     4 | Crash Landing on You                 | International TV Shows, Korean TV Shows, Romantic TV Shows     | A paragliding mishap drop a South Korean heiress in North Korea – and into the life of an army officer, who decides he will help her hide.            |
    |     5 | Arthdal Chronicles                   | International TV Shows, Korean TV Shows, TV Action & Adventure | In a mythical land called Arth, the inhabitant of the ancient city of Arthdal and its surrounding region vie for power as they build a new society.   |
    |     6 | The Epic Tales of Captain Underpants | Kids' TV, TV Comedies                                          | Fourth-grade friend George and Harold have a shared love of prank and comic book – and turning their principal into an undies-wearing superhero.      |
    |     7 | Buddy Thunderstruck                  | Kids' TV, TV Comedies                                          | Follow the outrageous, high-octane adventure of Buddy Thunderstruck, a truck-racing dog who brings gut and good time to the town of Greasepit.        |
    |     8 | YooHoo to the Rescue                 | Kids' TV, Korean TV Shows                                      | In a series of magical missions, quick-witted YooHoo and his can-do crew travel the globe to help animal in need.                                     |
    |     9 | Octonauts: Above & Beyond            | British TV Shows, Kids' TV                                     | The Octonauts expand their exploration beyond the sea — and onto land! With new ride and new friends, they'll protect any habitat and animal at risk. |
    ```
    <br>Menggunakan _feature_ kita bisa membandingkan dengan hasil rekomendasi movie mana yang tidak relevan baik dengan `genre` ataupun `description`. Dari 10 hasil rekomendasi, dapat dilihat semua hasil rekomendasi memiliki setidaknya satu _feature_ baik pada `genre` atau `description` yang sesuai atau relevan dengan salah satu _feature_ pada tv show yang menjadi acuan. Dihitung menggunakan rumus _precision_ maka 
    ```html
    P = 10/10 = 100% presicion
    ```
    untuk model rekomendasi tv show.
    
  
## Conclusion
---
Model _machine learning_ dapat dibangun berdasarkan tipe yaitu Movie dan TV Show. Sistem rekomendasi _machine learning_ dibangun dengan metode _content-based filtering_ menggunakan dua fitur yaitu `genre` dan `description`. SIstem rekomendasi movie memiliki _precision_ sebesar 90% dan sistem rekomendasi tv show memiliki _precision_ sebesar 100%. Kedepannya, diharapkan sistem dapat dibangun menggunakan semua _feature_ yang ada pada dataset atau menggunakan metode clustering dengan algoritma K-Means.

# Referensi (APA7)  :
---
[1] Laily, F. T., & Purbantina, A. P. (2021). Digitalisasi Industri Perfilman Korea Selatan Melalui Netflix Sebagai Alternatif Pasar _Ekspor Film. Expose: Jurnal Ilmu Komunikasi, 4(2)_, 141-155. [Tautan](http://e-journal.president.ac.id/presunivojs/index.php/EXPOSE/article/view/1494)

[2] Fajriansyah, M., Adikara, P. P., & Widodo, A. W. (2021). Sistem Rekomendasi Film Menggunakan Content Based Filtering. _Jurnal Pengembangan Teknologi Informasi dan Ilmu Komputer e-ISSN, 2548_, 964X. [Tautan](http://j-ptiik.ub.ac.id/index.php/j-ptiik/article/download/9163/4159)

[3] FIRDAUS, D. (2020). _EVALUATION OF NETFLIX RECOMMENDER SYSTEM BY USING DELONE AND MCLEAN INFORMATION SYSTEM SUCCESS MODEL_ (Doctoral dissertation, Universitas Gadjah Mada). [Tautan](http://etd.repository.ugm.ac.id/penelitian/detail/196994)

[4] Zayyad, M. R. A., & Kurniawardhani, A. (2021). Penerapan Metode Deep Learning pada Sistem Rekomendasi Film. _AUTOMATA, 2_(1). [Tautan](https://journal.uii.ac.id/AUTOMATA/article/view/17426)
