# Zaki-Tegar-Alamsyah-M01!
# Laporan Proyek Machine Learning - prediksi dengan metode regresi.

*   Nama: ZAKI TEGAR ALAMSYAH
*   ID Siswa: M193X0390
*   SIB Grup: M01

## Project Overview
Di lihat jumlah karyawan yang lebih dari 1000 pekerja di perusahaan ini, berencana untuk melakukan penelitian terkait gaji yang diterima para pekerjannya. Hal ini bertujuan untuk memberlakukan sistem penggajian yang adil tanpa memandang dari sudut pandang apapun. Dan untuk pelaksanaannya, perusahaan membutuhkan model yang dapat memprediksi gaji pekerjanya berdasarkan parameter-parameter seperti tingkat kecantikan, dan lain sebagainya. 

J. Pers. Med. 2022, melakukan penelitian untuk melakukan prediksi ide bunuh diri dengan menggunakan metode Random Forest dan menghasilkan akurasi 98.9% dan 97,4% dapat diprediksi hanya menggunakan kondisi terkait.[[1]] (https://doi.org/10.3390/jpm12060945).
Sejalan dengan penelitian tersebut, solusi yang ditawarkan menggunakan metode KNN, Random Forest dan AdaBoosting untuk melakukan prediksi gaji pekerja berdasarkan kecantikan pekerja.

## Business Understanding

### Problem Statements

- Bagaimana cara membangun sistem prediksi gaji pekerja berdasarkan kecantikannya dengan model terbaik?

### Goals

- Dapat mengetahui cara membangun sistem prediksi gaji pekerja berdasarkan kecantikannya dengan model terbaik.

## Data Understanding

Tabel 1. Informasi Dataset

| | Keterangan |
|---|---|
| Sumber | [Kaggle - Beauty](https://www.kaggle.com/datasets/aungpyaeap/beauty) |
| Jumlah Data | 1260 |
| *Usability* | 7.06 |
| Lisensi | Data files © Original Authors |
| *Rating* | None |
| Jenis dan Ukuran Berkas | csv (8 kB) |

### Variabel-variabel pada Dataset

Berdasarkan informasi dari [Kaggle](https://www.kaggle.com/datasets/aungpyaeap/beauty), variabel-variabel pada Beauty dataset adalah sebagai berikut:

- wage: merepresentasikan gaji yang diterima 
- exper: merepresentasikan lama waktu pengalaman pengguna dalam bekerja
- union: merepresentasikan status pekerja apakah bergabung dalam sebuah kelompok (1) atau tidak (0)
- goodhlth: merepresentasikan status kesehatan pekerja, jika sehat (1) jika sedang tidak sehat (0)
- black: merepresentasikan jenis warna kulit pekerja, jika hitam (1) jika tidak (0)
- female: merepresentasikan jenis kelamin, jika pekerja wanita (1) jika pria (0)
- married: merepresentasikan status pekerja, jika sudah menikah (1) jika belum (0).
- service: merepresentasikan pelayanan dalam bekerja
- educ: merepresentasikan tingkat edukasi pekerja dalam rentang tingkat 5 hingga 17
- looks: merepresentasikan tingkat kecantikan tampilan pekerja

### Menangani Missing Value

Untuk mendeteksi *missing value* digunakan fungsi isnull().sum() dan diperoleh:

Tabel 2. Hasil Deteksi *Missing Value*

| Kolom | Jumlah *Missing Value* |
|---|:---:|
| wage | 0 |
| exper | 0 |
| union | 0 |
| goodhlth | 0 |
| black | 0 |
| female | 0 |
| married | 0 |
| service | 0 |
| educ | 0 |
| looks | 0 |

Dari Tabel 2 di atas, terlihat bahwa setiap fitur tidak memiliki *missing value*.

### Univariate Analysis

Selanjutnya, untuk fitur numerik, akan dilakukan visualisasi dengan histogram pada masing-masing fiturnya sebagai berikut.

![histogram_numerik](https://user-images.githubusercontent.com/90232788/205424530-3deb7edf-2a3f-4ee6-8c7f-d57825e9bc82.png)

Gambar 1. Histogram pada Setiap Fitur Numerik

Berdasarkan Gambar 1. di atas, diperoleh beberapa informasi, antara lain:
- Pada histogram wage dan histogram exper miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model.

### Multivariate Analysis

Untuk mengamati hubungan antara fitur numerik, akan digunakan fungsi pairplot(), dengan output sebagai berikut.

![pairplot_numerik](https://user-images.githubusercontent.com/90232788/205424558-b3cf465d-a28b-4a06-8f2f-61ea884e5493.png)

Gambar 2. Visualisasi Hubungan antar Fitur Numerik

Pada pola sebaran data grafik pairplot di atas, terlihat fitur exper memiliki korelasi cukup kuat (positif) dengan fitur wage (target). Untuk mengevaluasi skor korelasinya, kita akan gunakan fungsi corr() sebagai berikut.

![correlation_numerik](https://user-images.githubusercontent.com/90232788/205424585-ce56f453-935b-4579-916e-70c7b3a81908.png)

Gambar 3. Korelasi antar Fitur Menarik

Koefisien korelasi berkisar antara -1 dan +1. Semakin dekat nilainya ke 1 atau -1, maka korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0 maka korelasinya semakin lemah.

Dari grafik korelasi di atas, fitur exper memiliki korelasi yang cukup kuat (0.23) dengan fitur target wage.

## Data Preparation

### Reduksi Dimensi dengan PCA

PCA umumnya digunakan ketika variabel dalam data yang memiliki korelasi yang tinggi. Korelasi tinggi ini menunjukkan data yang berulang atau redundant. Sebelumnya perlu cek kembali korelasi antar fitur (selain fitur target) dengan menggunakan pairplot.

![pairplot_non_target](https://user-images.githubusercontent.com/90232788/205424662-80a2ab46-cc94-4786-b8f8-c727e751dc33.png)

Gambar 4. Visualisasi Hubungan antar Fitur Selain Fitur Target (wage)

Selanjutnya kita akan mereduksi fitur educ dan fitur service karena keduanya berkorelasi cukup kuat yang dapat dilihat pada visualisasi pairplot di atas.

Untuk implementasinya menggunakan fungsi PCA() dari sklearn dengan mengatur nilai parameter n_components sebanyak fitur yang akan dikenakan PCA.

Tabel 3. Proporsi *Principal Component* dari Hasil PCA

| PC Pertama | PC Kedua |
|:---:|:---:|
| 0.975 | 0.025 |

Arti dari output di atas adalah, 97.5% informasi pada kedua fitur (educ dan service) terdapat pada PC (Principal Component) pertama. Sedangkan sisanya sebesar 2.5% terdapat pada PC kedua

Berdasarkan hasil tersebut, kita akan mereduksi fitur dan hanya mempertahankan PC (komponen) pertama saja. PC pertama ini akan menjadi fitur yang menggantikan dua fitur lainnya (educ dan service). Kita beri nama fitur ini PCA_1

Tabel 4. Tampilan 5 Sampel dari Dataset Setelah Dilakukan Reduksi Fitur

|index|wage|exper|union|goodhlth|black|female|married|looks|PCA\_1|
|---|---|---|---|---|---|---|---|---|---|
|1235|8.75|24|1|1|0|0|0|3|-3.469934|
|834|7.78|9|1|1|0|0|1|4|0.577108|
|542|2.92|12|0|1|0|1|0|4|0.524534|
|56|10.99|14|0|1|0|0|1|4|-3.417360|
|1027|3.27|25|0|1|0|0|1|2|4.571576|

### Train Test Split

Pada tahap ini akan dibagi dataset menjadi data latih (train) dan data uji (test). Pada kasus ini akan menggunakan proporsi pembagian sebesar 80:20 dengan fungsi train_test_split dari sklearn.

Tabel 5. Jumlah Data Latih dan Uji

| Jumlah Data Latih | Jumlah Data Uji | Jumlah Total Data |
|:---:|:---:|:---:|
| 1008 | 252 | 1260 |

### Standarisasi

Proses standarisasi bertujuan untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn.

StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandarScaler menghasilkan distribusi deviasi sama dengan 1 dan mean sama dengan 0.

Tabel 6. Hasil Proses Standarisasi pada Setiap Fitur

|index|exper|union|goodhlth|black|female|married|looks|PCA\_1|
|---|---|---|---|---|---|---|---|---|
|816|-0.0235686409852959|-0.5956833971812706|0.262543234837744|-0.2814766811934419|-0.7293249574894728|0.6835560754590242|-0.26044841890099907|0.9600808632180087|
|31|2.0458730065041055|-0.5956833971812706|-3.808896468492194|-0.2814766811934419|-0.7293249574894728|0.6835560754590242|-0.26044841890099907|0.9801271329692356|
|245|-0.4374569704831762|-0.5956833971812706|0.262543234837744|-0.2814766811934419|1.3711309200802089|0.6835560754590242|1.1741527081602416|-0.5629815255764054|
|1244|-0.27190163868402406|1.6787441193290356|0.262543234837744|-0.2814766811934419|-0.7293249574894728|0.6835560754590242|-0.26044841890099907|-0.1621696586265751|
|1129|1.7147623429058014|1.6787441193290356|0.262543234837744|-0.2814766811934419|-0.7293249574894728|0.6835560754590242|-1.6950495459622397|2.883955118962253|

## Modeling
Pada tahap ini, kita akan menggunakan tiga algoritma untuk kasus regresi ini. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menetukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan kita gunakan, antara lain:

1. K-Nearest Neighbor

    Kelebihan algoritma KNN adalah mudah dipahami dan digunakan sedangkan kekurangannya kika dihadapkan pada jumlah fitur atau dimensi yang besar rawan terjadi bias.

2. Random Forest
    
    Kelebihan algoritma Random Forest adalah menggunakan teknik Bagging yang berusaha melawan *overfitting* dengan berjalan secara paralel. Sedangkan kekurangannya ada pada kompleksitas algoritma Random Forest yang membutuhkan waktu relatif lebih lama dan daya komputasi yang lebih tinggi dibanding algoritma seperti Decision Tree.

3. Boosting Algorithm

    Kelebihan algoritma Boosting adalah menggunakan teknik Boosting yang berusaha menurunkan bias dengan berjalan secara sekuensial (memperbaiki model di tiap tahapnya). Sedangkan kekurangannya hampir sama dengan algoritma Random Forest dari segi kompleksitas komputasi yang menjadikan waktu pelatihan relatif lebih lama, selain itu *noisy* dan *outliers* sangat berpengaruh dalam algoritma ini.

Langkah pertama membuat DataFrame baru df_models untuk menampung nilai metrik pada setiap model / algoritma. Hal ini berguna untuk melakukan analisa perbandingan antar model. Metrik yang digunakan untuk mengevaluasi model adalah (MSE - Mean Squared Error).

### Model KNN
KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih k tetangga terdekat. Pemilihan nilai k sangat penting dan berpengaruh terhadap performa model. Jika memilih k yang terlalu rendah, maka akan menghasilkan model yang *overfitting* dan hasil prediksinya memiliki varians tinggi. Sedangkan jika memilih k yang terlalu tinggi, maka model yang dihasilkan akan *underfitting* dan prediksinya memiliki bias yang tinggi [[2]](https://www.oreilly.com/library/view/machine-learning-with/9781617296574/).

Oleh karena itu, perlu mencoba beberapa nilai k yang berbeda (1 sampai 20) kemudian membandingan mana yang menghasilkan nilai metrik model (pada kasus ini memakai *mean squared error*) terbaik. Selain itu, akan digunakan metrik ukuran jarak secara *default* (Minkowski Distance) pada KNeighborsRegressor dari *library* sklearn.

Tabel 7. Perbandingan Nilai K terhadap Nilai MSE

| K | MSE |
|:---:|---|
| 1 | 22.005950000000002 |
| 2 | 18.51776091269841 |
| 3 | 22.868429805996467 |
| 4 | 19.898676711309523 |
| 5 | 18.610842904761906 |
| 6 | 18.2803285824515 |
| 7 | 17.553236167800453 |
| 8 | 16.202325378224206 |
| 9 | 16.241765970997452 |
| 10 | 15.837582888888887 |
| 11 | 15.876245579168305 |
| 12 | 15.726342485119046 |
| 13 | 15.732282783882782 |
| 14 | 15.371005432053773 |
| 15 | 15.664813924162257 |
| 16 | 15.656392545572915 |
| 17 | 15.825094828911954 |
| 18 | 15.702771946649028 |
| 19 | 15.435384046739655 |
| 20 | 15.348939552579363 |

Jika divisualisasikan dengan fungsi "plot()" diperoleh:

![tuning_knn](https://user-images.githubusercontent.com/90232788/205424713-35e9c96d-780b-4298-8b43-706b74ef7f53.png)

Gambar 5. Visualisai Nilai K terhadap MSE

Dari hasil output diatas, nilai MSE terbaik dicapai ketika k = 14 yaitu sebesar 15.371. Oleh karena itu kita akan menggunakan k = 14 dan menyimpan nilai MSE nya (terhadap data latih, untuk data uji akan dilakukan pada proses evaluasi) kedalam df_models yang telah kita siapkan sebelumnya.

### Model Random Forest

Random forest merupakan algoritma *supervised learning* yang termasuk ke dalam kategori *ensemble* (group) learning. Pada model *ensemble*, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model *ensemble* ini digabungkan untuk membuat prediksi akhir. Jenis metode *ensemble* yang digunakan pada Random Forest adalah teknik *Bagging*. Metode ini bekerja dengan membuat subset dari data train yang independen. Beberapa model awal (base model / weak model) dibuat untuk dijalankan secara simultan / paralel dan independen satu sama lain dengan subset data train yang independen. Hasil prediksi setiap model kemudian dikombinasikan untuk menentukan hasil prediksi final.

Kita akan menggunakan `RandomForestRegressor` dari *library* scikit-learn dengan base_estimator defaultnya yaitu DecisionTreeRegressor dan parameter-parameter (hyperparameter) yang digunakan antara lain:

- n_estimator: jumlah trees (pohon) di forest.
- max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.
- random_state: digunakan untuk mengontrol random number generator yang digunakan.
- n_jobs: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.

Untuk menentukan nilai *hyperparameter* (n_estimator & max_depth) di atas, akan dilakukan *tuning* dengan GridSearchCV.

Tabel 8. Hasil *Hyperparameter Tuning* model *GridSearchCV* dengan Random Forest

| | Daftar Nilai | Nilai Terbaik |
|---|---|---|
| n_estimators | 10, 20, 30, 40, 50, 60, 70, 80, 90 | 30 |
| learning_rate | 4, 8, 16, 32 | 4 |
| MSE data latih | | 14.348 |
| MSE data uji | | 14.586 |

Dari hasil output di atas diperoleh nilai MSE terbaik dalam jangkauan parameter params_rf yaitu 14.349 (dengan data train) dan 14.586 (dengan data test) dengan n_estimators: 30 dan max_depth: 4. Selanjutnya kita akan menggunakan pengaturan parameter tersebut dan menyimpan nilai MSE nya kedalam df_models yang telah kita siapkan sebelumnya.

### Model AdaBoosting

Jika sebelumnya kita menggunakan algoritma *bagging* (Random Forest). Selanjutnya kita akan menggunakan metode lain dalam model *ensemble* yaitu teknik *Boosting*. Algoritma *Boosting* bekerja dengan membangun model dari data train. Kemudian membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Teknik ini bekerja secara sekuensial.

Pada kasus ini kita akan menggunakan metode *Adaptive Boosting*. Untuk implementasinya kita menggunakan AdaBoostRegressor dari library sklearn dengan base_estimator defaultnya yaitu DecisionTreeRegressor hampir sama dengan RandomForestRegressor bedanya menggunakan metode teknik *Boosting*.

Parameter-parameter (hyperparameter) yang digunakan pada algoritma ini antara lain:

- n_estimator: jumlah *estimator* dan ketika mencapai nilai jumlah tersebut algoritma Boosting akan dihentikan.
- learning_rate: bobot yang diterapkan pada setiap *regressor* di masing-masing iterasi Boosting.
- random_state: digunakan untuk mengontrol *random number* generator yang digunakan.

Untuk menentukan nilai *hyperparameter* (n_estimator & learning_rate) di atas, kita akan melakukan *tuning* dengan GridSearchCV.

Tabel 9. Hasil *Hyperparameter Tuning* model *GridSearchCV* dengan AdaBoosting

| | Daftar Nilai | Nilai Terbaik |
|---|---|---|
| n_estimators | 10, 20, 30, 40, 50, 60, 70, 80, 90 | 50 |
| learning_rate | 0.001, 0.01, 0.1, 0.2 | 0.001 |
| MSE data latih | | 17.2047 |
| MSE data uji | | 14.5258 |

Dari hasil output di atas diperoleh nilai MSE terbaik dalam jangkauan parameter params_ab yaitu  17.204 (dengan data train) dan 14.525 (dengan data test) dengan n_estimators: 50 dan learning_rate: 0.001. Selanjutnya kita akan menggunakan pengaturan parameter tersebut dan menyimpan nilai MSE nya kedalam df_models yang telah kita siapkan sebelumnya.

## Evaluation
Dari proses sebelumnya, kita telah membuat tiga model yang berbeda dan juga telah melatihnya. Selanjutnya kita perlu mengevaluasi model-model tersebut menggunakan data uji dan metrik yang digunakan dalam kasus ini yaitu mean_squared_error. Hasil evaluasi kemudian kita simpan ke dalam df_models.

$$\text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2.$$

Dengan:
- $n_{\text{sample}}$ adalah banyaknya data
- $\hat{y}_i$ adalah hasil prediksi sedangkan $y_i$ adalah nilai yang akan diprediksi (nilai yang sebenarnya).

Berdasarkan DataFrame `df_models` diperoleh:

Tabel 10. Nilai MSE pada Setiap Model dengan Data Latih dan Data Uji

|index|KNN|RandomForest|Boosting|
|---|---|---|---|
|Train MSE|16.83163|13.902077|13.552458|
|Test MSE|15.371005|14.708304|15.429171|

Untuk memudahkan, dilakukan *plot* hasil evaluasi model dengan *bar chart* sebagai berikut:

![evaluasi_model](https://user-images.githubusercontent.com/90232788/205424739-42824dce-5f28-4738-854a-e472cf92f1cf.png)

Gambar 6. *Bar Chart* Hasil Evaluasi Model dengan Data Latih dan Uji

Dari gambar di atas, terlihat bahwa, model RandomForest memberikan nilai MSE (pada data uji) yang paling rendah. Sebelum memutuskan model terbaik untuk melakukan prediksi "wage" atau besarnya gaji yang diterima pekerja berdasarkan kecantikannya. Mari kita coba uji prediksi menggunakan beberapa sampel acak (10) pada data uji.

Tabel 11. Hasil Prediksi dari 10 Sampel Acak

|index_sample|y_true|prediksi_KNN|prediksi_RF|prediksi_Boosting|
|---|---|---|---|---|
|120|5.77|9.487857|9.438564|8.690217|
|19|7.69|4.032143|4.646064|5.840000|
|78|6.89|9.861429|9.408543|8.690217|
|918|2.26|4.513571|3.739868|3.854242|
|1020|5.31|5.997857|7.284644|8.372640|
|209|7.69|7.500000|7.596209|8.524872|
|86|1.98|2.940714|3.412684|3.741762|
|307|6.01|9.918571|9.942020|8.690217|
|615|2.60|4.162857|3.701709|3.605537|
|367|4.81|9.047143|8.621286|8.372640|

Dari Tabel 11, terlihat bahwa prediksi dengan Random Forest memberikan hasil yang paling mendekati.

## Conclusion
Berdasarkan hasil evaluasi model di atas, dapat disimpulkan bahwa model terbaik untuk melakukan prediksi "wage" atau besarnya gaji yang diterima pekerja (berdasarkan kecantikannya) adalah Random Forest.
