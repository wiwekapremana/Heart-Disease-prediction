# Laporan Proyek Machine Learning - Komang Wiweka Premana

## Domain Proyek
Sindrom koroner akut atau serangan jantung adalah gangguan jantung serius ketika otot jantung tidak mendapat aliran darah. Kondisi ini akan mengganggu fungsi jantung dalam mengalirkan darah ke seluruh tubuh. Dalam dunia kedokteran, serangan jantung disebut juga sebagai infark miokard.Serangan jantung terjadi akibat terhambatnya aliran darah ke otot jantung. Penyebab utama kondisi ini adalah penyakit jantung koroner, yaitu tersumbatnya pembuluh darah yang memasok darah ke jantung (pembuluh darah koroner), akibat timbunan kolesterol yang membentuk plak di dinding pembuluh darah. Hal inilah yang menjadi penyebab mengapa kolesterol tinggi bisa membuat seseorang berisiko terkena sakit jantung.Kondisi ini diperparah dengan terbentuknya gumpalan darah, yang dapat menyumbat total pembuluh darah dan menimbulkan serangan jantung.Pada tautan berikut terdapat [Cara mencegah penyakit serangan jantung](https://www.klikdokter.com/info-sehat/read/2696643/cara-mudah-mencegah-serangan-jantung/). Data dalam proyek ini berisi mengenai data individu seperti age, sex, cp, trestbps, chol, fbs, restecg, thalachh, exang, oldpeak, slope, ca , thal dan target. Untuk menyelesaikan kasus ini, predictive analytics diharapkan dapat memprediksi masalah tersebut dan mendapatkan solusi yang terbaik dengan menggunakan model machine learning KNN, Logistic regresion, dan Random Forest. 

## Business Understanding
### Problem Statement
Berikut adalah problem statement dari proyek ini:
* Apa saja fitur yang berkorelasi dengan target (heart attack)?</br>
* Model Machine Learning manakah yang memiliki accuracy tertinggi dalam menyelesaikan permasalahan ini?</br>
### Goals
Berikut adalah goals yang ingin dicapai dalam proyek ini:
*	Mengetahui fitur yang berkorelasi dengan target(heart attack)
*	Mengetahui model terbaik dalam Machine Learning untuk memprediksi penyakit serangan jantung pada seseorang.
### Solution Statements
Untuk mencapai tujuan memprediksi masalah ini saya menggunakan tiga model Machine Learning. Dimana ketiga model ini cocok digunakan untuk data regresi karena output yang diprediksi adalah sebuah angka. Berikut penjelasan secara singkat mengenai tiga model yang saya gunakan:
*	**Logistic Regression **
</br>[Logistic Regression](https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8)  adalah algoritma yang digunakan untuk memprediksi probabilitas variabel dependen kategoris. Dalam regresi logistik, variabel dependen adalah variabel biner yang berisi data berkode 1 (ya, berhasil, dll.) atau 0 (tidak, gagal, dll.).Kuncoro (2001) mengatakan bahwa regresi logistik memiliki beberapa kelebihan dibandingkan teknik analisis lain yaitu: 1. Regresi logistik tidak memiliki asumsi normalitas dan heteroskedastisitas atas variabel bebas yang digunakan dalam model sehingga tidak diperlukan uji asumsi klasik walaupun variabel independen berjumlah lebih dari satu. 2. Variabel independen dalam regresi logistik bisa campuran dari variabel kontinu, distrik, dan dikotomis. 3. Regresi logistik tidak membutuhkan keterbatasan dari variabel independennya. 4. Regresi logistik tidak mengharuskan variabel bebasnya dalam bentuk interval.
*	**KNN**
</br>[KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) algoritma yang menggunakan kesamaan fitur untuk memprediksi nilai baru. Nilai baru ini didasarkan pada seberapa mirip dengan tetangganya sejumlah k, oleh karena itu disebut K-Nearest Neighbor.KNN memiliki beberapa kelebihan, diantaranya adalah pelatihan sangat cepat, sederhana dan mudah dipelajari, tahan terhadap data pelatihan yang memiliki derau, dan efektif jika data pelatihan besar.
*	**Random Forest**
</br>[Regresi Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) adalah meta estimator yang cocok dengan sejumlah pengklasifikasi pohon keputusan pada berbagai sub-sampel dari dataset dan menggunakan rata-rata untuk meningkatkan akurasi prediksi dan kontrol over-fitting. Ukuran sub-sampel dikontrol dengan parameter max_samples jika bootstrap=True (default), jika tidak, seluruh dataset digunakan untuk membangun setiap pohon. Model ini mampu mengklasifiksi data yang memiliki atribut yang tidak lengkap, dapat digunakan untuk klasifikasi dan regresi akan tetapi tidak terlalu bagus untuk regresi, lebih cocok untuk pengklasifikasian data serta dapat digunakan untuk menangani data sampel yang banyak. Selain itu, karena model ini termasuk model gabungan dari beberapa decision tree (ensemble). Maka dari itu, tentunya akan memakan waktu yang lebih lama karena model ini akan menggunakan decision tree dalam jumlah yang banyak.

## Data Understanding
Dataset yang digunakan adalah dataset [Heart Disease Dataset](https://www.kaggle.com/johnsmith88/heart-disease-dataset) dari situs Kaggle yang berisi data tentang penyakit serangan jantung yang dialami seseorang berdasarkan data individu seperti age, sex, cp, trestbps, chol, fbs, restecg, thalachh, exang, oldpeak, slope, ca , thal dan target. Pada  Dataset ini memiliki 1025 baris 14 kolom dengan 14 numerikal sebagai berikut :
* age:- Usia pasien
* sex:- Jenis kelamin pasien (1 = 'Laki laki' and 0 = 'perempuan')
* cp:- cp adalah singkatan dari chest pain. Dalam dataset ini, ada empat jenis nyeri dada (0=asimtomatik, 1= angina tipikal, 2=angina atipikal, 3=nyeri non-angina)
* trestbps:- Resting blood pressure(in mm hg on admission to the hospital)
* chol:- Serum kolestrol dalam mg/dl
* fbs:- Gula darah>120mg/dl (1=true, 0=false)
* restecg:-Hasil elektrokardiografi (0=normal, 1=having ST-T wave normality, 2=hypertrophy)
* thalachh:- Detak jantung maksimum
* exang:-  latihan angina yang diinduksi (1=yes, 0=no)
* oldpeak:- Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat
* slope:- Kemiringan segmen ST latihan(0= dowmsloping, 1=flat, 2=upsloping)
* ca: -jumlah dari major vessels(0-3) coloured by flourosopy)
* thal:-Kelainan darah(1=fixed defect, 2=normal, 3=reversable defect)
* target:- target variable dan predicted attribute(0=less chance of heart attack, 1=high rate of heart attack)

Apabila dilakukan Data Loading adalah sebagai berikut.
![Overview dataset](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/overview.png?raw=true)

Dataset tersebut juga dapat dilihat deskripsi statistiknya seperti berikut:
![describe dataset](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/describe.png?raw=true)


**Visualisasi Data**
</br>Mengecek keseimbangan data berdasarkan kolom target.
![target](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/target.png?raw=true)
Berdasarkan diagram diatas dapat dilihat bahwa target variable dengan value 1 lebih besar dari value 0

</br>Melakukan explorasi data analysis terhadap seluruh kolom karena bertipe numerik.
![eda dataset](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/eda.jpg?raw=true)
Berdasarkan diagram diatas ternyata masih banyak kolom yang memiliki kemiringingan yang tinggi atau tidak seimbang

</br>Mengecek penderita tertinggi berdasarkan usia.
![grafik age](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/grafik%20age.png?raw=true)
Berdasarkan diagram diatas bahwa pasien yang memiliki umur 55-60 lebih rentan terkena penyakit gagal jantung 

</br>Mengecek perbandingan penderita penyakit jantung berdasarkan jenis kelamin
![perbandingan](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/perbandingan.png?raw=true).

</br>Mengecek tingkat kematian pada gender laki-laki.
![lakik](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/laki.png?raw=true)

</br>Mengecek tingkat kematian pada gender perempuan.
![perempuan](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/perempuan.png?raw=true)

</br>Mengecek outlier pada data dengan menggunakan boxplot.
![outlier](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/outlier.png?raw=true)
Dapat dilihat bahwa terdapat outliers di features kita maka dari itu kita akan melakukan IQR Method

**Data Pre-Processing:**
</br>Mengecek korelasi.
![korelasi](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/korelasi.png?raw=true)
Kita dapat melihat bahwa yang memiliki korelasi terendah dengan 'target'  adalah 'fbs' dan 'chol' sedangkan 'oldpeak' dan 'exang' memiliki korelasi negatif yang paling kuat .

)

## Data Preparation
Pada data preparation ini tidak terdapat kolom yang bertipe kategorik maka dari itu daya tidak perlu melakukan Encoding. Karena pada dataset sudah ini kecil setelah melakukan IQR , saya tidak akan menghapus outlier melainkan saya akan menggantinya dengan batas atas atau bawah yang dapat diterima.
![batas](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/batas.png?raw=true)
Kemudian melakukan proses Train-Test Split. Dimana proses ini adalah pembagian dataset menjadi data latih (train) dan data uji (test) merupakan hal yang saya pilih untuk lakukan sebelum membuat model. Hal ini karena data uji berperan sebagai data baru yang benar-benar belum pernah dilihat oleh model sebelumnya sehingga informasi yang terdapat pada data uji tidak mengotori informasi yang terdapat pada data latih, alasan lain mengapa menggunakan train test split karena untuk efisiensi dan tidak melakukan data leakage ketika melakukan scaling.

![Train](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/train.png?raw=true)

## Modeling
Pada Proyek yang dibuat, digunakan model Machine Learning yaitu Logistic Regression, KNN, dan Random Forest. Model tersebut digunakan karena permasalahan dari model Machine Learning yang saya buat adalah permasalahan regresi. Pada tahap ini saya juga melakukan improvement terhadap model dengan menggunakan hyperparameter tuning. Pada Logistic Regression menggunakan parameter random_state=0, pada KNN menggunakan parameter n_neighbors=7, dan pada Random Forest menggunakan parameter n_estimators=100, random_state=0. Lalu untuk membandingkan ketiga model yang saya gunakan ini dilakukan perhitungan dari nilai accuracy_score dari data. Setelah dilakukan pelatihan maka dapat disimpulkan bahwa dengan menggunakan model Random Forest Regression dimana pada model ini memiliki accuracy tertinggi pada f1-score yaitu 100%.

## Evaluation
Pada tahap evaluation akan dijelaskan mengenai metrik yang digunakan dalam prediksi proyek saya dengan menggunakan metrik accuracy f1-score, recall, dan precision. Pada gambar di bawah ini ditampilkan secara detail hasil pengukuran model dengan algoritma Random Forest dengan metriks akurasi, _f1-score_, _recall_, dan _precision_.. 
* Akurasi 
merupakan metrik untuk menghitung nilai ketepatan model dalam memprediksi data dengan data yang sebenarnya. Semakin tinggi nilai accuracy, semakin dekat nilai yang diprediksi   dan diamati. Untuk menghitung nilai dari Akurasi menggunakan rumus berikut:
![Rumus](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/acc.png?raw=true)</br>
Keterangan: 
* accuracy = Nilai akurasi
* n = jumlah data sampel
* i = urutan data
* Y = Nilai hasil observasi
* Ŷ = Nilai hasil prediksi
* Precision
Precision adalah metrik yang digunakan pada kasus klasifikasi untuk menghitung seberapa baik model dalam memprediksi kelas positif terhadap semua prediksi model berkelas positif. Untuk menghitung precision, perlu pemahaman mengenai TP, TN, FP,dan FN. Setelah memahami konsep tersebut, kemudian perhitungan nilai precision dapat menggunakan rumus berikut.
![Rumus](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/presisi.png?raw=true)
</br>

* Recall
Recall adalah metrik yang digunakan pada kasus klasifikasi yang digunakan untuk mengukur seberapa baik model dalam memprediksi kelas positif terhadap semua kelas data positif. Berikut adalah rumus untuk menghitungnya.
![Rumus](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/recall.png?raw=true)
Kelebihan dari metrik ini yaitu ikut menghitung kelas yang terprediksi negatif dari prediksi kelas positif (tidak seperti precision). Namun, metrik ini memiliki kekurangan ketika semua prediksi bernilai = 1 maka recall akan memiliki nilai 1 (tidak memperhitungkan prediksi negatif).
</br>

* f1-score
F1-score adalah metrik yang digunakan pada kasus klasifikasi untuk mengukur seberapa baik hasil prediksi model (precision) dan seberapa lengkap hasil prediksi model tersebut (recall). Rumus untuk menghitungnya dapat dilihat di bawah.
![Rumus](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/f_beta.png?raw=true)
Note : Nilai beta = 1 (F1-score)
</br>
Kelebihan dari metrik ini yaitu dapat menutup semua kekurangan yang ada pada precision dan recall. Namun, F1-score tidak memperhitungkan hasil prediksi benar pada kelas negatif.


Hasil dari evaluation model pada proyek ini mengenai prediksi penyakit jantung pada seseorang dapat dilihat pada gambar di bawah ini. 

![Score Evaluasi](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/score.png?raw=true)</br>
![Grafik Evaluasi](https://github.com/wiwekapremana/MLT-1-dicoding/blob/main/grafik.png?raw=true)</br>

Dari sini kita dapat menyimpulkan bahwa model yang lebih akurat dalam memprediksi serangan jantung pada seseorang adalah dengan menggunakan model Random Forest Regression dimana pada model ini memiliki accuracy tertinggi pada f1-score yaitu 100%..
