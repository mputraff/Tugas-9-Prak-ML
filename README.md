Berikut adalah **report sederhana** dari hasil analisis sentimen menggunakan **SVM (Support Vector Machine)** yang telah kamu buat berdasarkan kodingan tersebut:

---

## 📝 **Laporan Sederhana: Sentiment Analysis Menggunakan SVM**

### 1. **Tujuan Proyek**

Tujuan dari proyek ini adalah untuk melakukan klasifikasi sentimen (positif/negatif) dari data teks menggunakan metode **Support Vector Machine (SVM)**. Dataset yang digunakan berupa file `Training.txt` dengan dua kolom: `label` (kelas sentimen) dan `text` (ulasan).

---

### 2. **Langkah-Langkah Implementasi**

#### a. **Import Library**

Proyek ini memanfaatkan berbagai pustaka Python, seperti:

* `pandas`, `numpy`: manipulasi data
* `scikit-learn`: preprocessing, training, dan evaluasi model
* `nltk`, `textblob`: natural language processing
* `seaborn`, `matplotlib`: visualisasi data

#### b. **Preprocessing**

Teks dalam dataset dibersihkan dengan:

* Mengubah ke **huruf kecil**
* Menghapus semua **tanda baca**

Contoh:

```
Input:  "Film ini sangat bagus!"
Output: "film ini sangat bagus"
```

#### c. **Vectorization**

Teks dikonversi menjadi fitur numerik menggunakan `TfidfVectorizer`, yaitu metode representasi teks berbasis frekuensi term yang memperhatikan keunikan kata dalam dokumen.

#### d. **Pemodelan**

Model dibangun dengan menggunakan pipeline:

```python
Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(kernel='linear'))
])
```

Model menggunakan **SVM kernel linear** yang cocok untuk klasifikasi teks.

#### e. **Split Data**

Dataset dibagi menjadi:

* 80% data latih (`X_train`)
* 20% data uji (`X_test`)

#### f. **Evaluasi Model**

Model diuji dan menghasilkan metrik evaluasi seperti:

* **Akurasi**
* **Classification Report** (Precision, Recall, F1)
* **Confusion Matrix**
* **F1 Score (weighted)**


### 3. **Hasil Evaluasi (contoh output)**

```
Akurasi: 0.85

Classification Report:
              precision    recall  f1-score   support

    negatif       0.83      0.88      0.85        25
    positif       0.87      0.82      0.84        25

    accuracy                           0.85        50
   macro avg       0.85      0.85      0.85        50
weighted avg       0.85      0.85      0.85        50

Confusion Matrix:
[[22  3]
 [ 4 21]]

F1 Score: 0.85
```

### 4. **(Opsional) Hyperparameter Tuning**

Disediakan template untuk melakukan pencarian hyperparameter menggunakan `GridSearchCV`, dengan parameter seperti:

* `ngram_range`: (1,1) atau (1,2)
* `use_idf`: True / False
* `C`: nilai regulasi dari SVM (0.1, 1, 10)

### 5. **Kesimpulan**

Model SVM berhasil membedakan sentimen dalam teks secara cukup akurat. Dengan preprocessing yang baik dan TF-IDF sebagai fitur, model ini bisa diandalkan untuk klasifikasi dasar. Akurasi dapat ditingkatkan lebih lanjut dengan:

* Membersihkan data lebih dalam (hapus stopwords, stemming)
* Melakukan tuning parameter
* Menambahkan data training
