# Prediksi Nilai Akademik dari Kebiasaan Gaming

Model regresi machine learning yang memprediksi nilai akademik (`grades`) dari kebiasaan bermain game, jam belajar, tidur, kehadiran, tingkat kecanduan, dan stres. Tiga pendekatan ensemble dibandingkan, di-tuning dengan validasi silang, lalu dijelaskan dengan **SHAP**.

> Proyek mini, dikerjakan di Google Colab.

## Dataset

[Gaming vs Academic Performance](https://www.kaggle.com/datasets/aiexplorer77/gaming-vs-academic-performance) (Kaggle): 8.000 baris dan 14 kolom, target `grades`.
Satu fitur dibuat sendiri: `gaming_study_ratio = gaming_hours / study_hours`.
Data dibagi 80:20 (6.400 latih, 1.600 uji, `random_state=42`). Dataset tidak disertakan di repositori, silakan cek lisensinya di Kaggle.

## Alur

1. **Analisis karakteristik data dan error.** Bias dan variance dibandingkan pada Decision Tree, Random Forest, dan XGBoost.
2. **Arsitektur ensemble.** Bagging (Random Forest), boosting (XGBoost), dan stacking (RF + XGBoost dengan meta-learner).
3. **Tuning hyperparameter.** Pencarian dengan validasi silang 5-fold (36 kandidat untuk Random Forest, 30 untuk XGBoost).
4. **Validasi K-Fold (5-fold)** pada model yang sudah di-tuning.
5. **Interpretasi model (XAI)** dengan SHAP values.

## Hasil

**Perbandingan arsitektur (R² data uji, sebelum tuning)**

| Model | Metode | R² |
|---|---|---|
| Random Forest | Bagging (paralel) | 0,9234 |
| XGBoost | Boosting (sekuensial) | 0,9274 |
| Stacking (RF + XGB) | Meta-learner | 0,9254 |

**Setelah tuning**

| Model | CV R² | Train R² | Test R² | Rata-rata R² 5-fold |
|---|---|---|---|---|
| Random Forest | 0,9321 | 0,9778 | 0,9242 | 0,9311 |
| XGBoost | 0,9363 | 0,9589 | **0,9280** | **0,9346** |

Parameter terbaik XGBoost: `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=1.0`.
Parameter terbaik Random Forest: `n_estimators=200`, `max_depth=None`, `min_samples_split=10`.

**Interpretasi (SHAP).** Tiga fitur paling berpengaruh pada prediksi nilai: `gaming_study_ratio`, `study_hours`, dan `sleep_hours`.

> Model ini memprediksi, bukan membuktikan sebab-akibat. Hubungan antara gaming dan nilai pada data ini tidak otomatis berlaku untuk kasus lain.

![Analisis bias dan variance](images/grafik1.png)
![Perbandingan arsitektur ensemble](images/grafik2.png)
![Interpretasi SHAP](images/grafik4.png)

## Menjalankan

Cara termudah: unggah `notebooks/gaming_vs_academic.ipynb` ke Google Colab dan jalankan semua sel.
Secara lokal:
```bash
pip install -r requirements.txt
jupyter notebook notebooks/gaming_vs_academic.ipynb
```
Dataset diunduh otomatis lewat `kagglehub`
## Struktur

```
notebooks/   notebook analisis dan pelatihan
requirements.txt
```

## Teknologi

Python, pandas, scikit-learn, XGBoost, SHAP, matplotlib.

