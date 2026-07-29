# Uçtan Uca Müşteri Ayrılma Tahmini

Türkiye Yapay Zeka Akademisi Makine Öğrenmesi Final Ödevi kapsamında hazırlanmış uçtan uca bir sınıflandırma projesidir.

## Projenin Amacı

Bu projede müşterilerin temel kullanım ve üyelik özelliklerinden yararlanarak `churn` durumunu tahmin etmeye çalıştım. `churn = 1` müşterinin hizmeti bıraktığını, `churn = 0` ise devam ettiğini gösterir.

Problem türü: **Sınıflandırma**

## Veri Seti

Veri seti `churn_prediction.py` içinde sentetik olarak oluşturulur ve 600 müşteri içerir. Veri yapısı; eksik değer, kategorik değişken, aykırı gözlem ve farklı müşteri davranışlarını içerecek şekilde hazırlanmıştır.

Başlıca değişkenler:

- `yas`
- `gelir`
- `aylik_ucret`
- `abonelik_suresi`
- `destek_talebi_sayisi`
- `son_giris_gun_sayisi`
- `sehir`
- `uyelik_tipi`
- `odeme_yontemi`
- `otomatik_odeme`
- `churn`

## Uygulanan Adımlar

- Temel veri inceleme: `head`, `shape`, veri tipleri ve özet istatistikler
- Eksik değer kontrolü ve `SimpleImputer` ile doldurma
- Kategorik değişkenler için `OneHotEncoder`
- Sayısal değişkenler için `StandardScaler`
- IQR yöntemi ile aykırı değer incelemesi
- Öznitelik mühendisliği: `abonelik_yili`, `destek_talebi_var_mi`, `tahmini_musteri_degeri`
- `SelectKBest` ve `f_classif` ile öznitelik seçimi
- Train / validation / test ayrımı: %60 / %20 / %20
- Sınıf oranlarını korumak için `stratify`
- 5 katlı `StratifiedKFold` çapraz doğrulama
- Validation F1 skoruna göre model karşılaştırması
- Seçilen model için `GridSearchCV`
- Test setinde confusion matrix, accuracy, precision, recall ve F1-score
- Bonus olarak permutation importance ile açıklanabilirlik analizi

## Modeller

Projede üç farklı sınıflandırma modeli karşılaştırılır:

- Logistic Regression
- K-Nearest Neighbors
- Random Forest

## Örnek Sonuçlar

Sabit `random_state=42` ile yapılan çalışmada validation sonuçları yaklaşık olarak:

| Model | Validation F1 | 5-Fold CV F1 |
| --- | ---: | ---: |
| Logistic Regression | 0.6050 | 0.6567 |
| KNN | 0.6341 | 0.5906 |
| Random Forest | 0.6508 | 0.6283 |

Validation F1 skoruna göre **Random Forest** seçildi ve GridSearchCV ile hiperparametre ayarlaması yapıldı.

Örnek test sonuçları:

- Accuracy: `0.6250`
- Precision: `0.6462`
- Recall: `0.6562`
- F1-score: `0.6512`

Permutation importance sonucunda özellikle `son_giris_gun_sayisi`, `tahmini_musteri_degeri`, `aylik_ucret`, `gelir` ve `uyelik_tipi` değişkenleri öne çıktı.

Veri seti sentetik olduğu için bu performans gerçek müşteri davranışına doğrudan genellenmemelidir. Gerçek bir uygulamada daha fazla gözlem, zaman bilgisi ve iş bağlamı ile ek doğrulama yapılması gerekir.

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python churn_prediction.py
```

Kod çalıştırıldığında veri inceleme çıktıları, aykırı değer analizi, model karşılaştırmaları, çapraz doğrulama, GridSearchCV sonucu, test metrikleri ve açıklanabilirlik çıktıları terminalde gösterilir.
