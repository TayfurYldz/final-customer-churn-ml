"""
Uçtan Uca Müşteri Ayrılma Tahmini (Customer Churn Prediction)

Amaç:
    Müşteri özelliklerinden yararlanarak churn durumunu tahmin etmek,
    farklı sınıflandırma modellerini karşılaştırmak ve en iyi modeli
    hiperparametre ayarı sonrası test setinde değerlendirmek.

Kullanılan kütüphaneler:
    numpy, pandas ve scikit-learn

Çalıştırma:
    pip install -r requirements.txt
    python churn_prediction.py

Not:
    Veri seti eğitim amacıyla bu dosya içinde sentetik olarak üretilir.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def veri_seti_olustur(n_samples=600):
    """Eğitim amacıyla churn ile ilişkili sentetik müşteri verisi üretir."""
    rng = np.random.default_rng(RANDOM_STATE)

    df = pd.DataFrame({
        "yas": rng.integers(18, 71, size=n_samples),
        "gelir": np.clip(rng.normal(12000, 4500, size=n_samples), 3000, 40000),
        "aylik_ucret": np.clip(rng.normal(650, 220, size=n_samples), 150, 1600),
        "abonelik_suresi": rng.integers(1, 73, size=n_samples),
        "destek_talebi_sayisi": np.clip(rng.poisson(2.2, size=n_samples), 0, 12),
        "son_giris_gun_sayisi": np.clip(rng.gamma(2.0, 5.0, size=n_samples), 0, 60),
        "sehir": rng.choice(
            ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"],
            size=n_samples,
        ),
        "uyelik_tipi": rng.choice(
            ["Temel", "Standart", "Premium"],
            size=n_samples,
            p=[0.35, 0.45, 0.20],
        ),
        "odeme_yontemi": rng.choice(
            ["Kredi Kartı", "Banka Kartı", "Havale"],
            size=n_samples,
            p=[0.55, 0.30, 0.15],
        ),
        "otomatik_odeme": rng.choice(
            ["Evet", "Hayır"],
            size=n_samples,
            p=[0.62, 0.38],
        ),
    })

    # IQR incelemesinde görülebilecek birkaç makul uç değer ekliyorum.
    outlier_idx = rng.choice(df.index, size=6, replace=False)
    df.loc[outlier_idx[:3], "gelir"] *= 1.8
    df.loc[outlier_idx[3:], "aylik_ucret"] *= 1.6

    # Eksik değer işlemini göstermek için bazı değerleri bilinçli boş bırakıyorum.
    df.loc[rng.choice(df.index, size=18, replace=False), "gelir"] = np.nan
    df.loc[rng.choice(df.index, size=12, replace=False), "sehir"] = np.nan
    df.loc[rng.choice(df.index, size=10, replace=False), "odeme_yontemi"] = np.nan

    gelir_dolu = df["gelir"].fillna(df["gelir"].median())

    # Churn tamamen tahmin edilebilir olmasın diye ilişkilere rastgelelik ekleniyor.
    skor = (
        -0.45
        + 0.24 * df["destek_talebi_sayisi"]
        + 0.035 * df["son_giris_gun_sayisi"]
        - 0.025 * df["abonelik_suresi"]
        + 0.0012 * df["aylik_ucret"]
        - 0.000025 * gelir_dolu
        + df["uyelik_tipi"].map(
            {"Temel": 0.45, "Standart": 0.05, "Premium": -0.35}
        )
        + df["otomatik_odeme"].map({"Evet": -0.35, "Hayır": 0.30})
        + rng.normal(0, 0.65, size=n_samples)
    )

    olasilik = 1 / (1 + np.exp(-skor))
    df["churn"] = rng.binomial(1, olasilik)

    return df


def aykiri_deger_ozeti(df, kolonlar):
    """IQR ile aykırı değerleri raporlar ve neden silinmediğini açıklar."""
    print("\nAykırı Değer İncelemesi (IQR)")
    print("-" * 45)

    for kolon in kolonlar:
        seri = df[kolon].dropna()
        q1, q3 = seri.quantile([0.25, 0.75])
        iqr = q3 - q1
        alt_sinir = q1 - 1.5 * iqr
        ust_sinir = q3 + 1.5 * iqr

        aykiri_sayisi = ((seri < alt_sinir) | (seri > ust_sinir)).sum()

        print(
            f"{kolon}: {aykiri_sayisi} aykırı gözlem "
            f"| sınırlar: {alt_sinir:.2f} - {ust_sinir:.2f}"
        )

    print(
        "Yorum: Bu değerler gerçek hayatta mümkün müşteri profillerini "
        "temsil edebileceği için veri kaybı oluşturmamak adına silinmedi."
    )


def main():
    # 1) Veri seti ve problem tanımı
    df = veri_seti_olustur()

    print("Problem Türü: Sınıflandırma")
    print("Hedef Değişken: churn (1 = ayrıldı, 0 = kaldı)")

    # 2) Temel veri inceleme
    print("\nİlk 5 Satır:")
    print(df.head())

    print("\nVeri Boyutu:", df.shape)

    print("\nVeri Tipleri:")
    print(df.dtypes)

    print("\nTemel İstatistikler:")
    print(df.describe(include="all"))

    print("\nEksik Değerler:")
    print(df.isnull().sum())

    print("\nChurn Dağılımı:")
    print(df["churn"].value_counts())
    print(df["churn"].value_counts(normalize=True).rename("oran"))

    # 3) Aykırı değer analizi
    aykiri_deger_ozeti(
        df,
        ["gelir", "aylik_ucret", "abonelik_suresi", "destek_talebi_sayisi"],
    )

    # 4) Öznitelik mühendisliği
    df["abonelik_yili"] = df["abonelik_suresi"] / 12
    df["destek_talebi_var_mi"] = (df["destek_talebi_sayisi"] > 0).astype(int)
    df["tahmini_musteri_degeri"] = df["aylik_ucret"] * df["abonelik_suresi"]

    X = df.drop(columns="churn")
    y = df["churn"]

    # 5) %60 train, %20 validation, %20 test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    print("\nBölünmüş Veri Boyutları:")
    print("Eğitim:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    sayisal_kolonlar = [
        "yas",
        "gelir",
        "aylik_ucret",
        "abonelik_suresi",
        "destek_talebi_sayisi",
        "son_giris_gun_sayisi",
        "abonelik_yili",
        "tahmini_musteri_degeri",
    ]

    kategorik_kolonlar = [
        "sehir",
        "uyelik_tipi",
        "odeme_yontemi",
        "otomatik_odeme",
        "destek_talebi_var_mi",
    ]

    sayisal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    kategorik_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("sayisal", sayisal_pipeline, sayisal_kolonlar),
        ("kategorik", kategorik_pipeline, kategorik_kolonlar),
    ])

    # 6) SelectKBest ile öznitelik seçimi ve üç model karşılaştırması
    modeller = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    validation_sonuclari = {}

    print("\nModel Karşılaştırması")
    print("-" * 80)

    for isim, model in modeller.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(score_func=f_classif, k=12)),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict(X_val)

        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        cv_f1 = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
        ).mean()

        validation_sonuclari[isim] = {
            "accuracy": val_accuracy,
            "f1": val_f1,
            "cv_f1": cv_f1,
        }

        print(
            f"{isim:20s} | "
            f"Val Accuracy: {val_accuracy:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"5-Fold CV F1: {cv_f1:.4f}"
        )

    en_iyi_model_adi = max(
        validation_sonuclari,
        key=lambda isim: validation_sonuclari[isim]["f1"],
    )

    print(f"\nValidation F1'a göre seçilen model: {en_iyi_model_adi}")

    # 7) Seçilen model ailesi için GridSearchCV
    tuning_ayarlari = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1500, random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 1.0, 5.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [5, 7, 11],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [150, 250],
                "model__max_depth": [5, 8, None],
                "model__min_samples_split": [2, 5],
            },
        ),
    }

    secilen_model, param_grid = tuning_ayarlari[en_iyi_model_adi]

    tuning_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("selector", SelectKBest(score_func=f_classif, k=12)),
        ("model", secilen_model),
    ])

    grid = GridSearchCV(
        tuning_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )

    # Model seçimi bittiği için train + validation birleştirilerek tuning yapılır.
    grid.fit(X_train_val, y_train_val)
    final_model = grid.best_estimator_

    print("\nGrid Search Sonucu")
    print("-" * 45)
    print("En iyi parametreler:", grid.best_params_)
    print(f"En iyi CV F1: {grid.best_score_:.4f}")

    # 8) Test setinde final değerlendirme
    y_test_pred = final_model.predict(X_test)

    print("\nTest Seti Sonuçları")
    print("-" * 45)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
    print(
        f"Precision: "
        f"{precision_score(y_test, y_test_pred, zero_division=0):.4f}"
    )
    print(
        f"Recall:    "
        f"{recall_score(y_test, y_test_pred, zero_division=0):.4f}"
    )
    print(
        f"F1 Score:  "
        f"{f1_score(y_test, y_test_pred, zero_division=0):.4f}"
    )

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    # 9) Bonus: permutation importance ile açıklanabilirlik
    importance = permutation_importance(
        final_model,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "ozellik": X_test.columns,
        "onem": importance.importances_mean,
    }).sort_values("onem", ascending=False)

    print("\nPermutation Importance - En Önemli 8 Özellik")
    print("-" * 55)
    print(importance_df.head(8).to_string(index=False))

    print("\nSonuç Yorumu")
    print("-" * 45)
    print(
        f"Validation sonuçlarında en yüksek F1 değerini {en_iyi_model_adi} verdi. "
        "Bu model ailesi GridSearchCV ile ayarlandı ve yalnızca son aşamada "
        "test verisi üzerinde değerlendirildi. Permutation importance çıktısı "
        "hangi giriş değişkenlerinin tahmin üzerinde daha etkili olduğunu gösterir. "
        "Veri seti sentetik olduğu için sonuçlar gerçek müşteri davranışına "
        "doğrudan genellenmemelidir; gerçek projede daha fazla gözlem, zaman "
        "bilgisi ve iş bağlamı ile doğrulama yapılması gerekir."
    )


if __name__ == "__main__":
    main()
