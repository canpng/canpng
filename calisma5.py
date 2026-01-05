# Yapay Sinir Ağı Nedir PDF'indeki Soru-3'ün Çözümü
"""
import numpy as np
from sklearn.neural_network import MLPClassifier # MLPRegressor kullanılmaz çünkü sınıflandırma problemi var.
from sklearn.model_selection import train_test_split # Soruda belirtilmemiş ama derste bu şekilde yapmıştık.
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score # Doğruluk hesaplama için

x = np.array([[10.0, 160, 1.1], [10.1, 158, 1.2], [9.8, 150, 1.5], [9.7, 148, 1.6],
              [10.4, 140, 2.1], [10.5, 138, 2.3], [9.9, 152, 1.4], [10.2, 156, 1.3]]) # Çift köşeli parantez
y = np.array(["A", "A", "B", "B", "C", "C", "B", "A"]) # Tek köşeli parantez

# Veriyi eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# scaler (Ölçekleyici) Nedir ve Neden Kullanıyoruz?
# Eğer StandardScaler kullanmazsan, matematiksel formülde (z = w.x + b) 160 sayısı 1.1 sayısını ezer geçer. 
# Model şöyle düşünür: "Sertlik değeri çok büyük, demek ki sonuç üzerinde en önemli etki sertlikte, pürüzlülük önemsiz."
# Oysa mühendislikte 1.5 pürüzlülük kritik bir hata olabilir. StandardScaler tüm bu sütunları alır ve onları "ortak bir dile" çevirir (genellikle ortalaması 0, sapması 1 olacak şekilde dönüştürür).
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# max_iter (Maksimum İterasyon) Nedir?
# Yapay sinir ağı, tek seferde öğrenmez. Bir döngü içinde sürekli tahmin yapar, hatasını ölçer, hatasını azaltmak için kendini günceller ve tekrar tahmin yapar.
# max_iter=1000: Modele "Veri setini en fazla 1000 kere dönerek hatanı sıfırlamaya çalış" emrini verirsin.
# Eğer bu sayıyı çok küçük (örneğin 10) yaparsan, model daha öğrenemeden "süre bitti" der ve eğitimi keser (uyarı verir: ConvergenceWarning).

# hidden_layer_sizes (Gizli Katman Boyutları) Nedir?
# hidden_layer_sizes parametresi, o aradaki "düşünen nöronların" sayısını belirler.
# Kodumuzda: hidden_layer_sizes=(5,) Bu, 1 tane gizli katman olsun ve içinde 5 tane nöron (işlemci hücre) olsun demektir.
# Alternatif: hidden_layer_sizes=(10, 5) yapsaydın;
# 2 gizli katman olurdu. İlkinde 10 nöron, ikincisinde 5 nöron olurdu. Buna "Derin Sinir Ağı" denirdi.
# Neye göre seçilir? Az nöron: Model yetersiz kalır, sorunu çözemez.
# Çok nöron: Model veriyi ezberler (Overfitting). Verimiz küçük olduğu için (5,) uygundur.

model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
model.fit(x_train, y_train) # Modeli eğitme

# Test seti üzerinde tahmin yapma
y_pred = model.predict(x_test)

# Doğruluk hesaplama
dogruluk = accuracy_score(y_test, y_pred)
print("Model Doğruluğu:", dogruluk)

# Yeni parça için tahmin
yeni_parca = np.array([[10.3, 145, 2.0]])
yeni_parca_scaled = scaler.transform(yeni_parca)
tahmin = model.predict(yeni_parca_scaled)
print("Yeni parça için tahmin edilen sınıf:", tahmin)
"""

# Sınav Sorusu Olarak Sorduğunu Söylemişti
"""
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Veri Setini Yükleme
df = pd.read_excel("calisma5.xlsx")

# 2. Girdi ve Çıktı Değişkenlerini Ayırma
X = df.drop(columns=[df.columns[-1]]) # Sadece son sütunu al
y = df[df.columns[-1]]


# 3. Train/Test böl. Veri Setini Ayırma (Eğitim %80, Test %20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 4. Ölçeklendirme (StandardScaler)
# PDF'te Soru-1 çözümünde belirtildiği gibi scaler sadece train setine fit edilir [cite: 596-599].
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. ANN modeli (MLPRegressor)
model = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    max_iter= 1000,
    random_state=42,
    learning_rate_init= 0.001, # Ağırlıklar ne kadar güncellenecek.
    early_stopping=True,
    n_iter_no_change=10 # 10 EPAUCH boyunca ilerleme görülmezse eğitim duracak.
)

# Modeli Eğitme
model.fit(X_train_scaled, y_train)

# 6. Performans Değerlendirme
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performansı:\nMAE: {mae:.2f}\nR² Skoru: {r2:.2f}")

# 7. Yeni Karışım Tahmini
# Girdiler: [Çimento, Cüruf, Kül, Su, Akışkan, İri Agrega, İnce Agrega, Yaş]
yeni_karisim = np.array([[350, 120, 0, 190, 5, 1000, 700, 28]])

# Yeni veriyi de aynı scaler ile ölçeklendiriyoruz.
yeni_karisim_scaled = scaler.transform(yeni_karisim)

tahmin = model.predict(yeni_karisim_scaled)[0]

print("-" * 30)
print(f"Tahmin Edilen Beton Basınç Dayanımı: {tahmin:.2f} MPa")
"""