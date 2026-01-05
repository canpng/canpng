# Logistic Regresyon PDF'indeki İlk Alıştırma Sorusunun Çözümü
"""
import numpy as np 
from sklearn.linear_model import LinearRegression

x = np.array([[1], [2], [3], [4], [5], [6]]) # Çift köşeli parantez
y = np.array([12, 19, 29, 42, 58, 77]) # Tek köşeli parantez

model = LinearRegression()
# Çıktı değişkeni olan "Enerji (kWh)" sürekli bir sayısal değerdir.
# Eğer "Düşük/Yüksek" gibi kategoriler olsaydı sınıflandırma olurdu, ancak sayı tahmin ettiğimiz için regresyondur.

model.fit(x, y) # Modeli eğitme

tahmin = model.predict([[7]])  # Çift köşeli parantez
print("7 saatlik çalışma için tahmin edilen enerji (kWh):", tahmin)
"""

# İkinci Alıştırma Sorusunun Çözümü
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

x = np.array([[2, 20], [4, 40], [6, 50], [8, 70], [10, 80], [12, 95]]) # Çift köşeli parantez
y = np.array([0, 0, 0, 1, 1, 1]) # Tek köşeli parantez

model = LogisticRegression()
# Çünkü çıktı değişkeni olan "y" sadece iki sınıfa (0 ve 1) sahip bir değişkendir.
model.fit(x, y)

tahmin = model.predict([[9, 75]])  # Çift köşeli parantez
olasilik = model.predict_proba([[9, 75]])  # Proba = olasılık (Probability)

print("Tahmin edilen sınıf (Bitirdi=1):", tahmin)
print("Tahmin edilen olasılıklar (Bitirmeme=0, Bitirme=1):", olasilik)
"""

# Üçüncü Alıştırma Sorusunun Çözümü
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

x = np.array([[1, 2], [2, 3], [3, 3], [4, 5], [5, 6], [6, 7]]) # Çift köşeli parantez
y = np.array([0, 0, 0, 1, 1, 1]) # Tek köşeli parantez

model = SVC(kernel='linear', C=1.0)
model.fit(x, y) # Modeli eğitme

tahmin = model.predict([[4.5, 5]])  # Çift köşeli parantez
print("SVM Tahmin edilen sınıf:", tahmin)

# SVM Doğruluk hesaplama
y_pred = model.predict(x)
dogruluk = accuracy_score(y, y_pred)
print("SVM Model Doğruluğu:", dogruluk)

# Random Forest modeli
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x, y)
rf_tahmin = rf_model.predict([[4.5, 5]])
print("Random Forest Tahmin edilen sınıf:", rf_tahmin)

# Random Forest Doğruluk hesaplama
rf_y_pred = rf_model.predict(x)
rf_dogruluk = accuracy_score(y, rf_y_pred)
print("Random Forest Model Doğruluğu:", rf_dogruluk)
"""

# Dördüncü Alıştırma Sorusunun Çözümü
"""
import numpy as np
from sklearn.svm import SVC 

# Bir sınıflandırma problemidir.
# Çünkü hedef değişken sayısal bir değer değil, kategorik bir etiket ("Sağlam" veya "Hatalı") içermektedir.

x = np.array([[180, 30, 2.1], [190, 32, 2.3], [220, 45, 3.5], [240, 50, 4.0], [200, 35, 2.6]]) # Çift köşeli parantez
y = np.array(["Sağlam", "Sağlam", "Hatalı", "Hatalı", "Sağlam"]) # Tek köşeli parantez

model = SVC(kernel='linear', C=1.0)
model.fit(x, y) # Modeli eğitme

tahmin = model.predict([[210, 40, 3.0]])  # Çift köşeli parantez
print("Tahmin edilen sınıf:", tahmin)
"""

# Beşinci Alıştırma Sorusunun Çözümü
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 

X = np.array([[20], [30], [40], [50]]) # Çift köşeli parantez
Y = np.array([0, 0, 1, 1]) # Tek köşeli parantez

# Lineer Regresyon bu problem için hatalıdır.
# Çünkü sınıflandırma problemlerinde sonuç ya 0 ya da 1 olmalıdır (veya bu ikisi arasında bir olasılık). 
# Ancak Lineer Regresyon + sonsuz ile - sonsuz arasında değerler üretebilir.
lin_model = LinearRegression()
lin_model.fit(X, Y) # Modeli eğitme
lin_tahmin = lin_model.predict(X)
print("Lineer Regresyon Tahminleri:", lin_tahmin)
# Lineer Regresyon Tahminleri: [-0.1  0.3  0.7  1.1] çıktısını alacaksın. 
# Gördüğün gibi, bazı tahminler 0 ile 1 arasında değil (örneğin -0.1 ve 1.1).

# Doğru Model: LogisticRegression 
log_model = LogisticRegression()
log_model.fit(X, Y) # Modeli eğitme
log_tahmin = log_model.predict(X)
print("Logistic Regression Tahminleri:", log_tahmin)
# Logistic Regression Tahminleri: [0 0 1 1] çıktısını alacaksın. 
# Gördüğün gibi, tüm tahminler ya 0 ya da 1.
"""
