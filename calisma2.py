# SVR (Support Vector REGRESSION): Regresyon yapar. Yani sonucu bir Sayı olarak tahmin eder.
# Örnek: "Bu evin fiyatı ne kadar?", "Yarın hava kaç derece olacak?", "Çeliğe 50N kuvvet uygularsam kaç mm uzar?"

# SVC (Support Vector CLASSIFICATION): Sınıflandırma yapar. Yani sonucu bir Grup/Sınıf (Etiket) olarak tahmin eder.
# Örnek: "Bu parça Sağlam mı / Bozuk mu?", "Gelen e-posta Spam mi / Değil mi?", "Hasta mı / Sağlıklı mı?"

# Girişler daima iki köşeli parantez içinde verilir. Çünkü model her zaman 2 boyutlu veri bekler.
# Çıkışlar ise tek köşeli parantez içinde verilir. Çünkü sonuç tek bir değerdir.

# kernel='linear': Veriler düz bir çizgi gibiyse kullanılır (Kuvvet-uzama grafiği gibi)
# kernel='rbf': Verilerin dalgalıysa, kıvrımlıysa (doğrusal değilse) bu seçilir. SVR'ın bükülebilmesini, eğrisel yollar çizmesini sağlar.

# Epsilon: SVR modelinin tolerans seviyesini belirler. Yani modelin ne kadar hata yapabileceğini kontrol eder.
# Küçük epsilon değeri: Model veriye daha sıkı uyar, daha az hata toleransı. (Daha karmaşık model)
# Büyük epsilon değeri: Model veriye daha gevşek uyar, daha fazla hata toleransı. (Daha basit model)

# C parametresi: Modelin hatalara karşı ne kadar sert olacağıdır.
# Küçük C değeri: Model hatalara daha toleranslı olur, daha yumuşak bir sınır çizer.
# Büyük C değeri: Model hatalara daha az toleranslı olur, daha sert bir sınır çizer.

# Gamma parametresi (Sadece RBF kernel için): Modelin veri noktalarına ne kadar yakın etki edeceğini belirler.
# Küçük gamma değeri: Bir nokta çok uzakları bile etkiler. Grafik daha yumuşak ve pürüzsüz olur.
# Büyük gamma değeri: Her nokta sadece kendi yakın çevresini etkiler. Grafik çok kıvrımlı ve tırtıklı olur.


# Support Vector Regression PDF'indeki İlk Alıştırma Sorusunun Çözümü
"""
import numpy as np 
from sklearn.svm import SVC

X = np.array([[2.1, 50], [2.4, 55], [3.0, 60], [4.9, 120], [5.2, 130], [5.6, 125]]) # Çift köşeli parantez
y = np.array(["Sağlam", "Sağlam", "Sağlam", "Arızalı", "Arızalı", "Arızalı"]) # Tek köşeli parantez

model = SVC(kernel='linear', C=1.0)  # Soruda lineer kernel istendi. Gamma RBF için geçerli. Epsilon SVR için geçerli.

model.fit(X, y) # Modeli eğitme

tahmin = model.predict([[4.0, 100]])  # Çift köşeli parantez. Çünkü giriş 2 boyutlu.
print("Tahmin edilen sınıf:", tahmin)
"""

# İkinci Alıştırma Sorusunun Çözümü
"""
import numpy as np 
from sklearn.svm import SVC
import matplotlib.pyplot as plt # Grafik çizimi için

x = np.array([[10, 200], [9.8, 210], [10.5, 240], [10.7, 250]]) # Çift köşeli parantez
y = np.array(["Hatasız", "Hatasız", "Hatalı", "Hatalı"]) # Tek köşeli parantez

model = SVC(kernel='linear', C=1.0)  
model.fit(x, y) # Modeli eğitme
tahmin = model.predict([[10.2, 220]])  # Çift köşeli parantez
print("Tahmin edilen sınıf:", tahmin)

# Grafik çizimi
plt.scatter(x[:, 0], x[:, 1], c=['blue' if label == 'Hatasız' else 'red' for label in y]) # Verileri renklendirerek çizme
plt.xlabel('Kalınlık (mm)') # X ekseni etiketi
plt.ylabel('Sıcaklık (°C)') # Y ekseni etiketi
plt.title('Parça Durumu Sınıflandırması') # Grafik başlığı
plt.show() # Grafiği göstermek için

# Grafik üzerinde gördüğünde, "Hatasız" ve "Hatalı" sınıflarının birbirinden net bir şekilde ayrıldığını göreceksin.
# Veriler birbirine karışmış veya iç içe geçmiş olsaydı, RBF kernel kullanmak daha iyi olurdu.
"""

# Üçüncü Alıştırma Sorusunun Çözümü
"""
import numpy as np 
from sklearn.svm import SVR # Soruda SVR istendi.
import matplotlib.pyplot as plt # Grafik çizimi için

x = np.array([[100], [150], [200], [250]])
y = np.array([0.0005, 0.0009, 0.0012, 0.0016])

model = SVR(kernel='linear', C=1000, epsilon=0.0001)  
# Elastik bölge için lineer kernel uygundur. Hooke Kanununa göre gerilme ve şekil değiştirme doğru orantılıdır.
# C değerini büyük tutuyoruz çünkü hatalara karşı daha az toleranslı olmasını istiyoruz.
# Epsilon değerini küçük tutuyoruz çünkü hedef değerler (0.0005 gibi) çok küçük.

model.fit(x, y) # Modeli eğitme

tahmin = model.predict([[300]])  # Tek özellikli giriş için çift köşeli parantez
print("300 MPa gerilme için tahmin edilen şekil değiştirme (ε):", tahmin)

# Grafik çizimi
plt.scatter(x, y, color='blue', label='Veri Noktaları') # Veri noktalarını çizme
plt.plot(x, model.predict(x), color='red', label='SVR Modeli') # Modelin tahmin çizgisi
plt.xlabel('Gerilme (MPa)') # X ekseni etiketi
plt.ylabel('Şekil Değiştirme (ε)') # Y ekseni etiketi
plt.title('Malzeme Gerilme–Şekil Değiştirme Regresyonu') # Grafik başlığı
plt.legend() # Sol üst köşede açıklama kutusu gösterir
plt.show() # Grafiği göstermek için
"""
