# Random Forest PDF'inin İlk Alıştırma Sorusunun Çözümü
"""
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Veri seti
x = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([2.0, 4.1, 5.9, 6.8, 7.2, 7.4])

# scikit-learn kütüphanesinde n_estimators değeri varsayılan olarak 100'dür. Kodda bu standardı kullandım.
# Bu olayın mantığı: Bir proje için 1 mühendise danıştığını düşün. Hata yapabilir. 
# Proje için 10 mühendise danışırsan, ortak karar daha doğru olur. 100 mühendise danışırsan sonuç çok daha güvenilir olur.
# 1000 yapsak daha iyi olmaz mı? Genellikle evet, ağaç sayısı arttıkça hata oranı düşer ve sonuçlar daha kararlı hale gelir. 
# Ancak bir noktadan sonra (örneğin 500 ağaçtan sonra) doğruluk artışı durur. İşlem süresi artmaya devam eder.
# Az Ağaç (örn: 10): Hızlı çalışır ama hata yapma riski yüksektir.
# Çok Ağaç (örn: 1000): Daha doğru sonuçlar verir ama işlem süresi uzar.

# Random Forest'da algoritma her çalıştığında, veri setinden rastgele örnekler seçer. 
# Bu parametreye bir sayı verdiğimizde (42, 0, 1999 fark etmez), bilgisayarın ürettiği rastgele sayıları kilitleriz. 
# Yani; sen de bilgisayarında random_state=42 yazarsan, benimle birebir aynı ağaçları kurar, birebir aynı sonucu alırsın.
# Douglas Adams'ın "Otostopçunun Galaksi Rehberi" kitabında, "Hayatın, Evrenin ve Her Şeyin Anlamı"nı hesaplayan süper bilgisayarın verdiği cevap 42'dir.
# Yazılımcılar bu kitaba atıfta bulunarak genellikle 42 sayısını kullanırlar. Hoca da bu şekilde yapıyor.
# Sen oraya random_state=1 de yazsan, 2025 de yazsan kod yine sabitlenir. Sadece farklı bir sabit olur.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x, y)

# Grid oluştur
x_grid = np.arange(1, 8).reshape(-1, 1)

# RF tahminlerini hesapla
y_pred = rf_model.predict(x_grid)

# Grafik çizimi
x_plot = np.linspace(1, 6, 200).reshape(-1, 1)
y_plot = rf_model.predict(x_plot)
# np.linspace kodunun görevi, belirtilen başlangıç ve bitiş değerleri arasında eşit aralıklı sayılar üretmektir.
# Burada 1 ile 6 arasında 200 eşit aralıklı sayı üretiyoruz.
# Elimizde sadece 6 tane veri noktası var. Eğer sadece bunları çizdirirsek grafik kopuk kopuk veya çok köşeli olur.
# np.linspace ile 1 ile 6 arasında çok daha fazla nokta üretip, bu noktalara karşılık gelen tahminleri alıyoruz.
# Random Forest'ta "merdiven" yapısı görülmelidir.

# np.linspace komutu sayıları yan yana dizilmiş bir liste (vektör) olarak verir: [1.0, 1.1, 1.2...]
# Scikit-learn modelleri, girdileri bir tablo olarak ister. Yani "Satırlar" ve "Sütunlar" olmalıdır. 
# Bu yüzden .reshape(-1, 1) kullanıyoruz. Bu komut, tek sütunlu bir tablo oluşturur.
# -1: "Satır sayısını sen hesapla, ben uğraşmayayım" demektir. Python, eleman sayısına göre satır sayısını otomatik ayarlar.
# 1: "Bunu tek sütun haline getir" demektir.

# rf_model.predict(x_plot) ne demek? Eğittiğimiz modele aşağıdaki soruyu soruyoruz:
# Elimde gerçek veriler değil ama, 1 ile 6 arasında oluşturduğum 200 tane farklı nokta (x_plot) var. Eğer giriş değerleri bunlar olsaydı, sonucun (Y) ne olacağını tahmin eder misin?
# Model, bu 200 noktanın her biri için tek tek hesaplama yapar. 

# y_plot, modelin o 200 sanal nokta için verdiği cevaplardır (tahminlerdir). Grafiğin Y eksenini oluşturur.

plt.scatter(x, y, color='red', label='Veri Noktaları')
plt.plot(x_plot, y_plot, color='blue', label='Random Forest Tahmini')
# Eğer x_plot ve y_plot kullanmasaydık ve sadece plt.plot(x, y) deseydik; Python 1. noktadan 2. noktaya dümdüz bir çizgi çekerdi.
# Ama Random Forest düz çizgi çizmez, basamaklı (merdiven gibi) tahmin yapar.
plt.title('Random Forest Regresyon')
plt.xlabel('X Değeri')
plt.ylabel('Tahmin Edilen Y Değeri')
plt.legend()
plt.show()
"""

# İkinci Alıştırma Sorusunun Çözümü
"""
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

x = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([2.0, 4.1, 5.9, 6.8, 7.2, 7.4])

# max_depth, Random Forest içindeki her bir karar ağacının maksimum ne kadar derinleşebileceğini belirler.

# max_depth=None (none=sınırsız) Ağaç, her bir veri noktasını tek tek ayırt edene kadar sonsuz soru sorabilir.
# Model eğitim verisindeki her noktayı mükemmel ezberler. Grafik çok girintili çıkıntılı olur. Ancak yeni bir veri geldiğinde hata yapma olasılığı artar.

# max_depth=3 Ağacın sadece 3 soru sorma hakkı vardır. Model çok detaya inemez, daha genel tahminler yapar. Veriyi ezberleyemez.

# Makine öğrenmesinde hiçbir zaman verinin ezberlenmesi (overfitting) istenmez.
# Senaryo A: Ezberci Öğrenci = Bu öğrenci derste çözülen 50 sorunun cevabını sırasıyla ezberler.
# Sınavda aynı 50 soru çıkarsa mükemmel puan alır. Hoca sayıları değiştirip sorarsa öğrenci çuvallar. Çünkü formülü değil, cevabı ezberlemiştir.
# Senaryo B: Çalışkan Öğrenci = Bu öğrenci derste çözülen 50 sorunun mantığını kavrar, formülleri öğrenir.
# Belki %90 başarı sağlar (bazı detayları kaçırabilir). Yeni sorular gelse de %85-90 civarı not alır. İstenilen model budur.

# Model A: n_estimators=50, max_depth=None
model_a = RandomForestRegressor(n_estimators=50, max_depth=None, random_state=42)
model_a.fit(x, y)
# Model B: n_estimators=500, max_depth=3
model_b = RandomForestRegressor(n_estimators=500, max_depth=3, random_state=42)
model_b.fit(x, y)
# Grid oluştur
x_grid = np.arange(1, 8).reshape(-1, 1)

# Her iki model için tahminleri hesapla
y_pred_a = model_a.predict(x_grid)
y_pred_b = model_b.predict(x_grid)
# Grafik çizimi
x_plot = np.linspace(1, 6, 200).reshape(-1, 1)
y_plot_a = model_a.predict(x_plot)
y_plot_b = model_b.predict(x_plot)
plt.scatter(x, y, color='red', label='Veri Noktaları')
plt.plot(x_plot, y_plot_a, color='blue', label='Model A Tahmini (n_estimators=50, max_depth=None)')
plt.plot(x_plot, y_plot_b, color='green', label='Model B Tahmini (n_estimators=500, max_depth=3)')
plt.title('Random Forest Regresyon - Model A vs Model B')
plt.xlabel('X Değeri')
plt.ylabel('Tahmin Edilen Y Değeri')
plt.legend()
plt.show()
"""

# Üçüncü Alıştırma Sorusunun Çözümü
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# RMSE (Root Mean Square Error), Türkçe adıyla Kök Ortalama Kare Hata demektir. 
# Regresyon (sayı tahmini) problemlerinde modelin ne kadar hata yaptığını ölçer.
# O yüzden, mean_squared_error kütüphanesini kullanıyoruz. 
# mean_squared_error fonksiyonu, gerçek değerler ile modelin tahmin ettiği değerler arasındaki farkların karelerinin ortalamasını verir.
# Değer ne kadar düşükse (0'a yakınsa), model o kadar başarılıdır.

# Önceki sorunun veri setini kullandım.
x = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([2.0, 4.1, 5.9, 6.8, 7.2, 7.4])

n_estimators_list = [5, 10, 20, 50, 100, 300, 600]
rmse_sabit = [] 
rmse_degisken = []

# Neden for Döngüsü Kullanıyoruz? 
# Eğer for döngüsü kullanmasaydık, 7 farklı ağaç (n_estimators) sayısı için aynı kodları 7 kez alt alta kopyalayıp yapıştırmamız gerekirdi.

# Birinci istenen: Sabit ve değişken random_state ile modelleri eğitme
for n_estimators in n_estimators_list:
    # Sabit random_state
    model_sabit = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model_sabit.fit(x, y)
    y_pred_sabit = model_sabit.predict(x)
    rmse_sabit.append(np.sqrt(mean_squared_error(y, y_pred_sabit)))
    
    # Değişken random_state
    model_degisken = RandomForestRegressor(n_estimators=n_estimators, random_state=None)
    model_degisken.fit(x, y)
    y_pred_degisken = model_degisken.predict(x)
    rmse_degisken.append(np.sqrt(mean_squared_error(y, y_pred_degisken)))

# İkinci istenen: Ağaç sayısı – RMSE grafiğini çizme
plt.plot(n_estimators_list, rmse_sabit, marker='o', label='Sabit random_state (42)')
plt.plot(n_estimators_list, rmse_degisken, marker='x', label='Değişken random_state')
plt.title('Ağaç Sayısı vs RMSE')
plt.xlabel('Ağaç Sayısı (n_estimators)')
plt.ylabel('RMSE')
plt.legend()
plt.grid() # Grafikteki ızgara görünümünü sağlıyor, şart değil.
plt.show()

# Düşük Ağaç Sayısında (5, 10, 20) Değişken random_state çok oynaktır. Hata bazen yüksek, bazen düşük çıkar. 
# Çünkü az sayıda ağaç ile rastgele kararlar çok değişir.

# Yüksek Ağaç Sayısında (100, 300, 600) çizgiler daha stabil hale gelir. Hata değerleri birbirine yakınlaşır.
# Ağaç sayısını artırmak (n_estimators), sonuçları stabil (kararlı) hale getirir. 
# Bu yüzden n_estimators değeri genellikle 100 veya üzeri tercih edilir.
"""

# Dördüncü Alıştırma Sorusunun Çözümü
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Veri seti
x = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([2.0, 4.1, 5.9, 6.8, 7.2, 7.4])

# Modelleri oluşturma ve eğitme
# 1. Linear Regresyon
lr_model = LinearRegression()
lr_model.fit(x, y)

# 2. Support Vector Regresyon
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1) # Neden RBF çünkü veri doğrusal değil.
svr_model.fit(x, y)

# 3. Random Forest Regresyon
rf_model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
rf_model.fit(x, y)

# Grid oluşturma (Random Forest için gerekliydi, burada hepsi için kullanıyoruz)
x_plot = np.linspace(1, 6, 200).reshape(-1, 1)

# Tahminleri hesaplama
y_lr = lr_model.predict(x_plot)
y_svr = svr_model.predict(x_plot)
y_rf = rf_model.predict(x_plot)

# Grafik çizimi
plt.scatter(x, y, color='red', label='Veri Noktaları')
plt.plot(x_plot, y_lr, color='blue', label='Linear Regresyon')
plt.plot(x_plot, y_svr, color='green', label='Support Vector Regresyon')
plt.plot(x_plot, y_rf, color='orange', label='Random Forest Regresyon')
plt.title('Farklı Regresyon Modelleri Karşılaştırması')
plt.xlabel('X Değeri')
plt.ylabel('Tahmin Edilen Y Değeri')
plt.legend()
plt.show()
"""