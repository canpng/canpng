# PYTHON KÜTÜPHANELERİ VE LİNEER REGRESYON PDF'indeki Alıştırma Sorusunun Çözümü

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_excel("calisma1.xlsx")

x = df[["Kesme Hızı", "İlerleme", "Derinlik", "Sıcaklık"]] # Özellikler
y = df[["Aşınma"]] # Hedef değişken

model = LinearRegression() 
model.fit(x, y)  # Modeli eğitme

print("Katsayılar (w):", model.coef_) # Her bir özelliğin katsayısı
print("Bias (b):", model.intercept_) # Sabit terim

skor = model.score(x, y) # Modelin R2 skoru
print("Test R2 skoru:", skor) 

yeni_takim = [[145, 0.19, 1.6, 390]]
tahmin = model.predict(yeni_takim)
print("Yeni takım için tahmin edilen aşınma değeri:", tahmin)