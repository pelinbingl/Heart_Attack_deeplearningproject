# 🫀 Kalp Krizi Risk Analizi ve Sınıflandırması

Bu proje, bireylerin temel sağlık verilerine dayanarak kalp krizi riskini **düşük**, **orta** veya **yüksek** seviyede tahmin edebilen bir makine öğrenmesi sisteminin geliştirilmesini amaçlamaktadır. Python ile geliştirilen bu uygulama, çeşitli sınıflandırma algoritmalarını karşılaştırmalı olarak test eder ve kullanıcıdan alınan anlık verilere göre kişiselleştirilmiş risk tahminleri üretir.

---

## 🎯 Projenin Amacı ve Hedefi

Bu proje kapsamında, kalp krizi riskinin önceden tahmin edilebilmesi için kullanıcıdan alınan temel sağlık verileri kullanılarak makine öğrenmesi tabanlı bir sınıflandırma sistemi geliştirilmiştir.  

Projenin temel hedefi:  
✔️ Yaş, tansiyon, kolesterol, diyabet, sigara kullanımı ve göğüs ağrısı gibi risk faktörlerine göre kişilerin kalp krizi riskini (“düşük”, “orta”, “yüksek”) tahmin eden bir model geliştirmek  
✔️ Çeşitli makine öğrenmesi algoritmalarını karşılaştırmalı olarak değerlendirmek  
✔️ Kullanıcı girdilerine göre anlık ve kişiselleştirilmiş tahminler üretmek  
✔️ Sonuçları grafiklerle görselleştirerek yorumlamak

---

## 🧰 Kullanılan Materyaller

### 📦 Yazılımlar ve Kütüphaneler
- Python 3.11
- `pandas` – Veri işleme
- `numpy` – Sayısal hesaplamalar
- `matplotlib`, `seaborn` – Görselleştirme
- `scikit-learn` – Makine öğrenmesi modelleri ve preprocessing
- `warnings` – Uyarı bastırma

### 💾 Veri Seti
- **Heart Attack Risk Factors Dataset**  
- Kaynak: [Kaggle](https://www.kaggle.com/datasets/waqi786/heart-attack-dataset)  
- İçerik:  
  - Demografik veriler (Cinsiyet, Yaş)  
  - Tıbbi ölçümler (Tansiyon, Kolesterol)  
  - Sağlık geçmişi (Diyabet, Sigara, Göğüs ağrısı tipi)  
  - Etiket: Risk Seviyesi

### 🖥 Donanım ve Geliştirme Ortamı
- VS Code (Python IDE)
- Windows/Linux bilgisayar

---

## ⚙️ Proje Aşamaları

### 1. Veri Ön İşleme
- Eksik veri temizliği
- Kategorik verilerin sayısallaştırılması
- Özellik ölçekleme (`StandardScaler`)
- Eşik değerlere göre risk seviyelerinin sınıflandırılması (`Labeling`)

---
## 📚 2. Model Eğitimi – 5 Katlı Cross-Validation

Bu projede model eğitimi, klasik train-test bölme yaklaşımı yerine **Stratified K-Fold Cross-Validation (k=5)** yöntemiyle gerçekleştirilmiştir.

### ⚙️ Stratified K-Fold Nedir?

Stratified K-Fold, veri setini K eşit parçaya bölerken, her parçada hedef değişkenin (etiket) orijinal sınıf dağılımını korur. Bu sayede hem eğitim hem de doğrulama aşamalarında model, her sınıftan yeterli miktarda veri ile eğitilmiş ve test edilmiş olur.

### ✅ Neden Bu Yöntem Tercih Edildi?

- 🔹 **Sınıf dengesini korur** – Özellikle azınlık sınıfın temsil edildiği medikal veri setlerinde avantaj sağlar.
- 🔹 **Overfitting riskini azaltır** – Modelin sadece belirli bir bölüme değil, tüm veriye genellenebilirliğini test eder.
- 🔹 **İstatistiksel olarak güvenilir skorlar sunar** – Her model 5 farklı eğitim-test kombinasyonunda test edildiği için başarı skorları daha sağlam olur.

### 📊 Kullanılan Başarı Metrikleri

Her model için aşağıdaki metriklerin 5 katlı çapraz doğrulama ortalamaları alınmıştır:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)

  
## 🧠 Kullanılan Makine Öğrenmesi Modelleri

Bu projede kalp krizi riski tahmini için çeşitli denetimli sınıflandırma algoritmaları kullanılmıştır. Her model 5 katlı çapraz doğrulama (cross-validation) ile test edilmiştir. Aşağıda bu modellerin çalışma prensipleri, avantajları ve dezavantajları detaylı şekilde açıklanmıştır:

---

### 🌳 Decision Tree (Karar Ağacı)

#### ✅ Avantajları:
- Yorumlaması ve görselleştirmesi kolaydır.
- Veri ölçeklemeye ihtiyaç duymaz.
- Hem kategorik hem de sayısal verilerle iyi çalışır.
- Aşırı karmaşık olmayan veri kümelerinde yüksek doğruluk verir.

#### ❌ Dezavantajları:
- Aşırı öğrenmeye (overfitting) meyillidir.
- Küçük değişikliklere çok hassastır, farklı ağaçlar oluşturabilir.

#### 📌 Bu projede:
Karar ağacı modeli %99.4 doğruluk ile en başarılı sonuçları vermiştir. Modelin öğrenme yapısı, bu tür sınıflandırma problemleri için oldukça uyumludur.

---

### 🌲 Random Forest

#### ✅ Avantajları:
- Birden fazla karar ağacından oluştuğu için overfitting riski düşer.
- Daha kararlı ve genellenebilir sonuçlar üretir.
- Eksik verilere karşı daha toleranslıdır.

#### ❌ Dezavantajları:
- Yorumlaması zordur (black-box model).
- Eğitimi ve tahmin süresi, tek bir karar ağacına göre daha uzundur.

#### 📌 Bu projede:
%96.3 doğruluk ile ikinci en başarılı modeldir. Veri çeşitliliği yüksek olduğu için birden fazla ağacın karar birleştirmesi yüksek başarı sağlamıştır.

---

### 📈 Logistic Regression

#### ✅ Avantajları:
- Basit ve hızlıdır.
- Doğrusal sınıflandırma problemlerinde iyi performans gösterir.
- Aşırı parametre ayarına gerek duymaz.

#### ❌ Dezavantajları:
- Karmaşık ilişkileri yakalayamaz.
- Doğrusal olmayan verilerde düşük performans gösterir.

#### 📌 Bu projede:
%89.5 doğruluk ile iyi bir baseline (referans) model olmuştur.

---

### 💠 Support Vector Machine (SVM)

#### ✅ Avantajları:
- Karışık sınırlara sahip verilerde yüksek performans gösterir.
- Margin (sınır) genişliğini maksimize ederek genelleme başarısını artırır.

#### ❌ Dezavantajları:
- Büyük veri kümelerinde yavaş çalışabilir.
- Parametre ayarı (kernel, C, gamma) zordur.

#### 📌 Bu projede:
%87.4 doğruluk elde edilmiştir. Özellikle iyi ayrışan sınıflarda stabil sonuçlar vermiştir.

---

### 👥 K-Nearest Neighbors (KNN)

#### ✅ Avantajları:
- Uygulaması çok kolaydır.
- Model eğitimi gerekmez (lazy learning).

#### ❌ Dezavantajları:
- Tahmin süresi yavaştır (özellikle büyük veri setlerinde).
- Aykırı değerlere duyarlıdır.
- Özelliklerin ölçeklenmesi çok önemlidir.

#### 📌 Bu projede:
%80.4 doğruluk elde edilmiştir. Özellikle doğru ölçekleme ve uygun K değeri seçimi ile geliştirilebilir.

---

### 🧮 Naive Bayes

#### ✅ Avantajları:
- Basit ve çok hızlıdır.
- Küçük veri setlerinde dahi etkili çalışabilir.
- Gereksiz özellikleri otomatik olarak azaltma eğilimindedir.

#### ❌ Dezavantajları:
- Özellikler arası bağımsızlık varsayımı genellikle gerçek dünyada sağlanmaz.
- Karmaşık ilişkileri modelleyemez.

#### 📌 Bu projede:
%64.9 doğruluk ile en düşük performansa sahiptir. Kalp krizi gibi çok faktörlü durumlarda özellik bağımsızlığı varsayımı zayıf kalmıştır.

---

## 🔍 Sonuç ve Model Seçimi

Bu proje özelinde:
- **Decision Tree** ve **Random Forest**, hem doğruluk hem de yorumlanabilirlik açısından öne çıkmıştır.
- **Naive Bayes**, düşük başarısı nedeniyle bu problem için uygun değildir.
- Diğer modeller (SVM, Logistic Regression, KNN) de faydalı bilgiler sunmuş, ancak genel başarıda geride kalmıştır.

Çapraz doğrulama sayesinde tüm modellerin başarısı daha güvenilir şekilde ölçülmüş ve karşılaştırılmıştır.


| Model              | Ortalama Doğruluk (5-Fold CV) |
|--------------------|-------------------------------|
| Decision Tree      | %99.40                        |
| Random Forest      | %96.30                        |
| Logistic Regression| %89.50                        |
| SVM                | %87.40                        |
| KNN                | %80.40                        |
| Naive Bayes        | %64.90                        |
| MLP                | %91.60                        |

---

### 3. Kullanıcıdan Alınan Verilerle Tahmin

Her model:
- Kullanıcıdan alınan veriyi preprocess eder  
- Risk grubunu tahmin eder  
- Sonuçları `precision`, `recall`, `F1-score` gibi metriklerle değerlendirir  
- Tahmin ve model başarısı görsellerle desteklenir

---

## 📊 Sonuçların Görselleştirilmesi

### Genel Veri Görselleştirmeleri
- Hasta / Hasta Olmayan Dağılımı
![image](https://github.com/user-attachments/assets/52c649fc-0287-4059-8045-6aabc27b18ab)

- Yaşa Göre Risk Dağılımı
![image](https://github.com/user-attachments/assets/c30c8d02-cad2-4f5e-94e9-a5541d56eccf)

- Cinsiyete Göre Risk Dağılımı
![image](https://github.com/user-attachments/assets/6c2243ca-62a2-4e34-b4cf-6005b02691d4)
  
- Göğüs Ağrısı Tipine Göre Risk Seviyesi
![image](https://github.com/user-attachments/assets/d1b7dfd4-54f4-4ff3-a5d9-329319aa085f)


### Kullanıcı Girdisine Dayalı Risk Analizi

#### 🔵 Düşük Risk Sınıfı Tahmini
- Kullanıcı verisi
![image](https://github.com/user-attachments/assets/898ae9f5-4c99-451c-adc2-b20686de8230)


- Model tahmini
![image](https://github.com/user-attachments/assets/c22af694-c366-4bfb-99b1-9708fdec3fb7)


- F1-Score ve tahmin grafiği  
![image](https://github.com/user-attachments/assets/34a55d71-63c4-4b03-a6a2-2bfb177c6773)



---

#### 🟡 Orta Risk Sınıfı Tahmini
- Kullanıcı verisi
![image](https://github.com/user-attachments/assets/2cddfb05-8db8-4dea-9111-18fccbec7889)


- Model tahmini
![image](https://github.com/user-attachments/assets/afd4f7ff-307b-485b-83a9-6bcfcd53dee5)



- F1-Score ve tahmin grafiği  
![image](https://github.com/user-attachments/assets/f07620b8-769e-4db7-9d0b-80620608d162)

---

#### 🔴 Yüksek Risk Sınıfı Tahmini
- - Kullanıcı verisi
![image](https://github.com/user-attachments/assets/ba1c436f-5149-4093-800b-8026afb03cf8)

- Model tahmini
![image](https://github.com/user-attachments/assets/e40c5de4-2fa6-4cab-b56e-156861bbfea0)

- F1-Score ve tahmin grafiği  
![image](https://github.com/user-attachments/assets/bf219daa-9470-4627-946b-80492ed87cba)




---

## 📈 Model Başarı Karşılaştırması

Model performansları doğruluk tablosu ve başarı grafiği ile görsel olarak karşılaştırılmıştır. En başarılı iki model olarak:

✅ **Decision Tree**  
✅ **Random Forest**

öne çıkmıştır. Naive Bayes ise düşük doğruluk nedeniyle uygun bulunmamıştır.

![image](https://github.com/user-attachments/assets/ba2dd3a3-7bd7-433c-9b6a-e8ea5a584e5f)

![image](https://github.com/user-attachments/assets/85958cd7-5181-4f53-9471-25d3b428c1f7)


---

## 🔬 Sonuçların Yorumlanması

- Cross-validation, tüm modellerin genel başarısını güvenilir şekilde ölçtü  
- Decision Tree ve Random Forest %95+ doğrulukla başarılı sonuçlar verdi  
- Kullanıcı bazlı tahminlerde de yüksek uyum gözlemlendi  
- Sistem, dijital sağlık uygulamalarına temel teşkil edebilecek niteliktedir

---

