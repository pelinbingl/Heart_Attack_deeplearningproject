import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
import warnings
warnings.filterwarnings("ignore")

color = sns.color_palette("Set2")

# === VERİYİ YÜKLEME ===
df = pd.read_csv("C:\\Users\\Pelin\\OneDrive\\Desktop\\Heart_attack\\heart_attack_dataset_no_treatment.csv")

# === RİSK PUANI HESAPLAMA ===
def calculate_risk_score(row):
    risk_factors = 0
    if row["Age"] > 60:
        risk_factors += 1
    if row["Blood Pressure (mmHg)"] > 140:
        risk_factors += 1
    if row["Cholesterol (mg/dL)"] > 240:
        risk_factors += 1
    if row["Has Diabetes"] == "Yes":
        risk_factors += 1
    if row["Smoking Status"] == "Current":
        risk_factors += 1
    if row["Chest Pain Type"] == "Type 1":
        risk_factors += 1
    return risk_factors

df["risk_score"] = df.apply(calculate_risk_score, axis=1)

def assign_risk_level(score):
    if score == 0:
        return "düşük"
    elif score == 1:
        return "orta"
    else:
        return "yüksek"

df["risk_level"] = df["risk_score"].apply(assign_risk_level)

# === VERİ ÖN İŞLEME ===
X = df.drop(['risk_level', 'risk_score'], axis=1)
y = df['risk_level']
X_encoded = pd.get_dummies(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# === MODELLERİ EĞİT ===
for name, model in models.items():
    model.fit(X_scaled, y)

# === MODEL DOĞRULUK HESAPLAMASI ===
accuracy = []
precision = []
recall = []
f1 = []

# 5-fold Cross Validation ekleme
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    acc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    accuracy.append(acc_scores.mean())

    # Precision, recall ve F1 skoru hesaplama
    precision_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='precision_weighted')
    recall_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='recall_weighted')
    f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_weighted')

    precision.append(precision_scores.mean())
    recall.append(recall_scores.mean())
    f1.append(f1_scores.mean())

# === HASTA/SAĞLAM PASTA GRAFİĞİ ===
hasta = df[df['risk_level'] != 'düşük'].shape[0]
saglam = df[df['risk_level'] == 'düşük'].shape[0]
labels = ['Hasta Olan Kişiler', 'Hasta Olmayan Kişiler']
sizes = [hasta, saglam]
colors = ['#5f5f5f', '#5ba5bb']
plt.figure(figsize=(8,8))
plt.pie(sizes, labels=labels, autopct='%1.2f%%', colors=colors, startangle=90)
plt.axis('equal')
plt.savefig('hasta_saglam_pasta.png', transparent=True)
plt.show()

# === YAŞA GÖRE RİSK BOXPLOT ===
plt.figure(figsize=(14,7))
sns.boxplot(x='risk_level', y='Age', data=df, palette=color)
plt.title("Risk Seviyelerine Göre Yaş Dağılımı")
plt.savefig('yas_risk.png', transparent=True)
plt.show()

# === CİNSİYETE GÖRE RİSK GRAFİĞİ ===
plt.figure(figsize=(14,6))
sns.countplot(x='Gender', hue='risk_level', data=df, palette=color)
plt.title("Cinsiyete Göre Risk Dağılımı")
plt.savefig('cinsiyet_risk.png', transparent=True)
plt.show()

# === GÖĞÜS AĞRISI TİPİNE GÖRE RİSK GRAFİĞİ ===
plt.figure(figsize=(15,6))
sns.countplot(x='Chest Pain Type', hue='risk_level', data=df, palette=color)
plt.title('Göğüs Ağrısı Tipine Göre Risk Seviyesi')
plt.savefig('gogus_agrisi_risk.png', transparent=True)
plt.show()

# === KULLANICI GİRİŞİ ===
print("\n--- KULLANICI GİRİŞİ ---")

print("Cinsiyet Seçin:\n0 - Male\n1 - Female")
gender_input = input("Seçiminiz: ")
gender = "Male" if gender_input == "0" else "Female"

age = int(input("Yaş: "))
bp = float(input("Tansiyon (mmHg): "))
chol = float(input("Kolesterol (mg/dL): "))

print("Diyabet Durumu:\n0 - No\n1 - Yes")
diabetes_input = input("Seçiminiz: ")
diabetes = "No" if diabetes_input == "0" else "Yes"

print("Sigara Durumu:\n0 - Never\n1 - Former\n2 - Current")
smoke_input = input("Seçiminiz: ")
smoke_options = ["Never", "Former", "Current"]
smoke = smoke_options[int(smoke_input)]

print("Göğüs Ağrısı Tipi:")
print("0 - Type 1 (Typical Angina)")
print("1 - Type 2 (Atypical Angina)")
print("2 - Type 3 (Non-anginal Pain)")
print("3 - Type 4 (Asymptomatic)")
cp_input = input("Seçiminiz: ")
cp_type = f"Type {int(cp_input) + 1}"

# === KULLANICI VERİSİ HAZIRLA ===
user_data = pd.DataFrame({
    'Gender': [gender],
    'Age': [age],
    'Blood Pressure (mmHg)': [bp],
    'Cholesterol (mg/dL)': [chol],
    'Has Diabetes': [diabetes],
    'Smoking Status': [smoke],
    'Chest Pain Type': [cp_type]
})

user_score = calculate_risk_score(user_data.iloc[0])
true_class = assign_risk_level(user_score)

user_encoded = pd.get_dummies(user_data)
for col in X_encoded.columns:
    if col not in user_encoded.columns:
        user_encoded[col] = 0
user_encoded = user_encoded[X_encoded.columns]
user_scaled = scaler.transform(user_encoded)

# === MODEL TAHMİNİ ve METRİKLER ===
print(f"\n--- Gerçek Risk Seviyesi (etiket): {true_class} ---\n")

for name, model in models.items():
    pred = model.predict(user_scaled)[0]
    precision = precision_score([true_class], [pred], average='weighted', zero_division=0)
    recall = recall_score([true_class], [pred], average='weighted', zero_division=0)
    f1 = f1_score([true_class], [pred], average='weighted', zero_division=0)

    print(f"{name:20s} | Tahmin: {pred:6s} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}")

# === KULLANICI OLASILIK GRAFİĞİ ===
chosen_model = RandomForestClassifier()
chosen_model.fit(X_scaled, y)
probs = chosen_model.predict_proba(user_scaled)[0]
class_labels = chosen_model.classes_

plt.figure(figsize=(10,6))
sns.barplot(x=class_labels, y=probs*100, palette="coolwarm")
plt.title("Yeni Kullanıcının Her Sınıfa Ait Olma Olasılığı (%)")
plt.ylabel("Olasılık (%)")
plt.xlabel("Risk Seviyesi")
for i, p in enumerate(probs*100):
    plt.text(i, p + 1, f"{p:.2f}%", ha='center')
plt.savefig('kullanici_risk_olasiligi.png', transparent=True)
plt.show()

# === MODEL KARŞILAŞTIRMA GRAFİĞİ ===
methods = list(models.keys())
plt.figure(figsize=(14,6))
sns.barplot(x=methods, y=[a*100 for a in accuracy], palette="deep")
plt.ylabel("Başarı %")
plt.xlabel("Algoritmalar")
plt.title("Algoritmaların Başarı Karşılaştırması (Accuracy)")

for i in range(len(methods)):
    plt.text(i - 0.15,
             accuracy[i]*100 + 0.5,
             "{:.2f}%".format(accuracy[i]*100),
             horizontalalignment='left',
             size='large',
             color="black")

plt.savefig('karsilastirma.png', transparent=True)
plt.show()

# === MODEL BAŞARILARINI KONSOLA YAZDIR ===
print("\n--- Model Başarı Oranları (Accuracy %) ---")
for i in range(len(methods)):
    print(f"{methods[i]}: {accuracy[i]*100:.2f}%")
