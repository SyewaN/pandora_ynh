# train_isolation_forest.py
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Veriyi yükle
df = pd.read_csv('sensor_data.csv')
features = ['sicaklik', 'salt']  # DÜZELTİLMİŞ
X = df[features].values

print(f"📊 Toplam veri: {len(X)}")

# Normalizasyon
scaler_iso = StandardScaler()
X_scaled = scaler_iso.fit_transform(X)

# Model eğit
iso_model = IsolationForest(
    contamination=0.05,  # %5 anomali bekleniyor
    random_state=42,
    n_estimators=100
)
iso_model.fit(X_scaled)

# Test
predictions = iso_model.predict(X_scaled)
n_outliers = (predictions == -1).sum()
print(f"🚨 Tespit edilen anomali: {n_outliers} ({n_outliers/len(X)*100:.2f}%)")

# Anomali örnekleri göster
df['is_anomaly'] = predictions == -1
anomalies = df[df['is_anomaly']][['created_at', 'sicaklik', 'salt']].head(10)
print("\n📋 İlk 10 anomali:")
print(anomalies)

# Kaydet
joblib.dump(iso_model, 'isolation_forest.pkl')
joblib.dump(scaler_iso, 'scaler_isolation.pkl')
print("\n✅ Isolation Forest modeli kaydedildi")
