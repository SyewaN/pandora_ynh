# predict_next_values.py
import numpy as np
import joblib
from tensorflow import keras
from supabase import create_client

# ⚠️ BURAYA DOĞRU KEY'İ YAPIŞTIR
SUPABASE_URL = "https://flnjitprqlxytbcaoptc.supabase.co"
SUPABASE_KEY = "sb_publishable_5_seWugPhmNDYtGO24NLFQ_ndcG19aL"  # Settings → API → anon public

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Model ve scaler'ı yükle
print("📦 Model yükleniyor...")
model = keras.models.load_model('lstm_model.h5', compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

scaler = joblib.load('scaler.pkl')

print("📡 Son 50 veriyi çekiyor...")
response = supabase.table('sensor_data')\
    .select('sicaklik, salt')\
    .order('created_at', desc=False)\
    .limit(50)\
    .execute()

# Veriyi hazırla
data = [[float(d['sicaklik']), float(d['salt'])] for d in response.data]
data = np.array(data)

if len(data) < 50:
    print(f"❌ Yetersiz veri! {len(data)}/50")
    exit()

print(f"✅ {len(data)} veri alındı")

# Normalize et
data_scaled = scaler.transform(data)
X = data_scaled.reshape(1, 50, 2)

# Tahmin yap
print("🔮 Tahmin yapılıyor...")
prediction_scaled = model.predict(X, verbose=0)

# Denormalize et
prediction = scaler.inverse_transform(prediction_scaled)

sicaklik_tahmin = float(prediction[0][0])
salt_tahmin = float(prediction[0][1])

print(f"\n📊 TAHMİN SONUÇLARI:")
print(f"  🌡️  Sonraki Sıcaklık: {sicaklik_tahmin:.1f}°C")
print(f"  💧 Sonraki TDS (Salt): {salt_tahmin:.1f} ppm")

# Son gerçek değerlerle karşılaştır
son_sicaklik = float(response.data[-1]['sicaklik'])
son_salt = float(response.data[-1]['salt'])

print(f"\n📋 KARŞILAŞTIRMA:")
print(f"  Son Ölçülen Sıcaklık: {son_sicaklik:.1f}°C → Tahmin: {sicaklik_tahmin:.1f}°C")
print(f"  Son Ölçülen TDS: {son_salt:.1f} ppm → Tahmin: {salt_tahmin:.1f} ppm")

# Supabase'e kaydet
print("\n💾 Tahmin Supabase'e kaydediliyor...")
try:
    supabase.table('predictions').insert({
        'predicted_sicaklik': sicaklik_tahmin,
        'predicted_salt': salt_tahmin,
        'confidence': 0.85
    }).execute()
    print("✅ Tahmin kaydedildi!")
except Exception as e:
    print(f"❌ Kayıt hatası: {e}")
