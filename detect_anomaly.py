# detect_anomaly.py
import numpy as np
import joblib
from supabase import create_client

# ⚠️ BURAYA DOĞRU KEY'İ YAPIŞTIR
SUPABASE_URL = "https://flnjitprqlxytbcaoptc.supabase.co"
SUPABASE_KEY = "sb_publishable_5_seWugPhmNDYtGO24NLFQ_ndcG19aL"  # Settings → API → anon public

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Model yükle
print("📦 Isolation Forest yükleniyor...")
iso_model = joblib.load('isolation_forest.pkl')
scaler_iso = joblib.load('scaler_isolation.pkl')

# Son veriyi çek
print("📡 Son veri çekiliyor...")
response = supabase.table('sensor_data')\
    .select('id, sicaklik, salt, sensor_id, created_at')\
    .order('created_at', desc=True)\
    .limit(1)\
    .execute()

if not response.data:
    print("❌ Veri bulunamadı!")
    exit()

data = response.data[0]
X = np.array([[float(data['sicaklik']), float(data['salt'])]])

# Normalize et
X_scaled = scaler_iso.transform(X)

# Tahmin
prediction = iso_model.predict(X_scaled)

print(f"\n📊 KONTROL EDİLEN VERİ:")
print(f"  ID: {data['id']}")
print(f"  Tarih: {data['created_at']}")
print(f"  Sıcaklık: {data['sicaklik']}°C")
print(f"  TDS: {data['salt']} ppm")
print(f"  Sensör: {data['sensor_id']}")

if prediction[0] == -1:
    print("\n🚨 ANOMALİ TESPİT EDİLDİ!")
    
    # Anomali kaydet
    try:
        supabase.table('anomalies').insert({
            'sensor_data_id': data['id'],
            'anomaly_type': 'isolation_forest',
            'severity': 'high',
            'message': '🤖 AI Anomali Tespiti: Olağandışı veri paterni',
            'details': {
                'sicaklik': float(data['sicaklik']),
                'salt': float(data['salt']),
                'sensor_id': data['sensor_id']
            }
        }).execute()
        
        print("✅ Anomali Supabase'e kaydedildi!")
    except Exception as e:
        print(f"❌ Kayıt hatası: {e}")
else:
    print("\n✅ Normal veri - Anomali yok")


