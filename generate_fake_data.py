# generate_fake_data.py
import numpy as np
from supabase import create_client
import time

# ⚠️ KENDİ BİLGİLERİNİ YAZ
SUPABASE_URL = "https://flnjitprqlxytbcaoptc.supabase.co"
SUPABASE_KEY = "sb_publishable_5_seWugPhmNDYtGO24NLFQ_ndcG19aL"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("🎲 500 adet test verisi üretiliyor...")

for i in range(500):
    data = {
        "sicaklik": round(np.random.uniform(18.0, 32.0), 1),
        "salt": round(np.random.uniform(1500.0, 3000.0), 2),
        "sensor_id": "esp-t1"
    }
    
    try:
        supabase.table('sensor_data').insert(data).execute()
        
        if (i + 1) % 50 == 0:
            print(f"✅ {i + 1}/500 veri eklendi...")
    except Exception as e:
        print(f"❌ Hata: {e}")
        break
    
    time.sleep(0.01)  # Çok hızlı gönderme

print("🎉 Tamamlandı!")
