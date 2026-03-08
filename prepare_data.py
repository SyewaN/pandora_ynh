# prepare_data.py
import pandas as pd
from supabase import create_client, Client
import numpy as np

# Supabase bağlantısı
SUPABASE_URL = "https://flnjitprqlxytbcaoptc.supabase.co"
SUPABASE_KEY = "sb_publishable_5_seWugPhmNDYtGO24NLFQ_ndcG19aL"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Verileri çek (DÜZELTİLMİŞ: created_at, sicaklik, salt)
response = supabase.table('sensor_data')\
    .select('created_at, sicaklik, salt, sensor_id')\
    .order('created_at', desc=False)\
    .execute()

df = pd.DataFrame(response.data)
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.sort_values('created_at')

# Null değerleri temizle
df = df.dropna(subset=['sicaklik', 'salt'])

print(f"📊 Toplam veri sayısı: {len(df)}")
print(f"📅 İlk veri: {df['created_at'].min()}")
print(f"📅 Son veri: {df['created_at'].max()}")
print("\n📈 Özet istatistikler:")
print(df[['sicaklik', 'salt']].describe())

# CSV'ye kaydet
df.to_csv('sensor_data.csv', index=False)
print("\n✅ Veriler sensor_data.csv dosyasına kaydedildi")
