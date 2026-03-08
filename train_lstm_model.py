# train_lstm_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt

# Veriyi yükle
df = pd.read_csv('sensor_data.csv')
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.sort_values('created_at')

# Özellikler (DÜZELTİLMİŞ: sadece sicaklik ve salt)
features = ['sicaklik', 'salt']
data = df[features].values

print(f"📊 Veri boyutu: {data.shape}")

# Normalizasyon
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Veri hazırlama (son 50 veriyle sonraki 1 veriyi tahmin et)
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 50
X, y = create_sequences(data_scaled, SEQ_LENGTH)

if len(X) < 100:
    print(f"❌ HATA: Yetersiz veri! En az {SEQ_LENGTH + 100} kayıt gerekli.")
    print(f"   Şu an: {len(df)} kayıt var.")
    exit()

# Train/Test split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")

# Model oluştur (DÜZELTİLMİŞ: 2 özellik için)
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(SEQ_LENGTH, 2)),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(2)  # 2 çıkış: sicaklik, salt
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Eğit
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Test
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"\n✅ Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}")

# Kaydet
model.save('lstm_model.h5')
joblib.dump(scaler, 'scaler.pkl')
print("\n✅ Model kaydedildi: lstm_model.h5")
print("✅ Scaler kaydedildi: scaler.pkl")

# Grafik
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Model Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.legend()
plt.title('Model MAE')
plt.tight_layout()
plt.savefig('lstm_training.png')
print("✅ Grafik kaydedildi: lstm_training.png")
