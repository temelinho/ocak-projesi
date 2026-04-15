import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Ayarlar
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
NUM_CLASSES = 3  # kaynama, gurultu, pisme

# Resimleri oku ve zenginleştir (Data Augmentation)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory('./Spectrograms', target_size=IMG_SIZE,
                                        batch_size=BATCH_SIZE, class_mode='categorical', subset='training')

val_gen = datagen.flow_from_directory('./Spectrograms', target_size=IMG_SIZE,
                                      batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

# Sınıf haritasını göster ve kaydet
print(f"\n📋 Sınıf Haritası: {train_gen.class_indices}")
class_indices = train_gen.class_indices
# Ters çevir: index -> label
index_to_label = {v: k for k, v in class_indices.items()}
with open('class_indices.json', 'w', encoding='utf-8') as f:
    json.dump(index_to_label, f, ensure_ascii=False, indent=2)
print(f"📁 Sınıf haritası 'class_indices.json' olarak kaydedildi.\n")

# CNN Mimarisi (3 sınıflı)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')  # 3 sınıf için softmax
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Eğitimi başlat
print("\n🚀 Eğitim başlıyor...")
history = model.fit(train_gen, epochs=20, validation_data=val_gen)

model.save('ocak_siniflandirici.keras')
print("\n📦 Model 'ocak_siniflandirici.keras' olarak kaydedildi!")

# Eğitim sonuçlarını göster
print(f"\n📊 Son Epoch Sonuçları:")
print(f"   Eğitim Accuracy : {history.history['accuracy'][-1]:.4f}")
print(f"   Validasyon Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"   Eğitim Loss     : {history.history['loss'][-1]:.4f}")
print(f"   Validasyon Loss  : {history.history['val_loss'][-1]:.4f}")