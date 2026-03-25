import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ayarlar
IMG_SIZE = (128, 128)
BATCH_SIZE = 8

# Resimleri oku ve zenginleştir (Data Augmentation)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory('./Spectrograms', target_size=IMG_SIZE,
                                        batch_size=BATCH_SIZE, class_mode='binary', subset='training')

val_gen = datagen.flow_from_directory('./Spectrograms', target_size=IMG_SIZE,
                                      batch_size=BATCH_SIZE, class_mode='binary', subset='validation')

# CNN Mimarisi
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid') # 0 veya 1 tahmini için
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Eğitimi başlat
print("🚀 Eğitim başlıyor...")
model.fit(train_gen, epochs=15, validation_data=val_gen)

model.save('kaynama_cnn.h5')
print("📦 Model 'kaynama_cnn.h5' olarak kaydedildi!")