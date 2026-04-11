import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# 1. Ayarlar (Dosya yollarını 'r' kullanarak veya '/' kullanarak düzeltiyoruz)
MODEL_WEIGHTS_PATH = 'ocak_siniflandirici.h5'
JSON_PATH = 'class_indices.json'
# Windows dosya yolu hatasını önlemek için 'r' ekledik:
TEST_FILE = r'C:\Users\temel\PycharmProjects\PythonProject1\Dataset\test4.wav'
TEMP_IMG = 'temp_inference.png'

LABEL_MAP = {
    'gurultu': ('🔇', 'GÜRÜLTÜ', 'Ocakta bir şey pişmiyor.'),
    'kaynama': ('🫧', 'KAYNIYOR', 'Yemek kaynıyor!'),
    'pisme': ('🍳', 'PİŞİYOR', 'Yemek pişiyor!')
}


# 2. Model Mimarini Tanımla (Eğitimdeki mimarinle aynı olmalı)
def build_my_model(num_classes):
    model = models.Sequential([
        # 1. Conv Katmanı
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),

        # 2. Conv Katmanı
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # 3. Conv Katmanı
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        # 4. Dense Katmanı
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),  # Eğitimde kullandığın Dropout

        # 5. Dense Katmanı
        layers.Dense(64, activation='relu'),

        # 6. Çıkış Katmanı (Softmax)
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def run_test():
    # Sınıf indekslerini yükle
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        index_to_label = json.load(f)

    num_classes = len(index_to_label)

    # Modeli oluştur ve SADECE ağırlıkları yükle
    model = build_my_model(num_classes)
    try:
        model.load_weights(MODEL_WEIGHTS_PATH)
        print("✅ Model ağırlıkları başarıyla yüklendi.")
    except Exception as e:
        print(f"❌ Ağırlıklar yüklenirken hata oluştu: {e}")
        return

    # 3. Ses İşleme
    if not os.path.exists(TEST_FILE):
        print(f"❌ Ses dosyası bulunamadı: {TEST_FILE}")
        return

    y, sr = librosa.load(TEST_FILE, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_db)
    plt.axis('off')
    plt.savefig(TEMP_IMG, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 4. Tahmin
    img = image.load_img(TEMP_IMG, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array, verbose=0)[0]

    if os.path.exists(TEMP_IMG):
        os.remove(TEMP_IMG)

    # 5. Sonuç Gösterimi
    best_idx = np.argmax(predictions)
    best_label_key = index_to_label[str(best_idx)]
    emoji, durum, aciklama = LABEL_MAP[best_label_key]

    print("\n" + "━" * 40)
    print(f"  {emoji} ANALİZ SONUCU: {durum}")
    print("━" * 40)

    for i, prob in enumerate(predictions):
        label_key = index_to_label[str(i)]
        e, name, _ = LABEL_MAP[label_key]
        print(f"{e} {name:8s}: {'█' * int(prob * 20):<20} %{prob * 100:.1f}")

    print("━" * 40)
    print(f"💡 Karar: {aciklama}")
    print("━" * 40)


if __name__ == '__main__':
    run_test()