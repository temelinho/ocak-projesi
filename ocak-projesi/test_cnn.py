import tensorflow as tf
from tensorflow.keras.preprocessing import image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import sys

# Terminalde Türkçe karakter sorunu olmaması için
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Yeni formatta modeli yükle
MODEL_PATH = 'ocak_siniflandirici.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Sınıf haritasını yükle
with open('class_indices.json', 'r', encoding='utf-8') as f:
    index_to_label = json.load(f)

# Türkçe etiketler ve açıklamalar
LABEL_MAP = {
    'gurultu': ('🔇', 'GÜRÜLTÜ', 'Ocakta bir şey pişmiyor.'),
    'kaynama': ('🫧', 'KAYNIYOR', 'Yemek kaynıyor!'),
    'pisme': ('🍳', 'PİŞİYOR', 'Yemek pişiyor!')
}


def test_single_audio(file_path):
    temp_image = 'temp_test.png'

    # Sesi Spektrograma Çevir
    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_db)
    plt.axis('off')
    plt.savefig(temp_image, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Resmi İşle
    img = image.load_img(temp_image, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Tahmin Yap
    predictions = model.predict(img_array, verbose=0)[0]

    if os.path.exists(temp_image):
        os.remove(temp_image)

    # Sonuçları Göster
    best_idx = np.argmax(predictions)
    best_label = index_to_label[str(best_idx)]
    emoji, durum, aciklama = LABEL_MAP[best_label]

    print("\n" + "━" * 40)
    print(f"  {emoji} ANALİZ: {file_path.split('/')[-1]}")
    print("━" * 40)

    for i in range(len(predictions)):
        label = index_to_label[str(i)]
        label_emoji, label_name, _ = LABEL_MAP[label]
        bar = "█" * int(predictions[i] * 20)
        print(f"  {label_emoji} {label_name:8s} : {bar:<20} %{predictions[i] * 100:.1f}")

    print("━" * 40)
    print(f"  ✅ SONUÇ: {aciklama}")
    print("━" * 40)


# --- TOPLU TEST KISMI ---
if __name__ == '__main__':
    # Test etmek istediğin dosyaları buraya ekle
    test_files = [
        'C:/Users/Fey/Downloads/1.wav',
        'C:/Users/Fey/Downloads/2.wav',
        'C:/Users/Fey/Downloads/3.wav',
        'C:/Users/Fey/Downloads/4.wav',
        'C:/Users/Fey/Downloads/5.wav',
        # İstersen yeni pişme seslerini de ekleyebilirsin:

    ]

    print("\n" + "=" * 40)
    print("  🔬 TOPLU TEST BAŞLIYOR")
    print("=" * 40)

    for f in test_files:
        # Windows/Linux yol uyumu için / ve \ karakterlerini düzeltelim
        f_path = f.replace('/', os.sep).replace('\\', os.sep)

        if os.path.exists(f_path):
            test_single_audio(f_path)
        else:
            print(f"\n❌ Dosya bulunamadı: {f_path}")