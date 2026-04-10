import tensorflow as tf
from tensorflow.keras.preprocessing import image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import json

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Modeli yükle
model = tf.keras.models.load_model('ocak_siniflandirici.h5')

# Sınıf haritasını yükle
with open('class_indices.json', 'r', encoding='utf-8') as f:
    index_to_label = json.load(f)

# Türkçe etiketler
LABEL_MAP = {
    'gurultu': ('🔇', 'GÜRÜLTÜ', 'Ocakta bir şey pişmiyor, sadece gürültü.'),
    'kaynama': ('🫧', 'KAYNIYOR', 'Yemek kaynıyor!'),
    'pisme':   ('🍳', 'PİŞİYOR', 'Yemek pişiyor!')
}


def test_single_audio(file_path):
    temp_image = 'temp_test.png'

    # 1. Sesi Spektrograma Çevir (EĞİTİMDEKİYLE AYNI AYARLAR)
    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_db)
    plt.axis('off')
    plt.savefig(temp_image, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 2. Resmi Yükle ve İşle
    img = image.load_img(temp_image, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # 3. Tahmin Yap (3 sınıf olasılığı)
    predictions = model.predict(img_array, verbose=0)[0]

    # Geçici dosyayı temizle
    if os.path.exists(temp_image):
        os.remove(temp_image)

    # 4. Sonuçları göster
    best_idx = np.argmax(predictions)
    best_label = index_to_label[str(best_idx)]
    emoji, durum, aciklama = LABEL_MAP[best_label]

    print("\n" + "━" * 36)
    print(f"  {emoji} OCAK ANALİZ SONUCU")
    print("━" * 36)

    for i in range(len(predictions)):
        label = index_to_label[str(i)]
        label_emoji, label_name, _ = LABEL_MAP[label]
        bar_len = int(predictions[i] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {label_emoji} {label_name:8s} : {bar} %{predictions[i]*100:.1f}")

    print("━" * 36)

    if predictions[best_idx] < 0.50:
        print(f"  ⚠️  Sonuç belirsiz, tekrar deneyin.")
    else:
        print(f"  ✅ SONUÇ: {aciklama}")

    print("━" * 36)

    return best_label, predictions


# TEST
if __name__ == '__main__':
    test_files = [
        'Dataset/test.wav',
        'Dataset/test2.wav',
        'Dataset/test3.wav',
        'Dataset/test4.wav',
        'Dataset/son.wav',
    ]

    print("=" * 40)
    print("  🔬 TOPLU TEST BAŞLIYOR")
    print("=" * 40)

    for f in test_files:
        if os.path.exists(f):
            print(f"\n📂 Dosya: {f}")
            test_single_audio(f)
        else:
            print(f"\n❌ Dosya bulunamadı: {f}")