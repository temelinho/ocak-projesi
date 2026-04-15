import tensorflow as tf
from tensorflow.keras.preprocessing import image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import tempfile

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Türkçe etiketler
LABEL_MAP = {
    'gurultu': ('🔇', 'GÜRÜLTÜ', 'Ocakta bir şey pişmiyor, sadece gürültü.'),
    'kaynama': ('🫧', 'KAYNIYOR', 'Yemek kaynıyor!'),
    'pisme':   ('🍳', 'PİŞİYOR', 'Yemek pişiyor!')
}

def test_single_audio(file_path, ext_model=None, ext_label_map=None, show_output=True, y=None, sr=None):
    temp_image = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name

    # Model arguman olarak gelmezse, mevcut dizinden yüklemeyi dene
    current_model = ext_model
    current_index_to_label = ext_label_map

    if current_model is None or current_index_to_label is None:
        if os.path.exists('class_indices.json'):
            with open('class_indices.json', 'r', encoding='utf-8') as f:
                current_index_to_label = json.load(f)
        else:
            current_index_to_label = {"0": "gurultu", "1": "kaynama", "2": "pisme"}
            
        current_model = tf.keras.models.load_model('ocak_siniflandirici.keras')

    # 1. Sesi Spektrograma Çevir (EĞİTİMDEKİYLE AYNI AYARLAR)
    if y is None or sr is None:
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
    predictions = current_model.predict(img_array, verbose=0)[0]

    # Geçici dosyayı temizle
    if os.path.exists(temp_image):
        try:
            os.remove(temp_image)
        except Exception:
            pass

    # 4. Sonuçları göster
    best_idx = np.argmax(predictions)
    best_label = current_index_to_label[str(best_idx)]
    emoji, durum, aciklama = LABEL_MAP[best_label]

    if show_output:
        print("\n" + "━" * 36)
        print(f"  {emoji} OCAK ANALİZ SONUCU")
        print("━" * 36)

        for i in range(len(predictions)):
            label = current_index_to_label[str(i)]
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
    import argparse
    parser = argparse.ArgumentParser(description="Ocak Ses Test - Tek Ses Dosyası Sınıflandırma")
    parser.add_argument("audio_path", nargs="?", default="Dataset/test.wav", help="Test edilecek ses dosyası yolu (varsayılan: Dataset/test.wav)")
    args = parser.parse_args()
    
    file_path = args.audio_path

    print("=" * 40)
    print("  🔬 TEK SES TESTİ BAŞLIYOR")
    print("=" * 40)

    if os.path.exists(file_path):
        print(f"\n📂 Dosya: {file_path}")
        test_single_audio(file_path)
    else:
        print(f"\n❌ Dosya bulunamadı: {file_path}")