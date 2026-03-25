import tensorflow as tf
from tensorflow.keras.preprocessing import image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Modeli yükle
model = tf.keras.models.load_model('kaynama_cnn.h5')


def test_single_audio(file_path):
    temp_image = 'temp_test.png'

    # 1. Sesi Spektrograma Çevir (EĞİTİMDEKİYLE AYNI AYARLAR)
    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Resim boyutu eğitimdeki ImageDataGenerator çıktısıyla aynı olmalı
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_db)
    plt.axis('off')
    plt.savefig(temp_image, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 2. Resmi Yükle ve İşle (Keras Yöntemi)
    img = image.load_img(temp_image, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Batch boyutu ekle (1, 128, 128, 3)
    img_array /= 255.0  # Normalize et (Eğitimde rescale=1./255 yapmıştık!)

    # 3. Tahmin Yap
    prediction = model.predict(img_array)[0][0]

    # Geçici dosyayı temizle
    if os.listdir().count(temp_image) > 0:
        os.remove(temp_image)

    print("-" * 30)
    print(f"ANALİZ SONUCU:")
    # predict_proba yerine direkt sigmoid çıktısını (prediction) kullanıyoruz
    return f"🔥 Kaynama İhtimali: %{round(float(prediction) * 100, 4)}"


# TEST
ses = 'Dataset/test.wav'
print(test_single_audio(ses))