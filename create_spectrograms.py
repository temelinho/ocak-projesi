import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def create_spectrogram(audio_file, image_file):
    try:
        y, sr = librosa.load(audio_file, sr=22050)
        # Mel-Spektrogram hesapla
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_db, sr=sr)
        plt.axis('off')
        plt.savefig(image_file, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Hata: {audio_file} işlenemedi. {e}")


# Klasörleri tara ve dönüştür
base_audio = './Dataset'
base_spectro = './Spectrograms'

for cat in ['kaynama', 'gurultu']:
    audio_path = os.path.join(base_audio, cat)
    dest_path = os.path.join(base_spectro, cat)
    os.makedirs(dest_path, exist_ok=True)

    files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
    print(f"⌛ {cat} için {len(files)} dosya dönüştürülüyor...")
    for f in files:
        create_spectrogram(os.path.join(audio_path, f), os.path.join(dest_path, f.replace('.wav', '.png')))

print("✅ Spektrogramlar hazır!")