import os
import sys
import json
import time
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

# --- YAPILANDIRMA ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, 'ocak_siniflandirici.keras')
CLASS_INDICES_PATH = os.path.join(PROJECT_ROOT, 'class_indices.json')
SAMPLING_RATE = 22050
DURATION = 3  # Saniye cinsinden her analiz süresi
THRESHOLD = 0.005  # Sessizlik eşiği

LABEL_MAP = {
    'gurultu': {'emoji': '🔇', 'name': 'GÜRÜLTÜ', 'description': 'Ocakta bir şey pişmiyor.'},
    'kaynama': {'emoji': '🫧', 'name': 'KAYNIYOR', 'description': 'Yemek kaynıyor!'},
    'pisme':   {'emoji': '🍳', 'name': 'PİŞİYOR', 'description': 'Yemek pişiyor!'},
}

from test_cnn import test_single_audio

# Terminalde Türkçe karakter desteği
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# 1. Modeli ve Etiketleri Yükle
print("🚀 Sistem başlatılıyor...")
if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_INDICES_PATH):
    print(f"❌ Hata: Model ({MODEL_PATH}) veya etiket dosyası ({CLASS_INDICES_PATH}) bulunamadı!")
    sys.exit(1)

print("📦 Model yükleniyor (bu biraz zaman alabilir)...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, 'r', encoding='utf-8') as f:
    index_to_label = json.load(f)

print("✅ Model ve etiketler hazır!")
print(f"📋 Sınıflar: {list(index_to_label.values())}")
print("-" * 50)

def predict_audio(audio_data):
    """Gelen ham ses verisini analiz eder."""
    # RMS (Ses Seviyesi) Kontrolü
    rms = np.sqrt(np.mean(audio_data**2))
    
    if rms < THRESHOLD:
        return "SESSİZ", 0.0, rms, None

    # test_cnn.py üzerinden tahmin (dosya kullanmadan)
    best_label, probs = test_single_audio(
        file_path=None, 
        ext_model=model,
        ext_label_map=index_to_label,
        show_output=False,
        y=audio_data,
        sr=SAMPLING_RATE
    )
    
    best_idx = np.argmax(probs)
    confidence = probs[best_idx]
    
    return best_label, confidence, rms, probs

def start_listening():
    print(f"\n🎧 Mikrofon açıldı. {DURATION} saniyelik aralıklarla dinleniyor...")
    print("Durdurmak için Ctrl+C tuşlarına basın.\n")
    
    try:
        while True:
            # Kayıt al
            print("🎤 Dinleniyor...", end="\r")
            recording = sd.rec(int(DURATION * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1)
            sd.wait()  # Kayıt bitene kadar bekle
            
            # Analiz et
            label, conf, rms, probs = predict_audio(recording.flatten())
            
            # Sonucu yazdır
            print(" " * 40, end="\r") # Satırı temizle
            
            if label == "SESSİZ":
                print(f"💤 [SESSİZ] (RMS: {rms:.5f}) - Analiz edilmedi.")
            else:
                info = LABEL_MAP[label]
                color_code = "\033[93m" if label == 'pisme' else "\033[92m" if label == 'kaynama' else "\033[90m"
                reset_code = "\033[0m"
                
                print(f"{color_code}{info['emoji']}  {info['name']:8s} | Güven: %{conf*100:.1f} | RMS: {rms:.5f}{reset_code}")
                # print(f"   Olasılıklar: {probs}") # Detay görmek istersen bunu açabilirsin
            
    except KeyboardInterrupt:
        print("\n\n👋 Canlı dinleme durduruldu.")
    except Exception as e:
        print(f"\n❌ Hata: {e}")

if __name__ == "__main__":
    start_listening()
