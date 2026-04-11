import os
import glob
import librosa
import soundfile as sf
import numpy as np

def augment_kaynama_sounds(input_folder, output_folder):
    """
    Kaynama seslerini frekans yapısını bozmadan hafif varyasyonlarla çoğaltır.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Klasördeki tüm .wav dosyalarını bul
    wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
    print(f"{len(wav_files)} adet orijinal kaynama sesi işleniyor...")

    for file_path in wav_files:
        try:
            # 1. Sesi yükle
            y, sr = librosa.load(file_path, sr=None)
            base_name = os.path.basename(file_path).replace(".wav", "")

            # --- VARYASYON 1: Hafif Pitch Shift (Tencere boyutu değişimi gibi) ---
            # Sesi sadece 1.2 yarım ton inceltiyoruz (Frekans karakteri korunur)
            y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=1.2)
            sf.write(os.path.join(output_folder, f"{base_name}_v1_pitchup.wav"), y_pitch_up, sr)

            # --- VARYASYON 2: Hafif Pitch Shift (Daha derin tencere gibi) ---
            # Sesi sadece 1.2 yarım ton kalınlaştırıyoruz
            y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1.2)
            sf.write(os.path.join(output_folder, f"{base_name}_v2_pitchdown.wav"), y_pitch_down, sr)

            # --- VARYASYON 3: Hafif Zaman Esnetme (Kaynama hızı) ---
            # Sesi %10 hızlandırıyoruz (Fokurdama sıklığı artar)
            y_fast = librosa.effects.time_stretch(y, rate=1.1)
            sf.write(os.path.join(output_folder, f"{base_name}_v3_fast.wav"), y_fast, sr)

            # --- VARYASYON 4: Çok Hafif Beyaz Gürültü (Mutfak ambiyansı) ---
            # Sese %0.5 oranında gürültü ekliyoruz (Modelin gürültüye direnci artar)
            noise = np.random.randn(len(y))
            y_noise = y + 0.005 * noise
            sf.write(os.path.join(output_folder, f"{base_name}_v4_noise.wav"), y_noise, sr)

            print(f"Tamamlandı: {base_name} (4 yeni varyasyon oluşturuldu)")

        except Exception as e:
            print(f"Hata ({file_path}): {e}")

if __name__ == "__main__":
    # KLASÖR YOLLARINI KENDİNE GÖRE DÜZENLE:
    orijinal_kaynama_yolu = 'Dataset\kaynama'
    yeni_artirilmis_yol = 'kaynama_sesleri_artirilmis_YENI'

    augment_kaynama_sounds(orijinal_kaynama_yolu, yeni_artirilmis_yol)
    print(f"\n✅ İşlem bitti! Yeni sesler '{yeni_artirilmis_yol}' klasöründe.")