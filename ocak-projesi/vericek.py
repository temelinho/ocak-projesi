import os
import glob
from pydub import AudioSegment


def safe_segment_sounds(input_folder, output_folder, chunk_length_ms=3000):
    """
    Büyük dosyaları daha güvenli bir şekilde parçalara böler.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Klasördeki tüm wav dosyalarını bul
    wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
    print(f"Toplam {len(wav_files)} dosya işlenecek...")

    for file_path in wav_files:
        base_name = os.path.basename(file_path).replace(".wav", "").replace(" ", "_")
        print(f"İşleniyor: {base_name} (Bu biraz zaman alabilir...)")

        try:
            # Sesi yükle
            audio = AudioSegment.from_file(file_path)
            duration_ms = len(audio)

            # Parçalara böl ve hemen kaydet (RAM'i boşaltmak için)
            count = 0
            for i in range(0, duration_ms, chunk_length_ms):
                chunk = audio[i:i + chunk_length_ms]

                # 3 saniyeden kısa olan son parçayı atla
                if len(chunk) < chunk_length_ms:
                    continue

                output_filename = f"{base_name}_part{count}.wav"
                chunk.export(os.path.join(output_folder, output_filename), format="wav")
                count += 1

            # RAM'i temizlemek için objeyi sil
            del audio
            print(f"--- Bitti: {count} adet parça oluşturuldu. ---")

        except Exception as e:
            print(f"Hata oluştu ({base_name}): {e}")


if __name__ == "__main__":
    # Önemli: Eğer pydub ffmpeg'i bulamazsa bin klasöründeki 3 exe'yi bu kodun yanına at!
    safe_segment_sounds('cooking_dataset', 'pisme_veriseti_final')
    print("\n✅ Tüm işlem başarıyla tamamlandı!")