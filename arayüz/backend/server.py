"""
Ocak Ses Analiz API - FastAPI Backend
Mevcut ocak-projesi modelini okuyarak ses sınıflandırması yapar.
ocak-projesi klasörüne hiçbir değişiklik yapmaz, sadece model ve etiket dosyasını okur.
"""

import os
import sys
import json
import tempfile
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')  # GUI olmadan çalışması için
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ─── YAPILANDIRMA ──────────────────────────────────────────────
# ocak-projesi klasörünün yolu (sadece okuma yapılacak)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ocak-projesi'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'ocak_siniflandirici.keras')
CLASS_INDICES_PATH = os.path.join(PROJECT_ROOT, 'class_indices.json')

# Türkçe etiketler
LABEL_MAP = {
    'gurultu': {'emoji': '🔇', 'name': 'GÜRÜLTÜ', 'description': 'Ocakta bir şey pişmiyor.'},
    'kaynama': {'emoji': '🫧', 'name': 'KAYNIYOR', 'description': 'Yemek kaynıyor!'},
    'pisme':   {'emoji': '🍳', 'name': 'PİŞİYOR', 'description': 'Yemek pişiyor!'},
}

# ─── UYGULAMA ──────────────────────────────────────────────────
app = FastAPI(title="Ocak Ses Analiz API", version="1.0.0")

# CORS ayarları (React dev server'ından gelen istekleri kabul et)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODEL YÜKLEME ────────────────────────────────────────────
model = None
index_to_label = None

@app.on_event("startup")
def load_model():
    """Sunucu başladığında modeli ve etiket haritasını yükle."""
    global model, index_to_label
    
    import tensorflow as tf
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model dosyası bulunamadı: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(CLASS_INDICES_PATH):
        print(f"❌ Sınıf etiket dosyası bulunamadı: {CLASS_INDICES_PATH}")
        sys.exit(1)
    
    print(f"📦 Model yükleniyor: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(CLASS_INDICES_PATH, 'r', encoding='utf-8') as f:
        index_to_label = json.load(f)
    
    print("✅ Model ve etiketler başarıyla yüklendi!")
    print(f"   Sınıflar: {list(index_to_label.values())}")


def process_audio(file_path: str) -> dict:
    """
    Ses dosyasını spektrograma çevir, modele gönder ve sonuçları döndür.
    test_cnn.py dosyasındaki mantığın aynısı, ama print yerine dict döndürüyor.
    """
    from tensorflow.keras.preprocessing import image as keras_image
    
    # 1. Sesi yükle ve mel spektrogram oluştur
    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # 2. Spektrogramı geçici görsel dosya olarak kaydet
    temp_image = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_db)
    plt.axis('off')
    plt.savefig(temp_image, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 3. Resmi modelin beklediği formata getir
    img = keras_image.load_img(temp_image, target_size=(128, 128))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # 4. Tahmin yap
    predictions = model.predict(img_array, verbose=0)[0]
    
    # 5. Geçici dosyayı temizle
    if os.path.exists(temp_image):
        os.remove(temp_image)
    
    # 6. Sonuçları hazırla
    best_idx = int(np.argmax(predictions))
    best_label = index_to_label[str(best_idx)]
    label_info = LABEL_MAP[best_label]
    
    # Her sınıf için detaylı sonuçlar
    class_results = []
    for i in range(len(predictions)):
        label = index_to_label[str(i)]
        info = LABEL_MAP[label]
        class_results.append({
            'label': label,
            'name': info['name'],
            'emoji': info['emoji'],
            'percentage': round(float(predictions[i]) * 100, 1),
        })
    
    return {
        'success': True,
        'result': {
            'label': best_label,
            'name': label_info['name'],
            'emoji': label_info['emoji'],
            'description': label_info['description'],
            'confidence': round(float(predictions[best_idx]) * 100, 1),
        },
        'details': class_results,
    }


# ─── API UÇLARI ───────────────────────────────────────────────

@app.get("/")
def root():
    """Sağlık kontrolü."""
    return {"status": "ok", "message": "Ocak Ses Analiz API çalışıyor!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Bir ses dosyasını alır, analiz eder ve sonuçları döndürür.
    Kabul edilen formatlar: .wav, .mp3, .ogg, .webm, .m4a
    """
    allowed_extensions = {'.wav', '.mp3', '.ogg', '.webm', '.m4a', '.flac'}
    
    # Dosya uzantısı kontrolü
    _, ext = os.path.splitext(file.filename or "file.wav")
    ext = ext.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya formatı: {ext}. Desteklenen: {', '.join(allowed_extensions)}"
        )
    
    # Geçici dosyaya yaz
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        # Analiz yap
        result = process_audio(temp_file.name)
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")
    
    finally:
        # Geçici dosyayı temizle
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
