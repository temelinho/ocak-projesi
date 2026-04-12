import { useState, useRef, useEffect, useCallback } from 'react'

const API_URL = 'http://localhost:8000'

export default function App() {
  const [mode, setMode] = useState('record') // 'record' | 'upload'
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [audioBlob, setAudioBlob] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [waveformData, setWaveformData] = useState(new Array(40).fill(4))
  const [dragOver, setDragOver] = useState(false)

  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const analyserRef = useRef(null)
  const animationRef = useRef(null)
  const streamRef = useRef(null)
  const fileInputRef = useRef(null)

  // ─── WAVEFORM ANİMASYONU ─────────────────────────────────
  const updateWaveform = useCallback(() => {
    if (!analyserRef.current) return
    const data = new Uint8Array(analyserRef.current.frequencyBinCount)
    analyserRef.current.getByteFrequencyData(data)

    const bars = 40
    const step = Math.floor(data.length / bars)
    const newWaveform = []
    for (let i = 0; i < bars; i++) {
      const val = data[i * step]
      newWaveform.push(Math.max(4, (val / 255) * 70))
    }
    setWaveformData(newWaveform)
    animationRef.current = requestAnimationFrame(updateWaveform)
  }, [])

  // ─── KAYIT BAŞLAT / DURDUR ───────────────────────────────
  const startRecording = async () => {
    try {
      setError(null)
      setResult(null)
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      // Waveform analiz
      const audioCtx = new AudioContext()
      const source = audioCtx.createMediaStreamSource(stream)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      analyserRef.current = analyser

      // MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        setAudioBlob(blob)
        stream.getTracks().forEach(track => track.stop())
        if (animationRef.current) cancelAnimationFrame(animationRef.current)
        analyserRef.current = null
        setWaveformData(new Array(40).fill(4))
      }

      mediaRecorder.start(250)
      setIsRecording(true)
      setRecordingTime(0)

      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1)
      }, 1000)

      updateWaveform()
    } catch (err) {
      setError('Mikrofon erişimi reddedildi. Lütfen tarayıcı izinlerini kontrol edin.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    setIsRecording(false)
    if (timerRef.current) clearInterval(timerRef.current)
  }

  // ─── DOSYA YÖNETİMİ ─────────────────────────────────────
  const handleFileSelect = (file) => {
    if (!file) return
    const allowed = ['.wav', '.mp3', '.ogg', '.webm', '.m4a', '.flac']
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!allowed.includes(ext)) {
      setError(`Desteklenmeyen format: ${ext}. Desteklenen: ${allowed.join(', ')}`)
      return
    }
    setError(null)
    setResult(null)
    setSelectedFile(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFileSelect(file)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => {
    setDragOver(false)
  }

  // ─── ANALİZ ──────────────────────────────────────────────
  const analyzeAudio = async () => {
    setIsAnalyzing(true)
    setError(null)
    setResult(null)

    const formData = new FormData()

    if (mode === 'record' && audioBlob) {
      formData.append('file', audioBlob, 'recording.webm')
    } else if (mode === 'upload' && selectedFile) {
      formData.append('file', selectedFile, selectedFile.name)
    } else {
      setError('Lütfen önce bir ses kaydı yapın veya dosya yükleyin.')
      setIsAnalyzing(false)
      return
    }

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `Sunucu hatası: ${res.status}`)
      }

      const data = await res.json()
      if (data.success) {
        setResult(data)
      } else {
        throw new Error('Beklenmeyen yanıt formatı')
      }
    } catch (err) {
      if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        setError('Sunucuya bağlanılamadı. Lütfen backend sunucusunun (start_server.bat) çalıştığından emin olun.')
      } else {
        setError(err.message)
      }
    } finally {
      setIsAnalyzing(false)
    }
  }

  // ─── SIFIRLAMA ───────────────────────────────────────────
  const resetAll = () => {
    setAudioBlob(null)
    setSelectedFile(null)
    setResult(null)
    setError(null)
    setRecordingTime(0)
    setIsAnalyzing(false)
  }

  // ─── ZAMANLAYICI FORMAT ──────────────────────────────────
  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0')
    const s = (seconds % 60).toString().padStart(2, '0')
    return `${m}:${s}`
  }

  // ─── CLEANUP ─────────────────────────────────────────────
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop())
    }
  }, [])

  // ─── RENDER ──────────────────────────────────────────────
  const canAnalyze = (mode === 'record' && audioBlob) || (mode === 'upload' && selectedFile)

  return (
    <>
      <div className="app-bg" />
      <div className="app-container">
        {/* HEADER */}
        <header className="header">
          <span className="header-icon">🍳</span>
          <h1>Ocak Ses Analizi</h1>
          <p>Yapay zeka ile mutfak seslerinizi analiz edin</p>
        </header>

        {/* MODE TABS */}
        <div className="mode-tabs" role="tablist">
          <button
            className={`mode-tab${mode === 'record' ? ' active' : ''}`}
            onClick={() => { setMode('record'); resetAll() }}
            role="tab"
            aria-selected={mode === 'record'}
            id="tab-record"
          >
            <span className="tab-icon">🎙️</span>
            Anlık Kayıt
          </button>
          <button
            className={`mode-tab${mode === 'upload' ? ' active' : ''}`}
            onClick={() => { setMode('upload'); resetAll() }}
            role="tab"
            aria-selected={mode === 'upload'}
            id="tab-upload"
          >
            <span className="tab-icon">📁</span>
            Dosya Yükle
          </button>
        </div>

        {/* KAYIT MODU */}
        {mode === 'record' && !isAnalyzing && !result && (
          <div className="recorder-section">
            {/* Waveform */}
            {isRecording && (
              <>
                <div className="recording-timer">{formatTime(recordingTime)}</div>
                <div className="waveform-container">
                  {waveformData.map((h, i) => (
                    <div
                      key={i}
                      className={`waveform-bar${h > 8 ? ' active' : ''}`}
                      style={{ height: `${h}px` }}
                    />
                  ))}
                </div>
              </>
            )}

            {/* Mic Button */}
            <button
              className={`mic-button${isRecording ? ' recording' : ''}`}
              onClick={isRecording ? stopRecording : startRecording}
              id="mic-button"
              aria-label={isRecording ? 'Kaydı durdur' : 'Kayda başla'}
            >
              <span className="mic-icon">{isRecording ? '⏹️' : '🎙️'}</span>
            </button>

            <div className="mic-label">
              {isRecording
                ? 'Kaydediliyor... Durdurmak için tıklayın'
                : audioBlob
                  ? '✅ Kayıt tamamlandı! Analiz edebilirsiniz.'
                  : <>Kayda başlamak için tıklayın<span>Mutfak seslerinizi dinletelim</span></>
              }
            </div>

            {/* Analiz Butonu */}
            {audioBlob && (
              <button
                className="analyze-btn"
                onClick={analyzeAudio}
                disabled={!canAnalyze}
                id="analyze-btn-record"
              >
                🔬 Sesi Analiz Et
              </button>
            )}
          </div>
        )}

        {/* YÜKLEME MODU */}
        {mode === 'upload' && !isAnalyzing && !result && (
          <div>
            <div
              className={`dropzone${dragOver ? ' drag-over' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              id="dropzone"
              role="button"
              tabIndex={0}
            >
              <span className="dropzone-icon">🎵</span>
              <div className="dropzone-text">
                <h3>Ses dosyanızı sürükleyip bırakın</h3>
                <p>veya tıklayarak seçin  •  WAV, MP3, OGG, WebM, M4A, FLAC</p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".wav,.mp3,.ogg,.webm,.m4a,.flac"
                onChange={(e) => handleFileSelect(e.target.files[0])}
              />

              {selectedFile && (
                <div className="file-selected">
                  <span className="file-icon">🎶</span>
                  <span className="file-name">{selectedFile.name}</span>
                  <button
                    className="file-remove"
                    onClick={(e) => { e.stopPropagation(); setSelectedFile(null) }}
                    aria-label="Dosyayı kaldır"
                  >
                    ✕
                  </button>
                </div>
              )}
            </div>

            {selectedFile && (
              <button
                className="analyze-btn"
                onClick={analyzeAudio}
                disabled={!canAnalyze}
                id="analyze-btn-upload"
              >
                🔬 Sesi Analiz Et
              </button>
            )}
          </div>
        )}

        {/* YÜKLEME DURUMU */}
        {isAnalyzing && (
          <div className="loading-overlay">
            <div className="loading-spinner" />
            <div className="loading-text">
              Ses analiz ediliyor<span className="loading-dots"></span>
            </div>
          </div>
        )}

        {/* HATA */}
        {error && (
          <div className="error-box">
            <span className="error-icon">⚠️</span>
            <p className="error-text">{error}</p>
          </div>
        )}

        {/* SONUÇLAR */}
        {result && (
          <div className="results-section">
            <div className="result-card">
              {/* Sonuç Başlığı */}
              <div className={`result-header ${result.result.label}`}>
                <span className="result-emoji">{result.result.emoji}</span>
                <div className="result-title">{result.result.description}</div>
                <div className="result-description">{result.result.name}</div>
                <span className="result-confidence">
                  %{result.result.confidence} güven
                </span>
              </div>

              {/* Detaylı Barlar */}
              <div className="result-bars">
                {result.details.map((item) => (
                  <div className="result-bar-item" key={item.label}>
                    <div className="bar-label">
                      <div className="bar-label-left">
                        <span className="bar-emoji">{item.emoji}</span>
                        <span className="bar-name">{item.name}</span>
                      </div>
                      <span className="bar-percentage">%{item.percentage}</span>
                    </div>
                    <div className="bar-track">
                      <div
                        className={`bar-fill ${item.label}`}
                        style={{ width: `${item.percentage}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <button className="reset-btn" onClick={resetAll} id="reset-btn">
              🔄 Yeni Analiz Yap
            </button>
          </div>
        )}

        {/* FOOTER */}
        <footer className="footer">
          Ocak Ses Analiz Sistemi — Yapay zeka destekli mutfak asistanı
        </footer>
      </div>
    </>
  )
}
