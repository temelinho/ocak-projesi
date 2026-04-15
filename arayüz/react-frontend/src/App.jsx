import React, { useState, useRef, useEffect } from 'react';
import './index.css';

const API_URL = 'http://localhost:8000/predict';

function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [selectedFileName, setSelectedFileName] = useState('');
  
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  
  const fileInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);

  // Animations trigger logic
  const [progressWidths, setProgressWidths] = useState({});

  useEffect(() => {
    if (result && result.details) {
      setTimeout(() => {
        const newWidths = {};
        result.details.forEach(item => {
          newWidths[item.label] = item.percentage;
        });
        setProgressWidths(newWidths);
      }, 50);
    } else {
      setProgressWidths({});
    }
  }, [result]);

  const showError = (msg) => {
    setError(msg);
    setTimeout(() => setError(null), 4000);
  };

  const uploadFile = async (file) => {
    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      
      if (!response.ok || data.success === false) {
        throw new Error(data.detail || data.message || 'Analiz sırasında bir hata oluştu');
      }

      setResult(data);
    } catch (err) {
      showError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const onFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const selected = e.target.files[0];
      setSelectedFileName(selected.name);
      uploadFile(selected);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const file = new File([audioBlob], "live_recording.webm", { type: 'audio/webm' });
        uploadFile(file);
        
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (err) {
      showError('Mikrofona erişilemedi. Lütfen mikrofon izni verdiğinizden emin olun.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(timerRef.current);
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  const resetTest = () => {
    setResult(null);
    setSelectedFileName('');
    if(fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="App">
      <div className="background-blobs">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
      </div>

      <main className="glass-container">
        <header>
          <div className="logo-icon">🍳</div>
          <h1>Ocak <span>AI</span></h1>
          <p>Yapay zeka ile anlık ocak analizi</p>
          <div className="header-badges">
            <span className="badge">Anlik analiz</span>
            <span className="badge">3 sinif tahmin</span>
          </div>
        </header>

        {error && <div className="error-toast">{error}</div>}

        {!loading && !result && (
          <div className="flex-row">
            <input 
              type="file" 
              accept="audio/*" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              onChange={onFileChange} 
            />
            
            <button 
              className="action-btn" 
              onClick={() => fileInputRef.current.click()}
              disabled={isRecording}
            >
              <span className="icon-large">📁</span>
              <span>Dosya Seç</span>
              {isRecording && <div className="timer" style={{opacity:0}}>00:00</div>}
            </button>

            <button 
              className={`action-btn ${isRecording ? 'recording' : ''}`}
              onClick={toggleRecording}
            >
              <span className="icon-large">{isRecording ? '⏹️' : '🎙️'}</span>
              <span>{isRecording ? 'Kaydı Bitir' : 'Canlı Dinle'}</span>
              {isRecording && <div className="timer">{formatTime(recordingTime)}</div>}
            </button>
          </div>
        )}

        {!loading && !result && selectedFileName && (
          <div className="selected-file-card">
            <span className="selected-file-label">Secilen dosya</span>
            <span className="selected-file-name" title={selectedFileName}>{selectedFileName}</span>
          </div>
        )}

        {loading && (
          <div className="loading-state">
            <div className="spinner"></div>
            <p>Yapay Zeka Analiz Ediyor...</p>
          </div>
        )}

        {result && (
          <div className="result-section">
            <div className="main-result">
              <span className="result-emoji">{result.result.emoji}</span>
              <h2 id="res-title">{result.result.name}</h2>
              <p id="res-desc">{result.result.description} ({result.result.confidence}%)</p>
            </div>

            {result.warning?.type === 'LOW_VOLUME' && (
              <div className="inline-warning">
                {result.warning.message}
              </div>
            )}
            {result.quality_warning?.type === 'NOISY_OR_SILENT_ENV' && (
              <div className="inline-warning">
                {result.quality_warning.message}
              </div>
            )}

            <div className="details-list">
              {[...result.details].sort((a,b) => b.percentage - a.percentage).map((item, idx) => {
                let color = '#a29bfe';
                if (item.label === 'pisme') color = '#fdcb6e';
                if (item.label === 'kaynama') color = '#00b894';
                if (item.label === 'gurultu') color = '#b2bec3';
                
                const width = progressWidths[item.label] || 0;

                return (
                  <div className="detail-item" key={item.label}>
                    <div className="detail-header">
                      <span>{item.emoji} {item.name}</span>
                      <span>{item.percentage}%</span>
                    </div>
                    <div className="progress-bar-bg">
                      <div 
                        className="progress-bar-fill" 
                        style={{ width: `${width}%`, backgroundColor: color }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>

            <button className="btn-reset" onClick={resetTest}>Yeni Test Yap</button>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
