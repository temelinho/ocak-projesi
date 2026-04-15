const dropZone = document.getElementById('drop-zone');
const audioInput = document.getElementById('audio-input');
const loadingState = document.getElementById('loading-state');
const resultSection = document.getElementById('result-section');
const btnReset = document.getElementById('btn-reset');
const errorToast = document.getElementById('error-toast');

// Fast API arka ucu portu
const API_URL = 'http://localhost:8000/predict';

dropZone.addEventListener('click', () => audioInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

audioInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

btnReset.addEventListener('click', () => {
    resultSection.classList.add('hidden');
    dropZone.classList.remove('hidden');
    audioInput.value = '';
});

function showError(msg) {
    errorToast.textContent = msg;
    errorToast.classList.add('show');
    setTimeout(() => {
        errorToast.classList.remove('show');
    }, 4000);
}

async function handleFile(file) {
    if (!file.type.startsWith('audio/') && !file.name.match(/\.(wav|mp3|ogg|webm|m4a)$/i)) {
        showError('Lütfen geçerli bir ses dosyası yükleyin.');
        return;
    }

    dropZone.classList.add('hidden');
    loadingState.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        loadingState.classList.add('hidden');

        if (!response.ok || data.success === false) {
            throw new Error(data.detail || data.message || 'Analiz sırasında bir hata oluştu');
        }

        showResults(data);
    } catch (error) {
        loadingState.classList.add('hidden');
        dropZone.classList.remove('hidden');
        showError(error.message);
    }
}

function showResults(data) {
    const res = data.result;
    const details = data.details;

    document.getElementById('res-emoji').textContent = res.emoji;
    document.getElementById('res-title').textContent = res.name;
    document.getElementById('res-desc').textContent = `${res.description} (${res.confidence}%)`;

    const detailsList = document.getElementById('details-list');
    detailsList.innerHTML = '';

    details.sort((a,b) => b.percentage - a.percentage).forEach(item => {
        let color = '#a29bfe';
        if (item.label === 'pisme') color = '#fdcb6e';
        if (item.label === 'kaynama') color = '#00b894';
        if (item.label === 'gurultu') color = '#b2bec3';

        const detailHtml = `
            <div class="detail-item">
                <div class="detail-header">
                    <span>${item.emoji} ${item.name}</span>
                    <span>${item.percentage}%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: 0%; background:${color};"></div>
                </div>
            </div>
        `;
        detailsList.insertAdjacentHTML('beforeend', detailHtml);
        
        setTimeout(() => {
            const bars = detailsList.querySelectorAll('.progress-bar-fill');
            const lastBar = bars[bars.length-1];
            if(lastBar) lastBar.style.width = `${item.percentage}%`;
        }, 50);
    });

    resultSection.classList.remove('hidden');
}
