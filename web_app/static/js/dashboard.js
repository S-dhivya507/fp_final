// // =====================================================
// // GLOBAL VARIABLES
// // =====================================================

let emotionChart = null;
let liveOverlayStream = null;
const emotionLabels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'];
const VIDEO_DURATION = 10;
const AUDIO_DURATION = 10;
const emotionIcons = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😊',
    'Neutral': '😐',
    'Sad': '😔',
    'Surprise': '😲'
};

const stressRecommendations = {
    high: [
        'Try deep breathing exercises for 5 minutes',
        'Take a short walk to clear your mind',
        'Consider meditation or mindfulness practice',
        'Reach out to a friend or counselor'
    ],
    medium: [
        'Take a brief break from current activities',
        'Practice some light stretching',
        'Listen to calming music',
        'Have a healthy snack and water'
    ],
    low: [
        'You are doing great! Keep up the positive mood',
        'Continue with your current activities',
        'Maintain good sleep and exercise routines',
        'Share your positive energy with others'
    ]
};
async function startRecording() {
    const btn = document.getElementById('startRecordingBtn');
    const progressContainer = document.getElementById('progressContainer');
    const emptyState = document.getElementById('emptyState');
    const resultsGrid = document.getElementById('resultsGrid');
    const systemStatus = document.getElementById('systemStatus');
    const captureMode = document.getElementById('captureMode');

    // Disable button and show progress
    btn.disabled = true;
    progressContainer.style.display = 'block';
    emptyState.style.display = 'none';
    captureMode.textContent = 'Live Recording (Browser)';
    systemStatus.textContent = 'Recording';
    clearCapturedMedia();

    let captureSequence = null;
    try {
        // Show loading status
        updateProgress(10, 'Initializing camera...');
        resetCaptureUI();

        captureSequence = runCaptureSequence();

        const timestamp = Date.now();
        const videoBlob = await recordBrowserVideo(VIDEO_DURATION * 1000);
        const audioBlob = await recordBrowserAudio(AUDIO_DURATION * 1000);
        const formData = new FormData();
        formData.append('video', videoBlob, `live_${timestamp}.webm`);
        formData.append('audio', audioBlob, `live_${timestamp}.webm`);

        const response = await fetch('/api/upload-analysis', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Recording failed');

        const data = await response.json();

        if (data.success) {
            updateProgress(100, 'Analysis Complete!');
            captureSequence.complete();
            displayResults(data);
            resultsGrid.style.display = 'grid';
            setLastRun();
            systemStatus.textContent = 'Complete';
            
            setTimeout(() => {
                progressContainer.style.display = 'none';
                btn.disabled = false;
            }, 2000);
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
        progressContainer.style.display = 'none';
        btn.disabled = false;
        if (captureSequence) captureSequence.cancel();
        setCaptureError();
        systemStatus.textContent = 'Error';
    } finally {
        stopLivePreview();
    }
}

async function recordBrowserVideo(durationMs) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Browser media capture is not supported.');
    }

    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    const mimeTypes = [
        'video/webm;codecs=vp9,opus',
        'video/webm;codecs=vp8,opus',
        'video/webm'
    ];
    let options = {};
    for (const type of mimeTypes) {
        if (MediaRecorder.isTypeSupported(type)) {
            options = { mimeType: type };
            break;
        }
    }

    const recorder = new MediaRecorder(stream, options);
    const chunks = [];

    const stopped = new Promise((resolve, reject) => {
        recorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) chunks.push(event.data);
        };
        recorder.onerror = () => reject(new Error('Recording failed'));
        recorder.onstop = () => resolve();
    });

    recorder.start();
    setTimeout(() => recorder.stop(), durationMs);
    await stopped;

    stream.getTracks().forEach((track) => track.stop());

    const type = recorder.mimeType || 'video/webm';
    return new Blob(chunks, { type });
}

async function recordBrowserAudio(durationMs) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Browser media capture is not supported.');
    }

    const stream = await navigator.mediaDevices.getUserMedia({ video: false, audio: true });
    const mimeTypes = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus'
    ];
    let options = {};
    for (const type of mimeTypes) {
        if (MediaRecorder.isTypeSupported(type)) {
            options = { mimeType: type };
            break;
        }
    }

    const recorder = new MediaRecorder(stream, options);
    const chunks = [];

    const stopped = new Promise((resolve, reject) => {
        recorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) chunks.push(event.data);
        };
        recorder.onerror = () => reject(new Error('Recording failed'));
        recorder.onstop = () => resolve();
    });

    recorder.start();
    setTimeout(() => recorder.stop(), durationMs);
    await stopped;

    stream.getTracks().forEach((track) => track.stop());

    const type = recorder.mimeType || 'audio/webm';
    return new Blob(chunks, { type });
}

// // =====================================================
// // START RECORDING & ANALYSIS
// // =====================================================


// // =====================================================
// // QUICK DEMO ANALYSIS
// // =====================================================

async function quickAnalysis() {
    const btn = document.getElementById('quickAnalysisBtn');
    const resultsGrid = document.getElementById('resultsGrid');
    const emptyState = document.getElementById('emptyState');
    const systemStatus = document.getElementById('systemStatus');
    const captureMode = document.getElementById('captureMode');

    btn.disabled = true;
    emptyState.style.display = 'none';
    captureMode.textContent = 'Quick Demo';
    systemStatus.textContent = 'Analyzing';
    clearCapturedMedia();

    try {
        const response = await fetch('/api/quick-analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) throw new Error('Analysis failed');

        const data = await response.json();

        if (data.success) {
            displayResults(data);
            resultsGrid.style.display = 'grid';
            setLastRun();
            systemStatus.textContent = 'Complete';
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
        systemStatus.textContent = 'Error';
    } finally {
        btn.disabled = false;
    }
}

async function uploadAnalysis() {
    const btn = document.getElementById('uploadAnalysisBtn');
    const resultsGrid = document.getElementById('resultsGrid');
    const emptyState = document.getElementById('emptyState');
    const uploadStatus = document.getElementById('uploadStatus');
    const systemStatus = document.getElementById('systemStatus');
    const captureMode = document.getElementById('captureMode');

    const videoFile = document.getElementById('videoFile').files[0];
    const audioFile = document.getElementById('audioFile').files[0];

    if (!videoFile && !audioFile) {
        uploadStatus.innerHTML = '<i class="fas fa-exclamation-triangle"></i><span>Select a video or audio file to analyze.</span>';
        return;
    }

    btn.disabled = true;
    emptyState.style.display = 'none';
    uploadStatus.innerHTML = '<i class="fas fa-spinner"></i><span>Uploading and analyzing...</span>';
    captureMode.textContent = 'File Upload';
    systemStatus.textContent = 'Analyzing';
    clearCapturedMedia();

    try {
        const formData = new FormData();
        if (videoFile) formData.append('video', videoFile);
        if (audioFile) formData.append('audio', audioFile);

        const response = await fetch('/api/upload-analysis', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Upload analysis failed');

        const data = await response.json();

        if (data.success) {
            displayResults(data);
            resultsGrid.style.display = 'grid';
            uploadStatus.innerHTML = '<i class="fas fa-check-circle"></i><span>Analysis complete.</span>';
            setLastRun();
            systemStatus.textContent = 'Complete';
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
        uploadStatus.innerHTML = '<i class="fas fa-times-circle"></i><span>Upload failed. Try a different file.</span>';
        systemStatus.textContent = 'Error';
    } finally {
        btn.disabled = false;
    }
}

// // =====================================================
// // DISPLAY RESULTS
// // =====================================================

function displayResults(data) {
    const { fused_emotions, stress_level, face_emotions, voice_probs, voice_emotions } = data;

    // Display stress indicator
    const fused = ensureEmotionMap(fused_emotions || {});
    displayStressIndicator(stress_level, fused);

    // Display emotion chart
    displayEmotionChart(fused);

    // Display emotion details
    displayEmotionDetails(fused);


    // Display face emotions
    if (face_emotions) {
        const faceMap = ensureEmotionMap(face_emotions);
        displayEmotionBars('faceEmotions', faceMap, 'Face Analysis');
        if (data.face_detected === false) {
            const faceContainer = document.getElementById('faceEmotions');
            faceContainer.innerHTML = '<div class="empty-inline">No face detected. Please face the camera directly and try again.</div>';
        }
    }

    // Display voice emotions
    if (voice_emotions) {
        const voiceMap = ensureEmotionMap(voice_emotions);
        displayEmotionBars('voiceEmotions', voiceMap, 'Voice Analysis');
    } else if (voice_probs) {
        const voiceMap = {};
        emotionLabels.forEach((emotion, idx) => {
            const value = Number(voice_probs[idx]);
            voiceMap[emotion] = Number.isFinite(value) ? value * 100 : 0;
        });
        displayEmotionBars('voiceEmotions', ensureEmotionMap(voiceMap), 'Voice Analysis');
    }

    // Display recommendations
    displayRecommendations(stress_level);

    // Display captured media
    displayCapturedMedia(data);
}

// // =====================================================
// // STRESS INDICATOR
// // =====================================================

function displayStressIndicator(stressLevel, emotions) {
    const indicator = document.getElementById('stressIndicator');
    const stressValue = document.getElementById('stressValue');
    const stressLevelText = document.getElementById('stressLevel');
    const gaugeFill = document.getElementById('gaugeFill');

    const score = stressLevel.score || 50;
    const percentage = (score / 100) * 100;

    // Update gauge fill
    const circumference = 2 * Math.PI * 40; // radius = 40
    const offset = circumference - (percentage / 100) * circumference;
    gaugeFill.style.strokeDashoffset = offset;

    // Update color based on stress level
    if (stressLevel.level === 'High') {
        gaugeFill.style.stroke = '#ef4444';
    } else if (stressLevel.level === 'Medium') {
        gaugeFill.style.stroke = '#f97316';
    } else {
        gaugeFill.style.stroke = '#10b981';
    }

    stressValue.textContent = Math.round(score);
    stressLevelText.textContent = stressLevel.level + ' Stress';
    stressLevelText.style.color = stressLevel.color;
}

// // =====================================================
// // EMOTION CHART
// // =====================================================

function displayEmotionChart(emotions) {
    const ctx = document.getElementById('emotionChart').getContext('2d');

    const labels = emotionLabels.slice();
    const data = labels.map(label => {
        const value = Number(emotions[label]);
        return Number.isFinite(value) ? value : 0;
    });

    // Destroy previous chart if exists
    if (emotionChart) {
        emotionChart.destroy();
    }

    emotionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Emotion Score (%)',
                data: data,
                backgroundColor: [
                    '#ef4444', '#a78bfa', '#f97316', '#10b981',
                    '#64748b', '#0ea5e9', '#fbbf24'
                ],
                borderColor: '#3b82f6',
                borderWidth: 2,
                borderRadius: 8,
                barThickness: 40
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            indexAxis: 'x',
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#cbd5e1' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                x: {
                    ticks: { color: '#cbd5e1', autoSkip: false, maxRotation: 0, minRotation: 0, font: { size: 10 } },
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(30, 41, 59, 0.9)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#f1f5f9',
                    borderColor: '#3b82f6',
                    borderWidth: 1
                }
            }
        }
    });
}

// // =====================================================
// // EMOTION DETAILS
// // =====================================================

function displayEmotionDetails(emotions) {
    const container = document.getElementById('emotionDetails');
    container.innerHTML = '';

    emotionLabels.forEach((emotion) => {
        const score = Number(emotions[emotion]) || 0;
        const item = document.createElement('div');
        item.className = 'emotion-item';
        item.innerHTML = `
            <h4>${emotionIcons[emotion] || ''} ${emotion}</h4>
            <span class="score">${parseFloat(score).toFixed(2)}%</span>
        `;
        container.appendChild(item);
    });
}

// // =====================================================
// // EMOTION BARS
// // =====================================================

function displayEmotionBars(containerId, emotions, title) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    emotionLabels.forEach(emotion => {
        const raw = parseFloat(emotions[emotion] || 0);
        const score = Math.min(100, Math.max(0, raw));
        const bar = document.createElement('div');
        bar.className = 'emotion-bar';
        bar.innerHTML = `
            <div class="emotion-bar-label">${emotion}</div>
            <div class="emotion-bar-container">
                <div class="emotion-bar-fill" style="width: ${score}%; background: linear-gradient(90deg, 
                    ${getEmotionColor(emotion)}, 
                    ${getEmotionColor(emotion)}88);"></div>
            </div>
            <div class="emotion-bar-value">${score.toFixed(1)}%</div>
        `;
        container.appendChild(bar);
    });
}

function getEmotionColor(emotion) {
    const colors = {
        'Angry': '#ef4444',
        'Disgust': '#a78bfa',
        'Fear': '#f97316',
        'Happy': '#10b981',
        'Neutral': '#64748b',
        'Sad': '#0ea5e9',
        'Surprise': '#fbbf24'
    };
    return colors[emotion] || '#3b82f6';
}

function ensureEmotionMap(emotions) {
    const normalized = {};
    emotionLabels.forEach(label => {
        const value = Number(emotions[label]);
        const safe = Number.isFinite(value) ? value : 0;
        normalized[label] = Math.min(100, Math.max(0, safe));
    });
    return normalized;
}

// // =====================================================
// // RECOMMENDATIONS
// // =====================================================

function displayRecommendations(stressLevel) {
    const container = document.getElementById('recommendations');
    container.innerHTML = '';

    const level = stressLevel.level.toLowerCase();
    const recommendations = stressRecommendations[level] || stressRecommendations.low;

    recommendations.forEach(rec => {
        const item = document.createElement('div');
        item.className = `recommendation-item ${level}`;
        item.innerHTML = `<p>✓ ${rec}</p>`;
        container.appendChild(item);
    });
}

function displayCapturedMedia(data) {
    const videoPreview = document.getElementById('videoPreview');
    const audioPreview = document.getElementById('audioPreview');
    const videoLink = document.getElementById('videoLink');
    const audioLink = document.getElementById('audioLink');
    const videoPreviewNote = document.getElementById('videoPreviewNote');

    if (data.video_file) {
        const videoEl = document.createElement('video');
        videoEl.controls = true;
        const sourceEl = document.createElement('source');
        sourceEl.src = data.video_file;
        sourceEl.type = getVideoMimeType(data.video_file);
        videoEl.appendChild(sourceEl);
        videoEl.addEventListener('error', () => {
            videoPreview.textContent = 'Video saved, but preview is not supported in this browser.';
            if (videoPreviewNote) {
                videoPreviewNote.textContent = 'Video saved, but preview is not supported in this browser.';
                videoPreviewNote.style.display = 'block';
            }
        });
        videoPreview.innerHTML = '';
        videoPreview.appendChild(videoEl);
        videoLink.href = data.video_file;
        videoLink.style.display = 'inline-flex';
        if (videoPreviewNote) {
            videoPreviewNote.style.display = 'none';
            videoPreviewNote.textContent = '';
        }
        if (videoPreviewNote && data.video_preview_supported === false) {
            videoPreviewNote.textContent = 'Preview might not be supported. Please use the file link.';
            videoPreviewNote.style.display = 'block';
        }
    } else if (data.video_error) {
        videoPreview.textContent = data.video_error;
        if (videoPreviewNote) {
            videoPreviewNote.textContent = data.video_error;
            videoPreviewNote.style.display = 'block';
        }
        videoLink.style.display = 'none';
    } else {
        videoPreview.textContent = 'No video available yet.';
        videoLink.style.display = 'none';
        if (videoPreviewNote) {
            videoPreviewNote.style.display = 'none';
            videoPreviewNote.textContent = '';
        }
    }

    if (data.audio_file) {
        audioPreview.innerHTML = `<audio controls src="${data.audio_file}"></audio>`;
        audioLink.href = data.audio_file;
        audioLink.style.display = 'inline-flex';
    } else {
        audioPreview.textContent = 'No audio available yet.';
        audioLink.style.display = 'none';
    }
}

function getVideoMimeType(url) {
    if (!url) return 'video/mp4';
    const lower = url.toLowerCase();
    if (lower.endsWith('.webm')) return 'video/webm';
    if (lower.endsWith('.mp4')) return 'video/mp4';
    return 'video/mp4';
}

function clearCapturedMedia() {
    const videoPreview = document.getElementById('videoPreview');
    const audioPreview = document.getElementById('audioPreview');
    const videoLink = document.getElementById('videoLink');
    const audioLink = document.getElementById('audioLink');
    const videoPreviewNote = document.getElementById('videoPreviewNote');

    if (videoPreview) videoPreview.textContent = 'No video available yet.';
    if (audioPreview) audioPreview.textContent = 'No audio available yet.';
    if (videoLink) videoLink.style.display = 'none';
    if (audioLink) audioLink.style.display = 'none';
    if (videoPreviewNote) {
        videoPreviewNote.style.display = 'none';
        videoPreviewNote.textContent = '';
    }
}

// =====================================================
// UTILITY FUNCTIONS
// =====================================================

function updateProgress(value, text) {
    const fill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');

    fill.style.width = value + '%';
    progressText.textContent = text;
}

function runCaptureSequence() {
    let cancelled = false;

    const sequence = (async () => {
        await startLiveOverlay();
        setStepState('stepVideo', 'active');
        setStatusState('video', 'recording', 'Recording', 'Capturing facial signals...');
        updateProgress(30, 'Capturing video...');
        await countdown('videoTimer', 'videoMeter', VIDEO_DURATION, () => cancelled);
        if (cancelled) return;
        setStatusState('video', 'complete', 'Complete', 'Video capture complete.');
        setStepState('stepVideo', 'done');
        stopLiveOverlay();

        setStepState('stepAudio', 'active');
        setStatusState('audio', 'recording', 'Recording', 'Capturing voice signals...');
        updateProgress(60, 'Recording audio...');
        await countdown('audioTimer', 'audioMeter', AUDIO_DURATION, () => cancelled);
        if (cancelled) return;
        setStatusState('audio', 'complete', 'Complete', 'Audio recording complete.');
        setStepState('stepAudio', 'done');

        setStepState('stepFuse', 'active');
        updateProgress(85, 'Fusing signals...');
    })();

    return {
        complete: () => {
            cancelled = true;
            setStepState('stepFuse', 'done');
            setStatusState('video', 'complete', 'Complete', 'Video capture complete.');
            setStatusState('audio', 'complete', 'Complete', 'Audio recording complete.');
            updateProgress(100, 'Analysis complete.');
            stopLiveOverlay();
        },
        cancel: () => {
            cancelled = true;
            stopLiveOverlay();
        }
    };
}

async function startLiveOverlay() {
    const overlay = document.getElementById('liveOverlay');
    const videoEl = document.getElementById('liveOverlayVideo');
    const tag = document.getElementById('liveOverlayStatus');

    if (!overlay || !videoEl || liveOverlayStream) {
        return;
    }

    try {
        liveOverlayStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        videoEl.srcObject = liveOverlayStream;
        overlay.classList.add('active');
        if (tag) tag.textContent = 'Recording';
    } catch (err) {
        if (tag) tag.textContent = 'Blocked';
    }
}

function stopLiveOverlay() {
    const overlay = document.getElementById('liveOverlay');
    const videoEl = document.getElementById('liveOverlayVideo');

    if (liveOverlayStream) {
        liveOverlayStream.getTracks().forEach((track) => track.stop());
        liveOverlayStream = null;
    }

    if (videoEl) {
        videoEl.srcObject = null;
    }
    if (overlay) {
        overlay.classList.remove('active');
    }
}

function resetCaptureUI() {
    setStatusState('video', 'idle', 'Idle', 'Ready to capture facial data.');
    setStatusState('audio', 'idle', 'Idle', 'Ready to capture voice signals.');
    setStepState('stepVideo', 'idle');
    setStepState('stepAudio', 'idle');
    setStepState('stepFuse', 'idle');
    setTimerText('videoTimer', VIDEO_DURATION);
    setTimerText('audioTimer', AUDIO_DURATION);
    setMeterFill('videoMeter', 0);
    setMeterFill('audioMeter', 0);
}

function setCaptureError() {
    setStatusState('video', 'idle', 'Error', 'Capture failed. Check camera access.');
    setStatusState('audio', 'idle', 'Error', 'Recording failed. Check microphone access.');
    setStepState('stepVideo', 'idle');
    setStepState('stepAudio', 'idle');
    setStepState('stepFuse', 'idle');
}

function setStatusState(type, state, label, hint) {
    const dot = document.getElementById(`${type}Dot`);
    const tag = document.getElementById(`${type}Status`);
    const hintEl = document.getElementById(`${type}Hint`);

    dot.classList.remove('idle', 'recording', 'complete');
    dot.classList.add(state);
    tag.textContent = label;
    hintEl.textContent = hint;
}


function setStepState(id, state) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove('active', 'done');
    if (state === 'active') el.classList.add('active');
    if (state === 'done') el.classList.add('done');
}

function setTimerText(id, seconds) {
    const el = document.getElementById(id);
    if (!el) return;
    const value = Math.max(0, Math.floor(seconds));
    el.textContent = `00:${value.toString().padStart(2, '0')}`;
}

function setMeterFill(id, percent) {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.width = `${percent}%`;
}

async function countdown(timerId, meterId, seconds, shouldCancel) {
    for (let remaining = seconds; remaining >= 0; remaining -= 1) {
        if (shouldCancel && shouldCancel()) return;
        setTimerText(timerId, remaining);
        const progress = ((seconds - remaining) / seconds) * 100;
        setMeterFill(meterId, progress);
        await sleep(1000);
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function setLastRun() {
    const lastRun = document.getElementById('lastRun');
    if (!lastRun) return;
    const now = new Date();
    lastRun.textContent = now.toLocaleString();
}

// // =====================================================
// // PAGE LOAD
// // =====================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize if needed
    resetCaptureUI();

    const videoInput = document.getElementById('videoFile');
    const audioInput = document.getElementById('audioFile');
    const uploadStatus = document.getElementById('uploadStatus');

    const updateUploadStatus = () => {
        const videoFile = videoInput.files[0];
        const audioFile = audioInput.files[0];
        if (!videoFile && !audioFile) {
            uploadStatus.innerHTML = '<i class="fas fa-info-circle"></i><span>No files selected.</span>';
            return;
        }

        const parts = [];
        if (videoFile) parts.push(`Video: ${videoFile.name}`);
        if (audioFile) parts.push(`Audio: ${audioFile.name}`);
        uploadStatus.innerHTML = `<i class="fas fa-paperclip"></i><span>${parts.join(' | ')}</span>`;
    };

    if (videoInput && audioInput) {
        videoInput.addEventListener('change', updateUploadStatus);
        audioInput.addEventListener('change', updateUploadStatus);
    }

    console.log('Dashboard loaded successfully');
});


