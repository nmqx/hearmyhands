// HMH Learn Cards — Anki-style spaced repetition + webcam validation against
// the example video. Same MLP letter classifier as /translate is reused via
// the existing Socket.IO 'frame' handler (server is shared).
(function () {

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const STORAGE_KEY  = 'hmh-learn-state-v1';
const STORAGE_QKEY = 'hmh-learn-queue-v1';

// Letters with a static MLP class on the server. Others (J, P, X, Z) won't
// trigger auto-validation but the user can still rate them manually.
const MLP_LETTERS = new Set("ABCDEFGHIKLMNOQRSTUVWXY".split(""));

// Tuning
const SEND_WIDTH        = 480;
const JPEG_QUALITY      = 0.7;
const SEND_INTERVAL_MS  = 1000 / 15;       // 15 fps suffit pour la validation
const MAX_IN_FLIGHT     = 3;
const MIN_CONF          = 0.6;
const STABLE_FRAMES_OK  = 5;               // n frames consécutives = validé
const ACK_TIMEOUT_MS    = 2000;

// ── State persistence ────────────────────────────────────────────────────
const loadState = () => { try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch (e) { return {}; } };
const saveState = (s) => localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
const defaultLetterState = () => ({ status: 'new', reps: 0, nextDueIn: 0 });

// In-memory session queue, separate from persistent state
let state = loadState();
let queue = [];           // array of { letter, ... } in current session order
let current = null;

function buildQueue() {
    const items = LETTERS
        .filter(l => (state[l] || defaultLetterState()).status !== 'known')
        .map(l => ({ letter: l, due: (state[l] && state[l].nextDueIn) || 0 }));
    // Sort by due so already-due letters come first; stable order otherwise
    items.sort((a, b) => a.due - b.due);
    return items;
}

function tickQueue() {
    queue.forEach(it => { if (it.due > 0) it.due--; });
    // Re-sort: due == 0 first (those ready to be shown)
    queue.sort((a, b) => a.due - b.due);
}

function rate(letter, choice) {
    const st = state[letter] || defaultLetterState();
    if (choice === 'again') {
        st.status = 'learning';
        st.nextDueIn = 3;                                          // revient vite
    } else if (choice === 'learning') {
        st.status = 'learning';
        st.reps = (st.reps || 0) + 1;
        // Courbe d'espacement: 5, 10, 20, 40, 80 cartes max
        st.nextDueIn = Math.min(80, 5 * Math.pow(2, st.reps - 1));
    } else if (choice === 'known') {
        st.status = 'known';
        st.reps = (st.reps || 0) + 1;
        st.nextDueIn = 0;
    }
    state[letter] = st;
    saveState(state);
}

// ── DOM ──────────────────────────────────────────────────────────────────
const cardView      = document.getElementById('cardView');
const emptyDeck     = document.getElementById('emptyDeck');
const targetLetterEl= document.getElementById('targetLetter');
const exampleVideo  = document.getElementById('exampleVideo');
const exampleMissing= document.getElementById('exampleMissing');
const progressDoneEl= document.getElementById('progressDone');
const progressLeftEl= document.getElementById('progressLeft');
const learnHintEl   = document.getElementById('learnHint');

const detectedEl    = document.getElementById('learnDetected');
const confEl        = document.getElementById('learnConf');
const detectedBox   = document.getElementById('learnDetectedBox');
const validatedEl   = document.getElementById('learnValidated');
const placeholderEl = document.getElementById('learnPlaceholder');

const camStartBtn   = document.getElementById('learnStartBtn');
const video         = document.getElementById('learnVideoElement');
const skelCanvas    = document.getElementById('learnSkeletonCanvas');
const skelCtx       = skelCanvas.getContext('2d');

// ── Card lifecycle ───────────────────────────────────────────────────────
function showNext() {
    queue = buildQueue();
    tickQueue();
    current = queue.find(it => it.due <= 0) || queue[0] || null;

    const totalDone = LETTERS.filter(l => (state[l] || defaultLetterState()).status === 'known').length;
    progressDoneEl.textContent = totalDone;
    progressLeftEl.textContent = queue.length;

    if (!current) {
        cardView.hidden = true;
        emptyDeck.hidden = false;
        return;
    }
    cardView.hidden = false;
    emptyDeck.hidden = true;

    const L = current.letter;
    targetLetterEl.textContent = L;

    // Hint preview for "j'apprends"
    const st = state[L] || defaultLetterState();
    const nextLearningWait = Math.min(80, 5 * Math.pow(2, (st.reps || 0)));
    learnHintEl.textContent = `dans ${nextLearningWait} cartes`;

    // Reset visual
    resetValidation();
    loadExample(L);
    stableCount = 0;
    lastDetected = null;
}

function resetValidation() {
    validatedEl.hidden = true;
    detectedEl.textContent = '—';
    confEl.textContent = '';
    detectedBox.classList.remove('match');
}

// Bump si on ré-encode les vidéos — force Cloudflare/navigateur à refetch.
const VIDEO_VERSION = 'h264';
let loadSeq = 0;

function loadExample(letter) {
    const src = `/static/learn/${letter}.mp4?v=${VIDEO_VERSION}`;
    const mySeq = ++loadSeq;
    // Reset visuel
    exampleMissing.hidden = true;
    exampleVideo.style.display = '';
    // On NE FAIT PAS de removeAttribute('src')+load() pour reset — ça
    // génère un 'error' parasite (src vide) qui faisait basculer le video
    // sur le placeholder "Vidéo à venir". On laisse le simple set de src
    // suivant remplacer l'ancienne source.
    exampleVideo.onerror = () => {
        if (mySeq !== loadSeq) return;          // ignore les events d'un load périmé
        if (!exampleVideo.src || exampleVideo.src === window.location.href) return;
        exampleVideo.style.display = 'none';
        exampleMissing.hidden = false;
    };
    exampleVideo.oncanplay = () => {
        if (mySeq !== loadSeq) return;
        exampleVideo.play().catch(() => {/* autoplay refusé, l'utilisateur jouera à la main */});
    };
    exampleVideo.src = src;
}

// ── Rating buttons ───────────────────────────────────────────────────────
document.querySelectorAll('.learn-rate').forEach(btn => {
    btn.addEventListener('click', () => {
        if (!current) return;
        rate(current.letter, btn.dataset.rate);
        showNext();
    });
});

// ── Webcam + Socket.IO + prediction ──────────────────────────────────────
const encodeCanvas = document.createElement('canvas');
const encodeCtx    = encodeCanvas.getContext('2d');

let predicting = false;
let inFlight = 0;
let frameSeq = 0;
let lastAppliedSeq = -1;
let sendTimer = null;
let stableCount = 0;
let lastDetected = null;

const socket = io();
socket.on('connect',       () => console.log('Socket.IO connected:', socket.id));
socket.on('connect_error', (e) => console.error('Socket.IO error:', e));

camStartBtn.addEventListener('click', async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("getUserMedia indisponible — utilise Safari (iOS) ou Chrome/Firefox.");
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        });
        video.srcObject = stream;
        video.classList.add('active');
        camStartBtn.style.display = 'none';
        placeholderEl.hidden = true;
        await new Promise(r => {
            if (video.videoWidth) return r();
            video.addEventListener('loadedmetadata', r, { once: true });
        });
        alignCanvas();
        startPredictLoop();
    } catch (err) {
        console.error(err);
        alert("Impossible d'accéder à la caméra : " + (err.name || err.message));
    }
});

function alignCanvas() {
    const vw = video.videoWidth, vh = video.videoHeight;
    const cw = video.clientWidth, ch = video.clientHeight;
    if (!vw || !vh || !cw || !ch) return;
    const r = vw / vh, br = cw / ch;
    let w, h, x, y;
    if (r > br) { w = cw; h = cw / r; x = 0; y = (ch - h) / 2; }
    else        { h = ch; w = ch * r; x = (cw - w) / 2; y = 0; }
    skelCanvas.style.left = `${x}px`;
    skelCanvas.style.top = `${y}px`;
    skelCanvas.style.width = `${w}px`;
    skelCanvas.style.height = `${h}px`;
}
window.addEventListener('resize', alignCanvas);

function startPredictLoop() {
    predicting = true;
    sendTimer = setInterval(sendFrame, SEND_INTERVAL_MS);
}

function sendFrame() {
    if (!predicting || !video.srcObject || inFlight >= MAX_IN_FLIGHT) return;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return;
    const sendW = SEND_WIDTH;
    const sendH = Math.round(vh * SEND_WIDTH / vw);
    if (encodeCanvas.width !== sendW) { encodeCanvas.width = sendW; encodeCanvas.height = sendH; }
    encodeCtx.drawImage(video, 0, 0, sendW, sendH);
    const seq = ++frameSeq;
    inFlight++;
    const release = setTimeout(() => { inFlight = Math.max(0, inFlight - 1); }, ACK_TIMEOUT_MS);
    encodeCanvas.toBlob(async (blob) => {
        if (!blob) { clearTimeout(release); inFlight = Math.max(0, inFlight - 1); return; }
        const buf = await blob.arrayBuffer();
        socket.emit('frame', buf, (resp) => {
            clearTimeout(release);
            inFlight = Math.max(0, inFlight - 1);
            if (seq <= lastAppliedSeq) return;
            lastAppliedSeq = seq;
            applyPrediction(resp, sendW, sendH);
        });
    }, 'image/jpeg', JPEG_QUALITY);
}

function applyPrediction(data, w, h) {
    if (!data) return;
    if (skelCanvas.width !== w) { skelCanvas.width = w; skelCanvas.height = h; }
    alignCanvas();
    skelCtx.clearRect(0, 0, w, h);
    if (data.skeleton && data.skeleton.length >= 9) drawSkeleton(data.skeleton);
    if (data.hands    && data.hands.length)        drawHands(data.hands);

    const conf = data.confidence ?? 0;
    const letter = (data.letter && conf >= MIN_CONF) ? data.letter : null;
    detectedEl.textContent = letter || '—';
    confEl.textContent = letter ? `(${Math.round(conf * 100)}%)` : '';

    if (current && letter === current.letter) {
        detectedBox.classList.add('match');
        stableCount++;
        if (stableCount >= STABLE_FRAMES_OK) {
            validatedEl.hidden = false;
        }
    } else {
        detectedBox.classList.remove('match');
        validatedEl.hidden = true;
        if (letter !== lastDetected) stableCount = 0;
        lastDetected = letter;
    }
}

// ── Drawing helpers (skeleton + hands), reused from translate ────────────
const EDGES = [[0,1],[0,2],[1,3],[3,5],[2,4],[4,6],[1,7],[2,8],[7,8]];
const KP = { neck:'white', left:'cyan', right:'#ff5078', body:'#32ff78' };
const VIS = 0.3;
function color(i) {
    if (i === 0) return KP.neck;
    if (i === 1 || i === 3 || i === 5) return KP.left;
    if (i === 2 || i === 4 || i === 6) return KP.right;
    return KP.body;
}
function drawSkeleton(kp) {
    skelCtx.lineWidth = 4;
    EDGES.forEach(([i, j]) => {
        const a = kp[i], b = kp[j];
        if (a[2] < VIS || b[2] < VIS) return;
        skelCtx.strokeStyle = color(i);
        skelCtx.beginPath(); skelCtx.moveTo(a[0], a[1]); skelCtx.lineTo(b[0], b[1]); skelCtx.stroke();
    });
    skelCtx.lineWidth = 2; skelCtx.strokeStyle = 'rgba(0,0,0,0.6)';
    kp.forEach((p, idx) => {
        if (p[2] < VIS) return;
        skelCtx.fillStyle = color(idx);
        skelCtx.beginPath(); skelCtx.arc(p[0], p[1], 6, 0, 2 * Math.PI); skelCtx.fill(); skelCtx.stroke();
    });
}
const HAND_EDGES = [
    [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],[0,17],
];
function drawHands(hs) {
    skelCtx.lineWidth = 2; skelCtx.strokeStyle = 'rgba(255,200,50,.85)'; skelCtx.fillStyle = '#ff4444';
    hs.forEach(h => {
        HAND_EDGES.forEach(([i, j]) => {
            skelCtx.beginPath(); skelCtx.moveTo(h[i][0], h[i][1]); skelCtx.lineTo(h[j][0], h[j][1]); skelCtx.stroke();
        });
        h.forEach(p => { skelCtx.beginPath(); skelCtx.arc(p[0], p[1], 3, 0, 2 * Math.PI); skelCtx.fill(); });
    });
}

// Apply the mirror via CSS so coords don't need flipping
skelCanvas.style.transform = 'scaleX(-1)';

// ── Init ─────────────────────────────────────────────────────────────────
showNext();

})();
