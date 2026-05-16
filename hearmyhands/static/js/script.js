// ── Reveal au scroll ─────────────────────────────────────────────────────────
const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('active');
        observer.unobserve(entry.target);
    });
}, { threshold: 0.15, rootMargin: '0px 0px -50px 0px' });
document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));


// ── Menu mobile ──────────────────────────────────────────────────────────────
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const navLinks      = document.getElementById('navLinks');
if (mobileMenuBtn && navLinks) {
    const icon = mobileMenuBtn.querySelector('i');
    const toggle = (open) => {
        navLinks.classList.toggle('active', open);
        icon.classList.toggle('fa-bars', !open);
        icon.classList.toggle('fa-xmark', open);
    };
    mobileMenuBtn.addEventListener('click', () => toggle(!navLinks.classList.contains('active')));
    navLinks.querySelectorAll('a').forEach(link => link.addEventListener('click', () => toggle(false)));
}


// ── Logique de traduction (uniquement sur /translate) ────────────────────────
const video = document.getElementById('videoElement');
if (video) initTranslate();

function initTranslate() {
    const skeletonCanvas = document.getElementById('skeletonCanvas');
    const skelCtx        = skeletonCanvas.getContext('2d');
    const videoContainer = video.closest('.video-container');

    const startBtn         = document.getElementById('startBtn');
    const togglePredBtn    = document.getElementById('togglePredBtn');
    const clearBtn         = document.getElementById('clearBtn');
    const wordHistoryEl    = document.getElementById('wordHistory');
    const currentLetterEl  = document.getElementById('currentLetter');
    const statusDot        = document.querySelector('.dot');

    // ── Tuning du transport WebSocket ────────────────────────────────────────
    const SEND_WIDTH       = 480;   // downscale avant envoi (économise la bande passante)
    const JPEG_QUALITY     = 0.7;   // qualité JPEG (0..1)
    const SEND_INTERVAL_MS = 66;    // ~15 fps max
    const ACK_TIMEOUT_MS   = 2000;  // libère l'envoi si le serveur ne répond pas

    // ── Tuning de la reconnaissance de lettres ───────────────────────────────
    const MIN_LETTER_CONFIDENCE = 0.6;  // sous ce seuil, on ignore la prédiction
    const STABLE_FRAMES_TO_COMMIT = 10; // n frames identiques avant d'ajouter au mot

    let lastLetter  = null;
    let stableCount = 0;

    // Canvas offscreen réutilisé pour l'encodage
    const encodeCanvas = document.createElement('canvas');
    const encodeCtx    = encodeCanvas.getContext('2d');

    let isPredicting = false;
    let inFlight     = false;
    let sendTimer    = null;
    let waitTimer    = null;

    // ── Socket.IO ────────────────────────────────────────────────────────────
    const socket = io();
    socket.on('connect',       () => console.log('Socket.IO connected:', socket.id));
    socket.on('connect_error', (e) => console.error('Socket.IO error:', e));

    // ── Caméra ───────────────────────────────────────────────────────────────
    startBtn.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            video.classList.add('active');
            startBtn.style.display = 'none';
            togglePredBtn.disabled = false;
            statusDot.classList.add('active');
        } catch (err) {
            console.error(err);
            alert("Impossible d'accéder à la webcam.");
        }
    });

    // ── Traduction ON/OFF ────────────────────────────────────────────────────
    togglePredBtn.addEventListener('click', () => {
        isPredicting = !isPredicting;
        if (isPredicting) {
            togglePredBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Arrêter';
            togglePredBtn.classList.add('stop-mode');
            sendTimer = setInterval(sendFrame, SEND_INTERVAL_MS);
        } else {
            togglePredBtn.innerHTML = '<i class="fa-solid fa-play"></i> Reprendre';
            togglePredBtn.classList.remove('stop-mode');
            clearInterval(sendTimer);
            sendTimer = null;
            clearSkeleton();
        }
    });

    clearBtn.addEventListener('click', () => {
        wordHistoryEl.textContent = '...';
        lastLetter = null;
        stableCount = 0;
    });

    // ── Envoi d'une frame en binaire ─────────────────────────────────────────
    function sendFrame() {
        if (!video.srcObject || inFlight) return;
        const vw = video.videoWidth, vh = video.videoHeight;
        if (!vw || !vh) return;

        const sendW = SEND_WIDTH;
        const sendH = Math.round(vh * SEND_WIDTH / vw);
        if (encodeCanvas.width !== sendW || encodeCanvas.height !== sendH) {
            encodeCanvas.width  = sendW;
            encodeCanvas.height = sendH;
        }
        encodeCtx.drawImage(video, 0, 0, sendW, sendH);

        inFlight = true;
        encodeCanvas.toBlob(async (blob) => {
            if (!blob) { inFlight = false; return; }
            const buf = await blob.arrayBuffer();

            // Timeout de sécurité au cas où le serveur ne répond pas
            waitTimer = setTimeout(() => { inFlight = false; }, ACK_TIMEOUT_MS);

            socket.emit('frame', buf, (response) => {
                clearTimeout(waitTimer);
                inFlight = false;
                if (isPredicting) applyPrediction(response, sendW, sendH);
            });
        }, 'image/jpeg', JPEG_QUALITY);
    }

    // ── Affichage de la prédiction ───────────────────────────────────────────
    function applyPrediction(data, w, h) {
        if (!data) return;
        if (skeletonCanvas.width !== w || skeletonCanvas.height !== h) {
            skeletonCanvas.width  = w;
            skeletonCanvas.height = h;
        }
        alignCanvasWithVideo();
        skelCtx.clearRect(0, 0, w, h);
        if (data.skeleton && data.skeleton.length >= 9) drawSkeleton(data.skeleton);
        if (data.hands && data.hands.length)            drawHands(data.hands);

        const conf = data.confidence ?? 0;
        const letter = (data.letter && conf >= MIN_LETTER_CONFIDENCE) ? data.letter : null;
        updateLetter(letter);
    }

    // Le conteneur vidéo est plus large que l'image webcam (object-fit: contain),
    // donc on positionne le canvas exactement sur la zone vidéo affichée.
    function alignCanvasWithVideo() {
        const vw = video.videoWidth, vh = video.videoHeight;
        const cw = video.clientWidth, ch = video.clientHeight;
        if (!vw || !vh || !cw || !ch) return;
        const videoRatio = vw / vh;
        const boxRatio   = cw / ch;
        let dispW, dispH, dispX, dispY;
        if (videoRatio > boxRatio) {
            dispW = cw;            dispH = cw / videoRatio;
            dispX = 0;             dispY = (ch - dispH) / 2;
        } else {
            dispH = ch;            dispW = ch * videoRatio;
            dispX = (cw - dispW) / 2; dispY = 0;
        }
        skeletonCanvas.style.left   = `${dispX}px`;
        skeletonCanvas.style.top    = `${dispY}px`;
        skeletonCanvas.style.width  = `${dispW}px`;
        skeletonCanvas.style.height = `${dispH}px`;
    }
    window.addEventListener('resize', alignCanvasWithVideo);
    video.addEventListener('loadedmetadata', () => {
        // Adapte le container à l'aspect réel de la webcam (4:3, 16:9, …)
        if (videoContainer && video.videoWidth && video.videoHeight) {
            videoContainer.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
        }
        alignCanvasWithVideo();
    });

    function updateLetter(letter) {
        if (currentLetterEl) currentLetterEl.textContent = letter ?? '-';
        if (letter && letter === lastLetter) {
            stableCount++;
            if (stableCount === STABLE_FRAMES_TO_COMMIT) {
                const cur = wordHistoryEl.textContent === '...' ? '' : wordHistoryEl.textContent;
                wordHistoryEl.textContent = cur + letter;
            }
        } else {
            lastLetter = letter;
            stableCount = letter ? 1 : 0;
        }
    }

    function clearSkeleton() {
        skelCtx.clearRect(0, 0, skeletonCanvas.width, skeletonCanvas.height);
        if (currentLetterEl) currentLetterEl.textContent = '-';
        lastLetter = null;
        stableCount = 0;
    }

    // ── Dessin du squelette ──────────────────────────────────────────────────
    const EDGES = [[0,1],[0,2],[1,3],[3,5],[2,4],[4,6],[1,7],[2,8],[7,8]];
    const KP_COLORS = { neck: 'white', left: 'cyan', right: '#ff5078', body: '#32ff78' };
    const VIS_THRESHOLD = 0.3;

    function getKpColor(idx) {
        if (idx === 0) return KP_COLORS.neck;
        if (idx === 1 || idx === 3 || idx === 5) return KP_COLORS.left;
        if (idx === 2 || idx === 4 || idx === 6) return KP_COLORS.right;
        return KP_COLORS.body;
    }

    function drawSkeleton(keypoints) {
        skelCtx.lineWidth = 4;
        EDGES.forEach(([i, j]) => {
            const p1 = keypoints[i], p2 = keypoints[j];
            if (p1[2] < VIS_THRESHOLD || p2[2] < VIS_THRESHOLD) return;
            skelCtx.strokeStyle = getKpColor(i);
            skelCtx.beginPath();
            skelCtx.moveTo(p1[0], p1[1]);
            skelCtx.lineTo(p2[0], p2[1]);
            skelCtx.stroke();
        });

        skelCtx.lineWidth = 2;
        skelCtx.strokeStyle = 'rgba(0,0,0,0.6)';
        keypoints.forEach((p, idx) => {
            if (p[2] < VIS_THRESHOLD) return;
            skelCtx.fillStyle = getKpColor(idx);
            skelCtx.beginPath();
            skelCtx.arc(p[0], p[1], 6, 0, 2 * Math.PI);
            skelCtx.fill();
            skelCtx.stroke();
        });
    }

    // ── Dessin des mains (MediaPipe) ─────────────────────────────────────────
    const HAND_CONNECTIONS = [
        [0,1],[1,2],[2,3],[3,4],
        [0,5],[5,6],[6,7],[7,8],
        [5,9],[9,10],[10,11],[11,12],
        [9,13],[13,14],[14,15],[15,16],
        [13,17],[17,18],[18,19],[19,20],
        [0,17],
    ];

    function drawHands(hands) {
        skelCtx.lineWidth = 2;
        skelCtx.strokeStyle = 'rgba(255, 200, 50, 0.85)';
        skelCtx.fillStyle = '#ff4444';
        hands.forEach(hand => {
            HAND_CONNECTIONS.forEach(([i, j]) => {
                const p1 = hand[i], p2 = hand[j];
                skelCtx.beginPath();
                skelCtx.moveTo(p1[0], p1[1]);
                skelCtx.lineTo(p2[0], p2[1]);
                skelCtx.stroke();
            });
            hand.forEach(p => {
                skelCtx.beginPath();
                skelCtx.arc(p[0], p[1], 3, 0, 2 * Math.PI);
                skelCtx.fill();
            });
        });
    }
}
