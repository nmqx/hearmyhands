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
    const placeholderEl  = videoContainer ? videoContainer.querySelector('.video-placeholder') : null;

    const startBtn         = document.getElementById('startBtn');
    const togglePredBtn    = document.getElementById('togglePredBtn');
    const clearBtn         = document.getElementById('clearBtn');
    const modeBtn          = document.getElementById('modeBtn');
    const wordHistoryEl    = document.getElementById('wordHistory');
    const currentLetterEl  = document.getElementById('currentLetter');
    const currentLabelEl   = document.getElementById('currentLabel');
    const statusDot        = document.querySelector('.dot');

    // ── Tuning du transport WebSocket ────────────────────────────────────────
    const TARGET_FPS       = 30;   // le serveur CPU ne sustain pas plus, inutile de gaspiller
    const SEND_WIDTH       = 480;                  // downscale avant envoi
    const JPEG_QUALITY     = 0.7;
    const SEND_INTERVAL_MS = 1000 / TARGET_FPS;    // ~16 ms (60 fps)
    const MAX_IN_FLIGHT    = 4;                    // pipeline: max N frames en vol simultanément
    const ACK_TIMEOUT_MS   = 2000;                 // libère un slot si le serveur ne répond pas

    // ── Tuning de la reconnaissance de lettres ───────────────────────────────
    const MIN_LETTER_CONFIDENCE = 0.6;  // sous ce seuil, on ignore la prédiction
    const STABLE_FRAMES_TO_COMMIT = 10; // n frames identiques avant d'ajouter au mot
    const MIN_SIGN_CONFIDENCE = 0.7;    // GRU temporel (Ocarina), plus exigeant
    const SIGN_COOLDOWN_MS = 1500;      // anti-doublons sur le signe temporel

    let lastLetter  = null;
    let stableCount = 0;
    let lastSign      = null;
    let lastSignTime  = 0;

    // 'static' = lettre par frame via MLP. 'dynamic' = signe temporel via GRU Ocarina.
    let mode = 'static';

    // Canvas offscreen réutilisé pour l'encodage
    const encodeCanvas = document.createElement('canvas');
    const encodeCtx    = encodeCanvas.getContext('2d');

    let isPredicting    = false;
    let inFlightCount   = 0;
    let frameSeq        = 0;
    let lastAppliedSeq  = -1;
    let sendTimer       = null;

    // ── Socket.IO ────────────────────────────────────────────────────────────
    const socket = io();
    socket.on('connect',       () => console.log('Socket.IO connected:', socket.id));
    socket.on('connect_error', (e) => console.error('Socket.IO error:', e));

    // ── Caméra ───────────────────────────────────────────────────────────────
    startBtn.addEventListener('click', async () => {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("getUserMedia indisponible — ouvre le site dans Safari ou Chrome (pas dans un navigateur in-app type Instagram/Discord).");
            return;
        }
        try {
            // Sur iOS, min:30 lève OverconstrainedError si la cam ne peut pas garantir
            // 30 fps — on reste sur des contraintes "ideal" uniquement (best-effort).
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    frameRate: { ideal: TARGET_FPS },
                    width:     { ideal: 1280 },
                    height:    { ideal: 720 },
                    facingMode: 'user',
                },
            });
            video.srcObject = stream;
            // iOS exige playsinline + autoplay; on force play() au cas où
            try { await video.play(); } catch (_) {}
            video.classList.add('active');
            startBtn.style.display = 'none';
            togglePredBtn.disabled = false;
            statusDot.classList.add('active');
            if (placeholderEl) placeholderEl.style.display = 'none';

            // Attend les dimensions réelles puis ajuste container + canvas
            await new Promise(r => {
                if (video.videoWidth) return r();
                video.addEventListener('loadedmetadata', r, { once: true });
            });
            if (videoContainer && video.videoWidth && video.videoHeight) {
                videoContainer.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
            }
            alignCanvasWithVideo();
        } catch (err) {
            console.error('getUserMedia failed:', err);
            const map = {
                NotAllowedError: "Permission caméra refusée. Va dans Réglages → Safari → Caméra et autorise pour ce site.",
                NotFoundError:   "Aucune caméra trouvée sur l'appareil.",
                NotReadableError:"La caméra est déjà utilisée par une autre app. Ferme les autres apps qui s'en servent.",
                OverconstrainedError:"La caméra ne supporte pas les contraintes demandées.",
                SecurityError:   "Accès caméra bloqué (contexte non sécurisé ?). Vérifie que tu es bien en https://.",
            };
            const msg = map[err.name] || ("Impossible d'accéder à la webcam : " + (err.name || err.message));
            alert(msg);
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
        lastLetter   = null;
        stableCount  = 0;
        lastSign     = null;
        lastSignTime = 0;
    });

    // ── Bascule Statique / Dynamique ─────────────────────────────────────────
    function applyMode() {
        const isStatic = mode === 'static';
        modeBtn.innerHTML = isStatic
            ? '<i class="fa-solid fa-image"></i> Mode : Statique'
            : '<i class="fa-solid fa-clapperboard"></i> Mode : Dynamique';
        if (currentLabelEl) {
            currentLabelEl.textContent = isStatic ? 'Lettre détectée' : 'Signe détecté';
        }
        // Reset des accumulateurs pour éviter de mélanger les deux pipelines
        lastLetter = null; stableCount = 0;
        lastSign = null;   lastSignTime = 0;
        if (currentLetterEl) currentLetterEl.textContent = '-';
    }
    if (modeBtn) {
        modeBtn.addEventListener('click', () => {
            mode = (mode === 'static') ? 'dynamic' : 'static';
            applyMode();
        });
        applyMode();
    }

    // ── Envoi d'une frame en binaire ─────────────────────────────────────────
    // Pipeline: jusqu'à MAX_IN_FLIGHT frames en vol simultanément (backpressure).
    // Les ack peuvent revenir dans le désordre — on n'applique que la prédiction
    // la plus récente via un compteur de séquence.
    function sendFrame() {
        if (!video.srcObject || inFlightCount >= MAX_IN_FLIGHT) return;
        const vw = video.videoWidth, vh = video.videoHeight;
        if (!vw || !vh) return;

        const sendW = SEND_WIDTH;
        const sendH = Math.round(vh * SEND_WIDTH / vw);
        if (encodeCanvas.width !== sendW || encodeCanvas.height !== sendH) {
            encodeCanvas.width  = sendW;
            encodeCanvas.height = sendH;
        }
        encodeCtx.drawImage(video, 0, 0, sendW, sendH);

        const seq = ++frameSeq;
        inFlightCount++;
        const releaseTimer = setTimeout(() => {
            // Filet de sécurité: libère le slot si l'ack ne revient pas
            inFlightCount = Math.max(0, inFlightCount - 1);
        }, ACK_TIMEOUT_MS);

        encodeCanvas.toBlob(async (blob) => {
            if (!blob) {
                clearTimeout(releaseTimer);
                inFlightCount = Math.max(0, inFlightCount - 1);
                return;
            }
            const buf = await blob.arrayBuffer();
            socket.emit('frame', buf, (response) => {
                clearTimeout(releaseTimer);
                inFlightCount = Math.max(0, inFlightCount - 1);
                if (!isPredicting) return;
                // Seule la dernière prédiction (par seq) est appliquée
                if (seq <= lastAppliedSeq) return;
                lastAppliedSeq = seq;
                applyPrediction(response, sendW, sendH);
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

        if (mode === 'static') {
            const conf = data.confidence ?? 0;
            const letter = (data.letter && conf >= MIN_LETTER_CONFIDENCE) ? data.letter : null;
            updateLetter(letter);
        } else {
            // Dynamique : on n'affiche / n'accumule que les signes du GRU
            const sConf = data.sign_confidence ?? 0;
            if (currentLetterEl) currentLetterEl.textContent = data.sign ?? '-';
            if (data.sign && sConf >= MIN_SIGN_CONFIDENCE) {
                handleSign(data.sign);
            }
        }
    }

    function handleSign(sign) {
        const now = Date.now();
        if (sign === lastSign && now - lastSignTime < SIGN_COOLDOWN_MS) return;
        lastSign = sign;
        lastSignTime = now;
        const cur = wordHistoryEl.textContent === '...' ? '' : wordHistoryEl.textContent;
        wordHistoryEl.textContent = cur + sign;
        // Reset également le compteur "stable letter" pour éviter de doubler
        // immédiatement le signe via l'accumulation MLP.
        stableCount = 0;
    }

    function syncContainerAspect() {
        if (videoContainer && video.videoWidth && video.videoHeight) {
            const want = `${video.videoWidth} / ${video.videoHeight}`;
            if (videoContainer.style.aspectRatio !== want) {
                videoContainer.style.aspectRatio = want;
            }
        }
    }

    // Le canvas se cale exactement sur la zone vidéo réellement affichée
    // (object-fit: contain peut introduire des bandes si jamais l'aspect change).
    function alignCanvasWithVideo() {
        syncContainerAspect();
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
    ['loadedmetadata', 'loadeddata', 'playing', 'resize'].forEach(ev => {
        video.addEventListener(ev, alignCanvasWithVideo);
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
