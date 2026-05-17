// HMH Quiz — mode reconnaissance (3 sous-modes : hardcore / 10sec / survival)
// Leaderboards locaux (localStorage), top 10 par mode.
(function () {

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWY Z".replace(/\s/g, "").split("");
// X n'est pas dans le dataset vidéo — on l'exclut.
const LETTERS_WITH_VIDEO = LETTERS.filter(l => l !== 'X');

const STORE_PSEUDO = 'hmh-quiz-pseudo';
const STORE_LB     = (mode) => `hmh-quiz-lb-${mode}`;
const LB_LIMIT     = 10;

// ─── Utils localStorage ───────────────────────────────────────────────────
function loadLB(mode) {
    try { return JSON.parse(localStorage.getItem(STORE_LB(mode))) || []; }
    catch (e) { return []; }
}
function saveLB(mode, entries) {
    try { localStorage.setItem(STORE_LB(mode), JSON.stringify(entries)); } catch (e) {}
}
// Pour Hardcore/10sec : score = nombre de bonnes réponses. Plus haut = mieux.
// Pour Survival     : score = nombre de lettres survies. Plus haut = mieux.
// On stocke aussi un timestamp pour départager les ex aequo.
function pushScore(mode, pseudo, score, extra) {
    const lb = loadLB(mode);
    lb.push({ pseudo: pseudo || 'Anonyme', score, ts: Date.now(), ...(extra || {}) });
    lb.sort((a, b) => (b.score - a.score) || (a.ts - b.ts));
    const trimmed = lb.slice(0, LB_LIMIT);
    saveLB(mode, trimmed);
    return trimmed.findIndex(e => e.ts === Date.now()
                                 || (e.score === score && e.pseudo === (pseudo || 'Anonyme')));
}

function getPseudo() {
    return localStorage.getItem(STORE_PSEUDO) || '';
}
function setPseudo(p) {
    if (p) localStorage.setItem(STORE_PSEUDO, p.trim().slice(0, 20));
}

// ─── Page d'accueil quiz (sélection mode + leaderboards) ──────────────────
function initLanding() {
    const pseudoInput = document.getElementById('quizPseudo');
    if (pseudoInput) {
        pseudoInput.value = getPseudo();
        pseudoInput.addEventListener('input', () => setPseudo(pseudoInput.value));
        pseudoInput.addEventListener('blur',  () => setPseudo(pseudoInput.value));
    }

    // Best score par mode sur les cartes
    ['hardcore', '10sec', 'survival'].forEach(mode => {
        const el = document.querySelector(`[data-mode-best="${mode}"]`);
        if (!el) return;
        const lb = loadLB(mode);
        if (!lb.length) { el.textContent = ''; return; }
        const top = lb[0];
        const unit = mode === 'survival' ? 'lettres' : '/ 10';
        el.innerHTML = `Meilleur : <strong>${top.score} ${unit}</strong> · ${escapeHtml(top.pseudo)}`;
    });

    // Leaderboards
    renderLB('hardcore', 'lbHardcore', s => `${s} / 10`);
    renderLB('10sec',    'lb10sec',    s => `${s} / 10`);
    renderLB('survival', 'lbSurvival', s => `${s} lettres`);
}

function renderLB(mode, listId, fmt) {
    const ol = document.getElementById(listId);
    if (!ol) return;
    const lb = loadLB(mode);
    ol.innerHTML = '';
    if (!lb.length) {
        ol.innerHTML = '<li class="quiz-lb-empty">Aucun score encore — sois le premier !</li>';
        return;
    }
    lb.forEach((e, i) => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span class="quiz-lb-rank">#${i + 1}</span>
            <span class="quiz-lb-pseudo">${escapeHtml(e.pseudo)}</span>
            <span class="quiz-lb-score">${fmt(e.score)}</span>
        `;
        ol.appendChild(li);
    });
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => (
        {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
    ));
}

// ─── Jeu ──────────────────────────────────────────────────────────────────
function initGame() {
    const mode = window.HMH_QUIZ_MODE;
    if (!mode) return;

    const modeLabel    = document.getElementById('modeLabel');
    const scoreBox     = document.getElementById('scoreBox');
    const introCard    = document.getElementById('quizIntro');
    const introTitle   = document.getElementById('introTitle');
    const introRules   = document.getElementById('introRules');
    const startBtn     = document.getElementById('startGameBtn');
    const gameCard     = document.getElementById('quizGame');
    const frame        = document.getElementById('quizFrame');
    const blackoutEl   = document.getElementById('quizBlackout');
    const inputEl      = document.getElementById('quizInput');
    const submitBtn    = document.getElementById('quizSubmit');
    const feedbackEl   = document.getElementById('quizFeedback');
    const timerFill    = document.getElementById('quizTimerFill');
    const timerText    = document.getElementById('quizTimerText');
    const endCard      = document.getElementById('quizEnd');
    const endTitle     = document.getElementById('endTitle');
    const endSub       = document.getElementById('endSub');
    const endRank      = document.getElementById('endRank');
    const playAgainBtn = document.getElementById('playAgainBtn');

    // Config par mode
    const MODES = {
        hardcore: {
            label:  'Hardcore',
            title:  'Mode Hardcore',
            rules:  'La vidéo joue une seule fois (≈ 2 s). Quand elle se termine, l\'écran devient noir et tu as 3 secondes pour taper la lettre. 10 questions au total.',
            total:  10,
            // Pour hardcore : on n'a pas un seul timer "depuis le start", c'est
            // séquencé : vidéo joue (~2s) puis 3s d'input. On le gère dans
            // startQuestion (cas particulier).
        },
        '10sec': {
            label:  '10 sec',
            title:  'Mode 10 secondes',
            rules:  'La vidéo boucle pendant 10 secondes. Tu peux répondre à tout moment dans ces 10 secondes. 10 questions au total.',
            total:  10,
            time:   10,
        },
        survival: {
            label:  'Survie',
            title:  'Mode Survie',
            rules:  'Illimité. Tu commences avec 15 s par lettre, ça diminue progressivement. Quand le temps passe sous la durée de la vidéo, la lecture accélère jusqu\'à ×3. Tu rates → game over. Vise les 40 !',
            total:  Infinity,
        },
    };
    const cfg = MODES[mode];
    if (!cfg) { window.location.href = '/learn/quiz'; return; }

    modeLabel.textContent = cfg.label;
    introTitle.textContent = cfg.title;
    introRules.textContent = cfg.rules;

    let score = 0;
    let qIndex = 0;
    let currentLetter = null;
    let questionTimerId = null;
    let videoEndedHandler = null;
    // Pour le mode hardcore, on a deux phases (video puis input). 'phase'
    // décrit où on en est dans la question courante.
    let phase = null;  // 'video' | 'input' (hardcore), 'live' (10sec/survival)
    let videoDurationSec = 2;  // valeur estimée, mise à jour quand la vidéo charge
    let questionStartTs = 0;
    let phaseDurationMs = 0;
    let rafTimerId = null;

    // ─── Sélection lettre ─────────────────────────────────────────────────
    function pickLetter() {
        // évite la même lettre deux fois de suite
        let l;
        do { l = LETTERS_WITH_VIDEO[Math.floor(Math.random() * LETTERS_WITH_VIDEO.length)]; }
        while (l === currentLetter);
        return l;
    }

    // ─── Démarrage ────────────────────────────────────────────────────────
    startBtn.addEventListener('click', () => {
        introCard.hidden = true;
        gameCard.hidden = false;
        score = 0; qIndex = 0;
        updateScoreBox();
        nextQuestion();
        inputEl.focus();
    });

    function updateScoreBox() {
        if (mode === 'survival') {
            scoreBox.textContent = `${score} lettres`;
        } else {
            scoreBox.textContent = `${score} / ${qIndex}`;
        }
    }

    // ─── Boucle question ──────────────────────────────────────────────────
    function nextQuestion() {
        if (mode !== 'survival' && qIndex >= cfg.total) {
            return endGame();
        }
        currentLetter = pickLetter();
        qIndex++;
        feedbackEl.textContent = '';
        feedbackEl.className = 'quiz-feedback';
        inputEl.value = '';
        inputEl.disabled = false;
        blackoutEl.hidden = true;
        // Options selon le mode
        if (mode === 'hardcore') {
            loadQuizVideo(currentLetter, { once: true, speed: 1 });
        } else if (mode === 'survival') {
            loadQuizVideo(currentLetter, { speed: survivalSpeed() });
        } else {
            loadQuizVideo(currentLetter, { speed: 1 });
        }
        updateScoreBox();

        if (mode === 'hardcore') {
            startHardcoreQuestion();
        } else if (mode === '10sec') {
            startTimedQuestion(cfg.time * 1000);
        } else {
            startSurvivalQuestion();
        }
        inputEl.focus();
    }

    // Charge l'iframe vidéo. Le hash transmet les params au wrapper :
    //   #once          -> pas de loop, lecture unique (mode hardcore)
    //   #speed=N       -> playbackRate (mode survie)
    function loadQuizVideo(letter, opts) {
        const parts = ['quiz'];
        if (opts && opts.once) parts.push('once');
        if (opts && opts.speed && opts.speed !== 1) parts.push(`speed=${opts.speed}`);
        const url = `/learn/play/${letter}#${parts.join('&')}`;
        try {
            const w = frame.contentWindow;
            if (w && w.location) { w.location.replace(url); return; }
        } catch (e) {}
        frame.src = url;
    }

    // Écoute le message 'video_ended' du wrapper (envoyé seulement quand
    // la vidéo s'arrête pour de bon en mode #once).
    window.addEventListener('message', (ev) => {
        const d = ev.data;
        if (!d || typeof d !== 'object') return;
        if (d.hmh === 'video_ended' && phase === 'video' && mode === 'hardcore') {
            // Passe en phase input + écran noir
            triggerHardcoreInputPhase();
        }
    });

    // ─── Mode HARDCORE ────────────────────────────────────────────────────
    // Phase 1 : la vidéo joue UNE SEULE FOIS (#once). On attend le message
    //           'video_ended' du wrapper (déclenché par l'event 'ended').
    //           Filet de sécurité : timeout à 4s si le message n'arrive pas.
    // Phase 2 : écran noir + 3 s pour taper la lettre.
    function startHardcoreQuestion() {
        phase = 'video';
        // La barre de timer reste à 100% pendant la lecture (pas de countdown)
        timerFill.style.width = '100%';
        timerFill.style.background = 'linear-gradient(90deg,#4A90E2,#50E3C2)';
        timerText.textContent = '...';
        // Safety net si video_ended ne vient pas (rare, vidéo corrompue/etc.)
        clearTimeout(questionTimerId);
        questionTimerId = setTimeout(() => {
            if (phase === 'video') triggerHardcoreInputPhase();
        }, 4000);
    }

    function triggerHardcoreInputPhase() {
        if (phase !== 'video') return;
        phase = 'input';
        blackoutEl.hidden = false;
        try { frame.contentWindow.postMessage({hmh: 'pause'}, '*'); } catch (e) {}
        clearTimeout(questionTimerId);
        // 3 secondes pour répondre
        questionStartTs = performance.now();
        phaseDurationMs = 3000;
        startTimerBar();
        questionTimerId = setTimeout(() => onTimeout(), 3000);
        inputEl.focus();
    }

    // ─── Mode 10 sec (et hardcore phase 2) ────────────────────────────────
    function startTimedQuestion(ms) {
        phase = 'live';
        questionStartTs = performance.now();
        phaseDurationMs = ms;
        startTimerBar();
        clearTimeout(questionTimerId);
        questionTimerId = setTimeout(() => onTimeout(), ms);
    }

    // ─── Mode SURVIE ──────────────────────────────────────────────────────
    function survivalTimeMs() {
        // Décroissance progressive : 15 s à la question 1, 1 s asymptotiquement.
        // f(n) = 1 + 14 * exp(-n/12)  (12 réglable, doux jusqu'à ~40)
        const n = qIndex - 1;
        const sec = 1 + 14 * Math.exp(-n / 12);
        return Math.max(1000, Math.round(sec * 1000));
    }
    function survivalSpeed() {
        // Quand le temps imparti est < durée vidéo, on accélère la vidéo
        // pour que ça défile en entier dans le temps. Cap ×3.
        const availSec = survivalTimeMs() / 1000;
        if (availSec >= videoDurationSec) return 1.0;
        return Math.min(3.0, videoDurationSec / availSec);
    }
    function startSurvivalQuestion() {
        startTimedQuestion(survivalTimeMs());
    }

    // ─── Barre de timer (RAF) ─────────────────────────────────────────────
    function startTimerBar() {
        cancelAnimationFrame(rafTimerId);
        const tick = () => {
            const elapsed = performance.now() - questionStartTs;
            const remainMs = Math.max(0, phaseDurationMs - elapsed);
            const ratio = phaseDurationMs > 0 ? (remainMs / phaseDurationMs) : 0;
            timerFill.style.width = `${ratio * 100}%`;
            timerText.textContent = (remainMs / 1000).toFixed(1);
            // Couleur rouge < 25%
            timerFill.style.background = ratio < 0.25
                ? 'linear-gradient(90deg,#ff5078,#ff8a4d)'
                : 'linear-gradient(90deg,#4A90E2,#50E3C2)';
            if (remainMs > 0) rafTimerId = requestAnimationFrame(tick);
        };
        rafTimerId = requestAnimationFrame(tick);
    }

    // ─── Soumission ───────────────────────────────────────────────────────
    function submitAnswer() {
        const v = (inputEl.value || '').trim().toUpperCase();
        if (!v) return;
        if (!/^[A-Z]$/.test(v)) {
            feedbackEl.textContent = 'Une seule lettre A–Z';
            feedbackEl.className = 'quiz-feedback bad';
            return;
        }
        onAnswerGiven(v);
    }
    submitBtn.addEventListener('click', submitAnswer);
    inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); submitAnswer(); }
    });
    // Auto-submit dès qu'une lettre est tapée (UX rapide)
    inputEl.addEventListener('input', () => {
        const v = (inputEl.value || '').trim().toUpperCase();
        inputEl.value = v;
        if (/^[A-Z]$/.test(v)) {
            // petit delay pour que le focus reste responsive
            setTimeout(submitAnswer, 50);
        }
    });

    function onAnswerGiven(letter) {
        const correct = letter === currentLetter;
        clearTimeout(questionTimerId);
        cancelAnimationFrame(rafTimerId);
        inputEl.disabled = true;

        if (correct) {
            score++;
            feedbackEl.textContent = `✓ Bien joué, c'était ${currentLetter}.`;
            feedbackEl.className = 'quiz-feedback ok';
        } else {
            feedbackEl.textContent = `✗ Raté, c'était ${currentLetter}.`;
            feedbackEl.className = 'quiz-feedback bad';
        }
        updateScoreBox();

        if (mode === 'survival' && !correct) {
            // En survie une erreur termine la partie
            return setTimeout(endGame, 900);
        }
        setTimeout(() => nextQuestion(), 700);
    }

    function onTimeout() {
        cancelAnimationFrame(rafTimerId);
        inputEl.disabled = true;
        feedbackEl.textContent = `⏱ Trop tard, c'était ${currentLetter}.`;
        feedbackEl.className = 'quiz-feedback bad';
        if (mode === 'survival') {
            return setTimeout(endGame, 900);
        }
        setTimeout(() => nextQuestion(), 700);
    }

    // ─── Fin de partie ────────────────────────────────────────────────────
    function endGame() {
        clearTimeout(questionTimerId);
        cancelAnimationFrame(rafTimerId);
        gameCard.hidden = true;
        endCard.hidden = false;

        const pseudo = getPseudo() || 'Anonyme';
        const totalLabel = mode === 'survival' ? 'lettres' : '/ 10';
        endTitle.textContent = mode === 'survival'
            ? `Tu as tenu ${score} lettres`
            : `Score : ${score} / 10`;
        endSub.textContent  = `Mode ${cfg.label} · pseudo "${pseudo}"`;

        // Save + rank
        const lb = loadLB(mode);
        const entry = { pseudo, score, ts: Date.now() };
        lb.push(entry); lb.sort((a, b) => (b.score - a.score) || (a.ts - b.ts));
        const trimmed = lb.slice(0, LB_LIMIT);
        saveLB(mode, trimmed);
        const rank = trimmed.findIndex(e => e.ts === entry.ts) + 1;
        if (rank > 0 && rank <= LB_LIMIT) {
            endRank.innerHTML = `<i class="fa-solid fa-medal"></i> Nouveau top ! Rang <strong>#${rank}</strong>`;
            endRank.className = 'quiz-end-rank good';
        } else {
            endRank.innerHTML = `Pas dans le top ${LB_LIMIT} cette fois.`;
            endRank.className = 'quiz-end-rank';
        }
    }

    playAgainBtn.addEventListener('click', () => {
        endCard.hidden = true;
        introCard.hidden = false;
    });
}

// ─── Détection page courante ──────────────────────────────────────────────
if (document.getElementById('quizGame')) {
    initGame();
} else if (document.querySelector('.quiz-modes')) {
    initLanding();
}

})();
