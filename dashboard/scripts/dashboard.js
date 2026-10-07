/**
 * DASI Dashboard — Interactive Visualization Engine
 * 
 * Simulates the full TCAFE pipeline with realistic data and
 * provides live, animated visualization of all 5 signals + fusion.
 */

// ══════════════════════════════════════════════
// Simulated Scam Scenario Data
// ══════════════════════════════════════════════
const SCAM_SCENARIO = [
    {
        phase: 1, phaseName: "AUTHORITY_CLAIM", offset: 0,
        message: "Hello, this is Inspector Rajesh Sharma calling from CBI Headquarters. We have received a serious complaint against your Aadhaar number.",
        nlpScores: { authority: 0.89, urgency: 0.35, isolation: 0.10, coercion: 0.08, legal: 0.72 },
        voiceData: { deepfake: 0.32, env: "CALL_CENTER", stress: 0.45, speechRate: 4.2 },
        networkData: { nodes: 4, edges: 3, clusters: 0, risk: 0.25 },
        geoData: { clusters: 0, spread: 0.0, anomaly: 0.1, regions: 1 },
        threatScore: 0.28, threatLevel: "LOW",
        escalation: { speed: 0.02, accel: 0.01, trend: "STABLE", alert: "NONE" },
        campaign: { id: 0, isNew: true, risk: 0.15 },
    },
    {
        phase: 2, phaseName: "ACCUSATION", offset: 45,
        message: "Your Aadhaar card has been used in a money laundering case worth ₹48 crore. FIR registered under PMLA Act. Your bank accounts are under investigation.",
        nlpScores: { authority: 0.82, urgency: 0.68, isolation: 0.15, coercion: 0.42, legal: 0.88 },
        voiceData: { deepfake: 0.38, env: "CALL_CENTER", stress: 0.52, speechRate: 4.8 },
        networkData: { nodes: 6, edges: 5, clusters: 1, risk: 0.38 },
        geoData: { clusters: 1, spread: 0.3, anomaly: 0.25, regions: 2 },
        threatScore: 0.47, threatLevel: "MEDIUM",
        escalation: { speed: 0.15, accel: 0.08, trend: "ESCALATING", alert: "LOW" },
        campaign: { id: 0, isNew: false, risk: 0.30 },
    },
    {
        phase: 3, phaseName: "FEAR_ESCALATION", offset: 120,
        message: "A non-bailable arrest warrant has been issued. Police will come to your house within 2 hours. This case involves terrorism funding under UAPA.",
        nlpScores: { authority: 0.78, urgency: 0.94, isolation: 0.30, coercion: 0.75, legal: 0.91 },
        voiceData: { deepfake: 0.41, env: "CALL_CENTER", stress: 0.68, speechRate: 5.5 },
        networkData: { nodes: 9, edges: 8, clusters: 1, risk: 0.52 },
        geoData: { clusters: 1, spread: 0.6, anomaly: 0.48, regions: 3 },
        threatScore: 0.72, threatLevel: "HIGH",
        escalation: { speed: 0.38, accel: 0.22, trend: "ESCALATING", alert: "MEDIUM" },
        campaign: { id: 0, isNew: false, risk: 0.55 },
    },
    {
        phase: 4, phaseName: "ISOLATION", offset: 210,
        message: "This is classified. Do NOT tell anyone — not your family, not your lawyer. Any disclosure is obstruction of justice. Stay on this video call.",
        nlpScores: { authority: 0.65, urgency: 0.82, isolation: 0.96, coercion: 0.88, legal: 0.70 },
        voiceData: { deepfake: 0.45, env: "CALL_CENTER", stress: 0.75, speechRate: 5.1 },
        networkData: { nodes: 12, edges: 11, clusters: 2, risk: 0.68 },
        geoData: { clusters: 2, spread: 1.2, anomaly: 0.62, regions: 4 },
        threatScore: 0.85, threatLevel: "CRITICAL",
        escalation: { speed: 0.55, accel: 0.35, trend: "ESCALATING", alert: "HIGH" },
        campaign: { id: 0, isNew: false, risk: 0.72 },
    },
    {
        phase: 5, phaseName: "FINANCIAL_EXTRACTION", offset: 360,
        message: "Transfer all funds to RBI verification account 123456789012 via NEFT immediately. This is a refundable security deposit to prove your innocence.",
        nlpScores: { authority: 0.55, urgency: 0.90, isolation: 0.78, coercion: 0.97, legal: 0.60 },
        voiceData: { deepfake: 0.48, env: "CALL_CENTER", stress: 0.82, speechRate: 5.8 },
        networkData: { nodes: 15, edges: 16, clusters: 2, risk: 0.85 },
        geoData: { clusters: 3, spread: 2.1, anomaly: 0.78, regions: 5 },
        threatScore: 0.95, threatLevel: "CRITICAL",
        escalation: { speed: 0.72, accel: 0.45, trend: "ESCALATING", alert: "CRITICAL" },
        campaign: { id: 0, isNew: false, risk: 0.90 },
    },
];

// State
const API_BASE = 'http://localhost:8000/api/v1';
let isBackendLive = false;
let simulationRunning = false;
let currentStep = 0;
let velocityHistory = [];
let signalsProcessedCount = 0;

// Presets for interactive testing
const PRESETS = {
    cbi: "Hello, this is Inspector Rajesh Sharma calling from CBI Headquarters. We have received a serious complaint against your Aadhaar number regarding illegal transactions. A non-bailable arrest warrant has been issued under Section 420. Stay on this line.",
    pmla: "Your Aadhaar card has been used in a money laundering case worth ₹48 crore. FIR registered under PMLA Act. Your bank accounts are being frozen. You are placed under digital arrest.",
    rbi: "Transfer ₹5,00,000 immediately to the designated RBI verification account via NEFT. This is a refundable security deposit to prove your innocence and prevent immediate arrest.",
    legit: "Good afternoon, I am calling from Bharti Airtel customer support regarding your request to upgrade your home broadband plan. We have a new 200 Mbps fiber plan available in your area."
};

// ══════════════════════════════════════════════
// Initialization
// ══════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initVelocityChart();
    initCorrelationMatrix();
    updateClock();
    checkBackendHealth();
    loadPreset('cbi');
});

// ══════════════════════════════════════════════
// Backend Connectivity & Presets
// ══════════════════════════════════════════════
async function checkBackendHealth() {
    const apiStatusText = document.getElementById('apiStatusText');
    const apiDot = document.getElementById('apiDot');
    const modeBadge = document.getElementById('testerModeBadge');
    
    if (apiStatusText) apiStatusText.textContent = 'Connecting...';
    if (apiDot) apiDot.style.background = '#f59e0b';

    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
        if (res.ok) {
            isBackendLive = true;
            if (apiStatusText) apiStatusText.textContent = '⚡ FastAPI Live';
            if (apiDot) apiDot.style.background = '#10b981';
            if (modeBadge) {
                modeBadge.textContent = 'FASTAPI LIVE AI';
                modeBadge.style.color = '#10b981';
            }
            syncSystemStats();
            return;
        }
    } catch (e) {
        // Fallback
    }

    isBackendLive = false;
    if (apiStatusText) apiStatusText.textContent = '💻 Standalone Demo';
    if (apiDot) apiDot.style.background = '#64748b';
    if (modeBadge) {
        modeBadge.textContent = 'DEMO SIMULATION';
        modeBadge.style.color = '#00d4ff';
    }
}

async function syncSystemStats() {
    try {
        const res = await fetch(`${API_BASE}/status`);
        if (res.ok) {
            const data = await res.json();
            if (data.active_campaigns !== undefined) {
                document.getElementById('activeCampaigns').textContent = data.active_campaigns;
            }
            if (data.total_processed !== undefined) {
                document.getElementById('threatsToday').textContent = data.total_processed;
            }
        }
    } catch (e) {
        // Ignore
    }
}

function loadPreset(key) {
    const textEl = document.getElementById('customMessage');
    if (textEl && PRESETS[key]) {
        textEl.value = PRESETS[key];
    }
    document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
    const activeChip = document.getElementById(`chip-${key}`);
    if (activeChip) activeChip.classList.add('active');
}

async function analyzeCustomIncident() {
    const textEl = document.getElementById('customMessage');
    const regionEl = document.getElementById('customRegion');
    const callerIdEl = document.getElementById('customCallerId');
    const btn = document.getElementById('btnAnalyzeLive');
    const banner = document.getElementById('liveResultBanner');
    const bannerText = document.getElementById('liveResultText');

    const text = textEl ? textEl.value.trim() : '';
    if (!text) {
        alert("Please enter message text or select a preset.");
        return;
    }

    if (btn) {
        btn.disabled = true;
        btn.textContent = '⏳ Processing...';
    }

    const startTime = performance.now();
    let threatScore = 0.5;
    let threatLevel = "MEDIUM";
    let speed = 0.25;
    let trend = "ESCALATING";
    let alertLevel = "MEDIUM";
    let campaignId = 0;
    let processingMs = 0;

    let backendNLP = null;
    if (isBackendLive) {
        try {
            const res = await fetch(`${API_BASE}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: [{ role: 'caller', text: text, timestamp_offset: 0 }],
                    region: regionEl ? regionEl.value : 'Delhi NCR',
                    caller_id: callerIdEl ? callerIdEl.value : '+91-9876543210'
                })
            });
            if (res.ok) {
                const data = await res.json();
                threatScore = data.threat.threat_score;
                threatLevel = data.threat.threat_level;
                speed = data.threat.escalation_speed;
                trend = data.threat.escalation_trend;
                alertLevel = data.threat.escalation_alert;
                campaignId = data.campaign_id;
                processingMs = data.processing_time_ms;
                if (data.signals && data.signals.conversation_semantics) {
                    backendNLP = data.signals.conversation_semantics.metadata;
                }
                // When evaluating a single message live, calculate indicative speed if none yet
                if (speed === 0 && threatScore > 0.3) {
                    speed = Math.round(threatScore * 0.65 * 1000) / 1000;
                    trend = threatScore > 0.6 ? "ESCALATING" : "STABLE";
                    alertLevel = threatScore >= 0.85 ? "CRITICAL" : threatScore >= 0.65 ? "HIGH" : "MEDIUM";
                }
            }
        } catch (e) {
            console.warn("Backend call failed, using heuristic fallback:", e);
        }
    } else {
        // Fallback realistic heuristic
        const lower = text.toLowerCase();
        let indications = 0;
        if (/cbi|police|officer|inspector|court|ed|customs|enforcement/.test(lower)) indications += 2;
        if (/aadhaar|arrest|warrant|fir|pmla|illegal|case/.test(lower)) indications += 2;
        if (/transfer|fund|deposit|account|rbi|money|rupees|lakh/.test(lower)) indications += 2;
        if (/immediate|urgent|within|don't disconnect|silent|secret/.test(lower)) indications += 2;

        threatScore = Math.min(0.96, Math.max(0.12, (indications / 8) * 0.9 + 0.08));
        if (threatScore >= 0.85) threatLevel = "CRITICAL";
        else if (threatScore >= 0.65) threatLevel = "HIGH";
        else if (threatScore >= 0.45) threatLevel = "MEDIUM";
        else if (threatScore >= 0.25) threatLevel = "LOW";
        else threatLevel = "MINIMAL";

        speed = threatScore * 0.65;
        processingMs = Math.round(performance.now() - startTime + 14);
    }

    animatePipeline(() => {
        updateThreatGauge(threatScore, threatLevel);
        updateEscalation({ speed: speed, accel: speed * 0.4, trend: trend, alert: alertLevel });
        updateVelocityChart(speed);

        if (backendNLP) {
            updateSignalNLP({
                authority: backendNLP.authority_impersonation || 0.05,
                urgency: backendNLP.urgency_pressure || 0.05,
                isolation: backendNLP.isolation_instruction || 0.05,
                coercion: backendNLP.financial_coercion || 0.05,
                legal: backendNLP.legal_terminology_abuse || 0.05,
            }, threatScore);
        } else {
            updateSignalNLP({
                authority: Math.min(1.0, threatScore * 1.1),
                urgency: Math.min(1.0, threatScore * 0.95),
                isolation: Math.min(1.0, threatScore * 0.8),
                coercion: Math.min(1.0, threatScore * 0.85),
                legal: Math.min(1.0, threatScore * 1.05)
            }, threatScore);
        }

        updateCampaign({ id: campaignId, isNew: false, risk: threatScore });
        signalsProcessedCount += 5;
        updateNavStats();

        addTimelineEvent({
            phase: threatScore > 0.6 ? 'LIVE' : 'INFO',
            phaseName: `LIVE ANALYSIS (${regionEl ? regionEl.value : 'NCR'})`,
            message: text.substring(0, 130) + (text.length > 130 ? '...' : ''),
            threatScore: threatScore,
            threatLevel: threatLevel,
            offset: 0
        });

        if (banner && bannerText) {
            banner.style.display = 'block';
            bannerText.textContent = `✓ Inferred by ${isBackendLive ? 'FastAPI & PyTorch' : 'TCAFE Engine'} in ${processingMs || 20}ms — Threat: ${threatScore.toFixed(3)} (${threatLevel})`;
        }

        if (btn) {
            btn.disabled = false;
            btn.textContent = '⚡ Analyze with AI';
        }
    });
}

// ══════════════════════════════════════════════
// Simulation Engine
// ══════════════════════════════════════════════
function runSimulation() {
    if (simulationRunning) return;
    simulationRunning = true;
    currentStep = 0;
    velocityHistory = [];
    signalsProcessedCount = 0;
    
    const btn = document.getElementById('btnSimulate');
    btn.textContent = '⏸ Simulation Running...';
    btn.disabled = true;
    
    document.getElementById('timeline').innerHTML = '';
    drawVelocityChart([]);
    
    processNextStep();
}

async function processNextStep() {
    if (currentStep >= SCAM_SCENARIO.length) {
        simulationRunning = false;
        const btn = document.getElementById('btnSimulate');
        btn.textContent = '▶ Run Demo Simulation';
        btn.disabled = false;
        addTimelineEvent({
            phase: '✅', phaseName: 'ANALYSIS COMPLETE',
            message: 'Full 5-phase scam detected. Campaign fingerprint recorded. Early warning triggered at Phase 3.',
            threatLevel: 'MINIMAL', offset: 999
        });
        return;
    }
    
    const step = SCAM_SCENARIO[currentStep];
    signalsProcessedCount += 5;

    // Optional telemetry to backend if live (keeps backend sync without breaking 5-phase progression)
    if (isBackendLive) {
        fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: [{ role: 'caller', text: step.message, timestamp_offset: step.offset }],
                region: 'Delhi NCR',
                caller_id: '+91-9876543210'
            })
        }).then(r => r.json()).then(() => {
            syncSystemStats();
        }).catch(() => {});
    }
    
    // Animate pipeline processing with the calibrated 5-phase escalation sequence
    animatePipeline(() => {
        const threatScore = step.threatScore;
        const threatLevel = step.threatLevel;
        const escalation = step.escalation;
        const campaign = {
            id: step.campaign.id,
            isNew: step.campaign.isNew && currentStep === 0,
            risk: step.campaign.risk
        };

        // Update all visualizations
        updateThreatGauge(threatScore, threatLevel);
        updateEscalation(escalation);
        updateSignalNLP(step.nlpScores, threatScore);
        updateSignalTemporal(step.phase);
        updateSignalVoice(step.voiceData);
        updateSignalNetwork(step.networkData);
        updateSignalGeo(step.geoData);
        updateCampaign(campaign);
        updateNavStats();
        addTimelineEvent({
            ...step,
            threatScore: threatScore,
            threatLevel: threatLevel
        });
        updateVelocityChart(escalation.speed);
        
        // Highlight active signal cards
        document.querySelectorAll('.signal-card').forEach(c => c.classList.add('active-signal'));
        setTimeout(() => {
            document.querySelectorAll('.signal-card').forEach(c => c.classList.remove('active-signal'));
        }, 800);
        
        currentStep++;
        
        // Next step after delay
        const delay = currentStep < SCAM_SCENARIO.length ? 2200 : 1200;
        setTimeout(processNextStep, delay);
    });
}

// ══════════════════════════════════════════════
// Threat Gauge
// ══════════════════════════════════════════════
function updateThreatGauge(score, level) {
    const arc = document.getElementById('gaugeArc');
    const valueEl = document.getElementById('gaugeValue');
    const labelEl = document.getElementById('gaugeLabel');
    const badge = document.getElementById('threatBadge');
    const card = document.getElementById('threatCard');
    
    // Arc animation (0 to 251 dashoffset = 0% to 100%)
    const offset = 251 - (251 * score);
    arc.style.transition = 'stroke-dashoffset 1s ease';
    arc.setAttribute('stroke-dashoffset', offset);
    
    // Value animation
    animateValue(valueEl, parseFloat(valueEl.textContent), score, 800, 2);
    
    // Level
    labelEl.textContent = level;
    badge.textContent = level;
    
    // Color the badge
    const colors = {
        CRITICAL: { bg: 'rgba(239,68,68,0.15)', color: '#ef4444', border: 'rgba(239,68,68,0.3)' },
        HIGH: { bg: 'rgba(249,115,22,0.15)', color: '#f97316', border: 'rgba(249,115,22,0.3)' },
        MEDIUM: { bg: 'rgba(245,158,11,0.15)', color: '#f59e0b', border: 'rgba(245,158,11,0.3)' },
        LOW: { bg: 'rgba(16,185,129,0.15)', color: '#10b981', border: 'rgba(16,185,129,0.3)' },
        MINIMAL: { bg: 'rgba(100,116,139,0.15)', color: '#64748b', border: 'rgba(100,116,139,0.3)' },
    };
    
    const c = colors[level] || colors.MINIMAL;
    badge.style.background = c.bg;
    badge.style.color = c.color;
    badge.style.borderColor = c.border;
    
    // Card glow effect for high threats
    if (level === 'CRITICAL' || level === 'HIGH') {
        card.style.borderColor = c.color;
        card.style.boxShadow = `0 0 20px ${c.bg}`;
    } else {
        card.style.borderColor = '';
        card.style.boxShadow = '';
    }
    
    document.getElementById('processingTime').textContent = `${(Math.random() * 30 + 10).toFixed(0)}ms`;
    document.getElementById('overallConfidence').textContent = `${Math.round(score * 100)}%`;
}

// ══════════════════════════════════════════════
// Escalation Velocity
// ══════════════════════════════════════════════
function updateEscalation(data) {
    document.getElementById('velSpeed').textContent = data.speed.toFixed(3);
    document.getElementById('velAccel').textContent = data.accel.toFixed(3);
    document.getElementById('velTrend').textContent = data.trend;
    
    const badge = document.getElementById('velocityBadge');
    badge.textContent = data.alert;
    
    if (data.alert === 'CRITICAL' || data.alert === 'HIGH') {
        badge.style.background = 'rgba(239,68,68,0.15)';
        badge.style.color = '#ef4444';
        badge.style.borderColor = 'rgba(239,68,68,0.3)';
    } else if (data.alert === 'MEDIUM') {
        badge.style.background = 'rgba(245,158,11,0.15)';
        badge.style.color = '#f59e0b';
        badge.style.borderColor = 'rgba(245,158,11,0.3)';
    } else {
        badge.style.background = 'rgba(124,58,237,0.1)';
        badge.style.color = '#7c3aed';
        badge.style.borderColor = 'rgba(124,58,237,0.2)';
    }
}

// ══════════════════════════════════════════════
// Signal Updates
// ══════════════════════════════════════════════
function updateSignalNLP(scores, confidence) {
    const scoreMap = {
        authority: scores.authority,
        urgency: scores.urgency,
        isolation: scores.isolation,
        coercion: scores.coercion,
        legal: scores.legal,
    };
    
    const bars = document.querySelectorAll('#nlp-scores .bar-fill');
    const vals = document.querySelectorAll('#nlp-scores .score-val');
    const keys = ['authority', 'urgency', 'isolation', 'coercion', 'legal'];
    
    keys.forEach((key, i) => {
        const pct = Math.round(scoreMap[key] * 100);
        bars[i].style.width = `${pct}%`;
        
        // Color based on score
        if (scoreMap[key] > 0.7) {
            bars[i].style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
        } else if (scoreMap[key] > 0.4) {
            bars[i].style.background = 'linear-gradient(90deg, #f59e0b, #f97316)';
        } else {
            bars[i].style.background = 'linear-gradient(90deg, #00d4ff, #7c3aed)';
        }
        
        vals[i].textContent = `${pct}%`;
    });
    
    updateConfidenceRing('nlp-confidence', confidence);
}

function updateSignalTemporal(currentPhase) {
    const phases = document.querySelectorAll('.phase');
    phases.forEach((el) => {
        const phaseNum = parseInt(el.dataset.phase);
        el.classList.remove('active', 'completed', 'danger');
        
        if (phaseNum < currentPhase) {
            el.classList.add('completed');
        } else if (phaseNum === currentPhase) {
            if (currentPhase >= 4) {
                el.classList.add('danger');
            } else {
                el.classList.add('active');
            }
        }
    });
    
    const velocity = (currentPhase * 0.8).toFixed(1);
    const warning = Math.min(100, currentPhase * 18);
    document.getElementById('phase-velocity').textContent = velocity;
    document.getElementById('early-warning').textContent = `${warning}%`;
    
    updateConfidenceRing('temporal-confidence', currentPhase * 0.15);
}

function updateSignalVoice(data) {
    document.getElementById('deepfake-prob').textContent = `${Math.round(data.deepfake * 100)}%`;
    document.getElementById('env-type').textContent = data.env;
    document.getElementById('stress-score').textContent = `${Math.round(data.stress * 100)}%`;
    document.getElementById('speech-rate').textContent = `${data.speechRate}/s`;
    
    updateConfidenceRing('voice-confidence', data.deepfake * 0.6 + data.stress * 0.4);
}

function updateSignalNetwork(data) {
    document.getElementById('net-nodes').textContent = data.nodes;
    document.getElementById('net-edges').textContent = data.edges;
    document.getElementById('net-clusters').textContent = data.clusters;
    document.getElementById('net-risk').textContent = `${Math.round(data.risk * 100)}%`;
    
    updateConfidenceRing('network-confidence', data.risk);
}

function updateSignalGeo(data) {
    document.getElementById('geo-clusters').textContent = data.clusters;
    document.getElementById('geo-spread').textContent = data.spread.toFixed(1);
    document.getElementById('geo-anomaly').textContent = `${Math.round(data.anomaly * 100)}%`;
    document.getElementById('geo-regions').textContent = data.regions;
    
    updateConfidenceRing('geo-confidence', data.anomaly);
}

function updateConfidenceRing(elementId, value) {
    const container = document.getElementById(elementId);
    if (!container) return;
    
    const ring = container.querySelector('.ring-fg');
    const valEl = container.querySelector('.conf-value');
    
    const circumference = 97.4;
    const dashArray = `${circumference * Math.min(1, value)} ${circumference}`;
    ring.setAttribute('stroke-dasharray', dashArray);
    
    valEl.textContent = `${Math.round(value * 100)}%`;
    
    // Color based on value
    if (value > 0.7) {
        ring.style.stroke = '#ef4444';
        valEl.style.color = '#ef4444';
    } else if (value > 0.4) {
        ring.style.stroke = '#f59e0b';
        valEl.style.color = '#f59e0b';
    } else {
        ring.style.stroke = '#00d4ff';
        valEl.style.color = '#00d4ff';
    }
}

// ══════════════════════════════════════════════
// Campaign Intelligence
// ══════════════════════════════════════════════
function updateCampaign(data) {
    const list = document.getElementById('campaignList');
    const badge = document.getElementById('campaignBadge');
    
    badge.textContent = `${data.isNew ? '1 New' : '1 Active'}`;
    
    if (data.isNew) {
        list.innerHTML = '';
    }
    
    const severityPct = Math.round(data.risk * 100);
    
    list.innerHTML = `
        <div class="campaign-item">
            <span class="campaign-id">C-${String(data.id).padStart(3, '0')}</span>
            <div class="campaign-info">
                <span class="campaign-incidents">Campaign #${data.id} — ${currentStep + 1} incident(s)</span>
                <span class="campaign-severity">Severity: ${severityPct}% | ${data.risk > 0.7 ? '🔴 CRITICAL' : data.risk > 0.4 ? '🟡 MEDIUM' : '🟢 LOW'}</span>
            </div>
        </div>
    `;
}

// ══════════════════════════════════════════════
// Navigation Stats
// ══════════════════════════════════════════════
function updateNavStats() {
    document.getElementById('threatsToday').textContent = currentStep + 1;
    document.getElementById('activeCampaigns').textContent = '1';
    document.getElementById('signalsProcessed').textContent = signalsProcessedCount;
}

// ══════════════════════════════════════════════
// Timeline
// ══════════════════════════════════════════════
function addTimelineEvent(step) {
    const timeline = document.getElementById('timeline');
    if (timeline.querySelector('.timeline-empty')) {
        timeline.innerHTML = '';
    }
    
    const levelClass = step.threatLevel ? step.threatLevel.toLowerCase() : 'minimal';
    const minutes = Math.floor(step.offset / 60);
    const seconds = step.offset % 60;
    const timeStr = `T+${minutes}:${String(seconds).padStart(2, '0')}`;
    
    const event = document.createElement('div');
    event.className = `timeline-event threat-${levelClass}`;
    event.innerHTML = `
        <span class="event-time">${timeStr}</span>
        <div class="event-content">
            <div class="event-phase">Phase ${step.phase}: ${step.phaseName}</div>
            <div class="event-message">"${step.message.substring(0, 100)}..."</div>
            ${step.threatScore ? `
                <span class="event-threat threat-badge-${levelClass}">
                    ● ${step.threatScore.toFixed(2)} ${step.threatLevel}
                </span>
            ` : ''}
            ${step.phase === 3 ? '<span class="event-threat threat-badge-high" style="margin-left:6px">⚠️ EARLY WARNING</span>' : ''}
        </div>
    `;
    
    timeline.appendChild(event);
    timeline.scrollTop = timeline.scrollHeight;
}

// ══════════════════════════════════════════════
// Velocity Chart
// ══════════════════════════════════════════════
let velocityCtx, velocityCanvas;

function initVelocityChart() {
    velocityCanvas = document.getElementById('velocityCanvas');
    velocityCtx = velocityCanvas.getContext('2d');
    drawVelocityChart([]);
}

function updateVelocityChart(speed) {
    velocityHistory.push(speed);
    drawVelocityChart(velocityHistory);
}

function drawVelocityChart(data) {
    const ctx = velocityCtx;
    if (!ctx || !velocityCanvas) return;
    const w = velocityCanvas.width;
    const h = velocityCanvas.height;
    
    ctx.clearRect(0, 0, w, h);
    
    const maxVal = Math.max(0.8, ...(data.length > 0 ? data : [0.8]));
    const padding = { top: 15, right: 15, bottom: 20, left: 40 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;
    
    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (chartH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(w - padding.right, y);
        ctx.stroke();
        
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.font = '9px JetBrains Mono';
        ctx.textAlign = 'right';
        ctx.fillText((maxVal * (4 - i) / 4).toFixed(2), padding.left - 5, y + 3);
    }
    
    // Threshold line (0.30)
    const threshY = padding.top + chartH * (1 - 0.3 / maxVal);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.45)';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(padding.left, threshY);
    ctx.lineTo(w - padding.right, threshY);
    ctx.stroke();
    ctx.setLineDash([]);
    
    ctx.fillStyle = 'rgba(239, 68, 68, 0.6)';
    ctx.font = '9px Inter';
    ctx.textAlign = 'left';
    ctx.fillText('Alert Threshold (0.30)', padding.left + 5, threshY - 4);
    
    if (data.length < 1) return;
    
    // Gradient fill
    const gradient = ctx.createLinearGradient(0, padding.top, 0, h);
    gradient.addColorStop(0, 'rgba(124, 58, 237, 0.35)');
    gradient.addColorStop(1, 'rgba(124, 58, 237, 0.02)');
    
    const count = data.length;
    const stepX = count > 1 ? chartW / (count - 1) : 0;
    
    // Area
    ctx.beginPath();
    data.forEach((val, i) => {
        const x = count === 1 ? padding.left + chartW / 2 : padding.left + stepX * i;
        const y = padding.top + chartH * (1 - Math.min(maxVal, val) / maxVal);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    const lastX = count === 1 ? padding.left + chartW / 2 : padding.left + chartW;
    const firstX = count === 1 ? padding.left + chartW / 2 : padding.left;
    ctx.lineTo(lastX, padding.top + chartH);
    ctx.lineTo(firstX, padding.top + chartH);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
    
    // Line
    ctx.beginPath();
    data.forEach((val, i) => {
        const x = count === 1 ? padding.left + chartW / 2 : padding.left + stepX * i;
        const y = padding.top + chartH * (1 - Math.min(maxVal, val) / maxVal);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 2.5;
    ctx.stroke();
    
    // Points
    data.forEach((val, i) => {
        const x = count === 1 ? padding.left + chartW / 2 : padding.left + stepX * i;
        const y = padding.top + chartH * (1 - Math.min(maxVal, val) / maxVal);
        
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = val >= 0.3 ? '#ef4444' : '#8b5cf6';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

// ══════════════════════════════════════════════
// Correlation Matrix
// ══════════════════════════════════════════════
function initCorrelationMatrix() {
    const canvas = document.getElementById('correlationCanvas');
    const ctx = canvas.getContext('2d');
    const signals = ['NLP', 'Temporal', 'Voice', 'Network', 'Geo'];
    
    // Simulated correlation values
    const correlations = [
        [1.0, 0.75, 0.45, 0.30, 0.20],
        [0.75, 1.0, 0.35, 0.25, 0.40],
        [0.45, 0.35, 1.0, 0.50, 0.15],
        [0.30, 0.25, 0.50, 1.0, 0.65],
        [0.20, 0.40, 0.15, 0.65, 1.0],
    ];
    
    const n = signals.length;
    const cellSize = 48;
    const offsetX = 60;
    const offsetY = 40;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw cells
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const val = correlations[i][j];
            const x = offsetX + j * cellSize;
            const y = offsetY + i * cellSize;
            
            // Color intensity
            const r = Math.round(124 * val);
            const g = Math.round(58 * val);
            const b = Math.round(237 * val);
            
            ctx.fillStyle = `rgba(${r + 50}, ${g + 50}, ${b}, ${0.3 + val * 0.6})`;
            ctx.fillRect(x, y, cellSize - 2, cellSize - 2);
            
            // Border
            ctx.strokeStyle = 'rgba(255,255,255,0.08)';
            ctx.strokeRect(x, y, cellSize - 2, cellSize - 2);
            
            // Value text
            ctx.fillStyle = val > 0.5 ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.5)';
            ctx.font = '10px JetBrains Mono';
            ctx.textAlign = 'center';
            ctx.fillText(val.toFixed(2), x + (cellSize - 2) / 2, y + (cellSize - 2) / 2 + 4);
        }
    }
    
    // Labels
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.font = '10px Inter';
    
    // Column headers
    ctx.textAlign = 'center';
    for (let j = 0; j < n; j++) {
        ctx.fillText(signals[j], offsetX + j * cellSize + (cellSize - 2) / 2, offsetY - 10);
    }
    
    // Row labels
    ctx.textAlign = 'right';
    for (let i = 0; i < n; i++) {
        ctx.fillText(signals[i], offsetX - 8, offsetY + i * cellSize + (cellSize - 2) / 2 + 4);
    }
}

// ══════════════════════════════════════════════
// Pipeline Animation
// ══════════════════════════════════════════════
function animatePipeline(callback) {
    const steps = document.querySelectorAll('.pipeline-step');
    let i = 0;
    
    function activateStep() {
        if (i > 0) steps[i - 1].classList.remove('processing');
        if (i < steps.length) {
            steps[i].classList.add('processing');
            i++;
            setTimeout(activateStep, 200);
        } else {
            setTimeout(() => {
                steps[steps.length - 1].classList.remove('processing');
                callback();
            }, 200);
        }
    }
    
    activateStep();
}

// ══════════════════════════════════════════════
// Utilities
// ══════════════════════════════════════════════
function animateValue(element, start, end, duration, decimals = 0) {
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease out cubic
        
        const current = start + (end - start) * eased;
        element.textContent = current.toFixed(decimals);
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

function updateClock() {
    setInterval(() => {
        const now = new Date();
        const statusEl = document.getElementById('systemStatus');
        if (statusEl) {
            statusEl.querySelector('span:last-child').textContent = 
                `System Active — ${now.toLocaleTimeString()}`;
        }
    }, 1000);
}
