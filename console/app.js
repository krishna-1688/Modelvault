// ModelVault Defense Console -- polls /admin/stats on the same origin (this
// page is served directly by the gateway) and renders everything client-side.

const params = new URLSearchParams(window.location.search);
const API_KEY = params.get('key') || '';
const POLL_MS = 1000;
const TIER_NORMAL_MAX = 35;
const TIER_ELEVATED_MAX = 70;
const TIER_COLORS = { normal: '#34d399', elevated: '#fbbf24', critical: '#f87171', blocked: '#8b8d98' };
const MAX_TRACE_ROWS = 150;

let lastCriticalCount = 0;
let previousValues = {};
let currentFilter = 'all';
let renderedKeys = new Set();
let requestHistory = [];   // rolling {t, total} samples for the sparkline
let suspiciousHistory = []; // rolling suspicious% samples
let referenceZone = null;
let plottedPoints = new Set();

document.getElementById('footer-url').textContent = window.location.origin;

// ---------------- Feature space map ----------------
// The 2D projection is fit ONCE at training time (train.py) purely for
// display -- it has no bearing on detection math. This turns Layer 2's
// abstract "macro distortion" score into something spatial and literal: an
// attacker's queries visibly drift away from where real transactions sit.
const FM_CENTER_X = 170, FM_CENTER_Y = 110, FM_PIXELS_PER_UNIT = 24;
const FM_POINT_LIFETIME_MS = 20000;

async function loadReferenceZone() {
  try {
    const res = await fetch('/admin/reference-zone', { headers: { 'X-API-Key': API_KEY } });
    if (!res.ok) return;
    referenceZone = await res.json();
    const [sx, sy] = referenceZone.std;
    document.getElementById('fm-zone-inner').setAttribute('rx', sx * FM_PIXELS_PER_UNIT);
    document.getElementById('fm-zone-inner').setAttribute('ry', sy * FM_PIXELS_PER_UNIT);
    document.getElementById('fm-zone-outer').setAttribute('rx', sx * FM_PIXELS_PER_UNIT * 2);
    document.getElementById('fm-zone-outer').setAttribute('ry', sy * FM_PIXELS_PER_UNIT * 2);
  } catch (e) { /* map stays empty; everything else still works */ }
}

function fmProject(x, y) {
  const mx = referenceZone ? referenceZone.mean[0] : 0;
  const my = referenceZone ? referenceZone.mean[1] : 0;
  const px = FM_CENTER_X + (x - mx) * FM_PIXELS_PER_UNIT;
  const py = FM_CENTER_Y + (y - my) * FM_PIXELS_PER_UNIT;
  return [Math.max(8, Math.min(332, px)), Math.max(22, Math.min(212, py))];
}

function updateFeatureMap(events) {
  const group = document.getElementById('fm-points');
  events.forEach(event => {
    const key = String(event.timestamp);
    if (plottedPoints.has(key)) return;
    const proj = event.trace && event.trace.layer2 && event.trace.layer2.projection;
    if (!proj) return;
    plottedPoints.add(key);
    const [px, py] = fmProject(proj[0], proj[1]);
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', px);
    circle.setAttribute('cy', py);
    circle.setAttribute('r', event.tier === 'critical' ? 3.2 : 2.4);
    circle.setAttribute('fill', TIER_COLORS[event.tier] || '#8b8d98');
    circle.setAttribute('class', 'fm-point');
    circle.dataset.key = key;
    circle.dataset.ts = event.timestamp * 1000;
    group.appendChild(circle);
  });

  // Age out old points so the map reflects recent behavior, not the entire
  // session's history piling up indefinitely.
  const now = Date.now();
  Array.from(group.children).forEach(circle => {
    const age = now - Number(circle.dataset.ts);
    if (age > FM_POINT_LIFETIME_MS) {
      circle.remove();
      plottedPoints.delete(circle.dataset.key);
    } else if (age > FM_POINT_LIFETIME_MS - 3000) {
      circle.style.opacity = (FM_POINT_LIFETIME_MS - age) / 3000;
    }
  });
}

// ---------------- Clock ----------------
function tickClock() { document.getElementById('clock').textContent = new Date().toLocaleTimeString(); }
setInterval(tickClock, 1000);
tickClock();

// ---------------- Audio alert (Web Audio API, no asset files) ----------------
let audioCtx = null;
function playAlertTone() {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.18);
    gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.start(); osc.stop(audioCtx.currentTime + 0.35);
  } catch (e) { /* autoplay policies may block until first user interaction -- silently skip */ }
}

// ---------------- Sparklines (plain inline SVG, no chart library needed) ----------------
function drawSparkline(svgId, values, color) {
  const svg = document.getElementById(svgId);
  if (!svg || values.length < 2) return;
  const w = 100, h = 24;
  const min = Math.min(...values), max = Math.max(...values);
  const range = max - min || 1;
  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = h - ((v - min) / range) * (h - 4) - 2;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  svg.innerHTML = `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>`;
}

// ---------------- Charts ----------------
const bandsPlugin = {
  id: 'tierBands',
  beforeDraw(chart) {
    const { ctx, chartArea, scales } = chart;
    if (!chartArea) return;
    const y = scales.y;
    const zones = [
      [0, TIER_NORMAL_MAX, 'rgba(52,211,153,0.05)'],
      [TIER_NORMAL_MAX, TIER_ELEVATED_MAX, 'rgba(251,191,36,0.05)'],
      [TIER_ELEVATED_MAX, 100, 'rgba(248,113,113,0.07)'],
    ];
    ctx.save();
    zones.forEach(([lo, hi, color]) => {
      const yTop = y.getPixelForValue(hi);
      const yBot = y.getPixelForValue(lo);
      ctx.fillStyle = color;
      ctx.fillRect(chartArea.left, yTop, chartArea.right - chartArea.left, yBot - yTop);
    });
    ctx.restore();
  }
};

const timelineCtx = document.getElementById('timeline-chart').getContext('2d');
const gradient = timelineCtx.createLinearGradient(0, 0, 0, 210);
gradient.addColorStop(0, 'rgba(91,141,239,0.28)');
gradient.addColorStop(1, 'rgba(91,141,239,0.01)');

const timelineChart = new Chart(timelineCtx, {
  type: 'line',
  data: { labels: [], datasets: [{ data: [], borderColor: '#5b8def', backgroundColor: gradient, fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 }] },
  options: {
    responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
    plugins: { legend: { display: false }, tooltip: { intersect: false, mode: 'index' } },
    scales: {
      y: { min: 0, max: 100, grid: { color: '#1a1c1f' }, ticks: { color: '#52545c', font: { size: 10 } } },
      x: { grid: { display: false }, ticks: { color: '#52545c', maxTicksLimit: 8, font: { size: 10 } } },
    },
  },
  plugins: [bandsPlugin],
});

const donutChart = new Chart(document.getElementById('donut-chart').getContext('2d'), {
  type: 'doughnut',
  data: { labels: ['Normal', 'Elevated', 'Critical'], datasets: [{ data: [0, 0, 0], backgroundColor: [TIER_COLORS.normal, TIER_COLORS.elevated, TIER_COLORS.critical], borderWidth: 0, hoverOffset: 4 }] },
  options: { responsive: true, maintainAspectRatio: false, cutout: '72%', animation: { duration: 250 }, plugins: { legend: { display: false } } },
});

function renderDonutLegend(counts) {
  const total = (counts.normal || 0) + (counts.elevated || 0) + (counts.critical || 0);
  document.getElementById('donut-legend').innerHTML = ['normal', 'elevated', 'critical'].map(tier => {
    const pct = total ? Math.round(((counts[tier] || 0) / total) * 100) : 0;
    return `<span><span class="legend-dot" style="background:${TIER_COLORS[tier]}"></span>${tier} ${pct}%</span>`;
  }).join('');
}

// ---------------- Mini gauge ----------------
function updateGauge(value) {
  const clamped = Math.max(0, Math.min(100, value));
  let color, tierText;
  if (clamped <= TIER_NORMAL_MAX) { color = TIER_COLORS.normal; tierText = 'NORMAL'; }
  else if (clamped <= TIER_ELEVATED_MAX) { color = TIER_COLORS.elevated; tierText = 'ELEVATED'; }
  else { color = TIER_COLORS.critical; tierText = 'CRITICAL'; }
  document.getElementById('gauge-value').textContent = Math.round(clamped);
  document.getElementById('gauge-value').style.color = color;
  document.getElementById('gauge-tier').textContent = tierText;
  document.getElementById('gauge-tier').style.color = color;
  const fill = document.getElementById('mini-gauge-fill');
  fill.style.width = clamped + '%';
  fill.style.background = color;
}

// ---------------- Stat bump ----------------
function setStat(id, value) {
  const el = document.getElementById(id);
  if (previousValues[id] !== undefined && previousValues[id] !== value) {
    el.classList.remove('bump'); void el.offsetWidth; el.classList.add('bump');
  }
  previousValues[id] = value;
  el.textContent = value;
}

// ---------------- Alert banner + attack signature ----------------
function updateBanner(stats) {
  const counts = stats.tier_counts;
  const total = (counts.normal || 0) + (counts.elevated || 0) + (counts.critical || 0);
  const banner = document.getElementById('alert-banner');
  const text = document.getElementById('alert-text');
  if (total === 0) { banner.className = 'alert-banner alert-ok'; text.textContent = 'Waiting for traffic…'; return; }
  const suspiciousShare = ((counts.elevated || 0) + (counts.critical || 0)) / total;
  if (suspiciousShare > 0.5) {
    banner.className = 'alert-banner alert-critical';
    text.textContent = `HIGH ALERT — ${Math.round(suspiciousShare*100)}% of recent traffic is elevated/critical tier. Likely active extraction attempt.`;
  } else if (suspiciousShare > 0.2) {
    banner.className = 'alert-banner alert-warning';
    text.textContent = `Elevated suspicion — ${Math.round(suspiciousShare*100)}% of recent traffic is elevated/critical tier.`;
  } else {
    banner.className = 'alert-banner alert-ok';
    text.textContent = `Traffic looks normal (${Math.round(suspiciousShare*100)}% elevated/critical).`;
  }
}

// Classifies the CURRENT dominant attack shape from the macro/micro signal
// mix of recent suspicious traffic -- a simple, transparent heuristic over
// the two real signals Layer 2 computes (not a trained classifier), but it
// gives a live "what kind of attack is this" read exactly like feature-
// attribution panels in real fraud/SOC tooling.
function updateAttackSignature(events) {
  const badge = document.getElementById('signature-badge');
  const suspicious = events.filter(e => !e.blocked && e.trace && e.trace.layer2 && !e.trace.layer2.skipped && (e.tier === 'elevated' || e.tier === 'critical'));
  if (suspicious.length < 6) { badge.style.display = 'none'; return; }

  const sample = suspicious.slice(0, 30);
  const avgMacro = sample.reduce((s, e) => s + e.trace.layer2.macro_feature_distortion, 0) / sample.length;
  const avgMicro = sample.reduce((s, e) => s + e.trace.layer2.micro_coverage_density, 0) / sample.length;

  let label;
  if (avgMacro > 55 && avgMicro < 35) label = 'PATTERN: RANDOM PROBING';
  else if (avgMacro < 50 && avgMicro > 50) label = 'PATTERN: IN-DISTRIBUTION SWEEP';
  else if (avgMacro > 45 && avgMicro > 45) label = 'PATTERN: BOUNDARY MAPPING';
  else label = 'PATTERN: LOW-CONFIDENCE ANOMALY';

  badge.textContent = `${label} (macro ${avgMacro.toFixed(0)} · micro ${avgMicro.toFixed(0)})`;
  badge.style.display = 'inline-block';
}

// ---------------- Cross-client correlation (Sybil cluster) panel ----------------
// Groups recent suspicious events by client_id. Many DISTINCT identities
// each showing up only a few times, all suspicious at once, is exactly the
// Sybil pattern Layer 2's pooled (cross-client) reservoir is designed to
// catch -- this panel makes that pooling visible instead of implicit.
function updateClusterPanel(events) {
  const note = document.getElementById('cluster-note');
  const list = document.getElementById('cluster-list');
  const suspicious = events.filter(e => !e.blocked && (e.tier === 'elevated' || e.tier === 'critical'));

  if (suspicious.length === 0) {
    note.className = 'cluster-note';
    note.textContent = 'Monitoring for coordinated (Sybil-style) patterns across identities…';
    list.innerHTML = '';
    return;
  }

  const byClient = {};
  suspicious.forEach(e => {
    if (!byClient[e.client_id]) byClient[e.client_id] = { count: 0, threatSum: 0 };
    byClient[e.client_id].count += 1;
    byClient[e.client_id].threatSum += e.threat_index || 0;
  });
  const clients = Object.entries(byClient).map(([id, v]) => ({ id, count: v.count, avg: v.threatSum / v.count }));
  clients.sort((a, b) => b.avg - a.avg);

  const distinctClients = clients.length;
  const avgPerClient = suspicious.length / distinctClients;

  if (distinctClients >= 5 && avgPerClient <= 4) {
    note.className = 'cluster-note hot';
    note.textContent = `⚠ ${distinctClients} distinct identities showing correlated suspicious behavior, ${avgPerClient.toFixed(1)} requests each on average -- consistent with a Sybil attack splitting traffic across fake accounts.`;
  } else {
    note.className = 'cluster-note';
    note.textContent = `${distinctClients} distinct identities with suspicious traffic so far.`;
  }

  const maxAvg = Math.max(...clients.map(c => c.avg), 1);
  list.innerHTML = clients.slice(0, 8).map(c => {
    const color = c.avg > TIER_ELEVATED_MAX ? TIER_COLORS.critical : TIER_COLORS.elevated;
    return `
      <div class="cluster-row">
        <span class="cluster-client">${c.id}</span>
        <span class="cluster-count">×${c.count}</span>
        <span class="cluster-avg" style="color:${color}">${c.avg.toFixed(0)}</span>
        <div class="cluster-bar-track"><div class="cluster-bar-fill" style="width:${(c.avg / maxAvg) * 100}%; background:${color}"></div></div>
      </div>`;
  }).join('');
}

// ---------------- Trace inspector ----------------
function signalSummary(event) {
  if (event.blocked) return 'rejected before scoring';
  const t = event.trace || {};
  if (t.layer2 && t.layer2.skipped) return 'defense disabled — unscored';
  if (t.layer2) {
    const parts = [`macro ${t.layer2.macro_feature_distortion}`, `micro ${t.layer2.micro_coverage_density}`];
    if (t.layer4 && t.layer4.triggered) parts.push('watermarked');
    if (t.layer3 && t.layer3.boundary_perturbed) parts.push('boundary-flipped');
    return parts.join(' · ');
  }
  return '';
}

// Plain-English "why" sentence for the expanded panel's header -- the
// research on explainable SOC tooling is consistent that analysts want a
// one-line human summary before the raw numbers, not instead of them.
function reasoningSentence(event) {
  if (event.blocked) return 'Rejected before any scoring: this client exceeded its request-rate budget.';
  const t = event.trace || {};
  if (t.layer2 && t.layer2.skipped) return 'Defense is currently disabled, so this request bypassed all scoring and was answered in full.';
  if (!t.layer2) return '';
  const macro = t.layer2.macro_feature_distortion, micro = t.layer2.micro_coverage_density;
  const reasons = [];
  if (macro > 55) reasons.push(`sits far from the training manifold (macro ${macro})`);
  if (micro > 55) reasons.push(`is redundant with recent queries across other clients (micro ${micro})`);
  if (reasons.length === 0) reasons.push('looked statistically ordinary');
  let sentence = `This query ${reasons.join(' and ')}.`;
  if (t.layer3 && t.layer3.tier !== 'normal') sentence += ` Routed to the ${t.layer3.tier} tier: ${t.layer3.degradation}.`;
  if (t.layer4 && t.layer4.triggered) sentence += ` A watermark was planted (label ${t.layer4.original_label} → ${t.layer4.final_label}) for later ownership proof.`;
  return sentence;
}

function wfBar(value, color) {
  return `<div class="wf-bar"><div class="wf-bar-fill" style="width:${Math.min(100, value)}%; background:${color}"></div></div>`;
}

function renderWaterfall(event) {
  const t = event.trace || {};
  const rows = [];

  if (event.blocked) {
    rows.push(`<div class="wf-stage"><div class="wf-stage-name">L1 · Ingress</div><div class="wf-stage-body"><span class="wf-fail">✗ BLOCKED</span> — rate limit exceeded for this client</div></div>`);
  } else {
    rows.push(`<div class="wf-stage"><div class="wf-stage-name">L1 · Ingress</div><div class="wf-stage-body"><span class="wf-pass">✓ allowed</span> <span class="dim">within rate limit</span></div></div>`);
  }

  if (t.layer2 && t.layer2.skipped) {
    rows.push(`<div class="wf-stage"><div class="wf-stage-name">L2 · Detection</div><div class="wf-stage-body dim">skipped — ${t.layer2.reason}</div></div>`);
  } else if (t.layer2) {
    rows.push(`<div class="wf-stage"><div class="wf-stage-name">L2 · Detection</div><div class="wf-stage-body">
      threat index <strong>${t.layer2.threat_index}</strong> = macro ${t.layer2.macro_feature_distortion} + micro ${t.layer2.micro_coverage_density}
      ${wfBar(t.layer2.threat_index, TIER_COLORS[event.tier] || '#5b8def')}
    </div></div>`);
  }

  if (t.layer3) {
    const extra = t.layer3.boundary_perturbed ? ' <span style="color:#a78bfa">· boundary-adjacent label flip applied</span>' : '';
    let eyeView = '';
    if (t.layer3.undefended_probabilities) {
      const undef = t.layer3.undefended_probabilities.map(p => p.toFixed(3)).join(', ');
      const disclosed = t.layer3.disclosed_probabilities ? t.layer3.disclosed_probabilities.map(p => p.toFixed(3)).join(', ') : 'none (label only)';
      eyeView = `<div class="eye-view">
        <div class="eye-row"><span class="eye-label">Undefended API would return</span><span class="eye-val eye-bad">[${undef}]</span></div>
        <div class="eye-row"><span class="eye-label">ModelVault actually returned</span><span class="eye-val eye-good">[${disclosed}]</span></div>
      </div>`;
    }
    rows.push(`<div class="wf-stage"><div class="wf-stage-name">L3 · Response</div><div class="wf-stage-body">tier <strong>${t.layer3.tier}</strong> — ${t.layer3.degradation}${extra}${eyeView}</div></div>`);
  }

  if (t.layer4) {
    if (t.layer4.triggered) {
      rows.push(`<div class="wf-stage"><div class="wf-stage-name">L4 · Watermark</div><div class="wf-stage-body"><span style="color:#a78bfa">⭐ triggered</span> — label flipped ${t.layer4.original_label} → ${t.layer4.final_label}</div></div>`);
    } else if (t.layer4.reason) {
      rows.push(`<div class="wf-stage"><div class="wf-stage-name">L4 · Watermark</div><div class="wf-stage-body dim">not evaluated — ${t.layer4.reason}</div></div>`);
    } else {
      rows.push(`<div class="wf-stage"><div class="wf-stage-name">L4 · Watermark</div><div class="wf-stage-body dim">not triggered this request</div></div>`);
    }
  }

  const reasoning = reasoningSentence(event);
  return `${reasoning ? `<div class="trace-reason">${reasoning}</div><div style="height:8px"></div>` : ''}<div class="trace-waterfall">${rows.join('')}</div>`;
}

function verdictBadge(event) {
  if (event.blocked) return `<span class="verdict-badge verdict-blocked">BLOCKED</span>`;
  return `<span class="verdict-badge verdict-${event.tier}">${event.tier.toUpperCase()}</span>`;
}

function matchesFilter(event) {
  if (currentFilter === 'all') return true;
  if (currentFilter === 'blocked') return event.blocked;
  return event.tier === currentFilter;
}

function buildRowElement(event) {
  const key = String(event.timestamp);
  const time = new Date(event.timestamp * 1000).toLocaleTimeString();
  const wm = event.watermarked ? '<span class="wm-badge">⭐</span>' : '';

  const row = document.createElement('div');
  row.className = 'trace-row';
  row.dataset.key = key;
  row.dataset.tier = event.blocked ? 'blocked' : event.tier;
  row.style.display = matchesFilter(event) ? '' : 'none';

  row.innerHTML = `
    <div class="trace-row-main">
      <span class="trace-time">${time}</span>
      <span class="trace-client">${event.client_id}</span>
      <span>${verdictBadge(event)}${wm}</span>
      <span class="trace-threat">${event.threat_index !== null ? event.threat_index.toFixed(0) : '—'}</span>
      <span class="trace-signal">${signalSummary(event)}</span>
      <span class="trace-caret">▶</span>
    </div>
    <div class="trace-detail"></div>`;

  row.querySelector('.trace-row-main').addEventListener('click', () => {
    const expanded = row.classList.toggle('expanded');
    const detail = row.querySelector('.trace-detail');
    if (expanded && !detail.dataset.built) {
      detail.innerHTML = renderWaterfall(event);
      detail.dataset.built = '1';
    }
  });

  return row;
}

// Only NEW events get a DOM node created (and therefore only they play the
// "just arrived" animation) -- previously the entire list was rebuilt via
// innerHTML on every single poll, which replayed the fade-in animation on
// EVERY row, EVERY second, forever. That was the "log keeps blinking" bug.
function updateTraceList(events) {
  const list = document.getElementById('trace-list');
  const placeholder = list.querySelector('.trace-placeholder');
  const newEvents = events.filter(e => !renderedKeys.has(String(e.timestamp)));

  if (newEvents.length > 0) {
    if (placeholder) placeholder.remove();
    // events arrive most-recent-first; insert oldest-of-the-new-batch first
    // so the final DOM order still has the newest event at the very top.
    for (let i = newEvents.length - 1; i >= 0; i--) {
      const event = newEvents[i];
      renderedKeys.add(String(event.timestamp));
      list.insertBefore(buildRowElement(event), list.firstChild);
    }
    while (list.children.length > MAX_TRACE_ROWS) {
      const last = list.lastElementChild;
      renderedKeys.delete(last.dataset.key);
      list.removeChild(last);
    }
  }

  if (list.children.length === 0) {
    list.innerHTML = `<div class="trace-placeholder">No requests yet…</div>`;
  }
}

function applyFilterToDom() {
  document.querySelectorAll('.trace-row').forEach(row => {
    row.style.display = (currentFilter === 'all' || row.dataset.tier === currentFilter) ? '' : 'none';
  });
}

document.getElementById('trace-filters').addEventListener('click', (e) => {
  if (!e.target.classList.contains('filter-pill')) return;
  document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
  e.target.classList.add('active');
  currentFilter = e.target.dataset.filter;
  applyFilterToDom();
});

// ---------------- Flash + status ----------------
function maybeFlash(criticalCount) {
  if (criticalCount > lastCriticalCount) {
    const overlay = document.getElementById('flash-overlay');
    overlay.classList.remove('flash'); void overlay.offsetWidth; overlay.classList.add('flash');
    playAlertTone();
  }
  lastCriticalCount = criticalCount;
}
function setStatus(online) {
  document.getElementById('status-pill').classList.toggle('offline', !online);
  document.getElementById('status-text').textContent = online ? 'LIVE' : 'OFFLINE';
}

// ---------------- Main poll loop ----------------
async function poll() {
  try {
    const res = await fetch('/admin/stats', { headers: { 'X-API-Key': API_KEY } });
    if (!res.ok) throw new Error('bad status ' + res.status);
    render(await res.json());
    setStatus(true);
  } catch (err) {
    setStatus(false);
  }
}

function render(stats) {
  const counts = stats.tier_counts || { normal: 0, elevated: 0, critical: 0 };
  const total = (counts.normal || 0) + (counts.elevated || 0) + (counts.critical || 0);
  const suspiciousPct = total ? Math.round(((counts.elevated || 0) + (counts.critical || 0)) / total * 100) : 0;

  setStat('kpi-requests', stats.total_requests);
  setStat('kpi-clients', stats.total_clients);
  setStat('kpi-blocked', stats.blocked_count || 0);
  setStat('kpi-watermarks', stats.watermark_triggers);
  setStat('kpi-suspicious', suspiciousPct + '%');

  requestHistory.push(stats.total_requests);
  if (requestHistory.length > 30) requestHistory.shift();
  drawSparkline('spark-requests', requestHistory, '#5b8def');

  suspiciousHistory.push(suspiciousPct);
  if (suspiciousHistory.length > 30) suspiciousHistory.shift();
  drawSparkline('spark-suspicious', suspiciousHistory, '#fbbf24');

  const recent = stats.recent_threat_indices || [];
  updateGauge(recent.length ? recent[recent.length - 1] : 0);

  timelineChart.data.labels = recent.map((_, i) => i);
  timelineChart.data.datasets[0].data = recent;
  timelineChart.update('none');

  donutChart.data.datasets[0].data = [counts.normal || 0, counts.elevated || 0, counts.critical || 0];
  donutChart.update('none');
  renderDonutLegend(counts);

  updateBanner(stats);

  const events = stats.recent_events || [];
  updateAttackSignature(events);
  updateClusterPanel(events);
  updateTraceList(events);
  updateFeatureMap(events);

  maybeFlash(counts.critical || 0);

  const toggle = document.getElementById('defense-toggle');
  if (document.activeElement !== toggle) toggle.checked = stats.defense_enabled;
  document.getElementById('defense-label').textContent = stats.defense_enabled ? 'DEFENSE ON' : 'DEFENSE OFF';
}

document.getElementById('defense-toggle').addEventListener('change', async (e) => {
  try {
    await fetch('/admin/toggle-defense', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
      body: JSON.stringify({ enabled: e.target.checked }),
    });
  } catch (err) { /* next poll resyncs */ }
});

loadReferenceZone();
poll();
setInterval(poll, POLL_MS);
