// ModelVault Defense Console -- polls /admin/stats on the same origin (this
// page is served directly by the gateway) and renders everything client-side.

const params = new URLSearchParams(window.location.search);
const API_KEY = params.get('key') || '';
const POLL_MS = 1000;
const TIER_NORMAL_MAX = 35;
const TIER_ELEVATED_MAX = 70;
const TIER_COLORS = { normal: '#34d399', elevated: '#fbbf24', critical: '#f87171', blocked: '#8b8d98' };

let lastEventTimestamp = 0;
let lastCriticalCount = 0;
let previousValues = {};
let expandedRows = new Set();
let currentFilter = 'all';
let allEvents = [];

document.getElementById('footer-url').textContent = window.location.origin;

// ---------------- Clock ----------------
function tickClock() { document.getElementById('clock').textContent = new Date().toLocaleTimeString(); }
setInterval(tickClock, 1000);
tickClock();

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

// ---------------- Alert banner ----------------
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

function wfBar(value, color) {
  return `<div class="wf-bar"><div class="wf-bar-fill" style="width:${Math.min(100, value)}%; background:${color}"></div></div>`;
}

function renderWaterfall(event) {
  const t = event.trace || {};
  const rows = [];

  // Layer 1
  if (event.blocked) {
    rows.push(`<div class="wf-stage"><div class="wf-stage-name">L1 · Ingress</div><div class="wf-stage-body"><span class="wf-fail">✗ BLOCKED</span> — rate limit exceeded for this client</div></div>`);
  } else {
    rows.push(`<div class="wf-stage"><div class="wf-stage-name">L1 · Ingress</div><div class="wf-stage-body"><span class="wf-pass">✓ allowed</span> <span class="dim">within rate limit</span></div></div>`);
  }

  // Layer 2
  if (t.layer2 && t.layer2.skipped) {
    rows.push(`<div class="wf-stage"><div class="wf-stage-name">L2 · Detection</div><div class="wf-stage-body dim">skipped — ${t.layer2.reason}</div></div>`);
  } else if (t.layer2) {
    rows.push(`<div class="wf-stage"><div class="wf-stage-name">L2 · Detection</div><div class="wf-stage-body">
      threat index <strong>${t.layer2.threat_index}</strong> = macro ${t.layer2.macro_feature_distortion} + micro ${t.layer2.micro_coverage_density}
      ${wfBar(t.layer2.threat_index, TIER_COLORS[event.tier] || '#5b8def')}
    </div></div>`);
  }

  // Layer 3
  if (t.layer3) {
    const extra = t.layer3.boundary_perturbed ? ' <span style="color:#a78bfa">· boundary-adjacent label flip applied</span>' : '';
    rows.push(`<div class="wf-stage"><div class="wf-stage-name">L3 · Response</div><div class="wf-stage-body">tier <strong>${t.layer3.tier}</strong> — ${t.layer3.degradation}${extra}</div></div>`);
  }

  // Layer 4
  if (t.layer4) {
    if (t.layer4.triggered) {
      rows.push(`<div class="wf-stage"><div class="wf-stage-name">L4 · Watermark</div><div class="wf-stage-body"><span style="color:#a78bfa">⭐ triggered</span> — label flipped ${t.layer4.original_label} → ${t.layer4.final_label}</div></div>`);
    } else if (t.layer4.reason) {
      rows.push(`<div class="wf-stage"><div class="wf-stage-name">L4 · Watermark</div><div class="wf-stage-body dim">not evaluated — ${t.layer4.reason}</div></div>`);
    } else {
      rows.push(`<div class="wf-stage"><div class="wf-stage-name">L4 · Watermark</div><div class="wf-stage-body dim">not triggered this request</div></div>`);
    }
  }

  return `<div class="trace-waterfall">${rows.join('')}</div>`;
}

function verdictBadge(event) {
  if (event.blocked) return `<span class="verdict-badge verdict-blocked">BLOCKED</span>`;
  const cls = `verdict-${event.tier}`;
  return `<span class="verdict-badge ${cls}">${event.tier.toUpperCase()}</span>`;
}

function renderTraceList() {
  const list = document.getElementById('trace-list');
  const filtered = currentFilter === 'all' ? allEvents : allEvents.filter(e => currentFilter === 'blocked' ? e.blocked : e.tier === currentFilter);

  if (filtered.length === 0) {
    list.innerHTML = `<div class="trace-placeholder">No ${currentFilter === 'all' ? '' : currentFilter + ' '}requests yet…</div>`;
    return;
  }

  const scrollTop = list.scrollTop;
  list.innerHTML = filtered.slice(0, 80).map(event => {
    const key = String(event.timestamp);
    const isExpanded = expandedRows.has(key);
    const time = new Date(event.timestamp * 1000).toLocaleTimeString();
    const wm = event.watermarked ? '<span class="wm-badge">⭐</span>' : '';
    return `
      <div class="trace-row ${isExpanded ? 'expanded' : ''}" data-key="${key}">
        <div class="trace-row-main">
          <span class="trace-time">${time}</span>
          <span class="trace-client">${event.client_id}</span>
          <span>${verdictBadge(event)}${wm}</span>
          <span class="trace-threat">${event.threat_index !== null ? event.threat_index.toFixed(0) : '—'}</span>
          <span class="trace-signal">${signalSummary(event)}</span>
          <span class="trace-caret">▶</span>
        </div>
        <div class="trace-detail">${isExpanded ? renderWaterfall(event) : ''}</div>
      </div>`;
  }).join('');
  list.scrollTop = scrollTop;

  list.querySelectorAll('.trace-row').forEach(row => {
    row.querySelector('.trace-row-main').addEventListener('click', () => {
      const key = row.dataset.key;
      if (expandedRows.has(key)) expandedRows.delete(key); else expandedRows.add(key);
      renderTraceList();
    });
  });
}

document.getElementById('trace-filters').addEventListener('click', (e) => {
  if (!e.target.classList.contains('filter-pill')) return;
  document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
  e.target.classList.add('active');
  currentFilter = e.target.dataset.filter;
  renderTraceList();
});

// ---------------- Flash + status ----------------
function maybeFlash(criticalCount) {
  if (criticalCount > lastCriticalCount) {
    const overlay = document.getElementById('flash-overlay');
    overlay.classList.remove('flash'); void overlay.offsetWidth; overlay.classList.add('flash');
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

  const recent = stats.recent_threat_indices || [];
  updateGauge(recent.length ? recent[recent.length - 1] : 0);

  timelineChart.data.labels = recent.map((_, i) => i);
  timelineChart.data.datasets[0].data = recent;
  timelineChart.update('none');

  donutChart.data.datasets[0].data = [counts.normal || 0, counts.elevated || 0, counts.critical || 0];
  donutChart.update('none');
  renderDonutLegend(counts);

  updateBanner(stats);

  allEvents = stats.recent_events || [];
  renderTraceList();

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

poll();
setInterval(poll, POLL_MS);
