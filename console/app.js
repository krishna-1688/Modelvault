// ModelVault Live Console -- polls /admin/stats on the SAME origin (this
// page is served directly by the gateway, see modelvault/gateway/api.py's
// StaticFiles mount) and renders everything client-side. No build step, no
// framework -- plain DOM + Chart.js, so it works offline and loads instantly.

const params = new URLSearchParams(window.location.search);
const API_KEY = params.get('key') || '';
const POLL_MS = 1000;
const TIER_NORMAL_MAX = 35;
const TIER_ELEVATED_MAX = 70;
const TIER_COLORS = { normal: '#22c55e', elevated: '#f59e0b', critical: '#ef4444' };

let lastEventTimestamp = 0;
let lastCriticalCount = 0;
let previousKpis = {};

document.getElementById('footer-url').textContent = window.location.origin;

// ---------------- Clock ----------------
function tickClock() {
  document.getElementById('clock').textContent = new Date().toLocaleTimeString();
}
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
      [0, TIER_NORMAL_MAX, 'rgba(34,197,94,0.06)'],
      [TIER_NORMAL_MAX, TIER_ELEVATED_MAX, 'rgba(245,158,11,0.06)'],
      [TIER_ELEVATED_MAX, 100, 'rgba(239,68,68,0.08)'],
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
const gradient = timelineCtx.createLinearGradient(0, 0, 0, 270);
gradient.addColorStop(0, 'rgba(34,211,238,0.35)');
gradient.addColorStop(1, 'rgba(34,211,238,0.02)');

const timelineChart = new Chart(timelineCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      data: [],
      borderColor: '#22d3ee',
      backgroundColor: gradient,
      fill: true,
      tension: 0.35,
      pointRadius: 0,
      borderWidth: 2.5,
    }],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: { legend: { display: false }, tooltip: { intersect: false, mode: 'index' } },
    scales: {
      y: { min: 0, max: 100, grid: { color: 'rgba(148,163,184,0.08)' }, ticks: { color: '#64748b' } },
      x: { grid: { display: false }, ticks: { color: '#64748b', maxTicksLimit: 8 } },
    },
  },
  plugins: [bandsPlugin],
});

const donutCtx = document.getElementById('donut-chart').getContext('2d');
const donutChart = new Chart(donutCtx, {
  type: 'doughnut',
  data: {
    labels: ['Normal', 'Elevated', 'Critical'],
    datasets: [{
      data: [0, 0, 0],
      backgroundColor: [TIER_COLORS.normal, TIER_COLORS.elevated, TIER_COLORS.critical],
      borderWidth: 0,
      hoverOffset: 6,
    }],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    cutout: '70%',
    animation: { duration: 300 },
    plugins: { legend: { display: false } },
  },
});

function renderDonutLegend(counts) {
  const total = (counts.normal || 0) + (counts.elevated || 0) + (counts.critical || 0);
  const legend = document.getElementById('donut-legend');
  legend.innerHTML = ['normal', 'elevated', 'critical'].map(tier => {
    const pct = total ? Math.round(((counts[tier] || 0) / total) * 100) : 0;
    return `<span><span class="legend-dot" style="background:${TIER_COLORS[tier]}"></span>${tier} ${pct}%</span>`;
  }).join('');
}

// ---------------- Gauge ----------------
const GAUGE_CIRCUMFERENCE = 251.2;
function updateGauge(value) {
  const clamped = Math.max(0, Math.min(100, value));
  const offset = GAUGE_CIRCUMFERENCE * (1 - clamped / 100);
  const arc = document.getElementById('gauge-arc');
  arc.style.strokeDashoffset = offset;

  let color, tierText;
  if (clamped <= TIER_NORMAL_MAX) { color = '#22c55e'; tierText = 'NORMAL'; }
  else if (clamped <= TIER_ELEVATED_MAX) { color = '#f59e0b'; tierText = 'ELEVATED'; }
  else { color = '#ef4444'; tierText = 'CRITICAL'; }
  arc.style.stroke = color;

  document.getElementById('gauge-value').textContent = Math.round(clamped);
  document.getElementById('gauge-value').style.color = color;
  document.getElementById('gauge-tier').textContent = tierText;
}

// ---------------- KPI bump animation ----------------
function setKpi(id, value) {
  const el = document.getElementById(id);
  if (previousKpis[id] !== undefined && previousKpis[id] !== value) {
    el.classList.remove('bump'); void el.offsetWidth; el.classList.add('bump');
  }
  previousKpis[id] = value;
  el.textContent = value;
}

// ---------------- Pipeline flow ----------------
function pulseNode(nodeId, color) {
  const node = document.getElementById(nodeId);
  node.style.setProperty('--pipe-color', color);
  node.classList.add('active');
  clearTimeout(node._t);
  node._t = setTimeout(() => node.classList.remove('active'), 1400);
}
function pulseLine(lineId) {
  const line = document.getElementById(lineId);
  line.classList.remove('flow'); void line.offsetWidth; line.classList.add('flow');
}

function updatePipeline(event) {
  if (!event) return;
  pulseNode('node-1', '#22d3ee');
  setTimeout(() => pulseLine('line-1'), 50);

  setTimeout(() => {
    pulseNode('node-2', '#22d3ee');
    document.getElementById('node-2-sub').textContent = event.threat_index.toFixed(0);
    pulseLine('line-2');
  }, 150);

  setTimeout(() => {
    const tierColor = TIER_COLORS[event.tier] || '#6366f1';
    pulseNode('node-3', tierColor);
    document.getElementById('node-3-sub').textContent = event.tier;
    pulseLine('line-3');
  }, 300);

  setTimeout(() => {
    if (event.watermarked) {
      pulseNode('node-4', '#facc15');
      document.getElementById('node-4-sub').textContent = 'triggered';
    } else {
      document.getElementById('node-4-sub').textContent = 'clear';
    }
  }, 450);
}

// ---------------- Alert banner ----------------
function updateBanner(stats) {
  const counts = stats.tier_counts;
  const total = (counts.normal || 0) + (counts.elevated || 0) + (counts.critical || 0);
  const banner = document.getElementById('alert-banner');
  const icon = document.getElementById('alert-icon');
  const text = document.getElementById('alert-text');

  if (total === 0) {
    banner.className = 'alert-banner alert-ok';
    icon.textContent = '⏳'; text.textContent = 'Waiting for traffic…';
    return;
  }
  const suspiciousShare = ((counts.elevated || 0) + (counts.critical || 0)) / total;
  if (suspiciousShare > 0.5) {
    banner.className = 'alert-banner alert-critical';
    icon.textContent = '🚨';
    text.textContent = `HIGH ALERT — ${Math.round(suspiciousShare*100)}% of recent traffic is elevated/critical. Likely active extraction attempt.`;
  } else if (suspiciousShare > 0.2) {
    banner.className = 'alert-banner alert-warning';
    icon.textContent = '⚠️';
    text.textContent = `Elevated suspicion — ${Math.round(suspiciousShare*100)}% of recent traffic is elevated/critical tier.`;
  } else {
    banner.className = 'alert-banner alert-ok';
    icon.textContent = '✅';
    text.textContent = `Traffic looks normal (${Math.round(suspiciousShare*100)}% elevated/critical).`;
  }
}

// ---------------- Activity log ----------------
function appendLogLines(events) {
  const terminal = document.getElementById('log-terminal');
  const newest = events.filter(e => e.timestamp > lastEventTimestamp).sort((a, b) => a.timestamp - b.timestamp);
  if (newest.length === 0) return;

  const placeholder = terminal.querySelector('.log-placeholder');
  if (placeholder) placeholder.remove();

  newest.forEach(e => {
    const line = document.createElement('div');
    line.className = 'log-line';
    const time = new Date(e.timestamp * 1000).toLocaleTimeString();
    const wm = e.watermarked ? '<span class="log-wm">⭐ watermarked</span>' : '';
    line.innerHTML = `
      <span class="log-time">${time}</span>
      <span class="log-client">${e.client_id}</span>
      <span class="log-tier ${e.tier}">● ${e.tier}</span>
      <span class="log-threat">${e.threat_index.toFixed(1)}</span>
      ${wm}
    `;
    terminal.insertBefore(line, terminal.firstChild);
  });
  lastEventTimestamp = Math.max(...newest.map(e => e.timestamp));

  while (terminal.children.length > 60) terminal.removeChild(terminal.lastChild);
}

// ---------------- Flash overlay on new critical ----------------
function maybeFlash(criticalCount) {
  if (criticalCount > lastCriticalCount) {
    const overlay = document.getElementById('flash-overlay');
    overlay.classList.remove('flash'); void overlay.offsetWidth; overlay.classList.add('flash');
  }
  lastCriticalCount = criticalCount;
}

// ---------------- Status ----------------
function setStatus(online) {
  const pill = document.getElementById('status-pill');
  const text = document.getElementById('status-text');
  pill.classList.toggle('offline', !online);
  text.textContent = online ? 'LIVE' : 'OFFLINE';
}

// ---------------- Main poll loop ----------------
async function poll() {
  try {
    const res = await fetch('/admin/stats', { headers: { 'X-API-Key': API_KEY } });
    if (!res.ok) throw new Error('bad status ' + res.status);
    const stats = await res.json();
    setStatus(true);
    render(stats);
  } catch (err) {
    setStatus(false);
  }
}

function render(stats) {
  const counts = stats.tier_counts || { normal: 0, elevated: 0, critical: 0 };
  const total = (counts.normal || 0) + (counts.elevated || 0) + (counts.critical || 0);
  const suspiciousPct = total ? Math.round(((counts.elevated || 0) + (counts.critical || 0)) / total * 100) : 0;

  setKpi('kpi-requests', stats.total_requests);
  setKpi('kpi-clients', stats.total_clients);
  setKpi('kpi-watermarks', stats.watermark_triggers);
  setKpi('kpi-suspicious', suspiciousPct + '%');

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
  appendLogLines(events);
  if (events.length > 0) updatePipeline(events[0]); // most recent first from the API

  maybeFlash(counts.critical || 0);

  const toggle = document.getElementById('defense-toggle');
  if (document.activeElement !== toggle) toggle.checked = stats.defense_enabled;
  document.getElementById('defense-label').textContent = stats.defense_enabled ? 'DEFENSE ON' : 'DEFENSE OFF';
}

// ---------------- Defense toggle ----------------
document.getElementById('defense-toggle').addEventListener('change', async (e) => {
  try {
    await fetch('/admin/toggle-defense', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
      body: JSON.stringify({ enabled: e.target.checked }),
    });
  } catch (err) { /* next poll will resync the toggle state */ }
});

poll();
setInterval(poll, POLL_MS);
