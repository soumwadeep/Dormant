/* Dormant · mission control
 *
 * No build step, no framework, no CDN. three.js is vendored next to this file so
 * the dashboard renders with the network off, which matters because the product it
 * is visualising is the one that never needs the network on.
 */

import * as THREE from './vendor/three.module.js';

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const isTouch = window.matchMedia('(hover: none)').matches;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const state = { data: null, points: [], sort: 'conf', filter: '', shown: 60, job: null, source: 'api' };

/* ------------------------------------------------------------------ prefs */

const prefs = {
  get(key, fallback) { try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } },
};

function applyPrefs() {
  const root = document.documentElement;
  // Query params win once, so a shared link can open the light theme or the cyan
  // accent without the visitor touching the toggles.
  const params = new URLSearchParams(location.search);
  const forcedTheme = params.get('theme');
  const forcedAccent = params.get('accent');
  if (forcedTheme === 'dark' || forcedTheme === 'light') { prefs.set('dormant.theme', forcedTheme); }
  if (['amber', 'cyan', 'violet', 'moss'].includes(forcedAccent)) { prefs.set('dormant.accent', forcedAccent); }
  root.dataset.theme = prefs.get('dormant.theme', 'dark');
  root.dataset.accent = prefs.get('dormant.accent', 'amber');
  root.dataset.density = prefs.get('dormant.density', 'roomy');
  $$('[data-theme-set]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.themeSet === root.dataset.theme)));
  $$('[data-accent-set]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.accentSet === root.dataset.accent)));
  const dens = $('#densityBtn');
  if (dens) {
    dens.setAttribute('aria-pressed', String(root.dataset.density === 'compact'));
    dens.textContent = root.dataset.density === 'compact' ? 'Roomy' : 'Compact';
  }
  const meta = document.querySelector('meta[name="theme-color"]') || (() => {
    const m = document.createElement('meta'); m.name = 'theme-color'; document.head.appendChild(m); return m;
  })();
  meta.content = getComputedStyle(root).getPropertyValue('--bg').trim() || '#05070b';
  field.recolor();
}

$$('[data-theme-set]').forEach(btn => btn.addEventListener('click', () => {
  prefs.set('dormant.theme', btn.dataset.themeSet); applyPrefs();
}));
$$('[data-accent-set]').forEach(btn => btn.addEventListener('click', () => {
  prefs.set('dormant.accent', btn.dataset.accentSet); applyPrefs();
}));
$('#densityBtn')?.addEventListener('click', () => {
  const cur = document.documentElement.dataset.density;
  prefs.set('dormant.density', cur === 'compact' ? 'roomy' : 'compact');
  applyPrefs();
});

/* ------------------------------------------------------------------ helpers */

const gb = n => ((n || 0) / 1e9);
const fmtGb = n => gb(n).toFixed(n && gb(n) < 10 ? 2 : 1);
const pct = n => (n * 100).toFixed(n >= 0.1 ? 0 : 1) + '%';
const day = s => (s ? String(s).slice(0, 10) : '—');
const daysSince = iso => {
  if (!iso) return null;
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return null;
  return Math.max(0, Math.round((Date.now() - t) / 86400000));
};
const compact = n => n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n);

function toast(msg, ms = 3400) {
  const el = $('#toast'); el.textContent = msg; el.hidden = false;
  requestAnimationFrame(() => el.classList.add('show'));
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.classList.remove('show'); setTimeout(() => { el.hidden = true; }, 320); }, ms);
}

function logLine(msg, kind = 'info') {
  const ul = $('#log'); if (!ul) return;
  const li = document.createElement('li');
  li.className = 'new';
  li.innerHTML = `<time>${new Date().toLocaleTimeString([], { hour12: false })}</time><span class="lg-${kind}">${msg}</span>`;
  ul.prepend(li);
  while (ul.children.length > 40) ul.lastElementChild.remove();
}

function animateNumber(el, to, fmt = v => v.toFixed(1)) {
  if (!el) return;
  if (reduceMotion) { el.textContent = fmt(to); return; }
  const from = parseFloat(el.textContent.replace(/[^0-9.\-]/g, '')) || 0;
  const t0 = performance.now(), dur = 900;
  const step = now => {
    const k = Math.min(1, (now - t0) / dur);
    const eased = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(from + (to - from) * eased);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

async function getJSON(url) {
  const r = await fetch(url, { headers: { accept: 'application/json' } });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}
async function postJSON(url, body) {
  const r = await fetch(url, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

/* ------------------------------------------------------------------ 3D field */

const field = {
  ready: false, points: null, ring: null, core: null, scene: null, camera: null, renderer: null,
  group: null, drag: null, rot: { x: 0.32, y: 0 }, target: { x: 0.32, y: 0 }, spin: 0,
  colors: { acc: '#f0b429', alive: '#5fe3a1', bg: '#05070b' },

  init() {
    const canvas = $('#field');
    if (!canvas || !canvas.clientWidth) return;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'low-power' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x000000, 0.0016);
    this.camera = new THREE.PerspectiveCamera(48, 1, 1, 4000);
    this.camera.position.set(0, 130, 320);
    this.camera.lookAt(0, 0, 0);
    this.group = new THREE.Group();
    this.scene.add(this.group);

    const ringGeo = new THREE.TorusGeometry(150, 0.7, 6, 180);
    const ringMat = new THREE.MeshBasicMaterial({ color: this.colors.acc, transparent: true, opacity: 0.3 });
    this.ring = new THREE.Mesh(ringGeo, ringMat);
    this.ring.rotation.x = Math.PI / 2;
    this.group.add(this.ring);

    const grid = new THREE.PolarGridHelper(160, 8, 6, 48, 0x33445c, 0x1d2836);
    grid.material.transparent = true; grid.material.opacity = 0.34;
    this.group.add(grid);

    const coreGeo = new THREE.IcosahedronGeometry(11, 1);
    const coreMat = new THREE.MeshBasicMaterial({ color: this.colors.acc, wireframe: true, transparent: true, opacity: 0.55 });
    this.core = new THREE.Mesh(coreGeo, coreMat);
    this.group.add(this.core);

    this.bindPointer(canvas);
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.ready = true;
    this.loop();
  },

  bindPointer(canvas) {
    canvas.addEventListener('pointerdown', e => {
      this.drag = { x: e.clientX, y: e.clientY, rx: this.target.y, ry: this.target.x };
      canvas.setPointerCapture(e.pointerId);
      this.spin = 0;
    });
    canvas.addEventListener('pointermove', e => {
      if (this.drag) {
        this.target.y = this.drag.rx + (e.clientX - this.drag.x) * 0.006;
        this.target.x = Math.max(-1.2, Math.min(1.2, this.drag.ry + (e.clientY - this.drag.y) * 0.004));
      } else if (!isTouch) this.hover(e);
    });
    const stop = () => { this.drag = null; this.spin = reduceMotion ? 0 : 0.0016; };
    canvas.addEventListener('pointerup', stop);
    canvas.addEventListener('pointercancel', stop);
    canvas.addEventListener('pointerleave', () => { stop(); $('#tip').hidden = true; });
  },

  hover(e) {
    if (!this.points || !this.points.count) return;
    const rect = e.target.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    ray.params.Points.threshold = 6;
    const hits = ray.intersectObject(this.points);
    const tip = $('#tip');
    if (!hits.length) { tip.hidden = true; return; }
    const i = hits[0].index;
    const d = this.meta[i];
    if (!d) { tip.hidden = true; return; }
    tip.hidden = false;
    tip.style.left = Math.min(rect.width - 250, e.clientX - rect.left + 14) + 'px';
    tip.style.top = (e.clientY - rect.top + 12) + 'px';
    tip.innerHTML = `<b>${d.ext || 'file'}</b> · ${fmtGb(d.size)} GB<br>age ${Math.round(d.age)} d<br>p(reopen) ${d.p.toFixed(3)}`;
  },

  setData(points) {
    this.meta = points;
    if (!this.ready) return;
    const limit = isTouch ? 2400 : 6000;
    const pts = points.slice(0, limit);
    const pos = new Float32Array(pts.length * 3);
    const col = new Float32Array(pts.length * 3);
    const acc = new THREE.Color(this.colors.acc), alive = new THREE.Color(this.colors.alive), tmp = new THREE.Color();
    const golden = Math.PI * (3 - Math.sqrt(5));
    pts.forEach((p, i) => {
      const r = 26 + Math.min(p.age, 1400) / 1400 * 132;
      const a = i * golden;
      const h = (0.5 - p.p) * 150;
      pos[i * 3] = Math.cos(a) * r;
      pos[i * 3 + 1] = h;
      pos[i * 3 + 2] = Math.sin(a) * r;
      tmp.copy(alive).lerp(acc, Math.max(0, Math.min(1, 1 - p.p)));
      col[i * 3] = tmp.r; col[i * 3 + 1] = tmp.g; col[i * 3 + 2] = tmp.b;
    });
    if (this.points) { this.group.remove(this.points); this.points.geometry.dispose(); this.points.material.dispose(); }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const mat = new THREE.PointsMaterial({
      size: isTouch ? 2.6 : 3.4, vertexColors: true, transparent: true, opacity: 0.94,
      sizeAttenuation: true, depthWrite: false,
    });
    this.points = new THREE.Points(geo, mat);
    this.group.add(this.points);
    $('#fieldCount').textContent = `${pts.length.toLocaleString()} of ${points.length.toLocaleString()} files plotted`;
  },

  recolor() {
    const cs = getComputedStyle(document.documentElement);
    this.colors.acc = cs.getPropertyValue('--acc').trim() || '#f0b429';
    this.colors.alive = cs.getPropertyValue('--good').trim() || '#5fe3a1';
    if (this.ring) this.ring.material.color.set(this.colors.acc);
    if (this.core) this.core.material.color.set(this.colors.acc);
    if (this.points && state.points.length) this.setData(state.points);
  },

  resize() {
    const wrap = $('.field-wrap'); if (!wrap || !this.renderer) return;
    const w = wrap.clientWidth, h = wrap.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  },

  loop() {
    requestAnimationFrame(() => this.loop());
    if (!this.renderer || !this.scene) return;
    if (!this.drag) this.target.y += this.spin || 0;
    this.rot.y += (this.target.y - this.rot.y) * 0.08;
    this.rot.x += (this.target.x - this.rot.x) * 0.08;
    if (this.group) { this.group.rotation.y = this.rot.y; this.group.rotation.x = this.rot.x; }
    if (this.core) this.core.rotation.y += 0.004;
    if (this.points && !reduceMotion) this.points.material.opacity = 0.82 + Math.sin(performance.now() / 900) * 0.12;
    this.renderer.render(this.scene, this.camera);
  },
};

/* ------------------------------------------------------------------ charts */

function lineChart(canvas, series, opts = {}) {
  const dpr = Math.min(devicePixelRatio, 2);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const cs = getComputedStyle(document.documentElement);
  const ink3 = cs.getPropertyValue('--ink-3').trim();
  const hair = cs.getPropertyValue('--hair').trim();

  const all = series.flatMap(s => s.values).filter(v => Number.isFinite(v));
  if (!all.length) return;
  const minY = opts.minY ?? Math.min(...all), maxY = opts.maxY ?? Math.max(...all);
  const pad = { l: 6, r: 6, t: 10, b: 14 };
  const X = i => pad.l + (i / Math.max(1, (opts.xCount || (series[0].values.length - 1)))) * (w - pad.l - pad.r);
  const Y = v => h - pad.b - ((v - minY) / Math.max(1e-9, maxY - minY)) * (h - pad.t - pad.b);

  ctx.strokeStyle = hair; ctx.lineWidth = 1;
  for (let g = 0; g <= 3; g++) {
    const y = pad.t + (g / 3) * (h - pad.t - pad.b);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
  }

  series.forEach(s => {
    ctx.beginPath(); ctx.lineWidth = s.width || 1.8; ctx.strokeStyle = s.color; ctx.lineJoin = 'round';
    let started = false;
    s.values.forEach((v, i) => {
      if (!Number.isFinite(v)) return;
      const x = X(i), y = Y(v);
      started ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), started = true);
    });
    if (s.fill) {
      ctx.lineTo(X(s.values.length - 1), h - pad.b); ctx.lineTo(X(0), h - pad.b); ctx.closePath();
      ctx.fillStyle = s.fill; ctx.fill();
      ctx.beginPath();
      s.values.forEach((v, i) => { if (!Number.isFinite(v)) return; const x = X(i), y = Y(v); started ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), started = true); });
    }
    ctx.stroke();
    if (s.dotLast) {
      const li = s.values.length - 1;
      ctx.beginPath(); ctx.arc(X(li), Y(s.values[li]), 3, 0, Math.PI * 2); ctx.fillStyle = s.color; ctx.fill();
    }
  });

  if (opts.labels) {
    ctx.fillStyle = ink3; ctx.font = '9px ui-monospace, monospace';
    opts.labels.forEach(t => {
      const x = X(t.i);
      ctx.textAlign = t.last ? 'right' : (t.i === 0 ? 'left' : 'center');
      ctx.fillText(t.text, t.last ? w - pad.r : x, h - 2);
    });
    ctx.textAlign = 'left';
  }
}

/* ------------------------------------------------------------------ renderers */

function renderBrief(d) {
  const b = d.brief || {};
  const l1 = $('#brief1'), l2 = $('#brief2');
  if (l1.textContent !== b.line1) {
    l1.textContent = b.line1 || 'No brief yet.';
    l2.textContent = b.line2 || '';
    [l1, l2].forEach(el => { el.classList.remove('in'); void el.offsetWidth; el.classList.add('in'); });
  }
  const meta = $('#briefMeta');
  meta.innerHTML = (b.detail || []).map(x => `<span>${x}</span>`).join('')
    + (d.threshold ? `<span>operating point ${pct(d.threshold)} dormancy</span>` : '')
    + (d.precision ? `<span>holdout precision ${pct(d.precision)}</span>` : '');
}

function renderLedger(d) {
  const f = d.flagged || {};
  animateNumber($('#figGb'), gb(f.bytes));
  animateNumber($('#figFiles'), f.files || 0, v => Math.round(v).toLocaleString());
  animateNumber($('#figDays'), f.days_bought || 0, v => Math.round(v).toString());
  const a = d.archive || {};
  $('#assurance').textContent = a.reversible_manifests
    ? `${a.archived_files} files staged and fully recoverable across ${a.reversible_manifests} open manifest(s). No delete code path exists.`
    : 'Dormant moves files into a dated staging folder and keeps a manifest. It has no delete code path.';
}

function renderForecast(d) {
  const f = d.forecast || {};
  const arc = $('#gaugeFill');
  const frac = Math.max(0, Math.min(1, (f.free_pct || 0) / 100));
  arc.style.strokeDashoffset = String(258 * (1 - frac));
  $('#gaugePct').textContent = (f.free_pct != null ? f.free_pct.toFixed(0) + '%' : '--');
  $('#gaugeLbl').textContent = f.free_gb != null ? `${f.free_gb} GB free` : 'free';
  $('#forecastSource').textContent = f.source || 'no snapshots';

  const rows = [
    ['volume', f.volume || '—'],
    ['growth', f.growth_gb_per_day != null ? `${f.growth_gb_per_day.toFixed(2)} GB/day` : '—'],
    ['per week', f.growth_gb_per_week != null ? `${f.growth_gb_per_week.toFixed(1)} GB` : '—'],
    ['warning date', f.full_date || 'not filling'],
    ['snapshots', f.snapshots != null ? `${f.snapshots} nights` : '—'],
  ];
  $('#forecastKv').innerHTML = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');

  const proj = f.projections || {};
  $('#projections').innerHTML = Object.entries(proj).map(([k, v]) =>
    `<div><b>${v}</b><i>${k.replace('free_in_', 'in ').replace('_gb', ' GB')}</i></div>`).join('');

  const curve = (f.curve || []).map(c => c.cumulative_gb);
  if (curve.length) {
    const cs = getComputedStyle(document.documentElement);
    const accent = cs.getPropertyValue('--acc').trim() || '#f0b429';
    lineChart($('#curve'), [{ values: curve, color: accent, fill: accent + '22', dotLast: true }],
      { xCount: curve.length - 1, labels: [{ text: day((f.curve || [])[0]?.date), i: 0 }, { text: day((f.curve || [])[(f.curve || []).length - 1]?.date), i: curve.length - 1 }] });
  }
}

function renderBench(d) {
  const rows = (d.bench || []).slice().reverse();
  const best = Math.max(0, ...rows.map(r => r.auc || 0));
  const seen = new Set();
  const uniq = rows.filter(r => { if (seen.has(r.model)) return false; seen.add(r.model); return true; });
  $('#benchSplit').textContent = d.eval_split ? `holdout ${d.eval_split.test || 0} rows · cut ${day(d.eval_split.cut)}` : 'not benchmarked';
  $('#benchRows').innerHTML = uniq.map(r => {
    let note = '';
    try { note = JSON.parse(r.notes || '{}').source || ''; } catch { note = (r.notes || '').slice(0, 20); }
    return `<div class="bench-row ${r.auc === best && best ? 'winner' : ''}">
      <span class="bench-name">${r.model}</span>
      <span class="bench-bar"><i style="width:${((r.auc || 0) * 100).toFixed(1)}%"></i></span>
      <span class="bench-cell">${r.auc != null ? Number(r.auc).toFixed(3) : '—'}<small>auc</small></span>
      <span class="bench-cell">${r.precision != null ? pct(r.precision) : '—'}<small>prec</small></span>
      <span class="bench-cell">${r.recall != null ? pct(r.recall) : '—'}<small>rec</small></span>
      <span class="bench-cell" title="${note}">${r.reclaimed != null ? Number(r.reclaimed).toFixed(1) + 'GB' : '—'}<small>reclaim</small></span>
    </div>`;
  }).join('') || '<div class="bench-row"><span class="bench-name">no benchmark yet</span><span class="bench-bar"><i></i></span><span class="bench-cell">—<small>auc</small></span><span class="bench-cell">—<small>prec</small></span><span class="bench-cell">—<small>rec</small></span><span class="bench-cell">—<small>reclaim</small></span></div>';

  const sc = d.scarcity || [];
  if (sc.length) {
    const keys = [['tabpfn_auc', 'var(--acc)', 'TabPFN'], ['random_forest_auc', 'var(--good)', 'random forest'],
                  ['hist_gb_auc', 'var(--warn)', 'hist gb'], ['logistic_auc', 'var(--ink-3)', 'logistic']];
    const cs = getComputedStyle(document.documentElement);
    const vals = sc.flatMap(r => keys.map(([k]) => r[k])).filter(v => Number.isFinite(v));
    const lo = Math.max(0, Math.min(...vals) - 0.03), hi = Math.min(1, Math.max(...vals) + 0.03);
    lineChart($('#scarcity'), keys.map(([k, color]) => ({
      values: sc.map(r => r[k]), color: cs.getPropertyValue(color.replace(/var\(|\)/g, '')) || '#888', width: 2, dotLast: true,
    })), { minY: lo, maxY: hi, xCount: sc.length - 1,
           labels: sc.map((r, i) => ({ text: String(r.n_train), i, last: i === sc.length - 1 }))
             .filter((_, i) => i % 2 === 0 || i === sc.length - 1) });
    $('#scarcityLegend').innerHTML = keys.map(([k, color, label]) => {
      const last = sc[sc.length - 1][k];
      return `<span><i style="background:${color}"></i>${label} ${last != null ? Number(last).toFixed(3) : ''}</span>`;
    }).join('');
  }
}

function renderRegister(d) {
  const now = Date.now();
  let rows = (d.top || []).map(r => ({
    ...r, dormant: 1 - r.p_reopen,
    shut: r.last_used_at ? Math.round((now - Date.parse(r.last_used_at)) / 86400000) : Math.round((now - Date.parse(r.created_at)) / 86400000),
  }));
  if (state.filter) {
    const q = state.filter.toLowerCase();
    rows = rows.filter(r => (r.path || r.name).toLowerCase().includes(q));
  }
  rows.sort((a, b) => state.sort === 'size' ? b.size_bytes - a.size_bytes : b.dormant - a.dormant);
  const slice = rows.slice(0, state.shown);
  $('#registerBody').innerHTML = slice.map(r => `
    <tr data-id="${r.id}">
      <td class="fname"><b>${r.name}</b><span>${(r.path || '').replace(r.name, '')}</span></td>
      <td>${fmtGb(r.size_bytes)} GB</td>
      <td>${r.last_used_at ? day(r.last_used_at) : '<span class="never">never</span>'}</td>
      <td>${r.shut.toLocaleString()} d</td>
      <td class="num"><span class="conf"><span class="conf-bar"><i style="width:${(r.dormant * 100).toFixed(0)}%"></i></span><b>${pct(r.dormant)}</b></span></td>
    </tr>`).join('');
  $('#registerCount').textContent = `${rows.length.toLocaleString()} dormant-ranked files · showing ${slice.length}`;
  $('#moreBtn').style.display = rows.length > state.shown ? '' : 'none';
}

function renderStatus(d) {
  const chip = $('#modeChip');
  chip.textContent = (d.mode || 'real').toUpperCase();
  chip.dataset.mode = d.mode || 'real';
  const c = d.counts || {};
  $('#statusText').textContent = `${compact(c.files)} files indexed`;
  $('#footScan').textContent = d.last_scan
    ? `last scan ${day(d.last_scan.started_at)} · ${compact(d.last_scan.files_seen)} walked · ${compact(d.last_scan.with_history)} with recorded opens`
    : (d.mode === 'demo' ? 'synthetic corpus · seeded, never scanned' : 'index empty');
}

/* ------------------------------------------------------------------ data */

async function load() {
  try {
    const [d, f] = await Promise.all([getJSON('/api/state'), getJSON('/api/field?limit=6000').catch(() => ({ points: [] }))]);
    state.data = d; state.points = f.points || []; state.source = 'api';
  } catch (err) {
    logLine(`api unavailable (${err.message}); falling back to the bundled snapshot`, 'warn');
    try {
      const d = await getJSON('/static/snapshot.json');
      state.data = d; state.points = d.field || []; state.source = 'snapshot';
    } catch (e2) {
      $('#brief1').textContent = 'Dormant engine is not running. Start it with: python cli.py serve';
      $('#brief2').textContent = 'The dashboard reads a local FastAPI on 127.0.0.1; nothing is fetched from the internet.';
      return;
    }
  }
  renderStatus(state.data);
  renderBrief(state.data);
  renderLedger(state.data);
  renderForecast(state.data);
  renderBench(state.data);
  renderRegister(state.data);
  field.setData(state.points);
  if (!load._greeted) {
    load._greeted = true;
    const c = state.data.counts || {}, f = state.data.flagged || {};
    logLine(`engine online · ${compact(c.files)} files indexed, ${c.with_history || 0} with recorded opens`);
    if (f.files) logLine(`${f.files.toLocaleString()} files flagged dormant, ${(f.bytes / 1e9).toFixed(1)} GB`);
    if (state.data.mode === 'demo') logLine('demo corpus loaded: synthetic files, nothing real is on disk', 'warn');
  }
}

/* ------------------------------------------------------------------ actions */

$('#filter')?.addEventListener('input', e => { state.filter = e.target.value; renderRegister(state.data || {}); });
$('#sortConf').addEventListener('click', () => { state.sort = 'conf'; press('sortConf', 'sortSize'); renderRegister(state.data); });
$('#sortSize').addEventListener('click', () => { state.sort = 'size'; press('sortSize', 'sortConf'); renderRegister(state.data); });
function press(on, off) { $('#' + on).setAttribute('aria-pressed', 'true'); $('#' + off).setAttribute('aria-pressed', 'false'); }
$('#moreBtn').addEventListener('click', () => { state.shown += 200; renderRegister(state.data); });

async function pollJob(kind) {
  const led = $('#led'); led.className = 'led busy';
  $('#statusText').textContent = kind + ' running';
  const tick = async () => {
    let j;
    try { j = await getJSON('/api/job'); } catch { return finish(); }
    if (j.log && j.log.length) {
      const last = j.log[j.log.length - 1].msg;
      if (last !== pollJob._last) { pollJob._last = last; logLine(last); }
    }
    if (j.running) return setTimeout(tick, 1200);
    finish(j);
  };
  const finish = j => {
    led.className = 'led' + (j && j.error ? ' err' : ' idle');
    pollJob._last = null;
    if (j && j.error) logLine('failed: ' + j.error, 'err');
    else logLine(kind + ' complete');
    load();
  };
  tick();
}

$('#rescanBtn')?.addEventListener('click', async () => {
  try { await postJSON('/api/scan', {}); logLine('scan queued'); pollJob('scan'); }
  catch (e) { toast('scan could not start: ' + e.message); }
});
$('#rescoreBtn')?.addEventListener('click', async () => {
  try { await postJSON('/api/score', {}); logLine('score queued'); pollJob('score'); }
  catch (e) { toast('re-score failed: ' + e.message); }
});

$('#stageBtn')?.addEventListener('click', async () => {
  if (state.data?.mode === 'demo') return toast('Demo corpus files do not exist on disk, so staging is disabled. Run on a real index to stage.');
  try {
    const preview = await postJSON('/api/archive', { top: 50, max_gb: 10, confirm: false });
    const ok = window.confirm(
      `Dormant will move ${preview.would_stage} files (${fmtGb(preview.bytes)} GB) into a dated staging folder.\n` +
      `Nothing is deleted; undo restores every file to its exact original path.\n\nProceed?`);
    if (!ok) return logLine('archive preview dismissed');
    const res = await postJSON('/api/archive', { top: 50, max_gb: 10, confirm: true });
    logLine(`staged ${res.count} files, ${fmtGb(res.bytes)} GB, manifest ${res.manifest}`);
    toast(`staged ${res.count} files · manifest ${res.manifest}`);
    load();
  } catch (e) { toast('archive failed: ' + e.message); }
});

$('#undoBtn')?.addEventListener('click', async () => {
  try {
    const preview = await postJSON('/api/undo', { confirm: false });
    if (!preview.manifest) return toast('nothing to undo');
    const res = await postJSON('/api/undo', { manifest: preview.manifest, confirm: true });
    logLine(`restored ${res.restored} files from ${res.manifest}`);
    toast(`restored ${res.restored} files`);
    load();
  } catch (e) { toast('undo failed: ' + e.message); }
});

const SUGGEST = ['free up 12 gigabytes', 'when will my disk fill', 'which files are dead', 'compare the models', 'undo that'];
$('#agentSuggest').innerHTML = SUGGEST.map(s => `<button class="mini">${s}</button>`).join('');
$$('#agentSuggest button').forEach(b => b.addEventListener('click', () => { $('#agentInput').value = b.textContent; $('#agentForm').requestSubmit(); }));

$('#agentForm')?.addEventListener('submit', async e => {
  e.preventDefault();
  const text = $('#agentInput').value.trim();
  if (!text) return;
  const out = $('#agentOut');
  out.hidden = false; out.textContent = 'planning…';
  try {
    const res = await postJSON('/api/agent', { text });
    out.textContent = JSON.stringify(res, null, 2);
    $('#plannerTag').textContent = res.planner || 'keyword planner';
    logLine(`agent: ${res.intent} → ${(res.executed || []).map(c => c.tool).join(', ') || 'nothing'}`);
    if ((res.blocked || []).length) toast('file-moving tools stay planned until you confirm them in the ledger');
  } catch (err) { out.textContent = String(err); }
});

$$('.mobile-tabs button').forEach(b => b.addEventListener('click', () => {
  $$('.mobile-tabs button').forEach(x => x.classList.remove('on'));
  b.classList.add('on');
  document.documentElement.dataset.tab = b.dataset.goto;
  const panel = $(`.panel[data-panel="${b.dataset.goto}"]`);
  if (panel) panel.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
  if (b.dataset.goto === 'field') field.resize();
}));
if (window.matchMedia('(max-width: 820px)').matches) document.documentElement.dataset.tab = 'field';

/* ------------------------------------------------------------------ clock + boot */

setInterval(() => {
  const el = $('#clock');
  if (el) el.textContent = new Date().toLocaleTimeString([], { hour12: false });
}, 1000);

field.init();
applyPrefs();
load();
setInterval(() => { if (document.visibilityState === 'visible' && !state.job) load(); }, 60000);
window.addEventListener('resize', () => { field.resize(); renderForecast(state.data || {}); renderBench(state.data || {}); });
