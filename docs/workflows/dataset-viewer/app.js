// Dataset viewer front end: one shared top bar, four pages (#review #batch #quality #eval). Data from server.py.
'use strict';
const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const el = html => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };
const fx = (x, d = 2) => x == null ? '–' : (+x).toFixed(d);
const FPS = 30, sec = f => (f / FPS).toFixed(2);
const NAMES = ['左肩转', '左肩抬', '左肘', '左腕屈', '左腕转', '左夹爪', '右肩转', '右肩抬', '右肘', '右腕屈', '右腕转', '右夹爪'];
const TAGS = ['微调', '突变', '重抓', '放错位置', '停顿', '掉落', '卡住'];
const REF_NAME = 'v4-handover2-trimmed';
const RATES = [1, 2, 4, 8];
const LS = {get(k, d) { try { return JSON.parse(localStorage.getItem('dv_' + k)) ?? d } catch (e) { return d } },
            set(k, v) { try { localStorage.setItem('dv_' + k, JSON.stringify(v)) } catch (e) {} }};

async function api(path, body) {
  const r = await fetch(path, body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {});
  const j = await r.json().catch(() => ({error: r.statusText}));
  if (!r.ok) throw new Error(j.error || r.status);
  return j;
}
const status = (t, err) => { const s = $('#status'); s.textContent = t; s.className = err ? 'err' : ''; };
const qs = o => Object.entries(o).map(([k, v]) => k + '=' + encodeURIComponent(v)).join('&');
const frameUrl = (id, cam, ep, f) => `/api/frame?${qs({id, cam, ep, f: Math.max(0, Math.round(f))})}`;

// ---------- route ----------
const G = {ds: [], by: {}, git: null, cleanup: [], rate: 1};
// one playback speed for every video on every page, remembered; keys < > (or , .) step it
function setRate(r) {
  G.rate = r; LS.set('rate', r);
  $('#rate').value = r;
  for (const p of players) p.setRate(r);
  document.dispatchEvent(new Event('globalrate'));
  status(`播放速度 ${r}×`);
}
function route() {
  const [page, q] = (location.hash.slice(1) || 'review').split('?');
  return {page, q: Object.fromEntries(new URLSearchParams(q || ''))};
}
function setQuery(q) {
  const {page} = route();
  history.replaceState(null, '', '#' + page + (Object.keys(q).length ? '?' + new URLSearchParams(q) : ''));
}

// ---------- dataset picker ----------
function dsLabel(d) {
  if (!d) return '';
  const rev = d.revision ? 'HF ' + d.revision.slice(0, 8) : d.built ? `本地构建 ${d.built.started} · git ${d.built.git.commit.slice(0, 8)}` : '本地（无 HF 记录）';
  return `${d.id} · ${d.episodes} 条 · ${d.dims} 维 · ${rev}${d.building ? ' · 构建中' : ''}`;
}
function showInfo(ids, extra = '') {
  const g = G.git ? `代码 git ${G.git.commit.slice(0, 8)}${G.git.dirty ? '（有未提交修改）' : ''}` : '';
  $('#info').textContent = (extra || (ids.length === 1 ? dsLabel(G.by[ids[0]]) : '')) + (g ? ' · ' + g : '');
}
// mode 'single' | 'multi' | null; filter(d) limits the list; onchange(ids)
function picker(mode, ids, filter, onchange) {
  const box = $('#picker');
  box.innerHTML = '';
  if (!mode) return;
  const btn = el('<button></button>');
  const label = () => { btn.textContent = ids.length ? (mode === 'single' ? '数据集：' : `数据集（${ids.length}）：`) + ids.join('、') : '选择数据集…'; showInfo(ids); };
  label();
  box.append(btn);
  btn.onclick = ev => {
    ev.stopPropagation();
    if ($('.pop', box)) return $('.pop', box).remove();
    const pop = el('<div class="pop"><input placeholder="搜索数据集名"><div class="list"></div></div>');
    box.append(pop);
    const draw = () => {
      const s = $('input', pop).value.toLowerCase(), list = G.ds.filter(filter).filter(d => d.id.toLowerCase().includes(s));
      const row = d => `<label class="row"><input type="${mode === 'single' ? 'radio' : 'checkbox'}" name="pk" value="${esc(d.id)}" ${ids.includes(d.id) ? 'checked' : ''}>
        <span>${esc(d.id)}</span><span class="mute">${d.episodes} 条 · ${d.dims} 维 · ${d.revision ? d.revision.slice(0, 8) : '本地'}${d.building ? ' · <b class="err">构建中</b>' : ''}</span></label>`;
      const demo = list.filter(d => !d.eval), evl = list.filter(d => d.eval);
      $('.list', pop).innerHTML = (demo.length ? '<h4>训练 / 示范数据集</h4>' + demo.map(row).join('') : '') +
        (evl.length ? '<h4>评测数据集（有 success.csv）</h4>' + evl.map(row).join('') : '');
    };
    draw();
    $('input', pop).oninput = draw;
    $('input', pop).focus();
    pop.onclick = ev => ev.stopPropagation();
    pop.onchange = ev => {
      if (ev.target.name !== 'pk') return;
      const v = ev.target.value;
      if (mode === 'single') { ids = [v]; pop.remove(); } else ids = ev.target.checked ? [...ids, v] : ids.filter(x => x !== v);
      label(); onchange(ids);
    };
  };
}
document.addEventListener('click', () => $$('.pop').forEach(p => p.remove()));

// ---------- player: the dataset's own mp4s, seeked to the episode; no transcoding ----------
// A server-decoded still sits under each video and the video only shows once it has seeked, so mounting,
// switching episodes or looping never flashes black.
const players = new Set();
class Player {
  constructor(box, o) {
    Object.assign(this, {loop: true, range: null, onframe: null}, o, {rate: G.rate});
    this.f = Math.max(0, Math.min(o.n - 1, Math.round(o.start || 0)));
    this.cams = (o.cams || Object.keys(o.video)).filter(c => o.video[c]);
    this.box = box; this.mounted = false; this.playing = false;
    box.classList.add('player');
    box.innerHTML = this.cams.map((c, i) => `<div class="cam ${i ? 'small' : 'main'}"><img alt="" src="${frameUrl(o.id, c, o.ep, this.f)}"><video muted playsinline preload="none"></video></div>`).join('');
    this.vs = $$('video', box); this.imgs = $$('img', box);
    if (this.cams.length === 1) box.style.gridTemplateColumns = '1fr';
    box.onclick = () => this.toggle();
    players.add(this);
  }
  t0(i) { return this.video[this.cams[i]].t0; }
  at(i, f) { return this.t0(i) + (f + 0.3) / FPS; }
  mount() {
    if (this.mounted) return;
    this.mounted = true;
    this.vs.forEach((v, i) => {
      v.preload = 'auto';
      v.onloadedmetadata = () => { v.defaultPlaybackRate = v.playbackRate = this.rate; v.currentTime = this.at(i, this.f); };
      v.onseeked = () => v.classList.add('ready');
      v.src = this.video[this.cams[i]].url;
    });
  }
  unmount() {
    if (!this.mounted) return;
    this.mounted = false; this.playing = false;
    this.vs.forEach((v, i) => {
      v.pause(); v.onloadedmetadata = v.onseeked = null;
      this.imgs[i].src = frameUrl(this.id, this.cams[i], this.ep, this.f);  // the still under it shows the same frame
      v.classList.remove('ready'); v.removeAttribute('src'); v.load();
    });
  }
  destroy() { this.unmount(); players.delete(this); }
  seek(f) {
    this.f = Math.max(0, Math.min(this.n - 1, Math.round(f)));
    if (this.mounted) this.vs.forEach((v, i) => { if (v.readyState >= 1) v.currentTime = this.at(i, this.f); });
    this.onframe && this.onframe(this.f);
  }
  play() {
    this.mount(); this.playing = true; this.ended = false;
    const [lo, hi] = this.range || [0, this.n];
    if (this.f >= hi - 1 || this.f < lo) this.seek(lo);
    this.vs.forEach(v => { v.defaultPlaybackRate = v.playbackRate = this.rate; v.play().catch(() => {}); });
  }
  pause() { this.playing = false; this.vs.forEach(v => v.pause()); }
  toggle() { this.playing ? this.pause() : this.play(); }
  setRate(r) { this.rate = r; this.vs.forEach(v => { v.defaultPlaybackRate = v.playbackRate = r; }); }
  step() {
    const v = this.vs[0];
    if (!this.mounted || v.readyState < 1 || v.seeking) return;
    let f = Math.floor((v.currentTime - this.t0(0)) * FPS + 1e-3);
    if (this.playing) {
      const [lo, hi] = this.range || [0, this.n];
      if (f >= hi - 1 || f < lo - 3) {
        if (this.loop) { this.seek(lo); return; }
        this.pause(); this.ended = true;
      }
      for (let i = 1; i < this.vs.length; i++) {  // the wrist cameras follow the head camera
        const w = this.vs[i];
        if (w.readyState < 1 || w.seeking) continue;
        if (Math.abs((w.currentTime - this.t0(i)) * FPS - f) > 4) w.currentTime = this.at(i, f);
        if (w.paused && this.playing) w.play().catch(() => {});
      }
    }
    f = Math.max(0, Math.min(this.n - 1, f));
    if (f !== this.f) { this.f = f; this.onframe && this.onframe(f); }
  }
}
(function tick() { for (const p of players) p.step(); requestAnimationFrame(tick); })();

// ---------- joint curves: follower state (solid) vs leader command (dashed) of one arm, one lane per joint ----------
const curveCache = new Map();
const getCurves = (id, ep) => {
  const k = id + '|' + ep;
  if (!curveCache.has(k)) curveCache.set(k, api(`/api/curves?${qs({id, ep})}`));
  return curveCache.get(k);
};
const LANE_COLORS = ['#2a78d6', '#1baf7a', '#eb6834', '#9b59b6', '#c0392b', '#7a6a00'];
// curves fold one by one (button on each) or all at once (toolbar button); the all-at-once state is remembered
const allCurves = new Set();
function foldAll(v) { LS.set('fold', v); allCurves.forEach(c => c.fold(v)); }
function foldAllButton() {
  const b = el('<button title="折叠 / 展开所有关节曲线"></button>');
  const label = () => { b.textContent = LS.get('fold', false) ? '展开全部曲线' : '折叠全部曲线'; };
  b.onclick = () => { foldAll(!LS.get('fold', false)); label(); };
  label();
  return b;
}
class Curves {
  constructor(box, o) {
    Object.assign(this, {marks: {}, onseek: null}, o);
    this.arm = o.arm === 'R' ? 'R' : 'L';
    this.box = box;
    box.classList.add('curves');
    box.innerHTML = '<button class="foldsw" title="折叠 / 展开这条的关节曲线"></button><canvas></canvas><div class="ph"></div><button class="armsw" title="切换显示的手臂"></button>';
    this.cv = $('canvas', box); this.ph = $('.ph', box);
    $('.armsw', box).onclick = ev => { ev.stopPropagation(); this.arm = this.arm === 'L' ? 'R' : 'L'; this.draw(); };
    $('.foldsw', box).onclick = ev => { ev.stopPropagation(); this.fold(!this.folded); };
    this.fold(o.folded ?? LS.get('fold', false));
    allCurves.add(this);
    let drag = false;
    const at = ev => { const r = this.cv.getBoundingClientRect(); return Math.round((ev.clientX - r.left) / r.width * this.n); };
    this.cv.onpointerdown = ev => { drag = true; this.cv.setPointerCapture(ev.pointerId); this.onseek && this.onseek(at(ev)); };
    this.cv.onpointermove = ev => { if (drag && this.onseek) this.onseek(at(ev)); };
    this.cv.onpointerup = () => { drag = false; };
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(box);
    getCurves(o.id, o.ep).then(d => { this.data = d; this.draw(); }).catch(e => { box.title = e.message; });
  }
  fold(v) {
    this.folded = v;
    this.box.classList.toggle('folded', v);
    $('.foldsw', this.box).textContent = v ? '▸ 关节曲线' : '▾ 折叠';
    if (!v) this.draw();
  }
  draw() {
    if (this.folded) return;
    const W = this.box.clientWidth, LH = 26, H = 6 * LH + 2, dpr = devicePixelRatio || 1;
    if (!W) return;
    $('.armsw', this.box).textContent = this.arm === 'L' ? '左臂 ⇄' : '右臂 ⇄';
    this.cv.width = W * dpr; this.cv.height = H * dpr; this.cv.style.height = H + 'px';
    const c = this.cv.getContext('2d'), n = this.n, x = f => f / n * W, m = this.marks;
    c.scale(dpr, dpr);
    const css = getComputedStyle(document.body);
    c.fillStyle = css.getPropertyValue('--soft'); c.fillRect(0, 0, W, H);
    const shade = (a, b, col) => { c.fillStyle = col; c.fillRect(x(a), 0, x(b) - x(a), H); };
    if (m.keep) { shade(0, m.keep[0], '#8884'); shade(m.keep[1], n, '#8884'); }
    (m.cuts || []).forEach(([a, b]) => shade(a, b, '#8886'));
    (m.pauses || []).forEach(([a, b]) => shade(a, b, '#eda10040'));
    c.lineWidth = 1.5;
    (m.snaps || []).forEach(([t]) => { c.strokeStyle = '#e34948'; c.beginPath(); c.moveTo(x(t), 0); c.lineTo(x(t), H); c.stroke(); });
    if (m.fail != null) { c.strokeStyle = '#e34948'; c.setLineDash([4, 3]); c.beginPath(); c.moveTo(x(m.fail), 0); c.lineTo(x(m.fail), H); c.stroke(); c.setLineDash([]); }
    if (!this.data) return;
    const base = this.arm === 'L' ? 0 : 6, ink = css.getPropertyValue('--ink'), mute = css.getPropertyValue('--mute');
    for (let k = 0; k < 6; k++) {
      const j = base + k, top = k * LH + 1, s = this.data.state.map(r => r[j]), a = this.data.action.map(r => r[j]);
      let lo = Math.min(...s, ...a), hi = Math.max(...s, ...a);
      if (hi - lo < 2) { lo -= 1; hi += 1; }
      const y = v => top + LH - 3 - (v - lo) / (hi - lo) * (LH - 6);
      c.strokeStyle = '#8883'; c.lineWidth = 1; c.beginPath(); c.moveTo(0, top + LH - .5); c.lineTo(W, top + LH - .5); c.stroke();
      const line = (arr, dash, col, w) => {
        c.strokeStyle = col; c.lineWidth = w; c.setLineDash(dash); c.beginPath();
        arr.forEach((v, f) => f ? c.lineTo(x(f), y(v)) : c.moveTo(x(f), y(v))); c.stroke(); c.setLineDash([]);
      };
      line(a, [3, 2], mute, 1);
      line(s, [], LANE_COLORS[k], 1.4);
      c.fillStyle = ink; c.font = '10px system-ui'; c.fillText(`${NAMES[j]} ${Math.round(lo)}–${Math.round(hi)}`, 3, top + 10);
    }
  }
  setFrame(f) { this.ph.style.left = (100 * f / this.n) + '%'; }
  destroy() { this.ro.disconnect(); allCurves.delete(this); }
}

// ---------- labels (review / note / tags, saved in examples/xlerobot/dataset_labels.json) ----------
function labeler(box, id, ep, label, onchange) {
  label = {...(label || {})};
  const tags = new Set(label.tags || []);
  const all = [...new Set([...TAGS, ...tags])];
  box.innerHTML = `<button data-v="ok">通过</button><button data-v="bad">有问题</button>
    ${all.map(t => `<button class="tag${tags.has(t) ? ' on' : ''}" data-t="${esc(t)}">${esc(t)}</button>`).join('')}
    <input class="newtag" placeholder="+标签" size="6"><input data-note placeholder="备注" value="${esc(label.note)}">`;
  const mark = () => $$('[data-v]', box).forEach(b => b.classList.toggle('on', label.v === b.dataset.v));
  mark();
  const save = patch => api('/api/label', {id, ep, patch}).then(l => { label = l; status(`已保存标注 ${id} ep${ep}`); onchange && onchange(l); })
    .catch(e => status('保存失败：' + e.message, 1));
  box.onclick = ev => {
    const b = ev.target.closest('button');
    if (!b) return;
    if (b.dataset.v) { label.v = label.v === b.dataset.v ? '' : b.dataset.v; mark(); save({v: label.v}); return; }
    const t = b.dataset.t;
    if (!t) return;
    tags.has(t) ? tags.delete(t) : tags.add(t);
    b.classList.toggle('on', tags.has(t));
    save({tags: [...tags]});
  };
  $('.newtag', box).onkeydown = ev => {
    const t = ev.target.value.trim();
    if (ev.key !== 'Enter' || !t) return;
    tags.add(t); save({tags: [...tags]});
    labeler(box, id, ep, {...label, tags: [...tags]}, onchange);
  };
  let tm;
  $('[data-note]', box).oninput = ev => { clearTimeout(tm); tm = setTimeout(() => save({note: ev.target.value}), 500); };
}
const tagChips = l => (l && l.v ? `<span class="chip ${l.v}">${l.v === 'ok' ? '通过' : '有问题'}</span>` : '') + ((l && l.tags) || []).map(t => `<span class="chip">${esc(t)}</span>`).join('');

// ---------- episode modal (batch / quality / eval) ----------
// list = [{id, ep, n, video, arm, label, fail?, title}]; J/K moves inside the list
function openModal(list, k, start) {
  const dlg = $('#modal');
  let p, cv;
  const show = (k, start) => {
    p && p.destroy(); cv && cv.destroy();
    const e = list[k];
    dlg.innerHTML = `<header style="display:flex;gap:10px;align-items:center;margin-bottom:6px;flex-wrap:wrap"><b>${esc(e.id)} ep${e.ep}</b><span>${e.title || ''}</span>
      <span class="mute">${k + 1}/${list.length} · J/K 上下条 · 空格播放 · Esc 关闭</span><button style="margin-left:auto" class="x">关闭</button></header>
      <div class="pl"></div><div class="cv"></div><div class="labeler"></div>`;
    p = new Player($('.pl', dlg), {...e, start: start ?? e.start ?? 0});
    cv = new Curves($('.cv', dlg), {id: e.id, ep: e.ep, n: e.n, arm: e.arm, marks: e.fail != null ? {fail: e.fail} : {},
                                    onseek: f => { p.pause(); p.seek(f); }});
    p.onframe = f => cv.setFrame(f);
    cv.setFrame(p.f);
    labeler($('.labeler', dlg), e.id, e.ep, e.label, l => { e.label = l; e.onlabel && e.onlabel(l); });
    $('.x', dlg).onclick = () => dlg.close();
    p.play();
    dlg.k = k;
  };
  dlg.onkeydown = ev => {
    if (ev.target.tagName === 'INPUT') return;
    if ('jJ'.includes(ev.key)) { ev.preventDefault(); if (dlg.k < list.length - 1) show(dlg.k + 1); }
    else if ('kK'.includes(ev.key)) { ev.preventDefault(); if (dlg.k > 0) show(dlg.k - 1); }
    else if (ev.key === ' ') { ev.preventDefault(); p.toggle(); }
  };
  dlg.onclose = () => { p && p.destroy(); cv && cv.destroy(); dlg.innerHTML = ''; dlg.dispatchEvent(new Event('closed')); };
  show(k, start);
  if (!dlg.open) dlg.showModal();
}

// =====================================================================================================
// page 1: trim review — every episode of a spec, timeline editor, rule parameters, rebuild
// =====================================================================================================
async function pageReview(q, view, tools) {
  picker(null);
  const specs = await api('/api/specs');
  const spec = q.spec || LS.get('spec', 'v6_trim_all.json');
  const rule = {reset_tol: +(q.tol ?? 10), to_target: q.tt === '1', snap_rule: q.snap === '1'};
  tools.innerHTML = `
    <label>spec <select id="spec">${specs.filter(s => !s.segments).map(s => `<option ${s.name === spec ? 'selected' : ''}>${esc(s.name)}</option>`).join('')}</select></label>
    <button id="newspec" title="给一个数据集新建 examples/xlerobot/&lt;名字&gt;_trim.json">新建 spec…</button>
    <label title="尾部：从每个关节都保持在最终姿态这么多以内的第一帧结束（v6 = 10）">结尾容差 <input id="tol" type="number" step="1" min="1" style="width:56px" value="${rule.reset_tol}"></label>
    <label title="开头从抓取臂真正朝胶带出发处开始（reach_start）"><input type="checkbox" id="tt" ${rule.to_target ? 'checked' : ''}>--to_target</label>
    <label title="突变在前 30% → 起点移到突变后；后 30% → 终点移到突变前；中间 → 整条不要"><input type="checkbox" id="snap" ${rule.snap_rule ? 'checked' : ''}>突变规则</label>
    <span style="border-left:1px solid var(--line);height:18px"></span>
    <label>来源 <select id="fsrc"><option value="">全部</option></select></label>
    <label>标签 <select id="ftag"><option value="">全部</option></select></label>
    <label><input type="checkbox" id="fsnap">只看有突变</label><label><input type="checkbox" id="fpause">只看有停顿</label>
    <label><input type="checkbox" id="fedit">只看已修改</label><label><input type="checkbox" id="fdrop">隐藏整条不要</label>
    <label title="只循环播放保留的部分"><input type="checkbox" id="kloop">只播保留段</label>
    <span id="count" class="mute"></span><span id="foldall"></span>
    <button id="exp">导出</button><button id="rebuild" class="on">按修改重建数据集…</button>
    <span class="legend"><span><i style="background:var(--cut)"></i>剪掉</span><span><i style="background:var(--kept)"></i>保留</span><span><i style="background:var(--snap)"></i>突变</span>
      <span><i style="background:var(--pause)"></i>停顿&gt;1秒</span><span><i style="border:1px solid var(--ink)"></i>规则范围</span>
      <span>J/K 上下条 · 空格播放 · 在时间轴上按住拖动预览、松手接着播 · [ ] 设起点/终点 · 点剪切段或橙色把手后按 Delete 删除 · &lt; &gt; 调速度</span></span>`;
  $('#foldall').replaceWith(foldAllButton());
  const go = () => setQuery({spec: $('#spec').value, tol: $('#tol').value, tt: +$('#tt').checked, snap: +$('#snap').checked});
  $('#spec').onchange = () => { LS.set('spec', $('#spec').value); go(); render(); };
  ['#tol', '#tt', '#snap'].forEach(s => $(s).onchange = () => { go(); refresh(); });
  $('#newspec').onclick = () => {
    const id = prompt('数据集名（~/.cache/huggingface/lerobot/wwqcleverghost/ 下的目录名）', 'handover2');
    if (id) api('/api/spec/new', {id}).then(r => { LS.set('spec', r.name); location.hash = '#review?spec=' + r.name; location.reload(); }).catch(e => status(e.message, 1));
  };

  let D = [], cur = null, sel = null;
  const curRule = () => ({reset_tol: +$('#tol').value, to_target: $('#tt').checked, snap_rule: $('#snap').checked});
  const ruleQ = () => qs({spec: $('#spec').value, reset_tol: $('#tol').value, to_target: +$('#tt').checked, snap_rule: +$('#snap').checked});
  const eff = d => d.edit.range ? {a: d.edit.range[0], b: d.edit.range[1], cuts: d.edit.cut.map(x => [...x]), drop: d.edit.drop}
    : {a: d.pfirst, b: d.pend, cuts: d.edit.cut.map(x => [...x]), drop: d.edit.drop || d.pdrop};
  const marks = d => ({keep: [d.w.a, d.w.b], cuts: d.w.cuts, pauses: d.pauses, snaps: d.snaps});
  // a range in the spec counts as an edit only where it differs from what the current rule gives
  // (a rebuild writes every episode's range into the spec)
  const edited = d => d.edit.range && (d.edit.range[0] !== d.pfirst || d.edit.range[1] !== d.pend);
  const loopRange = d => $('#kloop').checked && !d.w.drop ? [d.w.a, d.w.b] : null;

  // players / curves only while a card is near the screen (few decoders at a time); autoplay while visible
  const io = new IntersectionObserver(es => es.forEach(en => {
    const d = D[+en.target.dataset.i];
    if (!d) return;
    if (en.isIntersecting) {
      if (!d.p) {
        d.p = new Player($('.pl', d.c), {id: d.ds, ep: d.ep, n: d.n, video: d.video, start: d.w.a, range: loopRange(d)});
        d.p.onframe = f => { $('.bar .ph', d.c).style.left = (100 * f / d.n) + '%'; d.cv && d.cv.setFrame(f); };
      }
      if (!d.cv) d.cv = new Curves($('.cvbox', d.c), {id: d.ds, ep: d.ep, n: d.n, arm: d.arm, marks: marks(d), onseek: f => { d.p.pause(); d.p.seek(f); }});
      d.p.mount();
      if (d.visible && !d.held) d.p.play();
    } else if (d.p) d.p.unmount();
  }), {rootMargin: '700px 0px'});
  const vis = new IntersectionObserver(es => es.forEach(en => {
    const d = D[+en.target.dataset.i];
    if (!d) return;
    d.visible = en.isIntersecting;
    if (!d.p) return;
    if (d.visible && !d.held) d.p.play(); else d.p.pause();
  }), {threshold: 0.35});
  G.cleanup.push(() => { io.disconnect(); vis.disconnect(); D.forEach(d => { d.p && d.p.destroy(); d.cv && d.cv.destroy(); }); });

  async function render() {
    io.disconnect(); vis.disconnect(); D.forEach(d => { d.p && d.p.destroy(); d.cv && d.cv.destroy(); });
    view.innerHTML = '<p class="mute">加载中…</p>';
    status('读取 ' + $('#spec').value);
    let R;
    try { R = await api('/api/review?' + ruleQ()); } catch (e) { view.innerHTML = `<p class="err">${esc(e.message)}</p>`; status(''); return; }
    D = R.episodes; D.forEach((d, i) => { d.i = i; d.w = eff(d); });
    showInfo([], R.sources.map(s => dsLabel(G.by[s.ds]) || s.ds).join('　'));
    $('#fsrc').innerHTML = '<option value="">全部</option>' + R.sources.map(s => `<option>${esc(s.ds)}</option>`).join('');
    view.innerHTML = '<div class="cards"></div>';
    const box = $('.cards', view);
    D.forEach(d => box.append(card(d)));
    counts(); filter();
    status(`${D.length} 条`);
  }
  async function refresh() {  // rule parameters changed: new proposals, same cards
    status('按新规则计算…');
    const R = await api('/api/review?' + ruleQ()).catch(e => status(e.message, 1));
    if (!R) return;
    R.episodes.forEach((r, i) => { const d = D[i]; Object.assign(d, r, {i}); d.w = eff(d); draw(d); });
    counts(); filter(); status('已按新规则更新（手动改过的条目不变）');
  }

  function card(d) {
    const sn = d.snaps.map(([t, j, v]) => `${sec(t)}秒 ${NAMES[j]} 跳 ${v}`).join('；') || '无';
    const pa = d.pauses.map(([a, b]) => `${sec(a)}–${sec(b)}秒`).join('；') || '无';
    const c = el(`<section class="card" data-i="${d.i}">
      <header><b>#${d.i}</b> ${esc(d.ds)} ep${d.ep}<span>原始 ${sec(d.n)} 秒</span><span class="sel"></span><span class="edited"></span><span class="chips"></span></header>
      <div class="bar"><div class="ph"></div></div><div class="cvbox"></div>
      <p class="meta">突变（红线）：${sn}。停顿 &gt; 1 秒（黄色）：${pa}。细框 = 规则算出的保留范围。曲线：实线 = 从臂 observation.state，虚线 = 主臂指令 action</p>
      <div class="row"><div class="pl"></div><div class="strip"></div></div>
      <footer><button class="drop">整条不要</button><button class="addcut">加剪切段</button><button class="reset" title="回到 spec 的原始版本 + 当前规则">恢复初始</button><span class="lab labeler"></span></footer></section>`);
    d.c = c;
    const stills = [[Math.max(d.first - 15, 0), '剪掉部分（起点前 0.5 秒）'], [d.first, '保留起点'], [d.end - 1, '保留终点'], [Math.min(d.end + 14, d.n - 1), '剪掉部分（终点后 0.5 秒）']];
    d.snaps.forEach(([t], k) => stills.push([t - 2, `突变${k + 1}之前`], [t + 1, `突变${k + 1}之后`]));
    $('.strip', c).innerHTML = stills.map(([f, t]) => `<figure data-f="${f}"><img loading="lazy" src="${frameUrl(d.ds, Object.keys(d.video)[0], d.ep, f)}"><figcaption>${t} · ${sec(f)}s</figcaption></figure>`).join('');
    $('.strip', c).onclick = ev => { const f = ev.target.closest('figure'); if (f && d.p) { d.p.pause(); d.p.seek(+f.dataset.f); } };
    labeler($('.lab', c), d.ds, d.ep, d.label, l => { d.label = l; head(d); counts(); });
    head(d); draw(d);
    c.addEventListener('pointerdown', () => setCur(d), true);
    bindBar(d);
    $('.drop', c).onclick = () => { d.w.drop = !d.w.drop; draw(d); save(d); };
    $('.addcut', c).onclick = () => {
      const m = d.p ? d.p.f : Math.round((d.w.a + d.w.b) / 2);
      d.w.cuts.push([Math.max(m - 30, d.w.a), Math.min(m + 30, d.w.b)]); sel = {d, k: d.w.cuts.length - 1}; draw(d); save(d);
    };
    $('.reset', c).onclick = () => api('/api/review/edit', {spec: $('#spec').value, repo: d.repo, ep: d.ep, edit: null})
      .then(e => { d.edit = e; d.w = eff(d); draw(d); counts(); status(`已恢复 ${d.ds} ep${d.ep}`); }).catch(e => status(e.message, 1));
    io.observe(c); vis.observe(c);
    return c;
  }
  function head(d) { $('.chips', d.c).innerHTML = tagChips(d.label); d.c.classList.toggle('ok', d.label.v === 'ok'); d.c.classList.toggle('bad', d.label.v === 'bad'); }
  function draw(d) {
    const e = d.w, bar = $('.bar', d.c), ph = $('.ph', bar), pct = f => (100 * f / d.n) + '%';
    bar.querySelectorAll(':scope > :not(.ph)').forEach(x => x.remove());
    const add = (cls, l, w, ds = {}) => { const x = document.createElement('div'); x.className = cls; x.style.left = pct(l); if (w != null) x.style.width = pct(w); Object.assign(x.dataset, ds); bar.insertBefore(x, ph); return x; };
    add('seg', e.a, e.b - e.a);
    e.cuts.forEach(([s, t], k) => add('cutseg', s, t - s, {k}));
    add('orig', d.first, d.end - d.first);
    d.pauses.forEach(([s, t]) => add('pause', s, t - s));
    d.snaps.forEach(([t]) => add('snap', t));
    add('hd', e.a, null, {h: 'a'}); add('hd', e.b, null, {h: 'b'});
    e.cuts.forEach(([s, t], k) => { add('hd c', s, null, {h: 's', k}); add('hd c', t, null, {h: 't', k}); });
    if (sel && sel.d === d) bar.querySelectorAll(`.cutseg[data-k="${sel.k}"]`).forEach(x => x.classList.add('selected'));
    const kept = e.b - e.a - e.cuts.reduce((s, [a, b]) => s + b - a, 0);
    $('.sel', d.c).textContent = e.drop ? '整条不要' : `保留 ${sec(e.a)}–${sec(e.b)} 秒（共 ${sec(kept)} 秒）` +
      (e.cuts.length ? '，中间剪掉 ' + e.cuts.map(([s, t]) => `${sec(s)}–${sec(t)}`).join('、') : '');
    $('.edited', d.c).textContent = edited(d) ? '已手动修改（和规则结果不同）' : d.edit.range ? '范围已固定在 spec 里' : '';
    d.c.classList.toggle('dropped', e.drop);
    if (d.p) d.p.range = loopRange(d);
    if (d.cv) { d.cv.marks = marks(d); d.cv.draw(); }
  }
  const saveT = {};
  function save(d) {
    clearTimeout(saveT[d.i]);
    saveT[d.i] = setTimeout(() => api('/api/review/edit', {spec: $('#spec').value, repo: d.repo, ep: d.ep, edit: d.w})
      .then(e => { d.edit = e; draw(d); counts(); status(`已保存到 ${$('#spec').value}：${d.ds} ep${d.ep}`); })
      .catch(e => status('保存失败：' + e.message, 1)), 150);
  }
  function bindBar(d) {
    const bar = $('.bar', d.c);
    const frameAt = x => { const r = bar.getBoundingClientRect(); return Math.max(0, Math.min(d.n, Math.round((x - r.left) / r.width * d.n))); };
    const seek = f => { if (!d.p) return; d.p.pause(); d.p.seek(Math.min(f, d.n - 1)); };
    let drag = null;
    const select = k => { sel = {d, k}; $$('.cutseg.selected').forEach(x => x.classList.remove('selected')); bar.querySelectorAll(`.cutseg[data-k="${k}"]`).forEach(x => x.classList.add('selected')); };
    bar.onpointerdown = ev => {
      const h = ev.target.dataset.h;
      if (h) {
        drag = {h, k: +ev.target.dataset.k}; bar.setPointerCapture(ev.pointerId); ev.preventDefault();
        if (h === 's' || h === 't') select(drag.k);
        seek(h === 'a' ? d.w.a : h === 'b' ? d.w.b : d.w.cuts[drag.k][h === 's' ? 0 : 1]);
        return;
      }
      if (ev.target.classList.contains('cutseg')) return select(+ev.target.dataset.k);
      // anywhere else: hold = paused preview that follows the pointer, release = play on from there
      drag = {h: 'scrub'}; bar.setPointerCapture(ev.pointerId); ev.preventDefault();
      seek(frameAt(ev.clientX));
    };
    bar.onpointermove = ev => {
      if (!drag) return;
      if (drag.h === 'scrub') return seek(frameAt(ev.clientX));
      const e = d.w, f = frameAt(ev.clientX);
      if (drag.h === 'a') e.a = Math.min(f, e.b - 15); else if (drag.h === 'b') e.b = Math.max(f, e.a + 15);
      else { const cu = e.cuts[drag.k]; if (drag.h === 's') cu[0] = Math.min(f, cu[1] - 1); else cu[1] = Math.max(f, cu[0] + 1); }
      seek(drag.h === 'a' ? e.a : drag.h === 'b' ? e.b : f); draw(d);
    };
    bar.onpointerup = () => {
      if (!drag) return;
      const h = drag.h; drag = null;
      if (h === 'scrub') { if (d.p) { d.held = false; d.p.play(); } } else save(d);
    };
  }

  function counts() {
    const n = k => D.filter(k).length;
    $('#count').textContent = `${n(d => d.label.v === 'ok')} 通过 · ${n(d => d.label.v === 'bad')} 有问题 · ${n(d => d.w.drop)} 整条不要 · ${n(edited)} 已修改 · 共 ${D.length} 条`;
    const tc = {};
    D.forEach(d => (d.label.tags || []).forEach(t => tc[t] = (tc[t] || 0) + 1));
    const v = $('#ftag').value;
    $('#ftag').innerHTML = '<option value="">全部</option>' + Object.entries(tc).map(([t, k]) => `<option value="${esc(t)}" ${t === v ? 'selected' : ''}>${esc(t)}（${k}）</option>`).join('');
  }
  const shown = d => (!$('#fsrc').value || d.ds === $('#fsrc').value) && (!$('#ftag').value || (d.label.tags || []).includes($('#ftag').value)) &&
    (!$('#fsnap').checked || d.snaps.length) && (!$('#fpause').checked || d.pauses.length) && (!$('#fedit').checked || edited(d)) && (!$('#fdrop').checked || !d.w.drop);
  function filter() { D.forEach(d => { d.c.style.display = shown(d) ? '' : 'none'; }); }
  ['#fsrc', '#ftag', '#fsnap', '#fpause', '#fedit', '#fdrop'].forEach(s => $(s).onchange = filter);
  $('#kloop').onchange = () => D.forEach(d => d.p && (d.p.range = loopRange(d)));

  function setCur(d) { if (cur && cur !== d) cur.c.classList.remove('cur'); cur = d; d.c.classList.add('cur'); }
  const onKey = ev => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(ev.target.tagName) || $('#modal').open) return;
    const shownD = D.filter(d => d.c.style.display !== 'none');
    if (!cur || cur.c.style.display === 'none') { const c0 = shownD.find(d => d.c.getBoundingClientRect().bottom > 120); if (c0) setCur(c0); }
    if (!cur) return;
    if ('jJkK'.includes(ev.key) && ev.key.length === 1) {
      ev.preventDefault();
      const k = shownD.indexOf(cur), nx = shownD[Math.max(0, Math.min(shownD.length - 1, k + ('jJ'.includes(ev.key) ? 1 : -1)))];
      if (nx) { setCur(nx); nx.c.scrollIntoView({block: 'start'}); }
    } else if (ev.key === ' ') {
      ev.preventDefault();
      if (cur.p) { cur.held = cur.p.playing; cur.p.toggle(); }
    } else if ((ev.key === '[' || ev.key === ']') && cur.p) {
      const f = cur.p.f, e = cur.w;
      if (ev.key === '[') e.a = Math.min(f, e.b - 15); else e.b = Math.max(f + 1, e.a + 15);
      draw(cur); save(cur); status(`${cur.ds} ep${cur.ep} ${ev.key === '[' ? '起点' : '终点'} = ${sec(ev.key === '[' ? e.a : e.b)} 秒`);
    } else if ((ev.key === 'Delete' || ev.key === 'Backspace') && sel) {
      ev.preventDefault();
      const d = sel.d; d.w.cuts.splice(sel.k, 1); sel = null; draw(d); save(d);
    }
  };
  document.addEventListener('keydown', onKey);
  G.cleanup.push(() => document.removeEventListener('keydown', onKey));

  $('#exp').onclick = () => {
    const out = D.map(d => ({i: d.i, src: d.ds, ep: d.ep, ...d.w, edited: !!edited(d), review: d.label.v || '', note: d.label.note || '', tags: d.label.tags || []}));
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'}));
    a.download = $('#spec').value.replace('.json', '') + '_review.json'; a.click();
  };
  $('#rebuild').onclick = () => rebuildDialog($('#spec').value, curRule());
  await render();
  buildPanel();
}

async function rebuildDialog(spec, rule) {
  const R = await api('/api/review?' + qs({spec, reset_tol: rule.reset_tol, to_target: +rule.to_target, snap_rule: +rule.snap_rule}));
  const dlg = $('#modal');
  dlg.onkeydown = null; dlg.onclose = () => { dlg.innerHTML = ''; };
  dlg.innerHTML = `<h3 style="margin-top:0">按修改重建数据集</h3>
    <p>先把每条的精确保留范围写进 <code>examples/xlerobot/${esc(spec)}</code> 的 <code>range</code>（手动改过的不变，其余按当前规则：结尾容差 ${rule.reset_tol}、
    ${rule.to_target ? '用' : '不用'} --to_target、${rule.snap_rule ? '用' : '不用'}突变规则），再在后台（setsid）运行 <code>trim_stagnant_frames.py --spec</code>。
    目标名字相同的来源合并成一个数据集。</p>
    ${R.sources.map(s => `<p><label>${esc(s.repo)} → wwqcleverghost/<input data-repo="${esc(s.repo)}" value="${esc(s.dst || s.ds + '-trimmed-new')}" size="34"></label> <span class="ex"></span></p>`).join('')}
    <p><label><input type="checkbox" id="ow">目标已存在时覆盖（旧的改名为 &lt;名字&gt;.old_&lt;时间&gt;，不删除）</label>
    <label><input type="checkbox" id="arm">只保留 12 个手臂关节（--arm_only）</label></p>
    <p><button class="on" id="go">开始重建</button> <button id="cancel">取消</button> <span id="rbmsg"></span></p>`;
  const mark = () => $$('[data-repo]', dlg).forEach(i => { const d = G.by[i.value.trim()]; i.closest('p').querySelector('.ex').innerHTML = d ? `<span class="err">已存在${d.building ? '，正在构建' : ''}</span>` : ''; });
  $$('[data-repo]', dlg).forEach(i => i.oninput = mark); mark();
  $('#cancel', dlg).onclick = () => dlg.close();
  $('#go', dlg).onclick = () => {
    const dsts = Object.fromEntries($$('[data-repo]', dlg).map(i => [i.dataset.repo, i.value.trim()]));
    $('#rbmsg', dlg).textContent = '启动中…';
    api('/api/rebuild', {spec, rule, dsts, overwrite: $('#ow', dlg).checked, arm_only: $('#arm', dlg).checked})
      .then(jobs => { dlg.close(); status(`已开始重建 ${jobs.map(j => j.dst).join('、')}`); buildPanel(true); })
      .catch(e => { $('#rbmsg', dlg).innerHTML = `<span class="err">${esc(e.message)}</span>`; });
  };
  dlg.showModal();
}

function buildPanel(open) {
  let box = $('#builds');
  if (!box) {
    box = el('<div id="builds"><header><b>构建日志</b><span class="sum mute"></span><button style="margin-left:auto" class="tg">展开</button></header><div class="body" hidden><div class="jobs"></div><pre></pre></div></div>');
    document.body.append(box);
    $('header', box).onclick = () => { const b = $('.body', box); b.hidden = !b.hidden; $('.tg', box).textContent = b.hidden ? '展开' : '收起'; poll(); };
  }
  if (open) { $('.body', box).hidden = false; $('.tg', box).textContent = '收起'; }
  let pick = 0;
  async function poll() {
    clearTimeout(G.buildT);
    if (!document.body.contains(box)) return;
    const jobs = await api('/api/builds').catch(() => []);
    const st = j => j.running ? '运行中' : j.ok ? '完成' : '失败/中断';
    $('.sum', box).textContent = jobs.length ? `${jobs[0].dst}：${st(jobs[0])}` : '还没有用本页重建过';
    $('.jobs', box).innerHTML = jobs.map((j, k) => `<button data-k="${k}" class="${pick === k ? 'on' : ''}">${esc(j.dst)} · ${j.started} · ${st(j)}</button>`).join('');
    $$('.jobs button', box).forEach(b => b.onclick = ev => { ev.stopPropagation(); pick = +b.dataset.k; poll(); });
    const j = jobs[pick];
    $('pre', box).textContent = j ? `$ ${j.cmd}\n（git ${j.git.commit.slice(0, 8)}${j.git.dirty ? ' + 未提交修改' : ''}，日志 ${j.log}）\n\n` + j.tail.join('\n') : '';
    const pre = $('pre', box); pre.scrollTop = pre.scrollHeight;
    if (jobs.some(j => j.running)) G.buildT = setTimeout(poll, 2000);
  }
  poll();
}

// =====================================================================================================
// page 3: batch playback — many episodes from one or more datasets, as many columns as fit, one shared clock
// =====================================================================================================
async function pageBatch(q, view, tools) {
  let ids = (q.ds ? q.ds.split(',') : LS.get('batch_ds', ['v6-handover-curated'])).filter(i => G.by[i] && !G.by[i].building);
  // #batch?pick=1: an episode list handed over by the curate page (kept or dropped episodes of its current result)
  let pick = q.pick === '1' ? LS.get('batch_pick', null) : null;
  if (pick) ids = pick.ds.filter(i => G.by[i] && !G.by[i].building);
  picker('multi', ids, d => !d.building, v => { ids = v; pick = null; $('.pickbar') && $('.pickbar').remove(); LS.set('batch_ds', ids); setQuery({ds: ids.join(',')}); load(); });
  const cw = LS.get('batch_cw', 260);
  tools.innerHTML = `<button id="play" class="on">▶ 全部播放</button><button id="rst">⏮ 从头</button>
    <label>画面 <select id="cams"><option value="head">只看头部相机</option><option value="all">三路相机</option></select></label>
    <label>格子宽 <input id="cw" type="range" min="160" max="640" step="20" value="${cw}"></label>
    <label>每页 <select id="ps"><option>12</option><option selected>24</option><option>48</option></select></label>
    <span style="border-left:1px solid var(--line);height:18px"></span>
    <label>来源 <select id="fsrc"></select></label><label>标签 <select id="ftag"></select></label>
    <label>结果 <select id="fres"><option value="">全部</option><option value="1">成功</option><option value="0">失败</option></select></label>
    <label title="每条的质量分（report.py quality，基准 ${REF_NAME} 中位数）">质量分 <input id="qmin" type="number" step="0.05" min="0" max="1" style="width:58px" placeholder="≥"> – <input id="qmax" type="number" step="0.05" min="0" max="1" style="width:58px" placeholder="≤"></label>
    <span id="count" class="mute"></span><span class="mute">空格 全部播放/暂停 · 点格子标题看大图、曲线、打标签</span>`;
  if (pick) {
    tools.prepend(el(`<span class="pickbar">来自筛选：预设 ${esc(pick.preset)}，${esc(pick.what)}共 ${pick.eps.length} 条 <button id="back">返回筛选</button></span>`));
    $('#back').onclick = () => { location.hash = '#curate'; };
  }
  view.style.setProperty('--cw', cw + 'px');
  $('#cw').oninput = ev => { view.style.setProperty('--cw', ev.target.value + 'px'); LS.set('batch_cw', +ev.target.value); };
  let E = [], cells = [], page = 0;
  const clock = {playing: false, t: 0, t0: 0, rate: G.rate}, scores = {};
  async function load() {
    view.innerHTML = ids.length ? '<p class="mute">加载中…</p>' : '<p class="mute">在上面选择一个或多个数据集。</p>';
    const res = await Promise.all(ids.map(id => api('/api/episodes?id=' + encodeURIComponent(id)).catch(e => ({id, error: e.message, episodes: []}))));
    E = res.flatMap(r => r.episodes.map(e => ({...e, id: r.id})));
    if (pick) {
      const info = Object.fromEntries(pick.eps.map(x => [x.ds + '|' + x.ep, x]));
      E = E.filter(e => info[e.id + '|' + e.ep]).map(e => ({...e, pick: info[e.id + '|' + e.ep]}));
    }
    $('#fsrc').innerHTML = '<option value="">全部</option>' + ids.map(i => `<option>${esc(i)}</option>`).join('');
    tagOptions();
    page = 0; clock.t = 0; draw();
    const errs = res.filter(r => r.error);
    status(errs.length ? errs.map(r => `${r.id}: ${r.error}`).join('；') : `${E.length} 条`, errs.length);
    ids.filter(i => !scores[i]).forEach(i => api('/api/quality?id=' + encodeURIComponent(i)).then(r => {
      scores[i] = Object.fromEntries(r.rows.map(x => [x.ep, x.q]));
      cells.forEach(c => c.e.id === i && caption(c));
      if ($('#qmin').value || $('#qmax').value) { page = 0; draw(); }
    }).catch(() => {}));
  }
  function tagOptions() {
    const tc = {}; E.forEach(e => (e.label.tags || []).forEach(t => tc[t] = (tc[t] || 0) + 1));
    const v = $('#ftag').value;
    $('#ftag').innerHTML = '<option value="">全部</option>' + Object.entries(tc).map(([t, k]) => `<option value="${esc(t)}" ${t === v ? 'selected' : ''}>${esc(t)}（${k}）</option>`).join('');
  }
  const qOf = e => scores[e.id] && scores[e.id][e.ep];
  const shown = () => E.filter(e => (!$('#fsrc').value || e.id === $('#fsrc').value) && (!$('#ftag').value || (e.label.tags || []).includes($('#ftag').value)) &&
    ($('#fres').value === '' || e.success === +$('#fres').value) &&
    (!$('#qmin').value || (qOf(e) ?? -1) >= +$('#qmin').value) && (!$('#qmax').value || (qOf(e) ?? 2) <= +$('#qmax').value));
  function caption(c) {
    const e = c.e, q = qOf(e);
    $('.cap', c.box).innerHTML = `<b>${esc(e.id)}</b> ep${e.ep}` + (e.success != null ? ` <span class="chip ${e.success ? 'ok' : 'bad'}">${e.success ? '成功' : '失败'}</span>` : '') +
      `<span class="mute">${sec(e.n)}s${q != null ? ' · 质量 ' + fx(q) : ''}</span>` + tagChips(e.label) + pickChip(e);
  }
  const pickChip = e => !e.pick ? '' : ` <span class="chip ${e.pick.kept ? 'ok' : 'bad'}" title="${esc(e.pick.why.join('；'))}">${e.pick.kept ? '保留' : '去掉'}</span>` +
    (e.pick.why.length ? `<span class="mute">${esc(e.pick.why.join('；'))}</span>` : '');
  function draw() {
    cells.forEach(c => c.p.destroy());
    const list = shown(), ps = +$('#ps').value, pages = Math.max(1, Math.ceil(list.length / ps));
    page = Math.min(page, pages - 1);
    const sl = list.slice(page * ps, page * ps + ps);
    $('#count').textContent = `显示 ${list.length} / ${E.length} 条`;
    view.innerHTML = `<div class="grid"></div><div class="pager"><button id="prev">上一页</button><span>${page + 1} / ${pages}</span><button id="next">下一页</button></div>`;
    $('#prev').onclick = () => { if (page > 0) { page--; draw(); } };
    $('#next').onclick = () => { if (page < pages - 1) { page++; draw(); } };
    const grid = $('.grid', view), head = $('#cams').value === 'head';
    cells = sl.map((e, k) => {
      const box = el('<div class="cell"><div class="cap"></div><div class="pl"></div></div>');
      grid.append(box);
      const c = {e, box, p: new Player($('.pl', box), {id: e.id, ep: e.ep, n: e.n, video: e.video, loop: false,
        start: Math.min(clock.t * FPS, e.n - 1), cams: head ? [Object.keys(e.video)[0]] : undefined})};
      c.p.box.onclick = togglePlay;
      $('.cap', box).onclick = () => {
        pause();
        openModal(list.map(x => ({...x, title: (x.success != null ? (x.success ? '<span class="chip ok">成功</span>' : '<span class="chip bad">失败</span>') : '') + pickChip(x),
                                  onlabel: l => { x.label = l; cells.forEach(caption); tagOptions(); }})), page * ps + k, c.p.f);
      };
      caption(c);
      c.p.mount();
      return c;
    });
    seekAll(clock.playing ? now() : clock.t);
    if (clock.playing) cells.forEach(c => c.p.play());
  }
  const now = () => clock.playing ? clock.t + (performance.now() - clock.t0) / 1000 * clock.rate : clock.t;
  function seekAll(t) { cells.forEach(c => { c.p.seek(Math.min(t * FPS, c.e.n - 1)); c.box.classList.toggle('endd', t * FPS >= c.e.n - 1); }); }
  function play() { clock.playing = true; clock.t0 = performance.now(); $('#play').textContent = '⏸ 全部暂停'; cells.forEach(c => { if (clock.t * FPS < c.e.n - 1) c.p.play(); }); }
  function pause() { clock.t = now(); clock.playing = false; $('#play').textContent = '▶ 全部播放'; cells.forEach(c => c.p.pause()); }
  function togglePlay() { clock.playing ? pause() : play(); }
  // one shared clock: every cell shows frame t * fps of its episode; finished episodes freeze on their last frame;
  // when the longest one is done all start again
  const syncT = setInterval(() => {
    if (!clock.playing || !cells.length) return;
    const f = now() * FPS;
    if (f > Math.max(...cells.map(c => c.e.n)) + 15) { clock.t = 0; clock.t0 = performance.now(); seekAll(0); cells.forEach(c => c.p.play()); return; }
    cells.forEach(c => {
      const end = f >= c.e.n - 1;
      c.box.classList.toggle('endd', end);
      if (end) { if (c.p.playing) { c.p.pause(); c.p.seek(c.e.n - 1); } return; }
      if (!c.p.playing) c.p.play();
      if (Math.abs(c.p.f - f) > 8 * clock.rate) c.p.seek(f);
    });
  }, 250);
  G.cleanup.push(() => { clearInterval(syncT); cells.forEach(c => c.p.destroy()); });
  $('#play').onclick = togglePlay;
  $('#rst').onclick = () => { clock.t = 0; clock.t0 = performance.now(); seekAll(0); if (clock.playing) cells.forEach(c => c.p.play()); };
  const onRate = () => { clock.t = now(); clock.t0 = performance.now(); clock.rate = G.rate; };
  document.addEventListener('globalrate', onRate);
  G.cleanup.push(() => document.removeEventListener('globalrate', onRate));
  ['#cams', '#ps', '#fsrc', '#ftag', '#fres', '#qmin', '#qmax'].forEach(s => $(s).onchange = () => { page = 0; draw(); });
  const onKey = ev => { if (ev.key === ' ' && !['INPUT', 'SELECT'].includes(ev.target.tagName) && !$('#modal').open) { ev.preventDefault(); togglePlay(); } };
  document.addEventListener('keydown', onKey);
  G.cleanup.push(() => document.removeEventListener('keydown', onKey));
  await load();
}


// =====================================================================================================
// shared bits of the two score pages
// =====================================================================================================
// categorical series colours in a fixed order; a dataset keeps its slot while it stays selected (colour follows the entity)
const SLOTS = 8;
function colorSlots(ids) {
  const old = LS.get('slots', {}), used = new Set(), out = {};
  ids.forEach(i => { if (old[i] != null && !used.has(old[i])) { out[i] = old[i]; used.add(old[i]); } });
  ids.forEach(i => { if (out[i] == null) { for (let k = 0; k < SLOTS; k++) if (!used.has(k)) { out[i] = k; used.add(k); break; } } });
  LS.set('slots', {...old, ...out});
  return out;
}
const sv = k => `var(--s${k + 1})`;
const ciBar = (v, w = 140) => !v || v[0] == null ? '–' :
  `<svg width="${w}" height="14" class="ci"><line x1="0" x2="${w}" y1="7" y2="7" class="ax"/><line x1="${w / 2}" x2="${w / 2}" y1="3" y2="11" class="ax"/>` +
  `<line x1="${v[1] * w}" x2="${v[2] * w}" y1="7" y2="7" class="w"/><circle cx="${v[0] * w}" cy="7" r="4" class="d"/></svg> ` +
  `<b>${fx(v[0])}</b> <span class="mute">[${fx(v[1])}–${fx(v[2])}]</span>`;
const tip = (ev, text) => { const t = $('#tip'); t.hidden = !text; if (text) { t.textContent = text; t.style.left = (ev.clientX + 12) + 'px'; t.style.top = (ev.clientY + 12) + 'px'; } };
const verName = v => v ? `v${v}` : '早期（无版本前缀）';
// sortable table: cols = [[key, title, fmt(row) -> html, sortValue(row)]], onrow(row, rowsInShownOrder, index)
function table(box, rows, cols, onrow, sortKey) {
  let key = sortKey || null, dir = 1;
  const draw = () => {
    const col = cols.find(c => c[0] === key), r = col ? [...rows].sort((a, b) => {
      const x = (col[3] || (z => z[key]))(a), y = (col[3] || (z => z[key]))(b);
      return x == null ? 1 : y == null ? -1 : (x > y ? 1 : x < y ? -1 : 0) * dir;
    }) : rows;
    box.innerHTML = `<table><thead><tr>${cols.map(c => `<th data-k="${c[0]}">${c[1]}${c[0] === key ? (dir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('')}</tr></thead>
      <tbody>${r.map((x, i) => `<tr class="click${x._sel ? ' selrow' : ''}" data-i="${i}">${cols.map(c => `<td>${c[2](x)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
    $$('th', box).forEach(th => th.onclick = () => { dir = key === th.dataset.k ? -dir : 1; key = th.dataset.k; draw(); });
    $$('tbody tr', box).forEach(tr => tr.onclick = ev => onrow(r[+tr.dataset.i], r, +tr.dataset.i, ev));
  };
  draw();
}
async function episodeInfo(id) {  // video / length / arm / labels per episode, for the modal player
  G.epi = G.epi || {};
  if (!G.epi[id]) G.epi[id] = api('/api/episodes?id=' + encodeURIComponent(id)).then(r => Object.fromEntries(r.episodes.map(e => [e.ep, {...e, id}])));
  return G.epi[id];
}
// history tables poll while the server is still scoring datasets in the background
async function historyRows(kind, onrows) {
  const h = await api('/api/history?kind=' + kind);
  onrows(h);
  if (h.pending.length && document.body.contains($('#hist'))) G.histT = setTimeout(() => historyRows(kind, onrows), 3000);
}

// =====================================================================================================
// page 2: training-set quality — per-episode metrics, rank distribution, several datasets side by side, history
// =====================================================================================================
const QM = [  // key, label, format, note
  ['duration_s', '时长', x => fx(x, 1) + ' s', '整条时长'],
  ['median_jerk', '中位 jerk', x => fx(x, 0), '从臂 observation.state 三阶差分 × fps³ 的中位数'],
  ['p95_jerk', 'p95 jerk', x => fx(x, 0), '从臂 jerk 的 95 百分位'],
  ['max_velocity', '峰值速度', x => fx(x, 0), '从臂 12 关节速度范数最大值（单位/秒）'],
  ['static_fraction', '停顿占比', x => fx(100 * x, 1) + '%', '主臂指令速度 < 3 单位/秒的帧占比'],
  ['max_pause_s', '最长停顿', x => fx(x, 1) + ' s', '主臂指令最长连续停顿'],
  ['rev_rate', '小幅来回', x => fx(x, 2) + '/s', '运动中每秒小幅关节来回次数（smooth.py）'],
  ['q', '质量分', x => fx(x, 2), 'report.py quality：0.4 平滑 + 0.3 速度 + 0.3 不停顿，对比人类基准中位数'],
];
async function pageQuality(q, view, tools) {
  const def = ['v4-handover2-trimmed', 'v5-handover-perfect', 'v6-handover-curated', 'raw-handover-20260929-trimmed', 'raw-handover-20261001-trimmed'];
  let ids = (q.ds ? q.ds.split(',') : LS.get('quality_ds', def)).filter(i => G.by[i] && !G.by[i].eval && !G.by[i].building).slice(0, SLOTS);
  let metric = LS.get('quality_metric', 'p95_jerk');
  const setIds = v => { ids = v.slice(0, SLOTS); LS.set('quality_ds', ids); setQuery({ds: ids.join(',')}); picker('multi', ids, d => !d.eval && !d.building, setIds); hist(); detail(); };
  picker('multi', ids, d => !d.eval && !d.building, setIds);
  tools.innerHTML = `<span class="mute">jerk、峰值速度看从臂 observation.state；停顿看主臂指令 action；质量分对比人类基准 ${REF_NAME} 中位数；
    IQR 标记 = 超过该数据集 Q3 + 1.5×IQR（时长也标过短）。点历史表的行加入/移出对比（最多 ${SLOTS} 个），点图上的点或逐条表的行看视频。</span>`;
  view.innerHTML = `<div class="panel" id="hist"><h3>历史：所有训练 / 示范数据集（每列为该数据集各条的中位数）</h3><div class="tb"></div></div>
    <div class="panel" id="rank"><h3>排名分布</h3><div class="chips"></div><div class="legend lg"></div><div class="chart"></div></div>
    <div class="panel" id="eps"><h3>逐条 <label class="mute" style="font-weight:400"><input type="checkbox" id="fflag"> 只看有 IQR 标记的</label></h3><div class="tb"></div></div>`;
  let H = null, data = {};
  function hist() {
    if (!H) return;
    const rows = H.rows.map(r => ({...r, _sel: ids.includes(r.id)})).sort((a, b) => b.version - a.version || a.id.localeCompare(b.id));
    $('#hist h3').innerHTML = `历史：所有训练 / 示范数据集（每列为该数据集各条的中位数）${H.pending.length ? ` <span class="mute">· 后台计算中，还剩 ${H.pending.length} 个…</span>` : ''}`;
    table($('#hist .tb'), rows, [
      ['id', '数据集', r => `${r._sel ? `<i class="sw" style="background:${sv(colorSlots(ids)[r.id])}"></i>` : ''}<span class="nm" title="${esc(r.id)}">${esc(r.id)}</span>${r.error ? ` <span class="err" title="${esc(r.error)}">读取失败</span>` : ''}`],
      ['version', '版本', r => verName(r.version)],
      ['n', '条数', r => r.n ?? '–'],
      ['quality', '质量分 [95% CI]', r => ciBar(r.quality, 100), r => r.quality && r.quality[0]],
      ...QM.slice(0, 7).map(([k, t, f]) => [k, t, r => r[k] == null ? '–' : f(r[k])]),
      ['flagged', 'IQR 标记', r => r.flagged == null ? '–' : `${r.flagged}/${r.n}`],
    ], r => { if (r.error) return; setIds(ids.includes(r.id) ? ids.filter(x => x !== r.id) : [...ids, r.id]); });
  }
  historyRows('quality', h => { H = h; hist(); });
  async function detail() {
    $('#rank .chart').innerHTML = $('#eps .tb').innerHTML = ids.length ? '<p class="mute">加载中…</p>' : '<p class="mute">在历史表里点选数据集。</p>';
    await Promise.all(ids.filter(i => !data[i]).map(async i => {
      try { const [r, info] = await Promise.all([api('/api/quality?id=' + encodeURIComponent(i)), episodeInfo(i)]); data[i] = r.rows.map(x => ({...info[x.ep], ...x, id: i})); }
      catch (e) { status(`${i}: ${e.message}`, 1); data[i] = []; }
    }));
    chart(); eps();
  }
  function chart() {
    $('#rank .chips').innerHTML = QM.map(([k, t, , note]) => `<button class="${k === metric ? 'on' : ''}" data-k="${k}" title="${esc(note)}">${t}</button>`).join(' ');
    $$('#rank .chips button').forEach(b => b.onclick = () => { metric = b.dataset.k; LS.set('quality_metric', metric); chart(); });
    const slots = colorSlots(ids), m = QM.find(x => x[0] === metric), box = $('#rank .chart');
    $('#rank .lg').innerHTML = ids.map(i => `<span><i style="background:${sv(slots[i])}"></i>${esc(i)}</span>`).join('');
    const series = ids.map(i => ({id: i, pts: (data[i] || []).filter(r => r[metric] != null).sort((a, b) => a[metric] - b[metric])})).filter(s => s.pts.length);
    if (!series.length) { box.innerHTML = '<p class="mute">没有数据</p>'; return; }
    const W = Math.max(box.clientWidth, 480), Hh = 300, L = 64, R = 170, T = 12, B = 36;
    const all = series.flatMap(s => s.pts.map(p => p[metric]));
    let lo = metric === 'q' ? Math.max(0, Math.min(...all) - 0.05) : 0, hi = metric === 'q' ? 1 : Math.max(...all) * 1.05;
    if (hi <= lo) hi = lo + 1;
    const x = r => L + r * (W - L - R), y = v => T + (Hh - T - B) * (1 - (v - lo) / (hi - lo));
    let g = '';
    for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4; g += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" class="grid"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end" class="axt">${m[2](v)}</text>`; }
    for (let k = 0; k <= 4; k++) g += `<text x="${x(k / 4)}" y="${Hh - B + 16}" text-anchor="middle" class="axt">${k * 25}%</text>`;
    g += `<text x="${(L + W - R) / 2}" y="${Hh - 4}" text-anchor="middle" class="axt">数据集内排名（从小到大）</text>`;
    const ends = [];
    series.forEach(s => {
      const n = s.pts.length, px = i => x(n > 1 ? i / (n - 1) : 0.5), col = sv(slots[s.id]);
      g += `<polyline fill="none" stroke="${col}" stroke-width="2" points="${s.pts.map((p, i) => `${px(i)},${y(p[metric])}`).join(' ')}"/>`;
      s.pts.forEach((p, i) => { g += `<circle cx="${px(i)}" cy="${y(p[metric])}" r="4" fill="${col}" class="pt"/><circle cx="${px(i)}" cy="${y(p[metric])}" r="9" class="hit" data-s="${esc(s.id)}" data-i="${i}"/>`; });
      ends.push({id: s.id, y: y(s.pts[n - 1][metric]), col});
    });
    ends.sort((a, b) => a.y - b.y).forEach((e, i, a) => { if (i && e.y - a[i - 1].y < 13) e.y = a[i - 1].y + 13; });
    ends.forEach(e => { g += `<line x1="${W - R + 8}" x2="${W - R + 20}" y1="${e.y}" y2="${e.y}" stroke="${e.col}" stroke-width="2"/><text x="${W - R + 24}" y="${e.y + 4}" class="lbl">${esc(e.id)}</text>`; });
    box.innerHTML = `<svg width="${W}" height="${Hh}" class="rk">${g}</svg>`;
    const find = t => { const s = series.find(z => z.id === t.dataset.s); return s && {s, i: +t.dataset.i, p: s.pts[+t.dataset.i]}; };
    $$('.hit', box).forEach(c => {
      c.onmousemove = ev => { const f = find(c); tip(ev, `${f.s.id} ep${f.p.ep}\n${m[1]} ${m[2](f.p[metric])}\n排名 ${f.i + 1}/${f.s.pts.length}${f.p.flags.length ? '\nIQR 标记 ' + f.p.flags.join(' ') : ''}`); };
      c.onmouseleave = () => tip();
      c.onclick = () => { tip(); const f = find(c); openModal(f.s.pts.map(p => ({...p, title: `${m[1]} ${m[2](p[metric])}`})), f.i); };
    });
  }
  function eps() {
    const slots = colorSlots(ids), flagged = $('#fflag').checked;
    const rows = ids.flatMap(i => data[i] || []).filter(r => !flagged || r.flags.length);
    table($('#eps .tb'), rows, [
      ['id', '数据集', r => `<i class="sw" style="background:${sv(slots[r.id])}"></i>${esc(r.id)}`],
      ['ep', 'ep', r => r.ep],
      ...QM.map(([k, t, f]) => [k, t, r => r[k] == null ? '–' : f(r[k])]),
      ['flags', 'IQR 标记', r => r.flags.map(esc).join(' '), r => r.flags.length],
      ['label', '标注', r => tagChips(r.label), r => (r.label && r.label.tags || []).join()],
    ], (r, shown, k) => openModal(shown.map(x => ({...x, title: `质量 ${fx(x.q)}`, onlabel: l => { x.label = l; }})), k), metric);
  }
  $('#fflag').onchange = eps;
  $('#modal').addEventListener('closed', () => document.body.contains($('#eps')) && eps());
  const onResize = () => chart();
  window.addEventListener('resize', onResize);
  G.cleanup.push(() => window.removeEventListener('resize', onResize));
  detail();
}

// =====================================================================================================
// page 4: eval scores (docs/workflows/eval-scoring): completion, quality, total with 95% CI, stage funnel, per trial
// =====================================================================================================
const STAGE_ZH = ['前往', '抓住', '交接前不掉', '交接成功', '放好'];
async function pageEval(q, view, tools) {
  let ids = (q.ds ? q.ds.split(',') : LS.get('eval_ds', [])).filter(i => G.by[i] && G.by[i].eval);
  const setIds = v => { ids = v; LS.set('eval_ds', ids); setQuery({ds: ids.join(',')}); picker('multi', ids, d => d.eval, setIds); hist(); detail(); };
  picker('multi', ids, d => d.eval, setIds);
  tools.innerHTML = `<span class="mute">总分 = 0.5 × 完成度 + 0.5 × 质量。完成度 = 5 个阶段通过率 (通过+1)/(进入+2) 截断后平均（≤0.5 记 0）；
    质量 = 0.4 平滑 + 0.3 速度 + 0.3 不停顿，对比人类基准 ${REF_NAME} 中位数。区间：完成度/质量/总分按试次自助抽样 2000 次，阶段和成功率用 Wilson。
    规则见 docs/workflows/eval-scoring/README.zh.md。点历史表的行看详情。</span>`;
  view.innerHTML = `<div class="panel" id="hist"><h3>历史：所有评测</h3><div class="tb"></div></div><div id="det"></div>`;
  let H = null;
  function hist() {
    if (!H) return;
    const rows = H.rows.map(r => ({...r, _sel: ids.includes(r.id)})).sort((a, b) => b.version - a.version || (b.total ? b.total[0] : -1) - (a.total ? a.total[0] : -1));
    $('#hist h3').innerHTML = `历史：所有评测（新版本在上，同版本按总分排）${H.pending.length ? ` <span class="mute">· 后台计算中，还剩 ${H.pending.length} 个…</span>` : ''}`;
    table($('#hist .tb'), rows, [
      ['id', '评测数据集', r => `<span class="nm" title="${esc(r.id)}">${esc(r.id)}</span>` + (r.error ? ` <span class="err" title="${esc(r.error)}">读取失败</span>` : '')],
      ['version', '版本', r => verName(r.version)],
      ['success', '成功', r => r.success || '–', r => r.success_rate && r.success_rate[0]],
      ['success_rate', '成功率 [Wilson]', r => ciBar(r.success_rate, 100), r => r.success_rate && r.success_rate[0]],
      ['completion', '完成度 [95% CI]', r => ciBar(r.completion, 100), r => r.completion && r.completion[0]],
      ['quality', '质量 [95% CI]', r => ciBar(r.quality, 100), r => r.quality && r.quality[0]],
      ['total', '总分 [95% CI]', r => ciBar(r.total, 100), r => r.total && r.total[0]],
      ['stages', '阶段漏斗（通过/进入）', r => r.stages ? r.stages.map(s => `${s.k}/${s.n}`).join(' → ') : '–', r => r.stages && r.stages[4].k],
    ], r => { if (r.error) return; setIds(ids.includes(r.id) ? ids.filter(x => x !== r.id) : [r.id, ...ids]); });
  }
  historyRows('eval', h => { H = h; hist(); });
  async function detail() {
    const det = $('#det');
    det.innerHTML = ids.length ? '' : '<p class="mute">在历史表里点一个或多个评测，看阶段漏斗和逐试次结果。</p>';
    for (const id of ids) {
      const box = el(`<div class="panel"><h3>${esc(id)}</h3><p class="mute">加载中…</p></div>`);
      det.append(box);
      let r, info;
      try { [r, info] = await Promise.all([api('/api/eval?id=' + encodeURIComponent(id)), episodeInfo(id)]); }
      catch (e) { box.innerHTML = `<h3>${esc(id)}</h3><p class="err">${esc(e.message)}</p>`; continue; }
      const W = 300, funnel = r.stages.map((s, i) => `<tr><td>${STAGE_ZH[i]}</td><td>${s.k}/${s.n}</td><td>${s.n ? ciBar([s.k / s.n, s.ci[0], s.ci[1]], 160) : '没走到'}</td>
        <td class="mute">计分 ${fx(s.rate)}</td></tr>`).join('');
      box.innerHTML = `<h3>${esc(id)} <button class="x" style="float:right">移出</button></h3>
        <p>成功 <b>${r.success}</b> ${ciBar(r.success_rate, 100)}　完成度 ${ciBar(r.completion, 100)}　质量 ${ciBar(r.quality, 100)}　<b>总分</b> ${ciBar(r.total, 100)}</p>
        <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start"><div><h4 style="margin:4px 0">阶段漏斗（通过/进入，Wilson 95%）</h4><table class="fn">${funnel}</table></div>
        <div class="tr" style="min-width:0;flex:1"></div></div>`;
      $('.x', box).onclick = () => setIds(ids.filter(x => x !== id));
      const list = r.trials.map(t => ({...info[t.ep], ...t, id,
        start: t.fail != null ? Math.max(t.fail - 45, 0) : 0,
        title: (t.success ? '<span class="chip ok">成功</span>' : `<span class="chip bad">失败</span> 失败时刻约 ${sec(t.fail)} 秒（红色虚线）`) + ` · 质量 ${fx(t.q)}`}));
      const dots = t => t.stages.map((v, i) => { const reached = t.stages.slice(0, i).every(Boolean); return `<i class="dot ${!reached ? 'n' : v ? 'p' : 'f'}" title="${STAGE_ZH[i]}：${!reached ? '没走到' : v ? '通过' : '失败'}"></i>`; }).join('');
      table($('.tr', box), list, [
        ['ep', '试次', t => `T${t.ep + 1}`],
        ['success', '人工标签', t => `<span class="chip ${t.success ? 'ok' : 'bad'}">${t.success ? '成功' : '失败'}</span>`],
        ['fail', '失败时刻', t => t.fail == null ? '' : `<button>▶ ${sec(t.fail)} 秒</button>`],
        ['stages', '阶段 前往→放好', dots, t => t.stages.reduce((a, b) => a + b, 0)],
        ['q_smooth', '平滑', t => fx(t.q_smooth)], ['q_speed', '速度', t => fx(t.q_speed)], ['q_idle', '不停顿', t => fx(t.q_idle)],
        ['q', '质量', t => `<b>${fx(t.q)}</b>`],
        ['task_s', '用时', t => t.task_s == null ? '–' : fx(t.task_s, 1) + ' s'],
        ['rev_rate', '小幅来回', t => fx(t.rev_rate) + '/s'], ['idle_frac', '停顿占比', t => fx(100 * t.idle_frac, 0) + '%'],
        ['label', '标注', t => tagChips(t.label), t => (t.label && t.label.tags || []).join()],
      ], (t, shown, k) => openModal(shown.map(x => ({...x, onlabel: l => { x.label = l; }})), k, t.start));
    }
  }
  detail();
}
// =====================================================================================================
// boot
// =====================================================================================================

// =====================================================================================================
// page 5: curate — rules with thresholds (curate.py, the same code as the command line), what each drops, forced
// keep / drop, presets, trim spec + rebuild of the kept episodes
// =====================================================================================================
async function pageCurate(q, view, tools) {
  const def = ['raw-handover-20260929-trimmed', 'raw-handover-20261001-trimmed'];
  let ids = (q.ds ? q.ds.split(',') : LS.get('curate_ds', def)).filter(i => G.by[i] && !G.by[i].eval && !G.by[i].building);
  const P = await api('/api/curate/presets');
  let pname = LS.get('curate_preset', 'default');
  if (!P.presets[pname]) pname = 'default';
  let cur = JSON.parse(JSON.stringify({...P.default, ...(P.presets[pname] || {})})), R = null;
  let show = LS.get('curate_show', 'all'), mode = LS.get('curate_mode', 'list');  // show: all | kept | gone
  const setIds = v => { ids = v; LS.set('curate_ds', ids); setQuery({ds: ids.join(',')}); picker('multi', ids, d => !d.eval && !d.building, setIds); run(); };
  picker('multi', ids, d => !d.eval && !d.building, setIds);
  tools.innerHTML = `<label>预设 <select id="pre">${Object.keys(P.presets).map(n => `<option ${n === pname ? 'selected' : ''}>${esc(n)}</option>`).join('')}</select></label>
    <input id="pname" placeholder="新预设名" size="12"><button id="psave">另存为预设</button>
    <span id="count" class="mute"></span>
    <span class="segs" id="show">${[['all', '全部'], ['kept', '只看保留的'], ['gone', '只看去掉的']].map(([k, t]) => `<button data-s="${k}" class="${show === k ? 'on' : ''}">${t}</button>`).join('')}</span>
    <span class="segs" id="mode">${[['list', '列表'], ['frames', '起止画面审阅']].map(([k, t]) => `<button data-m="${k}" class="${mode === k ? 'on' : ''}">${t}</button>`).join('')}</span>
    <button id="bkept">批量播放保留的</button><button id="bgone">批量播放去掉的</button>
    <span class="mute">规则和数字与命令行 curate.py 相同；点行看视频，行尾按钮强制保留 / 去掉（存在 dataset_labels.json）</span>`;
  view.innerHTML = `<div class="panel" id="rules"><h3>筛选条件</h3><div class="tb"></div></div>
    <div class="panel" id="sum"><h3>结果</h3><div class="body"></div>
      <p><input id="sname" placeholder="spec 名，例如 v7-handover-curated" size="30"> <button id="mkspec">按筛选结果生成 spec</button>
      <button id="build" class="on" disabled>构建数据集…</button> <span id="smsg" class="mute"></span></p></div>
    <div class="panel" id="eps"><h3>逐条</h3><div class="tb"></div></div>
    <div class="panel" id="frames" hidden><h3>起止画面审阅 <span class="prog mute"></span></h3>
      <p class="mute">每条左边是保留范围第一帧、右边是最后一帧（头部相机）。按键：J / K 或方向键换条，1 = 开头立着，2 = 放进框里，3 = 没问题（标完跳到下一条，再按一次取消）。
      标了前两个就去掉这条；"没问题"只记作看过。<label><input type="checkbox" id="unseen">只看还没看过的</label>
      <label><input type="checkbox" id="hinted">只看有疑似提示的</label></p><div class="fgrid"></div><div class="pager"></div></div>`;
  const ruleTable = () => {
    const box = $('#rules .tb'), gone = k => R ? R.rows.filter(r => k in r.reasons).length : '';
    box.innerHTML = `<table class="rt"><tr><th>开</th><th>条件</th><th>阈值</th><th>不满足</th></tr>${P.rules.map(r => {
      const c = cur[r.key];
      const inputs = Object.keys(r.params).map(k => k === 'ref'
        ? `基准 <select data-k="${r.key}" data-p="ref">${G.ds.filter(d => !d.eval).map(d => `<option ${d.id === c.ref ? 'selected' : ''}>${esc(d.id)}</option>`).join('')}</select>`
        : `${k} <input type="number" step="any" style="width:70px" data-k="${r.key}" data-p="${k}" value="${c[k]}">`).join(' ');
      return `<tr><td><input type="checkbox" data-k="${r.key}" data-p="on" ${c.on ? 'checked' : ''}></td>
        <td>${esc(r.label)}${r.new ? ' <span class="chip bad">新指标，待确认</span>' : ''}<div class="mute help">${esc(r.help)}</div></td>
        <td>${inputs}</td><td>${c.on ? gone(r.key) + ' 条' : ''}</td></tr>`;
    }).join('')}</table>`;
    box.onchange = ev => {
      const t = ev.target, k = t.dataset.k, p = t.dataset.p;
      if (!k) return;
      cur[k][p] = p === 'on' ? t.checked : p === 'ref' ? t.value : +t.value;
      run();
    };
  };
  let tm, seq = 0;
  async function run(patch) {  // patch: only refresh marks / counts, keep the cards on screen (no reload, no flicker)
    clearTimeout(tm);
    tm = setTimeout(async () => {
      const my = ++seq;
      if (!ids.length) { $('#sum .body').innerHTML = '<p class="mute">在上面选择数据集。</p>'; return; }
      status('计算中…（第一次算一个数据集要解码首尾画面，约十几秒）');
      try { R = await api('/api/curate/run', {ids, preset: cur}); } catch (e) { status(e.message, 1); return; }
      if (my !== seq) return;
      status(''); ruleTable(); summary(); render(patch);
    }, patch ? 0 : 120);
  }
  const labelOf = k => k === 'all' ? '全部' : k === 'manual' ? '人工标记' : P.rules.find(r => r.key === k).short;
  function summary() {
    const kept = R.rows.filter(r => r.kept).length;
    $('#count').textContent = `保留 ${kept} / ${R.rows.length} 条`;
    $('#sum .body').innerHTML = `<p class="mute">基准：小幅来回 ${fx(R.refs.rev, 3)} 次/秒（${esc(cur.rev.ref)} 中位数），从臂 jerk ${fx(R.refs.jerk, 0)}（${esc(cur.jerk.ref)} 中位数）</p>` +
      Object.entries(R.steps).map(([ds, st]) => {
        const rows = R.rows.filter(r => r.ds === ds), forced = rows.filter(r => r.force).length;
        return `<p><b>${esc(ds)}</b>：${st.map(x => `${labelOf(x.key)} <b>${x.left}</b>`).join(' → ')}　最终保留 <b>${rows.filter(r => r.kept).length}</b> / ${rows.length}${forced ? `（含手动强制 ${forced} 条）` : ''}</p>`;
      }).join('') + `<p>合计保留 <b>${kept}</b> / ${R.rows.length} 条（按顺序累计；去掉的条可能同时不满足好几条）</p>` +
      `<p class="mute">人工起止画面审阅：已看 ${R.rows.filter(r => r.manual).length} / ${R.rows.length}，开头立着 ${R.rows.filter(r => r.manual === 'standing').length} 条，放进框里 ${R.rows.filter(r => r.manual === 'basket').length} 条</p>`;
    $('#bkept').textContent = `批量播放保留的 ${kept} 条`;
    $('#bgone').textContent = `批量播放去掉的 ${R.rows.length - kept} 条`;
  }
  function eps() {
    const rows = R.rows.filter(shown);
    const ratio = (v, ref) => v == null || !ref ? '–' : fx(v / ref, 2) + '×';
    table($('#eps .tb'), rows, [
      ['ds', '数据集', r => esc(r.ds)], ['ep', 'ep', r => r.ep],
      ['kept', '结果', r => `<span class="chip ${r.kept ? 'ok' : 'bad'}">${r.kept ? '保留' : '去掉'}</span>${r.force ? ' <span class="chip">手动</span>' : ''}`, r => +r.kept],
      ['reasons', '原因', r => Object.values(r.reasons).map(esc).join('；'), r => Object.keys(r.reasons).length],
      ['snaps', '突变', r => r.snaps], ['pauses', '最长停顿', r => r.pauses.length ? fx(Math.max(...r.pauses), 1) + 's' : '', r => Math.max(0, ...r.pauses)],
      ['rev_rate', '抖动', r => ratio(r.rev_rate, R.refs.rev)], ['jerk', 'jerk', r => ratio(r.jerk, R.refs.jerk)],
      ['small_snap', '小卡顿', r => r.small_snap], ['stall', '卡住', r => r.stall],
      ['closes', '闭合 L/R', r => `${r.closesL}/${r.closesR}`, r => Math.max(r.closesL, r.closesR)],
      ['tape_px_end', '末帧胶带', r => r.tape_px_end], ['tape_x_start', '先抓 / x', r => `${r.first_arm || '-'} ${r.tape_x_start ?? '–'}`],
      ['hw_start', '开头高/宽', r => r.hw_start],
      ['force', '强制', r => `<button data-f="keep" class="${r.force === 'keep' ? 'on' : ''}">保留</button><button data-f="drop" class="${r.force === 'drop' ? 'on' : ''}">去掉</button>${r.force ? '<button data-f="">自动</button>' : ''}`],
    ], async (r, shown, k, ev) => {
      const f = ev.target.dataset.f;
      if (f !== undefined) {
        await api('/api/label', {id: r.ds, ep: r.ep, patch: {force: f}}).catch(e => status(e.message, 1));
        return run();
      }
      const info = await episodeInfo(r.ds);
      const list = await Promise.all(shown.map(async x => ({...(await episodeInfo(x.ds))[x.ep], id: x.ds,
        title: `<span class="chip ${x.kept ? 'ok' : 'bad'}">${x.kept ? '保留' : '去掉'}</span> ${Object.values(x.reasons).map(esc).join('；')}`})));
      openModal(list, k);
    });
  }
  const shown = r => show === 'all' || (show === 'kept') === r.kept;
  $('#show').onclick = ev => {
    const b = ev.target.closest('[data-s]'); if (!b) return;
    show = b.dataset.s; LS.set('curate_show', show); $$('#show button').forEach(x => x.classList.toggle('on', x === b)); fpage = 0; render();
  };
  $('#mode').onclick = ev => {
    const b = ev.target.closest('[data-m]'); if (!b) return;
    mode = b.dataset.m; LS.set('curate_mode', mode); $$('#mode button').forEach(x => x.classList.toggle('on', x === b)); render();
  };
  // batch playback of the current result: the episode list goes through localStorage, #batch?pick=1 reads it
  const toBatch = kept => {
    const rows = R.rows.filter(r => r.kept === kept);
    LS.set('batch_pick', {preset: pname, what: kept ? '保留的' : '去掉的', ds: ids, eps: rows.map(r => ({ds: r.ds, ep: r.ep, kept: r.kept,
      why: [...Object.values(r.reasons), ...(r.force ? [r.force === 'drop' ? '手动强制去掉' : '手动强制保留'] : [])]}))});
    location.hash = '#batch?pick=1';
  };
  $('#bkept').onclick = () => R && toBatch(true);
  $('#bgone').onclick = () => R && toBatch(false);

  // ---- start / end frame review: big first / last head frames, one-key manual marks (manual in dataset_labels.json)
  const FPAGE = 20, MARKS = [['standing', '开头立着'], ['basket', '放进框里'], ['ok', '没问题']];
  let fpage = 0, focus = 0, flist = [], heads = {};
  const hints = r => [r.tape_px_end === 0 ? '疑似放进框里（末帧没有胶带像素）' : '', (r.hw_start || 0) >= cur.standing.r ? `疑似立着（开头色块高/宽 ${r.hw_start}）` : ''].filter(Boolean);
  async function frames(patch) {
    const box = $('#frames .fgrid');
    if (!patch) {
      await Promise.all(ids.filter(i => !heads[i]).map(async i => { heads[i] = Object.keys(Object.values(await episodeInfo(i))[0].video)[0]; }));
      flist = R.rows.filter(shown).filter(r => (!$('#unseen').checked || !r.manual) && (!$('#hinted').checked || hints(r).length));
      fpage = Math.min(fpage, Math.max(0, Math.ceil(flist.length / FPAGE) - 1));
      const sl = flist.slice(fpage * FPAGE, fpage * FPAGE + FPAGE);
      focus = Math.min(focus, sl.length - 1);
      box.innerHTML = sl.map((r, k) => {
        const img = f => `<div class="fimg"><img loading="lazy" src="${frameUrl(r.ds, heads[r.ds], r.ep, f)}&w=640"></div>`;
        return `<div class="fcard" data-k="${k}"><header><b>${esc(r.ds)} ep${r.ep}</b><span class="st"></span><span class="hint">${hints(r).map(esc).join(' · ')}</span></header>
          <div class="pair"><figure>${img(0)}<figcaption>开头（第 0 帧）</figcaption></figure><figure>${img(r.n - 1)}<figcaption>结尾（第 ${r.n - 1} 帧）</figcaption></figure></div>
          <footer>${MARKS.map(([v, t], i) => `<button data-v="${v}">${i + 1} ${t}</button>`).join('')}</footer></div>`;
      }).join('') || '<p class="mute">没有符合条件的条</p>';
      const pages = Math.max(1, Math.ceil(flist.length / FPAGE));
      $('#frames .pager').innerHTML = `<button data-p="-1">上一页</button><span>${fpage + 1} / ${pages}（${flist.length} 条）</span><button data-p="1">下一页</button>`;
      $$('#frames .fcard').forEach(c => c.onclick = ev => {
        focus = +c.dataset.k; mark();
        const b = ev.target.closest('[data-v]'); if (b) setMark(flist[fpage * FPAGE + focus], b.dataset.v);  // current marks, not the render-time copy
        else if (ev.target.tagName === 'IMG') openModal(sl.map(x => ({...(G.epi_cache[x.ds] || {})[x.ep], id: x.ds})).filter(x => x.video), focus);
      });
      await Promise.all(ids.map(async i => { G.epi_cache = G.epi_cache || {}; G.epi_cache[i] = await episodeInfo(i); }));
    }
    const byKey = Object.fromEntries(R.rows.map(r => [r.ds + '|' + r.ep, r]));
    flist = flist.map(r => byKey[r.ds + '|' + r.ep] || r);
    $$('#frames .fcard').forEach(c => {
      const r = flist[fpage * FPAGE + +c.dataset.k];
      c.classList.toggle('seen', !!r.manual);
      $$('[data-v]', c).forEach(b => b.classList.toggle('on', r.manual === b.dataset.v));
      $('.st', c).innerHTML = `<span class="chip ${r.kept ? 'ok' : 'bad'}">${r.kept ? '保留' : '去掉'}</span> ${Object.values(r.reasons).map(esc).join('；')}`;
    });
    $('#frames .prog').textContent = `已看 ${R.rows.filter(r => r.manual).length} / ${R.rows.length} · 开头立着 ${R.rows.filter(r => r.manual === 'standing').length} · 放进框里 ${R.rows.filter(r => r.manual === 'basket').length} · 最终保留 ${R.rows.filter(r => r.kept).length}`;
    mark();
  }
  const mark = () => $$('#frames .fcard').forEach(c => c.classList.toggle('focus', +c.dataset.k === focus));
  async function setMark(r, v) {
    const val = r.manual === v ? '' : v;
    try { await api('/api/label', {id: r.ds, ep: r.ep, patch: {manual: val}}); } catch (e) { return status(e.message, 1); }
    status(`${r.ds} ep${r.ep}：${val ? MARKS.find(m => m[0] === val)[1] : '已取消标记'}`);
    run(true);
    if (val) move(1);
  }
  function move(d) {
    const n = $$('#frames .fcard').length;
    if (focus + d >= n && (fpage + 1) * FPAGE < flist.length) { fpage++; focus = 0; frames(); return window.scrollTo(0, $('#frames').offsetTop - 100); }
    if (focus + d < 0 && fpage > 0) { fpage--; focus = FPAGE - 1; frames(); return; }
    focus = Math.max(0, Math.min(n - 1, focus + d)); mark();
    const c = $(`#frames .fcard[data-k="${focus}"]`); c && c.scrollIntoView({block: 'nearest'});
  }
  $('#frames .pager').onclick = ev => { const b = ev.target.closest('[data-p]'); if (!b) return; fpage = Math.max(0, fpage + +b.dataset.p); focus = 0; frames(); };
  $('#unseen').onchange = $('#hinted').onchange = () => { fpage = 0; frames(); };
  const onKey = ev => {
    if (mode !== 'frames' || ['INPUT', 'SELECT', 'TEXTAREA'].includes(ev.target.tagName) || $('#modal').open || !flist.length) return;
    const r = flist[fpage * FPAGE + focus];
    if (['j', 'J', 'ArrowDown', 'ArrowRight'].includes(ev.key)) { ev.preventDefault(); move(1); }
    else if (['k', 'K', 'ArrowUp', 'ArrowLeft'].includes(ev.key)) { ev.preventDefault(); move(-1); }
    else if (['1', '2', '3'].includes(ev.key) && r) { ev.preventDefault(); setMark(r, MARKS[+ev.key - 1][0]); }
  };
  document.addEventListener('keydown', onKey);
  G.cleanup.push(() => document.removeEventListener('keydown', onKey));
  function render(patch) {
    $('#eps').hidden = mode !== 'list'; $('#frames').hidden = mode !== 'frames';
    if (!R) return;
    if (mode === 'list') eps(); else frames(patch);
  }
  $('#pre').onchange = ev => { pname = ev.target.value; LS.set('curate_preset', pname); cur = JSON.parse(JSON.stringify({...P.default, ...P.presets[pname]})); run(); };
  $('#psave').onclick = async () => {
    const n = $('#pname').value.trim();
    if (!n) return status('先填预设名', 1);
    try { P.presets = await api('/api/curate/preset', {name: n, preset: cur}); } catch (e) { return status(e.message, 1); }
    pname = n; LS.set('curate_preset', n);
    $('#pre').innerHTML = Object.keys(P.presets).map(x => `<option ${x === n ? 'selected' : ''}>${esc(x)}</option>`).join('');
    status(`已保存预设 ${n}（presets/${n}.json）`);
  };
  let made = null;
  $('#mkspec').onclick = async () => {
    const n = $('#sname').value.trim();
    try { made = (await api('/api/curate/spec', {ids, preset: cur, name: n, preset_name: pname})).name; }
    catch (e) { $('#smsg').innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
    $('#smsg').textContent = `已写 examples/xlerobot/${made}（保留的条整条保留，其余进 drop）`;
    $('#build').disabled = false;
  };
  $('#build').onclick = () => made && rebuildDialog(made, {reset_tol: 10, to_target: false, snap_rule: false});
  ruleTable();
  await run();
  buildPanel();
}

const PAGES = {review: pageReview, batch: pageBatch, quality: pageQuality, eval: pageEval, curate: pageCurate};
async function show() {
  G.cleanup.splice(0).forEach(f => f());
  clearTimeout(G.buildT); clearTimeout(G.histT);
  $('#builds') && $('#builds').remove();
  $('#modal').open && $('#modal').close();
  const {page, q} = route();
  $$('.tabs a').forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + page));
  const view = $('#view'), tools = $('#tools');
  view.innerHTML = ''; view.style.cssText = ''; tools.innerHTML = '';
  try { await (PAGES[page] || pageReview)(q, view, tools); } catch (e) { console.error(e); view.innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}
window.addEventListener('hashchange', show);
document.addEventListener('keydown', ev => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(ev.target.tagName)) return;
  const k = RATES.indexOf(G.rate);
  if (ev.key === '>' || ev.key === '.') setRate(RATES[Math.min(RATES.length - 1, k + 1)]);
  else if (ev.key === '<' || ev.key === ',') setRate(RATES[Math.max(0, k - 1)]);
});
(async () => {
  G.rate = RATES.includes(LS.get('rate', 1)) ? LS.get('rate', 1) : 1;
  $('#rate').innerHTML = RATES.map(r => `<option value="${r}" ${r === G.rate ? 'selected' : ''}>${r}×</option>`).join('');
  $('#rate').onchange = ev => setRate(+ev.target.value);
  try {
    const r = await api('/api/datasets');
    G.ds = r.datasets; G.git = r.git; G.by = Object.fromEntries(G.ds.map(d => [d.id, d]));
  } catch (e) { status('读不到数据集列表：' + e.message, 1); }
  show();
})();
