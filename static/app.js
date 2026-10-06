const labels = { tb: 'Tốt', loai1: 'Loại 1', loai2: 'Loại 2', loai3: 'Loại 3', lbw: 'LBW', bad_output: 'Lỗi' };
const state = { file: null, img: null, predictions: [], model: null, ready: false, progressTimers: [] };
const $ = id => document.getElementById(id);
const canvas = $('imageCanvas');
const ctx = canvas.getContext('2d');

function status(ok, text) {
  state.ready = ok;
  $('connectionDot').className = `dot ${ok ? 'ready' : 'offline'}`;
  $('connectionText').textContent = text;
  $('retryConnection').classList.toggle('hidden', ok);
  $('analyzeButton').disabled = !ok || !state.file;
}

async function refreshHealth() {
  try {
    const health = await fetch('/health').then(r => { if (!r.ok) throw Error(); return r.json(); });
    const models = await fetch('/models').then(r => r.json());
    const select = $('modelSelect');
    const wanted = state.model || health.model;
    select.replaceChildren(...models.models.filter(m => m.available).map(m => new Option(m.name, m.id, m.id === wanted, m.id === wanted)));
    state.model = select.value;
    status(true, `Sẵn sàng · ${select.selectedOptions[0].text} · ${health.cuda ? 'GPU' : 'CPU'}`);
  } catch {
    status(false, 'Không kết nối được máy chủ phân tích');
  }
}

function notice(text) { $('notice').textContent = text; $('notice').classList.remove('hidden'); }
function clearProgress() { state.progressTimers.forEach(clearTimeout); state.progressTimers = []; }
function showProgress() {
  clearProgress();
  notice('Đang gửi ảnh đến máy chủ…');
  state.progressTimers.push(setTimeout(() => notice('Đang phát hiện hạt…'), 350));
  state.progressTimers.push(setTimeout(() => notice('Đang phân loại từng hạt…'), 1100));
}

function selectInput(file) {
  if (!file?.type.startsWith('image/')) return notice('Vui lòng chọn một tệp ảnh hợp lệ.');
  state.file = file; state.predictions = [];
  const img = new Image();
  img.onload = () => {
    state.img = img; canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    canvas.classList.remove('hidden'); $('emptyState').classList.add('hidden'); draw();
    $('chooseAnother').classList.remove('hidden'); $('analyzeButton').disabled = !state.ready;
    notice('Ảnh đã sẵn sàng. Bấm “Phân tích ảnh” để xem từng hạt.');
  };
  img.src = URL.createObjectURL(file);
}

function draw() {
  if (!state.img) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(state.img, 0, 0);
  state.predictions.forEach((p, i) => {
    const [x1, y1, x2, y2] = p.bbox;
    ctx.strokeStyle = p.class === 'bad_output' ? '#be3434' : '#17845f'; ctx.lineWidth = 4;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.fillStyle = ctx.strokeStyle; ctx.font = '18px system-ui';
    ctx.fillText(`${i + 1}. ${labels[p.class]} ${(p.conf_class * 100).toFixed(0)}%`, x1, Math.max(20, y1 - 7));
  });
}

function render(data) {
  clearProgress(); state.predictions = data.predictions || []; draw();
  $('totalCount').textContent = data.count; $('latency').textContent = `${data.latency_ms} ms`; $('modelUsed').textContent = data.model_used;
  if (!state.predictions.length) { $('topClass').textContent = 'Không có'; notice('Không phát hiện hạt điều trong ảnh này. Hãy thử ảnh khác rõ hơn.'); return; }
  const counts = getQualitySummary(state.predictions);
  const top = Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0];
  $('topClass').textContent = labels[top]; $('acceptedCount').textContent = ['tb', 'loai1', 'loai2'].reduce((n, key) => n + (counts[key] || 0), 0);
  $('yoloConfidence').textContent = `${(Math.max(...state.predictions.map(p => p.conf_detect)) * 100).toFixed(1)}%`;
  $('classConfidence').textContent = `${(Math.max(...state.predictions.map(p => p.conf_class)) * 100).toFixed(1)}%`;
  const list = $('perNut'); list.replaceChildren(...state.predictions.map((p, i) => {
    const item = document.createElement('li'); item.textContent = `Hạt ${i + 1}: ${labels[p.class]} — ${(p.conf_class * 100).toFixed(1)}%`; return item;
  }));
  $('resultPanel').classList.remove('hidden'); notice('Phân tích hoàn tất. Mỗi hạt đã có nhãn phân loại và độ tin cậy.');
}

function getQualitySummary(predictions) {
  return predictions.reduce((counts, prediction) => {
    counts[prediction.class] = (counts[prediction.class] || 0) + 1;
    return counts;
  }, {});
}

async function runAnalysis() {
  if (!state.file || !state.ready) return;
  const button = $('analyzeButton'); button.disabled = true; button.textContent = 'Đang phân tích…'; showProgress();
  try {
    const body = new FormData(); body.append('file', state.file);
    const modelParam = state.model ? `?model=${state.model}` : '';
    const result = await fetch(`/predict${modelParam}`, { method: 'POST', body }).then(async r => { if (!r.ok) throw Error((await r.json()).detail); return r.json(); });
    render(result);
  } catch (error) { clearProgress(); notice(error.message || 'Không thể phân tích ảnh.'); }
  finally { button.textContent = 'Phân tích ảnh'; button.disabled = !state.ready || !state.file; }
}

async function samples() {
  try {
    const data = await fetch('/samples/manifest.json').then(r => r.json());
    data.samples.forEach((sample, index) => {
      const number = index + 1; const button = document.createElement('button'); button.className = 'sample';
      button.innerHTML = `<img src="${sample.path}" alt="Ảnh mẫu ${number}"><span>Ảnh ${number}</span><small>Dùng thử</small>`;
      button.onclick = async () => { const blob = await fetch(sample.path).then(r => r.blob()); selectInput(new File([blob], `${sample.id}.png`, { type: blob.type || 'image/png' })); $('workspace').scrollIntoView({ behavior: 'smooth' }); };
      $('sampleGrid').append(button);
    });
  } catch { notice('Không tải được ảnh mẫu.'); }
}

function addResearchLink() {
  const link = document.createElement('a'); link.href = '/research.html'; link.textContent = 'Dữ liệu & nghiên cứu'; link.className = 'research-link';
  document.querySelector('.top')?.append(link);
}

$('modelSelect').onchange = event => { state.model = event.target.value; refreshHealth(); };
$('chooseImage').onclick = () => $('fileInput').click(); $('chooseAnother').onclick = () => $('fileInput').click();
$('fileInput').onchange = event => selectInput(event.target.files[0]); $('analyzeButton').onclick = runAnalysis; $('retryConnection').onclick = refreshHealth;
$('useSamples').onclick = () => $('sampleSection').scrollIntoView({ behavior: 'smooth' });
$('dropzone').ondragover = event => event.preventDefault(); $('dropzone').ondrop = event => { event.preventDefault(); selectInput(event.dataTransfer.files[0]); };
addResearchLink(); refreshHealth(); samples();
