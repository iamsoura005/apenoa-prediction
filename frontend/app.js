/**
 * ApneaSphere Enterprise Web Client
 * High-performance clinical PPG visualizer, telemetry client, and persistent record manager.
 */

const API_BASE = window.API_BASE_URL 
  || localStorage.getItem("APNEA_API_URL") 
  || (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" ? "" : "https://apneasphere-api.onrender.com");

let ppgChartInstance = null;
let currentSignalData = [];
let currentFs = 80;
let currentSubjectId = "Anonymous";
let lastAnalysisResult = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }
  await fetchSampleBenchmarks();
  initEmptyChart();
});

function switchTab(tabId) {
  ['workspace', 'history', 'transparency', 'science'].forEach(t => {
    const el = document.getElementById(`tab-${t}`);
    const btn = document.getElementById(`tab-btn-${t}`);
    if (el) el.classList.add('hidden');
    if (btn) btn.classList.remove('active');
  });

  const activeEl = document.getElementById(`tab-${tabId}`);
  const activeBtn = document.getElementById(`tab-btn-${tabId}`);
  if (activeEl) activeEl.classList.remove('hidden');
  if (activeBtn) activeBtn.classList.add('active');

  if (tabId === 'history') {
    fetchHistory();
  }

  if (window.lucide) window.lucide.createIcons();
}

async function fetchSampleBenchmarks() {
  const container = document.getElementById("sample-buttons-container");
  try {
    const res = await fetch(`${API_BASE}/api/samples`);
    const samples = await res.json();
    
    if (!samples || samples.length === 0) {
      container.innerHTML = `<span class="text-xs text-slate-500">No pre-loaded cases found.</span>`;
      return;
    }

    container.innerHTML = "";
    samples.forEach((sample) => {
      const isApnea = sample.condition.toLowerCase().includes("apnea");
      const badgeColor = isApnea ? "text-rose-400 bg-rose-950/60 border-rose-800" : "text-emerald-400 bg-emerald-950/60 border-emerald-800";
      
      const btn = document.createElement("button");
      btn.id = `sample-btn-${sample.id}`;
      btn.className = "benchmark-btn p-3 rounded-xl border transition text-left flex flex-col justify-between cursor-pointer";
      btn.onclick = () => {
        document.querySelectorAll('.benchmark-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        loadBenchmarkCase(sample.id);
      };
      
      btn.innerHTML = `
        <div>
          <div class="flex items-center justify-between mb-1.5">
            <span class="text-[10px] font-bold px-2 py-0.5 rounded-full border ${badgeColor}">${sample.condition}</span>
            <span class="text-[10px] text-slate-500 font-mono">${sample.fs} Hz</span>
          </div>
          <div class="font-bold text-xs text-slate-200 line-clamp-1">${sample.name}</div>
        </div>
        <p class="text-[10px] text-slate-400 mt-2 line-clamp-2">${sample.description}</p>
      `;
      container.appendChild(btn);
    });

    if (samples.length > 0) {
      document.getElementById(`sample-btn-${samples[0].id}`)?.classList.add('selected');
      loadBenchmarkCase(samples[0].id);
    }
  } catch (err) {
    console.error("Failed to load sample cases:", err);
    container.innerHTML = `<span class="text-xs text-rose-400">Failed to connect to backend samples.</span>`;
  }
}

async function loadBenchmarkCase(sampleId) {
  try {
    showLoader(true);
    const res = await fetch(`${API_BASE}/api/samples/${sampleId}`);
    const data = await res.json();
    
    currentSignalData = data.signal;
    currentFs = data.fs || 80;
    currentSubjectId = data.name || sampleId;

    await runInference();
  } catch (err) {
    alert("Error loading benchmark case: " + err.message);
  } finally {
    showLoader(false);
  }
}

async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);
  formData.append("fs", "80");
  formData.append("subject_id", file.name.replace(/\.[^/.]+$/, ""));

  try {
    showLoader(true);
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed");
    }

    const result = await res.json();
    displayResults(result);
  } catch (err) {
    alert("Upload Error: " + err.message);
  } finally {
    showLoader(false);
  }
}

async function runInference() {
  if (!currentSignalData || currentSignalData.length === 0) {
    alert("Please select a benchmark case or upload a PPG file first.");
    return;
  }

  showLoader(true);
  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        signal: currentSignalData,
        fs: currentFs,
        subject_id: currentSubjectId
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Inference failed");
    }

    const result = await res.json();
    displayResults(result);
  } catch (err) {
    alert("Inference Error: " + err.message);
  } finally {
    showLoader(false);
  }
}

function displayResults(res) {
  lastAnalysisResult = res;

  const btnPdf = document.getElementById("btn-download-pdf");
  if (btnPdf) btnPdf.removeAttribute("disabled");

  const hrv = res.biomarkers?.hrv || {};
  const spec = res.biomarkers?.spectral || {};
  const sqi = res.signal_quality || {};

  document.getElementById("val-hr").textContent = hrv.mean_hr_bpm ? `${hrv.mean_hr_bpm}` : "--";
  document.getElementById("val-sdnn").textContent = hrv.sdnn_ms ? `${hrv.sdnn_ms}` : "--";
  document.getElementById("val-rmssd").textContent = hrv.rmssd_ms ? `${hrv.rmssd_ms}` : "--";
  document.getElementById("val-lfhf").textContent = spec.lf_hf_ratio !== undefined ? `${spec.lf_hf_ratio}` : "--";
  document.getElementById("val-peaks").textContent = hrv.peak_count !== undefined ? `${hrv.peak_count}` : "--";
  document.getElementById("val-sqi").textContent = sqi.sqi_score !== undefined ? `${sqi.sqi_score}` : "--";
  document.getElementById("telemetry-sqi").textContent = `${sqi.quality.toUpperCase()} (${sqi.sqi_score})`;

  const banner = document.getElementById("results-banner");
  banner.classList.remove("hidden");

  const isApnea = res.prediction === "APNEA_EVENT";
  const bgStyle = isApnea 
    ? "bg-rose-950/40 border-rose-800/80 text-rose-200" 
    : "bg-emerald-950/40 border-emerald-800/80 text-emerald-200";
  const pillStyle = isApnea ? "bg-rose-500 text-white" : "bg-emerald-500 text-slate-950";

  banner.innerHTML = `
    <div class="p-5 rounded-2xl border ${bgStyle} flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-xl">
      <div class="flex items-start space-x-3.5">
        <div class="p-2.5 rounded-xl ${isApnea ? 'bg-rose-500/20 text-rose-400' : 'bg-emerald-500/20 text-emerald-400'} mt-0.5">
          <i data-lucide="${isApnea ? 'alert-triangle' : 'check-circle'}" class="w-6 h-6"></i>
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <span class="text-xs font-bold uppercase tracking-wider px-2 py-0.5 rounded ${pillStyle}">
              ${res.prediction.replace('_', ' ')}
            </span>
            <span class="text-xs font-semibold text-slate-300">Model Probability: <b>${(res.probability * 100).toFixed(1)}%</b></span>
            <span class="text-xs font-mono text-slate-400">Score Tier: <b>${res.risk_level}</b></span>
          </div>
          <p class="text-sm font-semibold text-slate-100 mt-1.5">${res.clinical_interpretation}</p>
          <p class="text-xs text-slate-400 mt-0.5">Subject: <span class="font-mono text-slate-300">${res.subject_id}</span> | Lead Time: <b>5.0s</b> | Analysis ID: <span class="font-mono text-slate-400">${res.analysis_id.slice(0, 8)}</span></p>
        </div>
      </div>
      <div class="flex items-center space-x-2 flex-shrink-0">
        <button onclick="downloadReport()" class="btn-primary text-xs">
          <i data-lucide="file-down" class="w-4 h-4 mr-1"></i> Export Clinical Report
        </button>
      </div>
    </div>
  `;

  if (window.lucide) window.lucide.createIcons();

  updateChart(res.filtered_signal_preview, res.peak_indices_preview);
}

function initEmptyChart() {
  const ctx = document.getElementById("ppg-chart").getContext("2d");
  ppgChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: Array.from({ length: 600 }, (_, i) => (i * 0.05).toFixed(1)),
      datasets: [{
        label: "PPG Amplitude",
        data: Array(600).fill(0),
        borderColor: "#38bdf8",
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid: { color: "rgba(51, 65, 85, 0.3)" },
          ticks: { color: "#64748b", font: { size: 10 }, maxTicksLimit: 12 },
          title: { display: true, text: "Time (seconds)", color: "#94a3b8", font: { size: 11, weight: 'bold' } }
        },
        y: {
          grid: { color: "rgba(51, 65, 85, 0.3)" },
          ticks: { color: "#64748b", font: { size: 10 } },
          title: { display: true, text: "Normalized Amplitude", color: "#94a3b8", font: { size: 11, weight: 'bold' } }
        }
      }
    }
  });
}

function updateChart(signalPreview, peakIndices) {
  if (!ppgChartInstance) return;

  const labels = signalPreview.map((_, i) => (i * 0.05).toFixed(1));
  const pointRadii = signalPreview.map((_, i) => (peakIndices.includes(i) ? 5 : 0));
  const pointColors = signalPreview.map((_, i) => (peakIndices.includes(i) ? "#f43f5e" : "#38bdf8"));

  ppgChartInstance.data.labels = labels;
  ppgChartInstance.data.datasets = [{
    label: "PPG Pulse Waveform",
    data: signalPreview,
    borderColor: "#38bdf8",
    borderWidth: 1.8,
    pointRadius: pointRadii,
    pointBackgroundColor: pointColors,
    pointBorderColor: "#ffffff",
    pointBorderWidth: 1.5,
    tension: 0.1
  }];
  ppgChartInstance.update();
}

async function fetchHistory() {
  const tbody = document.getElementById("history-table-body");
  if (!tbody) return;

  try {
    const res = await fetch(`${API_BASE}/api/history?limit=30`);
    const history = await res.json();

    if (!history || history.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="text-center py-6 text-slate-500 font-sans">No persistent analyses logged yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = "";
    history.forEach(item => {
      const isApnea = item.prediction === "APNEA_EVENT";
      const badge = isApnea 
        ? `<span class="px-2 py-0.5 rounded text-[10px] bg-rose-950 border border-rose-800 text-rose-300 font-bold">APNEA</span>`
        : `<span class="px-2 py-0.5 rounded text-[10px] bg-emerald-950 border border-emerald-800 text-emerald-300 font-bold">NORMAL</span>`;

      const tr = document.createElement("tr");
      tr.className = "hover:bg-slate-900/60 transition";
      tr.innerHTML = `
        <td class="py-3 px-4 text-slate-400 font-mono text-[11px]">${item.timestamp}</td>
        <td class="py-3 px-4 font-bold text-slate-200">${item.subject_id}</td>
        <td class="py-3 px-4">${badge}</td>
        <td class="py-3 px-4 text-cyan-400 font-bold">${(item.probability * 100).toFixed(1)}%</td>
        <td class="py-3 px-4 text-slate-300">${item.risk_level}</td>
        <td class="py-3 px-4 text-slate-200">${item.mean_hr_bpm} bpm</td>
        <td class="py-3 px-4 text-slate-200">${item.rmssd_ms} ms</td>
        <td class="py-3 px-4 text-right">
          <a href="${API_BASE}/api/report/${item.id}" target="_blank" class="text-cyan-400 hover:text-cyan-300 font-sans text-xs underline font-semibold">
            PDF Report
          </a>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

function downloadReport() {
  if (!lastAnalysisResult || !lastAnalysisResult.analysis_id) {
    alert("Please run an analysis first.");
    return;
  }
  window.open(`${API_BASE}/api/report/${lastAnalysisResult.analysis_id}`, "_blank");
}

function showLoader(visible) {
  const loader = document.getElementById("chart-loader");
  if (loader) {
    if (visible) loader.classList.remove("hidden");
    else loader.classList.add("hidden");
  }
}
