const corpusFile = document.getElementById("corpusFile");
const uploadCorpusButton = document.getElementById("uploadCorpusButton");
const uploadState = document.getElementById("uploadState");
const corpusStatus = document.getElementById("corpusStatus");
const corpusKpis = document.getElementById("corpusKpis");
const runAllButton = document.getElementById("runAllButton");
const calibrateButton = document.getElementById("calibrateButton");
const landscapeButton = document.getElementById("landscapeButton");
const landscapeTopButton = document.getElementById("landscapeTopButton");
const figureButton = document.getElementById("figureButton");
const figureTopButton = document.getElementById("figureTopButton");
const calibrationSort = document.getElementById("calibrationSort");
const calibrationList = document.getElementById("calibrationList");
const gapList = document.getElementById("gapList");
const calibrationCount = document.getElementById("calibrationCount");
const gapCount = document.getElementById("gapCount");
const figureList = document.getElementById("figureList");
const figureCount = document.getElementById("figureCount");
const embeddingState = document.getElementById("embeddingState");
const embeddingApiUrl = document.getElementById("embeddingApiUrl");
const embeddingModel = document.getElementById("embeddingModel");
const embeddingBatchSize = document.getElementById("embeddingBatchSize");
const embeddingTimeout = document.getElementById("embeddingTimeout");
const saveEmbeddingButton = document.getElementById("saveEmbeddingButton");
const testEmbeddingButton = document.getElementById("testEmbeddingButton");
const loraState = document.getElementById("loraState");
const loraBaseModel = document.getElementById("loraBaseModel");
const loraEmbeddingModel = document.getElementById("loraEmbeddingModel");
const loraMaxExamples = document.getElementById("loraMaxExamples");
const loraButton = document.getElementById("loraButton");
const loraLinks = document.getElementById("loraLinks");
const standardState = document.getElementById("standardState");
const standardSourceDir = document.getElementById("standardSourceDir");
const standardMode = document.getElementById("standardMode");
const standardAllPdfs = document.getElementById("standardAllPdfs");
const extractStandardsButton = document.getElementById("extractStandardsButton");
const standardCount = document.getElementById("standardCount");
const standardSummary = document.getElementById("standardSummary");
const standardList = document.getElementById("standardList");
const standardDomainFilter = document.getElementById("standardDomainFilter");
const standardCompileFilter = document.getElementById("standardCompileFilter");

let cachedCalibrations = [];
const corpusLanguage = () => window.PlatformI18n?.getLanguage?.() || localStorage.getItem("concrete_platform_language") || "zh";
const isEnglishCorpus = () => corpusLanguage() === "en";
const hasChineseText = (value) => /[\u4e00-\u9fff]/.test(String(value || ""));
const corpusText = (zh, en) => (isEnglishCorpus() ? en : zh);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function settingsPayload() {
  return {
    top_k: Number(document.getElementById("topKInput").value || 8),
    max_docs: Number(document.getElementById("maxDocsInput").value || 0),
  };
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.status === "error") {
    throw new Error(data.message || `请求失败: ${response.status}`);
  }
  return data;
}

function setBusy(isBusy, label = "") {
  [uploadCorpusButton, runAllButton, calibrateButton, landscapeButton, landscapeTopButton, figureButton, figureTopButton, saveEmbeddingButton, testEmbeddingButton, loraButton, extractStandardsButton].forEach((button) => {
    if (button) button.disabled = isBusy;
  });
  if (label) corpusStatus.textContent = label;
}

function embeddingPayload() {
  return {
    api_url: embeddingApiUrl.value.trim(),
    model: embeddingModel.value.trim(),
    batch_size: Number(embeddingBatchSize.value || 32),
    timeout_seconds: Number(embeddingTimeout.value || 120),
  };
}

function fillEmbeddingForm(config = {}) {
  if (config.api_url) embeddingApiUrl.value = config.api_url;
  if (config.model) embeddingModel.value = config.model;
  if (config.batch_size) embeddingBatchSize.value = config.batch_size;
  if (config.timeout_seconds) embeddingTimeout.value = config.timeout_seconds;
}

function renderKpis(status) {
  const corpus = status.corpus || {};
  const calibration = status.calibration || {};
  const items = [
    [corpusText("摘要记录", "Abstract Records"), corpus.doc_count ?? 0, corpus.source_filename || corpusText("未上传", "Not uploaded")],
    [corpusText("精选 KG 链", "Curated KG Chains"), calibration.kg_chain_count ?? 0, calibration.updated_at ? corpusText("已校准", "Calibrated") : corpusText("待校准", "Pending calibration")],
    [corpusText("高置信链", "High-Confidence Chains"), calibration.high_confidence_count ?? 0, corpusText("证据层支持", "Evidence-layer support")],
    [corpusText("证据用途", "Evidence Usage"), corpusText("不扩图", "No graph expansion"), corpusText("仅校准、检索、缺口识别", "Calibration, retrieval, and gap analysis only")],
  ];
  corpusKpis.innerHTML = items.map(([label, value, note]) => `
    <article>
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(note)}</small>
    </article>
  `).join("");
  corpusStatus.textContent = corpus.doc_count ? corpusText(`语料 ${corpus.doc_count} 条`, `Corpus ${corpus.doc_count} records`) : corpusText("等待上传", "Waiting for upload");
  uploadState.textContent = corpus.doc_count ? corpusText("已解析", "Parsed") : corpusText("待上传", "Waiting");
}

function evidenceBadge(item) {
  const score = Number(item.confidence_score || 0);
  if (score >= 0.72) return "high";
  if (score >= 0.52) return "mid";
  return "low";
}

function renderCalibrationRows(rows) {
  calibrationCount.textContent = rows.length;
  if (!rows.length) {
    calibrationList.innerHTML = '<div class="empty-state">尚未运行证据校准</div>';
    return;
  }
  calibrationList.innerHTML = rows.map((item) => {
    const support = item.supporting_abstracts || [];
    const contra = item.contradicting_abstracts || [];
    return `
      <article class="calibration-row ${evidenceBadge(item)}">
        <header>
          <div>
            <strong>${escapeHtml(localizedKgRoute(item.route || []))}</strong>
            <span>${escapeHtml(localizedKgValue(item.material, "Material"))} / ${escapeHtml(localizedKgValue(item.performance, "Performance"))} / ${escapeHtml(localizedKgValue(item.relation, "Relation"))}</span>
          </div>
          <b>${escapeHtml(item.confidence_score)}</b>
        </header>
        <div class="mini-metrics">
        <span>${escapeHtml(corpusText("支撑证据", "Supporting evidence"))} ${escapeHtml(item.evidence_count)}</span>
        <span>${escapeHtml(corpusText("近年证据", "Recent evidence"))} ${escapeHtml(item.recent_evidence_count)}</span>
        <span>${escapeHtml(corpusText("反向线索", "Contradictory signals"))} ${escapeHtml(contra.length)}</span>
        </div>
        <details>
          <summary>查看支撑摘要</summary>
          ${support.slice(0, 5).map((doc) => `
            <section class="abstract-hit">
              <strong>${escapeHtml(doc.title)}</strong>
              <span>${escapeHtml(doc.year || "-")} · ${escapeHtml(doc.journal || "-")} · score ${escapeHtml(doc.score)}</span>
              <p>${escapeHtml(doc.abstract)}</p>
            </section>
          `).join("")}
        </details>
      </article>
    `;
  }).join("");
}

function gapType(item) {
  const contra = item.contradicting_abstracts || [];
  if ((item.evidence_count || 0) < 3) return "证据不足";
  if (contra.length >= 3) return "适用边界";
  if ((item.recent_evidence_count || 0) === 0) return "近期证据弱";
  if ((item.confidence_score || 0) < 0.55) return "低置信";
  return "可作为证据链";
}

function renderGapRows(rows) {
  const gaps = rows
    .map((item) => ({ ...item, gap_type: gapType(item) }))
    .filter((item) => item.gap_type !== "可作为证据链")
    .sort((a, b) => (a.evidence_count || 0) - (b.evidence_count || 0) || (b.contradicting_abstracts || []).length - (a.contradicting_abstracts || []).length)
    .slice(0, 80);
  gapCount.textContent = gaps.length;
  if (!gaps.length) {
    gapList.innerHTML = '<div class="empty-state">当前 Gold KG 均有摘要证据支撑；可按“证据缺口”排序继续审查。</div>';
    return;
  }
  gapList.innerHTML = gaps.map((item) => `
    <article class="candidate-row">
      <header>
        <strong>${escapeHtml(item.gap_type)}</strong>
        <b>${escapeHtml(item.confidence_score)}</b>
      </header>
      <div class="candidate-route">
        <span>${escapeHtml(localizedKgValue(item.material, "Material"))}</span>
        <span>${escapeHtml(localizedKgValue(item.performance, "Performance"))}</span>
        <span>${escapeHtml(localizedKgValue(item.relation, "Relation"))}</span>
      </div>
      <p>${escapeHtml(localizedKgRoute(item.route || []))}</p>
      <small>${escapeHtml(corpusText("支撑证据", "Supporting evidence"))} ${escapeHtml(item.evidence_count)} · ${escapeHtml(corpusText("近年证据", "Recent evidence"))} ${escapeHtml(item.recent_evidence_count)} · ${escapeHtml(corpusText("反向线索", "Contradictory signals"))} ${(item.contradicting_abstracts || []).length}</small>
    </article>
  `).join("");
}

function figureUrl(filePath) {
  const name = String(filePath || "").split(/[\\/]/).pop();
  return `/api/abstract-corpus/figures/${encodeURIComponent(name)}`;
}

function renderFigures(figures = []) {
  figureCount.textContent = figures.length;
  if (!figures.length) {
    figureList.innerHTML = '<div class="empty-state">尚未生成图件</div>';
    return;
  }
  figureList.innerHTML = figures.map((fig) => {
    const files = fig.files || {};
    const png = files.png ? figureUrl(files.png) : "";
    return `
      <article class="figure-card">
        <header>
          <strong>${escapeHtml(fig.title || fig.id)}</strong>
          <span>${escapeHtml(fig.id || "")}</span>
        </header>
        ${png ? `<img src="${escapeHtml(png)}" alt="${escapeHtml(fig.title || fig.id)}" />` : ""}
        <div class="figure-links">
          ${["svg", "pdf", "tiff", "png"].map((ext) => files[ext] ? `<a href="${escapeHtml(figureUrl(files[ext]))}" target="_blank">${ext.toUpperCase()}</a>` : "").join("")}
        </div>
      </article>
    `;
  }).join("");
}

function renderLora(status = {}) {
  const manifest = status.manifest || {};
  const files = status.files || {};
  if (!status.exists && !manifest.train_examples) {
    loraState.textContent = "未生成";
    loraLinks.innerHTML = '<div class="empty-state small-empty">先完成 KG 校准和 Evidence Landscape，再生成 LoRA 数据。</div>';
    return;
  }
  loraState.textContent = `${manifest.train_examples || 0}/${manifest.validation_examples || 0}`;
  if (manifest.base_model) loraBaseModel.value = manifest.base_model;
  if (manifest.embedding_model) loraEmbeddingModel.value = manifest.embedding_model;
  const links = [
    ["train", "训练集"],
    ["validation", "验证集"],
    ["manifest", "Manifest"],
    ["script", "训练脚本"],
  ];
  loraLinks.innerHTML = `
    <div class="lora-summary">
      <strong>${escapeHtml(manifest.base_model || "Qwen/Qwen3-4B-Instruct-2507")}</strong>
      <span>${escapeHtml((manifest.tasks || []).join(" · ") || "evidence classification · mechanism extraction · agent routing")}</span>
    </div>
    <div class="button-grid">
      ${links.map(([key, label]) => files[key]?.url ? `<a href="${escapeHtml(files[key].url)}" target="_blank">${escapeHtml(label)}</a>` : "").join("")}
    </div>
  `;
}

const domainLabels = {
  general_concrete: { zh: "普通混凝土", en: "General Concrete" },
  mix_design_concrete: { zh: "配合比设计", en: "Mix Design Concrete" },
  pumped_concrete: { zh: "泵送混凝土", en: "Pumped Concrete" },
  self_compacting_concrete: { zh: "自密实混凝土", en: "Self-Compacting Concrete" },
  high_strength_concrete: { zh: "高强混凝土", en: "High-Strength Concrete" },
  high_performance_concrete: { zh: "高性能混凝土", en: "High-Performance Concrete" },
  reactive_powder_concrete: { zh: "活性粉末/UHPC", en: "Reactive Powder / UHPC" },
  marine_concrete: { zh: "水运/海工混凝土", en: "Waterway / Marine Concrete" },
  durability_concrete: { zh: "耐久性混凝土", en: "Durability Concrete" },
};
const localizedDomain = (key) => {
  const item = domainLabels[key];
  if (!item) return key || corpusText("规范", "Standard");
  return isEnglishCorpus() ? item.en : item.zh;
};
const localizedStandardSource = (name) => {
  if (!isEnglishCorpus() || !hasChineseText(name)) return name || "-";
  return "Standards source PDF";
};
const localizedStandardEvidence = (evidence) => {
  if (!isEnglishCorpus() || !hasChineseText(evidence)) return evidence || "";
  return "Original standard clause is retained in the Chinese source text. Switch to Chinese mode to inspect the full original excerpt.";
};
const localizedKgValue = (value, fallback = "KG concept") => {
  if (!isEnglishCorpus() || !hasChineseText(value)) return value || "-";
  return fallback;
};
const localizedKgRoute = (route = []) => {
  const joined = (route || []).join(" -> ");
  if (!isEnglishCorpus() || !hasChineseText(joined)) return joined;
  return "Gold KG mechanism route";
};

function renderStandardSummary(summary = {}) {
  const metrics = summary.metrics || {};
  const domains = summary.domains || {};
  const metricText = Object.entries(metrics).map(([key, value]) => `${key} ${value}`).join(" · ") || "无";
  const domainText = Object.entries(domains)
    .map(([key, value]) => `${localizedDomain(key)} ${value}`)
    .join(" · ") || corpusText("无", "None");
  standardSummary.innerHTML = `
    <div class="mini-metrics">
      <span>PDF ${escapeHtml(summary.processed_pdf_count || 0)}</span>
      <span>${escapeHtml(corpusText("约束/依据", "Constraints / references"))} ${escapeHtml(summary.constraint_count || 0)}</span>
      <span>${escapeHtml(corpusText("可优化", "Optimizable"))} ${escapeHtml(summary.optimizable_count || 0)}</span>
      <span>${escapeHtml(summary.mode || "precision")}</span>
    </div>
    <p><b>${escapeHtml(corpusText("指标", "Metrics"))}</b> ${escapeHtml(isEnglishCorpus() && metricText === "无" ? "None" : metricText)}</p>
    <p><b>${escapeHtml(corpusText("类型", "Types"))}</b> ${escapeHtml(domainText)}</p>
  `;
}

function constraintExpr(row) {
  const metric = row.metric || "-";
  const unit = row.unit || "";
  if (row.operator === "range") return `${metric}: ${row.lower ?? "-"} - ${row.upper ?? "-"} ${unit}`;
  if (row.operator === "<=" || row.operator === "<") return `${metric}: <= ${row.upper ?? row.value ?? "-"} ${unit}`;
  if (row.operator === ">=" || row.operator === ">") return `${metric}: >= ${row.lower ?? row.value ?? "-"} ${unit}`;
  if (row.operator === "class") return `${metric}: C${row.value ?? "-"} ${unit}`;
  return `${metric}: ${row.value ?? row.lower ?? row.upper ?? "-"} ${unit}`;
}

function renderStandards(payload = {}) {
  const rows = payload.rows || [];
  const summary = payload.summary || {};
  standardCount.textContent = payload.total ?? rows.length;
  standardState.textContent = `${summary.optimizable_count || 0}/${summary.constraint_count || 0}`;
  renderStandardSummary(summary);
  if (!rows.length) {
    standardList.innerHTML = '<div class="empty-state">尚未抽取规范，或当前筛选条件无结果。</div>';
    return;
  }
  standardList.innerHTML = rows.map((row) => `
    <article class="candidate-row ${row.compile_status === "optimizable" ? "standard-optimizable" : ""}">
      <header>
        <strong>${escapeHtml(localizedDomain(row.domain))}</strong>
        <b>${escapeHtml(row.compile_status === "optimizable" ? corpusText("优化约束", "Optimization constraint") : corpusText("依据", "Reference"))}</b>
      </header>
      <div class="candidate-route">
        <span>${escapeHtml(localizedStandardSource(row.source_file))}</span>
        <span>p.${escapeHtml(row.page || "-")}</span>
        <span>${escapeHtml(constraintExpr(row))}</span>
      </div>
      <p>${escapeHtml(localizedStandardEvidence(row.evidence))}</p>
      <small>confidence ${escapeHtml(row.confidence || "-")} · ${escapeHtml(row.extraction_method || "-")}</small>
    </article>
  `).join("");
}

async function loadStandards() {
  try {
    const params = new URLSearchParams();
    if (standardDomainFilter.value) params.set("domain", standardDomainFilter.value);
    if (standardCompileFilter.value) params.set("compile_status", standardCompileFilter.value);
    params.set("limit", "120");
    const data = await api(`/api/standards/constraints?${params.toString()}`);
    renderStandards(data);
  } catch {
    renderStandards({ rows: [], total: 0, summary: {} });
  }
}

async function ensureStandardsReady() {
  try {
    const status = await api("/api/standards/status");
    renderStandardSummary(status.summary || {});
    standardState.textContent = status.has_outputs
      ? `${status.optimizable_count || 0}/${status.constraint_count || 0}`
      : "待处理";
    if (!status.has_outputs || !status.constraint_count) {
      setBusy(true, "正在预处理默认规范库");
      const data = await api("/api/standards/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_dir: standardSourceDir.value.trim() || status.source_dir,
          mode: standardMode.value || "precision",
          all_pdfs: false,
        }),
      });
      renderStandardSummary(data.summary || {});
      standardState.textContent = `${data.summary?.optimizable_count || 0}/${data.summary?.constraint_count || 0}`;
    }
  } catch (error) {
    standardState.textContent = "失败";
    corpusStatus.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

async function refreshStatus() {
  const status = await api("/api/abstract-corpus/status");
  renderKpis(status);
  if (status.embedding) fillEmbeddingForm(status.embedding);
  if (status.lora) renderLora(status.lora);
  return status;
}

async function loadCalibration() {
  try {
    const data = await api(`/api/abstract-corpus/calibration?limit=108&sort=${encodeURIComponent(calibrationSort.value)}`);
    cachedCalibrations = data.calibrations || [];
    renderCalibrationRows(cachedCalibrations.slice(0, 80));
    renderGapRows(cachedCalibrations);
  } catch {
    cachedCalibrations = [];
    renderCalibrationRows([]);
    renderGapRows([]);
  }
}

async function loadFigures() {
  try {
    const data = await api("/api/abstract-corpus/figures");
    renderFigures(data.figures || []);
  } catch {
    renderFigures([]);
  }
}

async function loadEmbeddingConfig() {
  try {
    const data = await api("/api/abstract-corpus/embedding-config");
    fillEmbeddingForm(data.config || {});
  } catch {
    embeddingState.textContent = "读取失败";
  }
}

async function loadLoraStatus() {
  try {
    const data = await api("/api/abstract-corpus/lora");
    renderLora(data);
  } catch {
    renderLora({});
  }
}

uploadCorpusButton.addEventListener("click", async () => {
  const file = corpusFile.files?.[0];
  if (!file) {
    corpusStatus.textContent = "请选择摘要文件";
    return;
  }
  const form = new FormData();
  form.append("file", file);
  setBusy(true, "正在解析摘要语料");
  try {
    const data = await api("/api/abstract-corpus/upload", { method: "POST", body: form });
    uploadState.textContent = `已解析 ${data.meta.doc_count}`;
    await refreshStatus();
  } catch (error) {
    corpusStatus.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

async function runCalibration() {
  setBusy(true, "正在校准精选 KG");
  try {
    await api("/api/abstract-corpus/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settingsPayload()),
    });
    await refreshStatus();
    await loadCalibration();
  } catch (error) {
    corpusStatus.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

calibrateButton.addEventListener("click", runCalibration);
runAllButton.addEventListener("click", runCalibration);
calibrationSort.addEventListener("change", loadCalibration);

async function runLandscape() {
  setBusy(true, "正在计算 evidence landscape 和智能体先验");
  try {
    const data = await api("/api/abstract-corpus/landscape", { method: "POST" });
    corpusStatus.textContent = `完成：${data.summary?.pair_count || 0} 个材料-性能证据单元`;
    await refreshStatus();
  } catch (error) {
    corpusStatus.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

async function runFigures() {
  setBusy(true, "正在生成 NC 级别图件");
  try {
    await api("/api/abstract-corpus/landscape", { method: "POST" });
    const data = await api("/api/abstract-corpus/figures", { method: "POST" });
    renderFigures(data.figures || []);
    corpusStatus.textContent = `已生成 ${data.figures?.length || 0} 组图件`;
  } catch (error) {
    corpusStatus.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

landscapeButton.addEventListener("click", runLandscape);
landscapeTopButton.addEventListener("click", runLandscape);
figureButton.addEventListener("click", runFigures);
figureTopButton.addEventListener("click", runFigures);

saveEmbeddingButton.addEventListener("click", async () => {
  setBusy(true, "正在保存 Embedding 设置");
  try {
    const data = await api("/api/abstract-corpus/embedding-config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(embeddingPayload()),
    });
    fillEmbeddingForm(data.config || {});
    embeddingState.textContent = "已保存";
    corpusStatus.textContent = "Embedding 设置已保存";
  } catch (error) {
    embeddingState.textContent = "保存失败";
    corpusStatus.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

testEmbeddingButton.addEventListener("click", async () => {
  setBusy(true, "正在测试 Embedding 连接");
  try {
    const data = await api("/api/abstract-corpus/embedding-test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(embeddingPayload()),
    });
    embeddingState.textContent = `${data.result.dimension} 维`;
    corpusStatus.textContent = `Embedding 测试成功：${data.result.vector_count} 条，${data.result.dimension} 维`;
  } catch (error) {
    embeddingState.textContent = "测试失败";
    corpusStatus.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

loraButton.addEventListener("click", async () => {
  setBusy(true, "正在生成 LoRA 微调数据与训练脚本");
  try {
    const data = await api("/api/abstract-corpus/lora/dataset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        max_examples: Number(loraMaxExamples.value || 5000),
        base_model: loraBaseModel.value.trim(),
        embedding_model: loraEmbeddingModel.value.trim(),
      }),
    });
    renderLora(data);
    corpusStatus.textContent = `LoRA 数据已生成：训练 ${data.manifest?.train_examples || 0}，验证 ${data.manifest?.validation_examples || 0}`;
  } catch (error) {
    loraState.textContent = "生成失败";
    corpusStatus.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

extractStandardsButton.addEventListener("click", async () => {
  setBusy(true, "正在抽取规范强度/流动性约束");
  try {
    const data = await api("/api/standards/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_dir: standardSourceDir.value.trim(),
        mode: standardMode.value,
        all_pdfs: standardAllPdfs.checked,
      }),
    });
    renderStandardSummary(data.summary || {});
    await loadStandards();
    corpusStatus.textContent = `规范抽取完成：${data.summary?.processed_pdf_count || 0} 个 PDF，${data.summary?.optimizable_count || 0} 条可优化约束`;
  } catch (error) {
    standardState.textContent = "失败";
    corpusStatus.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

standardDomainFilter.addEventListener("change", loadStandards);
standardCompileFilter.addEventListener("change", loadStandards);

refreshStatus().then(async () => {
  await loadEmbeddingConfig();
  await loadLoraStatus();
  await ensureStandardsReady();
  await loadStandards();
  await loadCalibration();
  await loadFigures();
});
