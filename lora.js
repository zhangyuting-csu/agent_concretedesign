const pageStatus = document.getElementById("pageStatus");
const loraStage = document.getElementById("loraStage");
const datasetState = document.getElementById("datasetState");
const completionBadge = document.getElementById("completionBadge");
const completionDetails = document.getElementById("completionDetails");
const loraKpis = document.getElementById("loraKpis");
const figureList = document.getElementById("figureList");
const figureCount = document.getElementById("figureCount");
const artifactLinks = document.getElementById("artifactLinks");
const trainCommand = document.getElementById("trainCommand");

const baseModel = document.getElementById("baseModel");
const embeddingModel = document.getElementById("embeddingModel");
const outputDir = document.getElementById("outputDir");
const maxExamples = document.getElementById("maxExamples");
const loraR = document.getElementById("loraR");
const loraAlpha = document.getElementById("loraAlpha");
const loraDropout = document.getElementById("loraDropout");
const epochs = document.getElementById("epochs");
const learningRate = document.getElementById("learningRate");
const maxSeqLength = document.getElementById("maxSeqLength");
const saveConfigButton = document.getElementById("saveConfigButton");
const datasetButton = document.getElementById("datasetButton");
const figureButton = document.getElementById("figureButton");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
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
  [saveConfigButton, datasetButton, figureButton].forEach((button) => {
    button.disabled = isBusy;
  });
  if (label) pageStatus.textContent = label;
}

function configPayload() {
  return {
    base_model: baseModel.value.trim(),
    embedding_model: embeddingModel.value.trim(),
    output_dir: outputDir.value.trim(),
    max_examples: Number(maxExamples.value || 5000),
    lora_r: Number(loraR.value || 16),
    lora_alpha: Number(loraAlpha.value || 32),
    lora_dropout: Number(loraDropout.value || 0.05),
    epochs: Number(epochs.value || 2),
    learning_rate: Number(learningRate.value || 0.00015),
    max_seq_length: Number(maxSeqLength.value || 2048),
  };
}

function fillConfig(config = {}) {
  if (config.base_model) baseModel.value = config.base_model;
  if (config.embedding_model) embeddingModel.value = config.embedding_model;
  if (config.output_dir) outputDir.value = config.output_dir;
  if (config.max_examples) maxExamples.value = config.max_examples;
  if (config.lora_r) loraR.value = config.lora_r;
  if (config.lora_alpha) loraAlpha.value = config.lora_alpha;
  if (config.lora_dropout !== undefined) loraDropout.value = config.lora_dropout;
  if (config.epochs) epochs.value = config.epochs;
  if (config.learning_rate) learningRate.value = config.learning_rate;
  if (config.max_seq_length) maxSeqLength.value = config.max_seq_length;
}

function figureUrl(filePath) {
  const name = String(filePath || "").split(/[\\/]/).pop();
  return `/api/abstract-corpus/lora/figures/${encodeURIComponent(name)}`;
}

function renderFigures(figures = []) {
  figureCount.textContent = figures.length;
  if (!figures.length) {
    figureList.innerHTML = '<div class="empty-state">尚未生成 LoRA 图件</div>';
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

function renderArtifacts(files = {}) {
  const labels = {
    train: "训练集",
    validation: "验证集",
    manifest: "Manifest",
    script: "训练脚本",
  };
  artifactLinks.innerHTML = Object.entries(labels)
    .map(([key, label]) => files[key]?.url ? `<a href="${escapeHtml(files[key].url)}" target="_blank">${escapeHtml(label)}</a>` : "")
    .join("") || '<span class="empty-state small-empty">暂无可导出文件</span>';
}

function renderStatus(data) {
  const manifest = data.manifest || {};
  const config = data.config || {};
  const training = data.training || {};
  fillConfig(config);
  renderArtifacts(data.files || {});
  renderFigures(data.figures || []);

  loraStage.textContent = training.stage_label || "未检测";
  pageStatus.textContent = training.stage_label || "LoRA 状态未知";
  datasetState.textContent = manifest.train_examples ? `${manifest.train_examples}/${manifest.validation_examples}` : "未生成";
  completionBadge.textContent = training.completed ? "已完成" : "未完成";
  completionBadge.className = training.completed ? "trained" : "not-trained";

  const adapterText = training.adapter_path ? training.adapter_path : "未检测到 adapter_model.safetensors 或 adapter_model.bin";
  const command = `cd /d path/to/authorized-lora-workspace\npython train_qwen3_lora.py`;
  trainCommand.textContent = command;

  loraKpis.innerHTML = [
    ["训练样本", manifest.train_examples || 0, "JSONL supervised examples"],
    ["验证样本", manifest.validation_examples || 0, "held-out validation"],
    ["训练日志", training.log_points || 0, training.trainer_state_exists ? "trainer_state.json" : "未检测"],
    ["Adapter", training.completed ? "存在" : "不存在", training.stage || "not_ready"],
  ].map(([label, value, note]) => `
    <article>
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(note)}</small>
    </article>
  `).join("");

  completionDetails.innerHTML = [
    ["数据集", manifest.train_examples ? "已生成" : "未生成", "这只是微调前准备，不代表已训练。"],
    ["训练脚本", data.files?.script?.exists ? "已生成" : "未生成", "脚本需要在 GPU 环境中执行。"],
    ["Adapter 权重", training.completed ? "已检测" : "未检测", adapterText],
    ["Checkpoint", `${training.checkpoint_count || 0} 个`, "有 checkpoint 但无 adapter 时，通常表示训练未完整保存或未完成。"],
  ].map(([label, value, note]) => `
    <article>
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(note)}</small>
    </article>
  `).join("");
}

async function refresh() {
  const data = await api("/api/abstract-corpus/lora");
  renderStatus(data);
  return data;
}

saveConfigButton.addEventListener("click", async () => {
  setBusy(true, "正在保存 LoRA 设置");
  try {
    const data = await api("/api/abstract-corpus/lora/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(configPayload()),
    });
    renderStatus(data);
    pageStatus.textContent = "LoRA 设置已保存";
  } catch (error) {
    pageStatus.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

datasetButton.addEventListener("click", async () => {
  setBusy(true, "正在生成 LoRA 数据集和训练脚本");
  try {
    const data = await api("/api/abstract-corpus/lora/dataset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(configPayload()),
    });
    renderStatus(data);
    pageStatus.textContent = `LoRA 数据集已生成：训练 ${data.manifest?.train_examples || 0}，验证 ${data.manifest?.validation_examples || 0}`;
  } catch (error) {
    pageStatus.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

figureButton.addEventListener("click", async () => {
  setBusy(true, "正在生成 LoRA 图件");
  try {
    const data = await api("/api/abstract-corpus/lora/figures", { method: "POST" });
    renderStatus(data.lora_status || data);
    renderFigures(data.figures || []);
    pageStatus.textContent = `已生成 ${data.figures?.length || 0} 组 LoRA 图件`;
  } catch (error) {
    pageStatus.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

refresh().catch((error) => {
  pageStatus.textContent = error.message;
});
