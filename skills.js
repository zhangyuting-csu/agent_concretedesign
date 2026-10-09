const skillNav = document.getElementById("skillNav");
const skillName = document.getElementById("skillName");
const skillVersion = document.getElementById("skillVersion");
const skillDescription = document.getElementById("skillDescription");
const skillPrompt = document.getElementById("skillPrompt");
const skillArchitectureVersion = document.getElementById("skillArchitectureVersion");
const skillArchitectureTitle = document.getElementById("skillArchitectureTitle");
const skillArchitectureFlow = document.getElementById("skillArchitectureFlow");
const skillContract = document.getElementById("skillContract");
const skillRetrieval = document.getElementById("skillRetrieval");
const skillToolchain = document.getElementById("skillToolchain");
const skillValidators = document.getElementById("skillValidators");
const skillReflection = document.getElementById("skillReflection");
const skillBenchmarks = document.getElementById("skillBenchmarks");
const skillContractView = document.getElementById("skillContractView");
const skillRetrievalView = document.getElementById("skillRetrievalView");
const skillToolchainView = document.getElementById("skillToolchainView");
const skillValidatorsView = document.getElementById("skillValidatorsView");
const skillReflectionView = document.getElementById("skillReflectionView");
const skillBenchmarksView = document.getElementById("skillBenchmarksView");
const saveSkillButton = document.getElementById("saveSkillButton");
const skillSaveStatus = document.getElementById("skillSaveStatus");
const skillAssetFile = document.getElementById("skillAssetFile");
const skillAssetFolder = document.getElementById("skillAssetFolder");
const skillAssetList = document.getElementById("skillAssetList");
const bulkUploadStatus = document.getElementById("bulkUploadStatus");
const ragFlow = document.getElementById("ragFlow");
const ragFlowStatus = document.getElementById("ragFlowStatus");
const assetPreview = document.getElementById("assetPreview");
const documentJobs = document.getElementById("documentJobs");
const skillEvalScore = document.getElementById("skillEvalScore");
const skillEvalPanel = document.getElementById("skillEvalPanel");
const kgAuditCount = document.getElementById("kgAuditCount");
const kgAuditPanel = document.getElementById("kgAuditPanel");
const evidenceTotal = document.getElementById("evidenceTotal");
const evidenceCounts = document.getElementById("evidenceCounts");
const rebuildEvidenceButton = document.getElementById("rebuildEvidenceButton");
const deleteAllAssetsButton = document.getElementById("deleteAllAssetsButton");

let currentSkillId = null;
let skillCache = [];
let currentLiteratureAsset = null;
const skillLanguage = () => window.PlatformI18n?.getLanguage?.() || localStorage.getItem("concrete_platform_language") || "zh";
const localizedSkillName = (skill) => skillLanguage() === "en" ? (skill.name_en || skill.name) : skill.name;
const localizedSkillDescription = (skill) => skillLanguage() === "en" ? (skill.description_en || skill.description) : skill.description;
const hasChineseText = (value) => /[\u4e00-\u9fff]/.test(String(value || ""));
const localizedAssetName = (asset) => {
  if (skillLanguage() !== "en" || !hasChineseText(asset.name)) return asset.name;
  const suffix = asset.suffix ? ` ${String(asset.suffix).replace(".", "").toUpperCase()}` : "";
  return `Standards source${suffix} asset`;
};
const EN_SKILL_PROTOCOL = {
  requirement_analysis: {
    retrieval_policy: {
      query_focus: ["Terminology definitions", "Performance thresholds", "Material semantics", "Engineering type"],
      evidence_role: "Map natural language to computable specifications and identify whether specific standards are triggered.",
      primary_sources: ["standard", "literature"],
    },
    toolchain: ["Semantic parser", "Material-intent classifier", "Evidence retriever", "JSON schema validator"],
    validators: [
      "Target strength and curing age must not be missing",
      "allowed / required / avoided material sets must not conflict",
      "User-provided numerical thresholds must be preserved as hard constraints",
      "Every key parsing item must be traceable to the original request",
    ],
    reflection_loop: [
      "Check whether explicitly prohibited materials were missed",
      "Check whether allowed materials were incorrectly treated as required",
      "Check whether construction performance was reduced to an abstract preference",
      "Mark unresolved items when evidence is insufficient",
    ],
    benchmark_cases: [
      { case: "Allow fly ash and silica fume", expect: "allowed_materials contains both; other binders are not automatically prohibited" },
      { case: "I want to use fly ash and silica fume", expect: "required_materials contains both" },
      { case: "Control flowability within 550-650 mm", expect: "Create a hard interval constraint for flowability" },
    ],
  },
  optimization_modeling: {
    retrieval_policy: {
      primary_sources: ["standard", "literature", "kg"],
      query_focus: ["Empirical formulae", "Variable bounds", "Objective functions", "Standards thresholds", "Minimum meaningful dosage"],
      evidence_role: "Convert the requirement into a computable optimization problem and audit physical and engineering boundaries.",
    },
    toolchain: ["Bolomey prior", "Absolute-volume method", "Multi-objective optimizer", "Calculator", "Constraint auditor"],
    validators: [
      "Strength must satisfy the target lower bound",
      "Water-to-binder ratio must be constrained by the Bolomey prior window",
      "Prohibited materials must have zero lower and upper bounds",
      "Flowability intervals must enter constraints, not only narrative text",
      "Output dosages must not fall below meaningful engineering minima",
    ],
    reflection_loop: [
      "Check whether objective functions match user priorities",
      "Check whether a Pareto extreme was incorrectly selected",
      "Check whether empirical checks conflict with ML prediction",
      "Trigger second-pass optimization when necessary",
    ],
    benchmark_cases: [
      { case: "80 MPa low-carbon concrete", expect: "Constrain the water-to-binder ratio by a Bolomey window before optimization" },
      { case: "No silica fume", expect: "Set both lower and upper bounds of silicafume to 0" },
      { case: "Flowability greater than 500 mm", expect: "Create a flowability >= 500 constraint" },
    ],
  },
  mechanism_retrieval: {
    retrieval_policy: {
      primary_sources: ["kg", "literature"],
      query_focus: ["Material-feature-mechanism-performance routes", "Risk mechanisms", "Alternative materials"],
      evidence_role: "Use multi-hop mechanisms to explain the current design and decide whether design feedback is needed.",
    },
    toolchain: ["Multi-hop graph retrieval", "Hybrid retrieval", "Causal-chain scoring", "Second-pass optimization trigger"],
    validators: [
      "Every recommendation must distinguish support, risk, substitution, or validation",
      "KG suggestions must not be presented as model-verified results",
      "Mechanistic routes must be complete and readable",
      "If a recommendation changes a variable, state whether second-pass optimization is required",
    ],
    reflection_loop: [
      "Check whether only positive evidence was retained",
      "Check whether counter-risks were omitted",
      "Check whether literature and KG evidence support each other",
      "Lower confidence when route evidence is weak",
    ],
    benchmark_cases: [
      { case: "High flowability + high silica fume", expect: "Flag PCE compatibility and viscosity risks" },
      { case: "Low carbon + high fly ash", expect: "Report both flowability benefits and early-strength risks" },
      { case: "Pumped concrete", expect: "Retrieve pumping-related standards and cohesion mechanisms" },
    ],
  },
  report_generation: {
    retrieval_policy: {
      primary_sources: ["standard", "literature", "kg"],
      query_focus: ["Conclusion support", "Standards evaluation", "Experimental recommendations", "Risk statement", "Evidence cross-validation"],
      evidence_role: "Integrate computation, standards, mechanisms, and literature into a report suitable for engineering or scientific review.",
    },
    toolchain: ["Report planner", "Citation orchestrator", "Table generator", "Formula renderer", "Completeness auditor"],
    validators: [
      "The body must contain inline citations",
      "Literature, standards, and knowledge-graph citation roles must be distinguished",
      "Required sections must be complete",
      "All conclusions must distinguish prediction, standard, mechanism, and experimental validation",
    ],
    reflection_loop: [
      "Check citation coverage",
      "Check whether figures, experimental matrix, or risk controls are missing",
      "Check for unsupported conclusions",
      "Check whether the report can be read independently",
    ],
    benchmark_cases: [
      { case: "Engineering application report", expect: "Include design basis, recommendation, checks, testing, and risk" },
      { case: "Research guidance report", expect: "Include mechanism interpretation, experimental matrix, failure criteria, and next-round optimization" },
      { case: "Evidence includes standards and literature", expect: "Explain the roles of standards, literature, and KG evidence separately" },
    ],
  },
};

function displaySkillForLanguage(skill) {
  if (skillLanguage() !== "en") return skill;
  return { ...skill, ...(EN_SKILL_PROTOCOL[skill.id] || {}) };
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function loadSkills() {
  const response = await fetch(`/api/skills?language=${encodeURIComponent(skillLanguage())}`);
  const payload = await response.json();
  skillCache = payload.skills || [];
  if (!currentSkillId && skillCache.length) currentSkillId = skillCache[0].id;
  renderSkillNav();
  if (currentSkillId) renderSkillEditor(skillCache.find((item) => item.id === currentSkillId));
}

function renderSkillNav() {
  skillNav.innerHTML = skillCache.map((skill) => `
    <button type="button" class="${skill.id === currentSkillId ? "active" : ""}" data-skill-open="${escapeHtml(skill.id)}">
      <strong>${escapeHtml(localizedSkillName(skill))}</strong>
      <small>${escapeHtml(skill.agent)} · ${escapeHtml(skill.architecture_version || "v2")}</small>
    </button>
  `).join("");
}

function renderSkillEditor(skill) {
  if (!skill) return;
  const displaySkill = displaySkillForLanguage(skill);
  currentSkillId = skill.id;
  skillName.value = localizedSkillName(skill) || "";
  skillVersion.value = skill.version || "";
  skillDescription.value = localizedSkillDescription(skill) || "";
  skillPrompt.value = displaySkill.prompt || "";
  skillArchitectureVersion.textContent = skill.architecture_version || "v2";
  skillArchitectureTitle.textContent = skillLanguage() === "en" ? `${localizedSkillName(skill) || "Agent"} · Protocol-driven` : `${skill.name || "智能体"} · 协议驱动`;
  skillContract.value = JSON.stringify(displaySkill.io_contract || {}, null, 2);
  skillRetrieval.value = JSON.stringify(displaySkill.retrieval_policy || {}, null, 2);
  skillToolchain.value = JSON.stringify(displaySkill.toolchain || [], null, 2);
  skillValidators.value = JSON.stringify(displaySkill.validators || [], null, 2);
  skillReflection.value = JSON.stringify(displaySkill.reflection_loop || [], null, 2);
  skillBenchmarks.value = JSON.stringify(displaySkill.benchmark_cases || [], null, 2);
  renderSkillProtocol(displaySkill);
  skillAssetList.innerHTML = (skill.assets || []).map((asset) => `
    <article data-asset-name="${escapeHtml(asset.name)}">
      <button type="button" data-asset-preview="${escapeHtml(asset.name)}">
        <strong>${escapeHtml(localizedAssetName(asset))}</strong>
        <small>${escapeHtml(asset.suffix || "")} · ${escapeHtml(asset.size)} bytes</small>
      </button>
      <div>
        <a href="/api/skills/${escapeHtml(currentSkillId)}/assets/${encodeURIComponent(asset.name)}/file" target="_blank">查看</a>
        <button type="button" data-asset-reprocess="${escapeHtml(asset.name)}">重解析</button>
        <button type="button" data-asset-delete="${escapeHtml(asset.name)}">删除</button>
      </div>
    </article>
  `).join("") || '<p class="empty-copy">暂无知识资产。</p>';
  renderSkillNav();
}

document.addEventListener("platform-language-change", () => {
  loadSkills();
});

function renderSkillProtocol(skill) {
  const flow = [
    ["契约", "输入/输出结构"],
    ["检索", "证据路由"],
    ["工具", "计算与执行"],
    ["验证", "硬约束审计"],
    ["反思", "自检修订"],
    ["评估", "基准回归"],
  ];
  skillArchitectureFlow.innerHTML = flow.map(([label, detail], index) => `
    <div style="--i:${index}">
      <strong>${label}</strong>
      <span>${detail}</span>
    </div>
  `).join("");
  renderObjectView(skillContractView, skill.io_contract || {});
  renderObjectView(skillRetrievalView, skill.retrieval_policy || {});
  renderListView(skillToolchainView, skill.toolchain || []);
  renderListView(skillValidatorsView, skill.validators || []);
  renderListView(skillReflectionView, skill.reflection_loop || []);
  renderBenchmarkView(skillBenchmarksView, skill.benchmark_cases || []);
}

function renderObjectView(root, value) {
  root.innerHTML = Object.entries(value).map(([key, item]) => `
    <div>
      <b>${escapeHtml(key)}</b>
      <span>${escapeHtml(Array.isArray(item) ? item.join(" / ") : item)}</span>
    </div>
  `).join("");
}

function renderListView(root, items) {
  root.innerHTML = items.map((item) => `<span>${escapeHtml(item)}</span>`).join("");
}

function renderBenchmarkView(root, items) {
  root.innerHTML = items.map((item) => `
    <div>
      <b>${escapeHtml(item.case || "")}</b>
      <span>${escapeHtml(item.expect || "")}</span>
    </div>
  `).join("");
}

async function saveSkill() {
  if (!currentSkillId) return;
  skillSaveStatus.textContent = "保存中...";
  const language = skillLanguage();
  const currentSkill = skillCache.find((item) => item.id === currentSkillId) || {};
  const localizedFields = language === "en"
    ? { name: currentSkill.name, description: currentSkill.description, name_en: skillName.value, description_en: skillDescription.value }
    : { name: skillName.value, description: skillDescription.value, name_en: currentSkill.name_en, description_en: currentSkill.description_en };
  const response = await fetch(`/api/skills/${currentSkillId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...localizedFields,
      language,
      version: skillVersion.value,
      prompt: skillPrompt.value,
      architecture_version: skillArchitectureVersion.textContent || "v2",
      io_contract: parseJsonField(skillContract, {}),
      retrieval_policy: parseJsonField(skillRetrieval, {}),
      toolchain: parseJsonField(skillToolchain, []),
      validators: parseJsonField(skillValidators, []),
      reflection_loop: parseJsonField(skillReflection, []),
      benchmark_cases: parseJsonField(skillBenchmarks, []),
    }),
  });
  const payload = await response.json();
  if (!response.ok || payload.status !== "success") throw new Error(payload.message || "保存失败");
  const index = skillCache.findIndex((item) => item.id === currentSkillId);
  if (index >= 0) skillCache[index] = payload.skill;
  renderSkillEditor(payload.skill);
  skillSaveStatus.textContent = "已保存";
}

function parseJsonField(input, fallback) {
  try {
    return JSON.parse(input.value || JSON.stringify(fallback));
  } catch (error) {
    throw new Error(`${input.previousElementSibling?.textContent || "JSON"} 格式不正确`);
  }
}

async function uploadSkillAsset() {
  if (!currentSkillId || !skillAssetFile.files?.[0]) return;
  const form = new FormData();
  form.append("file", skillAssetFile.files[0]);
  skillSaveStatus.textContent = "上传中...";
  const response = await fetch(`/api/skills/${currentSkillId}/assets/upload`, { method: "POST", body: form });
  const payload = await response.json();
  if (!response.ok || payload.status !== "success") throw new Error(payload.message || "上传失败");
  const skill = await fetch(`/api/skills/${currentSkillId}?language=${encodeURIComponent(skillLanguage())}`).then((res) => res.json()).then((data) => data.skill);
  const index = skillCache.findIndex((item) => item.id === currentSkillId);
  if (index >= 0) skillCache[index] = skill;
  renderSkillEditor(skill);
  skillAssetFile.value = "";
  const articleSuffix = payload.processing?.literature_record_count ? `，识别 ${payload.processing.literature_record_count} 篇文献` : "";
  skillSaveStatus.textContent = payload.duplicate ? `文件已存在，已刷新索引${articleSuffix}` : `上传完成${articleSuffix}`;
  playRagFlow(payload.processing, payload.duplicate ? "已命中重复文件" : "单文件入库完成");
  await previewAsset(payload.asset.name);
  await loadEvidenceStatus();
  await loadKgCandidates();
}

async function uploadSkillFolder() {
  if (!currentSkillId || !skillAssetFolder.files?.length) return;
  const pdfFiles = [...skillAssetFolder.files].filter((file) => file.name.toLowerCase().endsWith(".pdf"));
  if (!pdfFiles.length) {
    bulkUploadStatus.textContent = "该文件夹内未找到 PDF。";
    return;
  }
  const form = new FormData();
  pdfFiles.forEach((file) => form.append("files", file, file.webkitRelativePath || file.name));
  bulkUploadStatus.textContent = `正在递归导入 ${pdfFiles.length} 个 PDF...`;
  const response = await fetch(`/api/skills/${currentSkillId}/assets/bulk-upload`, { method: "POST", body: form });
  const payload = await response.json();
  if (!response.ok || payload.status !== "success") throw new Error(payload.message || "批量上传失败");
  const skill = await fetch(`/api/skills/${currentSkillId}?language=${encodeURIComponent(skillLanguage())}`).then((res) => res.json()).then((data) => data.skill);
  const index = skillCache.findIndex((item) => item.id === currentSkillId);
  if (index >= 0) skillCache[index] = skill;
  renderSkillEditor(skill);
  skillAssetFolder.value = "";
  bulkUploadStatus.textContent = `已导入 ${payload.imported_count} 个 PDF，跳过 ${payload.skipped_count} 个文件。`;
  playRagFlow(payload.assets?.[0]?.processing, `批量导入完成：${payload.imported_count} 个新 PDF`);
  await loadEvidenceStatus();
  await loadKgCandidates();
}

function playRagFlow(processing, headline = "处理中") {
  if (!processing) return;
  ragFlowStatus.textContent = headline;
  ragFlow.innerHTML = (processing.stages || []).map((stage, index) => `
    <div class="rag-step" style="--i:${index}">
      <span>${String(index + 1).padStart(2, "0")}</span>
      <div><strong>${escapeHtml(stage.label)}</strong><small>${escapeHtml(stage.detail)}</small></div>
    </div>
  `).join("");
}

async function previewAsset(name, offset = 0) {
  const payload = await fetch(`/api/skills/${currentSkillId}/assets/${encodeURIComponent(name)}/preview?offset=${offset}&limit=4000`).then((res) => res.json());
  if (payload.status !== "success") throw new Error(payload.message || "预览失败");
  playRagFlow(payload.processing, "资产处理链回放");
  assetPreview.innerHTML = `
    <header>
      <strong>${escapeHtml(payload.asset.name)}</strong>
      <span>${escapeHtml(payload.processing.source_type)} · ${escapeHtml(payload.processing.char_count)} 字符 · ${escapeHtml(payload.processing.chunk_count)} 块 · ${escapeHtml(payload.processing.table_count || 0)} 张表${payload.processing.literature_record_count ? ` · ${escapeHtml(payload.processing.literature_record_count)} 篇文献` : ""}</span>
    </header>
    <div class="embedded-file">
      ${payload.asset.name.toLowerCase().endsWith(".pdf") ? `<iframe src="/api/skills/${escapeHtml(currentSkillId)}/assets/${encodeURIComponent(payload.asset.name)}/file"></iframe>` : ""}
    </div>
    <pre>${escapeHtml(payload.window?.text || payload.processing.excerpt || "未提取到可预览文本。")}</pre>
    <div class="preview-actions">
      ${payload.window?.offset > 0 ? `<button type="button" data-preview-page="${escapeHtml(payload.asset.name)}" data-offset="${Math.max(0, payload.window.offset - payload.window.limit)}">上一段</button>` : ""}
      ${payload.window?.has_more ? `<button type="button" data-preview-page="${escapeHtml(payload.asset.name)}" data-offset="${payload.window.offset + payload.window.limit}">下一段</button>` : ""}
    </div>
    ${renderExtractedTables(payload.processing.tables || [])}
    ${payload.processing.literature_record_count ? renderLiteratureTableShell(payload.asset.name, payload.processing.literature_record_count) : ""}
    ${renderNormativeFacts(payload.processing.normative_requirements || [])}
    ${renderDerivedTriples(payload.processing.derived_triples || [])}
  `;
  if (payload.processing.literature_record_count) {
    currentLiteratureAsset = payload.asset.name;
    await loadLiteratureRecords(payload.asset.name, 1, "");
  } else {
    currentLiteratureAsset = null;
  }
}

function renderExtractedTables(tables = []) {
  if (!tables.length) return "";
  return `
    <section class="preview-evidence-block">
      <h3>表格抽取</h3>
      ${tables.slice(0, 2).map((table) => `
        <article>
          <strong>第 ${table.page} 页</strong>
          <table>
            ${table.rows.slice(0, 8).map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}
          </table>
        </article>
      `).join("")}
    </section>
  `;
}

function renderLiteratureTableShell(name, count) {
  return `
    <section class="preview-evidence-block literature-records">
      <div class="literature-toolbar">
        <h3>已解析文献记录</h3>
        <label>
          <span>${escapeHtml(count)} 篇</span>
          <input type="search" data-literature-search="${escapeHtml(name)}" placeholder="检索题名、作者、DOI、摘要" />
        </label>
      </div>
      <div id="literatureTableHost" class="literature-table-host"></div>
      <div id="literaturePager" class="literature-pager"></div>
    </section>
  `;
}

async function loadLiteratureRecords(name, page = 1, query = "") {
  const params = new URLSearchParams({ page, page_size: 10, q: query });
  const payload = await fetch(`/api/skills/${currentSkillId}/assets/${encodeURIComponent(name)}/literature-records?${params}`).then((res) => res.json());
  if (payload.status !== "success") throw new Error(payload.message || "文献表加载失败");
  const host = document.getElementById("literatureTableHost");
  const pager = document.getElementById("literaturePager");
  if (!host || !pager) return;
  host.innerHTML = `
    <table class="literature-table">
      <thead><tr><th>#</th><th>题名</th><th>作者</th><th>期刊</th><th>年份</th><th>DOI</th><th>摘要</th></tr></thead>
      <tbody>
        ${payload.records.map((item, index) => `
          <tr>
            <td>${(payload.page - 1) * payload.page_size + index + 1}</td>
            <td>${escapeHtml(item.title || "-")}</td>
            <td>${escapeHtml((item.authors || []).join("; ") || "-")}</td>
            <td>${escapeHtml(item.journal || "-")}</td>
            <td>${escapeHtml(item.year || "-")}</td>
            <td>${escapeHtml(item.doi || "-")}</td>
            <td>
              ${item.abstract ? `
                <details>
                  <summary>${escapeHtml(String(item.abstract).slice(0, 180))}${String(item.abstract).length > 180 ? "..." : ""}</summary>
                  <p>${escapeHtml(item.abstract)}</p>
                </details>
              ` : "-"}
            </td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
  pager.dataset.query = payload.query || "";
  pager.innerHTML = `
    <span>共 ${payload.total} 条，第 ${payload.page} / ${payload.total_pages} 页</span>
    <div>
      <button type="button" data-literature-page="${Math.max(1, payload.page - 1)}" ${payload.page <= 1 ? "disabled" : ""}>上一页</button>
      <button type="button" data-literature-page="${Math.min(payload.total_pages, payload.page + 1)}" ${payload.page >= payload.total_pages ? "disabled" : ""}>下一页</button>
    </div>
  `;
}

function renderNormativeFacts(items = []) {
  if (!items.length) return "";
  return `
    <section class="preview-evidence-block">
      <h3>抽取到的规范指标</h3>
      ${items.slice(0, 8).map((item) => `<div><strong>${escapeHtml(item.metric)}</strong><span>${escapeHtml(item.value)} ${escapeHtml(item.unit)}</span></div>`).join("")}
    </section>
  `;
}

function renderDerivedTriples(items = []) {
  if (!items.length) return "";
  return `
    <section class="preview-evidence-block">
      <h3>自动生成的知识图谱三元组</h3>
      ${items.slice(0, 8).map((item) => `<div><strong>${escapeHtml(item.subject)}</strong><span>${escapeHtml(item.predicate)} → ${escapeHtml(item.object)}</span></div>`).join("")}
    </section>
  `;
}

async function deleteAsset(name) {
  const response = await fetch(`/api/skills/${currentSkillId}/assets/${encodeURIComponent(name)}`, { method: "DELETE" });
  const payload = await response.json();
  if (!response.ok || payload.status !== "success") throw new Error(payload.message || "删除失败");
  const skill = await fetch(`/api/skills/${currentSkillId}?language=${encodeURIComponent(skillLanguage())}`).then((res) => res.json()).then((data) => data.skill);
  const index = skillCache.findIndex((item) => item.id === currentSkillId);
  if (index >= 0) skillCache[index] = skill;
  renderSkillEditor(skill);
  assetPreview.innerHTML = "<p>资产已删除，索引已同步更新。</p>";
  ragFlowStatus.textContent = "索引已更新";
  await loadEvidenceStatus();
  await loadKgCandidates();
}

async function deleteAllAssets() {
  if (!currentSkillId) return;
  const skill = skillCache.find((item) => item.id === currentSkillId);
  const count = skill?.assets?.length || 0;
  if (!count) {
    bulkUploadStatus.textContent = "当前 Skill 没有可删除资产。";
    return;
  }
  if (!window.confirm(`确认删除当前 Skill 的 ${count} 个知识资产？该操作会同步移除对应证据索引和自动 KG 候选。`)) return;
  bulkUploadStatus.textContent = "正在删除全部资产...";
  const response = await fetch(`/api/skills/${currentSkillId}/assets`, { method: "DELETE" });
  const payload = await response.json();
  if (!response.ok || payload.status !== "success") throw new Error(payload.message || "全部删除失败");
  const updated = await fetch(`/api/skills/${currentSkillId}?language=${encodeURIComponent(skillLanguage())}`).then((res) => res.json()).then((data) => data.skill);
  const index = skillCache.findIndex((item) => item.id === currentSkillId);
  if (index >= 0) skillCache[index] = updated;
  renderSkillEditor(updated);
  assetPreview.innerHTML = "<p>当前 Skill 的知识资产已全部删除，证据索引和自动 KG 候选已同步更新。</p>";
  bulkUploadStatus.textContent = `已删除 ${payload.deleted_count || 0} 个资产。`;
  await loadEvidenceStatus();
  await loadKgCandidates();
}

async function reprocessAsset(name) {
  const response = await fetch(`/api/skills/${currentSkillId}/assets/${encodeURIComponent(name)}/reprocess`, { method: "POST" });
  const payload = await response.json();
  if (!response.ok || payload.status !== "success") throw new Error(payload.message || "重解析失败");
  bulkUploadStatus.textContent = "已加入后台解析队列。";
  await loadDocumentJobs();
  await loadKgCandidates();
}

async function loadEvidenceStatus() {
  const payload = await fetch("/api/evidence/status").then((res) => res.json());
  evidenceTotal.textContent = `${payload.total_records || 0} 条`;
  evidenceCounts.innerHTML = [
    ["文献摘要", payload.counts?.literature || 0],
    ["规范全文", payload.counts?.standard || 0],
    ["KG 机理", payload.counts?.kg || 0],
    ["自动 KG", payload.counts?.auto_kg || 0],
  ].map(([label, value]) => `<div><strong>${value}</strong><span>${label}</span></div>`).join("");
}

async function loadDocumentJobs() {
  const payload = await fetch("/api/document-jobs").then((res) => res.json());
  documentJobs.innerHTML = (payload.jobs || []).slice(0, 8).map((job) => `
    <article>
      <strong>${escapeHtml(job.asset_name)}</strong>
      <span>${escapeHtml(job.stage)} · ${escapeHtml(job.progress)}%</span>
      <i style="--p:${escapeHtml(job.progress)}%"></i>
    </article>
  `).join("") || "<p>暂无后台任务。</p>";
}

function scoreDialSvg(score, size = 128, label = "Overall") {
  const value = Math.max(0, Math.min(Number(score || 0), 100));
  const radius = 46;
  const circumference = 2 * Math.PI * radius;
  const dash = (value / 100) * circumference;
  return `
    <svg class="eval-dial-svg" width="${size}" height="${size}" viewBox="0 0 128 128" role="img" aria-label="${escapeHtml(label)} ${escapeHtml(value)}分">
      <circle cx="64" cy="64" r="${radius}" class="dial-track"></circle>
      <circle cx="64" cy="64" r="${radius}" class="dial-value" stroke-dasharray="${dash} ${circumference}"></circle>
      <text x="64" y="59" text-anchor="middle" class="dial-score">${escapeHtml(value)}</text>
      <text x="64" y="78" text-anchor="middle" class="dial-label">${escapeHtml(label)}</text>
    </svg>
  `;
}

function evalSparklineSvg(checks = []) {
  const points = checks.map((check, index) => {
    const x = 12 + index * 18;
    const y = check.passed ? 22 : 45;
    return `${x},${y}`;
  }).join(" ");
  return `
    <svg class="eval-spark-svg" width="148" height="58" viewBox="0 0 148 58" role="img" aria-label="检查项状态曲线">
      <line x1="8" y1="45" x2="140" y2="45" class="spark-base"></line>
      <polyline points="${points}" class="spark-line"></polyline>
      ${checks.map((check, index) => `<circle cx="${12 + index * 18}" cy="${check.passed ? 22 : 45}" r="4" class="${check.passed ? "ok" : "bad"}"></circle>`).join("")}
    </svg>
  `;
}

function evalCheckMatrix(checks = []) {
  return `
    <div class="eval-check-matrix">
      ${checks.map((check, index) => `
        <button type="button" class="${check.passed ? "pass" : "fail"}" title="${escapeHtml(check.name)}：${escapeHtml(check.detail)}">
          <span>${String(index + 1).padStart(2, "0")}</span>
          <b>${escapeHtml(check.name)}</b>
        </button>
      `).join("")}
    </div>
  `;
}

async function loadSkillEvaluation() {
  const payload = await fetch("/api/skill-evaluation").then((res) => res.json());
  if (payload.status !== "success") return;
  skillEvalScore.textContent = `${payload.overall_score} / 100`;
  const score = Number(payload.overall_score || 0);
  const results = payload.results || [];
  skillEvalPanel.innerHTML = `
    <section class="eval-dashboard">
      <div class="eval-score-hero">
        ${scoreDialSvg(score, 148, "Overall")}
        <div>
          <b>Skill v2 协议健康度</b>
          <p>契约、证据路由、工具链、验证规则、反思闭环和基准样例的自动审计结果。</p>
          <div class="eval-mini-legend"><span></span>通过项 <i></i>待修正项</div>
        </div>
      </div>
      <div class="eval-agent-grid">
        ${results.map((item) => {
          const checks = item.checks || [];
          const passed = checks.filter((check) => check.passed).length;
          return `
            <article class="eval-agent-card">
              <header>
                ${scoreDialSvg(item.score, 86, item.name?.slice(0, 4) || "Skill")}
                <div>
                  <strong>${escapeHtml(item.name)}</strong>
                  <small>${escapeHtml(passed)} / ${escapeHtml(checks.length)} checks passed</small>
                </div>
              </header>
              ${evalSparklineSvg(checks)}
              ${evalCheckMatrix(checks)}
              <details>
                <summary>查看审计明细</summary>
                <div class="eval-detail-list">
                  ${checks.map((check) => `<p class="${check.passed ? "pass" : "fail"}"><b>${check.passed ? "✓" : "!"}</b><span>${escapeHtml(check.name)}</span><em>${escapeHtml(check.detail)}</em></p>`).join("")}
                </div>
              </details>
            </article>
          `;
        }).join("")}
      </div>
    </section>
  `;
}

async function loadKgCandidates() {
  const payload = await fetch("/api/kg-candidates").then((res) => res.json());
  if (payload.status !== "success") return;
  const items = payload.candidates || [];
  kgAuditCount.textContent = `${items.length} 条`;
  kgAuditPanel.innerHTML = items.slice(0, 12).map((item) => `
    <article>
      <header>
        <strong>${escapeHtml(item.material)} → ${escapeHtml(item.performance)}</strong>
        <span>${escapeHtml(item.status)} · ${escapeHtml(item.confidence)} · ${escapeHtml(item.extractor || "rule_fallback")}</span>
      </header>
      <p>${escapeHtml([item.feature, item.mechanism].filter(Boolean).join("；"))}</p>
      <small>${escapeHtml(item.source || "")}</small>
      <div>
        <button type="button" data-kg-status="${escapeHtml(item.id)}" data-status="approved">通过</button>
        <button type="button" data-kg-status="${escapeHtml(item.id)}" data-status="rejected">驳回</button>
      </div>
    </article>
  `).join("") || "<p>暂无候选三元组。上传或重解析文献、规范后会自动生成。</p>";
}

async function updateKgCandidate(id, status) {
  const response = await fetch(`/api/kg-candidates/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  const payload = await response.json();
  if (!response.ok || payload.status !== "success") throw new Error(payload.message || "更新失败");
  await loadKgCandidates();
  await loadEvidenceStatus();
}

skillNav.addEventListener("click", (event) => {
  const button = event.target.closest("[data-skill-open]");
  if (!button) return;
  renderSkillEditor(skillCache.find((item) => item.id === button.dataset.skillOpen));
});

saveSkillButton.addEventListener("click", () => {
  saveSkill().catch((error) => {
    skillSaveStatus.textContent = `保存失败：${error.message}`;
  });
});

skillAssetFile.addEventListener("change", () => {
  uploadSkillAsset().catch((error) => {
    skillSaveStatus.textContent = `上传失败：${error.message}`;
  });
});

skillAssetFolder.addEventListener("change", () => {
  uploadSkillFolder().catch((error) => {
    bulkUploadStatus.textContent = `批量导入失败：${error.message}`;
  });
});

skillAssetList.addEventListener("click", (event) => {
  const preview = event.target.closest("[data-asset-preview]");
  if (preview) {
    previewAsset(preview.dataset.assetPreview).catch((error) => {
      bulkUploadStatus.textContent = `预览失败：${error.message}`;
    });
    return;
  }
  const remove = event.target.closest("[data-asset-delete]");
  if (remove) {
    deleteAsset(remove.dataset.assetDelete).catch((error) => {
      bulkUploadStatus.textContent = `删除失败：${error.message}`;
    });
    return;
  }
  const reprocess = event.target.closest("[data-asset-reprocess]");
  if (reprocess) {
    reprocessAsset(reprocess.dataset.assetReprocess).catch((error) => {
      bulkUploadStatus.textContent = `重解析失败：${error.message}`;
    });
    return;
  }
  const page = event.target.closest("[data-preview-page]");
  if (page) {
    previewAsset(page.dataset.previewPage, Number(page.dataset.offset || 0)).catch((error) => {
      bulkUploadStatus.textContent = `预览失败：${error.message}`;
    });
  }
});

kgAuditPanel.addEventListener("click", (event) => {
  const button = event.target.closest("[data-kg-status]");
  if (!button) return;
  updateKgCandidate(button.dataset.kgStatus, button.dataset.status).catch((error) => {
    bulkUploadStatus.textContent = `三元组审核失败：${error.message}`;
  });
});

assetPreview.addEventListener("click", (event) => {
  const page = event.target.closest("[data-literature-page]");
  if (!page || !currentLiteratureAsset) return;
  const query = document.getElementById("literaturePager")?.dataset.query || "";
  loadLiteratureRecords(currentLiteratureAsset, Number(page.dataset.literaturePage || 1), query).catch((error) => {
    bulkUploadStatus.textContent = `文献表加载失败：${error.message}`;
  });
});

assetPreview.addEventListener("input", (event) => {
  const input = event.target.closest("[data-literature-search]");
  if (!input || !currentLiteratureAsset) return;
  window.clearTimeout(input._literatureTimer);
  input._literatureTimer = window.setTimeout(() => {
    loadLiteratureRecords(currentLiteratureAsset, 1, input.value.trim()).catch((error) => {
      bulkUploadStatus.textContent = `文献检索失败：${error.message}`;
    });
  }, 220);
});

rebuildEvidenceButton.addEventListener("click", async () => {
  skillSaveStatus.textContent = "索引重建中...";
  await fetch("/api/evidence/rebuild", { method: "POST" });
  await loadEvidenceStatus();
  skillSaveStatus.textContent = "索引已重建";
});

deleteAllAssetsButton?.addEventListener("click", () => {
  deleteAllAssets().catch((error) => {
    bulkUploadStatus.textContent = `全部删除失败：${error.message}`;
  });
});

loadSkills().catch((error) => {
  skillSaveStatus.textContent = `加载失败：${error.message}`;
});
loadEvidenceStatus().catch(() => {});
loadDocumentJobs().catch(() => {});
loadSkillEvaluation().catch(() => {});
loadKgCandidates().catch(() => {});
setInterval(() => loadDocumentJobs().catch(() => {}), 2500);
