const CATEGORIES = [
  "高强", "低碳", "低收缩", "自密实", "泵送", "材料指定", "材料禁用", "流动性区间", "耐久环境", "成本约束", "模糊口语"
];

const STRENGTHS = [30, 40, 50, 60, 70, 80, 90, 100];
const AGES = [3, 7, 14, 28, 56, 90];
const MATERIAL_POLICIES = [
  { text: "允许使用粉煤灰、矿渣和硅灰", required: [], allowed: ["flyash", "GGBFS", "silicafume"], avoided: [] },
  { text: "想使用粉煤灰和硅灰，不希望再加其他胶凝材料", required: ["flyash", "silicafume"], allowed: ["flyash", "silicafume"], avoided: ["GGBFS", "metakaolin", "limestone"] },
  { text: "禁止使用硅灰，矿渣可以用", required: [], allowed: ["GGBFS"], avoided: ["silicafume"] },
  { text: "尽量不用粉煤灰，优先矿渣和少量硅灰", required: [], allowed: ["GGBFS", "silicafume"], avoided: ["flyash"] },
  { text: "只允许水泥、矿渣和减水剂体系", required: ["GGBFS"], allowed: ["cement", "GGBFS", "superplasticizer"], avoided: ["flyash", "silicafume", "metakaolin", "limestone"] },
];
const FLOW_REQUESTS = [
  { text: "坍落扩展度大于 500 mm", min: 500, max: null },
  { text: "流动性控制在 550 到 650 mm", min: 550, max: 650 },
  { text: "希望高流动但不能离析", min: 560, max: null },
  { text: "泵送施工，坍落度保持在 180 到 220 mm", min: 180, max: 220 },
  { text: "自密实，T500 不要太慢", min: 650, max: 760 },
];
const DURABILITY = [
  { text: "用于海工氯盐环境", tag: "chloride" },
  { text: "地下工程，要求抗渗和抗硫酸盐侵蚀", tag: "sulfate_impermeability" },
  { text: "寒冷地区，需要考虑冻融循环", tag: "freeze_thaw" },
  { text: "强紫外和碳化环境", tag: "carbonation_uv" },
  { text: "普通室内构件，无特殊耐久性要求", tag: "normal" },
];
const CONSTRUCTION = [
  { text: "预制构件", tag: "precast" },
  { text: "泵送施工", tag: "pumping" },
  { text: "自密实浇筑", tag: "self_compacting" },
  { text: "3D 打印材料", tag: "3d_printing" },
  { text: "现场大体积浇筑", tag: "mass_concrete" },
];
const CARBON_COST = [
  { text: "尽量降低碳排放，成本可以略高", priority: ["low_carbon", "strength"] },
  { text: "预算紧张，优先控制成本，同时满足强度", priority: ["low_cost", "strength"] },
  { text: "低碳和流动性优先，成本作为次要目标", priority: ["low_carbon", "flowability", "low_cost"] },
  { text: "强度必须达标，其次降低水泥用量", priority: ["strength", "low_cement", "low_carbon"] },
  { text: "希望综合平衡强度、碳排放、成本和流动性", priority: ["strength", "low_carbon", "low_cost", "flowability"] },
];
const SHRINKAGE_REQUESTS = [
  { text: "并且要控制干燥收缩和开裂风险", priority: ["low_shrinkage"], shrinkage_max: null, type: "drying" },
  { text: "56 天干燥收缩最好小于 400 微应变", priority: ["low_shrinkage"], shrinkage_max: 400, type: "drying" },
  { text: "低碳同时要低收缩，避免早期开裂", priority: ["low_shrinkage", "low_carbon"], shrinkage_max: null, type: "cracking_risk" },
  { text: "水胶比不能太激进，自收缩要尽量低", priority: ["low_shrinkage"], shrinkage_max: null, type: "autogenous" },
  { text: "收缩不是主要目标", priority: [], shrinkage_max: null, type: "" },
];
const TEMPLATES = [
  "设计一个 {strength} MPa、{age} 天龄期的混凝土，{flow}，{material}，{durability}，{carbon}，{shrinkage}。",
  "我需要 {strength}MPa 的配合比，龄期按 {age} 天考虑，{construction}，{flow}，并且{material}，{carbon}，{shrinkage}。",
  "帮我做一组 {age}d 达到 {strength} MPa 的低碳混凝土，{durability}，{construction}，{material}，{flow}，{shrinkage}。",
  "工程要求 C{strength} 左右，评价龄期 {age} 天，{carbon}，施工条件是{construction}，另外{flow}，{material}，{shrinkage}。",
];

function pick(list, index, offset = 0) {
  return list[(index + offset) % list.length];
}

function inferCategories(item) {
  const cats = new Set();
  if (item.expected.target_mpa >= 70) cats.add("高强");
  if (item.expected.priorities.includes("low_carbon") || item.text.includes("低碳")) cats.add("低碳");
  if (item.expected.priorities.includes("low_shrinkage") || item.text.includes("收缩")) cats.add("低收缩");
    if (item.expected.construction === "self_compacting" || item.text.includes("自密实")) cats.add("自密实");
  if (item.expected.construction === "pumping" || item.text.includes("泵送")) cats.add("泵送");
  if (item.expected.required_materials.length || item.text.includes("只允许") || item.text.includes("想使用")) cats.add("材料指定");
  if (item.expected.avoided_materials.length || item.text.includes("禁止") || item.text.includes("不用")) cats.add("材料禁用");
  if (item.expected.flowability_min || item.expected.flowability_max) cats.add("流动性区间");
  if (!["normal", ""].includes(item.expected.durability)) cats.add("耐久环境");
  if (item.expected.priorities.includes("low_cost")) cats.add("成本约束");
  return [...cats];
}

function generatedCases() {
  const cases = [];
  for (let i = 0; i < 200; i += 1) {
    const strength = pick(STRENGTHS, i);
    const age = pick(AGES, i, 2);
    const material = pick(MATERIAL_POLICIES, i, 1);
    const flow = pick(FLOW_REQUESTS, i, 3);
    const durability = pick(DURABILITY, i, 4);
    const construction = pick(CONSTRUCTION, i, 2);
    const carbon = pick(CARBON_COST, i, 1);
    const shrinkage = pick(SHRINKAGE_REQUESTS, i, 2);
    const template = pick(TEMPLATES, i);
    const text = template
      .replace("{strength}", strength)
      .replace("{age}", age)
      .replace("{material}", material.text)
      .replace("{flow}", flow.text)
      .replace("{durability}", durability.text)
      .replace("{construction}", construction.text)
      .replace("{carbon}", carbon.text)
      .replace("{shrinkage}", shrinkage.text);
    const item = {
      id: `G-${String(i + 1).padStart(3, "0")}`,
      source: "programmatic",
      difficulty: "standard",
      text,
      expected: {
        target_mpa: strength,
        age_days: age,
        required_materials: material.required,
        allowed_materials: material.allowed,
        avoided_materials: material.avoided,
        flowability_min: flow.min,
        flowability_max: flow.max,
        durability: durability.tag,
        construction: construction.tag,
        priorities: [...new Set([...(carbon.priority || []), ...(shrinkage.priority || [])])],
        shrinkage_max: shrinkage.shrinkage_max,
        shrinkage_type: shrinkage.type,
      },
    };
    item.categories = inferCategories(item);
    cases.push(item);
  }
  return cases;
}

const HARD_CASE_TEXTS = [
  ["H-001", "老板说这个项目要做 80 兆帕左右，28 天能交付，最好别太依赖水泥，可以用粉煤灰和矿渣，但硅灰别加，泵送不能太黏。", 80, 28, [], ["flyash", "GGBFS"], ["silicafume"], 180, 230, "pumping", ["strength", "low_cement", "flowability"]],
  ["H-002", "想做自密实高强混凝土，强度别低于 70MPa，扩展度最好 650 到 720，粉煤灰、硅灰都可以，但不要石灰石粉。", 70, 28, [], ["flyash", "silicafume"], ["limestone"], 650, 720, "self_compacting", ["flowability", "strength"]],
  ["H-003", "海边预制构件，56 天强度 90MPa，低氯离子渗透优先，矿渣必须有，粉煤灰可用可不用，硅灰最多少量。", 90, 56, ["GGBFS"], ["flyash", "silicafume"], [], null, null, "precast", ["durability", "strength"]],
  ["H-004", "不要给我那种硅灰 1 kg/m3 的假配方，材料要有工程意义；目标 100MPa，28d，低碳只是第二目标。", 100, 28, [], [], [], null, null, "", ["strength", "low_carbon"]],
  ["H-005", "地下防水混凝土，抗渗和硫酸盐都要考虑，强度 50MPa 就行，尽量便宜，别用钢纤维。", 50, 28, [], [], ["steel_fiber"], null, null, "underground", ["low_cost", "durability"]],
  ["H-006", "我不是要自密实，只是希望泵送舒服一点，C60，28 天，坍落度 200 左右，粉煤灰不要超过普通合理范围。", 60, 28, [], ["flyash"], [], 180, 220, "pumping", ["flowability", "strength"]],
  ["H-007", "C80，7 天要有早强，粉煤灰别太多，硅灰可以用，碳排放也要压低。", 80, 7, [], ["silicafume"], ["high_flyash"], null, null, "", ["early_strength", "low_carbon"]],
  ["H-008", "如果能不用矿渣就不用矿渣，我想比较粉煤灰体系和硅灰体系，目标 60MPa，28 天，流动性大于 500。", 60, 28, [], ["flyash", "silicafume"], ["GGBFS"], 500, null, "", ["flowability", "scheme_comparison"]],
  ["H-009", "3D 打印砂浆，不追求很高流动性，要求可建造性，28 天 40MPa，尽量降低水胶比但别堵管。", 40, 28, [], [], [], null, null, "3d_printing", ["buildability", "strength"]],
  ["H-010", "大体积承台，56 天 50MPa，核心是低热和低碳，早期强度不是第一位，禁止硅灰。", 50, 56, [], [], ["silicafume"], null, null, "mass_concrete", ["low_heat", "low_carbon"]],
  ["H-011", "我想做 60MPa、28 天的低碳低收缩混凝土，粉煤灰和矿渣都可以用，重点是干燥收缩不要太大，后续要做 56 天收缩验证。", 60, 28, [], ["flyash", "GGBFS"], [], null, null, "", ["low_carbon", "low_shrinkage", "strength"], 56, null, "drying"],
  ["H-012", "80MPa 高强低碳，但是别为了强度把硅灰堆太高导致自收缩开裂，要求自收缩风险低，28 天。", 80, 28, [], [], [], null, null, "", ["strength", "low_carbon", "low_shrinkage"], 28, null, "autogenous"],
  ["H-013", "预制板材 50MPa，要求 56d 干燥收缩小于 400 微应变，允许矿渣和石灰石粉，尽量低碳。", 50, 56, [], ["GGBFS", "limestone"], [], null, null, "precast", ["low_shrinkage", "low_carbon", "strength"], 56, 400, "drying"],
];

function hardCases() {
  const extra = [];
  const motifs = ["冻融", "碳化", "氯盐", "硫酸盐", "低热", "早强", "超高强", "高流动", "材料禁用", "多方案比较"];
  for (let i = 0; i < 40; i += 1) {
    const strength = pick([45, 55, 65, 75, 85, 95], i);
    const age = pick([7, 28, 56, 90], i, 1);
    const motif = pick(motifs, i);
    const material = pick(MATERIAL_POLICIES, i, 2);
    const flow = pick(FLOW_REQUESTS, i, 4);
    const text = `表达可能不规范：这个项目大概是 ${strength}MPa 等级，按 ${age} 天验收，重点是${motif}，${material.text}，${flow.text}，如果传统公式和模型预测冲突，请优先给出可施工且能解释的方案。`;
    extra.push({
      id: `H-${String(i + 11).padStart(3, "0")}`,
      source: "manual_hard",
      difficulty: "hard",
      text,
      expected: {
        target_mpa: strength,
        age_days: age,
        required_materials: material.required,
        allowed_materials: material.allowed,
        avoided_materials: material.avoided,
        flowability_min: flow.min,
        flowability_max: flow.max,
        durability: motif,
        construction: "",
        priorities: ["strength", motif.includes("低") ? "low_carbon" : "durability"],
      },
    });
  }
  return [
    ...HARD_CASE_TEXTS.map(([id, text, target, age, required, allowed, avoided, fmin, fmax, construction, priorities, shrinkageAge, shrinkageMax, shrinkageType]) => ({
      id,
      source: "manual_hard",
      difficulty: "hard",
      text,
      expected: {
        target_mpa: target,
        age_days: age,
        required_materials: required,
        allowed_materials: allowed,
        avoided_materials: avoided,
        flowability_min: fmin,
        flowability_max: fmax,
        durability: "",
        construction,
        priorities,
        shrinkage_age_days: shrinkageAge || null,
        shrinkage_max: shrinkageMax || null,
        shrinkage_type: shrinkageType || "",
      },
    })),
    ...extra,
  ].map((item) => ({ ...item, categories: [...new Set([...inferCategories(item), "模糊口语"])] }));
}

const BENCHMARK = [...generatedCases(), ...hardCases()];
let activeCategory = "";
let visibleCases = BENCHMARK;
let runPaused = false;
let runActive = false;
let runStatsCache = {};
const DB_NAME = "concrete_demand_benchmark_v1";
const DB_STORE = "runs";

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function expectedHtml(expected) {
  const rows = [
    ["强度", `${expected.target_mpa} MPa`],
    ["龄期", `${expected.age_days} d`],
    ["必须材料", expected.required_materials.join(", ") || "-"],
    ["允许材料", expected.allowed_materials.join(", ") || "-"],
    ["禁用材料", expected.avoided_materials.join(", ") || "-"],
    ["流动性", [expected.flowability_min ? `>=${expected.flowability_min}` : "", expected.flowability_max ? `<=${expected.flowability_max}` : ""].filter(Boolean).join(" ") || "-"],
    ["施工/环境", [expected.construction, expected.durability].filter(Boolean).join(" / ") || "-"],
    ["优先级", expected.priorities.join(" > ") || "-"],
  ];
  return rows.map(([key, value]) => `<span><b>${escapeHtml(key)}</b>${escapeHtml(value)}</span>`).join("");
}

function renderFilters() {
  const root = document.getElementById("categoryFilters");
  root.innerHTML = CATEGORIES.map((category) => {
    const count = BENCHMARK.filter((item) => item.categories.includes(category)).length;
    return `<button type="button" class="${activeCategory === category ? "active" : ""}" data-category="${escapeHtml(category)}"><span>${escapeHtml(category)}</span><b>${count}</b></button>`;
  }).join("");
}

function applyFilters() {
  const q = document.getElementById("benchmarkSearch").value.trim().toLowerCase();
  const difficulty = document.getElementById("difficultyFilter").value;
  visibleCases = BENCHMARK.filter((item) => {
    if (activeCategory && !item.categories.includes(activeCategory)) return false;
    if (difficulty && item.difficulty !== difficulty) return false;
    if (q && !`${item.text} ${item.categories.join(" ")} ${JSON.stringify(item.expected)}`.toLowerCase().includes(q)) return false;
    return true;
  });
  renderSummary();
  renderList();
}

function renderSummary() {
  document.getElementById("benchmarkTotal").textContent = BENCHMARK.length;
  const hard = visibleCases.filter((item) => item.difficulty === "hard").length;
  const standard = visibleCases.length - hard;
  document.getElementById("benchmarkSummary").innerHTML = [
    ["当前显示", visibleCases.length],
    ["标准样例", standard],
    ["高难样例", hard],
    ["类别覆盖", new Set(visibleCases.flatMap((item) => item.categories)).size],
  ].map(([label, value]) => `<article><strong>${value}</strong><span>${label}</span></article>`).join("");
}

function pct(value, digits = 0) {
  if (!Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(digits)}%`;
}

function mean(values) {
  const nums = values.map(Number).filter(Number.isFinite);
  return nums.length ? nums.reduce((sum, value) => sum + value, 0) / nums.length : null;
}

function metricBar(label, value, color = "#0f766e") {
  const numeric = Math.max(0, Math.min(1, Number(value) || 0));
  return `
    <div class="metric-bar">
      <span>${escapeHtml(label)}</span>
      <i><b style="width:${(numeric * 100).toFixed(1)}%;background:${color}"></b></i>
      <strong>${pct(numeric)}</strong>
    </div>
  `;
}

function svgBars(items, { width = 620, height = 230, color = "#0f766e", max = null } = {}) {
  const pad = { left: 42, right: 18, top: 26, bottom: 58 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;
  const vals = items.map((item) => Number(item.value) || 0);
  const maxV = max ?? Math.max(...vals, 1);
  const band = chartW / Math.max(items.length, 1);
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img">
      <line x1="${pad.left}" y1="${pad.top + chartH}" x2="${width - pad.right}" y2="${pad.top + chartH}" class="viz-axis"></line>
      ${items.map((item, index) => {
        const h = ((Number(item.value) || 0) / Math.max(maxV, 1e-9)) * chartH;
        const x = pad.left + index * band + band * 0.18;
        const y = pad.top + chartH - h;
        return `
          <rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(4, band * 0.64).toFixed(1)}" height="${h.toFixed(1)}" rx="3" fill="${item.color || color}"></rect>
          <text x="${(x + band * 0.32).toFixed(1)}" y="${Math.max(14, y - 5).toFixed(1)}" text-anchor="middle" class="viz-value">${escapeHtml(item.display ?? item.value ?? "-")}</text>
          <text x="${(x + band * 0.32).toFixed(1)}" y="${height - 25}" text-anchor="middle" class="viz-label">${escapeHtml(String(item.label).slice(0, 8))}</text>
        `;
      }).join("")}
    </svg>
  `;
}

function categoryHeatmap(records) {
  const completed = records.filter((record) => record.status === "completed");
  const rows = CATEGORIES.map((category) => {
    const subset = completed.filter((record) => (record.case?.categories || []).includes(category));
    const avg = mean(subset.map((record) => record.parsed_eval?.score));
    const failures = records.filter((record) => (record.case?.categories || []).includes(category) && record.status === "failed").length;
    return { category, count: subset.length, avg, failures };
  }).filter((row) => row.count || row.failures);
  if (!rows.length) return '<p class="empty-copy">暂无已完成记录，运行 benchmark 后生成类别热图。</p>';
  return `
    <div class="benchmark-heatmap">
      ${rows.map((row) => {
        const score = Number(row.avg || 0) / 100;
        const bg = `rgba(${Math.round(190 - 120 * score)}, ${Math.round(120 + 80 * score)}, ${Math.round(110 + 70 * score)}, 0.22)`;
        return `<article style="background:${bg}"><strong>${escapeHtml(row.category)}</strong><b>${row.avg === null ? "-" : Math.round(row.avg)}</b><span>${row.count} completed / ${row.failures} failed</span></article>`;
      }).join("")}
    </div>
  `;
}

function ablationRows(records) {
  const completed = records.filter((record) => record.status === "completed");
  const avgScore = mean(completed.map((record) => record.parsed_eval?.score));
  const lowShrinkCases = completed.filter((record) => (record.case?.expected?.priorities || []).includes("low_shrinkage"));
  const lowShrinkRecall = lowShrinkCases.length
    ? lowShrinkCases.filter((record) => record.parsed_eval?.checks?.low_shrinkage_priority).length / lowShrinkCases.length
    : null;
  const evidenceCoverage = completed.length
    ? completed.filter((record) => (record.result?.evidence_chain || []).length >= 3).length / completed.length
    : null;
  const kgCoverage = completed.length
    ? completed.filter((record) => (record.result?.kg_hits || []).length || (record.result?.recommendations || []).length).length / completed.length
    : null;
  const reproducibleCoverage = completed.length
    ? completed.filter((record) => record.full_report?.optimization?.objectives?.length && record.full_report?.optimization?.constraints?.length).length / completed.length
    : null;
  return [
    {
      method: "Full KG-enhanced multi-agent",
      status: "Observed",
      parse: avgScore === null ? "-" : `${Math.round(avgScore)}`,
      shrinkage: lowShrinkRecall === null ? "-" : pct(lowShrinkRecall),
      evidence: evidenceCoverage === null ? "-" : pct(evidenceCoverage),
      reproducibility: reproducibleCoverage === null ? "-" : pct(reproducibleCoverage),
      note: "Current platform: requirement parsing + ML optimization + KG evidence + theory checks.",
    },
    {
      method: "No-KG ablation",
      status: "Planned",
      parse: avgScore === null ? "-" : `${Math.round(avgScore)}`,
      shrinkage: "-",
      crack: "-",
      evidence: "0%",
      reproducibility: reproducibleCoverage === null ? "-" : pct(reproducibleCoverage),
      note: "Disable mechanism retrieval; compare recommendation quality and citation coverage.",
    },
    {
      method: "Data-only optimizer",
      status: "Baseline",
      parse: "-",
      shrinkage: "-",
      crack: "-",
      evidence: "0%",
      reproducibility: reproducibleCoverage === null ? "-" : pct(reproducibleCoverage),
      note: "Keep numerical search but remove semantic evidence and KG reasoning.",
    },
    {
      method: "LLM-only direct design",
      status: "Baseline",
      parse: avgScore === null ? "-" : `${Math.round(avgScore)}`,
      shrinkage: "-",
      crack: "-",
      evidence: "0%",
      reproducibility: "0%",
      note: "Text-only reference; no executable bounds or reproducible optimization trace.",
    },
  ];
}

function renderBenchmarkAnalytics(records = []) {
  const root = document.getElementById("benchmarkAnalytics");
  if (!root) return;
  const completed = records.filter((record) => record.status === "completed");
  const failed = records.filter((record) => record.status === "failed");
  const avgScore = mean(completed.map((record) => record.parsed_eval?.score));
  const lowShrinkCases = completed.filter((record) => (record.case?.expected?.priorities || []).includes("low_shrinkage"));
  const lowShrinkRecall = lowShrinkCases.length
    ? lowShrinkCases.filter((record) => record.parsed_eval?.checks?.low_shrinkage_priority).length / lowShrinkCases.length
    : null;
  const evidenceCoverage = completed.length ? completed.filter((record) => (record.result?.evidence_chain || []).length >= 3).length / completed.length : null;
  const kgCoverage = completed.length ? completed.filter((record) => (record.result?.kg_hits || []).length || (record.result?.recommendations || []).length).length / completed.length : null;
  const methodCoverage = completed.length ? completed.filter((record) => (record.result?.method_baselines || []).length >= 4).length / completed.length : null;
  const categoryItems = CATEGORIES.map((category) => {
    const subset = completed.filter((record) => (record.case?.categories || []).includes(category));
    return { label: category, value: subset.length };
  }).filter((item) => item.value);
  const scoreItems = completed.slice(-18).map((record) => ({
    label: record.id,
    value: Number(record.parsed_eval?.score || 0),
    display: record.parsed_eval?.score ?? "-",
    color: (record.case?.expected?.priorities || []).includes("low_shrinkage") ? "#b45309" : "#0f766e",
  }));
  const rows = ablationRows(records);
  root.innerHTML = `
    <header class="analytics-head">
      <div>
        <span>Benchmark Result Visualization</span>
        <strong>消融对照与运行结果图谱</strong>
      </div>
      <p>基于本机 IndexedDB 中已保存的运行记录生成；未实际运行的消融项标记为 Planned/Baseline，不伪造成已完成结果。</p>
    </header>
    <div class="analytics-kpis">
      <article><strong>${completed.length}</strong><span>completed</span></article>
      <article><strong>${failed.length}</strong><span>failed</span></article>
      <article><strong>${avgScore === null ? "-" : Math.round(avgScore)}</strong><span>mean parse score</span></article>
      <article><strong>${lowShrinkRecall === null ? "-" : pct(lowShrinkRecall)}</strong><span>low-shrinkage recall</span></article>
      <article><strong>${crackRecall === null ? "-" : pct(crackRecall)}</strong><span>crack-resistance recall</span></article>
    </div>
    <div class="analytics-grid">
      <section>
        <h3>Core Quality Metrics</h3>
        ${metricBar("Requirement parse score", avgScore === null ? 0 : avgScore / 100, "#0f766e")}
        ${metricBar("Low-shrinkage recognition", lowShrinkRecall ?? 0, "#b45309")}
        ${metricBar("Crack-resistance recognition", crackRecall ?? 0, "#c05621")}
        ${metricBar("Evidence-chain coverage", evidenceCoverage ?? 0, "#2563eb")}
        ${metricBar("KG/recommendation coverage", kgCoverage ?? 0, "#7c3aed")}
        ${metricBar("Method-baseline coverage", methodCoverage ?? 0, "#6b7280")}
      </section>
      <section>
        <h3>Recent Parse Scores</h3>
        ${svgBars(scoreItems, { color: "#0f766e", max: 100 })}
      </section>
      <section>
        <h3>Category Coverage</h3>
        ${svgBars(categoryItems, { color: "#2563eb" })}
      </section>
      <section>
        <h3>Category Accuracy Heatmap</h3>
        ${categoryHeatmap(records)}
      </section>
    </div>
    <section class="ablation-panel">
      <h3>Ablation / Baseline Panel</h3>
      <table>
        <thead><tr><th>Method</th><th>Status</th><th>Parse</th><th>Shrinkage</th><th>Evidence</th><th>Reproducibility</th><th>Interpretation</th></tr></thead>
        <tbody>${rows.map((row) => `
          <tr>
            <td>${escapeHtml(row.method)}</td>
            <td><span class="ablation-status ${escapeHtml(row.status.toLowerCase())}">${escapeHtml(row.status)}</span></td>
            <td>${escapeHtml(row.parse)}</td>
            <td>${escapeHtml(row.shrinkage)}</td>
            <td>${escapeHtml(row.crack)}</td>
            <td>${escapeHtml(row.evidence)}</td>
            <td>${escapeHtml(row.reproducibility)}</td>
            <td>${escapeHtml(row.note)}</td>
          </tr>
        `).join("")}</tbody>
      </table>
    </section>
  `;
}

function renderList() {
  document.getElementById("benchmarkList").innerHTML = visibleCases.map((item) => `
    <article class="benchmark-card" data-case-id="${escapeHtml(item.id)}">
      <header>
        <div><span>${escapeHtml(item.id)}</span><strong>${escapeHtml(item.difficulty === "hard" ? "高难度" : "标准表达")}</strong></div>
        <div class="benchmark-card-actions">
          <em class="case-run-state ${escapeHtml(runStatsCache[item.id]?.status || "pending")}">${escapeHtml(runStatsCache[item.id]?.status || "pending")}</em>
          <button type="button" data-run-one="${escapeHtml(item.id)}">运行</button>
          ${runStatsCache[item.id]?.status === "completed" ? `<button type="button" data-download-report="${escapeHtml(item.id)}">下载报告</button>` : ""}
          <button type="button" data-copy-case="${escapeHtml(item.id)}">复制需求</button>
        </div>
      </header>
      <p>${escapeHtml(item.text)}</p>
      <div class="benchmark-tags">${item.categories.map((tag) => `<span>${escapeHtml(tag)}</span>`).join("")}</div>
      <details>
        <summary>标准解析标签</summary>
        <div class="benchmark-expected">${expectedHtml(item.expected)}</div>
      </details>
    </article>
  `).join("");
}

function openBenchmarkDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(DB_STORE)) {
        const store = db.createObjectStore(DB_STORE, { keyPath: "id" });
        store.createIndex("status", "status", { unique: false });
        store.createIndex("updated_at", "updated_at", { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function dbPut(record) {
  const db = await openBenchmarkDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(DB_STORE, "readwrite");
    tx.objectStore(DB_STORE).put(record);
    tx.oncomplete = () => resolve(record);
    tx.onerror = () => reject(tx.error);
  });
}

async function dbGetAll() {
  const db = await openBenchmarkDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(DB_STORE, "readonly");
    const req = tx.objectStore(DB_STORE).getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
}

async function dbClear() {
  const db = await openBenchmarkDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(DB_STORE, "readwrite");
    tx.objectStore(DB_STORE).clear();
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

function compactDesignResult(payload) {
  const data = payload?.data || payload;
  const reqAgent = (data?.agents || []).find((agent) => agent.id === "requirement");
  const optAgent = (data?.agents || []).find((agent) => agent.id === "optimizer");
  const kgAgent = (data?.agents || []).find((agent) => agent.id === "kg");
  return {
    status: payload?.status || "success",
    lm_ok: data?.lm_ok,
    requirement: reqAgent?.details || null,
    requirement_summary: reqAgent?.summary || "",
    optimizer_summary: optAgent?.summary || "",
    kg_summary: kgAgent?.summary || "",
    kpis: data?.optimization?.kpis || {},
    ratios: data?.optimization?.ratios || {},
    traditional_theory: data?.optimization?.traditional_theory || {},
    solution_table: data?.optimization?.solution_table || [],
    scheme_comparison: data?.optimization?.scheme_comparison || [],
    method_baselines: data?.optimization?.method_baselines || [],
    evidence_chain: data?.optimization?.evidence_chain || [],
    constraint_status: data?.optimization?.constraint_status || {},
    kg_hits: data?.kg_hits || [],
    recommendations: data?.recommendations || [],
    answer_excerpt: String(data?.answer || "").slice(0, 4000),
  };
}

function compactFullPayload(payload) {
  const data = payload?.data || payload || {};
  const clone = typeof structuredClone === "function" ? structuredClone(data) : JSON.parse(JSON.stringify(data));
  if (clone.visualizations?.optimization_plots) {
    clone.visualizations.optimization_plots = clone.visualizations.optimization_plots.map((plot) => ({
      name: plot.name,
      has_image: Boolean(plot.image),
      image_omitted: true,
    }));
  }
  return clone;
}

function reportMarkdown(item, full, compact, parsedEval) {
  const opt = full.optimization || {};
  const req = (full.agents || []).find((agent) => agent.id === "requirement")?.details || compact.requirement || {};
  const kg = (full.agents || []).find((agent) => agent.id === "kg") || {};
  const rows = (opt.solution_table || compact.solution_table || []).map((row) => `| ${row.name || row.material || "-"} | ${row.value ?? "-"} | ${row.unit || ""} |`).join("\n");
  const schemes = (opt.scheme_comparison || compact.scheme_comparison || []).map((row) => `| ${row.label || "-"} | ${(row.roles || []).join(" / ") || "-"} | ${row.strength_mpa ?? "-"} | ${row.carbon_kgco2e_m3 ?? "-"} | ${row.shrinkage_risk_index ?? "-"} | ${row.flowability ?? "-"} |`).join("\n");
  const baselines = (opt.method_baselines || compact.method_baselines || []).map((row) => `| ${row.method || row.method_zh || "-"} | ${row.basis || row.basis_zh || "-"} | ${row.current_system_advantage || row.current_system_advantage_zh || "-"} |`).join("\n");
  const evidenceChain = (opt.evidence_chain || compact.evidence_chain || []).map((row) => `| ${row.claim || row.claim_zh || "-"} | ${row.value ?? "-"} | ${row.source_type || "-"} | ${row.evidence || row.evidence_zh || "-"} | ${row.validation || row.validation_zh || "-"} |`).join("\n");
  const recs = (full.recommendations || compact.recommendations || []).map((rec) => `- ${rec.topic_en || rec.topic || rec.material || "Recommendation"}: ${rec.recommendation_en || rec.recommendation || rec.mechanism_en || rec.mechanism || rec.evidence_en || rec.evidence || ""}`).join("\n") || "- No recommendation returned.";
  return `# Benchmark Report ${item.id}

## Requirement
${item.text}

## Expected Labels
- Target strength: ${item.expected.target_mpa} MPa
- Curing age: ${item.expected.age_days} d
- Required materials: ${(item.expected.required_materials || []).join(", ") || "-"}
- Allowed materials: ${(item.expected.allowed_materials || []).join(", ") || "-"}
- Avoided materials: ${(item.expected.avoided_materials || []).join(", ") || "-"}
- Priorities: ${(item.expected.priorities || []).join(" > ") || "-"}
- Shrinkage target: ${item.expected.shrinkage_max ? `${item.expected.shrinkage_max} microstrain` : (item.expected.priorities || []).includes("low_shrinkage") ? "low-shrinkage qualitative target" : "-"}
- Shrinkage type: ${item.expected.shrinkage_type || "-"}

## Parsed Requirement
- Target strength: ${req.target_mpa ?? "-"} MPa
- Curing age: ${req.age_days ?? "-"} d
- Required materials: ${(req.required_materials || []).join(", ") || "-"}
- Allowed materials: ${(req.allowed_materials || []).join(", ") || "-"}
- Avoided materials: ${(req.avoided_materials || []).join(", ") || "-"}
- Priorities: ${(req.priorities || []).join(" > ") || "-"}
- Parse score: ${parsedEval?.score ?? "-"}
- Parse checks: ${JSON.stringify(parsedEval?.checks || {})}

## Recommended Mix
| Variable | Value | Unit |
|---|---:|---|
${rows || "| - | - | - |"}

## KPIs
- Strength: ${opt.kpis?.strength_mpa ?? compact.kpis?.strength_mpa ?? "-"} MPa
- Carbon: ${opt.kpis?.carbon_kgco2e_m3 ?? compact.kpis?.carbon_kgco2e_m3 ?? "-"} kgCO2e/m3
- Shrinkage-risk index: ${opt.kpis?.shrinkage_risk_index ?? compact.kpis?.shrinkage_risk_index ?? "-"}
- Flowability: ${opt.kpis?.flowability ?? compact.kpis?.flowability ?? "-"}
- Water-to-binder ratio: ${opt.ratios?.water_binder_ratio ?? compact.ratios?.water_binder_ratio ?? "-"}
- Paste-to-aggregate ratio: ${opt.ratios?.paste_aggregate_ratio ?? compact.ratios?.paste_aggregate_ratio ?? "-"}

## Candidate Schemes
| Scheme | Role | Strength | Carbon | Shrinkage/cracking risk | Flowability |
|---|---|---:|---:|---:|---:|
${schemes || "| - | - | - | - | - | - |"}

## Method Baseline Audit
| Method | Basis | Current-system advantage |
|---|---|---|
${baselines || "| - | - | - |"}

## Design Evidence Chain
| Claim | Value | Source | Evidence | Validation |
|---|---:|---|---|---|
${evidenceChain || "| - | - | - | - | - |"}

## Mechanistic Recommendations
${recs}

## Agent Summaries
- Requirement agent: ${compact.requirement_summary || "-"}
- Optimization agent: ${compact.optimizer_summary || "-"}
- Mechanism retrieval agent: ${compact.kg_summary || "-"}

## Final Answer Excerpt
${compact.answer_excerpt || full.answer || ""}

## KG Trace
- Nodes: ${kg.trace?.nodes?.length ?? 0}
- Links: ${kg.trace?.links?.length ?? 0}
`;
}

function evaluateRequirementParse(item, result) {
  const parsed = result?.requirement || {};
  const expected = item.expected || {};
  const constraints = parsed.performance_constraints || [];
  const shrinkageConstraints = constraints.filter((entry) => entry?.metric === "shrinkage");
  const expectedLowShrinkage = (expected.priorities || []).includes("low_shrinkage");
    const checks = [
    ["target_mpa", Number(parsed.target_mpa) === Number(expected.target_mpa)],
    ["age_days", Number(parsed.age_days) === Number(expected.age_days)],
    ["required_materials", (expected.required_materials || []).every((mat) => (parsed.required_materials || []).includes(mat))],
    ["avoided_materials", (expected.avoided_materials || []).every((mat) => (parsed.avoided_materials || []).includes(mat))],
    ["flow_min", !expected.flowability_min || Number(parsed.flowability_min || parsed.slump_flow_min || constraints.find((entry) => entry.metric === "flowability")?.lower || 0) >= Number(expected.flowability_min) - 1],
    ["flow_max", !expected.flowability_max || Number(parsed.flowability_max || parsed.slump_flow_max || constraints.find((entry) => entry.metric === "flowability")?.upper || Infinity) <= Number(expected.flowability_max) + 1],
    ["low_shrinkage_priority", !expectedLowShrinkage || (parsed.priorities || []).includes("low_shrinkage")],
    ["shrinkage_constraint", !expected.shrinkage_max || shrinkageConstraints.some((entry) => Number(entry.upper || entry.target || 0) <= Number(expected.shrinkage_max) + 1)],
  ];
  const passed = checks.filter(([, ok]) => ok).length;
  return {
    score: Math.round((passed / checks.length) * 100),
    checks: Object.fromEntries(checks),
  };
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function recordsToCsv(records) {
  const header = [
    "id", "status", "difficulty", "score", "target_mpa", "age_days", "expected_priorities", "parsed_priorities",
    "strength_mpa", "carbon_kgco2e_m3", "shrinkage_risk_index", "water_binder_ratio", "paste_aggregate_ratio", "error"
  ];
  const rows = records.map((record) => {
    const result = record.result || {};
    const req = result.requirement || {};
    return [
      record.id,
      record.status,
      record.case?.difficulty,
      record.parsed_eval?.score,
      req.target_mpa,
      req.age_days,
      (record.case?.expected?.priorities || []).join(" > "),
      (req.priorities || []).join(" > "),
      result.kpis?.strength_mpa,
      result.kpis?.carbon_kgco2e_m3,
      result.kpis?.shrinkage_risk_index,
      result.ratios?.water_binder_ratio,
      result.ratios?.paste_aggregate_ratio,
      record.error || "",
    ].map(csvEscape).join(",");
  });
  return [header.join(","), ...rows].join("\n");
}

async function runOneCase(item, force = false) {
  const existing = runStatsCache[item.id];
  if (!force && existing?.status === "completed") return existing;
  const started = new Date().toISOString();
  await dbPut({ id: item.id, status: "running", case: item, started_at: started, updated_at: started });
  await refreshRunStats(false);
  const body = {
    prompt: item.text,
    language: window.PlatformI18n?.getLanguage?.() || localStorage.getItem("concrete_platform_language") || "zh",
    algorithm: "NSGA-II",
    pop_size: 48,
    generations: 20,
  };
  try {
    const response = await fetch("/api/agent/design", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok || payload.status !== "success") throw new Error(payload.message || "运行失败");
    const compact = compactDesignResult(payload);
    const parsed_eval = evaluateRequirementParse(item, compact);
    const full_report = compactFullPayload(payload);
    const markdown_report = reportMarkdown(item, full_report, compact, parsed_eval);
    const finished = new Date().toISOString();
    const record = {
      id: item.id,
      status: "completed",
      case: item,
      parsed_eval,
      result: compact,
      full_report,
      markdown_report,
      started_at: started,
      finished_at: finished,
      updated_at: finished,
    };
    await dbPut(record);
    return record;
  } catch (error) {
    const failed = new Date().toISOString();
    const record = {
      id: item.id,
      status: "failed",
      case: item,
      error: error.message,
      started_at: started,
      updated_at: failed,
    };
    await dbPut(record);
    return record;
  }
}

function appendRunnerLog(text, tone = "") {
  const root = document.getElementById("runnerLog");
  const line = document.createElement("div");
  line.className = tone;
  line.textContent = text;
  root.prepend(line);
  while (root.children.length > 80) root.lastElementChild.remove();
}

async function refreshRunStats(renderCards = true) {
  const records = await dbGetAll();
  runStatsCache = Object.fromEntries(records.map((record) => [record.id, record]));
  const completed = records.filter((item) => item.status === "completed").length;
  const failed = records.filter((item) => item.status === "failed").length;
  const running = records.filter((item) => item.status === "running").length;
  const total = BENCHMARK.length;
  const done = completed + failed;
  document.getElementById("runnerStats").innerHTML = [
    ["completed", completed],
    ["failed", failed],
    ["running", running],
    ["remaining", Math.max(0, total - done)],
  ].map(([label, value]) => `<article><strong>${value}</strong><span>${label}</span></article>`).join("");
  document.getElementById("runnerProgressBar").style.width = `${Math.round((done / total) * 100)}%`;
  renderBenchmarkAnalytics(records);
  if (renderCards) renderList();
  return records;
}

async function runCaseQueue(cases, { force = false, failedOnly = false } = {}) {
  if (runActive) return;
  runPaused = false;
  runActive = true;
  const records = await refreshRunStats(false);
  const byId = Object.fromEntries(records.map((record) => [record.id, record]));
  const queue = cases.filter((item) => {
    if (failedOnly) return byId[item.id]?.status === "failed";
    if (force) return true;
    return byId[item.id]?.status !== "completed";
  });
  document.getElementById("runnerStatus").textContent = `队列 ${queue.length} 条；已完成样例会自动跳过。`;
  for (let index = 0; index < queue.length; index += 1) {
    if (runPaused) {
      appendRunnerLog("已暂停，可点击断点续跑继续。", "warn");
      break;
    }
    const item = queue[index];
    document.getElementById("runnerStatus").textContent = `运行 ${index + 1}/${queue.length}: ${item.id}`;
    appendRunnerLog(`开始 ${item.id}: ${item.text.slice(0, 52)}...`);
    const record = await runOneCase(item, force);
    appendRunnerLog(`${record.status === "completed" ? "完成" : "失败"} ${item.id}${record.error ? `: ${record.error}` : ""}`, record.status === "completed" ? "ok" : "bad");
    await refreshRunStats();
    await new Promise((resolve) => setTimeout(resolve, 150));
  }
  runActive = false;
  document.getElementById("runnerStatus").textContent = runPaused ? "已暂停，运行记录已保存。" : "队列结束，运行记录已保存。";
}

async function copyText(text) {
  await navigator.clipboard.writeText(text);
}

function downloadJson() {
  const blob = new Blob([JSON.stringify(BENCHMARK, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "concrete-demand-understanding-benchmark.json";
  link.click();
  URL.revokeObjectURL(url);
}

function downloadBlob(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

async function downloadCaseReport(caseId) {
  const records = await dbGetAll();
  const record = records.find((item) => item.id === caseId);
  if (!record) return;
  const base = `benchmark-${caseId}-full-report`;
  downloadBlob(`${base}.md`, record.markdown_report || JSON.stringify(record, null, 2), "text/markdown;charset=utf-8");
  downloadBlob(`${base}.json`, JSON.stringify(record, null, 2), "application/json;charset=utf-8");
}

async function downloadFullReportPackage() {
  const records = await dbGetAll();
  const completed = records.filter((item) => item.status === "completed");
  const payload = {
    generated_at: new Date().toISOString(),
    total_records: records.length,
    completed: completed.length,
    failed: records.filter((item) => item.status === "failed").length,
    average_parse_score: completed.length ? Math.round(completed.reduce((sum, item) => sum + Number(item.parsed_eval?.score || 0), 0) / completed.length) : null,
    low_shrinkage_completed: completed.filter((item) => (item.case?.expected?.priorities || []).includes("low_shrinkage")).length,
    low_shrinkage_completed: completed.filter((item) => (item.case?.expected?.priorities || []).includes("low_shrinkage")).length,
    records,
    markdown_reports: Object.fromEntries(records.filter((item) => item.markdown_report).map((item) => [`${item.id}.md`, item.markdown_report])),
  };
  downloadBlob("concrete-demand-benchmark-full-report-package.json", JSON.stringify(payload, null, 2), "application/json;charset=utf-8");
  downloadBlob("concrete-demand-benchmark-summary.csv", recordsToCsv(records), "text/csv;charset=utf-8");
  downloadBlob(
    "concrete-demand-benchmark-combined-reports.md",
    records.filter((item) => item.markdown_report).map((item) => item.markdown_report).join("\n\n---\n\n"),
    "text/markdown;charset=utf-8"
  );
}

document.getElementById("categoryFilters").addEventListener("click", (event) => {
  const button = event.target.closest("[data-category]");
  if (!button) return;
  activeCategory = activeCategory === button.dataset.category ? "" : button.dataset.category;
  renderFilters();
  applyFilters();
});
document.getElementById("clearFilters").addEventListener("click", () => {
  activeCategory = "";
  document.getElementById("benchmarkSearch").value = "";
  document.getElementById("difficultyFilter").value = "";
  renderFilters();
  applyFilters();
});
document.getElementById("benchmarkSearch").addEventListener("input", applyFilters);
document.getElementById("difficultyFilter").addEventListener("change", applyFilters);
document.getElementById("benchmarkList").addEventListener("click", async (event) => {
  const runButton = event.target.closest("[data-run-one]");
  if (runButton) {
    const item = BENCHMARK.find((entry) => entry.id === runButton.dataset.runOne);
    if (!item) return;
    await runCaseQueue([item], { force: true });
    return;
  }
  const reportButton = event.target.closest("[data-download-report]");
  if (reportButton) {
    await downloadCaseReport(reportButton.dataset.downloadReport);
    return;
  }
  const button = event.target.closest("[data-copy-case]");
  if (!button) return;
  const item = BENCHMARK.find((entry) => entry.id === button.dataset.copyCase);
  if (!item) return;
  await copyText(item.text);
  button.textContent = "已复制";
  window.setTimeout(() => { button.textContent = "复制需求"; }, 900);
});
document.getElementById("copyVisible").addEventListener("click", async () => {
  await copyText(visibleCases.map((item) => `${item.id} ${item.text}`).join("\n"));
});
document.getElementById("downloadBenchmark").addEventListener("click", downloadJson);
document.getElementById("runVisibleCases").addEventListener("click", () => runCaseQueue(visibleCases));
document.getElementById("resumeCases").addEventListener("click", () => runCaseQueue(BENCHMARK));
document.getElementById("runFailedCases").addEventListener("click", () => runCaseQueue(BENCHMARK, { failedOnly: true }));
document.getElementById("pauseRunner").addEventListener("click", () => {
  runPaused = true;
  document.getElementById("runnerStatus").textContent = "正在请求暂停，当前样例结束后停止。";
});
document.getElementById("downloadRunResults").addEventListener("click", async () => {
  await downloadFullReportPackage();
});
document.getElementById("clearRunResults").addEventListener("click", async () => {
  if (!window.confirm("确认清空本机保存的 benchmark 运行记录？不会删除原始需求集。")) return;
  await dbClear();
  await refreshRunStats();
  document.getElementById("runnerLog").innerHTML = "";
  document.getElementById("runnerStatus").textContent = "运行记录已清空。";
});

renderFilters();
refreshRunStats(false).then(() => {
  applyFilters();
});
