(() => {
  const KEY = "concrete_platform_language";
  const DEFAULT_LANGUAGE = "zh";

  const dictionary = {
    "作者：张玉庭，CSU": "Author: Yuting Zhang, CSU",
    "作者：张玉庭 CSU": "Author: Yuting Zhang, CSU",
    "会话": "Sessions",
    "新建": "New",
    "新会话": "New session",
    "删除会话": "Delete session",
    "设置 / 多智能体 Skill": "Settings / Multi-Agent Skills",
    "多智能体 Skill 设置": "Multi-Agent Skill Settings",
    "多智能体 Skill": "Multi-Agent Skills",
    "机器学习平台": "Machine Learning Platform",
    "摘要证据增强平台": "Evidence Corpus Platform",
    "需求理解 Benchmark": "Requirement Benchmark",
    "模型": "Model",
    "检查中": "Checking",
    "已连接": "Connected",
    "未连接": "Disconnected",
    "保存中...": "Saving...",
    "已保存": "Saved",
    "保存失败": "Save failed",
    "Skill 不存在。": "Skill does not exist.",
    "Skill 提示词不能为空。": "Skill prompt cannot be empty.",
    "暂无知识资产。": "No knowledge assets.",
    "查看": "View",
    "重解析": "Reprocess",
    "删除": "Delete",
    "删除当前 Skill 全部资产": "Delete all assets for this Skill",
    "RAG 操作流": "RAG Operation Flow",
    "后台文档处理": "Background Document Processing",
    "Skill v2 自动评估": "Skill v2 Auto-Evaluation",
    "自动知识图谱候选": "Automatic KG Candidates",
    "当前先归档到对应 Skill；后续可接入分块、索引与检索。": "The file is first archived to the selected Skill; chunking, indexing, and retrieval can be connected afterward.",
    "选择一个根目录，自动递归导入其中所有子文件夹内的 PDF。": "Select a root directory and recursively import PDFs from all subfolders.",
    "上传 testML 导出的模型包": "Upload a model package exported from testML",
    "尚未加载模型": "No model loaded",
    "高级优化设置": "Advanced Optimization Settings",
    "默认由用户指令解析，必要时手动覆盖": "Parsed from the user request by default; override manually if needed",
    "算法": "Algorithm",
    "种群": "Population",
    "代数": "Generations",
    "变量边界 JSON": "Variable Bounds JSON",
    "多智能体协作的混凝土智能设计": "Multi-Agent Intelligent Concrete Design",
    "请上传模型包，然后直接输入需求。例如：": "Upload a model package, then enter the design requirement. Example:",
    "设计一个 80 MPa、28 天龄期的低碳高流动性混凝土，尽量减少水泥用量，允许使用粉煤灰、矿渣和硅灰。": "Design a low-carbon, high-flowability concrete targeting 80 MPa at 28 days, minimizing cement content and allowing fly ash, slag, and silica fume.",
    "发送": "Send",
    "运行中": "Running",
    "1. 需求解析智能体": "1. Requirement Parsing Agent",
    "2. 优化建模智能体": "2. Optimization Modeling Agent",
    "3. 机理检索智能体": "3. Mechanism Retrieval Agent",
    "等待用户需求": "Waiting for user requirement",
    "等待目标函数与约束设置": "Waiting for objectives and constraints",
    "等待知识图谱检索": "Waiting for knowledge graph retrieval",
    "快速指标": "Key Metrics",
    "推荐配合比": "Recommended Mix Design",
    "机理建议": "Mechanistic Recommendations",
    "返回设计 Copilot": "Back to Design Copilot",
    "返回主界面": "Back to Main",
    "多智能体 Skill 设置": "Multi-Agent Skill Settings",
    "管理协议驱动 Skill、知识资产、证据路由与验证闭环。": "Manage protocol-driven skills, knowledge assets, evidence routing, and validation loops.",
    "名称": "Name",
    "版本": "Version",
    "说明": "Description",
    "协议驱动智能体": "Protocol-Driven Agent",
    "任务契约、证据路由、工具链、校验器与反思闭环共同约束。": "Task contracts, evidence routing, toolchains, validators, and reflection loops jointly constrain agent behavior.",
    "输入 / 输出契约": "Input / Output Contract",
    "证据路由策略": "Evidence Routing Policy",
    "工具链": "Toolchain",
    "验证规则": "Validators",
    "反思闭环": "Reflection Loop",
    "评估基准集": "Evaluation Benchmarks",
    "契约": "Contract",
    "检索": "Retrieval",
    "工具": "Tools",
    "验证": "Validation",
    "反思": "Reflection",
    "评估": "Evaluation",
    "输入/输出结构": "Input/output schema",
    "证据路由": "Evidence routing",
    "计算与执行": "Computation and execution",
    "硬约束审计": "Hard-constraint audit",
    "自检修订": "Self-check and revision",
    "基准回归": "Benchmark regression",
    "术语定义": "Terminology definition",
    "性能阈值": "Performance thresholds",
    "材料语义": "Material semantics",
    "工程类型": "Engineering type",
    "经验公式": "Empirical formulae",
    "变量边界": "Variable bounds",
    "目标函数": "Objective functions",
    "规范阈值": "Standards thresholds",
    "材料最小掺量": "Minimum meaningful dosage",
    "材料-特征-机理-性能路线": "Material-feature-mechanism-performance routes",
    "风险机制": "Risk mechanisms",
    "替代材料": "Alternative materials",
    "结论支撑": "Conclusion support",
    "规范评价": "Standards evaluation",
    "实验建议": "Experimental recommendations",
    "风险说明": "Risk statement",
    "证据交叉验证": "Evidence cross-validation",
    "把自然语言映射为可计算规格，并识别是否触发专用规范。": "Map natural language to computable specifications and identify whether specific standards are triggered.",
    "将需求转为可计算优化问题，并核验物理与工程边界。": "Convert requirements into a computable optimization problem and audit physical and engineering boundaries.",
    "用多跳机理解释当前方案，并决定是否反馈修改。": "Explain the current design through multi-hop mechanisms and decide whether feedback revision is needed.",
    "把计算、规范、机理和文献整合成可用于工程或科研复核的正式报告。": "Integrate computation, standards, mechanisms, and literature into a report suitable for engineering or scientific review.",
    "语义解析器": "Semantic parser",
    "材料意图分类器": "Material-intent classifier",
    "证据检索器": "Evidence retriever",
    "JSON schema 校验器": "JSON schema validator",
    "鲍罗米先验": "Bolomey prior",
    "绝对体积法": "Absolute-volume method",
    "多目标优化器": "Multi-objective optimizer",
    "计算器": "Calculator",
    "约束校验器": "Constraint auditor",
    "多跳图检索": "Multi-hop graph retrieval",
    "混合检索": "Hybrid retrieval",
    "因果链评分": "Causal-chain scoring",
    "二次优化触发器": "Second-pass optimization trigger",
    "报告规划器": "Report planner",
    "引用编排器": "Citation orchestrator",
    "表格生成器": "Table generator",
    "公式渲染器": "Formula renderer",
    "完整性审计器": "Completeness auditor",
    "Skill 提示词": "Skill Prompt",
    "保存 Skill": "Save Skill",
    "知识资产工作台": "Knowledge Asset Workbench",
    "共享给全部智能体": "Shared by all agents",
    "上传摘要语料 TXT 或规范 PDF": "Upload abstract corpus TXT or standard PDF",
    "导入规范 PDF 文件夹": "Import standard PDF folder",
    "证据状态": "Evidence Status",
    "评估基准": "Evaluation Benchmark",
    "知识图谱候选": "Knowledge Graph Candidates",
    "仅删除当前智能体资产，并同步移除对应证据索引与自动 KG 候选。": "Deletes only the current agent assets and synchronizes the evidence index and automatic KG candidates.",
    "等待资产": "Waiting for assets",
    "OCR / 表格 / 索引": "OCR / Tables / Indexing",
    "选择一个资产查看提取文本、切块与索引过程。": "Select an asset to inspect extracted text, chunks, and indexing.",
    "统一证据索引": "Unified Evidence Index",
    "重建索引": "Rebuild Index",
    "LoRA 微调工作台": "LoRA Fine-Tuning Workbench",
    "语料上传": "Corpus Upload",
    "待上传": "Waiting",
    "上传 40000 篇 WOS 摘要": "Upload 40,000 WOS Abstracts",
    "支持 WOS plain text、TSV、CSV": "Supports WOS plain text, TSV, and CSV",
    "解析并建立语料": "Parse and Build Corpus",
    "运行设置": "Run Settings",
    "每条 KG 保留证据数": "Evidence Items per KG Edge",
    "调试文献上限": "Debug Document Limit",
    "运行证据校准": "Run Evidence Calibration",
    "计算智能体证据先验": "Compute Agent Evidence Priors",
    "生成 NC 级别图件": "Generate NC-Level Figures",
    "规范约束库": "Standards Constraint Library",
    "未加载": "Not loaded",
    "规范 PDF 文件夹": "Standards PDF Folder",
    "抽取模式": "Extraction Mode",
    "高精度": "High precision",
    "扩展召回": "Expanded recall",
    "扫描全部 PDF（2277 个，较慢）": "Scan all PDFs (2,277 files, slower)",
    "抽取规范约束": "Extract Standards Constraints",
    "Embedding 设置": "Embedding Settings",
    "未测试": "Not tested",
    "保存 Embedding 设置": "Save Embedding Settings",
    "测试 Embedding 连接": "Test Embedding Connection",
    "LoRA 微调": "LoRA Fine-Tuning",
    "未生成": "Not generated",
    "基座模型": "Base Model",
    "Embedding 模型": "Embedding Model",
    "最大样本数": "Maximum Examples",
    "生成 LoRA 数据与脚本": "Generate LoRA Data and Scripts",
    "导出": "Exports",
    "校准 JSON": "Calibration JSON",
    "证据表 CSV": "Evidence Table CSV",
    "大规模摘要语料驱动的精选知识图谱证据校准与缺口识别": "Curated KG Evidence Calibration and Gap Analysis Driven by Large-Scale Abstract Corpora",
    "加载中": "Loading",
    "NC 级别证据图件": "NC-Level Evidence Figures",
    "全部混凝土类型": "All Concrete Types",
    "普通混凝土": "General Concrete",
    "配合比设计": "Mix Design",
    "泵送混凝土": "Pumped Concrete",
    "自密实混凝土": "Self-Compacting Concrete",
    "高强混凝土": "High-Strength Concrete",
    "高性能混凝土": "High-Performance Concrete",
    "活性粉末/UHPC": "Reactive Powder / UHPC",
    "水运/海工混凝土": "Waterway / Marine Concrete",
    "耐久性混凝土": "Durability Concrete",
    "全部用途": "All Uses",
    "可进入优化器": "Optimizer-ready",
    "仅作规范依据": "Reference only",
    "运行 KG 证据校准": "Run KG Evidence Calibration",
    "计算 Evidence Landscape": "Compute Evidence Landscape",
    "生成图件": "Generate Figures",
    "按置信度": "By Confidence",
    "按证据缺口": "By Evidence Gap",
    "按近年证据": "By Recent Evidence",
    "KG 证据校准": "KG Evidence Calibration",
    "证据缺口与使用建议": "Evidence Gaps and Usage Guidance",
    "摘要记录": "Abstract Records",
    "精选 KG 链": "Curated KG Chains",
    "高置信链": "High-Confidence Chains",
    "证据用途": "Evidence Usage",
    "未上传": "Not uploaded",
    "已校准": "Calibrated",
    "待校准": "Pending calibration",
    "证据层支持": "Evidence-layer support",
    "不扩图": "No graph expansion",
    "仅校准、检索、缺口识别": "Calibration, retrieval, and gap analysis only",
    "尚未运行证据校准": "Evidence calibration has not been run.",
    "查看支撑摘要": "View Supporting Abstracts",
    "证据不足": "Insufficient evidence",
    "适用边界": "Applicability boundary",
    "近期证据弱": "Weak recent evidence",
    "低置信": "Low confidence",
    "可作为证据链": "Usable evidence chain",
    "当前 Gold KG 均有摘要证据支撑；可按“证据缺口”排序继续审查。": "All Gold KG items currently have abstract evidence support; sort by evidence gap for further review.",
    "尚未生成图件": "Figures have not been generated.",
    "先完成 KG 校准和 Evidence Landscape，再生成 LoRA 数据。": "Complete KG calibration and Evidence Landscape before generating LoRA data.",
    "训练集": "Training Set",
    "验证集": "Validation Set",
    "训练脚本": "Training Script",
    "无": "None",
    "约束/依据": "Constraints / References",
    "可优化": "Optimizable",
    "指标": "Metrics",
    "类型": "Types",
    "尚未抽取规范，或当前筛选条件无结果。": "No standards have been extracted, or the current filters returned no results.",
    "规范": "Standard",
    "优化约束": "Optimization constraint",
    "依据": "Reference",
    "待处理": "Pending",
    "失败": "Failed",
    "读取失败": "Read failed",
    "请选择摘要文件": "Please select an abstract file.",
    "已保存": "Saved",
    "保存失败": "Save failed",
    "测试失败": "Test failed",
    "生成失败": "Generation failed",
    "数据上传、EDA、预处理、模型训练、解释、优化与 Nature 级图件生成。": "Data upload, EDA, preprocessing, model training, interpretation, optimization, and Nature-grade figure generation.",
    "原 testML 功能不变；当前页仅统一入口、布局和视觉风格。": "Original testML functions are unchanged; this page only unifies entry, layout, and visual style.",
    "工作流": "Workflow",
    "返回设计 Copilot": "Back to Design Copilot",
    "数据上传": "Data Upload",
    "预处理": "Preprocessing",
    "模型训练": "Model Training",
    "优化": "Optimization",
    "当前": "Current",
    "导出当前": "Export Current",
    "导出数据": "Export Data",
    "一键导出本页全部图": "Export All Figures on This Page",
    "图像导入失败": "Image import failed",
    "请先上传数据集。": "Please upload a dataset first.",
    "请先选择模型。": "Please select a model first.",
    "请先运行优化。": "Please run optimization first.",
    "当前没有可导出的图像。": "No figure is available for export.",
    "当前页还没有可导出的图。": "No figures are available on this page.",
    "解析结果": "Parsed Result",
    "为什么这样解析": "Rationale",
    "传递给下游智能体的设计规格": "Design Specification for Downstream Agents",
    "本智能体使用的证据": "Evidence Used by This Agent",
    "参考文献与证据来源": "References and Evidence Sources",
    "中文": "Chinese",
    "English": "English"
  };

  const phraseDictionary = {
    "语料 ": "Corpus ",
    " 条": " records",
    "已解析 ": "Parsed ",
    "支撑证据": "Supporting evidence",
    "近年证据": "Recent evidence",
    "反向线索": "Contradictory signals",
    "共 ": "Total ",
    "第 ": "Page ",
    " 页": "",
    "上一页": "Previous",
    "下一页": "Next",
    "上一段": "Previous section",
    "下一段": "Next section",
    "篇文献": "literature records",
    "字符": "characters",
    "块": "chunks",
    "张表": "tables",
    "个 PDF": "PDFs",
    "条可优化约束": "optimizer-ready constraints",
    "正在": "",
    "预处理默认规范库": "Preprocessing the default standards library",
    "解析摘要语料": "Parsing abstract corpus",
    "校准精选 KG": "Calibrating curated KG",
    "计算 evidence landscape 和智能体先验": "Computing Evidence Landscape and agent priors",
    "生成 NC 级别图件": "Generating NC-level figures",
    "保存 Embedding 设置": "Saving embedding settings",
    "测试 Embedding 连接": "Testing embedding connection",
    "生成 LoRA 微调数据与训练脚本": "Generating LoRA fine-tuning data and training scripts",
    "抽取规范强度/流动性约束": "Extracting standards strength/flowability constraints",
    "完成：": "Complete: ",
    "个材料-性能证据单元": " material-performance evidence units",
    "已生成 ": "Generated ",
    "组图件": "figure sets",
    "Embedding 设置已保存": "Embedding settings saved",
    "Embedding 测试成功：": "Embedding test passed: ",
    "维": "dimensions",
    "LoRA 数据已生成：训练": "LoRA data generated: training",
    "验证": "validation",
    "规范抽取完成：": "Standards extraction complete: ",
    "保存中...": "Saving...",
    "上传中...": "Uploading...",
    "上传完成": "Upload complete",
    "文件已存在，已刷新索引": "File already exists; index refreshed",
    "已命中重复文件": "Duplicate file detected",
    "单文件入库完成": "Single file indexed",
    "该文件夹内未找到 PDF。": "No PDFs were found in this folder.",
    "正在递归导入": "Recursively importing",
    "已导入": "Imported",
    "跳过": "skipped",
    "个文件": "files",
    "批量导入完成：": "Batch import complete: ",
    "处理中": "Processing",
    "资产处理链回放": "Asset processing replay",
    "未提取到可预览文本。": "No previewable text was extracted.",
    "表格抽取": "Table Extraction",
    "已解析文献记录": "Parsed Literature Records",
    "抽取到的规范指标": "Extracted Standards Metrics",
    "自动生成的知识图谱三元组": "Automatically Generated KG Triples",
    "资产已删除，索引已同步更新。": "Asset deleted; index synchronized.",
    "索引已更新": "Index updated",
    "当前 Skill 没有可删除资产。": "The current Skill has no deletable assets.",
    "正在删除全部资产...": "Deleting all assets...",
    "当前 Skill 的知识资产已全部删除，证据索引和自动 KG 候选已同步更新。": "All knowledge assets for the current Skill have been deleted; evidence index and automatic KG candidates were synchronized.",
    "已删除": "Deleted",
    "已加入后台解析队列。": "Added to the background parsing queue.",
    "文献摘要": "Literature Abstracts",
    "规范全文": "Full Standards",
    "KG 机理": "KG Mechanisms",
    "自动 KG": "Automatic KG",
    "暂无后台任务。": "No background jobs.",
    "Skill v2 协议健康度": "Skill v2 Protocol Health",
    "契约、证据路由、工具链、验证规则、反思闭环和基准样例的自动审计结果。": "Automatic audit results for contracts, evidence routing, toolchains, validators, reflection loops, and benchmark cases.",
    "通过项": "Passed",
    "待修正项": "Needs revision",
    "查看审计明细": "View audit details",
    "通过": "Approve",
    "驳回": "Reject",
    "暂无候选三元组。上传或重解析文献、规范后会自动生成。": "No candidate triples yet. Upload or reprocess literature and standards to generate them automatically.",
    "索引重建中...": "Rebuilding index...",
    "索引已重建": "Index rebuilt"
  };

  const placeholders = {
    "例如：设计一个 80 MPa、28 天龄期的低碳高流动性混凝土，要求泵送施工，尽量少用水泥。":
      "Example: design a low-carbon, pumpable concrete targeting 80 MPa at 28 days, with high flowability and minimal cement content.",
    "搜索：80MPa、禁止硅灰、自密实、冻融、泵送、流动性...":
      "Search: 80 MPa, no silica fume, self-compacting, freeze-thaw, pumping, flowability..."
  };

  function getLanguage() {
    const queryLanguage = new URLSearchParams(window.location.search).get("lang");
    if (queryLanguage) {
      const normalized = queryLanguage.toLowerCase().startsWith("en") ? "en" : "zh";
      localStorage.setItem(KEY, normalized);
      return normalized;
    }
    return localStorage.getItem(KEY) || DEFAULT_LANGUAGE;
  }

  function setLanguage(language) {
    localStorage.setItem(KEY, language);
    applyLanguage(document);
    document.dispatchEvent(new CustomEvent("platform-language-change", { detail: { language } }));
  }

  function translateText(value) {
    const trimmed = value.trim();
    if (!trimmed) return value;
    const translated = dictionary[trimmed];
    if (translated) return value.replace(trimmed, translated);
    const phraseTranslated = translatePhraseText(trimmed);
    return phraseTranslated === trimmed ? value : value.replace(trimmed, phraseTranslated);
  }

  function translatePhraseText(value) {
    let output = value;
    Object.entries(phraseDictionary).forEach(([source, target]) => {
      output = output.split(source).join(target);
    });
    return output;
  }

  function applyLanguage(root = document) {
    const language = getLanguage();
    document.documentElement.lang = language === "en" ? "en" : "zh-CN";
    document.body?.classList.toggle("lang-en", language === "en");
    document.body?.classList.toggle("lang-zh", language !== "en");

    if (language !== "en") return;

    const walker = document.createTreeWalker(root.body || root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        const parent = node.parentElement;
        if (!parent || parent.closest("script,style,textarea,input,code,pre,.chat-log,.message,.bubble,.run-report,.publication-report-shell,[data-no-i18n]")) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((node) => {
      if (!node.__zhText || node.nodeValue !== node.__i18nLastText) node.__zhText = node.nodeValue;
      node.__i18nLastText = translateText(node.__zhText);
      node.nodeValue = node.__i18nLastText;
    });

    document.querySelectorAll("[placeholder]").forEach((node) => {
      if (!node.dataset.zhPlaceholder) node.dataset.zhPlaceholder = node.getAttribute("placeholder") || "";
      const text = placeholders[node.dataset.zhPlaceholder] || dictionary[node.dataset.zhPlaceholder];
      if (text) node.setAttribute("placeholder", text);
    });
    document.querySelectorAll("[title]").forEach((node) => {
      if (!node.dataset.zhTitle) node.dataset.zhTitle = node.getAttribute("title") || "";
      const text = dictionary[node.dataset.zhTitle];
      if (text) node.setAttribute("title", text);
    });
  }

  function restoreChinese() {
    document.querySelectorAll("[placeholder][data-zh-placeholder]").forEach((node) => {
      node.setAttribute("placeholder", node.dataset.zhPlaceholder);
    });
    document.querySelectorAll("[title][data-zh-title]").forEach((node) => {
      node.setAttribute("title", node.dataset.zhTitle);
    });
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const parent = node.parentElement;
        if (!parent || parent.closest(".chat-log,.message,.bubble,.run-report,.publication-report-shell,[data-no-i18n]")) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((node) => {
      if (node.__zhText) node.nodeValue = node.__zhText;
    });
    document.documentElement.lang = "zh-CN";
    document.body.classList.remove("lang-en");
    document.body.classList.add("lang-zh");
  }

  function installToggle() {
    if (document.querySelector(".language-toggle")) return;
    const wrap = document.createElement("div");
    wrap.className = "language-toggle";
    wrap.innerHTML = `
      <button type="button" data-lang="zh">中文</button>
      <button type="button" data-lang="en">EN</button>
    `;
    wrap.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-lang]");
      if (!button) return;
      const language = button.dataset.lang;
      if (language === "zh") {
        localStorage.setItem(KEY, "zh");
        restoreChinese();
      } else {
        setLanguage("en");
      }
      updateToggle();
    });
    document.body.appendChild(wrap);
    updateToggle();
  }

  function updateToggle() {
    const language = getLanguage();
    document.querySelectorAll(".language-toggle button").forEach((button) => {
      button.classList.toggle("active", button.dataset.lang === language);
    });
  }

  window.PlatformI18n = {
    key: KEY,
    getLanguage,
    setLanguage,
    applyLanguage,
    translate: (value) => (getLanguage() === "en" ? translateText(value) : value),
  };

  document.addEventListener("DOMContentLoaded", () => {
    installToggle();
    if (getLanguage() === "en") applyLanguage(document);
    const observer = new MutationObserver(() => {
      if (getLanguage() === "en") {
        window.clearTimeout(observer._timer);
        observer._timer = window.setTimeout(() => applyLanguage(document), 50);
      }
    });
    observer.observe(document.body, { childList: true, characterData: true, subtree: true });
  });
})();
