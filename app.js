const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const promptInput = document.getElementById("promptInput");
const sendButton = document.getElementById("sendButton");
const modelFile = document.getElementById("modelFile");
const modelStatus = document.getElementById("modelStatus");
const coreStatus = document.getElementById("coreStatus");
const lmStatus = document.getElementById("lmStatus");
const sessionList = document.getElementById("sessionList");
const newSessionButton = document.getElementById("newSessionButton");
let followChatTail = true;
let latestReportPayload = null;
let currentSessionId = null;
const citationDetailCache = new Map();
let citationHoverTimer = null;
const SESSION_STORAGE_KEY = "concrete_design_sessions_v1";
const SESSION_ACTIVE_KEY = "concrete_design_active_session_v1";
const currentLanguage = () => window.PlatformI18n?.getLanguage?.() || localStorage.getItem("concrete_platform_language") || "zh";
const EN_UI = {
  "设计推理主舞台": "Design Reasoning Main Stage",
  "语义解析、约束收缩、Pareto 搜索和知识图谱命中同步展开": "Semantic parsing, constraint contraction, Pareto search, and knowledge-graph hits unfold in parallel",
  "实时操作流": "Live Operation Stream",
  "实时 Pareto 搜索": "Live Pareto Search",
  "实时优化曲线": "Live optimization curve",
  "语义拆解轨道": "Semantic Decomposition Track",
  "目标与约束装配": "Objective and Constraint Assembly",
  "知识图谱命中": "Knowledge-Graph Hits",
  "多智能体协同设计中": "Multi-Agent Co-Design Running",
  "实时监控": "Live Monitor",
  "等待启动": "Waiting to start",
  "需求解析智能体": "Requirement Parsing Agent",
  "优化建模智能体": "Optimization Modeling Agent",
  "机理检索智能体": "Mechanism Retrieval Agent",
  "总报告智能体": "Final Report Agent",
  "解析": "Parse",
  "运行": "Run",
  "回答": "Output",
  "读取自然语言，识别强度、龄期、材料与性能目标": "Read natural language and identify strength, curing age, materials, and performance targets",
  "语义切分中": "Segmenting semantics",
  "等待输出": "Waiting for output",
  "等待需求规格": "Waiting for requirement specification",
  "等待目标函数": "Waiting for objective functions",
  "等待优化结果": "Waiting for optimization results",
  "等待材料与性能查询": "Waiting for material and performance queries",
  "等待图谱遍历": "Waiting for graph traversal",
  "等待机理建议": "Waiting for mechanistic recommendations",
  "智能体逐步输出": "Stepwise Agent Outputs",
  "已接收需求规格": "Requirement specification received",
  "构造多目标问题并搜索 Pareto 解": "Formulating the multi-objective problem and searching Pareto solutions",
  "已接收优化解和性能目标": "Optimization solution and performance targets received",
  "沿材料因果链检索": "Retrieving along material causal chains",
  "初始化种群": "Initialize population",
  "种群": "population",
  "目标函数": "Objective function",
  "约束条件": "Constraint",
  "优化完成": "Optimization completed",
  "构造图谱查询": "Build graph query",
  "统一证据查询": "Unified evidence query",
  "候选证据": "Candidate evidence",
  "证据重排完成": "Evidence reranking completed",
  "展开命中子图": "Expand hit subgraph",
  "命中节点": "Hit node",
  "扩展关系": "Expanded relation",
  "图谱检索完成": "Graph retrieval completed",
  "机理复审": "Mechanistic review",
  "启动二次优化": "Start second-pass optimization",
  "二次优化完成": "Second-pass optimization completed",
  "二次优化失败": "Second-pass optimization failed",
  "启动报告生成": "Start report generation",
  "开始生成文本": "Start text generation",
  "文本输出完成": "Text output completed",
  "算法": "algorithm",
  "预测强度": "predicted strength",
  "文献": "literature",
  "规范": "standards",
  "精选 KG": "curated KG",
  "自动 KG": "automatic KG",
  "保留": "retained",
  "条高相关证据": "high-relevance evidence items",
  "抽取": "extracted",
  "条多跳路线": "multi-hop routes",
  "个命中节点": "hit nodes",
  "条命中关系": "hit relations",
  "生成": "generated",
  "条建议": "recommendations",
  "触发二次优化": "trigger second-pass optimization",
  "不触发二次优化": "do not trigger second-pass optimization",
  "应用": "apply",
  "项机理修正": "mechanistic adjustments",
  "调用三个子智能体与总报告智能体": "Call the three sub-agents and the final report agent",
};
const t = (text) => (currentLanguage() === "en" ? (EN_UI[text] || window.PlatformI18n?.translate?.(text) || text) : text);

const RECOVERED_KG_EN_FALLBACK = {
  "GGBFS": [
    {
      "feature": "Calcium aluminosilicate glass",
      "mechanism": "Latent hydraulic reaction",
      "consequence": "C-A-S-H gel",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase at late ages",
      "evidence_count": 156,
      "route_id": "route-023"
    },
    {
      "feature": "Smooth glass surface",
      "mechanism": "Reduces friction",
      "consequence": "Yield stress lowers",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "increase",
      "evidence_count": 67,
      "route_id": "route-024"
    },
    {
      "feature": "High replacement early",
      "mechanism": "Reduces clinker contribution",
      "consequence": "Early solid volume lowers",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "decrease at early ages",
      "evidence_count": 300,
      "route_id": "route-025"
    },
    {
      "feature": "Fine slag",
      "mechanism": "Accelerates dissolution",
      "consequence": "Continuous matrix",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 404,
      "route_id": "route-026"
    },
    {
      "feature": "Very fine dosage",
      "mechanism": "Raises water demand",
      "consequence": "Viscosity increases",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 97,
      "route_id": "route-027"
    },
    {
      "feature": "Latent hydraulic reaction",
      "mechanism": "Continued hydration changes pore humidity over time",
      "consequence": "Autogenous and drying shrinkage are dosage dependent",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 188,
      "route_id": "route-081"
    },
    {
      "feature": "Fine slag powder",
      "mechanism": "Water demand and early self-desiccation can increase",
      "consequence": "Shrinkage risk requires curing validation",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires validation",
      "evidence_count": 229,
      "route_id": "route-082"
    },
    {
      "feature": "Slag replaces clinker",
      "mechanism": "Industrial by-product reduces clinker content",
      "consequence": "Embodied carbon decreases",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "decreases",
      "evidence_count": 246,
      "route_id": "route-084"
    },
    {
      "feature": "High-volume slag binder",
      "mechanism": "Low-carbon binder retains late strength",
      "consequence": "Carbon per strength unit decreases",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "decreases at later ages",
      "evidence_count": 256,
      "route_id": "route-085"
    },
    {
      "feature": "Activation demand",
      "mechanism": "Alkali or sulfate activation may add processing impact",
      "consequence": "Net carbon benefit is mix-specific",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "dosage dependent",
      "evidence_count": 121,
      "route_id": "route-086"
    },
    {
      "feature": "Latent hydraulic slag replacement",
      "mechanism": "Lowers clinker content and shifts hydration toward later C-A-S-H formation",
      "consequence": "Shrinkage depends on slag fineness, activation, and curing humidity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 237,
      "route_id": "route-135"
    },
    {
      "feature": "Latent hydraulic slag replacement",
      "mechanism": "Lowers clinker content and shifts hydration toward later C-A-S-H formation",
      "consequence": "Shrinkage depends on slag fineness, activation, and curing humidity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "controls",
      "evidence_count": 237,
      "route_id": "route-136"
    },
    {
      "feature": "Latent hydraulic slag replacement",
      "mechanism": "Lowers clinker content and shifts hydration toward later C-A-S-H formation",
      "consequence": "Shrinkage depends on slag fineness, activation, and curing humidity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires validation",
      "evidence_count": 237,
      "route_id": "route-137"
    },
    {
      "feature": "Latent hydraulic slag replacement",
      "mechanism": "Lowers clinker content and shifts hydration toward later C-A-S-H formation",
      "consequence": "Lower clinker emissions",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces when optimized",
      "evidence_count": 233,
      "route_id": "route-141"
    },
    {
      "feature": "Latent hydraulic slag replacement",
      "mechanism": "Lowers clinker content and shifts hydration toward later C-A-S-H formation",
      "consequence": "Lower clinker emissions",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "trade-off dependent",
      "evidence_count": 230,
      "route_id": "route-142"
    },
    {
      "feature": "Latent hydraulic slag replacement",
      "mechanism": "Lowers clinker content and shifts hydration toward later C-A-S-H formation",
      "consequence": "Lower clinker emissions",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires life-cycle check",
      "evidence_count": 236,
      "route_id": "route-143"
    },
    {
      "feature": "Latent hydraulic slag replacement",
      "mechanism": "Lowers clinker content and shifts hydration toward later C-A-S-H formation",
      "consequence": "Shrinkage depends on slag fineness, activation, and curing humidity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "age dependent",
      "evidence_count": 237,
      "route_id": "route-253"
    },
    {
      "feature": "Latent hydraulic slag replacement",
      "mechanism": "Lowers clinker content and shifts hydration toward later C-A-S-H formation",
      "consequence": "Shrinkage depends on slag fineness, activation, and curing humidity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "affects",
      "evidence_count": 237,
      "route_id": "route-254"
    },
    {
      "feature": "Latent hydraulic slag replacement",
      "mechanism": "Lowers clinker content and shifts hydration toward later C-A-S-H formation",
      "consequence": "Lower clinker emissions",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires strength normalization",
      "evidence_count": 258,
      "route_id": "route-259"
    },
    {
      "feature": "Latent hydraulic slag replacement",
      "mechanism": "Lowers clinker content and shifts hydration toward later C-A-S-H formation",
      "consequence": "Lower clinker emissions",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "supports service-life benefit",
      "evidence_count": 235,
      "route_id": "route-260"
    },
    {
      "feature": "Latent hydraulic slag replacement",
      "mechanism": "Lowers clinker content and shifts hydration toward later C-A-S-H formation",
      "consequence": "Lower clinker emissions",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires shrinkage-cracking check",
      "evidence_count": 233,
      "route_id": "route-261"
    },
    {
      "feature": "Latent hydraulic slag replacement",
      "mechanism": "Lowers clinker content and shifts hydration toward later C-A-S-H formation",
      "consequence": "Lower clinker emissions",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires curing-compatible design",
      "evidence_count": 246,
      "route_id": "route-262"
    },
    {
      "feature": "Latent hydraulic slag replacement",
      "mechanism": "Lowers clinker content and shifts hydration toward later C-A-S-H formation",
      "consequence": "Lower clinker emissions",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires service-life justification",
      "evidence_count": 234,
      "route_id": "route-263"
    },
    {
      "feature": "Latent hydraulic slag replacement",
      "mechanism": "Lowers clinker content and shifts hydration toward later C-A-S-H formation",
      "consequence": "Lower clinker emissions",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces binder only if workability is retained",
      "evidence_count": 247,
      "route_id": "route-264"
    }
  ],
  "cement": [
    {
      "feature": "C3S-rich clinker",
      "mechanism": "Rapid alite hydration",
      "consequence": "Early C-S-H skeleton",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 197,
      "route_id": "route-006"
    },
    {
      "feature": "C2S fraction",
      "mechanism": "Slow belite hydration",
      "consequence": "Late C-S-H growth",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 301,
      "route_id": "route-007"
    },
    {
      "feature": "C3A sulfate balance",
      "mechanism": "Controls ettringite",
      "consequence": "Stable hydrate assemblage",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 87,
      "route_id": "route-008"
    },
    {
      "feature": "High fineness",
      "mechanism": "Raises surface demand",
      "consequence": "Rapid stiffening",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 19,
      "route_id": "route-009"
    },
    {
      "feature": "Portlandite generation",
      "mechanism": "Feeds pozzolanic reaction",
      "consequence": "Secondary C-S-H",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 219,
      "route_id": "route-010"
    },
    {
      "feature": "Paste flocculation",
      "mechanism": "Traps lubricating water",
      "consequence": "Suspension stiffens",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 61,
      "route_id": "route-011"
    },
    {
      "feature": "High paste volume",
      "mechanism": "Shrinkable cement paste fraction increases",
      "consequence": "Drying shrinkage and volume change increase",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "increases",
      "evidence_count": 1406,
      "route_id": "route-068"
    },
    {
      "feature": "High clinker factor",
      "mechanism": "Calcination and kiln fuel dominate CO2 emissions",
      "consequence": "Embodied carbon increases",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "increases",
      "evidence_count": 171,
      "route_id": "route-069"
    },
    {
      "feature": "Binder reduction",
      "mechanism": "Paste volume and clinker dosage decrease",
      "consequence": "Carbon footprint decreases if strength is retained",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "decreases",
      "evidence_count": 662,
      "route_id": "route-070"
    },
    {
      "feature": "SCM replacement",
      "mechanism": "Clinker fraction is diluted by reactive low-carbon solids",
      "consequence": "CO2 intensity per cubic metre decreases",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "decreases",
      "evidence_count": 158,
      "route_id": "route-071"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "High paste and heat release amplify shrinkage stress accumulation",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 758,
      "route_id": "route-109"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "High paste and heat release amplify shrinkage stress accumulation",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "controls",
      "evidence_count": 691,
      "route_id": "route-110"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "High paste and heat release amplify shrinkage stress accumulation",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires validation",
      "evidence_count": 691,
      "route_id": "route-111"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "Lower clinker factor",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces when optimized",
      "evidence_count": 157,
      "route_id": "route-115"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "Lower clinker factor",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "trade-off dependent",
      "evidence_count": 138,
      "route_id": "route-116"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "Lower clinker factor",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires life-cycle check",
      "evidence_count": 296,
      "route_id": "route-117"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "High paste and heat release amplify shrinkage stress accumulation",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "age dependent",
      "evidence_count": 691,
      "route_id": "route-213"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "High paste and heat release amplify shrinkage stress accumulation",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "affects",
      "evidence_count": 691,
      "route_id": "route-214"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "High paste and heat release amplify shrinkage stress accumulation",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires curing control",
      "evidence_count": 990,
      "route_id": "route-215"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "High paste and heat release amplify shrinkage stress accumulation",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires workability balance",
      "evidence_count": 762,
      "route_id": "route-216"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "Lower clinker factor",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires strength normalization",
      "evidence_count": 379,
      "route_id": "route-222"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "Lower clinker factor",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "supports service-life benefit",
      "evidence_count": 220,
      "route_id": "route-223"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "Lower clinker factor",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires shrinkage-cracking check",
      "evidence_count": 181,
      "route_id": "route-224"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "Lower clinker factor",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires curing-compatible design",
      "evidence_count": 266,
      "route_id": "route-225"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "Lower clinker factor",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires service-life justification",
      "evidence_count": 186,
      "route_id": "route-226"
    },
    {
      "feature": "Clinker-rich binder fraction",
      "mechanism": "Controls heat release and clinker-related paste volume",
      "consequence": "Lower clinker factor",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces binder only if workability is retained",
      "evidence_count": 242,
      "route_id": "route-227"
    }
  ],
  "coarseaggregate": [
    {
      "feature": "Strong aggregate",
      "mechanism": "Provides load skeleton",
      "consequence": "Composite stiffness rises",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 469,
      "route_id": "route-049"
    },
    {
      "feature": "Rough texture",
      "mechanism": "Improves interlock",
      "consequence": "Bond strengthens",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 49,
      "route_id": "route-050"
    },
    {
      "feature": "Angular shape",
      "mechanism": "Raises friction",
      "consequence": "Rearrangement worsens",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 3,
      "route_id": "route-051"
    },
    {
      "feature": "Rounded gravel",
      "mechanism": "Reduces resistance",
      "consequence": "Flow improves",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "increase",
      "evidence_count": 63,
      "route_id": "route-052"
    },
    {
      "feature": "Poor grading",
      "mechanism": "Creates skeleton voids",
      "consequence": "Paste need rises",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 25,
      "route_id": "route-053"
    },
    {
      "feature": "High coarse aggregate volume",
      "mechanism": "Rigid aggregate restrains paste deformation",
      "consequence": "Total drying shrinkage decreases",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "decreases",
      "evidence_count": 156,
      "route_id": "route-094"
    },
    {
      "feature": "Large maximum particle size",
      "mechanism": "Paste volume requirement decreases",
      "consequence": "Shrinkable phase decreases",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "decreases",
      "evidence_count": 56,
      "route_id": "route-095"
    },
    {
      "feature": "Coarse aggregate volume and stiffness",
      "mechanism": "Forms a rigid paste-restraining aggregate skeleton",
      "consequence": "Lower total shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 156,
      "route_id": "route-179"
    },
    {
      "feature": "Coarse aggregate volume and stiffness",
      "mechanism": "Forms a rigid paste-restraining aggregate skeleton",
      "consequence": "Lower total shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "controls",
      "evidence_count": 155,
      "route_id": "route-180"
    },
    {
      "feature": "Coarse aggregate volume and stiffness",
      "mechanism": "Forms a rigid paste-restraining aggregate skeleton",
      "consequence": "Lower total shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires validation",
      "evidence_count": 155,
      "route_id": "route-181"
    },
    {
      "feature": "Coarse aggregate volume and stiffness",
      "mechanism": "Forms a rigid paste-restraining aggregate skeleton",
      "consequence": "Lower carbon intensity",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces when optimized",
      "evidence_count": 154,
      "route_id": "route-185"
    },
    {
      "feature": "Coarse aggregate volume and stiffness",
      "mechanism": "Forms a rigid paste-restraining aggregate skeleton",
      "consequence": "Lower carbon intensity",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "trade-off dependent",
      "evidence_count": 153,
      "route_id": "route-186"
    },
    {
      "feature": "Coarse aggregate volume and stiffness",
      "mechanism": "Forms a rigid paste-restraining aggregate skeleton",
      "consequence": "Lower carbon intensity",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires life-cycle check",
      "evidence_count": 157,
      "route_id": "route-187"
    },
    {
      "feature": "Coarse aggregate volume and stiffness",
      "mechanism": "Forms a rigid paste-restraining aggregate skeleton",
      "consequence": "Lower total shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "age dependent",
      "evidence_count": 155,
      "route_id": "route-311"
    },
    {
      "feature": "Coarse aggregate volume and stiffness",
      "mechanism": "Forms a rigid paste-restraining aggregate skeleton",
      "consequence": "Lower total shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "affects",
      "evidence_count": 155,
      "route_id": "route-312"
    },
    {
      "feature": "Coarse aggregate volume and stiffness",
      "mechanism": "Forms a rigid paste-restraining aggregate skeleton",
      "consequence": "Lower total shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires curing control",
      "evidence_count": 156,
      "route_id": "route-313"
    },
    {
      "feature": "Coarse aggregate volume and stiffness",
      "mechanism": "Forms a rigid paste-restraining aggregate skeleton",
      "consequence": "Lower total shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires workability balance",
      "evidence_count": 155,
      "route_id": "route-314"
    },
    {
      "feature": "Coarse aggregate volume and stiffness",
      "mechanism": "Forms a rigid paste-restraining aggregate skeleton",
      "consequence": "Lower carbon intensity",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires strength normalization",
      "evidence_count": 162,
      "route_id": "route-322"
    },
    {
      "feature": "Coarse aggregate volume and stiffness",
      "mechanism": "Forms a rigid paste-restraining aggregate skeleton",
      "consequence": "Lower carbon intensity",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "supports service-life benefit",
      "evidence_count": 156,
      "route_id": "route-323"
    },
    {
      "feature": "Coarse aggregate volume and stiffness",
      "mechanism": "Forms a rigid paste-restraining aggregate skeleton",
      "consequence": "Lower carbon intensity",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires shrinkage-cracking check",
      "evidence_count": 153,
      "route_id": "route-324"
    },
    {
      "feature": "Coarse aggregate volume and stiffness",
      "mechanism": "Forms a rigid paste-restraining aggregate skeleton",
      "consequence": "Lower carbon intensity",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires service-life justification",
      "evidence_count": 156,
      "route_id": "route-325"
    }
  ],
  "fineaggregate": [
    {
      "feature": "Well graded sand",
      "mechanism": "Maximizes packing",
      "consequence": "Paste demand lowers",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 683,
      "route_id": "route-044"
    },
    {
      "feature": "Rounded sand",
      "mechanism": "Reduces friction",
      "consequence": "Mortar mobility improves",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "increase",
      "evidence_count": 284,
      "route_id": "route-045"
    },
    {
      "feature": "Angular sand",
      "mechanism": "Raises friction",
      "consequence": "Shear resistance rises",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 165,
      "route_id": "route-046"
    },
    {
      "feature": "Gap grading",
      "mechanism": "Creates packing voids",
      "consequence": "Paste demand rises",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 66,
      "route_id": "route-047"
    },
    {
      "feature": "Large specific surface",
      "mechanism": "Expands ITZ area",
      "consequence": "Weak zones increase",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "decrease",
      "evidence_count": 191,
      "route_id": "route-048"
    },
    {
      "feature": "Optimized sand grading",
      "mechanism": "Paste demand decreases",
      "consequence": "Drying shrinkage decreases",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "decreases",
      "evidence_count": 322,
      "route_id": "route-093"
    },
    {
      "feature": "Sand grading and angularity",
      "mechanism": "Controls packing density, mortar friction, paste demand, and local restraint",
      "consequence": "Lower drying shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 336,
      "route_id": "route-170"
    },
    {
      "feature": "Sand grading and angularity",
      "mechanism": "Controls packing density, mortar friction, paste demand, and local restraint",
      "consequence": "Lower drying shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "controls",
      "evidence_count": 335,
      "route_id": "route-171"
    },
    {
      "feature": "Sand grading and angularity",
      "mechanism": "Controls packing density, mortar friction, paste demand, and local restraint",
      "consequence": "Lower drying shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires validation",
      "evidence_count": 335,
      "route_id": "route-172"
    },
    {
      "feature": "Sand grading and angularity",
      "mechanism": "Controls packing density, mortar friction, paste demand, and local restraint",
      "consequence": "Lower binder demand",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces when optimized",
      "evidence_count": 162,
      "route_id": "route-176"
    },
    {
      "feature": "Sand grading and angularity",
      "mechanism": "Controls packing density, mortar friction, paste demand, and local restraint",
      "consequence": "Lower binder demand",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "trade-off dependent",
      "evidence_count": 150,
      "route_id": "route-177"
    },
    {
      "feature": "Sand grading and angularity",
      "mechanism": "Controls packing density, mortar friction, paste demand, and local restraint",
      "consequence": "Lower binder demand",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires life-cycle check",
      "evidence_count": 204,
      "route_id": "route-178"
    },
    {
      "feature": "Sand grading and angularity",
      "mechanism": "Controls packing density, mortar friction, paste demand, and local restraint",
      "consequence": "Lower drying shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "age dependent",
      "evidence_count": 335,
      "route_id": "route-299"
    },
    {
      "feature": "Sand grading and angularity",
      "mechanism": "Controls packing density, mortar friction, paste demand, and local restraint",
      "consequence": "Lower drying shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "affects",
      "evidence_count": 335,
      "route_id": "route-300"
    },
    {
      "feature": "Sand grading and angularity",
      "mechanism": "Controls packing density, mortar friction, paste demand, and local restraint",
      "consequence": "Lower drying shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires curing control",
      "evidence_count": 347,
      "route_id": "route-301"
    },
    {
      "feature": "Sand grading and angularity",
      "mechanism": "Controls packing density, mortar friction, paste demand, and local restraint",
      "consequence": "Lower drying shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires workability balance",
      "evidence_count": 339,
      "route_id": "route-302"
    },
    {
      "feature": "Sand grading and angularity",
      "mechanism": "Controls packing density, mortar friction, paste demand, and local restraint",
      "consequence": "Lower binder demand",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires strength normalization",
      "evidence_count": 276,
      "route_id": "route-308"
    },
    {
      "feature": "Sand grading and angularity",
      "mechanism": "Controls packing density, mortar friction, paste demand, and local restraint",
      "consequence": "Lower binder demand",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "supports service-life benefit",
      "evidence_count": 183,
      "route_id": "route-309"
    },
    {
      "feature": "Sand grading and angularity",
      "mechanism": "Controls packing density, mortar friction, paste demand, and local restraint",
      "consequence": "Lower binder demand",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires shrinkage-cracking check",
      "evidence_count": 155,
      "route_id": "route-310"
    }
  ],
  "flyash": [
    {
      "feature": "Spherical particles",
      "mechanism": "Ball-bearing effect",
      "consequence": "Lower interparticle friction",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "increase",
      "evidence_count": 52,
      "route_id": "route-012"
    },
    {
      "feature": "Aluminosilicate glass",
      "mechanism": "Late pozzolanic reaction",
      "consequence": "Secondary C-A-S-H fills pores",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase at late ages",
      "evidence_count": 221,
      "route_id": "route-013"
    },
    {
      "feature": "High replacement",
      "mechanism": "Dilutes clinker early",
      "consequence": "Lower early hydrate volume",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "decrease at early ages",
      "evidence_count": 295,
      "route_id": "route-014"
    },
    {
      "feature": "Fine fraction",
      "mechanism": "Improves packing",
      "consequence": "Paste voids reduce",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 355,
      "route_id": "route-015"
    },
    {
      "feature": "Unburned carbon",
      "mechanism": "Adsorbs PCE",
      "consequence": "Dispersion loss",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 5,
      "route_id": "route-016"
    },
    {
      "feature": "Low water-demand morphology",
      "mechanism": "Reduces viscosity",
      "consequence": "Pumpability improves",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "increase",
      "evidence_count": 106,
      "route_id": "route-017"
    },
    {
      "feature": "Late pore refinement",
      "mechanism": "Internal humidity evolution changes after continued reaction",
      "consequence": "Late-age shrinkage requires dosage validation",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 155,
      "route_id": "route-072"
    },
    {
      "feature": "Clinker replacement by fly ash",
      "mechanism": "Industrial by-product displaces high-carbon cement",
      "consequence": "Embodied carbon decreases",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "decreases",
      "evidence_count": 247,
      "route_id": "route-074"
    },
    {
      "feature": "Late strength contribution",
      "mechanism": "Pozzolanic C-A-S-H recovers strength at lower clinker content",
      "consequence": "Carbon efficiency improves at later ages",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "decreases at later ages",
      "evidence_count": 195,
      "route_id": "route-075"
    },
    {
      "feature": "Excess replacement",
      "mechanism": "Binder reactivity becomes insufficient at early age",
      "consequence": "Extra binder or curing may offset carbon benefit",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "dosage dependent",
      "evidence_count": 155,
      "route_id": "route-076"
    },
    {
      "feature": "Spherical low-clinker pozzolanic particles",
      "mechanism": "Reduces water demand and later consumes portlandite to form secondary gel",
      "consequence": "Lower early-shrinkage stress",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 209,
      "route_id": "route-118"
    },
    {
      "feature": "Spherical low-clinker pozzolanic particles",
      "mechanism": "Reduces water demand and later consumes portlandite to form secondary gel",
      "consequence": "Lower early-shrinkage stress",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "controls",
      "evidence_count": 203,
      "route_id": "route-119"
    },
    {
      "feature": "Spherical low-clinker pozzolanic particles",
      "mechanism": "Reduces water demand and later consumes portlandite to form secondary gel",
      "consequence": "Lower early-shrinkage stress",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires validation",
      "evidence_count": 203,
      "route_id": "route-120"
    },
    {
      "feature": "Spherical low-clinker pozzolanic particles",
      "mechanism": "Reduces water demand and later consumes portlandite to form secondary gel",
      "consequence": "Clinker substitution lowers carbon intensity at comparable strength",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces when optimized",
      "evidence_count": 224,
      "route_id": "route-123"
    },
    {
      "feature": "Spherical low-clinker pozzolanic particles",
      "mechanism": "Reduces water demand and later consumes portlandite to form secondary gel",
      "consequence": "Clinker substitution lowers carbon intensity at comparable strength",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "trade-off dependent",
      "evidence_count": 224,
      "route_id": "route-124"
    },
    {
      "feature": "Spherical low-clinker pozzolanic particles",
      "mechanism": "Reduces water demand and later consumes portlandite to form secondary gel",
      "consequence": "Clinker substitution lowers carbon intensity at comparable strength",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires life-cycle check",
      "evidence_count": 243,
      "route_id": "route-125"
    },
    {
      "feature": "Spherical low-clinker pozzolanic particles",
      "mechanism": "Reduces water demand and later consumes portlandite to form secondary gel",
      "consequence": "Lower early-shrinkage stress",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "age dependent",
      "evidence_count": 203,
      "route_id": "route-228"
    },
    {
      "feature": "Spherical low-clinker pozzolanic particles",
      "mechanism": "Reduces water demand and later consumes portlandite to form secondary gel",
      "consequence": "Lower early-shrinkage stress",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "affects",
      "evidence_count": 203,
      "route_id": "route-229"
    },
    {
      "feature": "Spherical low-clinker pozzolanic particles",
      "mechanism": "Reduces water demand and later consumes portlandite to form secondary gel",
      "consequence": "Lower early-shrinkage stress",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires curing control",
      "evidence_count": 232,
      "route_id": "route-230"
    },
    {
      "feature": "Spherical low-clinker pozzolanic particles",
      "mechanism": "Reduces water demand and later consumes portlandite to form secondary gel",
      "consequence": "Clinker substitution lowers carbon intensity at comparable strength",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires strength normalization",
      "evidence_count": 224,
      "route_id": "route-235"
    },
    {
      "feature": "Spherical low-clinker pozzolanic particles",
      "mechanism": "Reduces water demand and later consumes portlandite to form secondary gel",
      "consequence": "Clinker substitution lowers carbon intensity at comparable strength",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "supports service-life benefit",
      "evidence_count": 243,
      "route_id": "route-236"
    },
    {
      "feature": "Spherical low-clinker pozzolanic particles",
      "mechanism": "Reduces water demand and later consumes portlandite to form secondary gel",
      "consequence": "Clinker substitution lowers carbon intensity at comparable strength",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires shrinkage-cracking check",
      "evidence_count": 232,
      "route_id": "route-237"
    },
    {
      "feature": "Spherical low-clinker pozzolanic particles",
      "mechanism": "Reduces water demand and later consumes portlandite to form secondary gel",
      "consequence": "Clinker substitution lowers carbon intensity at comparable strength",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires curing-compatible design",
      "evidence_count": 245,
      "route_id": "route-238"
    },
    {
      "feature": "Spherical low-clinker pozzolanic particles",
      "mechanism": "Reduces water demand and later consumes portlandite to form secondary gel",
      "consequence": "Clinker substitution lowers carbon intensity at comparable strength",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires service-life justification",
      "evidence_count": 239,
      "route_id": "route-239"
    },
    {
      "feature": "Spherical low-clinker pozzolanic particles",
      "mechanism": "Reduces water demand and later consumes portlandite to form secondary gel",
      "consequence": "Clinker substitution lowers carbon intensity at comparable strength",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces binder only if workability is retained",
      "evidence_count": 239,
      "route_id": "route-240"
    }
  ],
  "limestone": [
    {
      "feature": "Fine calcite",
      "mechanism": "Filler packing",
      "consequence": "Initial voids reduce",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase at low dosage",
      "evidence_count": 314,
      "route_id": "route-028"
    },
    {
      "feature": "Calcite surface",
      "mechanism": "Nucleates C-S-H",
      "consequence": "Early hydrates form",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 174,
      "route_id": "route-029"
    },
    {
      "feature": "Optimized grading",
      "mechanism": "Improves lubrication",
      "consequence": "Paste friction reduces",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "increase",
      "evidence_count": 20,
      "route_id": "route-030"
    },
    {
      "feature": "High replacement",
      "mechanism": "Dilutes binder",
      "consequence": "Gel volume lowers",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "decrease",
      "evidence_count": 101,
      "route_id": "route-031"
    },
    {
      "feature": "Excess ultrafines",
      "mechanism": "Raises surface demand",
      "consequence": "Viscosity rises",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 33,
      "route_id": "route-032"
    },
    {
      "feature": "Soft mineral phase",
      "mechanism": "Weakens diluted skeleton",
      "consequence": "Load transfer drops",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "decrease",
      "evidence_count": 76,
      "route_id": "route-033"
    },
    {
      "feature": "Excess limestone powder",
      "mechanism": "Binder dilution and high fines alter water demand",
      "consequence": "Shrinkage benefit may reverse",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 116,
      "route_id": "route-087"
    },
    {
      "feature": "Excessive dilution",
      "mechanism": "Strength loss requires compensating cement",
      "consequence": "Carbon saving is reduced",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "dosage dependent",
      "evidence_count": 196,
      "route_id": "route-088"
    },
    {
      "feature": "Calcite filler and nucleation surface",
      "mechanism": "Improves packing and early nucleation",
      "consequence": "Reduced paste demand",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 47,
      "route_id": "route-144"
    },
    {
      "feature": "Calcite filler and nucleation surface",
      "mechanism": "Improves packing and early nucleation",
      "consequence": "Reduced paste demand",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "controls",
      "evidence_count": 43,
      "route_id": "route-145"
    },
    {
      "feature": "Calcite filler and nucleation surface",
      "mechanism": "Improves packing and early nucleation",
      "consequence": "Lower embodied carbon",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces when optimized",
      "evidence_count": 108,
      "route_id": "route-149"
    },
    {
      "feature": "Calcite filler and nucleation surface",
      "mechanism": "Improves packing and early nucleation",
      "consequence": "Lower embodied carbon",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "trade-off dependent",
      "evidence_count": 102,
      "route_id": "route-150"
    },
    {
      "feature": "Calcite filler and nucleation surface",
      "mechanism": "Improves packing and early nucleation",
      "consequence": "Lower embodied carbon",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires life-cycle check",
      "evidence_count": 125,
      "route_id": "route-151"
    },
    {
      "feature": "Calcite filler and nucleation surface",
      "mechanism": "Improves packing and early nucleation",
      "consequence": "Reduced paste demand",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "age dependent",
      "evidence_count": 43,
      "route_id": "route-265"
    },
    {
      "feature": "Calcite filler and nucleation surface",
      "mechanism": "Improves packing and early nucleation",
      "consequence": "Reduced paste demand",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "affects",
      "evidence_count": 43,
      "route_id": "route-266"
    },
    {
      "feature": "Calcite filler and nucleation surface",
      "mechanism": "Improves packing and early nucleation",
      "consequence": "Lower embodied carbon",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires strength normalization",
      "evidence_count": 166,
      "route_id": "route-271"
    },
    {
      "feature": "Calcite filler and nucleation surface",
      "mechanism": "Improves packing and early nucleation",
      "consequence": "Lower embodied carbon",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "supports service-life benefit",
      "evidence_count": 123,
      "route_id": "route-272"
    },
    {
      "feature": "Calcite filler and nucleation surface",
      "mechanism": "Improves packing and early nucleation",
      "consequence": "Lower embodied carbon",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires shrinkage-cracking check",
      "evidence_count": 111,
      "route_id": "route-273"
    },
    {
      "feature": "Calcite filler and nucleation surface",
      "mechanism": "Improves packing and early nucleation",
      "consequence": "Lower embodied carbon",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires service-life justification",
      "evidence_count": 116,
      "route_id": "route-274"
    },
    {
      "feature": "Calcite filler and nucleation surface",
      "mechanism": "Improves packing and early nucleation",
      "consequence": "Lower embodied carbon",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces binder only if workability is retained",
      "evidence_count": 138,
      "route_id": "route-275"
    }
  ],
  "metakaolin": [
    {
      "feature": "Reactive aluminosilicate",
      "mechanism": "Fast pozzolanic reaction",
      "consequence": "C-A-S-H forms",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 86,
      "route_id": "route-034"
    },
    {
      "feature": "Plate-like fines",
      "mechanism": "Raises friction",
      "consequence": "Mobility lowers",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 2,
      "route_id": "route-035"
    },
    {
      "feature": "Filler nucleation",
      "mechanism": "Builds dense matrix",
      "consequence": "Strength improves",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 171,
      "route_id": "route-036"
    },
    {
      "feature": "Excess dosage",
      "mechanism": "Consumes calcium rapidly",
      "consequence": "Unreacted particles remain",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "decrease",
      "evidence_count": 32,
      "route_id": "route-037"
    },
    {
      "feature": "PCE support",
      "mechanism": "Offsets surface demand",
      "consequence": "Suspension stabilizes",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "increase",
      "evidence_count": 9,
      "route_id": "route-038"
    },
    {
      "feature": "Optimized PCE and curing",
      "mechanism": "Dispersion and moisture supply offset water demand",
      "consequence": "Shrinkage risk can be controlled",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "conditionally decreases",
      "evidence_count": 36,
      "route_id": "route-089"
    },
    {
      "feature": "High calcination energy",
      "mechanism": "Processing energy is higher than some by-products",
      "consequence": "Net carbon benefit requires source-specific accounting",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires validation",
      "evidence_count": 38,
      "route_id": "route-090"
    },
    {
      "feature": "Reactive aluminosilicate platelets",
      "mechanism": "Refines pores through pozzolanic and alumina reactions",
      "consequence": "Higher early-shrinkage sensitivity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 31,
      "route_id": "route-152"
    },
    {
      "feature": "Reactive aluminosilicate platelets",
      "mechanism": "Refines pores through pozzolanic and alumina reactions",
      "consequence": "Higher early-shrinkage sensitivity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "controls",
      "evidence_count": 31,
      "route_id": "route-153"
    },
    {
      "feature": "Reactive aluminosilicate platelets",
      "mechanism": "Refines pores through pozzolanic and alumina reactions",
      "consequence": "Higher early-shrinkage sensitivity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires validation",
      "evidence_count": 31,
      "route_id": "route-154"
    },
    {
      "feature": "Reactive aluminosilicate platelets",
      "mechanism": "Refines pores through pozzolanic and alumina reactions",
      "consequence": "Partial clinker replacement",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces when optimized",
      "evidence_count": 39,
      "route_id": "route-158"
    },
    {
      "feature": "Reactive aluminosilicate platelets",
      "mechanism": "Refines pores through pozzolanic and alumina reactions",
      "consequence": "Partial clinker replacement",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "trade-off dependent",
      "evidence_count": 37,
      "route_id": "route-159"
    },
    {
      "feature": "Reactive aluminosilicate platelets",
      "mechanism": "Refines pores through pozzolanic and alumina reactions",
      "consequence": "Partial clinker replacement",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires life-cycle check",
      "evidence_count": 44,
      "route_id": "route-160"
    },
    {
      "feature": "Reactive aluminosilicate platelets",
      "mechanism": "Refines pores through pozzolanic and alumina reactions",
      "consequence": "Higher early-shrinkage sensitivity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "age dependent",
      "evidence_count": 31,
      "route_id": "route-276"
    },
    {
      "feature": "Reactive aluminosilicate platelets",
      "mechanism": "Refines pores through pozzolanic and alumina reactions",
      "consequence": "Higher early-shrinkage sensitivity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "affects",
      "evidence_count": 31,
      "route_id": "route-277"
    },
    {
      "feature": "Reactive aluminosilicate platelets",
      "mechanism": "Refines pores through pozzolanic and alumina reactions",
      "consequence": "Higher early-shrinkage sensitivity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires curing control",
      "evidence_count": 39,
      "route_id": "route-278"
    },
    {
      "feature": "Reactive aluminosilicate platelets",
      "mechanism": "Refines pores through pozzolanic and alumina reactions",
      "consequence": "Higher early-shrinkage sensitivity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires workability balance",
      "evidence_count": 39,
      "route_id": "route-279"
    },
    {
      "feature": "Reactive aluminosilicate platelets",
      "mechanism": "Refines pores through pozzolanic and alumina reactions",
      "consequence": "Partial clinker replacement",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires strength normalization",
      "evidence_count": 53,
      "route_id": "route-285"
    },
    {
      "feature": "Reactive aluminosilicate platelets",
      "mechanism": "Refines pores through pozzolanic and alumina reactions",
      "consequence": "Partial clinker replacement",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "supports service-life benefit",
      "evidence_count": 40,
      "route_id": "route-286"
    },
    {
      "feature": "Reactive aluminosilicate platelets",
      "mechanism": "Refines pores through pozzolanic and alumina reactions",
      "consequence": "Partial clinker replacement",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires shrinkage-cracking check",
      "evidence_count": 38,
      "route_id": "route-287"
    }
  ],
  "ricehuskash": [
    {
      "feature": "Amorphous biogenic silica",
      "mechanism": "Pozzolanic CH consumption",
      "consequence": "Secondary C-S-H",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 45,
      "route_id": "route-039"
    },
    {
      "feature": "Cellular porosity",
      "mechanism": "Absorbs water",
      "consequence": "Free water decreases",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 5,
      "route_id": "route-040"
    },
    {
      "feature": "High carbon",
      "mechanism": "Adsorbs admixture",
      "consequence": "Dispersion unstable",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 2,
      "route_id": "route-041"
    },
    {
      "feature": "Over-burned ash",
      "mechanism": "Loses amorphous phase",
      "consequence": "Reactivity drops",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "decrease",
      "evidence_count": 9,
      "route_id": "route-042"
    },
    {
      "feature": "High replacement",
      "mechanism": "Dilutes clinker early",
      "consequence": "Early hydrates insufficient",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "decrease at early ages",
      "evidence_count": 17,
      "route_id": "route-043"
    },
    {
      "feature": "Pozzolanic pore refinement",
      "mechanism": "Transport pores become finer",
      "consequence": "Drying shrinkage response becomes dosage dependent",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 10,
      "route_id": "route-091"
    },
    {
      "feature": "Controlled burning demand",
      "mechanism": "Processing conditions influence energy burden",
      "consequence": "Net carbon depends on production route",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires validation",
      "evidence_count": 29,
      "route_id": "route-092"
    },
    {
      "feature": "Porous amorphous silica ash",
      "mechanism": "Combines pozzolanic gel formation with internal-water regulation",
      "consequence": "Internal-curing balance",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 8,
      "route_id": "route-161"
    },
    {
      "feature": "Porous amorphous silica ash",
      "mechanism": "Combines pozzolanic gel formation with internal-water regulation",
      "consequence": "Internal-curing balance",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "controls",
      "evidence_count": 8,
      "route_id": "route-162"
    },
    {
      "feature": "Porous amorphous silica ash",
      "mechanism": "Combines pozzolanic gel formation with internal-water regulation",
      "consequence": "Internal-curing balance",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires validation",
      "evidence_count": 8,
      "route_id": "route-163"
    },
    {
      "feature": "Porous amorphous silica ash",
      "mechanism": "Combines pozzolanic gel formation with internal-water regulation",
      "consequence": "Lower clinker demand",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces when optimized",
      "evidence_count": 18,
      "route_id": "route-167"
    },
    {
      "feature": "Porous amorphous silica ash",
      "mechanism": "Combines pozzolanic gel formation with internal-water regulation",
      "consequence": "Lower clinker demand",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "trade-off dependent",
      "evidence_count": 15,
      "route_id": "route-168"
    },
    {
      "feature": "Porous amorphous silica ash",
      "mechanism": "Combines pozzolanic gel formation with internal-water regulation",
      "consequence": "Lower clinker demand",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires life-cycle check",
      "evidence_count": 21,
      "route_id": "route-169"
    },
    {
      "feature": "Porous amorphous silica ash",
      "mechanism": "Combines pozzolanic gel formation with internal-water regulation",
      "consequence": "Internal-curing balance",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "age dependent",
      "evidence_count": 8,
      "route_id": "route-288"
    },
    {
      "feature": "Porous amorphous silica ash",
      "mechanism": "Combines pozzolanic gel formation with internal-water regulation",
      "consequence": "Internal-curing balance",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "affects",
      "evidence_count": 8,
      "route_id": "route-289"
    },
    {
      "feature": "Porous amorphous silica ash",
      "mechanism": "Combines pozzolanic gel formation with internal-water regulation",
      "consequence": "Internal-curing balance",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires curing control",
      "evidence_count": 8,
      "route_id": "route-290"
    },
    {
      "feature": "Porous amorphous silica ash",
      "mechanism": "Combines pozzolanic gel formation with internal-water regulation",
      "consequence": "Lower clinker demand",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires strength normalization",
      "evidence_count": 36,
      "route_id": "route-295"
    },
    {
      "feature": "Porous amorphous silica ash",
      "mechanism": "Combines pozzolanic gel formation with internal-water regulation",
      "consequence": "Lower clinker demand",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "supports service-life benefit",
      "evidence_count": 21,
      "route_id": "route-296"
    },
    {
      "feature": "Porous amorphous silica ash",
      "mechanism": "Combines pozzolanic gel formation with internal-water regulation",
      "consequence": "Lower clinker demand",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires shrinkage-cracking check",
      "evidence_count": 16,
      "route_id": "route-297"
    },
    {
      "feature": "Porous amorphous silica ash",
      "mechanism": "Combines pozzolanic gel formation with internal-water regulation",
      "consequence": "Lower clinker demand",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires service-life justification",
      "evidence_count": 19,
      "route_id": "route-298"
    }
  ],
  "silicafume": [
    {
      "feature": "Amorphous SiO2",
      "mechanism": "Rapid pozzolanic reaction",
      "consequence": "Low-Ca/Si C-S-H",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 148,
      "route_id": "route-018"
    },
    {
      "feature": "Submicron size",
      "mechanism": "Microfiller packing",
      "consequence": "Large pores reduce",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 140,
      "route_id": "route-019"
    },
    {
      "feature": "High surface area",
      "mechanism": "Raises water demand",
      "consequence": "Viscosity rises",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 95,
      "route_id": "route-020"
    },
    {
      "feature": "Nucleation surface",
      "mechanism": "Accelerates precipitation",
      "consequence": "Earlier hydrate percolation",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 32,
      "route_id": "route-021"
    },
    {
      "feature": "No superplasticizer",
      "mechanism": "Promotes bridging flocs",
      "consequence": "Free water decreases",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 60,
      "route_id": "route-022"
    },
    {
      "feature": "High water demand",
      "mechanism": "Additional PCE or water is needed for dispersion",
      "consequence": "Shrinkage response depends on water adjustment",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 132,
      "route_id": "route-077"
    },
    {
      "feature": "Dense ITZ",
      "mechanism": "Interfacial porosity and local water loss paths are reduced",
      "consequence": "Drying shrinkage cracking may be restrained when cured",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "conditionally decreases cracking",
      "evidence_count": 182,
      "route_id": "route-078"
    },
    {
      "feature": "Overuse with high PCE demand",
      "mechanism": "Admixture demand and processing burden increase",
      "consequence": "Net carbon benefit becomes dosage dependent",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "dosage dependent",
      "evidence_count": 21,
      "route_id": "route-080"
    },
    {
      "feature": "Ultrafine reactive silica",
      "mechanism": "Densifies the ITZ and pore network but increases surface water demand",
      "consequence": "Higher autogenous-shrinkage sensitivity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 201,
      "route_id": "route-126"
    },
    {
      "feature": "Ultrafine reactive silica",
      "mechanism": "Densifies the ITZ and pore network but increases surface water demand",
      "consequence": "Higher autogenous-shrinkage sensitivity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "controls",
      "evidence_count": 201,
      "route_id": "route-127"
    },
    {
      "feature": "Ultrafine reactive silica",
      "mechanism": "Densifies the ITZ and pore network but increases surface water demand",
      "consequence": "Higher autogenous-shrinkage sensitivity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires validation",
      "evidence_count": 201,
      "route_id": "route-128"
    },
    {
      "feature": "Ultrafine reactive silica",
      "mechanism": "Densifies the ITZ and pore network but increases surface water demand",
      "consequence": "Strength-efficiency gain",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces when optimized",
      "evidence_count": 118,
      "route_id": "route-132"
    },
    {
      "feature": "Ultrafine reactive silica",
      "mechanism": "Densifies the ITZ and pore network but increases surface water demand",
      "consequence": "Strength-efficiency gain",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "trade-off dependent",
      "evidence_count": 118,
      "route_id": "route-133"
    },
    {
      "feature": "Ultrafine reactive silica",
      "mechanism": "Densifies the ITZ and pore network but increases surface water demand",
      "consequence": "Strength-efficiency gain",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires life-cycle check",
      "evidence_count": 118,
      "route_id": "route-134"
    },
    {
      "feature": "Ultrafine reactive silica",
      "mechanism": "Densifies the ITZ and pore network but increases surface water demand",
      "consequence": "Higher autogenous-shrinkage sensitivity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "age dependent",
      "evidence_count": 201,
      "route_id": "route-241"
    },
    {
      "feature": "Ultrafine reactive silica",
      "mechanism": "Densifies the ITZ and pore network but increases surface water demand",
      "consequence": "Higher autogenous-shrinkage sensitivity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "affects",
      "evidence_count": 201,
      "route_id": "route-242"
    },
    {
      "feature": "Ultrafine reactive silica",
      "mechanism": "Densifies the ITZ and pore network but increases surface water demand",
      "consequence": "Higher autogenous-shrinkage sensitivity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires curing control",
      "evidence_count": 201,
      "route_id": "route-243"
    },
    {
      "feature": "Ultrafine reactive silica",
      "mechanism": "Densifies the ITZ and pore network but increases surface water demand",
      "consequence": "Higher autogenous-shrinkage sensitivity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires workability balance",
      "evidence_count": 201,
      "route_id": "route-244"
    },
    {
      "feature": "Ultrafine reactive silica",
      "mechanism": "Densifies the ITZ and pore network but increases surface water demand",
      "consequence": "Strength-efficiency gain",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires strength normalization",
      "evidence_count": 118,
      "route_id": "route-251"
    },
    {
      "feature": "Ultrafine reactive silica",
      "mechanism": "Densifies the ITZ and pore network but increases surface water demand",
      "consequence": "Strength-efficiency gain",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "supports service-life benefit",
      "evidence_count": 118,
      "route_id": "route-252"
    }
  ],
  "steelfiber": [
    {
      "feature": "High aspect ratio",
      "mechanism": "Bridges cracks",
      "consequence": "Localization delays",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 269,
      "route_id": "route-060"
    },
    {
      "feature": "Hooked ends",
      "mechanism": "Raises pull-out resistance",
      "consequence": "Energy dissipation rises",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 231,
      "route_id": "route-061"
    },
    {
      "feature": "High volume fraction",
      "mechanism": "Increases entanglement",
      "consequence": "Flow resistance rises",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 138,
      "route_id": "route-062"
    },
    {
      "feature": "Dense matrix around fiber",
      "mechanism": "Improves bond",
      "consequence": "Stress transfer rises",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 847,
      "route_id": "route-063"
    },
    {
      "feature": "Poor dispersion",
      "mechanism": "Creates weak clusters",
      "consequence": "Compaction defects",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "decrease",
      "evidence_count": 14,
      "route_id": "route-064"
    },
    {
      "feature": "Long stiff inclusions",
      "mechanism": "Obstruct flow",
      "consequence": "Apparent viscosity rises",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 29,
      "route_id": "route-065"
    },
    {
      "feature": "Steel-fibre dosage, aspect ratio, and dispersion",
      "mechanism": "Bridges cracks, transfers post-cracking stress, and increases fracture energy",
      "consequence": "Lower restrained crack width",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 176,
      "route_id": "route-196"
    },
    {
      "feature": "Steel-fibre dosage, aspect ratio, and dispersion",
      "mechanism": "Bridges cracks, transfers post-cracking stress, and increases fracture energy",
      "consequence": "Lower restrained crack width",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "controls",
      "evidence_count": 176,
      "route_id": "route-197"
    },
    {
      "feature": "Steel-fibre dosage, aspect ratio, and dispersion",
      "mechanism": "Bridges cracks, transfers post-cracking stress, and increases fracture energy",
      "consequence": "Service-life carbon trade-off",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces when optimized",
      "evidence_count": 81,
      "route_id": "route-201"
    },
    {
      "feature": "Steel-fibre dosage, aspect ratio, and dispersion",
      "mechanism": "Bridges cracks, transfers post-cracking stress, and increases fracture energy",
      "consequence": "Service-life carbon trade-off",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "trade-off dependent",
      "evidence_count": 81,
      "route_id": "route-202"
    },
    {
      "feature": "Steel-fibre dosage, aspect ratio, and dispersion",
      "mechanism": "Bridges cracks, transfers post-cracking stress, and increases fracture energy",
      "consequence": "Service-life carbon trade-off",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires life-cycle check",
      "evidence_count": 81,
      "route_id": "route-203"
    },
    {
      "feature": "Steel-fibre dosage, aspect ratio, and dispersion",
      "mechanism": "Bridges cracks, transfers post-cracking stress, and increases fracture energy",
      "consequence": "Lower restrained crack width",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "age dependent",
      "evidence_count": 176,
      "route_id": "route-333"
    },
    {
      "feature": "Steel-fibre dosage, aspect ratio, and dispersion",
      "mechanism": "Bridges cracks, transfers post-cracking stress, and increases fracture energy",
      "consequence": "Lower restrained crack width",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "affects",
      "evidence_count": 176,
      "route_id": "route-334"
    },
    {
      "feature": "Steel-fibre dosage, aspect ratio, and dispersion",
      "mechanism": "Bridges cracks, transfers post-cracking stress, and increases fracture energy",
      "consequence": "Service-life carbon trade-off",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires strength normalization",
      "evidence_count": 85,
      "route_id": "route-342"
    },
    {
      "feature": "Steel-fibre dosage, aspect ratio, and dispersion",
      "mechanism": "Bridges cracks, transfers post-cracking stress, and increases fracture energy",
      "consequence": "Service-life carbon trade-off",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "supports service-life benefit",
      "evidence_count": 81,
      "route_id": "route-343"
    },
    {
      "feature": "Steel-fibre dosage, aspect ratio, and dispersion",
      "mechanism": "Bridges cracks, transfers post-cracking stress, and increases fracture energy",
      "consequence": "Service-life carbon trade-off",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires shrinkage-cracking check",
      "evidence_count": 82,
      "route_id": "route-344"
    }
  ],
  "superplasticizer": [
    {
      "feature": "PCE comb polymer",
      "mechanism": "Steric dispersion",
      "consequence": "Yield stress lowers",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "increase",
      "evidence_count": 278,
      "route_id": "route-054"
    },
    {
      "feature": "Water reduction",
      "mechanism": "Maintains flow at low w/b",
      "consequence": "Capillary pores reduce",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 357,
      "route_id": "route-055"
    },
    {
      "feature": "Clay contamination",
      "mechanism": "Consumes polymer",
      "consequence": "Dispersion lost",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 69,
      "route_id": "route-056"
    },
    {
      "feature": "Delayed addition",
      "mechanism": "Improves adsorption efficiency",
      "consequence": "Slump retention",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "increase",
      "evidence_count": 245,
      "route_id": "route-057"
    },
    {
      "feature": "Compatibility",
      "mechanism": "Controls competitive adsorption",
      "consequence": "Stable suspension",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "increase",
      "evidence_count": 49,
      "route_id": "route-058"
    },
    {
      "feature": "Overdosage",
      "mechanism": "Destabilizes suspension",
      "consequence": "Weak zones form",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "decrease",
      "evidence_count": 5,
      "route_id": "route-059"
    },
    {
      "feature": "Admixture production burden",
      "mechanism": "Polymer dosage adds embodied impact",
      "consequence": "Net carbon benefit depends on dosage",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "dosage dependent",
      "evidence_count": 38,
      "route_id": "route-096"
    },
    {
      "feature": "PCE dispersion and slump retention",
      "mechanism": "Reduces yield stress through binder-particle dispersion",
      "consequence": "Lower paste porosity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 51,
      "route_id": "route-188"
    },
    {
      "feature": "PCE dispersion and slump retention",
      "mechanism": "Reduces yield stress through binder-particle dispersion",
      "consequence": "Lower paste porosity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "controls",
      "evidence_count": 40,
      "route_id": "route-189"
    },
    {
      "feature": "PCE dispersion and slump retention",
      "mechanism": "Reduces yield stress through binder-particle dispersion",
      "consequence": "Lower paste porosity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires validation",
      "evidence_count": 40,
      "route_id": "route-190"
    },
    {
      "feature": "PCE dispersion and slump retention",
      "mechanism": "Reduces yield stress through binder-particle dispersion",
      "consequence": "Lower binder content",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces when optimized",
      "evidence_count": 37,
      "route_id": "route-193"
    },
    {
      "feature": "PCE dispersion and slump retention",
      "mechanism": "Reduces yield stress through binder-particle dispersion",
      "consequence": "Lower binder content",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "trade-off dependent",
      "evidence_count": 40,
      "route_id": "route-194"
    },
    {
      "feature": "PCE dispersion and slump retention",
      "mechanism": "Reduces yield stress through binder-particle dispersion",
      "consequence": "Lower binder content",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires life-cycle check",
      "evidence_count": 45,
      "route_id": "route-195"
    },
    {
      "feature": "PCE dispersion and slump retention",
      "mechanism": "Reduces yield stress through binder-particle dispersion",
      "consequence": "Lower paste porosity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "age dependent",
      "evidence_count": 40,
      "route_id": "route-326"
    },
    {
      "feature": "PCE dispersion and slump retention",
      "mechanism": "Reduces yield stress through binder-particle dispersion",
      "consequence": "Lower paste porosity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "affects",
      "evidence_count": 40,
      "route_id": "route-327"
    },
    {
      "feature": "PCE dispersion and slump retention",
      "mechanism": "Reduces yield stress through binder-particle dispersion",
      "consequence": "Lower paste porosity",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires curing control",
      "evidence_count": 49,
      "route_id": "route-328"
    }
  ],
  "water": [
    {
      "feature": "Low water-to-binder ratio",
      "mechanism": "Reduces initial capillary water",
      "consequence": "Dense discontinuous pore network",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 1704,
      "route_id": "route-001"
    },
    {
      "feature": "Adequate curing water",
      "mechanism": "Sustains hydration",
      "consequence": "More binding gel",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 957,
      "route_id": "route-002"
    },
    {
      "feature": "Free water",
      "mechanism": "Lowers yield stress",
      "consequence": "Particle rearrangement improves",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "increase",
      "evidence_count": 270,
      "route_id": "route-003"
    },
    {
      "feature": "Insufficient water",
      "mechanism": "Raises friction",
      "consequence": "Poor mobility",
      "performance": "Flowability",
      "performance_key": "flowability",
      "relation": "decrease",
      "evidence_count": 37,
      "route_id": "route-004"
    },
    {
      "feature": "Ionic medium",
      "mechanism": "Promotes dissolution precipitation",
      "consequence": "Early C-S-H network",
      "performance": "Compressive strength",
      "performance_key": "compressive strength",
      "relation": "increase",
      "evidence_count": 47,
      "route_id": "route-005"
    },
    {
      "feature": "Excess mixing water",
      "mechanism": "Evaporable water volume increases",
      "consequence": "Drying shrinkage and capillary pore collapse increase",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "increases",
      "evidence_count": 358,
      "route_id": "route-066"
    },
    {
      "feature": "Curing water supply",
      "mechanism": "External curing offsets internal humidity drop",
      "consequence": "Early shrinkage gradient is reduced",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "decreases",
      "evidence_count": 345,
      "route_id": "route-067"
    },
    {
      "feature": "Effective water regulation",
      "mechanism": "Changes internal humidity, capillary pressure, and hydration continuity",
      "consequence": "Curing-sensitive shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "dosage dependent",
      "evidence_count": 352,
      "route_id": "route-100"
    },
    {
      "feature": "Effective water regulation",
      "mechanism": "Changes internal humidity, capillary pressure, and hydration continuity",
      "consequence": "Curing-sensitive shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "controls",
      "evidence_count": 350,
      "route_id": "route-101"
    },
    {
      "feature": "Effective water regulation",
      "mechanism": "Changes internal humidity, capillary pressure, and hydration continuity",
      "consequence": "Curing-sensitive shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires validation",
      "evidence_count": 350,
      "route_id": "route-102"
    },
    {
      "feature": "Effective water regulation",
      "mechanism": "Changes internal humidity, capillary pressure, and hydration continuity",
      "consequence": "Higher binder efficiency",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "reduces when optimized",
      "evidence_count": 116,
      "route_id": "route-106"
    },
    {
      "feature": "Effective water regulation",
      "mechanism": "Changes internal humidity, capillary pressure, and hydration continuity",
      "consequence": "Higher binder efficiency",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "trade-off dependent",
      "evidence_count": 116,
      "route_id": "route-107"
    },
    {
      "feature": "Effective water regulation",
      "mechanism": "Changes internal humidity, capillary pressure, and hydration continuity",
      "consequence": "Higher binder efficiency",
      "performance": "Low carbon",
      "performance_key": "low carbon",
      "relation": "requires life-cycle check",
      "evidence_count": 132,
      "route_id": "route-108"
    },
    {
      "feature": "Effective water regulation",
      "mechanism": "Changes internal humidity, capillary pressure, and hydration continuity",
      "consequence": "Curing-sensitive shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "age dependent",
      "evidence_count": 350,
      "route_id": "route-204"
    },
    {
      "feature": "Effective water regulation",
      "mechanism": "Changes internal humidity, capillary pressure, and hydration continuity",
      "consequence": "Curing-sensitive shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "affects",
      "evidence_count": 350,
      "route_id": "route-205"
    },
    {
      "feature": "Effective water regulation",
      "mechanism": "Changes internal humidity, capillary pressure, and hydration continuity",
      "consequence": "Curing-sensitive shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "indirectly affects",
      "evidence_count": 350,
      "route_id": "route-206"
    },
    {
      "feature": "Effective water regulation",
      "mechanism": "Changes internal humidity, capillary pressure, and hydration continuity",
      "consequence": "Curing-sensitive shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires curing control",
      "evidence_count": 352,
      "route_id": "route-207"
    },
    {
      "feature": "Effective water regulation",
      "mechanism": "Changes internal humidity, capillary pressure, and hydration continuity",
      "consequence": "Curing-sensitive shrinkage",
      "performance": "Shrinkage",
      "performance_key": "shrinkage",
      "relation": "requires workability balance",
      "evidence_count": 352,
      "route_id": "route-208"
    }
  ]
};
const EN_FIELD_LABELS = {
  "目标强度": "target strength",
  "龄期": "curing age",
  "性能优先级": "performance priority",
  "提及材料": "mentioned material",
  "必用材料": "required material",
  "允许材料": "allowed material",
  "禁用材料": "avoided material",
  "流动性约束": "flowability constraint",
  "收缩约束": "shrinkage constraint",
};
const EN_PRIORITY_LABELS = {
  strength: "strength requirement",
  low_carbon: "low-carbon design",
  high_flowability: "high flowability / pumpability with viscosity control",
  low_shrinkage: "low-shrinkage design",
  "低收缩": "low-shrinkage design",
  "低碳优先": "low-carbon design",
  "高流动性": "high flowability / pumpability with viscosity control",
  "早强": "early strength",
  "强度满足": "strength requirement",
  low_cost: "low cost",
  early_strength: "early strength",
  pumpability: "pumpability",
};
const EN_REASON_LABELS = {
  "由用户显式关键词触发": "Triggered by explicit user keywords.",
  "强度目标是配合比设计硬约束": "Strength is a hard constraint in mix design.",
  "作为强度约束和强度目标函数的目标值": "Used as the target value for strength constraints and objective functions.",
  "若模型存在 age/龄期变量，将固定为该值参与预测": "If the model contains an age variable, it is fixed to this value for prediction.",
  "用于决定目标函数和机理检索方向": "Used to determine objective functions and mechanism-retrieval directions.",
  "仅用于语义理解和图谱检索，不自动改变配方边界": "Used for semantic parsing and graph retrieval only; it does not automatically change mix-design bounds.",
  "作为后续优化的材料硬约束": "Used as a hard material constraint in downstream optimization.",
  "允许参与搜索，但不强制使用": "Allowed to participate in the search, but not forced to be used.",
  "在优化边界中强制设为 0": "Forced to zero in optimization bounds.",
  "作为真实流动性预测器的数值约束": "Used as a numerical constraint for the actual flowability predictor.",
  "用于收缩风险识别、知识图谱检索和后续低收缩优化目标": "Used for shrinkage-risk identification, KG retrieval, and downstream low-shrinkage optimization targets.",
};
const EN_INTERPRETATION_LABELS = {
  "提高施工流动性/泵送性并控制黏度": "Improve construction flowability and pumpability while controlling viscosity.",
  "降低熟料/水泥相关碳排放": "Reduce clinker- and cement-related carbon emissions.",
  "满足目标抗压强度": "Satisfy the target compressive strength.",
  "控制材料成本": "Control material cost.",
  "提高早期强度发展": "Improve early-age strength development.",
  "转化为目标抗压强度和强度约束": "Converted into target compressive strength and strength constraints.",
  "固定 age/龄期预测变量": "Fix the age variable used for prediction.",
  "提高 Carbon 最小化目标优先级": "Increase the priority of the carbon-minimization objective.",
  "转化为流动性数值区间约束": "Converted into a numerical flowability interval constraint.",
  "引入真实流动性目标并检索流动性机理": "Introduce a real flowability target and retrieve flowability mechanisms.",
  "降低干燥收缩、自收缩和开裂风险": "Reduce drying shrinkage, autogenous shrinkage, and cracking risk.",
  "转化为收缩上限或低收缩定性约束，并触发收缩机理检索": "Converted into a shrinkage upper-bound or qualitative low-shrinkage constraint and used to trigger shrinkage-mechanism retrieval.",
  "作为变量边界、知识图谱检索权重或材料偏好": "Used as variable bounds, KG retrieval weights, or material preferences.",
  "用于最终回答的工程背景": "Used as engineering background for the final answer.",
};
function enField(value) {
  return currentLanguage() === "en" ? (EN_FIELD_LABELS[value] || englishSafe(value, "field")) : value;
}
function enPriority(value) {
  return currentLanguage() === "en" ? (EN_PRIORITY_LABELS[value] || EN_INTERPRETATION_LABELS[value] || englishSafe(value, "priority")) : value;
}
function enReason(value) {
  return currentLanguage() === "en" ? (EN_REASON_LABELS[value] || englishSafe(value, "Derived from the user request.")) : value;
}
function enInterpretation(value) {
  return currentLanguage() === "en" ? (EN_INTERPRETATION_LABELS[value] || EN_PRIORITY_LABELS[value] || englishSafe(value, "engineering interpretation")) : value;
}
function enTraceValue(item = {}) {
  if (currentLanguage() !== "en") return item.value;
  if (item.field === "性能优先级") return enPriority(item.value);
  return englishSafe(item.value, "parsed value");
}

function localizeSchemeLabel(value = "") {
  if (currentLanguage() !== "en") return value || "-";
  const map = {
    "综合推荐方案": "Recommended compromise scheme",
    "最低碳候选": "Lowest-carbon candidate",
    "最高强候选": "Highest-strength candidate",
    "最低成本候选": "Lowest-cost candidate",
    "最高流动性候选": "Highest-flowability candidate",
    "最低收缩风险候选": "Lowest shrinkage/cracking-risk candidate",
    "综合折中推荐": "Recommended compromise",
  };
  return map[value] || englishSafe(value, "Candidate scheme");
}

function localizeSchemeRoles(roles = []) {
  if (currentLanguage() !== "en") return roles.join(" / ");
  const map = {
    "综合折中推荐": "recommended compromise",
    "最低碳候选": "lowest-carbon candidate",
    "最高强候选": "highest-strength candidate",
    "最低成本候选": "lowest-cost candidate",
    "最高流动性候选": "highest-flowability candidate",
    "最低收缩风险候选": "lowest shrinkage/cracking-risk candidate",
  };
  const translated = roles.map((item) => map[item] || englishSafe(item, "")).filter(Boolean);
  return translated.join(" / ") || "-";
}

function localizeTrustValue(value = "") {
  if (currentLanguage() !== "en") return value || "-";
  const map = {
    "中低": "medium-low",
    "中": "medium",
    "较高": "relatively high",
    "高": "high",
    "边界邻近": "near applicability boundary",
    "适用": "applicable",
    "模型预测": "Model prediction",
    "优化约束": "Optimization constraints",
    "图谱机理": "Knowledge-graph mechanisms",
    "实验验证": "Experimental validation",
    "定量": "quantitative",
    "可复现": "reproducible",
    "机理支持": "mechanistic support",
    "待确认": "to be confirmed",
    "由上传模型直接输出的强度、流动性或相关目标值。": "Strength, flowability, or related target values are directly produced by the uploaded model.",
    "由显式变量边界、材料策略和工程约束决定。": "Determined by explicit variable bounds, material policy, and engineering constraints.",
    "当前未发现明显边界风险": "No obvious boundary risk is currently flagged.",
  };
  const text = String(value || "-");
  if (text.startsWith("当前命中")) return "Current KG retrieval provides mechanistic support and recommendations.";
  const boundaryMatch = text.match(/^([A-Za-z0-9_]+)\s+靠近搜索边界$/);
  if (boundaryMatch) return `${localizeVariableName({ material: boundaryMatch[1] })} is near the search boundary.`;
  return map[text] || englishSafe(text, "-");
}

function localizeVariableName(row = {}) {
  const value = row.name || row.material || "";
  if (currentLanguage() !== "en") return value || "-";
  const key = String(row.material || value || "").toLowerCase();
  const map = {
    cement: "Cement",
    water: "Water",
    flyash: "Fly ash",
    fly_ash: "Fly ash",
    ggbs: "GGBFS",
    ggbfs: "GGBFS",
    slag: "GGBFS",
    silicafume: "Silica fume",
    silica_fume: "Silica fume",
    superplasticizer: "Superplasticizer",
    pce: "PCE superplasticizer",
    fineaggregate: "Fine aggregate",
    fine_aggregate: "Fine aggregate",
    coarseaggregate: "Coarse aggregate",
    coarse_aggregate: "Coarse aggregate",
    steelfiber: "Steel fiber",
    steel_fiber: "Steel fiber",
    age: "Curing age",
  };
  const byChinese = {
    "水泥": "Cement",
    "水": "Water",
    "粉煤灰": "Fly ash",
    "矿渣": "GGBFS",
    "矿渣粉": "GGBFS",
    "硅灰": "Silica fume",
    "减水剂": "Superplasticizer",
    "外加剂": "Admixture",
    "细骨料": "Fine aggregate",
    "砂": "Fine aggregate",
    "粗骨料": "Coarse aggregate",
    "石子": "Coarse aggregate",
    "钢纤维": "Steel fiber",
    "龄期": "Curing age",
  };
  return map[key] || byChinese[value] || englishSafe(value, "Variable");
}

function localizeAgentShortLabel(id) {
  const en = {
    requirement: "Requirement Parsing",
    optimizer: "Optimization Modeling",
    kg: "Mechanism Retrieval",
    requirement_analysis: "Requirement Parsing",
    optimization_modeling: "Optimization Modeling",
    mechanism_retrieval: "Mechanism Retrieval",
    report_generation: "Final Report",
  };
  const zh = {
    requirement: "需求解析",
    optimizer: "优化建模",
    kg: "机理检索",
    requirement_analysis: "需求解析",
    optimization_modeling: "优化建模",
    mechanism_retrieval: "机理检索",
    report_generation: "总报告",
  };
  return currentLanguage() === "en" ? (en[id] || id) : (zh[id] || id);
}

function localizeAgentSummary(agent = {}) {
  if (currentLanguage() !== "en") return agent.summary || "等待运行";
  const details = agent.details || {};
  if (agent.id === "requirement") {
    const target = details.target_mpa ?? "-";
    const age = details.age_days ?? "-";
    const priorities = (details.priorities || details.priority_ranking || []).join(", ") || "not specified";
    return `Identified target strength ${target} MPa, curing age ${age} d, and priorities: ${priorities}.`;
  }
  if (agent.id === "optimizer") {
    const objectives = Array.isArray(details.objectives) ? details.objectives.length : "-";
    const constraints = Array.isArray(details.constraints) ? details.constraints.length : "-";
    return `Generated ${objectives} objective functions and ${constraints} constraints, then ran ${details.selected_algorithm || details.algorithm || "the optimizer"}.`;
  }
  if (agent.id === "kg") {
    const hits = Array.isArray(details.hits) ? details.hits.length : 0;
    const recs = Array.isArray(details.recommendations) ? details.recommendations.length : 0;
    return `Retrieved ${hits} material-mechanism groups and generated ${recs} optimization recommendations.`;
  }
  return englishSafe(agent.summary, "Waiting for run.");
}

function englishSanitizePayload(value) {
  if (currentLanguage() !== "en") return value;
  if (Array.isArray(value)) return value.map((item) => englishSanitizePayload(item));
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, englishSanitizePayload(item)]));
  }
  if (typeof value === "string") return englishSafe(value, "Chinese source text hidden in English mode");
  return value;
}

function initialChatHtml() {
  const isEnglish = currentLanguage() === "en";
  const intro = isEnglish
    ? "Upload a model package, then enter the design requirement. Example:"
    : "请上传模型包，然后直接输入需求。例如：";
  const example = isEnglish
    ? "Design a low-carbon, high-flowability concrete targeting 80 MPa at 28 days, minimizing cement content and allowing fly ash, slag, and silica fume."
    : "设计一个 80 MPa、28 天龄期的低碳高流动性混凝土，尽量减少水泥用量，允许使用粉煤灰、矿渣和硅灰。";
  return `
  <article class="message assistant">
    <div class="avatar">AI</div>
    <div class="bubble">
      <p>${escapeHtml(intro)}</p>
      <p class="example">${escapeHtml(example)}</p>
    </div>
  </article>
`;
}

const stageMap = {
  requirement: document.querySelector('[data-agent="requirement"]'),
  optimizer: document.querySelector('[data-agent="optimizer"]'),
  kg: document.querySelector('[data-agent="kg"]'),
};

const stageText = {
  requirement: document.getElementById("stageRequirement"),
  optimizer: document.getElementById("stageOptimizer"),
  kg: document.getElementById("stageKg"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function addMessage(role, content) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "YOU" : "AI";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  article.append(avatar, bubble);
  chatLog.appendChild(article);
  scrollChatIfFollowing();
  saveCurrentSession();
  return bubble;
}

function addHtmlMessage(role, html, options = {}) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "YOU" : "AI";
  const bubble = document.createElement("div");
  bubble.className = "bubble rich-bubble";
  if (options.kind) bubble.dataset.messageKind = options.kind;
  bubble.innerHTML = html;
  rememberMathSources(bubble);
  article.append(avatar, bubble);
  chatLog.appendChild(article);
  typesetMath(bubble);
  if (options.scroll !== false) scrollChatIfFollowing();
  saveCurrentSession();
  return bubble;
}

function getSessions() {
  try {
    return JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function rememberMathSources(root) {
  root.querySelectorAll?.(".math-formula").forEach((node) => {
    if (!node.dataset.mathSource) node.dataset.mathSource = node.innerHTML;
  });
}

function serializableChatHtml() {
  const clone = chatLog.cloneNode(true);
  clone.querySelectorAll(".math-formula[data-math-source]").forEach((node) => {
    node.innerHTML = node.dataset.mathSource;
  });
  clone.querySelectorAll("mjx-container").forEach((node) => {
    if (!node.closest(".math-formula[data-math-source]")) node.remove();
  });
  clone.querySelectorAll(".run-report").forEach((node) => {
    node.outerHTML = '<div class="run-report" data-session-regenerate="run-report"></div>';
  });
  clone.querySelectorAll(".publication-report-shell").forEach((node) => {
    node.outerHTML = '<details class="publication-report-shell" data-session-regenerate="publication-report"></details>';
  });
  return clone.innerHTML;
}

function writeSessions(sessions) {
  const ordered = [...sessions].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  while (ordered.length) {
    try {
      localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(ordered));
      return ordered;
    } catch (error) {
      if (error?.name !== "QuotaExceededError") throw error;
      ordered.pop();
    }
  }
  localStorage.removeItem(SESSION_STORAGE_KEY);
  return [];
}

function compactSessionPayload(payload) {
  if (!payload) return null;
  const clone = typeof structuredClone === "function" ? structuredClone(payload) : JSON.parse(JSON.stringify(payload));
  if (clone.visualizations) clone.visualizations.optimization_plots = [];
  if (clone.optimization) {
    clone.optimization.history = (clone.optimization.history || []).slice(-12);
    clone.optimization.top_solutions = (clone.optimization.top_solutions || []).slice(0, 8);
  }
  return clone;
}

function sessionTitleFromChat() {
  const firstUser = chatLog.querySelector(".message.user .bubble");
  const text = firstUser?.textContent?.trim() || "新会话";
  return text.length > 24 ? `${text.slice(0, 24)}...` : text;
}

function createSessionRecord() {
  return {
    id: crypto.randomUUID ? crypto.randomUUID() : `session-${Date.now()}`,
    title: "新会话",
    updatedAt: Date.now(),
    html: initialChatHtml(),
    latestReportPayload: null,
  };
}

function renderSessionList() {
  if (!sessionList) return;
  const sessions = getSessions().sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  sessionList.innerHTML = sessions.map((session) => `
    <article class="session-item ${session.id === currentSessionId ? "active" : ""}" data-session-id="${escapeHtml(session.id)}">
      <button type="button" data-session-open="${escapeHtml(session.id)}">
        <strong>${escapeHtml(session.title || "新会话")}</strong>
        <small>${escapeHtml(new Date(session.updatedAt || Date.now()).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }))}</small>
      </button>
      <button type="button" data-session-delete="${escapeHtml(session.id)}" aria-label="删除会话">×</button>
    </article>
  `).join("");
}

function saveCurrentSession() {
  if (!currentSessionId) return;
  const sessions = getSessions();
  const index = sessions.findIndex((item) => item.id === currentSessionId);
  const record = {
    id: currentSessionId,
    title: sessionTitleFromChat(),
    updatedAt: Date.now(),
    html: serializableChatHtml(),
    latestReportPayload: compactSessionPayload(latestReportPayload),
  };
  if (index >= 0) sessions[index] = record;
  else sessions.push(record);
  const persisted = writeSessions(sessions);
  if (!persisted.some((item) => item.id === currentSessionId)) {
    modelStatus.textContent = "会话过大，已保留当前页但未写入历史存档";
  }
  localStorage.setItem(SESSION_ACTIVE_KEY, currentSessionId);
  renderSessionList();
}

function loadSession(sessionId) {
  const session = getSessions().find((item) => item.id === sessionId);
  if (!session) return;
  currentSessionId = session.id;
  chatLog.innerHTML = session.html || initialChatHtml();
  syncDefaultWelcomeLanguage();
  latestReportPayload = session.latestReportPayload || null;
  if (latestReportPayload) {
    const kgAgent = (latestReportPayload.agents || []).find((agent) => agent.id === "kg");
    const runReportBubble = chatLog.querySelector(".run-report")?.closest(".bubble");
    const formalReportBubble = chatLog.querySelector(".publication-report-shell")?.closest(".bubble");
    if (runReportBubble) runReportBubble.innerHTML = renderRunReport(latestReportPayload);
    if (formalReportBubble) formalReportBubble.innerHTML = publicationReportHtml(latestReportPayload, kgAgent?.trace || {});
  }
  rememberMathSources(chatLog);
  typesetMath(chatLog);
  if (latestReportPayload) {
    const kgAgent = (latestReportPayload.agents || []).find((agent) => agent.id === "kg");
    renderSolution(latestReportPayload.optimization?.solution_table || []);
    renderKpis(latestReportPayload.optimization?.kpis || {});
    renderProcessDetails(latestReportPayload.agents || []);
    renderKgGraph(kgAgent?.trace || {});
    renderRecommendations(latestReportPayload.recommendations || latestReportPayload.kg_hits || []);
  } else {
    renderSolution([]);
    renderKpis({});
    renderRecommendations([]);
    renderKgGraph({});
  }
  localStorage.setItem(SESSION_ACTIVE_KEY, currentSessionId);
  renderSessionList();
  chatLog.scrollTop = 0;
}

function rerenderLatestReportViews() {
  if (!latestReportPayload) return;
  const kgAgent = (latestReportPayload.agents || []).find((agent) => agent.id === "kg");
  let runReportBubble = chatLog.querySelector('[data-message-kind="run-report"]');
  let formalReportBubble = chatLog.querySelector('[data-message-kind="publication-report"]');
  if (!runReportBubble) runReportBubble = chatLog.querySelector(".run-report")?.closest(".bubble");
  if (!formalReportBubble) formalReportBubble = chatLog.querySelector(".publication-report-shell")?.closest(".bubble");
  if (runReportBubble) {
    runReportBubble.dataset.messageKind = "run-report";
    runReportBubble.innerHTML = renderRunReport(latestReportPayload);
  }
  if (formalReportBubble) {
    formalReportBubble.dataset.messageKind = "publication-report";
    formalReportBubble.innerHTML = publicationReportHtml(latestReportPayload, kgAgent?.trace || {});
  }
  renderSolution(latestReportPayload.optimization?.solution_table || []);
  renderKpis(latestReportPayload.optimization?.kpis || {});
  renderProcessDetails(latestReportPayload.agents || []);
  renderKgGraph(kgAgent?.trace || {});
  renderRecommendations(latestReportPayload.recommendations || latestReportPayload.kg_hits || []);
  rememberMathSources(chatLog);
  typesetMath(chatLog);
}

function handlePlatformLanguageChange() {
  syncDefaultWelcomeLanguage();
  rerenderLatestReportViews();
  saveCurrentSession();
}

function syncDefaultWelcomeLanguage() {
  const hasUserMessage = Boolean(chatLog.querySelector(".message.user"));
  const bubbleText = chatLog.querySelector(".message.assistant .bubble")?.textContent || "";
  const isDefaultWelcome = /请上传模型包，然后直接输入需求|Upload a model package, then enter the design requirement/.test(bubbleText);
  if (!hasUserMessage && isDefaultWelcome) chatLog.innerHTML = initialChatHtml();
}

function startNewSession() {
  const session = createSessionRecord();
  currentSessionId = session.id;
  latestReportPayload = null;
  chatLog.innerHTML = initialChatHtml();
  const sessions = getSessions();
  sessions.push(session);
  writeSessions(sessions);
  localStorage.setItem(SESSION_ACTIVE_KEY, currentSessionId);
  renderSessionList();
  renderSolution([]);
  renderKpis({});
  renderRecommendations([]);
  renderKgGraph({});
}

function deleteSession(sessionId) {
  const sessions = getSessions().filter((item) => item.id !== sessionId);
  writeSessions(sessions);
  if (sessionId === currentSessionId) {
    if (sessions.length) loadSession(sessions.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0))[0].id);
    else startNewSession();
  } else {
    renderSessionList();
  }
}

function initializeSessions() {
  const sessions = getSessions();
  const activeId = localStorage.getItem(SESSION_ACTIVE_KEY);
  const preferred = sessions.find((item) => item.id === activeId) || sessions.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0))[0];
  if (preferred) loadSession(preferred.id);
  else startNewSession();
}


function typesetMath(root) {
  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetPromise(root ? [root] : undefined).catch(() => {});
  }
}

function scrollChatIfFollowing() {
  if (followChatTail) chatLog.scrollTop = chatLog.scrollHeight;
}

function scrollBubbleToTop(bubble) {
  const message = bubble?.closest(".message");
  if (!message) return;
  chatLog.scrollTo({ top: Math.max(message.offsetTop - 18, 0), behavior: "smooth" });
}

chatLog.addEventListener("scroll", () => {
  const distanceFromBottom = chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight;
  followChatTail = distanceFromBottom < 80;
});

function mathifyExpression(expression = "") {
  return String(expression)
    .replaceAll(">=", "\\ge ")
    .replaceAll("<=", "\\le ")
    .replaceAll("*", "\\cdot ")
    .replaceAll("prediction", "\\hat{f}_{strength}")
    .replaceAll("flow_pred", "\\hat{f}_{flow}");
}

function formatAssistantAnswer(text) {
  const escaped = escapeHtml(text || "");
  const lines = escaped.split("\n");
  const blocks = [];
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const next = lines[i + 1] || "";
    if (line.trim().startsWith("|") && next.trim().match(/^\|?\s*:?-{3,}/)) {
      const headers = line.split("|").slice(1, -1).map((cell) => cell.trim());
      const rows = [];
      i += 2;
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        rows.push(lines[i].split("|").slice(1, -1).map((cell) => cell.trim()));
        i += 1;
      }
      i -= 1;
      blocks.push(
        `<table class="report-table"><thead><tr>${headers.map((cell) => `<th>${cell}</th>`).join("")}</tr></thead><tbody>${rows
          .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
          .join("")}</tbody></table>`,
      );
      continue;
    }
    blocks.push(line);
  }
  return `<p>${blocks.join("\n")
    .replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>")
    .replace(/^##### (.*)$/gm, "<h5>$1</h5>")
    .replace(/^#### (.*)$/gm, "<h5>$1</h5>")
    .replace(/^### (.*)$/gm, "<h4>$1</h4>")
    .replace(/^## (.*)$/gm, "<h3>$1</h3>")
    .replace(/^# (.*)$/gm, "<h3>$1</h3>")
    .replace(/^---+$/gm, "<hr>")
    .replace(/^> (.*)$/gm, "<blockquote>$1</blockquote>")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/^- (.*)$/gm, "<li>$1</li>")
    .replace(/^\d+\.\s+(.*)$/gm, "<li class=\"ordered\">$1</li>")
    .replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>")
    .replace(/\n{2,}/g, "</p><p>")
    .replace(/\n/g, "<br>")}</p>`;
}

function collectCitations(data = {}) {
  const citations = [];
  const seen = new Set();
  const add = (items = []) => {
    items.forEach((item) => {
      const key = `${item.ref_no || ""}:${item.id || ""}`;
      if (!item.id || seen.has(key)) return;
      seen.add(key);
      citations.push(item);
    });
  };
  (data.agents || []).forEach((agent) => add(agent.details?.citations || []));
  Object.values(data.role_evidence || {}).forEach((bundle) => add(bundle.citations || []));
  return citations;
}

function linkCitationRefs(html, citations = []) {
  const byNo = new Map(citations.map((item) => [String(item.ref_no), item]));
  return String(html || "").replace(/\[(\d+)\]/g, (match, ref) => {
    const item = byNo.get(ref);
    if (!item?.id) return match;
    const title = currentLanguage() === "en" ? englishCitationTitle(item) : item.title || "";
    return `<button type="button" class="inline-citation" data-citation-id="${escapeHtml(item.id)}" data-citation-ref="${escapeHtml(ref)}" title="${escapeHtml(title)}">[${escapeHtml(ref)}]</button>`;
  });
}

async function fetchCitationRecord(id) {
  if (citationDetailCache.has(id)) return citationDetailCache.get(id);
  const payload = await fetch(`/api/evidence/${encodeURIComponent(id)}`).then((res) => res.json());
  if (payload.status !== "success") throw new Error(payload.message || (currentLanguage() === "en" ? "Failed to read evidence" : "证据读取失败"));
  citationDetailCache.set(id, payload.record || {});
  return payload.record || {};
}

function removeCitationPopover() {
  window.clearTimeout(citationHoverTimer);
  document.querySelector(".citation-popover")?.remove();
}

function citationLinksHtml(record) {
  const links = [];
  if (record.doi_url) links.push(`<a href="${escapeHtml(record.doi_url)}" target="_blank" rel="noreferrer">DOI</a>`);
  if (record.file_url) links.push(`<a href="${escapeHtml(record.file_url)}" target="_blank" rel="noreferrer">${currentLanguage() === "en" ? "PDF / Source" : "PDF / 原文"}</a>`);
  return links.length ? `<div class="citation-popover-links">${links.join("")}</div>` : "";
}

function citationDisplayTitle(record = {}) {
  return currentLanguage() === "en" ? englishCitationTitle(record) : (record.title || record.id || "");
}

function citationDisplayText(record = {}, fallback = "") {
  const text = String(record.text || record.evidence || fallback || "");
  if (currentLanguage() !== "en") return text || "暂无原文片段。";
  return englishSafe(text, "Original source text is indexed and available for traceability; Chinese source text is hidden in English mode.");
}

function citationDisplayRoute(route = []) {
  return currentLanguage() === "en" ? route.map((part) => englishSafe(part, "evidence route")) : route;
}

async function showCitationPopover(button) {
  const id = button?.dataset?.citationId;
  if (!id) return;
  const record = await fetchCitationRecord(id);
  removeCitationPopover();
  const rect = button.getBoundingClientRect();
  const popover = document.createElement("aside");
  popover.className = "citation-popover";
  popover.innerHTML = `
    <header><span>${escapeHtml(record.source_type || "evidence")}</span><strong>${escapeHtml(citationDisplayTitle(record))}</strong></header>
    ${record.route?.length ? `<div class="citation-route compact">${citationDisplayRoute(record.route).map((part) => `<span>${escapeHtml(part)}</span>`).join("<i>→</i>")}</div>` : ""}
    <p>${escapeHtml(citationDisplayText(record, "暂无原文片段。").slice(0, 520))}</p>
    ${citationLinksHtml(record)}
  `;
  document.body.appendChild(popover);
  const left = Math.min(window.innerWidth - popover.offsetWidth - 16, Math.max(16, rect.left - 24));
  const top = rect.bottom + 10 + window.scrollY;
  popover.style.left = `${left}px`;
  popover.style.top = `${top}px`;
}

async function openCitationDetail(id) {
  const record = await fetchCitationRecord(id);
  const existing = document.querySelector(".citation-drawer");
  if (existing) existing.remove();
  const drawer = document.createElement("aside");
  drawer.className = "citation-drawer";
  drawer.innerHTML = `
    <header>
      <div><span>${escapeHtml(record.source_type || "evidence")}</span><strong>${escapeHtml(citationDisplayTitle(record))}</strong></div>
      <button type="button" data-citation-close>${currentLanguage() === "en" ? "Close" : "关闭"}</button>
    </header>
    ${record.route?.length ? `<div class="citation-route">${citationDisplayRoute(record.route).map((part) => `<span>${escapeHtml(part)}</span>`).join("<i>→</i>")}</div>` : ""}
    <dl>
      ${Object.entries(record.metadata || {}).slice(0, 8).map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(Array.isArray(value) ? value.join("; ") : value)}</dd></div>`).join("")}
    </dl>
    ${citationLinksHtml(record)}
    <pre>${escapeHtml(citationDisplayText(record, "暂无原文片段。"))}</pre>
  `;
  document.body.appendChild(drawer);
}

function openReferencesDrawer() {
  if (!latestReportPayload) return;
  const citations = collectCitations(latestReportPayload);
  const existing = document.querySelector(".citation-drawer");
  if (existing) existing.remove();
  const drawer = document.createElement("aside");
  drawer.className = "citation-drawer references-drawer";
  const isEnglish = currentLanguage() === "en";
  drawer.innerHTML = `
    <header>
      <div><span>${isEnglish ? "Evidence" : "证据"}</span><strong>${isEnglish ? "References" : "参考引用"}</strong></div>
      <button type="button" data-citation-close>${isEnglish ? "Close" : "关闭"}</button>
    </header>
    <div class="citation-strip drawer-citation-strip">
      ${citations.map((item) => `<button type="button" data-citation-id="${escapeHtml(item.id)}">[${escapeHtml(item.ref_no || item.id)}] ${escapeHtml(isEnglish ? englishCitationTitle(item) : item.title)}</button>`).join("")}
    </div>
    ${citationTableHtml(citations)}
  `;
  document.body.appendChild(drawer);
}

function setBusy(busy) {
  sendButton.disabled = busy;
  modelFile.disabled = busy;
  sendButton.textContent = busy ? "运行中" : "发送";
}

function resetStages() {
  Object.values(stageMap).forEach((stage) => {
    stage.classList.remove("running", "completed");
  });
  const isEnglish = currentLanguage() === "en";
  stageText.requirement.textContent = isEnglish ? "Waiting for user requirement" : "等待用户需求";
  stageText.optimizer.textContent = isEnglish ? "Waiting for objectives and constraints" : "等待目标函数与约束设置";
  stageText.kg.textContent = isEnglish ? "Waiting for KG retrieval" : "等待知识图谱检索";
}

function markStage(id, state, summary) {
  const stage = stageMap[id];
  if (!stage) return;
  stage.classList.remove("running", "completed");
  if (state) stage.classList.add(state);
  if (summary && stageText[id]) stageText[id].textContent = summary;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function playAgentTrace(agents = []) {
  resetStages();
  for (const agent of agents) {
    markStage(agent.id, "running", currentLanguage() === "en" ? "Running..." : "运行中...");
    await sleep(420);
    markStage(agent.id, "completed", currentLanguage() === "en" ? localizeAgentSummary(agent) : agent.summary || "已完成");
  }
}

function renderSolution(rows = []) {
  const host = document.getElementById("solutionTable");
  host.replaceChildren();
  if (!rows.length) {
    host.innerHTML = currentLanguage() === "en"
      ? '<div class="solution-card"><strong>No results</strong><small>Upload a model and run the design workflow.</small></div>'
      : '<div class="solution-card"><strong>暂无结果</strong><small>上传模型并运行后显示</small></div>';
    return;
  }
  rows.slice(0, 16).forEach((row) => {
    const card = document.createElement("div");
    card.className = "solution-card";
    const value = Number.isFinite(Number(row.value)) ? Number(row.value).toFixed(2) : row.value;
    const label = localizeVariableName(row);
    card.innerHTML = `<strong title="${escapeHtml(label)}">${escapeHtml(label)}</strong><span>${escapeHtml(value)}</span><small>${escapeHtml(row.unit || "")}</small>`;
    host.appendChild(card);
  });
}

function solutionCardsHtml(rows = []) {
  if (!rows.length) return currentLanguage() === "en"
    ? '<div class="solution-card"><strong>No results</strong><small>The model did not return mix-design variables.</small></div>'
    : '<div class="solution-card"><strong>暂无结果</strong><small>模型未返回配合比</small></div>';
  return rows.slice(0, 20).map((row) => {
    const value = Number.isFinite(Number(row.value)) ? Number(row.value).toFixed(row.unit === "d" ? 0 : 2) : row.value;
    const label = localizeVariableName(row);
    return `<div class="solution-card"><strong title="${escapeHtml(label)}">${escapeHtml(label)}</strong><span>${escapeHtml(value)}</span><small>${escapeHtml(row.unit || "")}</small></div>`;
  }).join("");
}

function solutionTableHtml(rows = []) {
  if (!rows.length) return currentLanguage() === "en" ? '<p class="empty-copy">The model did not return displayable mix-design variables.</p>' : '<p class="empty-copy">模型未返回可展示配合比。</p>';
  return `
    <table class="mix-table">
      <thead><tr><th>${currentLanguage() === "en" ? "Variable" : "变量"}</th><th>${currentLanguage() === "en" ? "Recommended value" : "推荐值"}</th><th>${currentLanguage() === "en" ? "Unit" : "单位"}</th></tr></thead>
      <tbody>
        ${rows.map((row) => {
          const value = Number.isFinite(Number(row.value)) ? Number(row.value).toFixed(row.unit === "d" ? 0 : 2) : row.value;
          return `<tr><td>${escapeHtml(localizeVariableName(row))}</td><td>${escapeHtml(value)}</td><td>${escapeHtml(row.unit || "")}</td></tr>`;
        }).join("")}
      </tbody>
    </table>
  `;
}

function ratioCardsHtml(ratios = {}) {
  const items = currentLanguage() === "en"
    ? [
        ["Total binder", ratios.binder_total_kg_m3, "kg/m3"],
        ["Water-to-binder ratio", ratios.water_binder_ratio, "w/b"],
        ["Paste-to-aggregate ratio", ratios.paste_aggregate_ratio, "p/a"],
      ]
    : [
        ["胶凝材料总量", ratios.binder_total_kg_m3, "kg/m3"],
        ["水胶比", ratios.water_binder_ratio, "w/b"],
        ["浆骨比", ratios.paste_aggregate_ratio, "p/a"],
      ];
  return items.map(([label, value, unit]) => `
    <div class="ratio-card">
      <strong>${escapeHtml(label)}</strong>
      <span>${value === undefined ? "-" : escapeHtml(value)}</span>
      <small>${escapeHtml(unit)}</small>
    </div>
  `).join("");
}

function theoryCardsHtml(theory = {}) {
  const items = currentLanguage() === "en"
    ? [
        ["Absolute volume", theory.absolute_volume_m3, "m3"],
        ["Volume error", theory.absolute_volume_error, "m3"],
        ["Theoretical density", theory.theoretical_density_kg_m3, "kg/m3"],
        ["Bolomey strength", theory.bolomey_strength_mpa, "MPa"],
        ["Bolomey w/b upper bound", theory.bolomey_wb_max, "w/b"],
      ]
    : [
        ["绝对体积", theory.absolute_volume_m3, "m3"],
        ["体积误差", theory.absolute_volume_error, "m3"],
        ["理论容重", theory.theoretical_density_kg_m3, "kg/m3"],
        ["鲍罗米强度", theory.bolomey_strength_mpa, "MPa"],
        ["鲍罗米 w/b 上限", theory.bolomey_wb_max, "w/b"],
      ];
  return `
    <div class="theory-grid">
      ${items.map(([label, value, unit]) => `
        <div class="theory-card">
          <strong>${escapeHtml(label)}</strong>
          <span>${value === undefined || value === null ? "-" : escapeHtml(value)}</span>
          <small>${escapeHtml(unit)}</small>
        </div>
      `).join("")}
    </div>
  `;
}

function theoryStatusHtml(theory = {}) {
  if (currentLanguage() === "en") {
    return `
      <div class="theory-status">
        <span class="${theory.absolute_volume_pass ? "pass" : "fail"}">Absolute-volume method ${theory.absolute_volume_pass ? "passed" : "failed"}</span>
        <span class="${theory.bolomey_pass ? "pass" : "fail"}">Bolomey check ${theory.bolomey_pass ? "passed" : "failed"}</span>
      </div>
    `;
  }
  return `
    <div class="theory-status">
      <span class="${theory.absolute_volume_pass ? "pass" : "fail"}">绝对体积法 ${theory.absolute_volume_pass ? "通过" : "未通过"}</span>
      <span class="${theory.bolomey_pass ? "pass" : "fail"}">鲍罗米校核 ${theory.bolomey_pass ? "通过" : "未通过"}</span>
    </div>
  `;
}

function theoryVolumeFigureHtml(theory = {}) {
  const components = theory.components || [];
  if (!components.length) return "";
  const air = Number(theory.air_content || 0);
  const total = components.reduce((sum, item) => sum + (Number(item.volume) || 0), 0) + air;
  const colors = ["#0f766e", "#2563eb", "#b45309", "#7c3aed", "#8ab17d", "#b56576", "#64748b", "#dc2626"];
  return `<div class="theory-figure-wrap">${svgFrame("Absolute-volume closure", `
    ${components.map((item, index) => {
      const prev = components.slice(0, index).reduce((sum, current) => sum + (Number(current.volume) || 0), 0);
      const x = 34 + (prev / Math.max(total, 1e-9)) * 252;
      const w = ((Number(item.volume) || 0) / Math.max(total, 1e-9)) * 252;
      return `<rect x="${x.toFixed(1)}" y="82" width="${w.toFixed(1)}" height="38" fill="${colors[index % colors.length]}"></rect>`;
    }).join("")}
    <rect x="${(34 + (((total - air) / Math.max(total, 1e-9)) * 252)).toFixed(1)}" y="82" width="${((air / Math.max(total, 1e-9)) * 252).toFixed(1)}" height="38" fill="#d1d5db"></rect>
    <line x1="34" y1="144" x2="286" y2="144" class="fig-axis"></line>
    <text x="160" y="166" text-anchor="middle" class="fig-label">ΣVi + Vair = ${escapeHtml(theory.absolute_volume_m3 ?? "-")} m3</text>
  `, 320, 210, "Absolute-volume contribution of each constituent.")}</div>`;
}

function candidateSchemesHtml(schemes = []) {
  if (!schemes.length) return currentLanguage() === "en" ? '<p class="empty-copy">Only one constrained recommended scheme is retained for this task.</p>' : '<p class="empty-copy">当前任务仅保留一个受约束推荐方案。</p>';
  return `
    <div class="scheme-grid">
      ${schemes.map((scheme) => `
        <section class="scheme-card">
          <header>
            <strong>${escapeHtml(localizeSchemeLabel(scheme.label))}</strong>
            <span>w/b ${escapeHtml(scheme.ratios?.water_binder_ratio ?? "-")}</span>
          </header>
          ${solutionTableHtml(scheme.solution_table || [])}
        </section>
      `).join("")}
    </div>
  `;
}

function schemeComparisonHtml(rows = []) {
  if (!rows.length) return currentLanguage() === "en" ? '<p class="empty-copy">No comparable candidate schemes are currently available.</p>' : '<p class="empty-copy">当前尚无可比较候选方案。</p>';
  return `
    <table class="comparison-table">
      <thead><tr><th>${currentLanguage() === "en" ? "Scheme" : "方案"}</th><th>${currentLanguage() === "en" ? "Role" : "角色"}</th><th>${currentLanguage() === "en" ? "Strength" : "强度"}</th><th>${currentLanguage() === "en" ? "Carbon" : "碳排"}</th><th>${currentLanguage() === "en" ? "Cost" : "成本"}</th><th>${currentLanguage() === "en" ? "Flowability" : "流动性"}</th><th>${currentLanguage() === "en" ? "Shrinkage risk" : "收缩风险"}</th><th>w/b</th><th>p/a</th></tr></thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            <td>${escapeHtml(localizeSchemeLabel(row.label))}</td>
            <td>${escapeHtml(localizeSchemeRoles(row.roles || []))}</td>
            <td>${escapeHtml(row.strength_mpa ?? "-")}</td>
            <td>${escapeHtml(row.carbon_kgco2e_m3 ?? "-")}</td>
            <td>${escapeHtml(row.cost_index ?? "-")}</td>
            <td>${escapeHtml(row.flowability ?? "-")}</td>
            <td>${escapeHtml(row.shrinkage_risk_index ?? "-")}</td>
            <td>${escapeHtml(row.water_binder_ratio ?? "-")}</td>
            <td>${escapeHtml(row.paste_aggregate_ratio ?? "-")}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function methodBaselineHtml(rows = []) {
  if (!rows.length) return "";
  const isEnglish = currentLanguage() === "en";
  const value = (row, enKey, zhKey, fallback = "-") => isEnglish ? englishSafe(row[enKey], fallback) : (row[zhKey] || row[enKey] || fallback);
  return `
    <div class="comparison-block">
      <h4>${isEnglish ? "Method Baseline Audit" : "方法基线审计"}</h4>
      <table class="formal-table">
        <thead><tr>
          <th>${isEnglish ? "Method" : "方法"}</th>
          <th>${isEnglish ? "Basis" : "依据"}</th>
          <th>${isEnglish ? "Limitation" : "局限"}</th>
          <th>${isEnglish ? "Current advantage" : "当前平台优势"}</th>
        </tr></thead>
        <tbody>
          ${rows.map((row) => `
            <tr>
              <td>${escapeHtml(value(row, "method", "method_zh"))}</td>
              <td>${escapeHtml(value(row, "basis", "basis_zh"))}</td>
              <td>${escapeHtml(value(row, "observed_or_expected_limitation", "observed_or_expected_limitation_zh"))}</td>
              <td>${escapeHtml(value(row, "current_system_advantage", "current_system_advantage_zh"))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function evidenceChainHtml(rows = []) {
  if (!rows.length) return "";
  const isEnglish = currentLanguage() === "en";
  const value = (row, enKey, zhKey, fallback = "-") => isEnglish ? englishSafe(row[enKey], fallback) : (row[zhKey] || row[enKey] || fallback);
  return `
    <div class="comparison-block evidence-chain-block">
      <h4>${isEnglish ? "Design Evidence Chain" : "设计证据链"}</h4>
      <table class="formal-table">
        <thead><tr>
          <th>${isEnglish ? "Claim" : "结论"}</th>
          <th>${isEnglish ? "Value" : "数值/状态"}</th>
          <th>${isEnglish ? "Evidence source" : "证据类型"}</th>
          <th>${isEnglish ? "Evidence" : "证据说明"}</th>
          <th>${isEnglish ? "Validation" : "验证动作"}</th>
        </tr></thead>
        <tbody>
          ${rows.map((row) => `
            <tr>
              <td>${escapeHtml(value(row, "claim", "claim_zh"))}</td>
              <td>${escapeHtml(row.value ?? "-")}</td>
              <td>${escapeHtml(englishSafe(row.source_type, row.source_type || "-"))}</td>
              <td>${escapeHtml(value(row, "evidence", "evidence_zh"))}${(row.refs || []).length ? ` <small>${escapeHtml((isEnglish ? "Refs: " : "引用：") + row.refs.join(", "))}</small>` : ""}</td>
              <td>${escapeHtml(value(row, "validation", "validation_zh"))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function trustProfileHtml(profile = {}) {
  const evidence = profile.evidence_levels || [];
  const risks = profile.risk_flags || [];
  if (currentLanguage() === "en") {
    return `
      <section class="trust-panel">
        <header>
          <div><span>Confidence</span><strong>${escapeHtml(localizeTrustValue(profile.confidence || "-"))}</strong></div>
          <div><span>Applicability</span><strong>${escapeHtml(localizeTrustValue(profile.applicability || "-"))}</strong></div>
        </header>
        <div class="evidence-ledger">
          ${evidence.map((item) => `<article><b>${escapeHtml(localizeTrustValue(item.label))}</b><em>${escapeHtml(localizeTrustValue(item.level))}</em><p>${escapeHtml(localizeTrustValue(item.description))}</p></article>`).join("")}
        </div>
        <div class="risk-flags">
          ${(risks.length ? risks : ["No obvious boundary risk is currently flagged."]).map((item) => `<span>${escapeHtml(localizeTrustValue(item))}</span>`).join("")}
        </div>
      </section>
    `;
  }
  return `
    <section class="trust-panel">
      <header>
        <div><span>可信度</span><strong>${escapeHtml(profile.confidence || "-")}</strong></div>
        <div><span>适用域</span><strong>${escapeHtml(profile.applicability || "-")}</strong></div>
      </header>
      <div class="evidence-ledger">
        ${evidence.map((item) => `<article><b>${escapeHtml(item.label)}</b><em>${escapeHtml(item.level)}</em><p>${escapeHtml(item.description)}</p></article>`).join("")}
      </div>
      <div class="risk-flags">
        ${(risks.length ? risks : ["当前未发现明显边界风险"]).map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
      </div>
    </section>
  `;
}

function handoffBoardHtml(handoffs = {}) {
  const isEnglish = currentLanguage() === "en";
  const cards = [
    [isEnglish ? "Requirement -> Optimization" : "需求 -> 优化", handoffs.requirement_to_optimizer, "design-spec"],
    [isEnglish ? "Optimization -> KG" : "优化 -> 图谱", handoffs.optimizer_to_kg, "optimization-payload"],
    [isEnglish ? "KG -> Report" : "图谱 -> 报告", handoffs.kg_to_report, "evidence-payload"],
    [isEnglish ? "Report -> Supplements" : "报告 -> 补充", handoffs.report_feedback, "report-feedback"],
  ];
  return `
    <section class="handoff-board">
      ${cards.map(([title, payload, kind], index) => `
        <article class="${kind}" style="--i:${index}">
          <h4>${escapeHtml(title)}</h4>
          <pre>${escapeHtml(JSON.stringify(englishSanitizePayload(payload || {}), null, 2))}</pre>
        </article>
      `).join("")}
    </section>
  `;
}

function reportFrontMatterHtml(optimization = {}, reportRequests = {}) {
  const comparison = optimization.scheme_comparison || [];
  const best = comparison[0] || {};
  const abstract = `系统依据用户约束完成结构化需求解析、受约束多目标优化、知识图谱机理复审与证据化报告生成。当前推荐方案以 ${best.label || "综合推荐方案"} 为核心，并同步保留候选配方族用于权衡分析。`;
  return `
    <section class="report-frontmatter">
      <div>
        <h4>摘要</h4>
        <p>${escapeHtml(abstract)}</p>
      </div>
      <div>
        <h4>关键词</h4>
        <p>高强混凝土；低碳设计；低收缩；多目标优化；知识图谱；大语言模型；机理约束</p>
      </div>
      <div>
        <h4>补充材料请求</h4>
        <p>图：${escapeHtml((reportRequests.requested_figures || []).join("；") || "无")}</p>
        <p>表：${escapeHtml((reportRequests.requested_tables || []).join("；") || "无")}</p>
      </div>
    </section>
  `;
}

function renderKpis(kpis = {}) {
  const host = document.getElementById("kpiGrid");
  host.replaceChildren();
  const labels = {
    strength_mpa: ["预测强度", "MPa"],
    carbon_kgco2e_m3: ["碳排放", "kgCO2e/m3"],
    cost_index: ["成本指数", "index"],
    flowability: ["流动性", "model"],
    shrinkage_risk_index: ["收缩风险指数", "0-1"],
    crack_risk_index: ["收缩风险指数", "0-1"],
  };
  Object.entries(labels).forEach(([key, [label, unit]]) => {
    const value = kpis[key];
    const card = document.createElement("div");
    card.className = "kpi-card";
    card.innerHTML = `<strong>${label}</strong><span>${value === undefined ? "-" : escapeHtml(value)}</span><small>${unit}</small>`;
    host.appendChild(card);
  });
}

function kpiCardsHtml(kpis = {}) {
  const labels = currentLanguage() === "en"
    ? {
        strength_mpa: ["Predicted strength", "MPa"],
        carbon_kgco2e_m3: ["Carbon footprint", "kgCO2e/m3"],
        cost_index: ["Cost index", "index"],
        flowability: ["Flowability", "model"],
        shrinkage_risk_index: ["Shrinkage-cracking risk", "0-1 proxy"],
        crack_risk_index: ["Crack risk", "0-1 proxy"],
      }
    : {
        strength_mpa: ["预测强度", "MPa"],
        carbon_kgco2e_m3: ["碳排放", "kgCO2e/m3"],
        cost_index: ["成本指数", "index"],
        flowability: ["流动性", "model"],
        shrinkage_risk_index: ["收缩风险指数", "0-1 proxy"],
        crack_risk_index: ["收缩风险指数", "0-1 proxy"],
      };
  return Object.entries(labels).map(([key, [label, unit]]) => {
    const value = kpis[key];
    return `<div class="kpi-card"><strong>${label}</strong><span>${value === undefined ? "-" : escapeHtml(value)}</span><small>${unit}</small></div>`;
  }).join("");
}

function recommendationDisplay(item = {}) {
  const isEnglish = currentLanguage() === "en";
  if (!isEnglish) {
    return {
      topic: item.topic || item.material || "机理",
      text: item.mechanism || item.evidence || item.recommendation || "",
      action: item.recommendation && item.recommendation !== item.mechanism ? item.recommendation : "",
    };
  }
  const material = localizeVariableName({ material: item.material || item.topic || "" });
  const topic = item.topic_en && !containsChinese(item.topic_en)
    ? item.topic_en
    : (material && material !== "Variable" ? material : "Mechanistic recommendation");
  const directText = item.recommendation_en || item.mechanism_en || item.evidence_en;
  if (directText && !containsChinese(directText)) {
    return {
      topic,
      text: directText,
      action: item.validation_en || "",
    };
  }
  const chainParts = [
    item.feature_en,
    item.mechanism_en,
    item.consequence_en,
    item.performance_en || item.performance,
  ].filter((part) => part && !containsChinese(part));
  if (chainParts.length) {
    return {
      topic,
      text: chainParts.join(" -> "),
      action: item.relation_en ? `Expected effect: ${item.relation_en}.` : "",
    };
  }
  if (Array.isArray(item.chains) && item.chains.length) {
    const chain = item.chains.find((entry) => entry.feature_en || entry.mechanism_en || entry.consequence_en) || item.chains[0];
    const parts = [chain.feature_en, chain.mechanism_en, chain.consequence_en, chain.performance_en || chain.performance].filter((part) => part && !containsChinese(part));
    if (parts.length) return { topic, text: parts.join(" -> "), action: chain.relation_en ? `Expected effect: ${chain.relation_en}.` : "" };
  }
  const fallback = englishSafe(item.recommendation || item.mechanism || item.evidence || item.description || "", "");
  return {
    topic,
    text: fallback || "No structured mechanistic statement was returned for this item.",
    action: "",
  };
}

function renderRecommendations(items = []) {
  const host = document.getElementById("recommendations");
  host.replaceChildren();
  if (!items.length) {
    host.innerHTML = currentLanguage() === "en"
      ? '<div class="rec-card"><strong>No recommendations</strong><p>Knowledge-graph mechanistic recommendations will appear after the run.</p></div>'
      : '<div class="rec-card"><strong>暂无建议</strong><p>运行后显示知识图谱机理增强建议。</p></div>';
    return;
  }
  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "rec-card";
    const rec = recommendationDisplay(item);
    card.innerHTML = `
      <strong>${escapeHtml(rec.topic)}</strong>
      <p>${escapeHtml(rec.text)}</p>
      ${rec.action ? `<p>${escapeHtml(rec.action)}</p>` : ""}
    `;
    host.appendChild(card);
  });
}

function renderProcessDetails(agents = []) {
  const host = document.getElementById("processDetails");
  if (!host) return;
  host.replaceChildren();
  agents.forEach((agent) => {
    const block = document.createElement("div");
    block.className = "process-block";
    const title = document.createElement("strong");
    title.textContent = agent.name;
    block.appendChild(title);
    if (agent.id === "requirement") {
      const flow = document.createElement("div");
      flow.className = "token-flow";
      (agent.trace || []).forEach((item) => {
        const pill = document.createElement("span");
        pill.innerHTML = `${escapeHtml(item.token)} <em>${escapeHtml(item.field)}: ${escapeHtml(item.value)}</em>`;
        flow.appendChild(pill);
      });
      block.appendChild(flow);
    } else if (agent.id === "optimizer") {
      const trace = agent.trace || {};
      const formulaList = document.createElement("div");
      formulaList.className = "formula-list";
      (trace.objectives || []).forEach((obj) => {
        const row = document.createElement("div");
        row.innerHTML = `<b>${escapeHtml(obj.name)}</b><code>${escapeHtml(obj.goal)}: ${escapeHtml(obj.expression)}</code>`;
        formulaList.appendChild(row);
      });
      (trace.constraints || []).forEach((expr) => {
        const row = document.createElement("div");
        row.innerHTML = `<b>Constraint</b><code>${escapeHtml(expr)}</code>`;
        formulaList.appendChild(row);
      });
      block.appendChild(formulaList);
    } else if (agent.id === "kg") {
      const trace = agent.trace || {};
      const summary = document.createElement("p");
      summary.textContent = `命中节点 ${trace.nodes?.length || 0} 个，关系 ${trace.links?.length || 0} 条。`;
      block.appendChild(summary);
    }
    host.appendChild(block);
  });
}

function renderKgGraph(trace = {}) {
  const host = document.getElementById("kgGraph");
  if (!host) return;
  host.replaceChildren();
  const nodes = trace.nodes || [];
  if (!nodes.length) {
    host.innerHTML = '<div class="kg-empty">等待检索结果</div>';
    return;
  }
  nodes.slice(0, 12).forEach((node, index) => {
    const dot = document.createElement("div");
    dot.className = `kg-node ${node.type || "material"}`;
    dot.style.setProperty("--i", index);
    dot.textContent = node.label;
    host.appendChild(dot);
  });
}

function formulasHtml(trace = {}) {
  const objectives = trace.objectives || [];
  const constraints = trace.constraints || [];
  const wbWindow = trace.traditional_theory?.water_binder_window || {};
  return `
    <div class="formula-list">
      ${wbWindow.min != null && wbWindow.max != null ? `<div><b>Bolomey prior</b><span class="math-formula">\\(w/b \\in [${escapeHtml(wbWindow.min)}, ${escapeHtml(wbWindow.max)}]\\)</span><small>pre-search</small></div>` : ""}
      ${objectives.map((obj) => `<div><b>${escapeHtml(obj.name)}</b><span class="math-formula">\\(${escapeHtml(mathifyExpression(obj.expression))}\\)</span><small>${escapeHtml(obj.goal)}</small></div>`).join("")}
      ${constraints.map((expr) => `<div><b>Constraint</b><span class="math-formula">\\(${escapeHtml(mathifyExpression(expr))}\\)</span><small>hard</small></div>`).join("")}
    </div>
  `;
}

function liveRunHtml() {
  return `
    <div class="live-run">
      <div class="live-workbench">
        <section class="live-main-stage">
          <header class="live-stage-header">
            <span class="pulse-dot"></span>
            <div>
              <strong>${t("设计推理主舞台")}</strong>
              <p>${t("语义解析、约束收缩、Pareto 搜索和知识图谱命中同步展开")}</p>
            </div>
          </header>
          <div class="live-detail-board">
            <section>
              <h4>${t("实时操作流")}</h4>
              <div class="live-event-log"></div>
            </section>
            <section>
              <h4>${t("实时 Pareto 搜索")}</h4>
              <svg class="live-chart" viewBox="0 0 420 180" aria-label="${t("实时优化曲线")}">
                <path class="hv-path"></path>
                <path class="feasible-path"></path>
                <g class="pareto-points"></g>
              </svg>
              <div class="generation-stream"></div>
            </section>
          </div>
          <section class="live-visual-board">
            <article class="semantic-panel">
              <header><span></span><strong>${t("语义拆解轨道")}</strong></header>
              <div class="semantic-rail"></div>
            </article>
            <article class="objective-panel">
              <header><span></span><strong>${t("目标与约束装配")}</strong></header>
              <div class="objective-rail"></div>
            </article>
            <article class="kg-panel">
              <header><span></span><strong>${t("知识图谱命中")}</strong></header>
              <div class="kg-rail"></div>
            </article>
          </section>
        </section>
        <aside class="live-side-rail">
          <div class="live-header">
            <span class="pulse-dot"></span>
            <strong>${t("多智能体协同设计中")}</strong>
          </div>
          <div class="live-monitor-dock">
            <strong>${t("实时监控")}</strong>
            <b class="dock-event">${t("等待启动")}</b>
            <em class="dock-generation">Gen -</em>
          </div>
          <div class="live-agent-grid">
            <section class="live-agent active" data-live-agent="requirement">
              <header><span>01</span><strong>${t("需求解析智能体")}</strong></header>
              <div class="live-lane"><b>${t("解析")}</b><p>${t("读取自然语言，识别强度、龄期、材料与性能目标")}</p></div>
              <div class="live-lane"><b>${t("运行")}</b><p class="typing">${t("语义切分中")}</p></div>
              <div class="live-lane"><b>${t("回答")}</b><p>${t("等待输出")}</p></div>
            </section>
            <section class="live-agent" data-live-agent="optimizer">
              <header><span>02</span><strong>${t("优化建模智能体")}</strong></header>
              <div class="live-lane"><b>${t("解析")}</b><p>${t("等待需求规格")}</p></div>
              <div class="live-lane"><b>${t("运行")}</b><p>${t("等待目标函数")}</p></div>
              <div class="live-lane"><b>${t("回答")}</b><p>${t("等待优化结果")}</p></div>
            </section>
            <section class="live-agent" data-live-agent="kg">
              <header><span>03</span><strong>${t("机理检索智能体")}</strong></header>
              <div class="live-lane"><b>${t("解析")}</b><p>${t("等待材料与性能查询")}</p></div>
              <div class="live-lane"><b>${t("运行")}</b><p>${t("等待图谱遍历")}</p></div>
              <div class="live-lane"><b>${t("回答")}</b><p>${t("等待机理建议")}</p></div>
            </section>
          </div>
          <section class="live-output-board">
            <h4>${t("智能体逐步输出")}</h4>
            <div class="live-output-grid">
              <article class="agent-sheet" data-live-output="requirement"><strong>${t("需求解析智能体")}</strong><div></div></article>
              <article class="agent-sheet" data-live-output="optimizer"><strong>${t("优化建模智能体")}</strong><div></div></article>
              <article class="agent-sheet" data-live-output="kg"><strong>${t("机理检索智能体")}</strong><div></div></article>
              <article class="report-sheet" data-live-output="final"><strong>${t("总报告智能体")}</strong><div></div></article>
            </div>
          </section>
        </aside>
      </div>
    </div>
  `;
}

function updateLivePhase(root, phase) {
  if (!root) return;
  const req = root.querySelector('[data-live-agent="requirement"]');
  const opt = root.querySelector('[data-live-agent="optimizer"]');
  const kg = root.querySelector('[data-live-agent="kg"]');
  [req, opt, kg].forEach((item) => item?.classList.remove("active", "completed"));
  if (phase === "optimizer") {
    req?.classList.add("completed");
    opt?.classList.add("active");
    opt.querySelectorAll(".live-lane p")[0].textContent = t("已接收需求规格");
    opt.querySelectorAll(".live-lane p")[1].className = "typing";
    opt.querySelectorAll(".live-lane p")[1].textContent = t("构造多目标问题并搜索 Pareto 解");
  } else if (phase === "kg") {
    req?.classList.add("completed");
    opt?.classList.add("completed");
    kg?.classList.add("active");
    kg.querySelectorAll(".live-lane p")[0].textContent = t("已接收优化解和性能目标");
    kg.querySelectorAll(".live-lane p")[1].className = "typing";
    kg.querySelectorAll(".live-lane p")[1].textContent = t("沿材料因果链检索");
  } else if (phase === "done") {
    [req, opt, kg].forEach((item) => item?.classList.add("completed"));
  }
}

function appendLiveEvent(root, title, detail, tone = "") {
  const log = root.querySelector(".live-event-log");
  if (!log) return;
  const row = document.createElement("div");
  row.className = `live-event ${tone}`;
  row.innerHTML = `<strong>${escapeHtml(title)}</strong><span>${escapeHtml(detail)}</span>`;
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
  const dock = root.querySelector(".dock-event");
  if (dock) dock.textContent = currentLanguage() === "en" ? `${title}: ${detail}` : `${title}：${detail}`;
}

function appendGeneration(root, payload) {
  const host = root.querySelector(".generation-stream");
  if (!host) return;
  const row = document.createElement("div");
  row.className = "generation-row";
  row.innerHTML = `
    <b>Gen ${payload.generation}/${payload.generations}</b>
    <span>HV ${payload.hypervolume}</span>
    <span>Feasible ${payload.feasible_ratio}</span>
    <span>ND ${payload.nondominated}</span>
  `;
  host.appendChild(row);
  host.scrollTop = host.scrollHeight;
  const dock = root.querySelector(".dock-generation");
  if (dock) dock.textContent = `Gen ${payload.generation}/${payload.generations}`;
  updateLiveChart(root, payload);
}

function updateLiveChart(root, payload) {
  root._generationData = root._generationData || [];
  root._generationData.push(payload);
  const data = root._generationData;
  const hvPath = root.querySelector(".hv-path");
  const feasiblePath = root.querySelector(".feasible-path");
  const pointsHost = root.querySelector(".pareto-points");
  if (!hvPath || !feasiblePath || !pointsHost) return;
  const width = 420;
  const height = 180;
  const pad = 20;
  const maxGen = Math.max(payload.generations || 1, 1);
  const hvMax = Math.max(...data.map((item) => Number(item.hypervolume) || 0), 1);
  const toLine = (items, accessor, maxY) =>
    items.map((item, index) => {
      const x = pad + ((Number(item.generation) || index + 1) / maxGen) * (width - pad * 2);
      const y = height - pad - ((Number(accessor(item)) || 0) / maxY) * (height - pad * 2);
      return `${index ? "L" : "M"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    }).join(" ");
  hvPath.setAttribute("d", toLine(data, (item) => item.hypervolume, hvMax));
  feasiblePath.setAttribute("d", toLine(data, (item) => item.feasible_ratio, 1));
  const points = payload.pareto_points || [];
  const xs = points.map((item) => Number(item[0])).filter(Number.isFinite);
  const ys = points.map((item) => Number(item[1])).filter(Number.isFinite);
  if (xs.length && ys.length) {
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    pointsHost.innerHTML = points.slice(0, 80).map((item) => {
      const x = pad + ((item[0] - minX) / Math.max(maxX - minX, 1e-9)) * (width - pad * 2);
      const y = height - pad - ((item[1] - minY) / Math.max(maxY - minY, 1e-9)) * (height - pad * 2);
      return `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="2.3"></circle>`;
    }).join("");
  }
}

function appendAgentChunk(root, agentId, chunk) {
  const host = root.querySelector(`[data-live-output="${agentId}"] div`);
  if (!host) return;
  host.dataset.raw = `${host.dataset.raw || ""}${chunk || ""}`;
  if (host._renderTimer) return;
  host._renderTimer = window.setTimeout(() => {
    host._renderTimer = null;
    if (agentId === "final") {
      host.classList.add("streaming-plain");
      host.textContent = currentLanguage() === "en" && containsChinese(host.dataset.raw)
        ? ""
        : host.dataset.raw;
    } else {
      host.innerHTML = currentLanguage() === "en" && containsChinese(host.dataset.raw)
        ? ""
        : formatAssistantAnswer(host.dataset.raw);
    }
    host.scrollTop = host.scrollHeight;
  }, agentId === "final" ? 140 : 80);
}

function finalizeAgentOutput(root, agentId) {
  const host = root.querySelector(`[data-live-output="${agentId}"] div`);
  if (!host) return;
  if (host._renderTimer) {
    window.clearTimeout(host._renderTimer);
    host._renderTimer = null;
  }
  host.classList.remove("streaming-plain");
  host.innerHTML = currentLanguage() === "en" && containsChinese(host.dataset.raw || "")
    ? ""
    : formatAssistantAnswer(host.dataset.raw || "");
  typesetMath(host);
  host.scrollTop = host.scrollHeight;
}

function appendSemanticNode(root, segment) {
  const host = root.querySelector(".semantic-rail");
  if (!host) return;
  const node = document.createElement("div");
  node.className = "semantic-node";
  node.innerHTML = `<strong>${escapeHtml(currentLanguage() === "en" ? englishSafe(segment.text, "User requirement clause") : segment.text)}</strong><span>${escapeHtml(currentLanguage() === "en" ? englishSafe(segment.role, "semantic role") : segment.role)}</span><small>${escapeHtml(currentLanguage() === "en" ? englishSafe(segment.interpretation, "engineering interpretation") : segment.interpretation)}</small>`;
  host.appendChild(node);
}

function appendObjectiveNode(root, title, expression, kind) {
  const host = root.querySelector(".objective-rail");
  if (!host) return;
  const node = document.createElement("div");
  node.className = `objective-node ${kind}`;
  node.innerHTML = `<strong>${escapeHtml(title)}</strong><span class="math-formula">\\(${escapeHtml(mathifyExpression(expression))}\\)</span>`;
  rememberMathSources(node);
  host.appendChild(node);
  typesetMath(node);
}

function kgLabel(value, fallback = "KG concept") {
  if (currentLanguage() !== "en") return value || fallback;
  return englishSafe(value, fallback);
}

function recoveredKgFallbackLabel(node = {}) {
  const id = String(node.id || "");
  const material = node.material || id.split("-")[0] || "";
  const chains = RECOVERED_KG_EN_FALLBACK[material] || RECOVERED_KG_EN_FALLBACK[String(material).toLowerCase()];
  if (!chains?.length) return "";
  const indexMatch = id.match(/-(\d+)$/);
  const row = Number.isFinite(Number(node.row)) ? Number(node.row) : (indexMatch ? Number(indexMatch[1]) : 0);
  const chain = chains[Math.max(0, row) % chains.length] || chains[0];
  if (node.type === "feature") return chain.feature;
  if (node.type === "mechanism") return chain.mechanism;
  if (node.type === "consequence") return chain.consequence;
  if (node.type === "performance") return chain.performance;
  if (node.type === "material") return localizeVariableName({ material });
  return "";
}

function kgNodeLabel(node = {}) {
  if (currentLanguage() !== "en") return node.label || node.id || "概念";
  if (node.type === "material") return localizeVariableName({ material: node.material || node.label_en || node.label || node.id });
  const explicit = node.label_en || node.name_en || node.feature_en || node.mechanism_en || node.consequence_en || node.performance_en;
  if (explicit && !containsChinese(explicit)) return explicit;
  const recovered = recoveredKgFallbackLabel(node);
  if (recovered) return recovered;
  const genericLabel = String(node.label || "");
  if (/^[A-Za-z0-9_]+\s+(feature|mechanism|consequence)$/i.test(genericLabel)) {
    const [material, type] = genericLabel.split(/\s+/);
    const fallback = recoveredKgFallbackLabel({ material, type, row: node.row || 0 });
    if (fallback) return fallback;
  }
  if (["feature", "mechanism", "consequence"].includes(node.type)) return "Mechanistic concept";
  if (node.type === "performance") return englishSafe(node.performance_en || node.performance || node.label, "performance response");
  if (node.type === "recommendation") return "Mechanistic recommendation";
  return "KG concept";
}

function kgLinkLabel(link = {}) {
  if (currentLanguage() !== "en") return link.relation || link.label || "";
  return link.relation_en || kgLabel(link.relation || link.label, "mechanistic relation");
}

function kgTypeLabel(type = "") {
  if (currentLanguage() !== "en") return type;
  return {
    material: "Material",
    feature: "Physicochemical feature",
    mechanism: "Mechanism",
    consequence: "Consequence",
    performance: "Performance response",
    recommendation: "Recommendation",
  }[type] || "KG concept";
}

function coreKgPerformance(link = {}) {
  const perf = String(link.performance_en || link.performance || "").toLowerCase();
  return perf.includes("shrinkage") || perf.includes("收缩")
    ? "shrinkage"
    : perf.includes("carbon") || perf.includes("co2") || perf.includes("低碳") || perf.includes("碳")
    ? "low_carbon"
    : (perf.includes("strength") || perf.includes("强度")
      ? "strength"
      : (perf.includes("flow") || perf.includes("workability") || perf.includes("流动") || perf.includes("工作性")
        ? "flowability"
        : ""));
}

function corePerformanceLabel(key) {
  if (currentLanguage() === "en") {
    return { shrinkage: "Shrinkage", low_carbon: "Low carbon", strength: "Compressive strength", flowability: "Flowability" }[key] || key;
  }
  return { shrinkage: "收缩", low_carbon: "低碳", strength: "强度", flowability: "流动性" }[key] || key;
}

function appendKgNode(root, node) {
  const host = root.querySelector(".kg-rail");
  if (!host) return;
  const item = document.createElement("div");
  item.className = `kg-hit ${escapeHtml(node.type || "material")}`;
  item.innerHTML = `<span></span><strong>${escapeHtml(kgNodeLabel(node))}</strong><small>${escapeHtml(node.type)}</small>`;
  host.appendChild(item);
}

function liveKgSvg(trace = {}) {
  const nodes = trace.nodes || [];
  const links = trace.links || [];
  if (!nodes.length) return "";
  const layerX = { material: 100, feature: 330, mechanism: 585, consequence: 840, performance: 1090 };
  const rowYs = {};
  nodes.forEach((node) => {
    if (!(node.row in rowYs)) rowYs[node.row] = 54 + Object.keys(rowYs).length * 76;
  });
  const pos = {};
  nodes.forEach((node) => { pos[node.id] = { x: layerX[node.type] || 70, y: rowYs[node.row] || 42 }; });
  const height = 96 + Object.keys(rowYs).length * 76;
  return `
    <svg class="live-kg-svg" viewBox="0 0 1190 ${height}">
      ${links.map((link) => {
        const a = pos[link.source];
        const b = pos[link.target];
        if (!a || !b) return "";
        return `<path data-edge="${escapeHtml(link.source)}__${escapeHtml(link.target)}" class="${link.active ? "active" : ""} ${escapeHtml(link.evidence_role || "support")}" d="M ${a.x + 74} ${a.y} C ${(a.x + b.x) / 2} ${a.y}, ${(a.x + b.x) / 2} ${b.y}, ${b.x - 74} ${b.y}"></path>`;
      }).join("")}
      ${nodes.map((node) => {
        const p = pos[node.id];
        return `<g data-node="${escapeHtml(node.id)}" class="${node.active ? "active" : ""}" transform="translate(${p.x},${p.y})"><rect x="-76" y="-22" width="152" height="44" rx="7"></rect><text text-anchor="middle">${escapeHtml(String(kgNodeLabel(node)).slice(0, 20))}</text></g>`;
      }).join("")}
    </svg>`;
}

function renderLiveKgGraph(root, trace) {
  const host = root.querySelector(".kg-rail");
  if (host) host.innerHTML = liveKgSvg(trace);
}

function activateLiveKgNode(root, node) {
  root.querySelector(`[data-node="${CSS.escape(node.id)}"]`)?.classList.add("hit");
}

function activateLiveKgLink(root, link) {
  root.querySelector(`[data-edge="${CSS.escape(`${link.source}__${link.target}`)}"]`)?.classList.add("hit");
}

function handleStreamEvent(root, msg) {
  switch (msg.event) {
    case "requirement_started":
      appendLiveEvent(root, currentLanguage() === "en" ? "Read user request" : "读取用户请求", msg.raw_request, "accent");
      break;
    case "requirement_llm_spec":
      appendLiveEvent(root, currentLanguage() === "en" ? "LLM structured specification" : "LLM 结构化需求", currentLanguage() === "en" ? `target strength ${msg.validated.target_mpa} MPa; curing age ${msg.validated.age_days} d` : `目标强度 ${msg.validated.target_mpa} MPa；龄期 ${msg.validated.age_days} d`, "accent");
      break;
    case "requirement_segment":
      appendLiveEvent(root, currentLanguage() === "en" ? `Semantic segment ${msg.index}` : `语义切分 ${msg.index}`, currentLanguage() === "en" ? `${englishSafe(msg.segment.text, "user clause")} -> ${englishSafe(msg.segment.role, "role")} -> ${englishSafe(msg.segment.interpretation, "engineering interpretation")}` : `${msg.segment.text} -> ${msg.segment.role} -> ${msg.segment.interpretation}`);
      appendSemanticNode(root, msg.segment);
      break;
    case "requirement_mapping":
      appendLiveEvent(root, currentLanguage() === "en" ? `Field mapping ${msg.index}` : `字段映射 ${msg.index}`, currentLanguage() === "en" ? `${englishSafe(msg.mapping.token, "input token")} -> ${enField(msg.mapping.field)}: ${enTraceValue(msg.mapping)}` : `${msg.mapping.token} -> ${msg.mapping.field}: ${msg.mapping.value}`);
      break;
    case "requirement_completed":
      updateLivePhase(root, "optimizer");
      appendLiveEvent(root, currentLanguage() === "en" ? "Requirement specification completed" : "需求规格完成", currentLanguage() === "en" ? `priorities: ${englishList(msg.requirements.priorities)}` : `优先级：${(msg.requirements.priorities || []).join(", ")}`, "success");
      break;
    case "optimizer_started":
      appendLiveEvent(root, currentLanguage() === "en" ? "Start optimization modeling" : "启动优化建模", currentLanguage() === "en" ? "Assembling variables, objective functions, and constraints" : "开始装配变量、目标函数和约束", "accent");
      break;
    case "optimizer_llm_proposal":
      appendLiveEvent(root, currentLanguage() === "en" ? "LLM modeling proposal" : "LLM 建模提案", currentLanguage() === "en" ? `objectives: ${englishList(msg.proposal.objectives)}` : `目标：${(msg.proposal.objectives || []).join(", ")}`);
      break;
    case "optimizer_bounds":
      appendLiveEvent(root, currentLanguage() === "en" ? "Variable bounds" : "变量边界", currentLanguage() === "en" ? `assembled ${msg.bounds.length} design variables` : `装配 ${msg.bounds.length} 个设计变量`);
      break;
    case "optimizer_objective":
      appendLiveEvent(root, t("目标函数"), `${englishSafe(msg.objective.name, msg.objective.name)}: ${englishSafe(msg.objective.goal, msg.objective.goal)} ${msg.objective.expression}`);
      appendObjectiveNode(root, msg.objective.name, msg.objective.expression, "objective");
      break;
    case "optimizer_constraint":
      appendLiveEvent(root, t("约束条件"), currentLanguage() === "en" ? englishSafe(msg.constraint, "structured constraint") : msg.constraint);
      appendObjectiveNode(root, "Constraint", msg.constraint, "constraint");
      break;
    case "optimizer_progress":
      if (msg.phase === "population_initialized") appendLiveEvent(root, t("初始化种群"), currentLanguage() === "en" ? `${msg.algorithm}, population ${msg.population_size}` : `${msg.algorithm}，种群 ${msg.population_size}`);
      if (msg.phase === "generation_completed") appendGeneration(root, msg);
      break;
    case "optimizer_completed":
      updateLivePhase(root, "kg");
      appendLiveEvent(root, t("优化完成"), currentLanguage() === "en" ? `algorithm ${msg.best_algorithm}; predicted strength ${msg.kpis.strength_mpa ?? "-"}` : `算法 ${msg.best_algorithm}；预测强度 ${msg.kpis.strength_mpa ?? "-"}`, "success");
      break;
    case "kg_started":
      appendLiveEvent(root, t("构造图谱查询"), currentLanguage() === "en" ? englishList(msg.query_terms, "KG query terms") : (msg.query_terms || []).join(", "), "accent");
      break;
    case "evidence_query":
      appendLiveEvent(root, t("统一证据查询"), currentLanguage() === "en" ? `literature ${msg.counts?.literature || 0}, standards ${msg.counts?.standard || 0}, curated KG ${msg.counts?.kg || 0}, automatic KG ${msg.counts?.auto_kg || 0}` : `文献 ${msg.counts?.literature || 0}，规范 ${msg.counts?.standard || 0}，精选 KG ${msg.counts?.kg || 0}，自动 KG ${msg.counts?.auto_kg || 0}`);
      break;
    case "evidence_candidate":
      appendLiveEvent(root, t("候选证据"), `${msg.evidence.source_type} | ${currentLanguage() === "en" ? englishSafe(msg.evidence.title, "evidence item") : msg.evidence.title} | score ${msg.evidence.score}`);
      break;
    case "evidence_ranked":
      appendLiveEvent(root, t("证据重排完成"), currentLanguage() === "en" ? `retained ${msg.selected.length} high-relevance evidence items; extracted ${msg.routes.length} multi-hop routes` : `保留 ${msg.selected.length} 条高相关证据，抽取 ${msg.routes.length} 条多跳路线`, "success");
      break;
    case "kg_graph_loaded":
      renderLiveKgGraph(root, msg.graph);
      appendLiveEvent(root, t("展开命中子图"), currentLanguage() === "en" ? `${msg.graph.nodes.length} hit nodes, ${msg.graph.links.length} hit relations` : `${msg.graph.nodes.length} 个命中节点，${msg.graph.links.length} 条命中关系`);
      break;
    case "kg_node":
      appendLiveEvent(root, t("命中节点"), `${msg.node.type}: ${kgNodeLabel(msg.node)}`);
      activateLiveKgNode(root, msg.node);
      break;
    case "kg_link":
      appendLiveEvent(root, t("扩展关系"), `${msg.link.source} -> ${kgLinkLabel(msg.link)} -> ${msg.link.target}`);
      activateLiveKgLink(root, msg.link);
      break;
    case "kg_completed":
      appendLiveEvent(root, t("图谱检索完成"), currentLanguage() === "en" ? `generated ${msg.recommendations.length} recommendations` : `生成 ${msg.recommendations.length} 条建议`, "success");
      break;
    case "kg_revision_decision":
      appendLiveEvent(root, t("机理复审"), currentLanguage() === "en" ? (msg.review.trigger_second_pass ? `trigger second-pass optimization: ${englishSafe(msg.review.reason, "mechanistic risk detected")}` : `do not trigger second-pass optimization: ${englishSafe(msg.review.reason, "no material revision required")}`) : (msg.review.trigger_second_pass ? `触发二次优化：${msg.review.reason}` : `不触发二次优化：${msg.review.reason}`));
      break;
    case "optimizer_revision_started":
      appendLiveEvent(root, t("启动二次优化"), currentLanguage() === "en" ? `apply ${msg.adjustments.length} mechanistic adjustments` : `应用 ${msg.adjustments.length} 项机理修正`, "accent");
      break;
    case "optimizer_revision_completed":
      appendLiveEvent(root, t("二次优化完成"), currentLanguage() === "en" ? `predicted strength ${msg.kpis.strength_mpa ?? "-"}` : `预测强度 ${msg.kpis.strength_mpa ?? "-"}`, "success");
      break;
    case "optimizer_revision_failed":
      appendLiveEvent(root, t("二次优化失败"), currentLanguage() === "en" ? englishSafe(msg.message, "second-pass optimization did not find a feasible revision") : msg.message);
      break;
    case "report_started":
      updateLivePhase(root, "done");
      appendLiveEvent(root, t("启动报告生成"), t("调用三个子智能体与总报告智能体"), "accent");
      break;
    case "agent_output_started":
      appendLiveEvent(root, t("开始生成文本"), currentLanguage() === "en" ? englishSafe(msg.agent_name, "Agent") : msg.agent_name, "accent");
      break;
    case "agent_output_chunk":
      appendAgentChunk(root, msg.agent_id, msg.chunk);
      break;
    case "agent_output_completed":
      finalizeAgentOutput(root, msg.agent_id);
      appendLiveEvent(root, t("文本输出完成"), msg.agent_id, "success");
      break;
  }
}

function orchestrationHtml(agents = []) {
  const labels = [
    ["requirement", localizeAgentShortLabel("requirement"), currentLanguage() === "en" ? "Natural language" : "Natural language"],
    ["optimizer", localizeAgentShortLabel("optimizer"), currentLanguage() === "en" ? "Objectives and constraints" : "Objectives & constraints"],
    ["kg", localizeAgentShortLabel("kg"), currentLanguage() === "en" ? "Causal KG" : "Causal KG"],
  ];
  return `
    <section class="orchestration-board">
      <div class="orchestration-line"></div>
      ${labels.map(([id, title, subtitle], index) => {
        const agent = agents.find((item) => item.id === id) || {};
        return `
          <div class="orchestration-step" style="--i:${index}">
            <span>${index + 1}</span>
            <strong>${title}</strong>
            <small>${subtitle}</small>
            <p>${escapeHtml(localizeAgentSummary(agent))}</p>
          </div>
        `;
      }).join("")}
    </section>
  `;
}

function roleEvidenceRoutingHtml(data = {}) {
  const bundles = data.role_evidence || {};
  const rows = [
    ["requirement_analysis", localizeAgentShortLabel("requirement_analysis")],
    ["optimization_modeling", localizeAgentShortLabel("optimization_modeling")],
    ["mechanism_retrieval", localizeAgentShortLabel("mechanism_retrieval")],
    ["report_generation", localizeAgentShortLabel("report_generation")],
  ];
  if (!Object.keys(bundles).length) return "";
  const isEnglish = currentLanguage() === "en";
  return `
    <section class="role-rag-board">
      <header><strong>${isEnglish ? "Role-Specific RAG Routing" : "角色化 RAG 路由"}</strong><span>${isEnglish ? "Each agent retrieves evidence independently according to the Skill v2 policy." : "每个智能体按 Skill v2 检索策略独立召回证据"}</span></header>
      <div>
        ${rows.map(([key, label], index) => {
          const bundle = bundles[key] || {};
          const sources = bundle.policy?.primary_sources || [];
          const citations = bundle.citations || [];
          return `
            <article style="--i:${index}">
              <h4>${escapeHtml(label)}</h4>
              <p>${sources.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</p>
              <strong>${isEnglish ? `${citations.length} evidence items` : `${citations.length} 条证据`}</strong>
              <small>${escapeHtml(isEnglish ? (bundle.query_terms || []).slice(0, 8).map((item) => englishSafe(item, "query term")).join(" / ") : (bundle.query_terms || []).slice(0, 8).join(" / "))}</small>
            </article>
          `;
        }).join("")}
      </div>
    </section>
  `;
}

function requirementHtml(agent) {
  const details = agent?.details || {};
  const trace = agent?.trace || [];
  const priorities = details.priority_ranking || [];
  const segments = details.raw_segments || [];
  if (currentLanguage() === "en") {
    return `
      <section class="main-process-card requirement-card" style="--step:0">
        <div class="process-title"><span>01</span><strong>Requirement Parsing Agent</strong></div>
        <p class="process-copy">The original request is segmented into semantic clauses and mapped to hard constraints, soft objectives, material policies, and knowledge-graph query conditions.</p>
        <div class="agent-triptych">
          <div><b>Parse</b><p>Segment natural language into strength, curing-age, material, and performance clauses.</p></div>
          <div><b>Operate</b><p>Map clauses to computable variables, hard constraints, soft objectives, and downstream graph-retrieval conditions.</p></div>
          <div><b>Output</b><p>${escapeHtml(localizeAgentSummary(agent || { id: "requirement" }))}</p></div>
        </div>
        <div class="raw-segment-flow">
          ${segments.map((item, index) => `
            <div style="--i:${index}">
              <b>${escapeHtml(englishSafe(item.text, "User requirement clause"))}</b>
              <span>${escapeHtml(englishSafe(item.role, "semantic role"))}</span>
              <small>${escapeHtml(enInterpretation(item.interpretation))}</small>
            </div>
          `).join("")}
        </div>
        <div class="token-flow">
          ${trace.map((item) => `<span>${escapeHtml(englishSafe(item.token, "input token"))}<em>${escapeHtml(enField(item.field))}: ${escapeHtml(enTraceValue(item))}</em><small>${escapeHtml(enReason(item.reason || ""))}</small></span>`).join("")}
        </div>
        <div class="priority-stack">
          ${priorities.map((item, index) => `<div><b>${index + 1}</b><strong>${escapeHtml(enInterpretation(item.interpretation || item.priority))}</strong><small>${escapeHtml(enReason(item.reason))}</small></div>`).join("")}
        </div>
      </section>
    `;
  }
  return `
    <section class="main-process-card requirement-card" style="--step:0">
      <div class="process-title"><span>01</span><strong>需求解析智能体</strong></div>
      <p class="process-copy">原始自然语言首先被切分为语义片段，再映射为硬约束、软目标、材料偏好和知识图谱查询条件。</p>
      <div class="agent-triptych">
        <div><b>解析</b><p>将自然语言切分为语义片段，并判断它们属于强度指标、龄期输入、材料条件还是性能偏好。</p></div>
        <div><b>运作</b><p>把片段映射为可计算变量、硬约束、软目标和后续图谱检索条件。</p></div>
        <div><b>回答</b><p>${escapeHtml(agent?.summary || "需求规格已生成")}</p></div>
      </div>
      ${agentStepsHtml(agent)}
      <div class="raw-segment-flow">
        ${segments.map((item, index) => `
          <div style="--i:${index}">
            <b>${escapeHtml(item.text)}</b>
            <span>${escapeHtml(item.role)}</span>
            <small>${escapeHtml(item.interpretation)}</small>
          </div>
        `).join("")}
      </div>
      <div class="token-flow">
        ${trace.map((item) => `<span>${escapeHtml(item.token)}<em>${escapeHtml(item.field)}: ${escapeHtml(item.value)}</em><small>${escapeHtml(item.reason || "")}</small></span>`).join("")}
      </div>
      <div class="priority-stack">
        ${priorities.map((item, index) => `<div><b>${index + 1}</b><strong>${escapeHtml(item.interpretation)}</strong><small>${escapeHtml(item.reason)}</small></div>`).join("")}
      </div>
      ${requirementFigureSuiteHtml(details, trace)}
      ${semanticAuditHtml(details)}
      ${agentDialogueHtml(agent)}
    </section>
  `;
}

function optimizationHtml(agent, optimization, plots = []) {
  const trace = agent?.trace || {};
  const figureGroups = [
    ["Optimization landscape", ["Pareto Front", "Parallel Coordinates", "Trade-off Matrix", "Objective Correlation"]],
    ["Search dynamics", ["Progress History", "Spacing History", "Algorithm Hypervolume History"]],
    ["Decision support", ["TOPSIS Screening", "Ranking Agreement", "Solution Score Heatmap", "Best Solution Profile", "Algorithm Performance Matrix", "Algorithm Ranking Agreement"]],
  ];
  if (currentLanguage() === "en") {
    return `
      <section class="main-process-card optimizer-card" style="--step:1">
        <div class="process-title"><span>02</span><strong>Optimization Modeling Agent</strong></div>
        <p class="process-copy">The requirement specification is converted into objective functions, constraints, and a searchable design space for multi-objective optimization.</p>
        <div class="agent-triptych">
          <div><b>Parse</b><p>Receive target strength, curing age, low-carbon priority, and flowability priority, then determine fixed inputs and searchable variables.</p></div>
          <div><b>Operate</b><p>Construct objective functions and constraints, run the multi-objective algorithm, screen Pareto solutions, and select the recommended scheme.</p></div>
          <div><b>Output</b><p>${escapeHtml(localizeAgentSummary(agent || { id: "optimizer" }))}</p></div>
        </div>
        ${agentStepsHtml(agent)}
        <div class="opt-meta">
          <div><strong>${escapeHtml(trace.algorithm || "-")}</strong><small>Main algorithm</small></div>
          <div><strong>${escapeHtml(trace.bounds_count ?? "-")}</strong><small>Variables</small></div>
          <div><strong>${escapeHtml(optimization?.age_days ?? "-")} d</strong><small>Curing age</small></div>
          <div><strong>${escapeHtml(trace.material_policy?.mode === "specified" ? "Specified materials" : "Exploratory mix design")}</strong><small>Mix-design policy</small></div>
        </div>
        ${formulasHtml(trace)}
        ${constraintStatusHtml(optimization?.constraint_status || trace.constraint_status || {})}
        <div class="kpi-grid main-kpis">${kpiCardsHtml(optimization?.kpis || {})}</div>
        ${researchConclusionHtml(optimization)}
        ${scienceFigureSuiteHtml(optimization)}
        ${agentDialogueHtml(agent)}
      </section>
    `;
  }
  return `
    <section class="main-process-card optimizer-card" style="--step:1">
      <div class="process-title"><span>02</span><strong>优化建模智能体</strong></div>
      <p class="process-copy">将需求转换为目标函数、约束条件和搜索空间，并调用多目标优化模型。</p>
      <div class="agent-triptych">
        <div><b>解析</b><p>接收目标强度、龄期、低碳和流动性优先级，确定哪些变量固定、哪些变量进入搜索空间。</p></div>
        <div><b>运作</b><p>构造目标函数与约束，运行多目标算法，筛选 Pareto 解并选出推荐方案。</p></div>
        <div><b>回答</b><p>${escapeHtml(agent?.summary || "优化已完成")}</p></div>
      </div>
      ${agentStepsHtml(agent)}
      <div class="opt-meta">
        <div><strong>${escapeHtml(trace.algorithm || "-")}</strong><small>主算法</small></div>
        <div><strong>${escapeHtml(trace.bounds_count ?? "-")}</strong><small>变量数</small></div>
        <div><strong>${escapeHtml(optimization?.age_days ?? "-")} d</strong><small>龄期</small></div>
        <div><strong>${escapeHtml(trace.material_policy?.mode === "specified" ? "指定材料" : "探索配方")}</strong><small>配方策略</small></div>
      </div>
      ${formulasHtml(trace)}
      ${constraintStatusHtml(optimization?.constraint_status || trace.constraint_status || {})}
      <div class="kpi-grid main-kpis">${kpiCardsHtml(optimization?.kpis || {})}</div>
      ${researchConclusionHtml(optimization)}
      ${scienceFigureSuiteHtml(optimization)}
      <div class="figure-suite">
        ${figureGroups.map(([title, names]) => {
          const items = (plots || []).filter((plot) => names.includes(plot.name));
          if (!items.length) return "";
          return `
            <section>
              <h4>${escapeHtml(title)}</h4>
              <div class="plot-grid">
                ${items.map((plot) => `<figure><img src="${plot.image}" alt="${escapeHtml(plot.name)}" /><figcaption>${escapeHtml(plot.name)}</figcaption></figure>`).join("")}
              </div>
            </section>
          `;
        }).join("")}
      </div>
      ${agentDialogueHtml(agent)}
    </section>
  `;
}

function svgFrame(title, body, width = 320, height = 210, caption = "") {
  return `
    <figure class="science-figure">
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(title)}">
        <text x="16" y="22" class="fig-title">${escapeHtml(title)}</text>
        ${body}
      </svg>
      ${caption ? `<figcaption>${escapeHtml(caption)}</figcaption>` : ""}
    </figure>
  `;
}

function researchConclusionHtml(optimization = {}) {
  const rows = optimization.scheme_comparison || [];
  const best = rows.find((row) => (row.roles || []).includes("综合折中推荐")) || rows[0] || {};
  const lowCarbon = rows.find((row) => (row.roles || []).includes("最低碳候选"));
  const strongest = rows.find((row) => (row.roles || []).includes("最高强候选"));
  if (currentLanguage() === "en") {
    return `
      <section class="research-conclusions">
        <article><span>Recommendation logic</span><p>${escapeHtml(localizeSchemeLabel(best.label || "综合推荐方案"))} represents a compromise solution rather than a single-objective extreme.</p></article>
        <article><span>Low-carbon reference</span><p>${lowCarbon ? `${escapeHtml(localizeSchemeLabel(lowCarbon.label))} is used as the current lowest-carbon reference.` : "No independent lowest-carbon reference is currently available."}</p></article>
        <article><span>Strength reference</span><p>${strongest ? `${escapeHtml(localizeSchemeLabel(strongest.label))} is used as the current highest-strength reference.` : "No independent highest-strength reference is currently available."}</p></article>
      </section>
    `;
  }
  return `
    <section class="research-conclusions">
      <article><span>推荐逻辑</span><p>${escapeHtml(best.label || "综合推荐方案")} 被用于代表折中解，而不是单目标极端。</p></article>
      <article><span>低碳参照</span><p>${lowCarbon ? `${escapeHtml(lowCarbon.label)} 构成当前最低碳对照。` : "当前暂无独立最低碳对照。"}</p></article>
      <article><span>强度参照</span><p>${strongest ? `${escapeHtml(strongest.label)} 构成当前最高强对照。` : "当前暂无独立最高强对照。"}</p></article>
    </section>
  `;
}

function lineChartBody(values = [], color = "#0f766e", yLabel = "", xLabel = "Iteration") {
  if (!values.length) return '<text x="160" y="108" text-anchor="middle" class="fig-empty">No data</text>';
  const xs = values.map((_, i) => 34 + i * (252 / Math.max(values.length - 1, 1)));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const ys = values.map((value) => 176 - ((value - min) / Math.max(max - min, 1e-9)) * 126);
  const path = xs.map((x, i) => `${i ? "L" : "M"} ${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(" ");
  return `
    <line x1="34" y1="176" x2="286" y2="176" class="fig-axis"></line>
    <line x1="34" y1="50" x2="34" y2="176" class="fig-axis"></line>
    <path d="${path}" fill="none" stroke="${color}" stroke-width="2.4"></path>
    ${xs.map((x, i) => `<circle cx="${x.toFixed(1)}" cy="${ys[i].toFixed(1)}" r="2.4" fill="${color}"></circle>`).join("")}
    <text x="18" y="112" class="fig-label" transform="rotate(-90 18 112)">${escapeHtml(yLabel)}</text>
    <text x="160" y="201" text-anchor="middle" class="fig-label">${escapeHtml(xLabel)}</text>
  `;
}

function barChartBody(items = [], color = "#0f766e", yLabel = "Value") {
  if (!items.length) return '<text x="160" y="108" text-anchor="middle" class="fig-empty">No data</text>';
  const max = Math.max(...items.map((item) => Number(item.value) || 0), 1);
  return `
    <line x1="34" y1="176" x2="286" y2="176" class="fig-axis"></line>
    <line x1="34" y1="50" x2="34" y2="176" class="fig-axis"></line>
    ${items.map((item, i) => {
      const width = 220 / items.length;
      const h = ((Number(item.value) || 0) / max) * 118;
      const x = 42 + i * width;
      const y = 176 - h;
      return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(width - 8, 8).toFixed(1)}" height="${h.toFixed(1)}" fill="${color}"></rect><text x="${(x + width / 2 - 4).toFixed(1)}" y="193" text-anchor="middle" class="fig-label">${escapeHtml(item.label)}</text>`;
    }).join("")}
    <text x="18" y="112" class="fig-label" transform="rotate(-90 18 112)">${escapeHtml(yLabel)}</text>
  `;
}

function scatterBody(points = [], xLabel = "Carbon", yLabel = "Strength") {
  if (!points.length) return '<text x="160" y="108" text-anchor="middle" class="fig-empty">No data</text>';
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  return `
    <line x1="34" y1="176" x2="286" y2="176" class="fig-axis"></line>
    <line x1="34" y1="50" x2="34" y2="176" class="fig-axis"></line>
    ${points.map((point, index) => {
      const x = 34 + ((point.x - minX) / Math.max(maxX - minX, 1e-9)) * 252;
      const y = 176 - ((point.y - minY) / Math.max(maxY - minY, 1e-9)) * 126;
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${index === 0 ? 4.3 : 3}" class="${index === 0 ? "fig-point selected" : "fig-point"}"></circle>`;
    }).join("")}
    <text x="160" y="201" text-anchor="middle" class="fig-label">${escapeHtml(xLabel)}</text>
    <text x="18" y="112" class="fig-label" transform="rotate(-90 18 112)">${escapeHtml(yLabel)}</text>
  `;
}

function scienceFigureSuiteHtml(optimization = {}) {
  const history = optimization.history || [];
  const sols = optimization.top_solutions || [];
  const rows = optimization.solution_table || [];
  const objectives = optimization.objectives || [];
  const objectiveNames = objectives.map((item) => item.name);
  const paretoPoints = sols
    .filter((item) => Number.isFinite(Number(item.Carbon)) && Number.isFinite(Number(item.prediction)))
    .map((item) => ({ x: Number(item.Carbon), y: Number(item.prediction) }));
  const materialRows = rows.filter((row) => row.unit === "kg/m3").slice(0, 8);
  const carbonItems = materialRows.map((row) => ({ label: String(row.name).slice(0, 4), value: Number(row.value) || 0 }));
  const scoreItems = sols.slice(0, 8).map((item, index) => ({ label: String(index + 1), value: Number(item["TOPSIS Score"] ?? item["Compromise Rank"] ?? 0) }));
  const strengthItems = sols.slice(0, 8).map((item, index) => ({ label: String(index + 1), value: Number(item.prediction ?? item.StrengthTarget ?? 0) }));
  const carbonRankItems = sols.slice(0, 8).map((item, index) => ({ label: String(index + 1), value: Number(item.Carbon ?? 0) }));
  const compItems = (optimization.comparison_metrics || []).slice(0, 6).map((item) => ({ label: String(item.Algorithm || "").slice(0, 4), value: Number(item.Hypervolume ?? 0) }));
  const feasibleAlgoItems = (optimization.comparison_metrics || []).slice(0, 6).map((item) => ({ label: String(item.Algorithm || "").slice(0, 4), value: Number(item["Feasible Ratio"] ?? 0) }));
  const compositionTotal = materialRows.reduce((sum, row) => sum + (Number(row.value) || 0), 0);
  const compositionBody = materialRows.length ? `
    ${materialRows.map((row, index) => {
      const prev = materialRows.slice(0, index).reduce((sum, item) => sum + (Number(item.value) || 0), 0);
      const x = 34 + (prev / Math.max(compositionTotal, 1)) * 252;
      const w = ((Number(row.value) || 0) / Math.max(compositionTotal, 1)) * 252;
      const colors = ["#0f766e", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#b56576", "#6c757d", "#457b9d"];
      return `<rect x="${x.toFixed(1)}" y="84" width="${w.toFixed(1)}" height="38" fill="${colors[index % colors.length]}"></rect>`;
    }).join("")}
    <line x1="34" y1="140" x2="286" y2="140" class="fig-axis"></line>
  ` : '<text x="160" y="108" text-anchor="middle" class="fig-empty">No data</text>';

  return `
    <section class="science-suite">
      <div class="suite-head">
        <span>Optimization evidence</span>
        <strong>Synchronized optimization panels</strong>
      </div>
      <div class="science-grid">
        ${svgFrame("A. Hypervolume convergence", lineChartBody(history.map((item) => Number(item.hypervolume)), "#0f766e", "HV"), 320, 210, "Convergence of archive quality across iterations.")}
        ${svgFrame("B. Feasible ratio", lineChartBody(history.map((item) => Number(item.feasible_ratio)), "#b45309", "Ratio"), 320, 210, "Share of feasible candidates during the search.")}
        ${svgFrame("C. Nondominated count", lineChartBody(history.map((item) => Number(item.nondominated)), "#2563eb", "Count"), 320, 210, "Number of nondominated solutions retained.")}
        ${svgFrame("D. Spacing metric", lineChartBody(history.map((item) => Number(item.spacing)), "#7c3aed", "Spacing"), 320, 210, "Distribution regularity of Pareto solutions.")}
        ${svgFrame("E. Carbon-strength Pareto", scatterBody(paretoPoints, "Carbon", "Strength"), 320, 210, "Trade-off between embodied carbon and predicted strength.")}
        ${svgFrame("F. Binder composition", compositionBody, 320, 210, "Mass share of principal constituents in the selected mix.")}
        ${svgFrame("G. Recommended mix masses", barChartBody(carbonItems, "#2a9d8f", "kg/m3"), 320, 210, "Recommended material dosage profile.")}
        ${svgFrame("H. Top-solution scores", barChartBody(scoreItems, "#457b9d", "Score"), 320, 210, "Decision score among leading candidates.")}
        ${svgFrame("I. Strength among top solutions", barChartBody(strengthItems, "#0f766e", "MPa"), 320, 210, "Predicted strength of shortlisted candidates.")}
        ${svgFrame("J. Carbon among top solutions", barChartBody(carbonRankItems, "#b45309", "kgCO2e"), 320, 210, "Carbon intensity of shortlisted candidates.")}
        ${svgFrame("K. Algorithm hypervolume", barChartBody(compItems, "#6b7280", "HV"), 320, 210, "Cross-algorithm archive quality comparison.")}
        ${svgFrame("L. Algorithm feasible ratio", barChartBody(feasibleAlgoItems, "#8ab17d", "Ratio"), 320, 210, "Cross-algorithm feasibility comparison.")}
      </div>
    </section>
  `;
}

function requirementFigureSuiteHtml(details = {}, trace = []) {
  const segments = details.raw_segments || [];
  const priorities = details.priority_ranking || [];
  const roleCounts = Object.entries(segments.reduce((acc, item) => {
    acc[item.role] = (acc[item.role] || 0) + 1;
    return acc;
  }, {})).map(([label, value]) => ({ label: label.slice(0, 4), value }));
  const priorityBars = priorities.map((item, index) => ({ label: String(index + 1), value: priorities.length - index }));
  const entityBars = trace.slice(0, 8).map((item) => ({ label: String(item.field).slice(0, 4), value: String(item.value || "").length }));
  const sankeyBody = segments.length ? `
    ${segments.map((item, index) => {
      const y = 54 + index * 24;
      return `
        <rect x="24" y="${y}" width="82" height="16" rx="4" fill="#e7f4f1"></rect>
        <text x="65" y="${y + 11}" text-anchor="middle" class="fig-label">${escapeHtml(item.text.slice(0, 7))}</text>
        <path d="M 106 ${y + 8} C 132 ${y + 8}, 146 ${84 + index * 18}, 172 ${84 + index * 18}" class="fig-flow"></path>
        <rect x="172" y="${76 + index * 18}" width="104" height="16" rx="4" fill="#f5ead7"></rect>
        <text x="224" y="${87 + index * 18}" text-anchor="middle" class="fig-label">${escapeHtml(item.role)}</text>
      `;
    }).join("")}
  ` : '<text x="160" y="108" text-anchor="middle" class="fig-empty">No data</text>';
  return `
    <section class="science-suite compact-suite">
      <div class="suite-head"><span>Requirement evidence</span><strong>Semantic extraction</strong></div>
      <div class="science-grid">
        ${svgFrame("A. Semantic routing", sankeyBody, 320, 210, "Mapping from raw clauses to structured requirement classes.")}
        ${svgFrame("B. Segment classes", barChartBody(roleCounts, "#0f766e", "Count"), 320, 210, "Distribution of recognized semantic segment types.")}
        ${svgFrame("C. Priority ranking", barChartBody(priorityBars, "#b45309", "Rank score"), 320, 210, "Relative priority order inferred from the request.")}
        ${svgFrame("D. Parsed entity load", barChartBody(entityBars, "#2563eb", "Text length"), 320, 210, "Relative load of parsed entities passed downstream.")}
      </div>
    </section>
  `;
}

function kgFigureSuiteHtml(trace = {}) {
  const nodes = trace.nodes || [];
  const links = trace.links || [];
  const layerItems = Object.entries(nodes.reduce((acc, node) => {
    acc[node.type] = (acc[node.type] || 0) + 1;
    return acc;
  }, {})).map(([label, value]) => ({ label: label.slice(0, 4), value }));
  const perfItems = Object.entries(links.reduce((acc, link) => {
    const key = coreKgPerformance(link);
    if (!key) return acc;
    const label = corePerformanceLabel(key);
    acc[label] = (acc[label] || 0) + 1;
    return acc;
  }, {})).map(([label, value]) => ({ label: label.slice(0, 4), value }));
  const pos = links.filter((link) => String(link.polarity || "").includes("提高")).length;
  const neg = links.filter((link) => String(link.polarity || "").includes("降低")).length;
  const polarityBody = barChartBody([{ label: "pos", value: pos }, { label: "neg", value: neg }], "#7c3aed");
  const materialItems = Object.entries(nodes.filter((node) => node.type === "material").reduce((acc, node) => {
    const label = kgNodeLabel(node);
    acc[label] = (acc[label] || 0) + 1;
    return acc;
  }, {})).map(([label, value]) => ({ label: label.slice(0, 4), value }));
  return `
    <section class="science-suite compact-suite">
      <div class="suite-head"><span>Mechanism evidence</span><strong>Retrieved graph evidence</strong></div>
      <div class="science-grid">
        ${svgFrame("A. Node-layer composition", barChartBody(layerItems, "#0f766e", "Count"), 320, 210, "Distribution of retrieved graph nodes by layer.")}
        ${svgFrame("B. Core performance evidence", barChartBody(perfItems, "#b45309", "Count"), 320, 210, "Evidence counts for shrinkage, low carbon, strength, and flowability.")}
        ${svgFrame("C. Polarity balance", polarityBody, 320, 210, "Balance between supporting and risk-bearing relations.")}
        ${svgFrame("D. Hit material coverage", barChartBody(materialItems, "#2563eb", "Count"), 320, 210, "Coverage of retrieved material concepts.")}
      </div>
    </section>
  `;
}

function publicationKgMapHtml(trace = {}) {
  const nodes = trace.nodes || [];
  const links = (trace.links || []).filter((link) => coreKgPerformance(link));
  if (!nodes.length || !links.length) return "";
  const byId = Object.fromEntries(nodes.map((node) => [node.id, node]));
  const layerOrder = ["material", "feature", "mechanism", "consequence", "performance"];
  const layerLabels = currentLanguage() === "en"
    ? ["Material", "Feature", "Mechanism", "Consequence", "Performance"]
    : ["材料", "特征", "机理", "后果", "性能"];
  const coreLinks = links.filter((link) => {
    const target = byId[link.target];
    const source = byId[link.source];
    if (!target || !source) return false;
    if (target.type === "performance") return Boolean(coreKgPerformance(link));
    return true;
  });
  const linkedIds = new Set(coreLinks.flatMap((link) => [link.source, link.target]));
  const layered = Object.fromEntries(layerOrder.map((layer) => [layer, nodes.filter((node) => node.type === layer && linkedIds.has(node.id)).slice(0, 8)]));
  const width = 1120;
  const height = Math.max(330, 110 + Math.max(...Object.values(layered).map((items) => items.length), 1) * 58);
  const xByLayer = { material: 92, feature: 318, mechanism: 558, consequence: 798, performance: 1030 };
  const yPos = {};
  layerOrder.forEach((layer) => {
    const items = layered[layer];
    const gap = Math.min(70, Math.max(46, (height - 128) / Math.max(items.length, 1)));
    items.forEach((node, index) => {
      yPos[node.id] = 86 + index * gap;
    });
  });
  const nodeColor = {
    material: "#e8f2ff",
    feature: "#edf7ef",
    mechanism: "#fff4dc",
    consequence: "#f4efff",
    performance: "#fff1f1",
  };
  const edgeHtml = coreLinks.map((link) => {
    const source = byId[link.source];
    const target = byId[link.target];
    if (!source || !target || yPos[source.id] === undefined || yPos[target.id] === undefined) return "";
    const key = coreKgPerformance(link);
    const cls = key ? `core-${key}` : "";
    const x1 = xByLayer[source.type] + 76;
    const x2 = xByLayer[target.type] - 76;
    const y1 = yPos[source.id];
    const y2 = yPos[target.id];
    const mx = (x1 + x2) / 2;
    return `<path class="pub-kg-edge ${cls}" d="M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}"></path>`;
  }).join("");
  const nodeHtml = layerOrder.flatMap((layer) => layered[layer].map((node) => {
    const label = kgNodeLabel(node);
    return `
      <g class="pub-kg-node ${escapeHtml(layer)}" transform="translate(${xByLayer[layer]},${yPos[node.id]})">
        <rect x="-82" y="-22" width="164" height="44" rx="5" fill="${nodeColor[layer]}"></rect>
        <text text-anchor="middle" y="-2">${escapeHtml(label.length > 20 ? `${label.slice(0, 20)}...` : label)}</text>
        <text class="pub-kg-type" text-anchor="middle" y="14">${escapeHtml(kgTypeLabel(layer))}</text>
      </g>
    `;
  })).join("");
  const perfCounts = links.reduce((acc, link) => {
    const key = coreKgPerformance(link);
    if (key) acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  return `
    <section class="publication-kg-map">
      <header>
        <div>
          <span>${currentLanguage() === "en" ? "Publication KG View" : "论文式知识图谱视图"}</span>
          <strong>${currentLanguage() === "en" ? "Low-carbon and low-shrinkage causal routes" : "低碳-低收缩因果链"}</strong>
        </div>
        <p>${currentLanguage() === "en" ? "The manuscript-facing KG focuses on low carbon, shrinkage, compressive strength, and flowability." : "论文主图聚焦低碳、收缩、强度和流动性四类性能。"}</p>
      </header>
      <div class="pub-kg-legend">
        <span class="core-low_carbon">${corePerformanceLabel("low_carbon")} ${perfCounts.low_carbon || 0}</span>
        <span class="core-shrinkage">${corePerformanceLabel("shrinkage")} ${perfCounts.shrinkage || 0}</span>
        <span class="core-strength">${corePerformanceLabel("strength")} ${perfCounts.strength || 0}</span>
        <span class="core-flowability">${corePerformanceLabel("flowability")} ${perfCounts.flowability || 0}</span>
      </div>
      <div class="kg-causal-scroll">
        <svg class="publication-kg-svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img">
          ${layerOrder.map((layer, index) => `<text class="pub-kg-layer" x="${xByLayer[layer]}" y="30" text-anchor="middle">${escapeHtml(layerLabels[index])}</text>`).join("")}
          <g>${edgeHtml}</g>
          <g>${nodeHtml}</g>
        </svg>
      </div>
    </section>
  `;
}

function kgCausalSvg(trace = {}) {
  const nodes = trace.nodes || [];
  const links = trace.links || [];
  if (!nodes.length) return `<div class="kg-empty">${currentLanguage() === "en" ? "No KG concept was retrieved." : "未命中概念"}</div>`;
  const causalNodes = nodes.filter((node) => node.type !== "recommendation");
  const causalIds = new Set(causalNodes.map((node) => node.id));
  const causalLinks = links.filter((link) => causalIds.has(link.source) && causalIds.has(link.target));
  const layerX = { material: 95, feature: 285, mechanism: 500, consequence: 720, performance: 930 };
  const layerColor = {
    material: "#e8f2ff",
    feature: "#f0f7ee",
    mechanism: "#fff5df",
    consequence: "#f7efff",
    performance: "#eaf7f4",
    recommendation: "#fff1f2",
  };
  const rowYs = {};
  let rowCount = 0;
  causalNodes.forEach((node) => {
    const row = node.row ?? 6 + rowCount;
    if (!(row in rowYs)) {
      rowYs[row] = 76 + Object.keys(rowYs).length * 92;
    }
    rowCount += 1;
  });
  const pos = {};
  causalNodes.forEach((node) => {
    const x = layerX[node.type] || 90;
    const y = rowYs[node.row ?? 0] || 70;
    pos[node.id] = { x, y };
  });
  const height = Math.max(260, 120 + Object.keys(rowYs).length * 92);
  const width = 1040;
  const edgeHtml = causalLinks.map((link, index) => {
    const a = pos[link.source];
    const b = pos[link.target];
    if (!a || !b) return "";
    const mx = (a.x + b.x) / 2;
    return `<path class="kg-edge ${link.active ? "active" : "muted"} ${escapeHtml(link.evidence_role || "support")}" style="--i:${index}" d="M ${a.x + 72} ${a.y} C ${mx} ${a.y}, ${mx} ${b.y}, ${b.x - 72} ${b.y}" />`;
  }).join("");
  const nodeHtml = causalNodes.map((node, index) => {
    const p = pos[node.id];
    const label = escapeHtml(kgNodeLabel(node));
    const fill = layerColor[node.type] || "#fff";
    return `
      <g class="kg-svg-node ${escapeHtml(node.type)} ${node.active ? "active" : "muted"}" style="--i:${index}" transform="translate(${p.x},${p.y})">
        <rect x="-82" y="-24" width="164" height="48" rx="7" fill="${fill}" />
        <text text-anchor="middle" y="-3">${label.length > 18 ? `${label.slice(0, 18)}...` : label}</text>
        <text class="kg-node-type" text-anchor="middle" y="14">${escapeHtml(kgTypeLabel(node.type))}</text>
      </g>
    `;
  }).join("");
  return `
    <div class="kg-causal-scroll animated-hit-graph">
      <svg class="kg-causal-svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${currentLanguage() === "en" ? "Retrieved knowledge-graph causal chains" : "命中知识图谱因果链"}">
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="#8a9baa"></path>
          </marker>
        </defs>
        <g class="kg-edges">${edgeHtml}</g>
        <g class="kg-nodes">${nodeHtml}</g>
      </svg>
    </div>
  `;
}

function kgHtml(agent, recommendations = []) {
  if (currentLanguage() === "en") {
    const trace = agent?.trace || {};
    const evidence = agent?.details?.evidence || {};
    const citations = agent?.details?.citations || [];
    return `
      <section class="main-process-card kg-card-main" style="--step:2">
        <div class="process-title"><span>03</span><strong>Mechanism Retrieval Agent</strong></div>
        <p class="process-copy">Retrieved paths are organized as material -> physicochemical feature -> mechanism -> consequence -> performance response. The animation order represents the retrieval chain.</p>
        <div class="agent-triptych">
          <div><b>Parse</b><p>Convert the user target and optimized mix into graph queries, such as high flowability, low carbon intensity, or high silica-fume dosage.</p></div>
          <div><b>Operate</b><p>Traverse material, feature, mechanism, consequence, and performance nodes along causal chains.</p></div>
          <div><b>Output</b><p>${escapeHtml(localizeAgentSummary(agent || { id: "kg" }))}</p></div>
        </div>
        ${agentStepsHtml(agent)}
        ${publicationKgMapHtml(trace)}
        ${kgCausalSvg(trace)}
        ${kgFigureSuiteHtml(trace)}
        ${evidencePanelsHtml(evidence, citations)}
        ${englishRecommendationsHtml(recommendations || [])}
        ${agentDialogueHtml(agent)}
      </section>
    `;
  }
  const trace = agent?.trace || {};
  const evidence = agent?.details?.evidence || {};
  const citations = agent?.details?.citations || [];
  return `
    <section class="main-process-card kg-card-main" style="--step:2">
      <div class="process-title"><span>03</span><strong>机理检索智能体</strong></div>
      <p class="process-copy">命中路径按照“材料 → 物理化学特征 → 作用机理 → 后果 → 性能响应”展开，动画顺序表示检索链路。</p>
      <div class="agent-triptych">
        <div><b>解析</b><p>把用户目标与优化配合比转为图谱查询，例如高流动性、低碳、硅灰高掺量等。</p></div>
        <div><b>运作</b><p>沿因果链遍历材料节点、特征节点、机理节点和性能节点。</p></div>
        <div><b>回答</b><p>${escapeHtml(agent?.summary || "机理检索已完成")}</p></div>
      </div>
      ${agentStepsHtml(agent)}
      ${publicationKgMapHtml(trace)}
      ${kgCausalSvg(trace)}
      ${kgFigureSuiteHtml(trace)}
      ${evidencePanelsHtml(evidence, citations)}
      <div class="recommendations main-recommendations">
        ${(recommendations || []).map((item) => `<div class="rec-card"><strong>${escapeHtml(item.topic || item.material || "机理")}</strong><p>${escapeHtml(item.mechanism || item.evidence || "")}</p><p>${escapeHtml(item.recommendation || "")}</p></div>`).join("")}
      </div>
      ${agentDialogueHtml(agent)}
    </section>
  `;
}

function evidencePanelsHtml(evidence = {}, citations = []) {
  const selected = evidence.selected || [];
  const routes = evidence.multi_hop_routes || [];
  if (!selected.length && !routes.length) return "";
  const isEnglish = currentLanguage() === "en";
  const sourceGroups = [
    ["standard", isEnglish ? "Standards" : "规范"],
    ["literature", isEnglish ? "Literature" : "文献"],
    ["kg", isEnglish ? "Curated KG" : "精选图谱"],
    ["auto_kg", isEnglish ? "Auto-Extracted KG" : "自动图谱"],
  ].map(([key, label]) => ({
    key,
    label,
    items: selected.filter((item) => item.source_type === key),
  }));
  return `
    <section class="evidence-board">
      <header>
        <strong>${isEnglish ? "Unified Evidence Index" : "统一证据索引"}</strong>
        <span>${isEnglish ? "Hybrid retrieval + multi-hop mechanism routes + evidence scoring" : "混合检索 + 多跳机理路线 + 证据评分"}</span>
      </header>
      <div class="evidence-summary">
        <div><b>${evidence.counts?.literature || 0}</b><small>${isEnglish ? "Literature abstract blocks" : "文献摘要块"}</small></div>
        <div><b>${evidence.counts?.standard || 0}</b><small>${isEnglish ? "Full-standard blocks" : "规范全文块"}</small></div>
        <div><b>${evidence.counts?.kg || 0}</b><small>${isEnglish ? "Curated KG routes" : "精选 KG 机理链"}</small></div>
        <div><b>${evidence.counts?.auto_kg || 0}</b><small>${isEnglish ? "Auto-extracted KG routes" : "自动抽取 KG"}</small></div>
      </div>
      <div class="evidence-grid">
        ${sourceGroups.map((group) => `
          <section class="evidence-cluster">
            <h4>${group.label}</h4>
            ${group.items.slice(0, 4).map((item) => `
              <article>
                <div><span>${escapeHtml(item.source_type)}</span><em>${escapeHtml(item.grade?.tier || "-")} / ${escapeHtml(item.grade?.confidence || "-")}</em></div>
                <strong>${escapeHtml(isEnglish ? englishCitationTitle(item) : item.title)}</strong>
                ${item.metadata?.year || item.metadata?.journal ? `<small>${escapeHtml([item.metadata?.journal, item.metadata?.year].filter(Boolean).join(" · "))}</small>` : ""}
                <p>${escapeHtml(isEnglish ? englishSafe(String(item.text || "").slice(0, 150), "Evidence text is available in the indexed source.") : String(item.text || "").slice(0, 150))}</p>
                <small>score ${escapeHtml(item.score)}</small>
              </article>
            `).join("")}
          </section>
        `).join("")}
      </div>
      <div class="agent-evidence-matrix">
        <strong>${isEnglish ? "Shared Evidence Calls" : "共享证据调用"}</strong>
        <span>${localizeAgentShortLabel("requirement_analysis")}</span><span>${localizeAgentShortLabel("optimization_modeling")}</span><span>${localizeAgentShortLabel("mechanism_retrieval")}</span><span>${localizeAgentShortLabel("report_generation")}</span>
        <em>${isEnglish ? "All four agents read the same retrieved evidence set and apply task-specific prompts." : "四个智能体均读取同一批检索证据，并在各自提示词中执行不同任务。"}</em>
      </div>
      <div class="route-strip">
        ${routes.slice(0, 4).map((item) => `<div>${(item.route || []).map((part) => `<span>${escapeHtml(isEnglish ? englishSafe(part, "evidence route") : part)}</span>`).join("<i>→</i>")}</div>`).join("")}
      </div>
      <div class="citation-strip">
        ${citations.slice(0, 12).map((item) => `<button type="button" data-citation-id="${escapeHtml(item.id)}">[${escapeHtml(item.ref_no || item.id)}] ${escapeHtml(isEnglish ? englishCitationTitle(item) : item.title)}</button>`).join("")}
      </div>
    </section>
  `;
}

function citationTableHtml(citations = []) {
  if (!citations.length) return "";
  const isEnglish = currentLanguage() === "en";
  return `
    <table class="formal-table citation-table">
      <thead><tr><th>ID</th><th>${isEnglish ? "Source" : "来源"}</th><th>${isEnglish ? "Title" : "标题"}</th><th>${isEnglish ? "Metadata" : "元数据"}</th><th>${isEnglish ? "Grade" : "等级"}</th><th>${isEnglish ? "Excerpt" : "摘录"}</th></tr></thead>
      <tbody>
        ${citations.map((item) => `
          <tr>
            <td><button type="button" data-citation-id="${escapeHtml(item.id)}">[${escapeHtml(item.ref_no || item.id)}]</button></td>
            <td>${escapeHtml(item.source_type)}</td>
            <td>${escapeHtml(isEnglish ? englishCitationTitle(item) : item.title)}</td>
            <td>${escapeHtml([item.metadata?.journal, item.metadata?.year, item.metadata?.doi].filter(Boolean).join(" · "))}</td>
            <td>${escapeHtml(item.grade?.tier || "-")} / ${escapeHtml(item.grade?.confidence || "-")}</td>
            <td>${escapeHtml(isEnglish ? englishSafe(item.excerpt, "Source excerpt is available in the indexed evidence.") : item.excerpt || "")}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function localizeStepTitle(title = "") {
  if (currentLanguage() !== "en") return title;
  const map = {
    "鲍罗米先验收缩": "Bolomey-prior contraction",
    "装配优化变量": "Assemble optimization variables",
    "构造目标函数": "Construct objective functions",
    "构造约束条件": "Construct constraints",
    "运行多目标算法": "Run multi-objective algorithms",
    "生成 Pareto 解集": "Generate Pareto solution set",
    "筛选推荐方案": "Screen recommended scheme",
    "统一证据召回": "Unified evidence retrieval",
    "识别修正建议": "Identify revision recommendations",
    "读取原始请求": "Read raw request",
    "切分语义片段": "Segment semantic clauses",
    "识别数值实体": "Identify numerical entities",
    "识别材料意图": "Identify material intent",
    "排序设计优先级": "Rank design priorities",
    "生成下游规格": "Generate downstream specification",
    "固定预测输入": "Fix prediction inputs",
  };
  return map[title] || englishSafe(title, "Operation step");
}

function localizeStepDetail(detail = "") {
  if (currentLanguage() !== "en") return detail;
  let text = String(detail || "");
  if (text.includes("先反推水胶比可行区间")) return text.replace("先反推水胶比可行区间", "Back-calculate the feasible water-to-binder-ratio interval").replace("，再进入多目标搜索", ", then enter multi-objective search");
  if (text.includes("个设计变量")) return text.replace("装配", "assembled").replace("个设计变量", "design variables");
  if (text.includes("目标函数")) return text.replace("生成", "generated").replace("个目标函数", "objective functions");
  if (text.includes("个约束")) return text.replace("生成", "generated").replace("个约束", "constraints");
  if (text.includes("对比算法")) return text.replace("对比算法", "comparison algorithms");
  if (text.includes("选定算法")) return text.replace("选定算法", "selected algorithm").replace("；传统先验复筛后保留", "; retained").replace("个可行解", "prior-feasible solutions after traditional-prior rescreening");
  if (text.includes("预测强度")) return text.replace("预测强度", "predicted strength").replace("；碳排放", "; carbon footprint");
  if (text.includes("得到") && text.includes("个片段")) return text.replace("得到", "obtained").replace("个片段", "segments");
  if (text.includes("目标强度")) return text.replace("目标强度", "target strength").replace("；龄期", "; curing age");
  if (containsChinese(text)) return "Structured operation completed.";
  return text;
}

function agentStepsHtml(agent) {
  const steps = agent?.steps || [];
  if (!steps.length) return "";
  return `
    <div class="operation-timeline">
      ${steps.map((step, index) => `
        <div class="operation-step" style="--i:${index}">
          <span>${String(index + 1).padStart(2, "0")}</span>
          <div>
            <strong>${escapeHtml(localizeStepTitle(step.title))}</strong>
            <p>${escapeHtml(localizeStepDetail(step.detail))}</p>
            <small>${escapeHtml(step.artifact || "")}</small>
          </div>
        </div>
      `).join("")}
    </div>
  `;
}

function agentDialogueHtml(agent) {
  if (!agent?.dialogue) return "";
  const isEnglish = currentLanguage() === "en";
  if (isEnglish && containsChinese(agent.dialogue)) return "";
  const name = isEnglish ? (agent.id ? localizeAgentShortLabel(agent.id) : englishSafe(agent.name, "Agent")) : agent.name;
  return `
    <div class="agent-dialogue">
      <h4>${escapeHtml(name)} ${isEnglish ? "Output" : "输出"}</h4>
      <div>${formatAssistantAnswer(agent.dialogue)}</div>
    </div>
  `;
}

function finalTableHtml(optimization = {}) {
  const isEnglish = currentLanguage() === "en";
  return `
    <section class="main-process-card final-card" style="--step:3">
      <div class="process-title"><span>04</span><strong>${isEnglish ? "Recommended Design Scheme" : "推荐设计方案"}</strong></div>
      <div class="final-dashboard">
        <div class="final-scoreboard">${kpiCardsHtml(optimization.kpis || {})}</div>
        <div class="final-mix-panel">
          <h4>${isEnglish ? "Mix-Design Table" : "配合比清单"}</h4>
          ${solutionTableHtml(optimization.solution_table || [])}
        </div>
      </div>
      <div class="ratio-strip">${ratioCardsHtml(optimization.ratios || {})}</div>
      ${theoryStatusHtml(optimization.traditional_theory || {})}
      ${theoryCardsHtml(optimization.traditional_theory || {})}
      <div class="candidate-schemes">
        <h4>${isEnglish ? "Candidate Scheme Families" : "候选配方族"}</h4>
        ${candidateSchemesHtml(optimization.candidate_schemes || [])}
      </div>
      <div class="comparison-block">
        <h4>${isEnglish ? "Scheme Family Comparison" : "方案族比较"}</h4>
        ${schemeComparisonHtml(optimization.scheme_comparison || [])}
      </div>
      ${methodBaselineHtml(optimization.method_baselines || [])}
      ${evidenceChainHtml(optimization.evidence_chain || [])}
      ${trustProfileHtml(optimization.trust_profile || {})}
    </section>
  `;
}

function designBasisTableHtml(data = {}) {
  const requirement = (data.agents || []).find((agent) => agent.id === "requirement")?.details || {};
  const optimization = data.optimization || {};
  const rows = [
    ["目标强度", `${requirement.target_mpa ?? optimization.target_mpa ?? "-"} MPa`],
    ["龄期", `${requirement.age_days ?? optimization.age_days ?? "-"} d`],
    ["性能优先级", (requirement.priorities || []).join(" / ") || "-"],
    ["提及材料", (requirement.mentioned_materials || []).join(" / ") || "无"],
    ["必用材料", (requirement.required_materials || []).join(" / ") || "无"],
    ["允许材料", (requirement.allowed_materials || []).join(" / ") || "无"],
    ["禁用材料", (requirement.avoided_materials || []).join(" / ") || "无"],
    ["性能区间约束", (requirement.performance_constraints || []).map((item) => `${item.metric}: ${item.lower ?? "-"}~${item.upper ?? "-"}`).join(" / ") || "无"],
    ["语义解释器", requirement.semantic_interpreter === "llm_primary_with_program_validation" ? "LLM 主解析 + 程序校验" : "规则回退解析"],
    ["配方策略", requirement.material_mode === "specified" ? "指定材料模式" : "探索配方模式"],
    ["主算法", optimization.best_algorithm || "-"],
  ];
  return `
    <table class="formal-table">
      <tbody>${rows.map(([label, value]) => `<tr><th>${escapeHtml(label)}</th><td>${escapeHtml(value)}</td></tr>`).join("")}</tbody>
    </table>
  `;
}

function semanticAuditHtml(requirement = {}) {
  const parserHints = requirement.parser_hints || {};
  const llmOk = requirement.llm_spec_ok;
  const asTextList = (value) => Array.isArray(value) ? value : (value ? [value] : []);
  const rationale = [...asTextList(requirement.rationale), ...asTextList(requirement.validation_notes)];
  if (currentLanguage() === "en") {
    const list = (items = []) => items.map((item) => localizeVariableName({ material: item })).join(" / ") || "none";
    return `
      <section class="semantic-audit">
        <header>
          <strong>Semantic Interpretation Audit</strong>
          <span class="${llmOk ? "ok" : "warn"}">${llmOk ? "LLM-led parsing" : "Rule fallback"}</span>
        </header>
        <div class="semantic-audit-grid">
          <div><b>Final material intent</b><p>Required: ${escapeHtml(list(requirement.required_materials || []))}; Allowed: ${escapeHtml(list(requirement.allowed_materials || []))}; Avoided: ${escapeHtml(list(requirement.avoided_materials || []))}</p></div>
          <div><b>Programmatic parser hints</b><p>Required: ${escapeHtml(list(parserHints.required_materials || []))}; Allowed: ${escapeHtml(list(parserHints.allowed_materials || []))}; Avoided: ${escapeHtml(list(parserHints.avoided_materials || []))}</p></div>
          <div><b>Parsing notes</b><p>${escapeHtml(rationale.map((item) => englishSafe(item, "Parsing note")).join("; ") || "No additional notes returned.")}</p></div>
        </div>
      </section>
    `;
  }
  return `
    <section class="semantic-audit">
      <header>
        <strong>语义解释审计</strong>
        <span class="${llmOk ? "ok" : "warn"}">${llmOk ? "LLM 已主导解析" : "规则回退"}</span>
      </header>
      <div class="semantic-audit-grid">
        <div><b>最终材料意图</b><p>必用 ${(requirement.required_materials || []).join(" / ") || "无"}；允许 ${(requirement.allowed_materials || []).join(" / ") || "无"}；禁用 ${(requirement.avoided_materials || []).join(" / ") || "无"}</p></div>
        <div><b>程序候选提示</b><p>必用 ${(parserHints.required_materials || []).join(" / ") || "无"}；允许 ${(parserHints.allowed_materials || []).join(" / ") || "无"}；禁用 ${(parserHints.avoided_materials || []).join(" / ") || "无"}</p></div>
        <div><b>解析说明</b><p>${escapeHtml(rationale.join("；") || "未返回额外说明。")}</p></div>
      </div>
    </section>
  `;
}

function constraintStatusHtml(status = {}) {
  const applied = status.applied || [];
  const unresolved = status.unresolved || [];
  if (!applied.length && !unresolved.length) return "";
  if (currentLanguage() === "en") {
    const fmt = (item) => `${englishSafe(item.metric, "constraint")} ${item.lower ?? "-"}~${item.upper ?? "-"} -> ${englishSafe(item.model_field, "model field")}`;
    const fmtUnresolved = (item) => `${englishSafe(item.metric, "constraint")} ${item.lower ?? "-"}~${item.upper ?? "-"}: ${englishSafe(item.reason, "unresolved reason")}`;
    return `
      <section class="constraint-status ${unresolved.length ? "has-warning" : ""}">
        <header><strong>User Constraint Implementation Status</strong></header>
        <div class="constraint-status-grid">
          <div>
            <b>Applied</b>
            <p>${applied.length ? applied.map(fmt).join("; ") : "none"}</p>
          </div>
          <div>
            <b>Unresolved</b>
            <p>${unresolved.length ? unresolved.map(fmtUnresolved).join("; ") : "none"}</p>
          </div>
        </div>
      </section>
    `;
  }
  return `
    <section class="constraint-status ${unresolved.length ? "has-warning" : ""}">
      <header><strong>用户约束落地状态</strong></header>
      <div class="constraint-status-grid">
        <div>
          <b>已生效</b>
          <p>${applied.length ? applied.map((item) => `${item.metric} ${item.lower ?? "-"}~${item.upper ?? "-"} -> ${item.model_field}`).join("；") : "无"}</p>
        </div>
        <div>
          <b>未生效</b>
          <p>${unresolved.length ? unresolved.map((item) => `${item.metric} ${item.lower ?? "-"}~${item.upper ?? "-"}：${item.reason}`).join("；") : "无"}</p>
        </div>
      </div>
    </section>
  `;
}

function standardVizLabel(value = "") {
  const isEnglish = currentLanguage() === "en";
  const key = String(value || "unknown");
  const en = {
    slump_flow: "slump-flow",
    slump: "slump",
    flowability: "flowability",
    t500: "T500",
    strength_grade: "strength grade",
    compressive_strength: "compressive strength",
    self_compacting_concrete: "self-compacting concrete",
    pumped_concrete: "pumped concrete",
    general_concrete: "general concrete",
    marine_concrete: "marine concrete",
    high_strength_concrete: "high-strength concrete",
    mix_design_concrete: "mix-design concrete",
    optimizable: "optimizable",
    reference_only: "reference only",
    unknown: "unknown",
  };
  const zh = {
    slump_flow: "坍落扩展度",
    slump: "坍落度",
    flowability: "流动性",
    t500: "T500",
    strength_grade: "强度等级",
    compressive_strength: "抗压强度",
    self_compacting_concrete: "自密实混凝土",
    pumped_concrete: "泵送混凝土",
    general_concrete: "通用混凝土",
    marine_concrete: "海工混凝土",
    high_strength_concrete: "高强混凝土",
    mix_design_concrete: "配合比设计",
    optimizable: "可优化",
    reference_only: "参考",
    unknown: "未知",
  };
  return (isEnglish ? en : zh)[key] || (isEnglish ? englishSafe(key, "standard item") : key);
}

function standardBarSvg(items = [], width = 360, height = 160, color = "#2f7f8f") {
  const clean = items.filter(([, value]) => Number(value) > 0).slice(0, 6);
  if (!clean.length) return `<text x="18" y="82" class="empty-svg-text">No data</text>`;
  const max = Math.max(...clean.map(([, value]) => Number(value) || 0), 1);
  const left = 118;
  const barW = width - left - 34;
  const rowH = 19;
  return clean.map(([label, value], index) => {
    const y = 34 + index * rowH;
    const w = Math.max(3, (Number(value) / max) * barW);
    return `
      <text x="12" y="${y + 10}" class="std-axis-label">${escapeHtml(standardVizLabel(label))}</text>
      <rect x="${left}" y="${y}" width="${w}" height="11" rx="2" fill="${color}" opacity="${0.92 - index * 0.06}"></rect>
      <text x="${left + w + 6}" y="${y + 10}" class="std-value-label">${escapeHtml(value)}</text>
    `;
  }).join("");
}

function standardMatrixSvg(matrix = [], width = 620, height = 230) {
  const domains = [...new Set(matrix.map((item) => item.domain))].slice(0, 6);
  const metrics = [...new Set(matrix.map((item) => item.metric))].slice(0, 6);
  if (!domains.length || !metrics.length) return `<text x="18" y="112" class="empty-svg-text">No matrix data</text>`;
  const lookup = new Map(matrix.map((item) => [`${item.domain}::${item.metric}`, Number(item.count) || 0]));
  const max = Math.max(...matrix.map((item) => Number(item.count) || 0), 1);
  const left = 138;
  const top = 48;
  const cellW = Math.min(72, (width - left - 28) / Math.max(metrics.length, 1));
  const cellH = 24;
  const metricLabels = metrics.map((metric, index) => `<text x="${left + index * cellW + cellW / 2}" y="26" class="std-matrix-head" text-anchor="middle">${escapeHtml(standardVizLabel(metric))}</text>`).join("");
  const rows = domains.map((domain, rowIndex) => {
    const y = top + rowIndex * cellH;
    const cells = metrics.map((metric, colIndex) => {
      const value = lookup.get(`${domain}::${metric}`) || 0;
      const alpha = value ? 0.18 + 0.72 * (value / max) : 0.04;
      return `
        <rect x="${left + colIndex * cellW}" y="${y}" width="${cellW - 4}" height="${cellH - 4}" rx="3" fill="#0f766e" opacity="${alpha}"></rect>
        ${value ? `<text x="${left + colIndex * cellW + (cellW - 4) / 2}" y="${y + 14}" text-anchor="middle" class="std-matrix-value">${value}</text>` : ""}
      `;
    }).join("");
    return `<text x="12" y="${y + 14}" class="std-axis-label">${escapeHtml(standardVizLabel(domain))}</text>${cells}`;
  }).join("");
  return `${metricLabels}${rows}`;
}

function standardConstraintVisualizationHtml(viz = {}, status = {}) {
  const count = Number(viz.constraint_count || 0);
  const optimizable = Number(viz.optimizable_count || 0);
  const agent = viz.constraint_agent || {};
  const evidenceCounts = viz.evidence_counts || {};
  const abstractCorpus = viz.abstract_corpus || {};
  const rows = Array.isArray(viz.optimizable_rows) ? viz.optimizable_rows : [];
  const applied = Array.isArray(status.applied) ? status.applied.filter((item) => item.source_type === "standard") : [];
  const unresolved = Array.isArray(status.unresolved) ? status.unresolved.filter((item) => item.source_type === "standard") : [];
  if (!count && !rows.length && !applied.length && !unresolved.length) return "";
  const isEnglish = currentLanguage() === "en";
  const metricItems = Object.entries(viz.metrics || {}).sort((a, b) => Number(b[1]) - Number(a[1]));
  const domainItems = Object.entries(viz.domains || {}).sort((a, b) => Number(b[1]) - Number(a[1]));
  const statusItems = Object.entries(viz.compile_status || {}).sort((a, b) => Number(b[1]) - Number(a[1]));
  const activeRows = rows.slice(0, 5);
  if (Number(agent.text_evidence_count || 0) > 0) {
    const groupItems = (agent.country_groups || []).map((item) => [item.country_group, Number(item.standards) || 0]);
    const textItems = (agent.targets || []).map((item) => [item.target, Number(item.text_evidence) || 0]);
    const numericItems = (agent.targets || []).map((item) => [item.target, Number(item.numeric_records) || 0]);
    const textMatrix = (agent.group_target || []).map((item) => ({
      domain: item.country_group,
      metric: item.target,
      count: Number(item.text_evidence) || 0,
    }));
    const numericMatrix = (agent.group_target || []).map((item) => ({
      domain: item.country_group,
      metric: item.target,
      count: Number(item.numeric_records) || 0,
    }));
    return `
      <section class="standard-constraint-viz">
        <header>
          <div>
            <strong>${isEnglish ? "Constraint-Condition Agent Evidence Base" : "约束条件智能体证据库"}</strong>
            <span>${isEnglish ? "Standard text evidence → numeric records → context-aware constraint compilation" : "规范文本证据 → 数值记录 → 结合适用域与用户需求编译约束"}</span>
          </div>
          <a href="/api/standards/export/four-target-csv" target="_blank" rel="noreferrer">${isEnglish ? "Evidence CSV" : "证据 CSV"}</a>
        </header>
        <div class="standard-viz-kpis">
          <div><b>${escapeHtml(agent.standard_count ?? "-")}</b><small>${isEnglish ? "curated standards" : "规范总数"}</small></div>
          <div><b>${escapeHtml(agent.text_evidence_count ?? "-")}</b><small>${isEnglish ? "text evidence clauses" : "文本证据条文"}</small></div>
          <div><b>${escapeHtml(agent.numeric_record_count ?? "-")}</b><small>${isEnglish ? "numeric records" : "数值记录"}</small></div>
          <div><b>${escapeHtml((agent.targets || []).length)}</b><small>${isEnglish ? "design targets" : "设计目标"}</small></div>
          <div><b>${escapeHtml(count)}</b><small>${isEnglish ? "compiled legacy records" : "已有编译记录"}</small></div>
          <div><b>${escapeHtml(applied.length)}</b><small>${isEnglish ? "applied in this run" : "本次已生效"}</small></div>
        </div>
        <div class="standard-viz-grid">
          <figure>
            <figcaption>${isEnglish ? "Standard-source groups" : "规范来源分组"}</figcaption>
            <svg viewBox="0 0 360 160" role="img">${standardBarSvg(groupItems, 360, 160, "#2f6f5e")}</svg>
          </figure>
          <figure>
            <figcaption>${isEnglish ? "Text evidence by target" : "各目标文本证据"}</figcaption>
            <svg viewBox="0 0 360 160" role="img">${standardBarSvg(textItems, 360, 160, "#3e6ea8")}</svg>
          </figure>
          <figure>
            <figcaption>${isEnglish ? "Numeric records by target" : "各目标数值记录"}</figcaption>
            <svg viewBox="0 0 360 160" role="img">${standardBarSvg(numericItems, 360, 160, "#b9782d")}</svg>
          </figure>
        </div>
        <div class="standard-viz-grid">
          <figure class="standard-matrix-figure">
            <figcaption>${isEnglish ? "Country group × target text evidence" : "来源分组 × 目标文本证据"}</figcaption>
            <svg viewBox="0 0 620 190" role="img">${standardMatrixSvg(textMatrix, 620, 190)}</svg>
          </figure>
          <figure class="standard-matrix-figure">
            <figcaption>${isEnglish ? "Country group × target numeric records" : "来源分组 × 目标数值记录"}</figcaption>
            <svg viewBox="0 0 620 190" role="img">${standardMatrixSvg(numericMatrix, 620, 190)}</svg>
          </figure>
        </div>
        <div class="standard-active-constraints">
          <strong>${isEnglish ? "Agent target memory" : "智能体目标证据记忆"}</strong>
          ${(agent.targets || []).map((item) => `
            <article>
              <b>${escapeHtml(standardVizLabel(item.target))}</b>
              <span>${escapeHtml(item.text_evidence || 0)} ${isEnglish ? "text clauses" : "条文本证据"} · ${escapeHtml(item.numeric_records || 0)} ${isEnglish ? "numeric records" : "条数值记录"}</span>
              <small>${escapeHtml(item.source_standards || 0)} ${isEnglish ? "source standards" : "份来源规范"}</small>
            </article>
          `).join("")}
        </div>
        <p class="standard-viz-warning">${isEnglish ? "Numeric records are candidate constraints. The agent must verify application domain, age, test method and user intent before compiling them into optimization bounds." : "数值记录属于候选约束；智能体必须结合适用域、龄期、试验方法和用户需求后，才能编译为优化边界。"}</p>
      </section>
    `;
  }
  return `
    <section class="standard-constraint-viz">
      <header>
        <div>
          <strong>${isEnglish ? "Standards-Derived Constraint Map" : "规范提取约束可视化"}</strong>
          <span>${isEnglish ? "From standard clauses to computable optimization constraints" : "从规范条文到可计算优化约束"}</span>
        </div>
        <a href="/api/standards/export/csv" target="_blank" rel="noreferrer">${isEnglish ? "CSV" : "导出 CSV"}</a>
      </header>
      <div class="standard-viz-kpis">
        <div><b>${escapeHtml(abstractCorpus.doc_count ?? evidenceCounts.literature ?? "-")}</b><small>${isEnglish ? "abstract corpus papers" : "摘要语料文献"}</small></div>
        <div><b>${escapeHtml(evidenceCounts.literature ?? "-")}</b><small>${isEnglish ? "indexed literature records" : "索引文献记录"}</small></div>
        <div><b>${escapeHtml(evidenceCounts.standard ?? "-")}</b><small>${isEnglish ? "indexed standard blocks" : "索引规范块"}</small></div>
        <div><b>${escapeHtml(abstractCorpus.shrinkage_doc_count ?? "-")}</b><small>${isEnglish ? "shrinkage abstracts" : "收缩相关摘要"}</small></div>
        <div><b>${escapeHtml(viz.processed_pdf_count ?? "-")}</b><small>${isEnglish ? "processed standards" : "已处理规范"}</small></div>
        <div><b>${escapeHtml(count)}</b><small>${isEnglish ? "extracted constraints" : "抽取约束"}</small></div>
        <div><b>${escapeHtml(optimizable)}</b><small>${isEnglish ? "optimizable" : "可优化约束"}</small></div>
        <div><b>${escapeHtml(applied.length)}</b><small>${isEnglish ? "applied in this run" : "本次已生效"}</small></div>
      </div>
      <div class="standard-viz-grid">
        <figure>
          <figcaption>${isEnglish ? "Metric distribution" : "指标分布"}</figcaption>
          <svg viewBox="0 0 360 160" role="img">${standardBarSvg(metricItems, 360, 160, "#1f77b4")}</svg>
        </figure>
        <figure>
          <figcaption>${isEnglish ? "Concrete-domain distribution" : "适用领域分布"}</figcaption>
          <svg viewBox="0 0 360 160" role="img">${standardBarSvg(domainItems, 360, 160, "#b45309")}</svg>
        </figure>
        <figure>
          <figcaption>${isEnglish ? "Compile status" : "编译状态"}</figcaption>
          <svg viewBox="0 0 360 160" role="img">${standardBarSvg(statusItems, 360, 160, "#6d5bd0")}</svg>
        </figure>
      </div>
      <figure class="standard-matrix-figure">
        <figcaption>${isEnglish ? "Domain-metric constraint matrix" : "领域-指标约束矩阵"}</figcaption>
        <svg viewBox="0 0 620 230" role="img">${standardMatrixSvg(viz.matrix || [], 620, 230)}</svg>
      </figure>
      <div class="standard-active-constraints">
        <strong>${isEnglish ? "Representative computable clauses" : "代表性可计算条文"}</strong>
        ${activeRows.map((row) => `
          <article>
            <b>${escapeHtml(standardVizLabel(row.metric))}</b>
            <span>${escapeHtml(row.operator || "")} ${escapeHtml(row.value ?? row.lower ?? row.upper ?? "-")} ${escapeHtml(row.unit || "")}</span>
            <small>${escapeHtml(isEnglish ? englishSafe(row.source_file || "standard", "standard source") : row.source_file || "规范")} · p.${escapeHtml(row.page || "-")} · ${escapeHtml(standardVizLabel(row.domain))}</small>
          </article>
        `).join("") || `<p>${isEnglish ? "No optimizable standard clause is available for the current domain." : "当前领域没有可直接优化的规范条文。"}</p>`}
      </div>
      ${unresolved.length ? `<p class="standard-viz-warning">${isEnglish ? "Some standards-derived constraints were retrieved but not applied because the current model package lacks the corresponding predictor." : "部分规范约束已命中但未生效，因为当前模型包缺少对应预测输出。"}</p>` : ""}
    </section>
  `;
}

function reportFigureGalleryHtml(data = {}) {
  const plots = data.visualizations?.optimization_plots || [];
  if (!plots.length) return '<p class="empty-copy">当前没有可嵌入的优化图像。</p>';
  return `
    <div class="report-figure-gallery">
      ${plots.slice(0, 8).map((plot, index) => `
        <figure>
          <img src="${plot.image}" alt="${escapeHtml(plot.name)}" />
          <figcaption>图 ${index + 1}. ${escapeHtml(plot.name)}</figcaption>
        </figure>
      `).join("")}
    </div>
  `;
}

function implementationPlanHtml(optimization = {}, requests = {}) {
  const ratios = optimization.ratios || {};
  return `
    <table class="formal-table">
      <thead><tr><th>阶段</th><th>建议动作</th><th>判定依据</th></tr></thead>
      <tbody>
        <tr><td>原材复核</td><td>核查胶凝材料活性、骨料含水率、减水剂相容性</td><td>避免模型输入与现场原材脱节</td></tr>
        <tr><td>试拌确认</td><td>按推荐配比进行首轮拌合，实测坍落扩展度、黏聚性和含气量</td><td>水胶比 ${escapeHtml(ratios.water_binder_ratio ?? "-")}；浆骨比 ${escapeHtml(ratios.paste_aggregate_ratio ?? "-")}</td></tr>
        <tr><td>强度验证</td><td>至少测试 3 d、7 d、28 d 强度，并复核目标龄期</td><td>强度不得低于设计目标</td></tr>
        <tr><td>二次优化</td><td>${escapeHtml((requests.requested_experiments || []).join("；") || "若工作性或强度偏离目标，则依据实测结果重新优化")}</td><td>触发条件：预测与实测差异超限</td></tr>
      </tbody>
    </table>
  `;
}

function theoryVerificationHtml(theory = {}) {
  const wbWindow = theory.water_binder_window || {};
  return `
    ${theoryStatusHtml(theory)}
    ${theoryCardsHtml(theory)}
    ${theoryVolumeFigureHtml(theory)}
    <table class="formal-table">
      <thead><tr><th>理论</th><th>公式</th><th>结果</th></tr></thead>
      <tbody>
        <tr><td>绝对体积法</td><td>Σ(m_i / ρ_i) + V_air = 1</td><td>${escapeHtml(theory.absolute_volume_m3 ?? "-")} m3</td></tr>
        <tr><td>理论容重</td><td>ρ_theory = Σm_i</td><td>${escapeHtml(theory.theoretical_density_kg_m3 ?? "-")} kg/m3</td></tr>
        <tr><td>鲍罗米公式</td><td>f = A(1/(w/b) - B_0)</td><td>${escapeHtml(theory.bolomey_strength_mpa ?? "-")} MPa；先验 w/b 区间 ${escapeHtml(wbWindow.min ?? "-")} ~ ${escapeHtml(wbWindow.max ?? "-")}</td></tr>
      </tbody>
    </table>
    <div class="theory-assumption-note">
      <strong>${theory.calibration_status === "project_calibrated" ? "已标定参数" : "默认参考参数"}</strong>
      <p>${escapeHtml(theory.assumption_note || "")}</p>
    </div>
  `;
}

function validationMatrixHtml() {
  return `
    <table class="formal-table">
      <thead><tr><th>项目</th><th>建议指标</th><th>用途</th></tr></thead>
      <tbody>
        <tr><td>新拌性能</td><td>坍落扩展度、T500、离析/泌水、含气量</td><td>验证施工性与泵送适应性</td></tr>
        <tr><td>力学性能</td><td>3 d、7 d、28 d 抗压强度</td><td>验证早期与目标龄期强度</td></tr>
        <tr><td>低收缩性能</td><td>28 d/56 d 干燥收缩、自收缩、环形约束开裂、裂缝宽度观察；必要时劈裂抗拉或弯曲韧性</td><td>验证低收缩目标</td></tr>
        <tr><td>低碳-收缩协同</td><td>单位体积 CO2e、干燥/自收缩、环形约束开裂、裂缝宽度</td><td>验证论文核心设计目标</td></tr>
        <tr><td>低碳核算</td><td>单位体积 kgCO2e/m3</td><td>校核减碳收益</td></tr>
      </tbody>
    </table>
  `;
}

function structuredNarrativeHtml(data = {}) {
  const answer = data.answer || "";
  const citations = collectCitations(data);
  return `
    <section class="formal-section narrative-section">
      <h4>9. 技术论证正文</h4>
      <div class="formal-prose">${linkCitationRefs(formatAssistantAnswer(answer), citations)}</div>
    </section>
  `;
}

function containsChinese(value) {
  return /[\u4e00-\u9fff]/.test(String(value || ""));
}

function englishSafe(value, fallback = "-") {
  if (value === null || value === undefined || value === "") return fallback;
  return containsChinese(value) ? fallback : String(value);
}

function englishList(values, fallback = "None") {
  const mapItem = (item) => {
    if (EN_PRIORITY_LABELS[item]) return EN_PRIORITY_LABELS[item];
    const material = localizeVariableName({ material: item });
    return material && material !== "Variable" ? material : englishSafe(item, "");
  };
  const items = (values || []).map((item) => mapItem(item)).filter(Boolean);
  return items.length ? items.join(" / ") : fallback;
}

function englishSolutionTableHtml(rows = []) {
  if (!rows.length) return '<p class="empty-copy">No mix-design variables were returned.</p>';
  return `
    <table class="formal-table">
      <thead><tr><th>Variable</th><th>Dosage</th><th>Unit</th><th>Role</th></tr></thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            <td>${escapeHtml(englishSafe(row.name, "Variable"))}</td>
            <td>${escapeHtml(row.value ?? "-")}</td>
            <td>${escapeHtml(row.unit || "")}</td>
            <td>${escapeHtml(englishSafe(row.role, "Mix-design variable"))}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function englishValidationMatrixHtml() {
  return `
    <table class="formal-table">
      <thead><tr><th>Category</th><th>Recommended measurements</th><th>Purpose</th></tr></thead>
      <tbody>
        <tr><td>Fresh properties</td><td>Slump flow, T500, segregation/bleeding, air content</td><td>Verify workability and pumping suitability</td></tr>
        <tr><td>Mechanical properties</td><td>3 d, 7 d, and 28 d compressive strength</td><td>Verify early-age and target-age strength</td></tr>
        <tr><td>Low-shrinkage and crack-resistance performance</td><td>28 d/56 d drying shrinkage, autogenous shrinkage, restrained-ring cracking, crack-width observation, and splitting tensile or flexural toughness when required</td><td>Verify low-shrinkage and crack-resistance targets</td></tr>
        <tr><td>Low-carbon, shrinkage and cracking synergy</td><td>CO2e per cubic metre, drying/autogenous shrinkage, restrained-ring cracking, crack width</td><td>Verify the manuscript-facing design target</td></tr>
        <tr><td>Carbon accounting</td><td>kgCO2e per cubic metre</td><td>Audit the carbon-reduction benefit</td></tr>
      </tbody>
    </table>
  `;
}

function englishSchemeComparisonHtml(rows = []) {
  if (!rows.length) return '<p class="empty-copy">No alternative scheme family was returned.</p>';
  return `
    <table class="formal-table">
      <thead><tr><th>Scheme</th><th>Role</th><th>Strength</th><th>Carbon</th><th>Shrinkage risk</th><th>w/b</th><th>p/a</th></tr></thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            <td>${escapeHtml(localizeSchemeLabel(row.label))}</td>
            <td>${escapeHtml(localizeSchemeRoles(row.roles || []))}</td>
            <td>${escapeHtml(row.strength_mpa ?? "-")}</td>
            <td>${escapeHtml(row.carbon_kgco2e_m3 ?? "-")}</td>
            <td>${escapeHtml(row.shrinkage_risk_index ?? "-")}</td>
            <td>${escapeHtml(row.water_binder_ratio ?? "-")}</td>
            <td>${escapeHtml(row.paste_aggregate_ratio ?? "-")}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function englishRecommendationsHtml(items = []) {
  const rows = (items || []).map((item) => recommendationDisplay(item));
  if (!rows.length) return '<p class="empty-copy">No additional mechanistic recommendation was returned.</p>';
  return `<div class="recommendations main-recommendations">${rows.map((item) => `<div class="rec-card"><strong>${escapeHtml(item.topic)}</strong><p>${escapeHtml(item.text)}</p>${item.action ? `<p>${escapeHtml(item.action)}</p>` : ""}</div>`).join("")}</div>`;
}

function englishNarrativeHtml(data = {}) {
  if (data.answer && !containsChinese(data.answer)) {
    const citations = collectCitations(data);
    return linkCitationRefs(formatAssistantAnswer(data.answer), citations);
  }
  const optimization = data.optimization || {};
  const kpis = optimization.kpis || {};
  return `
    <p>The system translated the user request into structured constraints, executed constrained multi-objective mix-design optimization, and retained knowledge-graph evidence for mechanistic review. The recommended solution is interpreted as a balanced candidate rather than a single-objective extreme.</p>
    <p>The current candidate reaches a predicted compressive strength of ${escapeHtml(kpis.strength_mpa ?? "-")} MPa with an estimated carbon footprint of ${escapeHtml(kpis.carbon_kgco2e_m3 ?? "-")} kgCO2e/m3 and a mechanistic shrinkage/cracking-risk index of ${escapeHtml(kpis.shrinkage_risk_index ?? "-")}. The risk index is a design proxy for ranking and explanation, not a measured microstrain, crack-width, fracture-energy, or restrained-ring response.</p>
    <p>For a manuscript-facing workflow, the recommended next step is to report the model package, variable bounds, hard constraints, candidate-scheme comparison, and validation matrix together with the exported figures and source tables.</p>
  `;
}

function englishCitationTitle(item = {}) {
  const title = item.title || item.source || item.reference || "";
  if (title && !containsChinese(title)) return title;
  if (item.source_type === "standard") return "Chinese concrete standard clause";
  if (item.source_type === "literature") return "Concrete literature abstract evidence";
  if (item.source_type === "kg") return "Curated concrete knowledge-graph route";
  if (item.source_type === "auto_kg") return "Automatically extracted knowledge-graph candidate";
  return "Evidence record";
}

function englishEvidenceReferencesHtml(data = {}) {
  const citations = collectCitations(data);
  const counts = citations.reduce((acc, item) => {
    const key = item.source_type || "unknown";
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  if (!citations.length) {
    return '<p class="empty-copy">No evidence citations were attached to this run.</p>';
  }
  return `
    <div class="mini-metrics">
      <span>Standards ${escapeHtml(counts.standard || 0)}</span>
      <span>Literature ${escapeHtml(counts.literature || 0)}</span>
      <span>KG ${escapeHtml((counts.kg || 0) + (counts.auto_kg || 0))}</span>
    </div>
    <table class="formal-table citation-table">
      <thead><tr><th>Ref.</th><th>Type</th><th>Evidence source</th><th>Score</th></tr></thead>
      <tbody>
        ${citations.map((item) => `
          <tr>
            <td>[${escapeHtml(item.ref_no || item.id)}]</td>
            <td>${escapeHtml(item.source_type || "-")}</td>
            <td>${escapeHtml(englishCitationTitle(item))}</td>
            <td>${escapeHtml(item.score ?? "-")}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function publicationReportHtmlEn(data = {}) {
  const optimization = data.optimization || {};
  const requirement = (data.agents || []).find((agent) => agent.id === "requirement")?.details || {};
  const ratios = optimization.ratios || {};
  return `
    <details class="publication-report-shell">
      <summary>
        <div>
          <span>Final Output</span>
          <strong>High-Strength Low-Carbon Low-Shrinkage Crack-Resistant Concrete Engineering and Research Report</strong>
          <small>Open the complete English report; process traces remain above.</small>
        </div>
        <div class="report-summary-actions">
          <b>${escapeHtml(optimization.kpis?.strength_mpa ?? "-")} MPa</b>
          <b>${escapeHtml(optimization.kpis?.carbon_kgco2e_m3 ?? "-")} kgCO2e/m3</b>
        </div>
      </summary>
      <section class="publication-report">
        <header>
          <div class="report-head-row">
            <span>Design Report</span>
            <div class="report-actions">
              <button type="button" data-reference-panel>Refs</button>
              <button type="button" data-report-download="md">MD</button>
              <button type="button" data-report-download="json">JSON</button>
              <button type="button" data-report-download="word">Word</button>
              <button type="button" data-report-download="pdf">PDF</button>
            </div>
          </div>
          <h3>High-Strength Low-Carbon Low-Shrinkage Crack-Resistant Concrete Engineering and Research Report</h3>
          <p>A structured report for trial mixing, scientific validation, and next-round optimization.</p>
        </header>
        <div class="publication-summary">${kpiCardsHtml(optimization.kpis || {})}</div>
        <div class="formal-report-grid">
          <section class="formal-section">
            <h4>1. Design Basis and Input Boundary</h4>
            <table class="formal-table"><tbody>
              <tr><th>Target strength</th><td>${escapeHtml(requirement.target_mpa ?? optimization.target_mpa ?? "-")} MPa</td></tr>
              <tr><th>Curing age</th><td>${escapeHtml(requirement.age_days ?? optimization.age_days ?? "-")} d</td></tr>
              <tr><th>Priorities</th><td>${escapeHtml(englishList(requirement.priorities))}</td></tr>
              <tr><th>Required materials</th><td>${escapeHtml(englishList(requirement.required_materials))}</td></tr>
              <tr><th>Allowed materials</th><td>${escapeHtml(englishList(requirement.allowed_materials))}</td></tr>
              <tr><th>Avoided materials</th><td>${escapeHtml(englishList(requirement.avoided_materials))}</td></tr>
            </tbody></table>
          </section>
          <section class="formal-section">
            <h4>2. Recommended Mix and Key Ratios</h4>
            <div class="ratio-strip publication-ratios">
              <article><span>Total binder</span><strong>${escapeHtml(ratios.binder_total_kg_m3 ?? "-")}</strong><small>kg/m3</small></article>
              <article><span>Water-to-binder ratio</span><strong>${escapeHtml(ratios.water_binder_ratio ?? "-")}</strong><small>w/b</small></article>
              <article><span>Paste-to-aggregate ratio</span><strong>${escapeHtml(ratios.paste_aggregate_ratio ?? "-")}</strong><small>p/a</small></article>
            </div>
            ${englishSolutionTableHtml(optimization.solution_table || [])}
          </section>
          <section class="formal-section full">
            <h4>3. Candidate Scheme Comparison</h4>
            ${englishSchemeComparisonHtml(optimization.scheme_comparison || [])}
          </section>
          <section class="formal-section full">
            ${methodBaselineHtml(optimization.method_baselines || [])}
          </section>
          <section class="formal-section full">
            ${evidenceChainHtml(optimization.evidence_chain || [])}
          </section>
          <section class="formal-section full">
            <h4>6. Mechanistic Evidence and Risk Checks</h4>
            ${publicationKgMapHtml((data.agents || []).find((agent) => agent.id === "kg")?.trace || {})}
            ${englishRecommendationsHtml(data.recommendations || (Array.isArray(data.kg_hits) ? data.kg_hits : []))}
          </section>
          <section class="formal-section">
            <h4>7. Engineering Implementation Plan</h4>
            <table class="formal-table"><tbody>
              <tr><th>Raw-material audit</th><td>Measure binder activity, aggregate moisture, grading, and admixture compatibility before trial mixing.</td></tr>
              <tr><th>Trial mixing</th><td>Measure slump flow, cohesion, segregation, bleeding, air content, and density.</td></tr>
              <tr><th>Strength validation</th><td>Test 3 d, 7 d, and 28 d compressive strength and compare the results with model predictions.</td></tr>
              <tr><th>Next optimization trigger</th><td>Re-optimize if workability, strength, carbon, shrinkage, or crack-resistance targets deviate from the acceptance window.</td></tr>
            </tbody></table>
          </section>
          <section class="formal-section">
            <h4>8. Research Validation Matrix</h4>
            ${englishValidationMatrixHtml()}
          </section>
        </div>
        <section class="formal-section narrative-section">
          <h4>9. Technical Narrative</h4>
          <div class="formal-prose">${englishNarrativeHtml(data)}</div>
        </section>
        <section class="formal-section full">
          <h4>10. Evidence and References</h4>
          ${englishEvidenceReferencesHtml(data)}
        </section>
      </section>
    </details>
  `;
}

function publicationReportHtml(data = {}, kgTrace = {}) {
  if (currentLanguage() === "en") return publicationReportHtmlEn(data, kgTrace);
  const optimization = data.optimization || {};
  const reportRequests = data.report_requests || {};
  const handoffs = data.handoffs || {};
  const kgAgent = (data.agents || []).find((agent) => agent.id === "kg");
  return `
    <details class="publication-report-shell">
      <summary>
        <div>
          <span>正式成果</span>
          <strong>高强低碳低收缩混凝土工程应用与科研指导报告</strong>
          <small>点击展开完整报告，过程轨迹仍保留在上方</small>
        </div>
        <div class="report-summary-actions">
          <b>${escapeHtml(optimization.kpis?.strength_mpa ?? "-")} MPa</b>
          <b>${escapeHtml(optimization.kpis?.carbon_kgco2e_m3 ?? "-")} kgCO2e/m3</b>
        </div>
      </summary>
      <section class="publication-report">
      <header>
        <div class="report-head-row">
          <span>Design Report</span>
          <div class="report-actions">
            <button type="button" data-reference-panel>引用</button>
            <button type="button" data-report-download="md">MD</button>
            <button type="button" data-report-download="json">JSON</button>
            <button type="button" data-report-download="word">Word</button>
            <button type="button" data-report-download="pdf">PDF</button>
          </div>
        </div>
        <h3>高强低碳低收缩混凝土工程应用与科研指导报告</h3>
        <p>面向工程试配、科研验证与后续迭代优化的结构化正式报告</p>
      </header>
      <div class="publication-summary">
        ${kpiCardsHtml(optimization.kpis || {})}
      </div>
      ${reportFrontMatterHtml(optimization, reportRequests)}
      <div class="formal-report-grid">
        <section class="formal-section">
          <h4>1. 设计依据与输入边界</h4>
          ${designBasisTableHtml(data)}
          ${semanticAuditHtml((data.agents || []).find((agent) => agent.id === "requirement")?.details || {})}
          ${constraintStatusHtml(optimization.constraint_status || {})}
        </section>
        <section class="formal-section">
          <h4>2. 推荐方案与关键计算</h4>
          <div class="ratio-strip publication-ratios">${ratioCardsHtml(optimization.ratios || {})}</div>
          ${solutionTableHtml(optimization.solution_table || [])}
        </section>
        <section class="formal-section full">
          <h4>3. 传统配合比设计理论校核</h4>
          ${theoryVerificationHtml(optimization.traditional_theory || {})}
        </section>
        <section class="formal-section full">
          <h4>4. 候选配方族与方案比较</h4>
          ${candidateSchemesHtml(optimization.candidate_schemes || [])}
          ${schemeComparisonHtml(optimization.scheme_comparison || [])}
        </section>
        <section class="formal-section full">
          ${methodBaselineHtml(optimization.method_baselines || [])}
        </section>
        <section class="formal-section full">
          ${evidenceChainHtml(optimization.evidence_chain || [])}
        </section>
        <section class="formal-section full">
          <h4>7. 优化证据图</h4>
          ${reportFigureGalleryHtml(data)}
          ${scienceFigureSuiteHtml(optimization)}
        </section>
        <section class="formal-section full">
          <h4>8. 知识图谱机理证据</h4>
          ${publicationKgMapHtml(kgTrace)}
          ${kgCausalSvg(kgTrace)}
          ${kgFigureSuiteHtml(kgTrace)}
          ${evidencePanelsHtml(kgAgent?.details?.evidence || {}, kgAgent?.details?.citations || [])}
          ${citationTableHtml(kgAgent?.details?.citations || [])}
          <div class="recommendations main-recommendations">
            ${(data.recommendations || []).map((item) => `<div class="rec-card"><strong>${escapeHtml(item.topic || item.material || "机理")}</strong><p>${escapeHtml(item.mechanism || item.evidence || "")}</p><p>${escapeHtml(item.recommendation || "")}</p></div>`).join("")}
          </div>
        </section>
        <section class="formal-section">
          <h4>9. 工程实施建议</h4>
          ${implementationPlanHtml(optimization, reportRequests)}
        </section>
        <section class="formal-section">
          <h4>10. 科研验证矩阵</h4>
          ${validationMatrixHtml()}
        </section>
      </div>
      ${structuredNarrativeHtml(data)}
      <section class="publication-evidence">
        <h4>系统证据与协作轨迹</h4>
        ${trustProfileHtml(optimization.trust_profile || {})}
        ${handoffBoardHtml(handoffs)}
      </section>
      </section>
    </details>
  `;
}

function renderRunReport(data) {
  const agents = data.agents || [];
  const req = agents.find((agent) => agent.id === "requirement");
  const opt = agents.find((agent) => agent.id === "optimizer");
  const kg = agents.find((agent) => agent.id === "kg");
  const plots = data.visualizations?.optimization_plots || [];
  return `
    <div class="run-report">
      ${orchestrationHtml(agents)}
      ${handoffBoardHtml(data.handoffs || {})}
      ${roleEvidenceRoutingHtml(data)}
      ${requirementHtml(req)}
      ${optimizationHtml(opt, data.optimization || {}, plots)}
      ${kgHtml(kg, data.recommendations || data.kg_hits || [])}
      ${finalTableHtml(data.optimization || {})}
    </div>
  `;
}

function markdownReportTextEn(payload = {}) {
  const optimization = payload.optimization || {};
  const ratios = optimization.ratios || {};
  const requirement = (payload.agents || []).find((agent) => agent.id === "requirement")?.details || {};
  const comparison = (optimization.scheme_comparison || []).map((row) =>
    `| ${localizeSchemeLabel(row.label)} | ${localizeSchemeRoles(row.roles || [])} | ${row.strength_mpa ?? "-"} | ${row.carbon_kgco2e_m3 ?? "-"} | ${row.shrinkage_risk_index ?? "-"} | ${row.water_binder_ratio ?? "-"} | ${row.paste_aggregate_ratio ?? "-"} |`
  ).join("\n");
  const methodBaselines = (optimization.method_baselines || []).map((row) =>
    `| ${englishSafe(row.method, "Method")} | ${englishSafe(row.basis, "-")} | ${englishSafe(row.observed_or_expected_limitation, "-")} | ${englishSafe(row.current_system_advantage, "-")} |`
  ).join("\n") || "| - | - | - | - |";
  const evidenceChain = (optimization.evidence_chain || []).map((row) =>
    `| ${englishSafe(row.claim, "Claim")} | ${row.value ?? "-"} | ${englishSafe(row.source_type, "-")} | ${englishSafe(row.evidence, "-")} | ${englishSafe(row.validation, "-")} |`
  ).join("\n") || "| - | - | - | - | - |";
  const basis = [
    `- Target strength: ${requirement.target_mpa ?? optimization.target_mpa ?? "-"} MPa`,
    `- Curing age: ${requirement.age_days ?? optimization.age_days ?? "-"} d`,
    `- Performance priorities: ${englishList(requirement.priorities)}`,
    `- Mentioned materials: ${englishList(requirement.mentioned_materials)}`,
    `- Required materials: ${englishList(requirement.required_materials)}`,
    `- Allowed materials: ${englishList(requirement.allowed_materials)}`,
    `- Avoided materials: ${englishList(requirement.avoided_materials)}`,
  ].join("\n");
  const recommendations = (payload.recommendations || []).map((item) => {
    const rec = recommendationDisplay(item);
    return `- ${rec.topic}: ${rec.text}${rec.action ? ` ${rec.action}` : ""}`;
  }).join("\n") || "- No additional mechanistic recommendation was returned.";
  const narrative = payload.answer && !containsChinese(payload.answer)
    ? payload.answer
    : "The system translated the user request into structured constraints, performed constrained multi-objective optimization, and retained mechanistic evidence for validation. The recommended mix should be treated as a model-based candidate until trial mixing and target-age testing confirm its fresh and hardened properties.";
  const citations = collectCitations(payload);
  const refs = citations.map((item) =>
    `- [${item.ref_no || item.id}] ${item.source_type || "evidence"}: ${englishCitationTitle(item)}${item.score !== undefined ? ` (score ${item.score})` : ""}`
  ).join("\n") || "- No evidence citations were attached to this run.";
  return `# High-Strength Low-Carbon Low-Shrinkage Crack-Resistant Concrete Intelligent Design Report

## Abstract
The platform completed requirement parsing, constrained multi-objective optimization, low-shrinkage and crack-resistance risk auditing, mechanistic evidence review, and report generation for a high-strength low-carbon low-shrinkage crack-resistant concrete design task.

## Keywords
high-strength concrete; low-carbon design; low-shrinkage design; crack-resistant concrete; multi-objective optimization; knowledge graph; large language model

## 1. Design Basis and Input Boundary
${basis}

## 2. Recommended Mix and Key Ratios
- Total binder: ${ratios.binder_total_kg_m3 ?? "-"} kg/m3
- Water-to-binder ratio: ${ratios.water_binder_ratio ?? "-"}
- Paste-to-aggregate ratio: ${ratios.paste_aggregate_ratio ?? "-"}

${(optimization.solution_table || []).map((row) => `- ${localizeVariableName(row)}: ${row.value} ${row.unit || ""}`).join("\n")}

## 3. Candidate Scheme Comparison
| Scheme | Role | Strength | Carbon | Shrinkage risk | w/b | p/a |
|---|---|---:|---:|---:|---:|---:|
${comparison}

## 4. Mechanistic Recommendations
${recommendations}

## 5. Method Baseline Audit
| Method | Basis | Limitation | Current advantage |
|---|---|---|---|
${methodBaselines}

## 6. Design Evidence Chain
| Claim | Value | Evidence source | Evidence | Validation |
|---|---:|---|---|---|
${evidenceChain}

## 7. Engineering Implementation Plan
| Stage | Recommended action | Decision basis |
|---|---|---|
| Raw-material audit | Measure binder activity, aggregate moisture, grading, and admixture compatibility. | Avoid mismatch between model inputs and site materials. |
| Trial mixing | Measure slump flow, cohesion, segregation, bleeding, air content, and density. | Verify workability and pumping suitability. |
| Strength validation | Test 3 d, 7 d, and 28 d compressive strength. | Verify early-age and target-age strength. |
| Next optimization | Re-optimize if workability, strength, or carbon targets deviate from acceptance windows. | Prediction-to-test deviation exceeds the tolerance. |

## 8. Research Validation Matrix
| Category | Recommended measurements | Purpose |
|---|---|---|
| Fresh properties | Slump flow, T500, segregation/bleeding, air content | Verify workability and pumping suitability |
| Mechanical properties | 3 d, 7 d, and 28 d compressive strength | Verify early-age and target-age strength |
| Low-shrinkage and crack-resistance performance | 28 d/56 d drying shrinkage, autogenous shrinkage, restrained-ring cracking, crack-width observation, and splitting tensile or flexural toughness when required | Verify low-shrinkage and crack-resistance targets |
| Low-carbon, shrinkage and cracking synergy | CO2e per cubic metre, drying/autogenous shrinkage, restrained-ring cracking, crack width | Verify the manuscript-facing design target |
| Carbon accounting | kgCO2e per cubic metre | Audit the carbon-reduction benefit |

## 9. Technical Narrative
${narrative}

## 10. Evidence and References
${refs}`;
}

function markdownReportText(payload = {}) {
  if (currentLanguage() === "en") return markdownReportTextEn(payload);
  const optimization = payload.optimization || {};
  const ratios = optimization.ratios || {};
  const requirement = (payload.agents || []).find((agent) => agent.id === "requirement")?.details || {};
  const ratioBlock = [
    `- 胶凝材料总量：${ratios.binder_total_kg_m3 ?? "-" } kg/m3`,
    `- 水胶比：${ratios.water_binder_ratio ?? "-"}`,
    `- 浆骨比：${ratios.paste_aggregate_ratio ?? "-"}`,
  ].join("\n");
  const comparison = (optimization.scheme_comparison || []).map((row) =>
    `| ${row.label} | ${(row.roles || []).join(" / ")} | ${row.strength_mpa ?? "-"} | ${row.carbon_kgco2e_m3 ?? "-"} | ${row.shrinkage_risk_index ?? "-"} | ${row.water_binder_ratio ?? "-"} | ${row.paste_aggregate_ratio ?? "-"} |`
  ).join("\n");
  const methodBaselines = (optimization.method_baselines || []).map((row) =>
    `| ${row.method_zh || row.method || "-"} | ${row.basis_zh || row.basis || "-"} | ${row.observed_or_expected_limitation_zh || row.observed_or_expected_limitation || "-"} | ${row.current_system_advantage_zh || row.current_system_advantage || "-"} |`
  ).join("\n") || "| - | - | - | - |";
  const evidenceChain = (optimization.evidence_chain || []).map((row) =>
    `| ${row.claim_zh || row.claim || "-"} | ${row.value ?? "-"} | ${row.source_type || "-"} | ${row.evidence_zh || row.evidence || "-"} | ${row.validation_zh || row.validation || "-"} |`
  ).join("\n") || "| - | - | - | - | - |";
  const basis = [
    `- 目标强度：${requirement.target_mpa ?? optimization.target_mpa ?? "-"} MPa`,
    `- 龄期：${requirement.age_days ?? optimization.age_days ?? "-"} d`,
    `- 性能优先级：${(requirement.priorities || []).join(" / ") || "-"}`,
    `- 提及材料：${(requirement.mentioned_materials || []).join(" / ") || "无"}`,
    `- 必用材料：${(requirement.required_materials || []).join(" / ") || "无"}`,
    `- 允许材料：${(requirement.allowed_materials || []).join(" / ") || "无"}`,
    `- 禁用材料：${(requirement.avoided_materials || []).join(" / ") || "无"}`,
  ].join("\n");
  const recommendations = (payload.recommendations || []).map((item) =>
    `- ${item.topic || item.material || "机理"}：${item.recommendation || item.mechanism || item.evidence || ""}`
  ).join("\n") || "- 当前无额外机理建议。";
  const theory = optimization.traditional_theory || {};
  return `# 高强低碳低收缩混凝土智能设计报告

## 摘要
系统依据用户约束完成需求解析、多目标优化、低收缩风险审计、知识图谱复审与正式报告生成。

## 关键词
高强混凝土；低碳设计；低收缩；多目标优化；知识图谱；大语言模型

## 1. 设计依据与输入边界
${basis}

## 2. 推荐方案与关键计算
${ratioBlock}

${(optimization.solution_table || []).map((row) => `- ${row.name}: ${row.value} ${row.unit || ""}`).join("\n")}

## 3. 传统配合比设计理论校核
- 绝对体积：${theory.absolute_volume_m3 ?? "-"} m3
- 体积误差：${theory.absolute_volume_error ?? "-"} m3
- 理论容重：${theory.theoretical_density_kg_m3 ?? "-"} kg/m3
- 鲍罗米强度：${theory.bolomey_strength_mpa ?? "-"} MPa
- 绝对体积法：${theory.absolute_volume_pass ? "通过" : "未通过"}
- 鲍罗米校核：${theory.bolomey_pass ? "通过" : "未通过"}

## 4. 方案族比较
| 方案 | 角色 | 强度 | 碳排 | 收缩风险 | 水胶比 | 浆骨比 |
|---|---|---:|---:|---:|---:|---:|
${comparison}

## 5. 知识图谱机理建议
${recommendations}

## 6. 方法基线审计
| 方法 | 依据 | 局限 | 当前平台优势 |
|---|---|---|---|
${methodBaselines}

## 7. 设计证据链
| 结论 | 数值/状态 | 证据类型 | 证据说明 | 验证动作 |
|---|---:|---|---|---|
${evidenceChain}

## 8. 工程实施建议
| 阶段 | 建议动作 | 判定依据 |
|---|---|---|
| 原材复核 | 核查胶凝材料活性、骨料含水率、减水剂相容性 | 避免模型输入与现场原材脱节 |
| 试拌确认 | 实测坍落扩展度、黏聚性和含气量 | 验证施工性 |
| 强度验证 | 测试 3 d、7 d、28 d 抗压强度 | 验证目标龄期强度 |
| 二次优化 | 若工作性或强度偏离目标，则依据实测结果重新优化 | 预测与实测差异超限 |

## 9. 科研验证矩阵
| 项目 | 建议指标 | 用途 |
|---|---|---|
| 新拌性能 | 坍落扩展度、T500、离析/泌水、含气量 | 验证施工性与泵送适应性 |
| 力学性能 | 3 d、7 d、28 d 抗压强度 | 验证早期与目标龄期强度 |
| 低收缩性能 | 28 d/56 d 干燥收缩、自收缩、环形约束开裂、裂缝宽度观察；必要时劈裂抗拉或弯曲韧性 | 验证低收缩目标 |
| 低碳-收缩协同 | 单位体积 CO2e、干燥/自收缩、环形约束开裂、裂缝宽度 | 验证论文核心设计目标 |
| 低碳核算 | 单位体积 kgCO2e/m3 | 校核减碳收益 |

## 10. 技术论证正文
${payload.answer || ""}`;
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

function downloadWordReport(payload = {}) {
  const kgTrace = (payload.agents || []).find((agent) => agent.id === "kg")?.trace || {};
  const html = `
    <html><head><meta charset="utf-8"><style>
      body{font-family:Arial,"Microsoft YaHei",sans-serif;line-height:1.8;color:#202123}
      table{border-collapse:collapse;width:100%;margin:12px 0}
      th,td{border:1px solid #ccc;padding:8px;text-align:left}
      h1,h2,h3,h4,h5{font-family:Arial,"Microsoft YaHei",sans-serif}
      img{max-width:100%;height:auto}
      figure{margin:14px 0}
    </style></head><body>${publicationReportHtml(payload, kgTrace)}</body></html>`;
  downloadBlob("concrete-design-report.doc", html, "application/msword");
}

function structuredReportExport(payload = {}) {
  const clone = typeof structuredClone === "function" ? structuredClone(payload) : JSON.parse(JSON.stringify(payload || {}));
  if (clone.visualizations?.optimization_plots) {
    clone.visualizations.optimization_plots = clone.visualizations.optimization_plots.map((plot) => ({
      name: plot.name,
      plot_type: plot.plot_type || plot.type || "",
      has_image: Boolean(plot.image),
      image_omitted: true,
      table_file: plot.xlsx_file || plot.table_file || "",
    }));
  }
  return {
    exported_at: new Date().toISOString(),
    language: currentLanguage(),
    report_markdown: markdownReportText(payload),
    optimization: clone.optimization || {},
    agents: clone.agents || [],
    kg_hits: clone.kg_hits || [],
    recommendations: clone.recommendations || [],
    handoffs: clone.handoffs || {},
    report_requests: clone.report_requests || {},
    references: collectCitations(payload),
  };
}

function printPdfReport(payload = {}) {
  const win = window.open("", "_blank");
  if (!win) return;
  const kgTrace = (payload.agents || []).find((agent) => agent.id === "kg")?.trace || {};
  win.document.write(`
    <html><head><meta charset="utf-8"><title>Concrete Design Report</title>
    <style>body{font-family:Arial,"Microsoft YaHei",sans-serif;line-height:1.85;padding:28px;color:#202123}
    table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid #ccc;padding:8px;text-align:left}
    img{max-width:100%;height:auto}.report-actions{display:none}.formal-report-grid{display:block}.formal-section{margin:0 0 18px}.science-grid{display:block}
    </style>
    </head><body>${publicationReportHtml(payload, kgTrace)}</body></html>`);
  win.document.close();
  win.focus();
  win.print();
}

async function playDetailedReport(root, data) {
  const agents = data.agents || [];
  const req = agents.find((agent) => agent.id === "requirement");
  const opt = agents.find((agent) => agent.id === "optimizer");
  const kg = agents.find((agent) => agent.id === "kg");
  const plots = data.visualizations?.optimization_plots || [];
  root.innerHTML = `<div class="run-report">${orchestrationHtml(agents)}</div>`;
  const report = root.querySelector(".run-report");
  await sleep(450);
  report.insertAdjacentHTML("beforeend", requirementHtml(req));
  scrollChatIfFollowing();
  await sleep(650);
  report.insertAdjacentHTML("beforeend", optimizationHtml(opt, data.optimization || {}, plots));
  scrollChatIfFollowing();
  await sleep(650);
  report.insertAdjacentHTML("beforeend", kgHtml(kg, data.recommendations || data.kg_hits || []));
  scrollChatIfFollowing();
  await sleep(650);
  report.insertAdjacentHTML("beforeend", finalTableHtml(data.optimization || {}));
  scrollChatIfFollowing();
}

function parseBoundsJson() {
  const text = document.getElementById("boundsJson").value.trim();
  if (!text) return null;
  return JSON.parse(text);
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const payload = await response.json();
    coreStatus.textContent = payload.ml_core_loaded ? "已连接" : "异常";
    lmStatus.textContent = `LM Studio: ${payload.lm_studio_url.replace(/^https?:\/\//, "")}`;
    if (payload.model_loaded) modelStatus.textContent = `已加载：${payload.package_name}`;
  } catch (error) {
    coreStatus.textContent = "离线";
  }
}

async function uploadModel() {
  const file = modelFile.files?.[0];
  if (!file) return;
  modelStatus.textContent = "模型加载中...";
  const form = new FormData();
  form.append("file", file);
  try {
    const response = await fetch("/api/model/upload", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok || payload.status !== "success") throw new Error(payload.message || "模型加载失败");
    const variables = payload.variables || [];
    document.getElementById("boundsJson").value = JSON.stringify(
      variables.map((item) => ({ name: item.safe, lower: item.lower, upper: item.upper, original: item.original })),
      null,
      2
    );
    modelStatus.textContent = `已加载 ${payload.model_name || file.name}；变量 ${variables.length} 个`;
    addMessage("assistant", `模型包已加载：${payload.model_name || file.name}\n已识别 ${variables.length} 个优化变量。`);
  } catch (error) {
    modelStatus.textContent = `加载失败：${error.message}`;
    addMessage("assistant", `模型加载失败：${error.message}`);
  }
}

async function runDesign(prompt) {
  followChatTail = false;
  addMessage("user", prompt);
  const pending = addHtmlMessage("assistant", liveRunHtml(), { scroll: false, kind: "live-run" });
  scrollBubbleToTop(pending);
  resetStages();
  markStage("requirement", "running", currentLanguage() === "en" ? "Parsing strength, material, and performance constraints..." : "解析强度、材料和性能约束...");
  setBusy(true);

  try {
    const payload = {
      prompt,
      language: currentLanguage(),
      algorithm: document.getElementById("algorithmSelect").value,
      pop_size: Number(document.getElementById("popSize").value),
      generations: Number(document.getElementById("generations").value),
      variables: parseBoundsJson(),
    };

    await sleep(350);
    markStage("requirement", "completed", currentLanguage() === "en" ? "Requirement submitted for backend parsing" : "需求已提交后端解析");
    markStage("optimizer", "running", currentLanguage() === "en" ? "Building objective functions and constraints, then running multi-objective optimization..." : "构造目标函数、约束并运行多目标优化...");
    updateLivePhase(pending, "optimizer");

    const response = await fetch("/api/agent/design/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok || !response.body) throw new Error(currentLanguage() === "en" ? "Failed to start streaming endpoint" : "流式接口启动失败");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalData = null;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const msg = JSON.parse(line);
        if (msg.event === "error") throw new Error(msg.message);
        handleStreamEvent(pending, msg);
        if (msg.event === "done") finalData = msg.data;
      }
    }
    if (!finalData) throw new Error(currentLanguage() === "en" ? "No final result was received" : "未收到最终结果");
    markStage("optimizer", "completed", currentLanguage() === "en" ? "Optimization completed; retrieving the knowledge graph" : "优化完成，正在检索知识图谱");
    markStage("kg", "completed", currentLanguage() === "en" ? "Knowledge-graph retrieval completed" : "知识图谱检索完成");
    appendLiveEvent(pending, currentLanguage() === "en" ? "Process retained" : "过程保留", currentLanguage() === "en" ? "The complete structured trace has been retained; the formal report is generated below." : "完整思考轨迹已保留；正式报告在下方单独生成。", "success");
    latestReportPayload = finalData;
    const kgAgent = (finalData.agents || []).find((agent) => agent.id === "kg");
    const resultBubble = addHtmlMessage("assistant", renderRunReport(finalData), { scroll: false, kind: "run-report" });
    scrollBubbleToTop(resultBubble);
    addHtmlMessage("assistant", publicationReportHtml(finalData, kgAgent?.trace || {}), { scroll: false, kind: "publication-report" });
    renderSolution(finalData.optimization?.solution_table || []);
    renderKpis(finalData.optimization?.kpis || {});
    renderProcessDetails(finalData.agents || []);
    renderKgGraph(kgAgent?.trace || {});
    renderRecommendations(finalData.recommendations || finalData.kg_hits || []);
    modelStatus.textContent = finalData.lm_ok ? "LM Studio 已参与回答" : "LM Studio 未连接，已使用规则化回答";
  } catch (error) {
    appendLiveEvent(pending, "运行失败", error.message);
    addMessage("assistant", `运行失败：${error.message}`);
    markStage("optimizer", "", currentLanguage() === "en" ? "Run failed" : "运行失败");
  } finally {
    setBusy(false);
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt) return;
  promptInput.value = "";
  runDesign(prompt);
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

modelFile.addEventListener("change", uploadModel);
newSessionButton?.addEventListener("click", startNewSession);
document.addEventListener("platform-language-change", handlePlatformLanguageChange);

document.addEventListener("click", (event) => {
  const citationClose = event.target.closest("[data-citation-close]");
  if (citationClose) {
    citationClose.closest(".citation-drawer")?.remove();
    return;
  }
  const citation = event.target.closest("[data-citation-id]");
  if (citation) {
    openCitationDetail(citation.dataset.citationId).catch((error) => {
      modelStatus.textContent = `证据读取失败：${error.message}`;
    });
    return;
  }
  const openSession = event.target.closest("[data-session-open]");
  if (openSession) {
    loadSession(openSession.dataset.sessionOpen);
    return;
  }
  const deleteButton = event.target.closest("[data-session-delete]");
  if (deleteButton) {
    deleteSession(deleteButton.dataset.sessionDelete);
    return;
  }
  const referencePanel = event.target.closest("[data-reference-panel]");
  if (referencePanel) {
    openReferencesDrawer();
    return;
  }
  const button = event.target.closest("[data-report-download]");
  if (!button || !latestReportPayload) return;
  const kind = button.dataset.reportDownload;
  if (kind === "md") downloadBlob("concrete-design-report.md", markdownReportText(latestReportPayload), "text/markdown;charset=utf-8");
  if (kind === "json") downloadBlob("concrete-design-report-structured-package.json", JSON.stringify(structuredReportExport(latestReportPayload), null, 2), "application/json;charset=utf-8");
  if (kind === "word") downloadWordReport(latestReportPayload);
  if (kind === "pdf") printPdfReport(latestReportPayload);
});

document.addEventListener("mouseover", (event) => {
  const citation = event.target.closest("[data-citation-id]");
  if (!citation) return;
  window.clearTimeout(citationHoverTimer);
  citationHoverTimer = window.setTimeout(() => {
    showCitationPopover(citation).catch((error) => {
      modelStatus.textContent = `证据读取失败：${error.message}`;
    });
  }, 120);
});

document.addEventListener("mouseout", (event) => {
  const citation = event.target.closest("[data-citation-id]");
  const popover = event.target.closest(".citation-popover");
  if (!citation && !popover) return;
  const next = event.relatedTarget;
  if (next?.closest?.("[data-citation-id], .citation-popover")) return;
  citationHoverTimer = window.setTimeout(removeCitationPopover, 120);
});

renderSolution([]);
renderKpis({});
renderRecommendations([]);
renderKgGraph({});
initializeSessions();
loadStatus();
