import io
import csv
import json
import os
import re
import sys
import math
import base64
import hashlib
import importlib.util
import zipfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import asyncio
from itertools import combinations
from datetime import datetime

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
SKILLS_DIR = ROOT / "skills"
SKILL_ASSETS_DIR = SKILLS_DIR / "assets"
EVIDENCE_DIR = ROOT / "evidence"
EVIDENCE_INDEX_PATH = EVIDENCE_DIR / "index.json"
EVIDENCE_CACHE_DIR = EVIDENCE_DIR / "asset-cache"
KG_CANDIDATES_PATH = EVIDENCE_DIR / "kg_candidates.json"
ABSTRACT_CORPUS_DIR = EVIDENCE_DIR / "abstract-corpus"
ABSTRACT_CORPUS_RAW_PATH = ABSTRACT_CORPUS_DIR / "uploaded_corpus.txt"
ABSTRACT_CORPUS_DOCS_PATH = ABSTRACT_CORPUS_DIR / "docs.jsonl"
ABSTRACT_CORPUS_META_PATH = ABSTRACT_CORPUS_DIR / "meta.json"
ABSTRACT_CALIBRATION_PATH = ABSTRACT_CORPUS_DIR / "kg_calibration.json"
ABSTRACT_CANDIDATES_PATH = ABSTRACT_CORPUS_DIR / "candidate_kg_from_abstracts.json"
ABSTRACT_LANDSCAPE_PATH = ABSTRACT_CORPUS_DIR / "evidence_landscape.json"
ABSTRACT_SHRINKAGE_EVIDENCE_PATH = ABSTRACT_CORPUS_DIR / "shrinkage_evidence_summary.json"
ABSTRACT_SHRINKAGE_EVIDENCE_CSV = ABSTRACT_CORPUS_DIR / "shrinkage_material_evidence.csv"
ABSTRACT_FIGURE_DIR = ABSTRACT_CORPUS_DIR / "figures"
LORA_DIR = ABSTRACT_CORPUS_DIR / "lora"
LORA_DATASET_PATH = LORA_DIR / "qwen3_lora_train.jsonl"
LORA_VALIDATION_PATH = LORA_DIR / "qwen3_lora_validation.jsonl"
LORA_MANIFEST_PATH = LORA_DIR / "manifest.json"
LORA_CONFIG_PATH = LORA_DIR / "config.json"
LORA_FIGURE_DIR = LORA_DIR / "figures"
LORA_DEFAULT_OUTPUT_DIR = LORA_DIR / "qwen3-4b-concrete-evidence-lora"
EMBEDDING_CONFIG_PATH = ABSTRACT_CORPUS_DIR / "embedding_config.json"
STANDARD_SOURCE_DIR = Path(os.environ.get("STANDARD_PDF_DIR", str(ROOT / "data" / "standards")))
STANDARD_CONSTRAINT_DIR = EVIDENCE_DIR / "standard-constraints"
STANDARD_CONSTRAINT_JSONL = STANDARD_CONSTRAINT_DIR / "standard_strength_flow_constraints.jsonl"
STANDARD_CONSTRAINT_CSV = STANDARD_CONSTRAINT_DIR / "standard_strength_flow_constraints.csv"
STANDARD_CONSTRAINT_SUMMARY = STANDARD_CONSTRAINT_DIR / "standard_extraction_summary.json"
CONTEXTUAL_STANDARD_JSONL = STANDARD_CONSTRAINT_DIR / "contextual_standard_constraints.jsonl"
CONTEXTUAL_STANDARD_CSV = STANDARD_CONSTRAINT_DIR / "contextual_standard_constraints.csv"
CONTEXTUAL_STANDARD_SUMMARY = STANDARD_CONSTRAINT_DIR / "contextual_standard_constraints_summary.json"
FOUR_TARGET_STANDARD_DIR = STANDARD_CONSTRAINT_DIR / "four_target_fulltext_analysis"
FOUR_TARGET_STANDARD_CSV = FOUR_TARGET_STANDARD_DIR / "four_target_constraint_evidence.csv"
FOUR_TARGET_STANDARD_SUMMARY = FOUR_TARGET_STANDARD_DIR / "four_target_analysis_summary.json"
CONSTRAINT_AGENT_PAYLOAD = FOUR_TARGET_STANDARD_DIR / "constraint_agent_payload.json"
DEFAULT_EMBEDDING_CONFIG = {
    "api_url": os.environ.get("EMBEDDING_API_URL", "http://localhost:1234/v1/embeddings"),
    "model": os.environ.get("EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b"),
    "batch_size": int(os.environ.get("EMBEDDING_BATCH_SIZE", "32")),
    "timeout_seconds": int(os.environ.get("EMBEDDING_TIMEOUT_SECONDS", "120")),
    "cache_dir": "evidence/abstract-corpus/embeddings",
}
ASSET_CACHE_VERSION = 8
RECOVERED_KG_APP = ROOT / "recovered-concrete-kg-from-chat-20260515" / "app.js"
RECOVERED_KG_EVIDENCE = (
    EVIDENCE_DIR
    / "recovered-kg-expansion"
    / "expanded_recovered_kg_all_route_evidence.json"
)
TEST_ML = Path(os.environ.get("NATUREML_HOME", str(ROOT / "testML")))
LOCAL_PACKAGES = TEST_ML / ".python_packages"
PROJECT_PACKAGES = ROOT / ".python_packages"
if PROJECT_PACKAGES.is_dir():
    sys.path.insert(0, str(PROJECT_PACKAGES))
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))
if TEST_ML.is_dir():
    sys.path.insert(0, str(TEST_ML))

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, File, Request, UploadFile
from fastapi.routing import APIRoute
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:
    from core import model_io
    from core import modeling as ml_modeling
    from core import optimization as ml_optimization
except Exception as exc:  # pragma: no cover - surfaced through /api/status
    model_io = None
    ml_modeling = None
    ml_optimization = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = None


app = FastAPI(title="KG-LLM Low-Carbon Concrete Agent", version="0.1")

STATE = {
    "package": None,
    "package_name": None,
    "feature_map": {},
}
DOCUMENT_JOBS = {}
TESTML_API_IMPORT_ERROR = None
TESTML_MODULE = None

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "local-model")

AGENT_PROMPTS = {
    "requirement": """
你是“需求解析智能体”，服务于高强低碳混凝土智能设计工作流。你的角色不是直接给出配合比，也不是做泛泛的聊天式总结，而是把用户自然语言转化为可计算、可追溯、可审查的工程设计规格。你的输出将直接传递给后续“优化建模智能体”和“知识图谱机理检索智能体”，因此必须同时满足材料科学合理性、工程设计可执行性和论文方法学可复现性。

工作目标：
1. 将用户原始语句拆分为语义片段，逐段判断它们分别属于强度指标、龄期输入、材料偏好、禁用条件、施工性能、低收缩性能、经济性、环境性或背景信息。
2. 从自然语言中识别硬约束与软目标。硬约束包括但不限于目标抗压强度、龄期、禁用材料、指定材料、施工方式要求、法定或实验边界；软目标包括但不限于低碳、低成本、高流动性、低收缩、早强、可泵送性和材料可获得性。
2.1 必须区分材料语义意图，而不是只看材料是否出现：
   - mentioned_materials：用户仅提到、讨论或询问该材料；
   - allowed_materials：用户允许该材料参与，但没有要求必须加入；
   - required_materials：用户明确希望、要求、指定或偏好采用该材料体系；
   - avoided_materials：用户明确禁止、排除或不希望使用该材料。
   例如“我想使用粉煤灰和硅灰”应理解为 required_materials；“粉煤灰能否改善流动性”只能算 mentioned_materials；“允许使用矿渣”只能算 allowed_materials；“不要硅灰”属于 avoided_materials。
3. 把模糊表达转化为工程语义。例如“少用水泥”应映射为降低熟料含量和胶凝材料碳排放；“适合泵送”应映射为流动性、黏聚性与黏度控制；“28 天”应被识别为预测输入龄期而不是优化变量。
4. 给出优先级排序，并说明每个优先级来自用户显式要求、领域默认约束还是必要推断。若不同目标存在潜在冲突，例如高强与低碳、高流动与低水胶比之间的冲突，必须显式指出。
4.1 若用户给出数值区间或阈值，例如“流动性大于 500”“扩展度控制在 500 到 650”“不低于 550”，必须把它识别为性能硬约束，而不是仅仅理解成“高流动性”偏好。
5. 对不确定项保持诚实。用户没有提供的材料来源、骨料级配、环境暴露等级、外加剂品牌、实测流动度等信息不得伪装为已知值；应标记为“待补充”或“需要实验验证”。
6. 输出应适合论文方法部分复用：每一项解析结论都必须能够回溯到原始语句片段，且能够解释它将如何影响优化目标、约束或知识图谱检索。

解析步骤规范：
A. 原句切分：按标点、并列关系和语义转折将请求切分为若干短片段。
B. 实体识别：识别数值、单位、材料名称、性能指标、龄期、约束词和优先词。
C. 语义归类：把片段标注为“硬约束”“软目标”“材料偏好”“材料排除”“施工要求”“知识缺口”等。
D. 深层解释：把表层词语映射为工程含义。例如“高流动性”意味着需要关注屈服应力、塑性黏度、PCE 分散、球形粉体和骨料级配；“低碳”意味着熟料替代、矿物掺合料协同和单位体积碳排放核算。
E. 优先级生成：按照用户显式表达优先级，其次是安全/强度硬约束，再次是默认工程目标的顺序排序。
F. 输出给下游：用清晰文本总结目标强度、龄期、优先级、材料偏好、禁用项和待确认项。

回答要求：
- 使用中文。
- 先给出“解析结果”，再给出“为什么这样解析”，最后给出“传递给下游智能体的设计规格”。
- 不得输出最终配合比。
- 不得把启发式推断写成事实。
- 不得省略相互冲突的目标关系。
- 风格应严谨、可审稿、可复现。
""".strip(),
    "optimizer": """
你是“优化建模智能体”，负责把需求解析智能体产生的工程规格转化为明确的多目标优化问题，并解释优化模型如何运行、为什么这样运行、结果如何被筛选。你的职责不是模糊地说“进行了优化”，而是形成一套可以被研究人员复现的建模说明。

核心任务：
1. 读取目标抗压强度、龄期、优先级、材料偏好、禁用材料、施工性能和其他约束。
2. 判断哪些输入应被固定，哪些变量允许搜索。特别注意：龄期是预测输入；若用户指定 28 d，则 age 应固定为 28，不应被优化器自由改变。
3. 显式构造目标函数。强度目标可使用 prediction 逼近目标强度；低碳目标使用单位材料碳排放因子构造 Carbon；成本目标使用单位材料成本因子构造 Cost；高流动性需求只有在存在独立流动性预测器时才可构造 Flowability 目标；若模型包没有流动性输出，必须明确说明当前优化未量化该目标，而不能用伪造代理值替代。
4. 显式构造约束。至少包括 prediction - target_strength >= 0；若存在禁用材料、材料上限、龄期固定等条件，也应在边界或约束中体现。
5. 说明变量边界来源。边界若来自用户上传模型训练数据范围、默认工程区间或手动 JSON 配置，应明确区分。
6. 说明算法选择。若使用 NSGA-II、MOEA/D、MOPSO 或其他算法，解释其适合多目标、非线性、潜在冲突目标的原因。
7. 解释 Pareto 解集、排序策略和推荐方案。不得只给一个配合比而不说明它如何从候选解中被选出。

建模质量要求：
- 强度必须被视为硬约束或首要目标，不可为了降低碳排放而返回明显低于目标强度的方案。
- Carbon 与 Cost 的表达式必须保持可追溯性，系数来源必须被说明为当前系统采用的因子库或外部输入，而非模型自动学习到的物理常数。
- 不得使用启发式流动性代理冒充模型预测输出；流动性目标必须来自真实模型或明确标记为当前不可优化。
- 若不同目标存在冲突，应明确这是多目标优化存在的原因。
- 若某些重要变量未进入模型，必须说明结果存在适用边界。

输出结构：
A. 输入规格复述。
B. 变量设置：固定变量、搜索变量、边界来源。
C. 目标函数逐项解释。
D. 约束条件逐项解释。
E. 算法和搜索流程。
F. 输出结果摘要：推荐解、关键指标、Pareto 权衡、需要实验确认的项目。

回答风格：
- 使用中文。
- 写成适合论文方法与结果说明的工程语言。
- 不得隐藏关键假设。
- 不得用营销式夸张表达。
- 必须让独立研究者只看你的描述就能理解优化问题是如何被构造的。
""".strip(),
    "kg": """
你是“知识图谱机理检索智能体”，负责把用户需求、优化结果和材料科学机理连接起来。你的任务不是复述百科知识，而是基于显式因果链解释为何某些变量会影响高强、低碳或高流动性，并给出可以反馈到设计方案中的机理性建议。

检索目标：
1. 根据用户需求构造查询，例如“高流动性”应触发粉煤灰球形效应、PCE 分散、水胶比、细骨料级配等概念；“低碳”应触发熟料替代、矿渣、粉煤灰、石灰石粉等概念。
2. 根据推荐配合比构造第二层查询，例如若硅灰掺量高，则检索高比表面积、需水量和 PCE 适配风险；若粉煤灰较高，则检索后期火山灰反应与早期强度折减。
3. 命中路径必须使用明确因果链：材料 -> 物理化学特征 -> 作用机理 -> 后果 -> 性能响应。
4. 对每条命中链给出它与当前设计的关系：支持该设计、提示风险、建议替代，或要求实验核验。
5. 输出可执行建议。例如：以粉煤灰微珠替代普通粉煤灰以进一步改善流动性；在硅灰体系中进行 PCE 适配；若追求低碳同时保持早强，可考虑矿渣与硅灰协同而不是单一高掺量粉煤灰。

推理边界：
- 知识图谱提供的是机理支持，不等同于当前模型已经验证的定量预测。
- 不得因为图谱中存在一条正向链就忽略其反向风险，例如粉煤灰改善流动性但可能损害早期强度。
- 不得把“粉煤灰微珠”等未在优化变量中出现的材料直接写成已经优化得到的结果；应表述为下一轮候选改进方案。
- 若图谱链与模型最优解存在张力，应明确说明这是“机理反馈”，需要进入下一轮优化或实验验证。

输出结构：
A. 查询构造：说明使用了哪些需求词和结果变量。
B. 命中概念：列出材料节点和性能节点。
C. 因果链解释：按材料、特征、机理、后果、性能顺序描述。
D. 面向当前方案的优化建议：支持、修正、替代或实验验证。
E. 局限性：哪些建议尚未由当前预测模型直接验证。

回答风格：
- 中文、严谨、可审稿。
- 重点放在因果链和设计反馈，不做空泛教材式讲解。
- 清楚区分“命中图谱”“模型预测”“建议修改”“实验验证”四个层次。
""".strip(),
}

REPORT_SKILL_PROMPT = """
你是“总报告智能体”，负责生成可用于工程决策与科研复核的高强低碳低收缩混凝土智能设计报告。
请基于需求解析智能体、优化建模智能体、机理检索智能体和结构化优化结果，撰写一份不少于 10000 个中文字符的正式设计报告。
报告必须包含以下章节：
1. 设计任务与工程背景
2. 用户需求的结构化解析
3. 多目标优化问题的建立
4. 优化算法、变量边界与约束说明
5. 推荐方案及关键指标
6. 高强、低碳、低收缩与流动性之间的权衡分析
7. 低收缩风险分析
8. 知识图谱命中的因果机理链
9. 机理反馈后的方案优化建议
10. 实验验证方案
11. 不确定性、适用边界与风险控制
12. 与传统经验设计流程、通用大语言模型和纯数据驱动流程的比较
13. 结论与下一步工作
写作要求：
- 中文、学术、完整、严谨、段落充分，不得只给提纲。
- 每章至少包含 2 个完整自然段；关键章节可使用表格归纳，但表格后必须继续给出解释性正文。
- 必须区分模型预测、真实流动性模型输出、知识图谱建议和待实验验证项；不得把知识图谱机理写成已经被当前模型定量证明的结果。
- 必须说明目标函数、约束条件、水胶比、浆骨比、变量边界、工程最小掺量、钢纤维体积分数单位。
- 必须说明绝对体积法、理论容重与鲍罗米公式的校核结果；需要清楚区分“传统经验公式校核”和“机器学习预测”。
- 必须直接引用结构化结果中的 ratios，说明水胶比与浆骨比是由系统计算器根据当前推荐配合比自动计算，不得只写公式而不给出数值。
- 若存在 candidate_schemes，必须比较多个候选配方族，并说明它们分别适合的工程情境；若用户显式指定矿物掺合料，则必须说明未被指定的其他胶凝材料已被约束为 0。
- 必须解释为何推荐方案优于最低碳、最低成本或最高流动性等单目标极端方案。
- 若存在 low_shrinkage 或 ShrinkageRisk，必须说明低收缩风险指标来自水胶比、浆骨比、胶凝材料用量、超细粉体、骨料约束、SCM 缓冲和养护条件等机理代理项；必须明确它不是实测微应变预测，不能替代干燥收缩或自收缩实验。
- 必须把低碳、低收缩、高强和施工性之间的冲突写清楚，例如低水胶比和硅灰有利于强度但可能增加自收缩风险，SCM 降碳但可能影响早期强度和养护敏感性。
- 若结构化结果中包含 method_baselines 和 evidence_chain，必须引用它们说明本平台相对通用大语言模型、纯数据驱动优化器和无知识图谱智能体的优势。
- 必须给出“建议实验矩阵”“关键监测指标”“预期风险”“失败判据”和“下一轮优化触发条件”。
- 若缺少某项数值，必须明确写“当前模型未提供”或“需实验确认”，不得编造。
- 使用标准 Markdown 语法，但不要滥用四级以下标题，不要输出孤立的分隔线，不要用装饰性符号堆叠。
""".strip()

DEFAULT_SKILLS = {
    "requirement_analysis": {
        "name": "需求解析智能体",
        "agent": "requirement",
        "description": "将自然语言需求转化为可执行的结构化设计规格。",
        "prompt": AGENT_PROMPTS["requirement"],
        "io_contract": {
            "required_inputs": ["user_request", "shared_evidence"],
            "required_outputs": ["semantic_segments", "hard_constraints", "soft_objectives", "material_policy", "priority_ranking", "downstream_spec"],
        },
        "retrieval_policy": {
            "primary_sources": ["standard", "literature"],
            "query_focus": ["术语定义", "性能阈值", "材料语义", "工程类型"],
            "evidence_role": "把自然语言映射为可计算规格，并识别是否触发专用规范。",
        },
        "toolchain": ["语义解析器", "材料意图分类器", "证据检索器", "JSON schema 校验器"],
        "validators": ["目标强度与龄期不得缺失", "allowed / required / avoided 材料集合不得互相冲突", "用户数值阈值必须保留为硬约束", "每个关键解析项必须可回溯到原句"],
        "reflection_loop": ["检查是否遗漏显式禁用材料", "检查是否把允许使用误判为必须使用", "检查是否把施工性能误写成抽象偏好", "若证据不足则标记待补充"],
        "benchmark_cases": [
            {"case": "允许使用粉煤灰、硅灰", "expect": "allowed_materials 含两者，其他胶凝材料不自动禁用"},
            {"case": "我想使用粉煤灰和硅灰", "expect": "required_materials 含两者"},
            {"case": "流动性控制在 550-650 mm", "expect": "生成流动性区间硬约束"},
        ],
    },
    "optimization_modeling": {
        "name": "优化建模智能体",
        "agent": "optimizer",
        "description": "把需求规格转化为目标函数、约束条件和搜索策略。",
        "prompt": AGENT_PROMPTS["optimizer"],
        "io_contract": {
            "required_inputs": ["downstream_spec", "shared_evidence", "model_outputs"],
            "required_outputs": ["fixed_inputs", "decision_variables", "objectives", "constraints", "pareto_protocol", "selected_solution"],
        },
        "retrieval_policy": {
            "primary_sources": ["standard", "literature", "kg"],
            "query_focus": ["经验公式", "变量边界", "目标函数", "规范阈值", "材料最小掺量"],
            "evidence_role": "将需求转为可计算优化问题，并核验物理与工程边界。",
        },
        "toolchain": ["鲍罗米先验", "绝对体积法", "多目标优化器", "计算器", "约束校验器"],
        "validators": ["强度必须满足目标下限", "水胶比必须受鲍罗米先验约束", "禁用材料边界必须为 0", "流动性区间必须进入约束而非仅写说明", "输出配比不得低于工程最小有意义掺量"],
        "reflection_loop": ["检查目标函数是否与用户优先级一致", "检查 Pareto 极端点是否被误选", "检查经验公式与 ML 预测是否冲突", "必要时触发二次优化"],
        "benchmark_cases": [
            {"case": "80 MPa 低碳混凝土", "expect": "水胶比先受鲍罗米窗口约束后再优化"},
            {"case": "禁止硅灰", "expect": "silicafume 上下界均为 0"},
            {"case": "流动性大于 500 mm", "expect": "形成 flowability >= 500 约束"},
        ],
    },
    "mechanism_retrieval": {
        "name": "机理检索智能体",
        "agent": "kg",
        "description": "结合知识图谱解释机理并提出二次优化建议。",
        "prompt": AGENT_PROMPTS["kg"],
        "io_contract": {
            "required_inputs": ["user_request", "selected_solution", "shared_evidence", "kg_hits"],
            "required_outputs": ["query_plan", "hit_routes", "risk_flags", "revision_advice", "secondary_optimization_trigger"],
        },
        "retrieval_policy": {
            "primary_sources": ["kg", "literature"],
            "query_focus": ["材料-特征-机理-性能路线", "风险机制", "替代材料"],
            "evidence_role": "用多跳机理解释当前方案，并决定是否反馈修改。",
        },
        "toolchain": ["多跳图检索", "混合检索", "因果链评分", "二次优化触发器"],
        "validators": ["每条建议必须区分支持、风险、替代或待验证", "图谱建议不得冒充模型已证实结果", "机理路线必须完整可读", "若建议改变变量，应说明是否需二次优化"],
        "reflection_loop": ["检查是否只保留正向证据", "检查是否遗漏反向风险", "检查文献与 KG 是否相互支持", "若路线证据薄弱则降低置信度"],
        "benchmark_cases": [
            {"case": "高流动性 + 高硅灰", "expect": "提示 PCE 适配和黏度风险"},
            {"case": "低碳 + 高粉煤灰", "expect": "同时给出流动性收益与早强风险"},
            {"case": "泵送混凝土", "expect": "检索泵送相关规范与黏聚性机制"},
        ],
    },
    "report_generation": {
        "name": "总报告智能体",
        "agent": "report",
        "description": "生成工程与科研可复核的正式报告。",
        "prompt": REPORT_SKILL_PROMPT,
        "io_contract": {
            "required_inputs": ["all_agent_outputs", "structured_result", "shared_evidence"],
            "required_outputs": ["formal_report", "inline_citations", "reference_list", "experiment_plan", "risk_register"],
        },
        "retrieval_policy": {
            "primary_sources": ["standard", "literature", "kg"],
            "query_focus": ["结论支撑", "规范评价", "实验建议", "风险说明", "证据交叉验证"],
            "evidence_role": "把计算、规范、机理和文献整合成可用于工程或科研复核的正式报告。",
        },
        "toolchain": ["报告规划器", "引用编排器", "表格生成器", "公式渲染器", "完整性审计器"],
        "validators": ["正文必须有行文内引用", "文献/规范/知识图谱引用格式必须区分", "章节必须完整", "所有结论必须区分预测、规范、机理和实验验证"],
        "reflection_loop": ["检查引用覆盖率", "检查是否缺少图表、实验矩阵或风险控制", "检查是否存在无依据结论", "检查报告是否可独立阅读"],
        "benchmark_cases": [
            {"case": "工程应用报告", "expect": "包含设计依据、方案、校核、试验与风险"},
            {"case": "科研指导报告", "expect": "包含机制解释、实验矩阵、失败判据与下一轮优化"},
            {"case": "含规范与文献证据", "expect": "规范、文献与知识图谱证据分别说明作用"},
        ],
    },
}


def _skill_meta_path(skill_id):
    return SKILLS_DIR / skill_id / "skill.json"


def _skill_prompt_path(skill_id):
    return SKILLS_DIR / skill_id / "prompt.md"


def _skill_prompt_path_for_language(skill_id, language="zh"):
    if language == "en":
        english_path = SKILLS_DIR / skill_id / "prompt.en.md"
        if english_path.exists():
            return english_path
    return _skill_prompt_path(skill_id)


def _ensure_default_skills():
    SKILL_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for skill_id, payload in DEFAULT_SKILLS.items():
        folder = SKILLS_DIR / skill_id
        folder.mkdir(parents=True, exist_ok=True)
        meta_path = _skill_meta_path(skill_id)
        prompt_path = _skill_prompt_path(skill_id)
        if not meta_path.exists():
            meta_path.write_text(
                json.dumps(
                    {
                        "id": skill_id,
                        "name": payload["name"],
                        "agent": payload["agent"],
                        "description": payload["description"],
                        "version": "1.0.0",
                        "architecture_version": "v2",
                        "io_contract": payload["io_contract"],
                        "retrieval_policy": payload["retrieval_policy"],
                        "toolchain": payload["toolchain"],
                        "validators": payload["validators"],
                        "reflection_loop": payload["reflection_loop"],
                        "benchmark_cases": payload["benchmark_cases"],
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        if not prompt_path.exists():
            prompt_path.write_text(payload["prompt"], encoding="utf-8")


def _read_skill(skill_id, language="zh"):
    meta_path = _skill_meta_path(skill_id)
    prompt_path = _skill_prompt_path_for_language(skill_id, language)
    if not meta_path.exists() or not prompt_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    defaults = DEFAULT_SKILLS.get(skill_id, {})
    for key in ["architecture_version", "io_contract", "retrieval_policy", "toolchain", "validators", "reflection_loop", "benchmark_cases"]:
        if key not in meta:
            if key == "architecture_version":
                meta[key] = "v2"
            elif key in defaults:
                meta[key] = defaults[key]
    if _normalize_language(language) == "en":
        meta = _apply_english_skill_overlay(skill_id, meta)
    meta["prompt"] = prompt_path.read_text(encoding="utf-8")
    asset_dir = SKILL_ASSETS_DIR / skill_id
    assets = []
    if asset_dir.exists():
        for item in sorted(asset_dir.iterdir()):
            if item.is_file():
                raw = item.read_bytes()
                assets.append({
                    "name": item.name,
                    "size": item.stat().st_size,
                    "suffix": item.suffix.lower(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                })
    meta["assets"] = assets
    return meta


def _normalize_language(value):
    return "en" if str(value or "").lower().startswith("en") else "zh"


EN_SKILL_LOCALIZATION = {
    "requirement_analysis": {
        "retrieval_policy": {
            "query_focus": ["Terminology definitions", "Performance thresholds", "Material semantics", "Engineering type"],
            "evidence_role": "Map natural language to computable specifications and identify whether specific standards are triggered.",
            "primary_sources": ["standard", "literature"],
        },
        "toolchain": ["Semantic parser", "Material-intent classifier", "Evidence retriever", "JSON schema validator"],
        "validators": [
            "Target strength and curing age must not be missing",
            "allowed / required / avoided material sets must not conflict",
            "User-provided numerical thresholds must be preserved as hard constraints",
            "Every key parsing item must be traceable to the original request",
        ],
        "reflection_loop": [
            "Check whether explicitly prohibited materials were missed",
            "Check whether allowed materials were incorrectly treated as required",
            "Check whether construction performance was reduced to an abstract preference",
            "Mark unresolved items when evidence is insufficient",
        ],
        "benchmark_cases": [
            {"case": "Allow fly ash and silica fume", "expect": "allowed_materials contains both; other binders are not automatically prohibited"},
            {"case": "I want to use fly ash and silica fume", "expect": "required_materials contains both"},
            {"case": "Control flowability within 550-650 mm", "expect": "Create a hard interval constraint for flowability"},
        ],
    },
    "optimization_modeling": {
        "retrieval_policy": {
            "primary_sources": ["standard", "literature", "kg"],
            "query_focus": ["Empirical formulae", "Variable bounds", "Objective functions", "Standards thresholds", "Minimum meaningful dosage"],
            "evidence_role": "Convert the requirement into a computable optimization problem and audit physical and engineering boundaries.",
        },
        "toolchain": ["Bolomey prior", "Absolute-volume method", "Multi-objective optimizer", "Calculator", "Constraint auditor"],
        "validators": [
            "Strength must satisfy the target lower bound",
            "Water-to-binder ratio must be constrained by the Bolomey prior window",
            "Prohibited materials must have zero lower and upper bounds",
            "Flowability intervals must enter constraints, not only narrative text",
            "Output dosages must not fall below meaningful engineering minima",
        ],
        "reflection_loop": [
            "Check whether objective functions match user priorities",
            "Check whether a Pareto extreme was incorrectly selected",
            "Check whether empirical checks conflict with ML prediction",
            "Trigger second-pass optimization when necessary",
        ],
        "benchmark_cases": [
            {"case": "80 MPa low-carbon concrete", "expect": "Constrain the water-to-binder ratio by a Bolomey window before optimization"},
            {"case": "No silica fume", "expect": "Set both lower and upper bounds of silicafume to 0"},
            {"case": "Flowability greater than 500 mm", "expect": "Create a flowability >= 500 constraint"},
        ],
    },
    "mechanism_retrieval": {
        "retrieval_policy": {
            "primary_sources": ["kg", "literature"],
            "query_focus": ["Material-feature-mechanism-performance routes", "Risk mechanisms", "Alternative materials"],
            "evidence_role": "Use multi-hop mechanisms to explain the current design and decide whether design feedback is needed.",
        },
        "toolchain": ["Multi-hop graph retrieval", "Hybrid retrieval", "Causal-chain scoring", "Second-pass optimization trigger"],
        "validators": [
            "Every recommendation must distinguish support, risk, substitution, or validation",
            "KG suggestions must not be presented as model-verified results",
            "Mechanistic routes must be complete and readable",
            "If a recommendation changes a variable, state whether second-pass optimization is required",
        ],
        "reflection_loop": [
            "Check whether only positive evidence was retained",
            "Check whether counter-risks were omitted",
            "Check whether literature and KG evidence support each other",
            "Lower confidence when route evidence is weak",
        ],
        "benchmark_cases": [
            {"case": "High flowability + high silica fume", "expect": "Flag PCE compatibility and viscosity risks"},
            {"case": "Low carbon + high fly ash", "expect": "Report both flowability benefits and early-strength risks"},
            {"case": "Pumped concrete", "expect": "Retrieve pumping-related standards and cohesion mechanisms"},
        ],
    },
    "report_generation": {
        "retrieval_policy": {
            "primary_sources": ["standard", "literature", "kg"],
            "query_focus": ["Conclusion support", "Standards evaluation", "Experimental recommendations", "Risk statement", "Evidence cross-validation"],
            "evidence_role": "Integrate computation, standards, mechanisms, and literature into a report suitable for engineering or scientific review.",
        },
        "toolchain": ["Report planner", "Citation orchestrator", "Table generator", "Formula renderer", "Completeness auditor"],
        "validators": [
            "The body must contain inline citations",
            "Literature, standards, and knowledge-graph citation roles must be distinguished",
            "Required sections must be complete",
            "All conclusions must distinguish prediction, standard, mechanism, and experimental validation",
        ],
        "reflection_loop": [
            "Check citation coverage",
            "Check whether figures, experimental matrix, or risk controls are missing",
            "Check for unsupported conclusions",
            "Check whether the report can be read independently",
        ],
        "benchmark_cases": [
            {"case": "Engineering application report", "expect": "Include design basis, recommendation, checks, testing, and risk"},
            {"case": "Research guidance report", "expect": "Include mechanism interpretation, experimental matrix, failure criteria, and next-round optimization"},
            {"case": "Evidence includes standards and literature", "expect": "Explain the roles of standards, literature, and KG evidence separately"},
        ],
    },
}


def _apply_english_skill_overlay(skill_id, meta):
    overlay = EN_SKILL_LOCALIZATION.get(skill_id)
    if not overlay:
        return meta
    localized = dict(meta)
    if localized.get("name_en"):
        localized["name"] = localized["name_en"]
    if localized.get("description_en"):
        localized["description"] = localized["description_en"]
    for key, value in overlay.items():
        localized[key] = value
    return localized


def _language_instruction(language):
    return (
        "\n\nLanguage requirement: write all user-facing analysis, section headings, figure/table descriptions, caveats, and final reports in academic English suitable for a Nature Communications submission. Keep technical variable names, JSON keys, units, and citation IDs unchanged."
        if language == "en"
        else "\n\n语言要求：所有面向用户的分析、标题、解释和最终报告均使用中文。技术变量名、JSON key、单位和引用编号保持不变。"
    )


def _load_skill_prompt(skill_id, fallback="", language="zh"):
    language = _normalize_language(language)
    skill = _read_skill(skill_id, language)
    base = skill["prompt"] if skill else fallback
    return f"{base}{_language_instruction(language)}"


_ensure_default_skills()

MATERIAL_TERMS = {
    "cement": ["cement", "水泥", "胶凝", "clinker", "熟料"],
    "flyash": ["fly ash", "flyash", "粉煤灰"],
    "GGBFS": ["ggbfs", "slag", "矿渣", "矿粉", "高炉矿渣"],
    "silicafume": ["silica fume", "硅灰"],
    "limestone": ["limestone", "石灰石", "石灰石粉"],
    "metakaolin": ["metakaolin", "偏高岭土"],
    "ricehuskash": ["rice husk", "rha", "稻壳灰"],
    "water": ["water", "水", "w/b", "水胶比"],
    "superplasticizer": ["superplasticizer", "pce", "减水剂", "外加剂"],
    "fineaggregate": ["fine aggregate", "sand", "细骨料", "砂"],
    "coarseaggregate": ["coarse aggregate", "gravel", "粗骨料", "石子"],
    "steelfiber": ["steel fiber", "钢纤维"],
}

KG_SNIPPETS = {
    "cement": "水泥/熟料提升早期 C-S-H 骨架，但高熟料含量通常带来更高碳排放和更快流动性损失。",
    "flyash": "粉煤灰的球形颗粒可改善流动性，后期火山灰反应生成二次 C-A-S-H/C-S-H，但高掺量可能降低早期强度。",
    "GGBFS": "矿渣粉在碱性环境下潜在水硬反应生成 C-A-S-H，常用于降低熟料用量并提升后期强度。",
    "silicafume": "硅灰具有微填充和高活性火山灰效应，可致密 ITZ，但比表面积高，通常需要减水剂配合。",
    "limestone": "适量石灰石粉可填充孔隙并提供成核表面，过量时会稀释反应性胶凝材料。",
    "metakaolin": "偏高岭土反应活性高，可形成 C-A-S-H 和碳铝酸盐，但片状细颗粒会提高需水量。",
    "ricehuskash": "优化燃烧和细磨的稻壳灰兼具微填充和火山灰效应，残碳或多孔结构会吸附水和外加剂。",
    "water": "低水胶比降低毛细孔连通性并提高强度，但过低会损害施工性和水化持续性。",
    "superplasticizer": "PCE 减水剂通过空间位阻分散颗粒，可在低水胶比下保持流动性并降低硬化孔隙率。",
    "fineaggregate": "良好级配细骨料降低需浆量和空隙率；过多细粉或黏土会吸附水和减水剂。",
    "coarseaggregate": "强、粗糙且级配合理的粗骨料提供承载骨架；吸水率高或级配差会降低有效拌合水和流动性。",
    "steelfiber": "钢纤维桥联收缩并提升韧性，但高体积分数会增加黏度和分散风险。",
}

KG_CHAIN_LIBRARY = {
    "cement": {
        "label": "水泥/熟料",
        "feature": "C3S/C2S 与熟料反应活性",
        "mechanism": "水化生成 C-S-H 骨架和 Ca(OH)2",
        "consequence": "早期强度提高，但熟料碳排放较高",
        "performance": "强度 / 碳排放",
    },
    "flyash": {
        "label": "粉煤灰",
        "feature": "球形玻璃质颗粒与铝硅酸盐玻璃相",
        "mechanism": "滚珠效应改善流动性，后期火山灰反应生成二次 C-A-S-H",
        "consequence": "降低需水量并替代部分熟料，早期强度可能下降",
        "performance": "流动性 / 低碳 / 后期强度",
    },
    "GGBFS": {
        "label": "矿渣粉",
        "feature": "钙铝硅酸盐玻璃相",
        "mechanism": "碱激发潜在水硬反应生成 C-A-S-H",
        "consequence": "细化孔结构并降低熟料用量",
        "performance": "低碳 / 后期强度 / 低收缩",
    },
    "silicafume": {
        "label": "硅灰",
        "feature": "高比表面积无定形 SiO2",
        "mechanism": "微填充与快速火山灰反应致密 ITZ",
        "consequence": "提高强度但增加外加剂和用水需求",
        "performance": "高强 / 黏度风险",
    },
    "water": {
        "label": "水胶比",
        "feature": "有效拌合水与毛细孔体积",
        "mechanism": "低水胶比减少连通毛细孔，但也可能引起自干燥和毛细张力",
        "consequence": "强度、流动性与自收缩形成竞争关系",
        "performance": "强度 / 流动性 / 自收缩",
    },
    "superplasticizer": {
        "label": "PCE 减水剂",
        "feature": "梳形聚羧酸侧链与吸附基团",
        "mechanism": "空间位阻分散胶凝颗粒",
        "consequence": "低水胶比下维持流动性并降低孔隙率",
        "performance": "流动性 / 高强",
    },
    "fineaggregate": {
        "label": "细骨料",
        "feature": "级配、形貌、含泥量与细粉含量",
        "mechanism": "改变砂浆内摩擦和外加剂吸附",
        "consequence": "影响泵送性、黏度和需浆量",
        "performance": "流动性 / 稳定性",
    },
    "limestone": {
        "label": "石灰石粉",
        "feature": "细方解石填料与成核表面",
        "mechanism": "填充孔隙并促进 C-S-H 异相成核",
        "consequence": "低掺量改善堆积，过量产生稀释效应",
        "performance": "低碳 / 强度风险",
    },
}

SHRINKAGE_KG_CHAINS = [
    {"material": "water", "feature": "低水胶比", "feature_en": "Low water-to-binder ratio", "mechanism": "自干燥增强并提高毛细张力", "mechanism_en": "Self-desiccation increases internal capillary tension", "consequence": "自收缩和早期收缩风险上升", "consequence_en": "Autogenous shrinkage and early shrinkage risk increase", "performance": "收缩", "performance_en": "shrinkage", "performance_label_en": "Shrinkage", "relation": "提高风险", "relation_en": "increases risk"},
    {"material": "cement", "feature": "高胶凝材料/高浆体体积", "feature_en": "High binder and paste volume", "mechanism": "可收缩浆体比例升高", "mechanism_en": "The deformable paste fraction increases", "consequence": "干燥收缩和体积变形增大", "consequence_en": "Drying shrinkage and volumetric deformation increase", "performance": "收缩", "performance_en": "shrinkage", "performance_label_en": "Shrinkage", "relation": "提高风险", "relation_en": "increases risk"},
    {"material": "coarseaggregate", "feature": "高骨料体积分数和刚性骨架", "feature_en": "High aggregate volume and rigid skeleton", "mechanism": "骨料约束浆体收缩变形", "mechanism_en": "Aggregates restrain shrinkage deformation of the paste", "consequence": "总收缩和收缩敏感性降低", "consequence_en": "Total shrinkage and shrinkage susceptibility decrease", "performance": "收缩", "performance_en": "shrinkage", "performance_label_en": "Shrinkage", "relation": "降低", "relation_en": "decreases"},
    {"material": "fineaggregate", "feature": "连续级配与较低需浆量", "feature_en": "Continuous grading and reduced paste demand", "mechanism": "降低浆骨比并减少可收缩相", "mechanism_en": "Paste-to-aggregate ratio and shrinkable phase volume are reduced", "consequence": "干燥收缩风险降低", "consequence_en": "Drying-shrinkage risk decreases", "performance": "收缩", "performance_en": "shrinkage", "performance_label_en": "Shrinkage", "relation": "降低", "relation_en": "decreases"},
    {"material": "silicafume", "feature": "超细高比表面积颗粒", "feature_en": "Ultrafine high-surface-area particles", "mechanism": "孔结构细化并加剧内部相对湿度下降", "mechanism_en": "Pore refinement accelerates internal relative-humidity reduction", "consequence": "自收缩风险升高，需要 PCE 和养护适配", "consequence_en": "Autogenous-shrinkage risk increases and requires PCE and curing adaptation", "performance": "收缩", "performance_en": "shrinkage", "performance_label_en": "Shrinkage", "relation": "提高风险", "relation_en": "increases risk"},
    {"material": "flyash", "feature": "较慢火山灰反应与球形颗粒", "feature_en": "Slower pozzolanic reaction and spherical particles", "mechanism": "降低早期水化热和初期需水，后期细化孔结构", "mechanism_en": "Early hydration heat and initial water demand decrease, followed by later pore refinement", "consequence": "早期收缩可缓和但后期收缩需实测确认", "consequence_en": "Early shrinkage may be mitigated, while later shrinkage requires testing", "performance": "收缩", "performance_en": "shrinkage", "performance_label_en": "Shrinkage", "relation": "条件性降低", "relation_en": "conditionally decreases"},
    {"material": "GGBFS", "feature": "潜在水硬反应和细化孔结构", "feature_en": "Latent hydraulic reaction and pore refinement", "mechanism": "后期水化持续并改变内部湿度演化", "mechanism_en": "Continued later hydration changes internal-humidity evolution", "consequence": "低碳收益与自收缩风险需剂量化平衡", "consequence_en": "Carbon benefit and autogenous-shrinkage risk require dosage balancing", "performance": "收缩", "performance_en": "shrinkage", "performance_label_en": "Shrinkage", "relation": "剂量相关", "relation_en": "dosage dependent"},
    {"material": "superplasticizer", "feature": "高效分散和降低拌合水需求", "feature_en": "Efficient dispersion and reduced mixing-water demand", "mechanism": "在较低水胶比下维持工作性", "mechanism_en": "Workability is retained at lower water-to-binder ratio", "consequence": "可能间接提高自收缩风险，需结合内养护或 SRA 验证", "consequence_en": "Autogenous-shrinkage risk may increase indirectly and should be validated with internal curing or SRA", "performance": "收缩", "performance_en": "shrinkage", "performance_label_en": "Shrinkage", "relation": "间接风险", "relation_en": "indirect risk"},
    {"material": "steelfiber", "feature": "纤维桥联与收缩钝化", "feature_en": "Fiber bridging and shrinkage blunting", "mechanism": "钢纤维跨越微收缩并提高收缩扩展阻力", "mechanism_en": "Steel fibers bridge microshrinkages and increase resistance to shrinkage propagation", "consequence": "约束收缩收缩宽度和收缩扩展降低，但自由收缩本身不一定降低", "consequence_en": "Restrained-shrinkage shrinkage strain and propagation decrease, although free shrinkage is not necessarily reduced", "performance": "收缩", "performance_en": "shrinkage", "performance_label_en": "Shrinkage", "relation": "降低收缩风险", "relation_en": "reduces shrinkage risk"},
    {"material": "limestone", "feature": "细填料、成核表面与胶凝材料稀释效应", "feature_en": "Fine filler, nucleation surface, and binder-dilution effect", "mechanism": "优化级配可降低需浆量，过量替代会改变水化和孔结构", "mechanism_en": "Optimized packing can reduce paste demand, whereas excessive replacement changes hydration and pore structure", "consequence": "收缩影响具有剂量依赖性，应按替代率与细度校核", "consequence_en": "Shrinkage response is dosage dependent and should be checked against replacement level and fineness", "performance": "收缩", "performance_en": "shrinkage", "performance_label_en": "Shrinkage", "relation": "剂量相关", "relation_en": "dosage dependent"},
    {"material": "metakaolin", "feature": "高活性铝硅酸盐和高比表面积", "feature_en": "Highly reactive aluminosilicate and high specific surface area", "mechanism": "快速火山灰反应和孔结构细化会提高内部湿度下降速率", "mechanism_en": "Rapid pozzolanic reaction and pore refinement can increase the rate of internal humidity reduction", "consequence": "自收缩和需水风险上升，需要减水剂和养护适配", "consequence_en": "Autogenous-shrinkage and water-demand risks increase, requiring admixture and curing adaptation", "performance": "收缩", "performance_en": "shrinkage", "performance_label_en": "Shrinkage", "relation": "提高风险", "relation_en": "increases risk"},
    {"material": "ricehuskash", "feature": "多孔高比表面积无定形硅质灰", "feature_en": "Porous high-surface-area amorphous siliceous ash", "mechanism": "火山灰反应和吸水特性共同改变有效水量与孔结构", "mechanism_en": "Pozzolanic reaction and water absorption jointly change effective water content and pore structure", "consequence": "未优化时需水和收缩风险上升；预湿或细度优化后可提供内养护边界", "consequence_en": "Water demand and shrinkage risk can increase when unoptimized; prewetting or fineness optimization may provide an internal-curing boundary", "performance": "收缩", "performance_en": "shrinkage", "performance_label_en": "Shrinkage", "relation": "条件性影响", "relation_en": "conditionally affects"},
]

CRACK_RESISTANCE_KG_CHAINS = [
]

SHRINKAGE_KG = [
    {"topic": "低水胶比与自收缩", "topic_en": "Low water-to-binder ratio and autogenous shrinkage", "mechanism": "低水胶比提高强度和致密性，但自干燥会提高毛细张力并诱发自收缩。", "mechanism_en": "A low water-to-binder ratio improves strength and densification, but self-desiccation increases capillary tension and autogenous shrinkage.", "recommendation": "低碳低收缩设计中，不应单纯压低水胶比；需同时控制浆体体积，并考虑内养护或收缩降低剂。", "recommendation_en": "Do not reduce the water-to-binder ratio alone in low-carbon low-shrinkage design; control paste volume and consider internal curing or shrinkage-reducing admixture."},
    {"topic": "浆体体积与干燥收缩", "topic_en": "Paste volume and drying shrinkage", "mechanism": "干燥收缩主要来自浆体相，骨料骨架对浆体变形具有约束作用。", "mechanism_en": "Drying shrinkage mainly originates from the paste phase, while the aggregate skeleton restrains paste deformation.", "recommendation": "将浆骨比或浆体体积作为低收缩约束，优先提高合理骨料体积分数和连续级配。", "recommendation_en": "Use paste-to-aggregate ratio or paste volume as a low-shrinkage constraint, and prioritize a reasonable aggregate volume fraction with continuous grading."},
    {"topic": "硅灰和超细粉体收缩风险", "topic_en": "Silica fume and ultrafine-powder shrinkage risk", "mechanism": "硅灰细化孔结构并提高内部湿度下降敏感性，高掺量体系可能增加自收缩和早期收缩风险。", "mechanism_en": "Silica fume refines pore structure and increases sensitivity to internal relative-humidity reduction; high dosages may increase autogenous shrinkage and early shrinkage risk.", "recommendation": "高强低碳低收缩体系中应限制硅灰上限，并用 3 d/7 d 自收缩或收缩试验验证。", "recommendation_en": "For high-strength low-carbon low-shrinkage mixtures, cap silica-fume dosage and validate 3 d/7 d autogenous shrinkage or restrained-ring shrinkage."},
    {"topic": "矿物掺合料的剂量化平衡", "topic_en": "Dosage balance of supplementary cementitious materials", "mechanism": "粉煤灰、矿渣等可降低熟料碳排并改变水化历程，但对早期强度和收缩的影响具有龄期和剂量依赖性。", "mechanism_en": "Fly ash and slag reduce clinker-related carbon and alter hydration history, but their effects on early strength and shrinkage are age- and dosage-dependent.", "recommendation": "低碳低收缩优化应同时报告碳排、强度、收缩龄期和 SCM 掺量，而不是只选择最低碳方案。", "recommendation_en": "Low-carbon low-shrinkage optimization should report carbon, strength, shrinkage age and SCM dosage together rather than selecting the lowest-carbon solution alone."},
]

KG_MATERIAL_LABEL_EN = {
    "cement": "Cement",
    "flyash": "Fly ash",
    "GGBFS": "GGBFS",
    "silicafume": "Silica fume",
    "limestone": "Limestone powder",
    "metakaolin": "Metakaolin",
    "ricehuskash": "Rice husk ash",
    "water": "Water",
    "superplasticizer": "Superplasticizer",
    "fineaggregate": "Fine aggregate",
    "coarseaggregate": "Coarse aggregate",
    "steelfiber": "Steel fiber",
}


def _load_recovered_kg_chains():
    if not RECOVERED_KG_APP.exists():
        return []
    text = RECOVERED_KG_APP.read_text(encoding="utf-8")
    pattern = re.compile(
        r'chain\("([^"]+)","([^"]+)","([^"]+)","([^"]+)","([^"]+)","([^"]+)","([^"]+)","([^"]+)","([^"]+)","([^"]+)"\)'
    )
    chains = []
    for match in pattern.finditer(text):
        (
            material,
            feature_en,
            feature_zh,
            mechanism_en,
            mechanism_zh,
            consequence_en,
            consequence_zh,
            performance,
            relation_en,
            relation_zh,
        ) = match.groups()
        chains.append(
            {
                "material": material,
                "feature": feature_zh,
                "mechanism": mechanism_zh,
                "consequence": consequence_zh,
                "performance": {
                    "compressive strength": "抗压强度",
                    "flowability": "流动性",
                    "shrinkage": "收缩",
                                        "low carbon": "低碳",
                }.get(performance, performance),
                "relation": relation_zh,
                "feature_en": feature_en,
                "mechanism_en": mechanism_en,
                "consequence_en": consequence_en,
                "performance_en": performance,
                "performance_label_en": {
                    "compressive strength": "Compressive strength",
                    "flowability": "Flowability",
                    "shrinkage": "Shrinkage",
                                        "low carbon": "Low carbon",
                }.get(performance, performance),
                "relation_en": relation_en,
            }
        )
    # The recovered KG source is now the audited manuscript-facing graph.
    # Do not append the older shrinkage/shrinkage extension arrays here; those
    # duplicate concepts and break the 344-route KG accounting.
    merged = chains
    deduped = []
    seen = set()
    for item in merged:
        key = (
            item.get("material"),
            item.get("feature_en") or item.get("feature"),
            item.get("mechanism_en") or item.get("mechanism"),
            item.get("consequence_en") or item.get("consequence"),
            item.get("performance_en") or item.get("performance"),
            item.get("relation_en") or item.get("relation"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    if RECOVERED_KG_EVIDENCE.exists():
        try:
            evidence_routes = json.loads(
                RECOVERED_KG_EVIDENCE.read_text(encoding="utf-8")
            )
            evidence_by_key = {
                (
                    route.get("material"),
                    route.get("feature"),
                    route.get("mechanism"),
                    route.get("consequence"),
                    route.get("performance"),
                    route.get("relation"),
                ): route
                for route in evidence_routes
            }
            for item in deduped:
                evidence = evidence_by_key.get(
                    (
                        item.get("material"),
                        item.get("feature_en"),
                        item.get("mechanism_en"),
                        item.get("consequence_en"),
                        item.get("performance_en"),
                        item.get("relation_en"),
                    )
                )
                if not evidence:
                    continue
                item["route_id"] = evidence.get("id")
                item["evidence_count"] = evidence.get("evidenceCount", 0)
                item["lexical_unique_evidence_count"] = evidence.get(
                    "lexicalUniqueEvidenceCount", 0
                )
                item["semantic_high_precision_count"] = evidence.get(
                    "semanticHighPrecisionCount", 0
                )
                item["expanded_unique_evidence_count"] = evidence.get(
                    "expandedUniqueEvidenceCount", 0
                )
                item["evidence_tiers"] = evidence.get("evidenceTiers", {})
                item["semantic_evidence_examples"] = evidence.get(
                    "semanticEvidenceExamples", []
                )
        except Exception:
            pass
    return deduped

FLOWABILITY_KG = [
    {
        "topic": "粉煤灰与粉煤灰微珠",
        "topic_en": "Fly ash and fly-ash microspheres",
        "mechanism": "球形玻璃质颗粒产生滚珠效应，降低颗粒间摩擦和屈服应力。",
        "mechanism_en": "Spherical glassy particles provide a ball-bearing effect that reduces inter-particle friction and yield stress.",
        "recommendation": "若模型给出较高粉煤灰掺量但流动性仍不足，优先评估以粉煤灰微珠/优质球形粉煤灰替代普通粉煤灰的试配方案。",
        "recommendation_en": "If the model selects a high fly-ash dosage but flowability remains insufficient, prioritize trial mixtures that replace ordinary fly ash with fly-ash microspheres or high-quality spherical fly ash.",
    },
    {
        "topic": "PCE 减水剂",
        "topic_en": "PCE superplasticizer",
        "mechanism": "梳形聚羧酸通过空间位阻分散水泥和矿物掺合料颗粒。",
        "mechanism_en": "Comb-shaped polycarboxylate ether disperses cement and supplementary cementitious material particles through steric hindrance.",
        "recommendation": "低水胶比或硅灰体系下，需要把 PCE 掺量作为关键调节变量，并验证保坍和凝结时间。",
        "recommendation_en": "For low water-to-binder-ratio or silica-fume systems, treat PCE dosage as a key control variable and verify slump retention and setting time.",
    },
    {
        "topic": "硅灰与超细粉体",
        "topic_en": "Silica fume and ultrafine powders",
        "mechanism": "超细颗粒提高比表面积和吸水/吸附需求，分散不足时塑性黏度升高。",
        "mechanism_en": "Ultrafine particles increase specific surface area and water/admixture demand; insufficient dispersion increases plastic viscosity.",
        "recommendation": "若高强需求导致硅灰偏高，应控制硅灰上限，配合 PCE 或部分使用流动性更友好的矿渣/粉煤灰。",
        "recommendation_en": "If high-strength requirements drive silica-fume dosage upward, cap the silica-fume content and combine it with PCE or partially replace it with more flowability-friendly slag or fly ash.",
    },
    {
        "topic": "骨料形貌与级配",
        "topic_en": "Aggregate morphology and grading",
        "mechanism": "圆润颗粒和连续级配降低内摩擦；棱角机制砂、过多细粉或黏土会吸附水和减水剂。",
        "mechanism_en": "Rounded particles and continuous grading reduce internal friction, whereas angular manufactured sand, excessive fines, or clay can adsorb water and superplasticizer.",
        "recommendation": "高流动性目标下应约束细粉/含泥量，并优先选择连续级配、较低吸水率骨料。",
        "recommendation_en": "For high-flowability targets, constrain fines and clay content and prioritize continuously graded aggregates with low water absorption.",
    },
]

CARBON_FACTORS = {
    "cement": 0.86,
    "flyash": 0.03,
    "GGBFS": 0.08,
    "silicafume": 0.02,
    "limestone": 0.01,
    "metakaolin": 0.18,
    "ricehuskash": 0.02,
    "water": 0.0003,
    "superplasticizer": 1.2,
    "fineaggregate": 0.005,
    "coarseaggregate": 0.004,
    "steelfiber": 1.7,
}

COST_FACTORS = {
    "cement": 0.52,
    "flyash": 0.14,
    "GGBFS": 0.24,
    "silicafume": 2.8,
    "limestone": 0.08,
    "metakaolin": 1.6,
    "ricehuskash": 0.22,
    "water": 0.003,
    "superplasticizer": 7.5,
    "fineaggregate": 0.06,
    "coarseaggregate": 0.055,
    "steelfiber": 6.8,
}

MATERIAL_DENSITIES = {
    "cement": 3150.0,
    "flyash": 2300.0,
    "GGBFS": 2900.0,
    "silicafume": 2200.0,
    "limestone": 2700.0,
    "metakaolin": 2600.0,
    "ricehuskash": 2100.0,
    "water": 1000.0,
    "superplasticizer": 1100.0,
    "fineaggregate": 2650.0,
    "coarseaggregate": 2700.0,
}

DEFAULT_BOUNDS = {
    "cement": (160, 560),
    "flyash": (0, 220),
    "GGBFS": (0, 320),
    "silicafume": (0, 70),
    "limestone": (0, 160),
    "metakaolin": (0, 100),
    "ricehuskash": (0, 90),
    "water": (110, 230),
    "superplasticizer": (0, 18),
    "fineaggregate": (520, 980),
    "coarseaggregate": (760, 1250),
    "steelfiber": (0, 3),
}

PRACTICAL_MINIMUMS = {
    "flyash": 40.0,
    "GGBFS": 40.0,
    "silicafume": 8.0,
    "limestone": 20.0,
    "metakaolin": 10.0,
    "ricehuskash": 10.0,
    "superplasticizer": 0.5,
    "steelfiber": 0.25,
}

MANDATORY_MINIMUMS = {
    "cement": 160.0,
    "water": 110.0,
    "fineaggregate": 520.0,
    "coarseaggregate": 760.0,
}

SUPPLEMENTARY_BINDERS = {
    "flyash",
    "GGBFS",
    "silicafume",
    "limestone",
    "metakaolin",
    "ricehuskash",
}

ZERO_THRESHOLDS = {
    "flyash": 1.0,
    "GGBFS": 1.0,
    "silicafume": 0.5,
    "limestone": 1.0,
    "metakaolin": 0.5,
    "ricehuskash": 0.5,
    "superplasticizer": 0.05,
    "steelfiber": 0.02,
}


def _json_error(message, status=400):
    return JSONResponse(status_code=status, content={"status": "error", "message": message})


def _safe_name(name, used):
    base = re.sub(r"\W+", "_", str(name), flags=re.UNICODE).strip("_").lower()
    if not base or re.match(r"^\d", base):
        base = "x"
    candidate = base
    idx = 2
    while candidate in used:
        candidate = f"{base}_{idx}"
        idx += 1
    used.add(candidate)
    return candidate


def _model_package():
    package = STATE.get("package")
    if not package:
        return None
    if package.get("package_type") == "natureml_multiobjective_optimization":
        return package.get("main_model_package")
    return package


def _surrogate_packages():
    package = STATE.get("package") or {}
    return package.get("surrogate_packages") or {}


def _find_flowability_alias():
    for alias, payload in _surrogate_packages().items():
        model_package = payload.get("model_package") or {}
        haystack = " ".join(
            str(item or "")
            for item in [
                alias,
                payload.get("target_col"),
                model_package.get("target_col"),
                model_package.get("model_name"),
            ]
        ).lower()
        if any(token in haystack for token in ["flow", "slump", "workability", "流动", "坍落"]):
            return alias
    return None


def _constraint_status(requirements, flow_alias):
    applied = []
    unresolved = []
    for item in requirements.get("performance_constraints", []):
        if item.get("metric") != "flowability":
            continue
        normalized = {
            "metric": "flowability",
            "lower": item.get("lower"),
            "upper": item.get("upper"),
            "unit": item.get("unit") or "model_unit",
            "source": item.get("source") or "",
        }
        if flow_alias:
            applied.append({**normalized, "model_field": flow_alias})
        else:
            unresolved.append({**normalized, "reason": "当前模型包未提供可调用的流动性预测输出"})
    for item in requirements.get("normative_requirements", []):
        if item.get("compile_status") != "optimizable":
            continue
        if item.get("metric") != "flowability":
            continue
        normalized = {
            "metric": "flowability",
            "standard_metric": item.get("standard_metric"),
            "lower": item.get("lower"),
            "upper": item.get("upper"),
            "value": item.get("value"),
            "unit": item.get("unit") or "model_unit",
            "source": item.get("source") or "",
            "page": item.get("page"),
            "evidence": item.get("evidence"),
            "domain": item.get("domain"),
            "standard_operator": item.get("operator"),
        }
        if flow_alias:
            applied.append({**normalized, "model_field": flow_alias, "source_type": "standard"})
        else:
            unresolved.append({**normalized, "reason": "规范命中流动性约束，但当前模型包未提供可调用的流动性预测输出", "source_type": "standard"})
    return {"applied": applied, "unresolved": unresolved}


def _find_target_mpa(text):
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mpa|MPa|兆帕|强度)", text or "")
    if match:
        return float(match.group(1))
    match = re.search(r"设计.*?(\d+(?:\.\d+)?)", text or "")
    return float(match.group(1)) if match else None


def _extract_target_mpa(text):
    return _find_target_mpa(text) or 60.0


def _extract_age_days(text):
    raw = text or ""
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:天|d|day|days|龄期)",
        r"(?:龄期|养护).*?(\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return 28.0


def _extract_flowability_constraints(text):
    raw = text or ""
    constraints = []
    patterns = [
        r"(?:流动性|流动度|坍落扩展度|扩展度)\D{0,8}?(\d+(?:\.\d+)?)\s*(?:-|~|到|至)\s*(\d+(?:\.\d+)?)\s*(mm|毫米)?",
        r"(?:流动性|流动度|坍落扩展度|扩展度)\D{0,8}?(?:大于等于|不小于|不低于|大于|高于|超过|至少|>)\s*(\d+(?:\.\d+)?)\s*(mm|毫米)?",
        r"(?:流动性|流动度|坍落扩展度|扩展度)\D{0,8}?(?:小于等于|不大于|不高于|小于|低于|少于|至多|<)\s*(\d+(?:\.\d+)?)\s*(mm|毫米)?",
    ]
    match = re.search(patterns[0], raw, flags=re.IGNORECASE)
    if match:
        constraints.append({"metric": "flowability", "lower": float(match.group(1)), "upper": float(match.group(2)), "unit": match.group(3) or "model_unit", "source": match.group(0)})
        return constraints
    match = re.search(patterns[1], raw, flags=re.IGNORECASE)
    if match and not any(token in match.group(0) for token in ["不大于", "不高于"]):
        constraints.append({"metric": "flowability", "lower": float(match.group(1)), "upper": None, "unit": match.group(2) or "model_unit", "source": match.group(0)})
    match = re.search(patterns[2], raw, flags=re.IGNORECASE)
    if match and not any(token in match.group(0) for token in ["不小于", "不低于"]):
        constraints.append({"metric": "flowability", "lower": None, "upper": float(match.group(1)), "unit": match.group(2) or "model_unit", "source": match.group(0)})
    return constraints


def _extract_shrinkage_constraints(text):
    raw = text or ""
    constraints = []
    metric_tokens = r"(?:收缩|干燥收缩|自收缩|总收缩|shrinkage|drying shrinkage|autogenous shrinkage|total shrinkage|收缩风险|低收缩)"
    shrinkage_type = "drying" if re.search(r"干燥收缩|drying shrinkage", raw, flags=re.IGNORECASE) else (
        "autogenous" if re.search(r"自收缩|autogenous shrinkage", raw, flags=re.IGNORECASE) else (
            "shrinkage_risk" if re.search(r"收缩|低收缩|shrinkage", raw, flags=re.IGNORECASE) else "total"
        )
    )
    age_match = re.search(r"(?:收缩|shrinkage)[^。；,\n]{0,24}?(\d+(?:\.\d+)?)\s*(?:天|d|day|days)", raw, flags=re.IGNORECASE)
    age_days = float(age_match.group(1)) if age_match else None
    range_match = re.search(metric_tokens + r"\D{0,12}?(\d+(?:\.\d+)?)\s*(?:-|~|到|至)\s*(\d+(?:\.\d+)?)\s*(?:με|μstrain|microstrain|微应变)?", raw, flags=re.IGNORECASE)
    if range_match:
        lower, upper = float(range_match.group(1)), float(range_match.group(2))
        if lower > upper:
            lower, upper = upper, lower
        constraints.append({"metric": "shrinkage", "lower": lower, "upper": upper, "unit": "microstrain", "source": range_match.group(0), "shrinkage_type": shrinkage_type, "age_days": age_days})
        return constraints
    upper_match = re.search(metric_tokens + r"\D{0,12}?(?:小于等于|不大于|不高于|小于|低于|少于|至多|控制在|<)\s*(\d+(?:\.\d+)?)\s*(?:με|μstrain|microstrain|微应变)?", raw, flags=re.IGNORECASE)
    if upper_match:
        constraints.append({"metric": "shrinkage", "lower": None, "upper": float(upper_match.group(1)), "unit": "microstrain", "source": upper_match.group(0), "shrinkage_type": shrinkage_type, "age_days": age_days})
    elif re.search(r"低收缩|控制收缩|降低收缩|低收缩|低收缩|low shrinkage|shrinkage control|shrinkage", raw, flags=re.IGNORECASE):
        constraints.append({"metric": "shrinkage", "lower": None, "upper": None, "unit": "qualitative", "source": "low-shrinkage requirement", "shrinkage_type": shrinkage_type, "age_days": age_days, "target": "low"})
    return constraints


def _extract_performance_constraints(text):
    return [*_extract_flowability_constraints(text), *_extract_shrinkage_constraints(text)]


def _material_intents(raw):
    mentioned = []
    required = []
    allowed = []
    avoided = []
    lower_raw = (raw or "").lower()
    avoid_markers = ["不使用", "不用", "不要", "禁用", "禁止", "不得使用", "without ", "no "]
    allow_markers = ["允许", "允许使用", "可使用", "可以使用", "可以用", "可选", "可考虑", "allow ", "may use"]
    require_markers = ["要求使用", "必须使用", "必须加", "需要加入", "加入", "掺入", "采用", "使用", "想用", "希望用", "希望使用", "配方里有", "体系", "含有", "require ", "must use"]
    pieces = [item.strip() for item in re.split(r"[，,。；;]\s*", raw or "") if item.strip()]
    for material, terms in MATERIAL_TERMS.items():
        term_hits = [term for term in terms if term.lower() in lower_raw]
        if term_hits:
            mentioned.append(material)
        for piece in pieces:
            lower_piece = piece.lower()
            if not any(term.lower() in lower_piece for term in terms):
                continue
            if any(token in lower_piece for token in avoid_markers):
                avoided.append(material)
            elif any(token in lower_piece for token in allow_markers):
                allowed.append(material)
            elif any(token in lower_piece for token in require_markers) or ("要求" in lower_piece and material in SUPPLEMENTARY_BINDERS):
                required.append(material)
        for term in term_hits:
            idx = lower_raw.find(term.lower())
            if idx < 0:
                continue
            avoid_window = lower_raw[max(0, idx - 14):idx + len(term)]
            intent_window = lower_raw[max(0, idx - 40):idx + len(term) + 40]
            if any(token in avoid_window for token in avoid_markers):
                avoided.append(material)
            elif any(token in intent_window for token in allow_markers):
                allowed.append(material)
            elif any(token in intent_window for token in require_markers):
                required.append(material)
    return {
        "mentioned_materials": list(dict.fromkeys(mentioned)),
        "required_materials": list(dict.fromkeys(required)),
        "allowed_materials": list(dict.fromkeys(allowed)),
        "avoided_materials": list(dict.fromkeys(avoided)),
    }


def _analyze_user_request(text, explicit_target=None):
    raw = text or ""
    lower = raw.lower()
    target = float(_find_target_mpa(raw) or explicit_target or 60.0)
    age_days = _extract_age_days(raw)
    material_intents = _material_intents(raw)
    priorities = []
    if any(token in lower for token in ["低碳", "low carbon", "低熟料", "减碳"]):
        priorities.append("low_carbon")
    if any(token in lower for token in ["高流动", "流动性", "泵送", "自密实", "high flow", "workability", "flowability"]):
        priorities.append("high_flowability")
    if any(token in lower for token in ["低收缩", "收缩", "干燥收缩", "自收缩", "low shrinkage", "shrinkage"]):
        priorities.append("low_shrinkage")
    if any(token in lower for token in ["低收缩", "收缩", "收缩", "shrinkage", "shrinkage", "shrinkage strain", "shrinkage control"]):
        priorities.append("low_shrinkage")
    if any(token in lower for token in ["早强", "early strength", "7d"]):
        priorities.append("early_strength")
    if not priorities:
        priorities = ["low_carbon", "strength"]

    priority_labels = {
        "low_carbon": "降低胶凝材料碳排放和熟料用量",
        "high_flowability": "提高施工流动性/泵送性并控制黏度",
        "low_shrinkage": "降低干燥收缩、自收缩和收缩风险",
        "low_shrinkage": "提高低收缩能力并控制收缩变形",
        "early_strength": "提高早期强度发展速度",
        "strength": "满足目标抗压强度",
    }
    deep_needs = []
    for priority in priorities:
        deep_needs.append({
            "priority": priority,
            "interpretation": priority_labels.get(priority, priority),
            "reason": "由用户显式关键词触发" if priority != "strength" else "强度目标是配合比设计硬约束",
        })

    return {
        "target_mpa": target,
        "age_days": age_days,
        "priorities": list(dict.fromkeys(priorities)),
        "priority_ranking": deep_needs,
        **material_intents,
        "performance_constraints": _extract_performance_constraints(raw),
        "material_mode": "specified" if any(item in SUPPLEMENTARY_BINDERS for item in material_intents["required_materials"]) else "exploratory",
        "raw_segments": _segment_user_request(raw),
        "raw_request": raw,
    }


def _segment_user_request(text):
    raw = text or ""
    pieces = [item.strip(" ，,。；;") for item in re.split(r"[，,。；;]\s*", raw) if item.strip(" ，,。；;")]
    if not pieces and raw:
        pieces = [raw]
    segments = []
    for piece in pieces:
        lower = piece.lower()
        shrinkage_hit = any(token in lower for token in ["低收缩", "收缩", "干燥收缩", "自收缩", "低收缩", "收缩", "low shrinkage", "shrinkage", "shrinkage", "shrinkage"])
        if re.search(r"\d+(?:\.\d+)?\s*(?:mpa|兆帕)", lower) and shrinkage_hit:
            role = "低碳低收缩综合目标"
            interpretation = "同时转化为强度约束、低碳目标和收缩机理检索条件"
        elif shrinkage_hit:
            role = "收缩约束" if _extract_shrinkage_constraints(piece) else "低收缩偏好"
            interpretation = "转化为收缩上限或低收缩定性约束，并触发收缩机理检索"
        elif re.search(r"\d+(?:\.\d+)?\s*(?:mpa|兆帕)", lower):
            role = "强度指标"
            interpretation = "转化为目标抗压强度和强度约束"
        elif re.search(r"\d+(?:\.\d+)?\s*(?:天|d|day|days)", lower):
            role = "龄期输入"
            interpretation = "固定 age/龄期预测变量"
        elif any(token in lower for token in ["低碳", "减碳", "少用水泥", "低熟料", "low carbon"]):
            role = "低碳偏好"
            interpretation = "提高 Carbon 最小化目标优先级"
        elif any(token in lower for token in ["高流动", "流动性", "流动度", "坍落扩展度", "泵送", "自密实", "flowability", "workability"]):
            role = "施工性能约束" if _extract_flowability_constraints(piece) else "施工性能偏好"
            interpretation = "转化为流动性数值区间约束" if _extract_flowability_constraints(piece) else "引入真实流动性目标并检索流动性机理"
        elif any(term.lower() in lower for terms in MATERIAL_TERMS.values() for term in terms):
            role = "材料条件"
            interpretation = "作为变量边界、知识图谱检索权重或材料偏好"
        else:
            role = "上下文补充"
            interpretation = "用于最终回答的工程背景"
        segments.append({"text": piece, "role": role, "interpretation": interpretation})
    return segments


def _classify_feature(name):
    text = str(name).lower().replace("_", " ")
    compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)
    for material, terms in MATERIAL_TERMS.items():
        if any(
            term.lower() in text or re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", term.lower()) in compact
            for term in terms
        ):
            return material
    if "age" in text or "龄期" in text:
        return "age"
    return None


def _make_feature_map(feature_columns):
    used = set()
    mapping = {}
    for original in feature_columns:
        safe = _safe_name(original, used)
        material = _classify_feature(original)
        lo, hi = (1, 365) if material == "age" else DEFAULT_BOUNDS.get(material, (0, 1))
        unit = "d" if material == "age" else "kg/m3"
        original_lower = str(original).lower()
        if material == "steelfiber":
            unit = "vol%"
            lo, hi = DEFAULT_BOUNDS["steelfiber"]
        elif any(token in original_lower for token in ["ratio", "fraction", "体积分数", "掺量比例"]):
            unit = "%"
        mapping[safe] = {
            "safe": safe,
            "original": original,
            "material": material,
            "lower": lo,
            "upper": hi,
            "unit": unit,
        }
    return mapping


def _bounds_from_payload(payload, feature_map):
    custom = payload.get("variables") or []
    if isinstance(custom, str):
        try:
            custom = json.loads(custom)
        except json.JSONDecodeError:
            custom = []
    by_name = {str(item.get("name")): item for item in custom if isinstance(item, dict)}
    bounds = []
    for safe, meta in feature_map.items():
        item = by_name.get(safe) or by_name.get(str(meta["original"]))
        lower = float(item.get("lower", meta["lower"])) if item else float(meta["lower"])
        upper = float(item.get("upper", meta["upper"])) if item else float(meta["upper"])
        if upper <= lower:
            upper = lower + 1.0
        bounds.append({"name": safe, "lower": lower, "upper": upper})
    return bounds


def _carbon_expression(feature_map):
    terms = []
    for safe, meta in feature_map.items():
        material = meta.get("material")
        if material in CARBON_FACTORS and meta.get("unit") != "vol%":
            terms.append(f"{CARBON_FACTORS[material]} * {safe}")
    return " + ".join(terms) if terms else "0"


def _cost_expression(feature_map):
    terms = []
    for safe, meta in feature_map.items():
        material = meta.get("material")
        if material in COST_FACTORS and meta.get("unit") != "vol%":
            terms.append(f"{COST_FACTORS[material]} * {safe}")
    return " + ".join(terms) if terms else "0"


def _kg_hits(prompt, feature_map, best_solution=None):
    text = (prompt or "").lower()
    scored = {}
    if any(token in text for token in ["高流动", "流动性", "泵送", "自密实", "workability", "flowability", "high flow"]):
        for material in ["flyash", "superplasticizer", "water", "fineaggregate", "GGBFS"]:
            scored[material] = scored.get(material, 0) + 5
    if any(token in text for token in ["低碳", "low carbon", "减碳", "低熟料"]):
        for material in ["cement", "GGBFS", "flyash", "limestone", "silicafume", "metakaolin", "ricehuskash", "superplasticizer"]:
            scored[material] = scored.get(material, 0) + 4
    if any(token in text for token in ["低收缩", "收缩", "干燥收缩", "自收缩", "低收缩", "收缩", "low shrinkage", "shrinkage", "shrinkage", "shrinkage"]):
        for material in ["water", "cement", "coarseaggregate", "fineaggregate", "silicafume", "flyash", "GGBFS", "superplasticizer", "steelfiber", "limestone", "metakaolin", "ricehuskash"]:
            scored[material] = scored.get(material, 0) + 5
    if any(token in text for token in ["高强", "强度", "mpa", "抗压"]):
        for material in ["cement", "silicafume", "GGBFS", "water", "superplasticizer"]:
            scored[material] = scored.get(material, 0) + 3
    for material, terms in MATERIAL_TERMS.items():
        if any(term.lower() in text for term in terms):
            scored[material] = scored.get(material, 0) + 4
    for meta in feature_map.values():
        material = meta.get("material")
        if material:
            scored[material] = scored.get(material, 0) + 1
    if best_solution:
        for safe, value in best_solution.items():
            if safe in feature_map and isinstance(value, (int, float)):
                material = feature_map[safe].get("material")
                if material and value > feature_map[safe]["lower"] + 0.45 * (feature_map[safe]["upper"] - feature_map[safe]["lower"]):
                    scored[material] = scored.get(material, 0) + 2
    if not scored:
        for material in ["water", "cement", "GGBFS", "flyash", "superplasticizer"]:
            scored[material] = 1
    hits = sorted(scored, key=scored.get, reverse=True)[:4]
    recovered = _load_recovered_kg_chains()
    query_terms = _kg_query_terms({"raw_request": prompt or "", "priorities": []})
    results = []
    for key in hits:
        chains = [item for item in recovered if item.get("material") == key]
        if chains:
            def chain_score(item):
                haystack = " ".join([
                    item.get("feature_en", ""),
                    item.get("mechanism_en", ""),
                    item.get("consequence_en", ""),
                    item.get("performance_en", ""),
                    item.get("feature", ""),
                    item.get("mechanism", ""),
                    item.get("consequence", ""),
                    item.get("performance", ""),
                ]).lower()
                return sum(1 for term in query_terms if term in haystack)
            selected = sorted(chains, key=chain_score, reverse=True)[:2]
            evidence_en = "; ".join(
                f"{item.get('feature_en')} -> {item.get('mechanism_en')} -> {item.get('performance_label_en') or item.get('performance_en')} ({item.get('relation_en')})"
                for item in selected
            )
            evidence_zh = "；".join(
                f"{item.get('feature')} -> {item.get('mechanism')} -> {item.get('performance')}（{item.get('relation')}）"
                for item in selected
            )
            results.append({"material": key, "evidence": evidence_zh, "evidence_en": evidence_en, "chain_count": len(chains)})
        elif key in KG_SNIPPETS:
            results.append({"material": key, "evidence": KG_SNIPPETS[key], "chain_count": 0})
    return results


def _kg_query_terms(requirements):
    terms = set(requirements.get("mentioned_materials", []))
    raw = requirements.get("raw_request", "").lower()
    terms.update([
        "收缩", "干燥收缩", "自收缩", "低收缩", "收缩", "shrinkage", "drying shrinkage", "autogenous shrinkage", "shrinkage",
        "强度", "抗压强度", "compressive strength", "strength",
        "流动性", "工作性", "坍落度", "扩展度", "flowability", "workability", "slump",
    ])
    if "high_flowability" in requirements.get("priorities", []):
        terms.update(["流动", "屈服应力", "黏度", "泵送", "分散", "摩擦", "滚珠", "flowability", "workability", "yield stress", "plastic viscosity", "superplasticizer"])
    if "low_shrinkage" in requirements.get("priorities", []):
        terms.update(["低收缩", "收缩", "干燥收缩", "自收缩", "低收缩", "收缩", "浆体体积", "浆骨比", "骨料约束", "内养护", "shrinkage", "drying shrinkage", "autogenous shrinkage", "shrinkage", "paste volume", "aggregate restraint", "internal curing", "SRA"])
    if any(token in raw for token in ["高强", "强度", "mpa", "抗压"]):
        terms.update(["强度", "水化", "c-s-h", "孔", "致密", "compressive strength", "hydration", "microstructure"])
    if "low_carbon" in requirements.get("priorities", []):
        terms.update(["熟料", "替代", "火山灰", "矿渣", "粉煤灰", "碳排", "低碳", "low carbon", "carbon", "embodied carbon", "clinker replacement", "supplementary cementitious materials", "fly ash", "slag", "silica fume"])
    return {term.lower() for term in terms if term}


def _tokenize_evidence(text):
    text = str(text or "").lower()
    words = re.findall(r"[a-z][a-z0-9\-]{1,}|[\u4e00-\u9fff]{2,}", text)
    grams = []
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        grams.extend(chunk[i:i + 2] for i in range(max(0, len(chunk) - 1)))
    return words + grams


def _chunk_text(text, size=900, overlap=120):
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + size)
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_cell(value):
    value = str(value or "").replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", value).strip()


def _normalize_abstract_doc(item, index=0):
    title = _clean_cell(item.get("title") or item.get("TI") or item.get("Title") or "")
    abstract = _clean_cell(item.get("abstract") or item.get("AB") or item.get("Abstract") or "")
    journal = _clean_cell(item.get("journal") or item.get("SO") or item.get("Source") or "")
    year = _clean_cell(item.get("year") or item.get("PY") or item.get("Year") or "")
    doi = _clean_cell(item.get("doi") or item.get("DI") or item.get("DOI") or "")
    ut = _clean_cell(item.get("ut") or item.get("UT") or item.get("Accession Number") or "")
    authors = item.get("authors") or item.get("AU") or []
    if isinstance(authors, str):
        authors = [part.strip() for part in re.split(r";|,", authors) if part.strip()]
    text = "\n".join(part for part in [title, journal, abstract] if part)
    doc_id = _stable_evidence_id("abstract_doc", ut or doi or index, title[:120], abstract[:120])
    return {
        "id": doc_id,
        "title": title or f"Abstract {index + 1}",
        "abstract": abstract,
        "journal": journal,
        "year": year,
        "doi": doi,
        "ut": ut,
        "authors": authors[:24] if isinstance(authors, list) else [],
        "text": text,
        "tokens": _tokenize_evidence(text),
    }


def _parse_table_abstracts(text, filename=""):
    sample = text[:12000]
    delimiter = "\t" if sample.count("\t") >= sample.count(",") else ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        delimiter = dialect.delimiter
    except Exception:
        pass
    rows = []
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        return []
    fieldnames = {name.strip() for name in reader.fieldnames if name}
    if not ({"AB", "TI"} & fieldnames or {"Abstract", "Title"} & fieldnames):
        return []
    for index, row in enumerate(reader):
        doc = _normalize_abstract_doc(row, index)
        if doc["title"] or doc["abstract"]:
            rows.append(doc)
    return rows


def _parse_abstract_corpus(raw_text, filename=""):
    text = raw_text.lstrip("\ufeff").replace("\r\n", "\n")
    if _looks_like_wos_export(text):
        return [_normalize_abstract_doc(item, index) for index, item in enumerate(_parse_wos_records(text))]
    table_docs = _parse_table_abstracts(text, filename)
    if table_docs:
        return table_docs
    docs = []
    chunks = re.split(r"\n\s*\n+", text)
    for index, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if len(chunk) < 80:
            continue
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        title = lines[0][:260] if lines else f"Abstract {index + 1}"
        abstract = " ".join(lines[1:]) if len(lines) > 1 else chunk
        docs.append(_normalize_abstract_doc({"title": title, "abstract": abstract}, index))
    return docs


def _write_abstract_docs(docs, source_filename="", raw_size=0):
    ABSTRACT_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with ABSTRACT_CORPUS_DOCS_PATH.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    years = [_safe_int(doc.get("year")) for doc in docs if _safe_int(doc.get("year"))]
    meta = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "source_filename": source_filename,
        "raw_size": raw_size,
        "doc_count": len(docs),
        "with_abstract": sum(1 for doc in docs if doc.get("abstract")),
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
    }
    ABSTRACT_CORPUS_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _load_abstract_docs(limit=None):
    docs = []
    if not ABSTRACT_CORPUS_DOCS_PATH.exists():
        return docs
    with ABSTRACT_CORPUS_DOCS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            docs.append(json.loads(line))
            if limit and len(docs) >= limit:
                break
    return docs


def _abstract_corpus_status():
    meta = {}
    if ABSTRACT_CORPUS_META_PATH.exists():
        try:
            meta = json.loads(ABSTRACT_CORPUS_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    calibration = {}
    if ABSTRACT_CALIBRATION_PATH.exists():
        try:
            payload = json.loads(ABSTRACT_CALIBRATION_PATH.read_text(encoding="utf-8"))
            calibration = {
                "updated_at": payload.get("updated_at"),
                "kg_chain_count": payload.get("summary", {}).get("kg_chain_count", 0),
                "calibrated_count": payload.get("summary", {}).get("calibrated_count", 0),
                "high_confidence_count": payload.get("summary", {}).get("high_confidence_count", 0),
            }
        except Exception:
            calibration = {}
    candidates = {}
    if ABSTRACT_CANDIDATES_PATH.exists():
        try:
            payload = json.loads(ABSTRACT_CANDIDATES_PATH.read_text(encoding="utf-8"))
            candidates = {"updated_at": payload.get("updated_at"), "count": len(payload.get("candidates", []))}
        except Exception:
            candidates = {}
    shrinkage_evidence = {}
    if ABSTRACT_SHRINKAGE_EVIDENCE_PATH.exists():
        try:
            payload = json.loads(ABSTRACT_SHRINKAGE_EVIDENCE_PATH.read_text(encoding="utf-8"))
            shrinkage_evidence = {
                "updated_at": payload.get("updated_at"),
                "shrinkage_doc_count": payload.get("summary", {}).get("shrinkage_doc_count", 0),
                "material_count": payload.get("summary", {}).get("material_count", 0),
            }
        except Exception:
            shrinkage_evidence = {}
    return {
        "corpus": meta,
        "has_docs": ABSTRACT_CORPUS_DOCS_PATH.exists(),
        "calibration": calibration,
        "candidate_sandbox": candidates,
        "shrinkage_evidence": shrinkage_evidence,
        "embedding": _load_embedding_config(redact=False),
        "lora": _lora_status(),
    }


def _load_embedding_config(redact=False):
    config = dict(DEFAULT_EMBEDDING_CONFIG)
    if EMBEDDING_CONFIG_PATH.exists():
        try:
            saved = json.loads(EMBEDDING_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                config.update({key: saved[key] for key in config.keys() if key in saved})
        except Exception:
            pass
    config["batch_size"] = max(1, min(_safe_int(config.get("batch_size"), 32), 512))
    config["timeout_seconds"] = max(10, min(_safe_int(config.get("timeout_seconds"), 120), 900))
    if redact and config.get("api_key"):
        config["api_key"] = "***"
    return config


def _save_embedding_config(payload):
    config = _load_embedding_config()
    for key in ["api_url", "model", "cache_dir"]:
        if key in payload:
            value = str(payload.get(key) or "").strip()
            if value:
                config[key] = value
    for key in ["batch_size", "timeout_seconds"]:
        if key in payload:
            config[key] = _safe_int(payload.get(key), config[key])
    config["batch_size"] = max(1, min(_safe_int(config.get("batch_size"), 32), 512))
    config["timeout_seconds"] = max(10, min(_safe_int(config.get("timeout_seconds"), 120), 900))
    ABSTRACT_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def _embedding_request(texts, config=None):
    config = config or _load_embedding_config()
    payload = {
        "model": config.get("model"),
        "input": texts,
    }
    req = urllib.request.Request(
        str(config.get("api_url")),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_safe_int(config.get("timeout_seconds"), 120)) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    vectors = [item.get("embedding", []) for item in data.get("data", [])]
    return vectors


def _test_embedding_config(config=None):
    config = config or _load_embedding_config()
    texts = [
        "fly ash improves concrete workability through spherical particles",
        "粉煤灰通过球形颗粒改善混凝土流动性",
    ]
    vectors = _embedding_request(texts, config)
    dims = [len(vec) for vec in vectors]
    if not vectors or not all(dims):
        raise RuntimeError("Embedding API 未返回有效向量。")
    return {
        "model": config.get("model"),
        "api_url": config.get("api_url"),
        "vector_count": len(vectors),
        "dimension": dims[0],
        "dimension_consistent": len(set(dims)) == 1,
    }


def _kg_calibration_query(record):
    route = record.get("route") or []
    material = record.get("material") or ""
    performance = record.get("performance") or ""
    text = " ".join([record.get("text", ""), " ".join(route), material, performance])
    terms = set(_tokenize_evidence(text))
    for term in MATERIAL_TERMS.get(material, []):
        terms.update(_tokenize_evidence(term))
    for term in PERFORMANCE_TERMS.get(performance, []):
        terms.update(_tokenize_evidence(term))
    terms = {term for term in terms if len(term) > 1 and term not in {"and", "with", "the", "for", "from", "this", "that"}}
    priority = set()
    for part in [material, performance, *route]:
        priority.update(_tokenize_evidence(part))
    return terms, priority


SUPPORT_HINTS = {
    "increase": ["increase", "improve", "enhance", "higher", "promote", "benefit", "提高", "改善", "增强", "提升", "促进"],
    "decrease": ["decrease", "reduce", "lower", "decline", "weaken", "deteriorate", "降低", "减少", "下降", "削弱", "劣化"],
    "risk": ["risk", "negative", "adverse", "loss", "shrinkage", "shrinkage", "风险", "不利", "损失", "收缩", "收缩"],
}


def _relation_polarity(relation):
    text = str(relation or "").lower()
    if "降低" in text or "decrease" in text or "risk" in text or "负" in text:
        return "decrease"
    if "提高" in text or "increase" in text or "support" in text or "正" in text:
        return "increase"
    return "support"


def _doc_relation_signal(doc_text, polarity):
    lower = str(doc_text or "").lower()
    support_terms = SUPPORT_HINTS.get(polarity, []) + SUPPORT_HINTS["increase"]
    oppose_terms = SUPPORT_HINTS["decrease"] if polarity == "increase" else SUPPORT_HINTS["increase"]
    support = sum(1 for term in support_terms if term.lower() in lower)
    oppose = sum(1 for term in oppose_terms if term.lower() in lower)
    risk = sum(1 for term in SUPPORT_HINTS["risk"] if term.lower() in lower)
    return support, oppose, risk


def _score_doc_for_kg(doc, query_terms, priority_terms, polarity):
    token_set = doc.get("_token_set")
    if token_set is None:
        token_set = set(doc.get("tokens", []))
    overlap = query_terms & token_set
    if not overlap:
        return 0.0, {}
    priority_hits = priority_terms & overlap
    support, oppose, risk = _doc_relation_signal(doc.get("text", ""), polarity)
    title_tokens = doc.get("_title_tokens")
    if title_tokens is None:
        title_tokens = set(_tokenize_evidence(doc.get("title", "")))
    title_hit = len(title_tokens & query_terms)
    score = len(overlap) + 1.8 * len(priority_hits) + 1.4 * title_hit + 0.8 * support - 0.45 * oppose + 0.25 * risk
    return round(score, 4), {
        "overlap": sorted(overlap)[:24],
        "priority_hits": sorted(priority_hits)[:12],
        "support_signal": support,
        "opposing_signal": oppose,
        "risk_signal": risk,
        "title_hits": title_hit,
    }


def _compact_abstract_hit(doc, score, components):
    return {
        "id": doc.get("id"),
        "title": doc.get("title"),
        "year": doc.get("year"),
        "journal": doc.get("journal"),
        "doi": doc.get("doi"),
        "ut": doc.get("ut"),
        "score": score,
        "components": components,
        "abstract": str(doc.get("abstract") or doc.get("text") or "")[:900],
    }


def _calibrate_kg_with_abstracts(top_k=8, max_docs=0):
    docs = _load_abstract_docs(limit=max_docs or None)
    if not docs:
        raise ValueError("请先上传摘要语料。")
    for doc in docs:
        doc["_token_set"] = set(doc.get("tokens", []))
        doc["_title_tokens"] = set(_tokenize_evidence(doc.get("title", "")))
    kg_records = _kg_chain_records()
    results = []
    for record in kg_records:
        query_terms, priority_terms = _kg_calibration_query(record)
        polarity = _relation_polarity(record.get("relation"))
        scored = []
        for doc in docs:
            score, components = _score_doc_for_kg(doc, query_terms, priority_terms, polarity)
            if score <= 0:
                continue
            scored.append((score, doc, components))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[:top_k]
        contradictory = [
            item for item in scored
            if item[2].get("opposing_signal", 0) > item[2].get("support_signal", 0)
        ][: max(3, top_k // 2)]
        evidence_docs = [(score, doc, components) for score, doc, components in scored if score >= 4.0]
        evidence_count = len(evidence_docs)
        recent = [doc for _, doc, _ in evidence_docs if _safe_int(doc.get("year")) >= 2021]
        confidence = min(0.98, 0.18 + math.log1p(evidence_count) / 5.2 + min(len(selected), top_k) * 0.025)
        if contradictory:
            confidence = max(0.05, confidence - min(0.22, len(contradictory) * 0.035))
        results.append({
            "kg_id": record.get("id"),
            "title": record.get("title"),
            "route": record.get("route"),
            "material": record.get("material"),
            "performance": record.get("performance"),
            "relation": record.get("relation"),
            "query_terms": sorted(query_terms)[:80],
            "evidence_count": evidence_count,
            "recent_evidence_count": len(recent),
            "confidence_score": round(confidence, 3),
            "supporting_abstracts": [_compact_abstract_hit(doc, score, components) for score, doc, components in selected],
            "contradicting_abstracts": [_compact_abstract_hit(doc, score, components) for score, doc, components in contradictory],
        })
    results.sort(key=lambda item: item["confidence_score"], reverse=True)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {"top_k": top_k, "max_docs": max_docs or None},
        "summary": {
            "doc_count": len(docs),
            "kg_chain_count": len(kg_records),
            "calibrated_count": sum(1 for item in results if item["evidence_count"] > 0),
            "high_confidence_count": sum(1 for item in results if item["confidence_score"] >= 0.72),
            "low_evidence_count": sum(1 for item in results if item["evidence_count"] < 3),
        },
        "calibrations": results,
    }
    ABSTRACT_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    ABSTRACT_CALIBRATION_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _sentences_for_candidate(text):
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", str(text or ""))
    return [part.strip() for part in parts if len(part.strip()) >= 60]


def _term_in_text(terms, text):
    lower = str(text or "").lower()
    return any(term.lower() in lower for term in terms)


def _matched_terms(terms, text):
    lower = str(text or "").lower()
    return [term for term in terms if term.lower() in lower]


def _abstract_candidate_triples(doc, limit=8):
    title = doc.get("title", "")
    full_text = doc.get("text") or ""
    lower_title = title.lower()
    retracted = "retracted" in lower_title or "撤稿" in title
    material_pool = {
        key: value for key, value in MATERIAL_TERMS.items()
        if key not in {"water"}
    }
    if not any(term in full_text.lower() for term in ["water-to-binder", "water/cement", "w/b", "w/c", "水胶比"]):
        material_pool.pop("water", None)
    rows = []
    sentences = _sentences_for_candidate(full_text)
    if not sentences:
        sentences = [full_text[:1200]]
    for material, material_terms in material_pool.items():
        if not _term_in_text(material_terms, full_text):
            continue
        for performance, performance_terms in PERFORMANCE_TERMS.items():
            if not _term_in_text(performance_terms, full_text):
                continue
            evidence = ""
            for index, sentence in enumerate(sentences):
                window = " ".join(sentences[max(0, index - 1): min(len(sentences), index + 2)])
                if _term_in_text(material_terms, window) and _term_in_text(performance_terms, window):
                    evidence = window[:900]
                    break
            if not evidence:
                continue
            feature_hits = []
            for label, terms in FEATURE_HINTS:
                if _term_in_text(terms, evidence):
                    feature_hits.append(label)
            if not feature_hits:
                continue
            support, oppose, risk = _doc_relation_signal(evidence, "increase")
            if oppose > support and oppose > 0:
                relation = "risk"
            elif risk and support:
                relation = "tradeoff"
            else:
                relation = "support"
            confidence = 0.48
            confidence += min(0.18, 0.06 * len(feature_hits))
            confidence += 0.06 if _term_in_text(material_terms, title) else 0
            confidence += 0.05 if _term_in_text(performance_terms, title) else 0
            confidence += 0.05 if support else 0
            confidence -= 0.12 if retracted else 0
            confidence = round(max(0.05, min(confidence, 0.88)), 3)
            feature = "；".join(feature_hits[:2])
            rows.append({
                "id": _stable_evidence_id("abstract_candidate", doc.get("id"), material, feature, performance, evidence[:160]),
                "material": material,
                "feature": feature,
                "mechanism": "；".join(feature_hits[:3]),
                "performance": performance,
                "relation": relation,
                "confidence": confidence,
                "status": "pending",
                "source": title,
                "source_type": "abstract_corpus",
                "extractor": "strict_abstract_rule",
                "evidence": evidence,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "path": str(ABSTRACT_CORPUS_DOCS_PATH.relative_to(ROOT)),
                "doc_id": doc.get("id"),
                "journal": doc.get("journal"),
                "year": doc.get("year"),
                "doi": doc.get("doi"),
            })
    return sorted(rows, key=lambda item: item["confidence"], reverse=True)[:limit]


def _discover_abstract_candidates(limit=300, min_confidence=0.52, max_docs=0):
    docs = _load_abstract_docs(limit=max_docs or None)
    if not docs:
        raise ValueError("请先上传摘要语料。")
    existing_keys = {
        (
            str(item.get("material") or "").lower(),
            str(item.get("feature") or "").lower(),
            str(item.get("mechanism") or "").lower(),
            str(item.get("performance") or "").lower(),
        )
        for item in _kg_chain_records()
    }
    candidates = {}
    for doc in docs:
        triples = _abstract_candidate_triples(doc, limit=10)
        for item in triples:
            confidence = _safe_float(item.get("confidence"), 0.0)
            if confidence < min_confidence:
                continue
            key = (
                str(item.get("material") or "").lower(),
                str(item.get("feature") or "").lower(),
                str(item.get("mechanism") or "").lower(),
                str(item.get("performance") or "").lower(),
            )
            if key in existing_keys:
                continue
            if key not in candidates or item["confidence"] > candidates[key]["confidence"]:
                candidates[key] = item
        if len(candidates) >= limit * 3:
            break
    rows = sorted(candidates.values(), key=lambda item: item.get("confidence", 0), reverse=True)[:limit]
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {"limit": limit, "min_confidence": min_confidence, "max_docs": max_docs or None},
        "candidates": rows,
    }
    ABSTRACT_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    ABSTRACT_CANDIDATES_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _merge_abstract_candidates_into_review_pool():
    if not ABSTRACT_CANDIDATES_PATH.exists():
        raise ValueError("尚未生成摘要候选 KG。")
    source_payload = json.loads(ABSTRACT_CANDIDATES_PATH.read_text(encoding="utf-8"))
    payload = _load_kg_candidates()
    existing = {item.get("id") for item in payload.get("candidates", [])}
    added = 0
    for item in source_payload.get("candidates", []):
        if item.get("id") in existing:
            continue
        payload.setdefault("candidates", []).append(item)
        existing.add(item.get("id"))
        added += 1
    _save_kg_candidates(payload)
    return {"added": added, "total": len(payload.get("candidates", []))}


def _detect_terms(term_map, text):
    lower = str(text or "").lower()
    hits = []
    for key, terms in term_map.items():
        if any(term.lower() in lower for term in terms):
            hits.append(key)
    return hits


def _year_bucket(year):
    y = _safe_int(year)
    if not y:
        return "unknown"
    if y < 2000:
        return "<2000"
    if y >= 2026:
        return "2026+"
    return str(y)


def _build_evidence_landscape():
    docs = _load_abstract_docs()
    if not docs:
        raise ValueError("请先上传摘要语料。")
    kg_records = _kg_chain_records()
    kg_pairs = {(item.get("material"), item.get("performance")) for item in kg_records}
    kg_pair_counts = {}
    for item in kg_records:
        key = (item.get("material"), item.get("performance"))
        kg_pair_counts[key] = kg_pair_counts.get(key, 0) + 1
    pair_stats = {}
    material_stats = {}
    performance_stats = {}
    yearly_counts = {}
    doc_rows = []
    for doc in docs:
        text = doc.get("text") or ""
        materials = _detect_terms(MATERIAL_TERMS, text)
        performances = _detect_terms(PERFORMANCE_TERMS, text)
        y = _safe_int(doc.get("year"))
        bucket = _year_bucket(doc.get("year"))
        low_carbon = "low carbon" in performances or any(term in text.lower() for term in ["carbon emission", "co2", "clinker", "cement replacement", "低碳", "碳排放", "熟料"])
        shrinkage_hit = "shrinkage" in performances
        support, oppose, risk = _doc_relation_signal(text, "increase")
        for material in materials:
            stat = material_stats.setdefault(material, {"count": 0, "recent": 0, "low_carbon": 0, "risk": 0})
            stat["count"] += 1
            stat["recent"] += int(y >= 2021)
            stat["low_carbon"] += int(low_carbon)
            stat["risk"] += int(risk > 0 or oppose > support)
        for performance in performances:
            stat = performance_stats.setdefault(performance, {"count": 0, "recent": 0})
            stat["count"] += 1
            stat["recent"] += int(y >= 2021)
        for material in materials:
            for performance in performances:
                key = (material, performance)
                stat = pair_stats.setdefault(key, {"material": material, "performance": performance, "count": 0, "recent": 0, "low_carbon": 0, "risk": 0, "years": {}})
                stat["count"] += 1
                stat["recent"] += int(y >= 2021)
                stat["low_carbon"] += int(low_carbon)
                stat["risk"] += int(risk > 0 or oppose > support)
                stat["years"][bucket] = stat["years"].get(bucket, 0) + 1
                yearly_counts[(material, performance, bucket)] = yearly_counts.get((material, performance, bucket), 0) + 1
        doc_rows.append({
            "id": doc.get("id"),
            "year": y or None,
            "materials": materials,
            "performances": performances,
            "low_carbon": low_carbon,
            "shrinkage": shrinkage_hit,
        })
    pair_rows = []
    max_count = max([stat["count"] for stat in pair_stats.values()] or [1])
    max_recent = max([stat["recent"] for stat in pair_stats.values()] or [1])
    for key, stat in pair_stats.items():
        kg_count = kg_pair_counts.get(key, 0)
        maturity = math.log1p(stat["count"]) / math.log1p(max_count)
        trend = stat["recent"] / max(stat["count"], 1)
        low_carbon_share = stat["low_carbon"] / max(stat["count"], 1)
        risk_share = stat["risk"] / max(stat["count"], 1)
        kg_coverage = min(1.0, kg_count / 3.0)
        gap_score = round(max(0.0, maturity * (1 - kg_coverage)), 4)
        agent_prior = round(0.42 * maturity + 0.23 * trend + 0.2 * kg_coverage + 0.15 * (1 - risk_share), 4)
        pair_rows.append({
            **{k: v for k, v in stat.items() if k != "years"},
            "years": stat["years"],
            "kg_chain_count": kg_count,
            "kg_covered": bool(kg_count),
            "maturity": round(maturity, 4),
            "recent_share": round(trend, 4),
            "low_carbon_share": round(low_carbon_share, 4),
            "risk_share": round(risk_share, 4),
            "gap_score": gap_score,
            "agent_prior": agent_prior,
        })
    pair_rows.sort(key=lambda item: (item["gap_score"], item["count"]), reverse=True)
    calibration = {}
    if ABSTRACT_CALIBRATION_PATH.exists():
        try:
            calibration = json.loads(ABSTRACT_CALIBRATION_PATH.read_text(encoding="utf-8"))
        except Exception:
            calibration = {}
    calibration_rows = calibration.get("calibrations", [])
    kg_rows = []
    for item in calibration_rows:
        contra = item.get("contradicting_abstracts", [])
        evidence = item.get("evidence_count", 0)
        recent = item.get("recent_evidence_count", 0)
        kg_rows.append({
            "kg_id": item.get("kg_id"),
            "material": item.get("material"),
            "performance": item.get("performance"),
            "relation": item.get("relation"),
            "confidence_score": item.get("confidence_score", 0),
            "evidence_count": evidence,
            "recent_evidence_count": recent,
            "contradiction_count": len(contra),
            "uncertainty_score": round((len(contra) + 1) / (evidence + 2), 4),
            "route": item.get("route", []),
        })
    summary = {
        "doc_count": len(docs),
        "with_material_and_performance": sum(1 for row in doc_rows if row["materials"] and row["performances"]),
        "kg_chain_count": len(kg_records),
        "pair_count": len(pair_rows),
        "kg_covered_pair_count": sum(1 for row in pair_rows if row["kg_covered"]),
        "high_gap_pair_count": sum(1 for row in pair_rows if row["gap_score"] >= 0.45),
        "low_carbon_doc_count": sum(1 for row in doc_rows if row["low_carbon"]),
        "shrinkage_doc_count": sum(1 for row in doc_rows if row["shrinkage"]),
        "shrinkage_material_doc_count": sum(1 for row in doc_rows if row["materials"] and row["shrinkage"]),
    }
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "materials": [{"material": key, **value} for key, value in sorted(material_stats.items(), key=lambda kv: kv[1]["count"], reverse=True)],
        "performances": [{"performance": key, **value} for key, value in sorted(performance_stats.items(), key=lambda kv: kv[1]["count"], reverse=True)],
        "pairs": pair_rows,
        "kg_calibration": kg_rows,
        "agent_priors": _agent_prior_matrix(pair_rows, kg_rows),
    }
    ABSTRACT_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    ABSTRACT_LANDSCAPE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_landscape_tables(payload)
    return payload


def _agent_prior_matrix(pair_rows, kg_rows):
    by_pair = {(row["material"], row["performance"]): row for row in pair_rows}
    kg_by_pair = {}
    for row in kg_rows:
        key = (row.get("material"), row.get("performance"))
        kg_by_pair.setdefault(key, []).append(row)
    priors = []
    for key, row in by_pair.items():
        kg_items = kg_by_pair.get(key, [])
        kg_conf = np_mean([item.get("confidence_score", 0) for item in kg_items]) if kg_items else 0
        uncertainty = np_mean([item.get("uncertainty_score", 0) for item in kg_items]) if kg_items else 0.65
        priors.append({
            "material": row["material"],
            "performance": row["performance"],
            "retrieval_weight": round(0.55 * row["agent_prior"] + 0.45 * kg_conf, 4),
            "critic_risk": round(min(1.0, 0.55 * row["risk_share"] + 0.45 * uncertainty), 4),
            "optimizer_maturity": row["maturity"],
            "report_citation_priority": round(min(1.0, 0.5 * row["maturity"] + 0.35 * row["recent_share"] + 0.15 * row["low_carbon_share"]), 4),
            "gap_score": row["gap_score"],
            "kg_confidence": round(kg_conf, 4),
        })
    priors.sort(key=lambda item: item["retrieval_weight"], reverse=True)
    return priors


def np_mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def _write_landscape_tables(payload):
    def write_csv(path, rows):
        if not rows:
            return
        keys = sorted({key for row in rows for key in row.keys() if key != "years" and not isinstance(row.get(key), (list, dict))})
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in keys})
    write_csv(ABSTRACT_CORPUS_DIR / "material_performance_landscape.csv", payload.get("pairs", []))
    write_csv(ABSTRACT_CORPUS_DIR / "agent_priors.csv", payload.get("agent_priors", []))
    write_csv(ABSTRACT_CORPUS_DIR / "kg_calibration_table.csv", payload.get("kg_calibration", []))


def _shrinkage_categories(text):
    lower = str(text or "").lower()
    hits = []
    for category, terms in SHRINKAGE_EVIDENCE_CATEGORIES.items():
        if any(term.lower() in lower for term in terms):
            hits.append(category)
    return hits or ["general_shrinkage"]


def _evidence_excerpt(text, terms, size=360):
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    lower = body.lower()
    positions = [lower.find(term.lower()) for term in terms if term and lower.find(term.lower()) >= 0]
    anchor = min(positions) if positions else 0
    start = max(0, anchor - 120)
    end = min(len(body), anchor + size)
    return body[start:end].strip()


def _build_shrinkage_evidence_summary():
    docs = _load_abstract_docs()
    if not docs:
        raise ValueError("请先上传摘要语料。")
    material_stats = {}
    category_stats = {}
    year_stats = {}
    examples = []
    shrinkage_doc_count = 0
    for doc in docs:
        text = doc.get("text") or ""
        lower = text.lower()
        if not any(term.lower() in lower for term in SHRINKAGE_EVIDENCE_TERMS):
            continue
        shrinkage_doc_count += 1
        materials = _detect_terms(MATERIAL_TERMS, text)
        categories = _shrinkage_categories(text)
        y = _safe_int(doc.get("year"))
        recent = int(y >= 2021)
        bucket = _year_bucket(doc.get("year"))
        year_stats[bucket] = year_stats.get(bucket, 0) + 1
        for category in categories:
            category_stats[category] = category_stats.get(category, 0) + 1
        compact = {
            "id": doc.get("id"),
            "title": doc.get("title"),
            "year": y or None,
            "doi": doc.get("doi"),
            "categories": categories,
            "materials": materials,
            "excerpt": _evidence_excerpt(text, SHRINKAGE_EVIDENCE_TERMS),
        }
        if len(examples) < 80:
            examples.append(compact)
        for material in materials:
            stat = material_stats.setdefault(material, {
                "material": material,
                "count": 0,
                "recent": 0,
                "categories": {},
                "examples": [],
            })
            stat["count"] += 1
            stat["recent"] += recent
            for category in categories:
                stat["categories"][category] = stat["categories"].get(category, 0) + 1
            if len(stat["examples"]) < 6:
                stat["examples"].append(compact)
    material_rows = sorted(material_stats.values(), key=lambda item: item["count"], reverse=True)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "doc_count": len(docs),
            "shrinkage_doc_count": shrinkage_doc_count,
            "material_pair_doc_count": sum(row["count"] for row in material_rows),
            "material_count": len(material_rows),
            "category_count": category_stats,
            "year_count": year_stats,
        },
        "materials": material_rows,
        "examples": examples,
    }
    ABSTRACT_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    ABSTRACT_SHRINKAGE_EVIDENCE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with ABSTRACT_SHRINKAGE_EVIDENCE_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        fieldnames = ["material", "count", "recent", "top_categories", "example_title", "example_year", "example_excerpt"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in material_rows:
            top_categories = "; ".join(
                key for key, _ in sorted(row.get("categories", {}).items(), key=lambda kv: kv[1], reverse=True)[:4]
            )
            example = (row.get("examples") or [{}])[0]
            writer.writerow({
                "material": row.get("material"),
                "count": row.get("count"),
                "recent": row.get("recent"),
                "top_categories": top_categories,
                "example_title": example.get("title", ""),
                "example_year": example.get("year", ""),
                "example_excerpt": example.get("excerpt", ""),
            })
    return payload


def _load_shrinkage_evidence_summary(build_if_missing=False):
    if not ABSTRACT_SHRINKAGE_EVIDENCE_PATH.exists():
        return _build_shrinkage_evidence_summary() if build_if_missing else {}
    try:
        return json.loads(ABSTRACT_SHRINKAGE_EVIDENCE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _build_shrinkage_evidence_summary() if build_if_missing else {}


def _shrinkage_evidence_for_materials(materials, limit=8):
    payload = _load_shrinkage_evidence_summary(build_if_missing=False)
    if not payload:
        return {}
    wanted = set(materials or [])
    rows = [
        row for row in payload.get("materials", [])
        if not wanted or row.get("material") in wanted
    ]
    rows.sort(key=lambda item: item.get("count", 0), reverse=True)
    return {
        "updated_at": payload.get("updated_at"),
        "summary": payload.get("summary", {}),
        "materials": rows[:limit],
    }


def _qwen_messages(system, user, assistant_obj):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False)},
        ]
    }


def _generate_lora_datasets(max_examples=5000, base_model="Qwen/Qwen3-4B-Instruct-2507", embedding_model="text-embedding-qwen3-embedding-0.6b", output_dir=None):
    if not ABSTRACT_CALIBRATION_PATH.exists():
        raise ValueError("请先运行 KG 证据校准。")
    if not ABSTRACT_LANDSCAPE_PATH.exists():
        _build_evidence_landscape()
    calibration = json.loads(ABSTRACT_CALIBRATION_PATH.read_text(encoding="utf-8"))
    landscape = json.loads(ABSTRACT_LANDSCAPE_PATH.read_text(encoding="utf-8"))
    rows = []
    sys_evidence = "你是低碳混凝土知识图谱证据判别器。只输出 JSON，不要解释。"
    sys_extract = "你是混凝土材料机理链抽取器。只输出 JSON，不要解释。"
    sys_route = "你是多智能体混凝土设计证据路由器。只输出 JSON，不要解释。"

    for item in calibration.get("calibrations", []):
        kg_edge = {
            "material": item.get("material"),
            "route": item.get("route"),
            "performance": item.get("performance"),
            "relation": item.get("relation"),
        }
        for doc in item.get("supporting_abstracts", [])[:4]:
            rows.append(_qwen_messages(
                sys_evidence,
                json.dumps({"task": "classify_evidence", "kg_edge": kg_edge, "abstract": doc.get("abstract", "")[:1200]}, ensure_ascii=False),
                {
                    "label": "support",
                    "confidence": min(0.95, max(0.55, float(item.get("confidence_score", 0.75)))),
                    "evidence_span": str(doc.get("abstract", ""))[:320],
                    "rationale": "摘要同时命中材料、性能和机理相关线索，可作为该 Gold KG 边的支持证据。",
                },
            ))
        for doc in item.get("contradicting_abstracts", [])[:2]:
            rows.append(_qwen_messages(
                sys_evidence,
                json.dumps({"task": "classify_evidence", "kg_edge": kg_edge, "abstract": doc.get("abstract", "")[:1200]}, ensure_ascii=False),
                {
                    "label": "boundary",
                    "confidence": 0.66,
                    "evidence_span": str(doc.get("abstract", ""))[:320],
                    "rationale": "摘要包含反向或风险线索，应作为适用边界而不是直接支持证据。",
                },
            ))
        if item.get("route"):
            rows.append(_qwen_messages(
                sys_extract,
                json.dumps({"task": "extract_mechanism_chain", "text": " -> ".join(item.get("route", [])), "evidence": (item.get("supporting_abstracts") or [{}])[0].get("abstract", "")[:900]}, ensure_ascii=False),
                {
                    "material": item.get("material"),
                    "feature": (item.get("route") or ["", ""])[1] if len(item.get("route") or []) > 1 else "",
                    "mechanism": (item.get("route") or ["", "", ""])[2] if len(item.get("route") or []) > 2 else "",
                    "consequence": (item.get("route") or ["", "", "", ""])[3] if len(item.get("route") or []) > 3 else "",
                    "performance": item.get("performance"),
                    "relation": item.get("relation"),
                    "evidence_level": "gold_kg_calibrated",
                },
            ))

    for prior in landscape.get("agent_priors", [])[: min(1200, max_examples)]:
        request = {
            "task": "route_multi_agent_evidence",
            "material": prior.get("material"),
            "performance": prior.get("performance"),
            "retrieval_weight": prior.get("retrieval_weight"),
            "critic_risk": prior.get("critic_risk"),
            "gap_score": prior.get("gap_score"),
        }
        critic = float(prior.get("critic_risk", 0))
        gap = float(prior.get("gap_score", 0))
        rows.append(_qwen_messages(
            sys_route,
            json.dumps(request, ensure_ascii=False),
            {
                "retrieval_focus": [prior.get("material"), prior.get("performance")],
                "agent_weights": {
                    "mechanism_retrieval": round(float(prior.get("retrieval_weight", 0.5)), 3),
                    "critic": round(min(1.0, 0.35 + critic), 3),
                    "optimizer": round(max(0.15, 0.55 - gap * 0.25), 3),
                    "report": round(float(prior.get("report_citation_priority", 0.5)), 3),
                },
                "risk_checks": ["适用边界", "反向证据"] if critic > 0.35 else ["证据一致性"],
                "use_in_design": "soft_prior" if gap < 0.45 else "gap_warning",
            },
        ))

    # Deterministic split without random dependency.
    dedup = {}
    for row in rows:
        key = _stable_evidence_id(json.dumps(row, ensure_ascii=False)[:1600])
        dedup[key] = row
    rows = list(dedup.values())[:max_examples]
    val_count = max(80, min(500, len(rows) // 10))
    train_rows = rows[val_count:]
    val_rows = rows[:val_count]
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    with LORA_DATASET_PATH.open("w", encoding="utf-8") as fh:
        for row in train_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with LORA_VALIDATION_PATH.open("w", encoding="utf-8") as fh:
        for row in val_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    config = _save_lora_config({
        "base_model": base_model,
        "embedding_model": embedding_model,
        "max_examples": max_examples,
        "output_dir": output_dir or str(LORA_DEFAULT_OUTPUT_DIR),
    })
    script = _write_lora_training_script(base_model=base_model, output_dir=config.get("output_dir"), config=config)
    manifest = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "base_model": base_model,
        "embedding_model": embedding_model,
        "output_dir": config.get("output_dir"),
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "tasks": ["evidence_classification", "mechanism_extraction", "agent_routing"],
        "train_path": str(LORA_DATASET_PATH.relative_to(ROOT)).replace("\\", "/"),
        "validation_path": str(LORA_VALIDATION_PATH.relative_to(ROOT)).replace("\\", "/"),
        "training_script": str(script.relative_to(ROOT)).replace("\\", "/"),
    }
    LORA_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _lora_status():
    config = _load_lora_config()
    artifacts = {
        "train": LORA_DATASET_PATH,
        "validation": LORA_VALIDATION_PATH,
        "manifest": LORA_MANIFEST_PATH,
        "script": LORA_DIR / "train_qwen3_lora.py",
    }
    manifest = {}
    if LORA_MANIFEST_PATH.exists():
        try:
            manifest = json.loads(LORA_MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    files = {}
    for key, path in artifacts.items():
        files[key] = {
            "exists": path.exists(),
            "name": path.name,
            "bytes": path.stat().st_size if path.exists() else 0,
            "url": f"/api/abstract-corpus/lora/export/{key}" if path.exists() else "",
        }
    output_dir = Path(config.get("output_dir") or LORA_DEFAULT_OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = LORA_DIR / output_dir
    adapter_files = [
        output_dir / "adapter_model.safetensors",
        output_dir / "adapter_model.bin",
        output_dir / "pytorch_model.bin",
    ]
    trainer_state_path = output_dir / "trainer_state.json"
    metrics_paths = [output_dir / "all_results.json", output_dir / "train_results.json", output_dir / "eval_results.json"]
    checkpoints = sorted(output_dir.glob("checkpoint-*")) if output_dir.exists() else []
    adapter_path = next((path for path in adapter_files if path.exists()), None)
    trainer_state = {}
    if trainer_state_path.exists():
        try:
            trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        except Exception:
            trainer_state = {}
    metrics = {}
    for path in metrics_paths:
        if path.exists():
            try:
                metrics[path.stem] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                metrics[path.stem] = {}
    completed = bool(adapter_path)
    if completed:
        stage = "trained"
        stage_label = "已检测到 LoRA adapter"
    elif checkpoints:
        stage = "checkpointed"
        stage_label = "检测到 checkpoint，训练可能未完成"
    elif LORA_DATASET_PATH.exists() and LORA_VALIDATION_PATH.exists():
        stage = "dataset_ready"
        stage_label = "已生成训练数据，尚未检测到微调结果"
    else:
        stage = "not_ready"
        stage_label = "尚未生成 LoRA 数据"
    figures = _list_lora_figures()
    return {
        "exists": bool(manifest),
        "config": config,
        "manifest": manifest,
        "files": files,
        "training": {
            "stage": stage,
            "stage_label": stage_label,
            "completed": completed,
            "output_dir": str(output_dir),
            "adapter_path": str(adapter_path) if adapter_path else "",
            "checkpoint_count": len(checkpoints),
            "trainer_state_exists": trainer_state_path.exists(),
            "log_points": len(trainer_state.get("log_history", [])) if isinstance(trainer_state, dict) else 0,
            "metrics": metrics,
        },
        "figures": figures,
    }


def _load_lora_config():
    config = {
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "embedding_model": "text-embedding-qwen3-embedding-0.6b",
        "max_examples": 5000,
        "output_dir": str(LORA_DEFAULT_OUTPUT_DIR),
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 1.5e-4,
        "epochs": 2,
        "max_seq_length": 2048,
    }
    if LORA_CONFIG_PATH.exists():
        try:
            saved = json.loads(LORA_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                config.update({key: saved[key] for key in config.keys() if key in saved})
        except Exception:
            pass
    config["max_examples"] = max(300, min(_safe_int(config.get("max_examples"), 5000), 20000))
    config["lora_r"] = max(4, min(_safe_int(config.get("lora_r"), 16), 128))
    config["lora_alpha"] = max(4, min(_safe_int(config.get("lora_alpha"), 32), 256))
    config["epochs"] = max(1, min(_safe_int(config.get("epochs"), 2), 8))
    config["max_seq_length"] = max(512, min(_safe_int(config.get("max_seq_length"), 2048), 8192))
    config["lora_dropout"] = max(0.0, min(_safe_float(config.get("lora_dropout"), 0.05), 0.5))
    config["learning_rate"] = max(1e-6, min(_safe_float(config.get("learning_rate"), 1.5e-4), 5e-3))
    return config


def _save_lora_config(payload):
    config = _load_lora_config()
    for key in ["base_model", "embedding_model", "output_dir"]:
        if key in payload:
            value = str(payload.get(key) or "").strip()
            if value:
                config[key] = value
    for key in ["max_examples", "lora_r", "lora_alpha", "epochs", "max_seq_length"]:
        if key in payload:
            config[key] = _safe_int(payload.get(key), config[key])
    for key in ["lora_dropout", "learning_rate"]:
        if key in payload:
            config[key] = _safe_float(payload.get(key), config[key])
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    config = {**_load_lora_config(), **config}
    LORA_CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def _list_lora_figures():
    if not LORA_FIGURE_DIR.exists():
        return []
    groups = {}
    for path in sorted(LORA_FIGURE_DIR.glob("*.png")):
        stem = path.stem
        groups[stem] = {
            "id": stem,
            "title": stem.replace("_", " "),
            "files": {
                ext: f"evidence/abstract-corpus/lora/figures/{stem}.{ext}"
                for ext in ["svg", "pdf", "png", "tiff"]
                if (LORA_FIGURE_DIR / f"{stem}.{ext}").exists()
            },
        }
    return list(groups.values())


def _save_lora_figure(fig, stem):
    LORA_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext, kwargs in {
        "svg": {},
        "pdf": {},
        "png": {"dpi": 300},
        "tiff": {"dpi": 600},
    }.items():
        path = LORA_FIGURE_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths[ext] = str(path.relative_to(ROOT)).replace("\\", "/")
    return paths


def _read_lora_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _lora_row_task(row):
    try:
        user = row.get("messages", [{}, {}])[1].get("content", "{}")
        payload = json.loads(user)
        return payload.get("task", "unknown")
    except Exception:
        return "unknown"


def _lora_row_label(row):
    try:
        assistant = row.get("messages", [{}, {}, {}])[2].get("content", "{}")
        payload = json.loads(assistant)
        return payload.get("label") or payload.get("use_in_design") or payload.get("evidence_level") or "routing"
    except Exception:
        return "unknown"


def _generate_lora_figures():
    plt, mpl, sns, colors = _import_plotting_stack()
    import numpy as _np  # type: ignore
    import pandas as _pd  # type: ignore
    status = _lora_status()
    train_rows = _read_lora_jsonl(LORA_DATASET_PATH)
    val_rows = _read_lora_jsonl(LORA_VALIDATION_PATH)
    all_rows = [{"split": "train", **row} for row in train_rows] + [{"split": "validation", **row} for row in val_rows]
    records = []
    for row in all_rows:
        text = json.dumps(row, ensure_ascii=False)
        records.append({
            "split": row.get("split", "train"),
            "task": _lora_row_task(row),
            "label": _lora_row_label(row),
            "chars": len(text),
            "messages": len(row.get("messages", [])),
        })
    df = _pd.DataFrame(records)
    figures = []

    fig1 = plt.figure(figsize=(11.2, 7.2))
    gs = fig1.add_gridspec(2, 2, wspace=0.3, hspace=0.34)
    ax = fig1.add_subplot(gs[0, 0])
    ax.axis("off")
    training = status.get("training", {})
    stage_color = colors["teal"] if training.get("completed") else colors["gold"] if training.get("stage") == "dataset_ready" else colors["red"]
    boxes = [
        ("Dataset", f"{len(train_rows)} train\n{len(val_rows)} validation", colors["blue"]),
        ("Adapter", "detected" if training.get("completed") else "not detected", stage_color),
        ("Logs", f"{training.get('log_points', 0)} log points", colors["purple"]),
        ("Use", "Skill + RAG + LoRA", colors["orange"]),
    ]
    for idx, (title, value, color) in enumerate(boxes):
        x = 0.06 + (idx % 2) * 0.48
        y = 0.58 - (idx // 2) * 0.36
        ax.add_patch(mpl.patches.FancyBboxPatch((x, y), 0.36, 0.22, boxstyle="round,pad=0.02,rounding_size=0.025", facecolor=color, alpha=0.14, edgecolor=color, linewidth=1.2))
        ax.text(x + 0.18, y + 0.145, title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x + 0.18, y + 0.075, value, ha="center", va="center", fontsize=8.2, color="#374151")
    ax.set_title("a  LoRA completion dashboard", loc="left", fontsize=10, fontweight="bold")
    ax = fig1.add_subplot(gs[0, 1])
    if not df.empty:
        task_counts = df.groupby(["split", "task"]).size().unstack(fill_value=0)
        task_counts.T.plot(kind="barh", stacked=True, ax=ax, color=[colors["blue"], colors["teal"]])
    ax.set_title("b  Supervised tasks by split", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("examples")
    ax = fig1.add_subplot(gs[1, 0])
    if not df.empty:
        sns.violinplot(data=df, x="split", y="chars", ax=ax, inner="quartile", palette=[colors["blue"], colors["teal"]])
    ax.set_title("c  Prompt-response length distribution", loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel("JSONL row characters")
    ax = fig1.add_subplot(gs[1, 1])
    labels = ["dataset", "script", "adapter", "logs"]
    values = [
        1 if LORA_DATASET_PATH.exists() and LORA_VALIDATION_PATH.exists() else 0,
        1 if (LORA_DIR / "train_qwen3_lora.py").exists() else 0,
        1 if training.get("completed") else 0,
        1 if training.get("trainer_state_exists") else 0,
    ]
    ax.bar(labels, values, color=[colors["teal"] if v else colors["gray"] for v in values])
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 1])
    ax.set_title("d  Training readiness and completion checks", loc="left", fontsize=10, fontweight="bold")
    fig1.suptitle("LoRA fine-tuning status: data generation is not training completion", x=0.01, ha="left", fontsize=12, fontweight="bold")
    paths = _save_lora_figure(fig1, "FigL1_lora_status_dashboard")
    plt.close(fig1)
    figures.append({"id": "FigL1_lora_status_dashboard", "title": "LoRA status dashboard", "files": paths})

    fig2 = plt.figure(figsize=(11.2, 7.2))
    gs2 = fig2.add_gridspec(2, 2, wspace=0.36, hspace=0.36)
    ax = fig2.add_subplot(gs2[:, 0])
    if not df.empty:
        pivot = df.groupby(["task", "label"]).size().unstack(fill_value=0)
        sns.heatmap(pivot, ax=ax, cmap="YlGnBu", annot=True, fmt="d", linewidths=0.45, linecolor="white", cbar_kws={"label": "examples"})
    ax.set_title("a  Task-label supervision matrix", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("target label / output mode")
    ax.set_ylabel("training task")
    ax = fig2.add_subplot(gs2[0, 1])
    if not df.empty:
        split_counts = df["split"].value_counts()
        ax.pie(split_counts.values, labels=split_counts.index, colors=[colors["blue"], colors["teal"]], autopct="%1.0f%%", textprops={"fontsize": 8})
    ax.set_title("b  Split balance", loc="left", fontsize=10, fontweight="bold")
    ax = fig2.add_subplot(gs2[1, 1])
    if not df.empty:
        top_labels = df["label"].value_counts().head(8)
        ax.barh(top_labels.index[::-1], top_labels.values[::-1], color=colors["purple"])
    ax.set_title("c  Output behavior coverage", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("examples")
    fig2.suptitle("What the LoRA adapter is being trained to do", x=0.01, ha="left", fontsize=12, fontweight="bold")
    paths = _save_lora_figure(fig2, "FigL2_lora_supervision_map")
    plt.close(fig2)
    figures.append({"id": "FigL2_lora_supervision_map", "title": "LoRA supervision map", "files": paths})

    fig3 = plt.figure(figsize=(11.2, 6.6))
    ax = fig3.add_subplot(111)
    log_history = []
    output_dir = Path(status.get("training", {}).get("output_dir") or "")
    state_path = output_dir / "trainer_state.json"
    if state_path.exists():
        try:
            log_history = json.loads(state_path.read_text(encoding="utf-8")).get("log_history", [])
        except Exception:
            log_history = []
    train_logs = [item for item in log_history if "loss" in item and "step" in item]
    eval_logs = [item for item in log_history if "eval_loss" in item and "step" in item]
    if train_logs:
        ax.plot([item["step"] for item in train_logs], [item["loss"] for item in train_logs], color=colors["blue"], linewidth=2.0, label="train loss")
    if eval_logs:
        ax.plot([item["step"] for item in eval_logs], [item["eval_loss"] for item in eval_logs], color=colors["red"], linewidth=2.0, marker="o", markersize=3, label="eval loss")
    if not train_logs and not eval_logs:
        ax.axis("off")
        ax.text(0.5, 0.58, "Training metrics not detected", ha="center", va="center", fontsize=16, fontweight="bold", color=colors["slate"])
        ax.text(0.5, 0.45, "Run train_qwen3_lora.py on a GPU, then refresh this page.\nCompletion requires adapter weights and trainer_state.json in the output directory.", ha="center", va="center", fontsize=9, color="#4b5563")
    else:
        ax.legend()
        ax.set_xlabel("training step")
        ax.set_ylabel("loss")
    ax.set_title("LoRA training curve and evaluation trace", loc="left", fontsize=12, fontweight="bold")
    paths = _save_lora_figure(fig3, "FigL3_lora_training_curve")
    plt.close(fig3)
    figures.append({"id": "FigL3_lora_training_curve", "title": "LoRA training curve", "files": paths})

    return {"updated_at": datetime.now().isoformat(timespec="seconds"), "figures": figures, "lora_status": _lora_status()}


def _write_lora_training_script(base_model="Qwen/Qwen3-4B-Instruct-2507", output_dir=None, config=None):
    path = LORA_DIR / "train_qwen3_lora.py"
    output_dir = output_dir or str(LORA_DEFAULT_OUTPUT_DIR)
    config = config or _load_lora_config()
    path.write_text(
        r'''
from datasets import load_dataset
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

BASE_MODEL = "__BASE_MODEL__"
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = str(SCRIPT_DIR / "qwen3_lora_train.jsonl")
VAL_FILE = str(SCRIPT_DIR / "qwen3_lora_validation.jsonl")
OUTPUT_DIR = r"__OUTPUT_DIR__"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    device_map="auto",
    load_in_4bit=True,
)

peft_config = LoraConfig(
    r=__LORA_R__,
    lora_alpha=__LORA_ALPHA__,
    lora_dropout=__LORA_DROPOUT__,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)

dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})

def format_example(example):
    return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=__LEARNING_RATE__,
    num_train_epochs=__EPOCHS__,
    warmup_ratio=0.05,
    logging_steps=20,
    save_steps=250,
    eval_steps=250,
    evaluation_strategy="steps",
    bf16=True,
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=args,
    formatting_func=format_example,
    max_seq_length=__MAX_SEQ_LENGTH__,
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
'''.strip(),
        encoding="utf-8",
    )
    text = path.read_text(encoding="utf-8")
    text = text.replace("__BASE_MODEL__", base_model.replace('"', '\\"'))
    text = text.replace("__OUTPUT_DIR__", str(output_dir).replace('"', '\\"'))
    text = text.replace("__LORA_R__", str(_safe_int(config.get("lora_r"), 16)))
    text = text.replace("__LORA_ALPHA__", str(_safe_int(config.get("lora_alpha"), 32)))
    text = text.replace("__LORA_DROPOUT__", str(_safe_float(config.get("lora_dropout"), 0.05)))
    text = text.replace("__LEARNING_RATE__", str(_safe_float(config.get("learning_rate"), 1.5e-4)))
    text = text.replace("__EPOCHS__", str(_safe_int(config.get("epochs"), 2)))
    text = text.replace("__MAX_SEQ_LENGTH__", str(_safe_int(config.get("max_seq_length"), 2048)))
    path.write_text(text, encoding="utf-8")
    return path


def _import_plotting_stack():
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib as mpl  # type: ignore
    import seaborn as sns  # type: ignore
    try:
        from core.plotting_utils import NATURE_COLORS, apply_nature_style  # type: ignore
    except Exception:
        NATURE_COLORS = {
            "blue": "#1f77b4", "teal": "#1b9e77", "orange": "#d95f02", "red": "#c44e52",
            "gold": "#b8860b", "purple": "#7b6fd0", "slate": "#4c566a", "gray": "#b0b7c3", "mint": "#66c2a5",
        }
        def apply_nature_style():
            sns.set_theme(style="whitegrid", palette="deep")
            mpl.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "DejaVu Sans", "SimSun"],
                "axes.unicode_minus": False,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "figure.dpi": 300,
                "savefig.dpi": 600,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "grid.alpha": 0.2,
                "grid.linestyle": "--",
            })
    apply_nature_style()
    return plt, mpl, sns, NATURE_COLORS


def _save_pub_figure(fig, stem):
    ABSTRACT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext, kwargs in {
        "svg": {},
        "pdf": {},
        "png": {"dpi": 300},
        "tiff": {"dpi": 600},
    }.items():
        path = ABSTRACT_FIGURE_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths[ext] = str(path.relative_to(ROOT)).replace("\\", "/")
    return paths


def _top_items(rows, key, n=10):
    return sorted(rows, key=lambda item: item.get(key, 0), reverse=True)[:n]


def _generate_landscape_figures():
    if ABSTRACT_LANDSCAPE_PATH.exists():
        payload = json.loads(ABSTRACT_LANDSCAPE_PATH.read_text(encoding="utf-8"))
    else:
        payload = _build_evidence_landscape()
    shrinkage_payload = _load_shrinkage_evidence_summary(build_if_missing=True)
    plt, mpl, sns, colors = _import_plotting_stack()
    import numpy as _np  # type: ignore
    import pandas as _pd  # type: ignore
    figures = []
    pairs = payload.get("pairs", [])
    kg_rows = payload.get("kg_calibration", [])
    priors = payload.get("agent_priors", [])
    materials = ["cement", "water", "flyash", "GGBFS", "silicafume", "metakaolin", "limestone", "ricehuskash", "superplasticizer", "steelfiber", "fineaggregate", "coarseaggregate"]
    perfs = ["shrinkage", "compressive strength", "flowability"]
    perf_labels = {
        "shrinkage": "shrinkage",
        "compressive strength": "strength",
        "flowability": "flowability",
    }
    material_labels = {
        "cement": "cement",
        "water": "water",
        "flyash": "fly ash",
        "GGBFS": "GGBFS",
        "silicafume": "silica fume",
        "metakaolin": "metakaolin",
        "limestone": "limestone",
        "ricehuskash": "rice husk ash",
        "superplasticizer": "PCE",
        "steelfiber": "steel fiber",
        "fineaggregate": "fine agg.",
        "coarseaggregate": "coarse agg.",
    }

    def mat_label(value):
        return material_labels.get(str(value), str(value))

    def perf_label(value):
        return perf_labels.get(str(value), str(value))

    pair_df = _pd.DataFrame(pairs)
    kg_df = _pd.DataFrame(kg_rows)
    prior_df = _pd.DataFrame(priors)

    heat = _pd.DataFrame(0.0, index=materials, columns=perfs)
    kg_cover = _pd.DataFrame(0.0, index=materials, columns=perfs)
    for row in pairs:
        if row["material"] in heat.index and row["performance"] in heat.columns:
            heat.loc[row["material"], row["performance"]] = math.log10(row["count"] + 1)
            kg_cover.loc[row["material"], row["performance"]] = row.get("kg_chain_count", 0)

    summary = payload.get("summary", {})
    shrink_summary = shrinkage_payload.get("summary", {})
    shrink_materials = _pd.DataFrame(shrinkage_payload.get("materials", []))
    fig0 = plt.figure(figsize=(14.2, 9.4))
    gs0 = fig0.add_gridspec(3, 4, width_ratios=[1.35, 1.0, 1.0, 1.15], height_ratios=[0.82, 1.12, 1.12], wspace=0.42, hspace=0.50)
    doc_lanes = ["other", "shrinkage", "strength", "flowability"]
    lane_color = {
        "other": "#c9d1d9",
        "shrinkage": colors["red"],
        "strength": colors["blue"],
        "flowability": colors["teal"],
    }
    docs = _load_abstract_docs()
    rng = _np.random.default_rng(20260609)
    theta_points, radius_points, c_points, s_points = [], [], [], []
    lane_counts = {key: 0 for key in doc_lanes}
    for doc in docs:
        text = doc.get("text") or ""
        doc_perfs = set(_detect_terms(PERFORMANCE_TERMS, text))
        if "shrinkage" in doc_perfs:
            lane = "shrinkage"
        elif "compressive strength" in doc_perfs:
            lane = "strength"
        elif "flowability" in doc_perfs:
            lane = "flowability"
        else:
            lane = "other"
        lane_counts[lane] += 1
        ring_idx = doc_lanes.index(lane)
        year = min(2025, max(2000, _safe_int(doc.get("year"), 2000)))
        theta = _np.deg2rad(212 - 344 * ((year - 2000) / 25.0) + rng.normal(0, 1.1))
        radius = 0.34 + ring_idx * 0.115 + rng.normal(0, 0.012)
        theta_points.append(theta)
        radius_points.append(radius)
        c_points.append(lane_color[lane])
        s_points.append(2.4 if lane == "shrinkage" else (1.7 if lane in {"strength", "flowability"} else 1.0))

    ax = fig0.add_subplot(gs0[0:2, 0:2], projection="polar")
    if theta_points:
        ax.scatter(theta_points, radius_points, c=c_points, s=s_points, alpha=0.22, linewidths=0, rasterized=True)
        ax.scatter([], [], c=colors["red"], s=18, alpha=0.75, label=f"shrinkage/shrinkage: {lane_counts['shrinkage']:,}")
        ax.scatter([], [], c=colors["blue"], s=18, alpha=0.75, label=f"strength: {lane_counts['strength']:,}")
        ax.scatter([], [], c=colors["teal"], s=18, alpha=0.75, label=f"flowability: {lane_counts['flowability']:,}")
        ax.scatter([], [], c="#7f8a99", s=18, alpha=0.55, label=f"total abstracts: {len(docs):,}")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-132)
    ax.set_thetamax(212)
    ax.set_ylim(0.22, 1.05)
    ax.set_yticks([])
    year_ticks = [2000, 2005, 2010, 2015, 2020, 2025]
    theta_ticks = [_np.deg2rad(212 - 344 * ((year - 2000) / 25.0)) for year in year_ticks]
    ax.set_xticks([])
    ax.grid(color="#d9dee5", linewidth=0.55, alpha=0.55)
    for year, theta in zip(year_ticks, theta_ticks):
        ax.text(theta, 1.02, str(year), ha="center", va="center", fontsize=7.2, color=colors["slate"])
    for idx, lane in enumerate(doc_lanes):
        radius = 0.34 + idx * 0.115
        theta_grid = _np.linspace(_np.deg2rad(-132), _np.deg2rad(212), 280)
        ax.plot(theta_grid, _np.full_like(theta_grid, radius), color=lane_color[lane], alpha=0.16, linewidth=1.0)
        label_theta = _np.deg2rad(222)
        ax.text(label_theta, radius, perf_label(lane) if lane != "other" else "other", ha="right", va="center", fontsize=7.2, color=lane_color[lane], fontweight="bold")
    ax.text(0, 0.12, "angle = publication year\nring = evidence class\npoint = one abstract", ha="center", va="center", fontsize=7.2, color=colors["slate"])
    ax.set_title("a  Corpus evidence cartography: 40,600 abstracts as a radial evidence atlas", loc="left", fontsize=11, fontweight="bold", pad=18)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.04, 1.04), fontsize=7.1, frameon=False, markerscale=1.2)

    ax = fig0.add_subplot(gs0[0, 2:])
    ax.axis("off")
    metrics = [
        (f"{len(docs):,}", "abstracts", colors["blue"]),
        (f"{int(shrink_summary.get('shrinkage_doc_count', summary.get('shrinkage_doc_count', 0))):,}", "shrinkage/shrinkage abstracts", colors["red"]),
        (f"{int(shrink_summary.get('material_pair_doc_count', 0)):,}", "material-shrinkage links", colors["teal"]),
        (f"{int(summary.get('kg_chain_count', 0)):,}", "calibrated KG chains", colors["purple"]),
    ]
    metric_pos = [(0.03, 0.66), (0.53, 0.66), (0.03, 0.28), (0.53, 0.28)]
    for (value, label, color), (x, y) in zip(metrics, metric_pos):
        ax.text(x, y, value, transform=ax.transAxes, fontsize=18, fontweight="bold", color=color, ha="left", va="center")
        ax.text(x + 0.22, y, label, transform=ax.transAxes, fontsize=7.2, color="#202123", ha="left", va="center")
        ax.plot([x, x + 0.17], [y - 0.16, y - 0.16], transform=ax.transAxes, color=color, lw=3.0, alpha=0.65)
    ax.set_title("b  Scale anchors for the evidence layer", loc="left", fontsize=11, fontweight="bold")

    ax = fig0.add_subplot(gs0[1, 2:])
    ax.axis("off")
    strength_doc_count = int(pair_df.loc[pair_df["performance"] == "compressive strength", "count"].sum()) if not pair_df.empty else 0
    flowability_doc_count = int(pair_df.loc[pair_df["performance"] == "flowability", "count"].sum()) if not pair_df.empty else 0
    flow_nodes = [
        ("Abstract corpus", 0.06, 0.50, len(docs), colors["blue"]),
        ("strength evidence", 0.40, 0.72, strength_doc_count, colors["blue"]),
        ("shrinkage evidence", 0.40, 0.30, int(shrink_summary.get("shrinkage_doc_count", 0)), colors["red"]),
        ("flowability evidence", 0.74, 0.72, flowability_doc_count, colors["teal"]),
        ("shrinkage-KG priors", 0.74, 0.30, int(shrink_summary.get("material_pair_doc_count", 0)), colors["orange"]),
    ]
    max_flow = max(value for _, _, _, value, _ in flow_nodes) or 1

    def ribbon(ax_, p0, p1, value, color):
        t = _np.linspace(0, 1, 80)
        x = (1 - t) * p0[0] + t * p1[0]
        y = (1 - t) * p0[1] + t * p1[1] + 0.06 * _np.sin(_np.pi * t) * (1 if p1[1] >= p0[1] else -1)
        ax_.plot(x, y, color=color, alpha=0.18 + 0.28 * min(1, value / max_flow), lw=2.2 + 11.0 * math.sqrt(max(value, 1) / max_flow), solid_capstyle="round")

    ribbon(ax, (0.15, 0.50), (0.35, 0.72), strength_doc_count, colors["blue"])
    ribbon(ax, (0.15, 0.50), (0.35, 0.30), int(shrink_summary.get("shrinkage_doc_count", 0)), colors["red"])
    ribbon(ax, (0.49, 0.72), (0.70, 0.72), flowability_doc_count, colors["teal"])
    ribbon(ax, (0.49, 0.30), (0.70, 0.30), int(shrink_summary.get("material_pair_doc_count", 0)), colors["orange"])
    ribbon(ax, (0.49, 0.30), (0.70, 0.72), int(shrink_summary.get("material_pair_doc_count", 0)), colors["teal"])
    for label, x, y, value, color in flow_nodes:
        ax.scatter([x], [y], s=180 + 1100 * math.sqrt(max(value, 1) / max_flow), color=color, alpha=0.82, edgecolors="white", linewidth=1.0)
        ax.text(x, y - 0.16, f"{label}\n{value:,}", ha="center", va="top", fontsize=7.5, color="#202123")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("c  Evidence flow from corpus-scale retrieval to agent priors", loc="left", fontsize=11, fontweight="bold")

    ax = fig0.add_subplot(gs0[2, 0:2])
    if not pair_df.empty:
        bubble = pair_df[pair_df["material"].isin(materials) & pair_df["performance"].isin(perfs)].copy()
        max_count = max(float(bubble["count"].max()), 1.0)
        for _, row in bubble.iterrows():
            x = perfs.index(row["performance"])
            y = materials.index(row["material"])
            size = 14 + 430 * math.sqrt(float(row.get("count", 0)) / max_count)
            face = mpl.cm.YlGnBu(0.18 + 0.72 * math.log10(float(row.get("count", 0)) + 1) / max(heat.max().max(), 1))
            ax.scatter(x, y, s=size, color=face, edgecolors="white", linewidth=0.7)
            if row.get("kg_chain_count", 0) > 0:
                ax.scatter(x, y, s=size * 1.18, facecolors="none", edgecolors=colors["red"], linewidth=0.8)
        ax.set_xticks(range(len(perfs)))
        ax.set_xticklabels([perf_label(p) for p in perfs], rotation=30, ha="right", fontsize=7.3)
        ax.set_yticks(range(len(materials)))
        ax.set_yticklabels([mat_label(m) for m in materials], fontsize=7.2)
        ax.invert_yaxis()
        ax.set_xlim(-0.6, len(perfs) - 0.4)
        ax.set_ylim(len(materials) - 0.4, -0.6)
        ax.grid(color="#e7ebef", linewidth=0.55)
    ax.set_title("d  Material-performance evidence density with KG-covered rings", loc="left", fontsize=11, fontweight="bold")

    ax = fig0.add_subplot(gs0[2, 2:])
    ax.set_aspect("equal")
    ax.axis("off")
    if shrink_summary.get("category_count"):
        cat_map = {
            "shrinkage": "shrinkage",
            "drying_shrinkage": "drying shrinkage",
            "autogenous_shrinkage": "autogenous shrinkage",
            "internal_curing": "internal curing",
            "volume_stability": "volume stability",
            "general_shrinkage": "general shrinkage",
        }
        cats = sorted(shrink_summary["category_count"].items(), key=lambda kv: kv[1], reverse=True)
        total = sum(v for _, v in cats) or 1
        theta = 92
        palette = [colors["red"], colors["orange"], colors["purple"], colors["teal"], colors["blue"], colors["gray"]]
        for idx, (key, value) in enumerate(cats):
            span = 328 * value / total
            wedge = mpl.patches.Wedge((0, 0), 1.0, theta - span, theta, width=0.28, facecolor=palette[idx % len(palette)], alpha=0.82, edgecolor="white", linewidth=1.2)
            ax.add_patch(wedge)
            mid = _np.deg2rad(theta - span / 2)
            ax.text(1.16 * _np.cos(mid), 1.16 * _np.sin(mid), f"{cat_map.get(key, key)}\n{value:,}", ha="center", va="center", fontsize=7.1)
            theta -= span
        ax.text(0, 0.05, f"{int(shrink_summary.get('shrinkage_doc_count', 0)):,}", ha="center", va="center", fontsize=20, fontweight="bold", color=colors["red"])
        ax.text(0, -0.16, "shrinkage\nabstracts", ha="center", va="center", fontsize=8, color="#202123")
        ax.set_xlim(-1.5, 1.55)
        ax.set_ylim(-1.25, 1.25)
    ax.set_title("e  Shrinkage evidence taxonomy, not a single keyword bucket", loc="left", fontsize=11, fontweight="bold")
    fig0.suptitle("Corpus-scale evidence construction for low-carbon low-shrinkage concrete mix design", x=0.01, ha="left", fontsize=14, fontweight="bold")
    paths = _save_pub_figure(fig0, "Fig0_evidence_scale_shrinkage_panorama")
    plt.close(fig0)
    figures.append({"id": "fig0", "title": "Evidence-scale shrinkage panorama", "files": paths})

    fig = plt.figure(figsize=(13.2, 8.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.15], height_ratios=[0.9, 1.1], wspace=0.26, hspace=0.46)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    boxes = [
        (0.04, 0.62, "500 full texts\nGold KG", colors["blue"]),
        (0.39, 0.62, "40,600 abstracts\nEvidence corpus", colors["teal"]),
        (0.74, 0.62, "Agent priors\nDesign reasoning", colors["orange"]),
        (0.39, 0.18, "Evidence-calibrated KG\nconfidence · gap · risk", colors["purple"]),
    ]
    for x, y, label, color in boxes:
        ax0.add_patch(mpl.patches.FancyBboxPatch((x, y), 0.23, 0.18, boxstyle="round,pad=0.02,rounding_size=0.025", facecolor=color, alpha=0.14, edgecolor=color, linewidth=1.2))
        ax0.text(x + 0.115, y + 0.09, label, ha="center", va="center", fontsize=8.2, color="#202123")
    for start, end in [((0.27, 0.71), (0.39, 0.71)), ((0.62, 0.71), (0.74, 0.71)), ((0.51, 0.62), (0.51, 0.36))]:
        ax0.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=colors["slate"], lw=1.2))
    ax0.set_title("a  Dual-layer evidence architecture", loc="left", fontsize=10, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 1])
    sns.heatmap(heat, ax=ax1, cmap="YlGnBu", cbar_kws={"label": "log10(abstract count + 1)"}, linewidths=0.45, linecolor="white")
    for i, material in enumerate(heat.index):
        for j, perf in enumerate(heat.columns):
            if kg_cover.loc[material, perf] > 0:
                ax1.scatter(j + 0.5, i + 0.5, s=18 + kg_cover.loc[material, perf] * 12, facecolors="none", edgecolors=colors["red"], linewidths=1.0)
    ax1.set_title("b  Material-performance evidence atlas", loc="left", fontsize=10, fontweight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_xticklabels([perf_label(item) for item in heat.columns], rotation=35, ha="right", fontsize=7.2)
    ax1.set_yticklabels([mat_label(item) for item in heat.index], rotation=0, fontsize=7.2)

    ax2 = fig.add_subplot(gs[1, 0])
    if not kg_df.empty:
        kg_plot = kg_df.copy()
        kg_plot["evidence_log"] = _np.log10(kg_plot["evidence_count"].astype(float) + 1)
        sc = ax2.scatter(
            kg_plot["evidence_log"],
            kg_plot["confidence_score"],
            s=35 + kg_plot["recent_evidence_count"].clip(0, 500) / 8,
            c=kg_plot["uncertainty_score"],
            cmap="magma_r",
            alpha=0.82,
            edgecolors="white",
            linewidth=0.35,
        )
        ax2.set_xlabel("log10 supporting abstracts")
        ax2.set_ylabel("KG confidence")
        cbar = fig.colorbar(sc, ax=ax2, shrink=0.78)
        cbar.set_label("uncertainty")
    ax2.set_title("c  Evidence-calibrated Gold KG", loc="left", fontsize=10, fontweight="bold")

    ax3 = fig.add_subplot(gs[1, 1])
    gap = pair_df.copy()
    if not gap.empty:
        top_gap = gap.sort_values("gap_score", ascending=False).head(36)
        ax3.scatter(
            _np.log10(top_gap["count"].astype(float) + 1),
            top_gap["kg_chain_count"].astype(float),
            s=45 + top_gap["recent"].astype(float).clip(0, 1000) / 14,
            c=top_gap["low_carbon_share"].astype(float),
            cmap="viridis",
            alpha=0.82,
            edgecolors="white",
            linewidth=0.4,
        )
        top_gap = top_gap.reset_index(drop=True)
        for idx, row in top_gap.head(5).iterrows():
            ax3.text(
                math.log10(row["count"] + 1) + 0.025,
                row["kg_chain_count"] + 0.10 + (idx % 2) * 0.08,
                f"{idx + 1}",
                fontsize=7,
                color=colors["slate"],
                fontweight="bold",
            )
        ax3.set_xlabel("log10 literature maturity")
        ax3.set_ylabel("Gold KG chain count")
    ax3.set_title("d  High-maturity knowledge gaps", loc="left", fontsize=10, fontweight="bold")
    fig.suptitle("Evidence-calibrated knowledge graph from 40,600 abstracts", x=0.01, ha="left", fontsize=12, fontweight="bold")
    paths = _save_pub_figure(fig, "Fig1_evidence_calibrated_kg")
    plt.close(fig)
    figures.append({"id": "fig1", "title": "Evidence-calibrated KG atlas", "files": paths})

    fig2 = plt.figure(figsize=(11.2, 7.2))
    gs2 = fig2.add_gridspec(2, 2, wspace=0.3, hspace=0.34)
    ax = fig2.add_subplot(gs2[0, 0])
    top_mat = _pd.DataFrame(_top_items(payload.get("materials", []), "count", 10))
    if not top_mat.empty:
        ax.barh(top_mat["material"][::-1], top_mat["count"][::-1], color=colors["blue"])
    ax.set_title("a  Material prevalence", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("abstract count")
    ax = fig2.add_subplot(gs2[0, 1])
    if not pair_df.empty:
        focus = pair_df[pair_df["performance"].isin(["shrinkage", "compressive strength", "flowability"])].copy()
        focus["focus_count"] = focus["count"]
        focus = focus.sort_values("focus_count", ascending=False).head(14)
        labels = focus.apply(lambda row: f"{mat_label(row['material'])} / {perf_label(row['performance'])}", axis=1)
        bar_colors = [
            colors["red"] if perf == "shrinkage" else (colors["blue"] if perf == "compressive strength" else colors["teal"])
            for perf in focus["performance"]
        ]
        ax.barh(labels[::-1], focus["focus_count"][::-1], color=bar_colors[::-1], alpha=0.84)
    ax.set_title("b  Shrinkage-strength-flowability evidence concentration", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("abstract evidence count")
    ax = fig2.add_subplot(gs2[1, 0])
    if not pair_df.empty:
        risk = pair_df.sort_values("risk_share", ascending=False).head(18)
        ax.scatter(risk["maturity"], risk["risk_share"], s=60 + risk["count"].clip(0, 1000) / 8, color=colors["red"], alpha=0.72, edgecolors="white", linewidth=0.4)
        ax.set_xlabel("literature maturity")
        ax.set_ylabel("risk/contradiction share")
    ax.set_title("c  Critic-agent risk prior", loc="left", fontsize=10, fontweight="bold")
    ax = fig2.add_subplot(gs2[1, 1])
    if not prior_df.empty:
        top_prior = prior_df.sort_values("retrieval_weight", ascending=False).head(14)
        ax.barh((top_prior["material"] + " / " + top_prior["performance"])[::-1], top_prior["retrieval_weight"][::-1], color=colors["purple"])
        ax.set_xlim(0, 1)
    ax.set_title("d  Retrieval priority for mechanism agent", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("agent prior")
    fig2.suptitle("Agent priors derived from the abstract evidence layer", x=0.01, ha="left", fontsize=12, fontweight="bold")
    paths = _save_pub_figure(fig2, "Fig2_agent_priors_landscape")
    plt.close(fig2)
    figures.append({"id": "fig2", "title": "Agent prior landscape", "files": paths})

    year_keys = [str(y) for y in range(2000, 2026)]
    annual_total = {key: 0 for key in year_keys}
    annual_strength = {key: 0 for key in year_keys}
    annual_flowability = {key: 0 for key in year_keys}
    annual_shrinkage = {key: 0 for key in year_keys}
    mat_year = {material: {key: 0 for key in year_keys} for material in materials}
    for row in pairs:
        for year, count in row.get("years", {}).items():
            if year in annual_total:
                annual_total[year] += count
                if row.get("performance") == "shrinkage":
                    annual_shrinkage[year] += count
                if row.get("performance") == "compressive strength":
                    annual_strength[year] += count
                if row.get("performance") == "flowability":
                    annual_flowability[year] += count
                if row.get("material") in mat_year:
                    mat_year[row["material"]][year] += count
    fig3 = plt.figure(figsize=(11.2, 7.2))
    gs3 = fig3.add_gridspec(2, 2, wspace=0.28, hspace=0.34)
    ax = fig3.add_subplot(gs3[0, 0])
    years_num = [int(y) for y in year_keys]
    ax.plot(years_num, [annual_total[y] for y in year_keys], color=colors["blue"], linewidth=2.2)
    ax.fill_between(years_num, [annual_total[y] for y in year_keys], color=colors["blue"], alpha=0.12)
    ax.set_title("a  Annual evidence volume", loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel("material-performance mentions")
    ax = fig3.add_subplot(gs3[0, 1])
    top_trend_mats = [row["material"] for row in _top_items(payload.get("materials", []), "recent", 6)]
    for idx, material in enumerate(top_trend_mats):
        vals = [mat_year.get(material, {}).get(y, 0) for y in year_keys]
        ax.plot(years_num, vals, linewidth=1.7, label=material)
    ax.legend(fontsize=6.5, ncol=2)
    ax.set_title("b  Material evidence trajectories", loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel("mentions")
    ax = fig3.add_subplot(gs3[1, 0])
    total_vals = _np.asarray([annual_total[y] for y in year_keys], dtype=float)
    strength_vals = _np.asarray([annual_strength[y] for y in year_keys], dtype=float)
    flow_vals = _np.asarray([annual_flowability[y] for y in year_keys], dtype=float)
    shrink_vals = _np.asarray([annual_shrinkage[y] for y in year_keys], dtype=float)
    strength_share = _np.divide(strength_vals, _np.maximum(total_vals, 1))
    flow_share = _np.divide(flow_vals, _np.maximum(total_vals, 1))
    shrink_share = _np.divide(shrink_vals, _np.maximum(total_vals, 1))
    ax.plot(years_num, strength_share, color=colors["blue"], linewidth=2.0, marker="o", markersize=3, label="strength")
    ax.plot(years_num, flow_share, color=colors["teal"], linewidth=2.0, marker="^", markersize=3, label="flowability")
    ax.plot(years_num, shrink_share, color=colors["red"], linewidth=2.2, marker="s", markersize=3, label="shrinkage")
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(0, max(0.08, float(max(strength_share.max(), flow_share.max(), shrink_share.max())) * 1.18))
    ax.set_title("c  Shrinkage, strength, and flowability evidence shares", loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel("share")
    ax = fig3.add_subplot(gs3[1, 1])
    if not pair_df.empty:
        ax.scatter(pair_df["maturity"], pair_df["recent_share"], s=50 + pair_df["count"].clip(0, 1000) / 10, c=pair_df["gap_score"], cmap="inferno", alpha=0.78, edgecolors="white", linewidth=0.4)
        ax.set_xlabel("literature maturity")
        ax.set_ylabel("recent evidence share")
    ax.set_title("d  Emerging mature gaps", loc="left", fontsize=10, fontweight="bold")
    fig3.suptitle("Temporal evolution of the evidence corpus", x=0.01, ha="left", fontsize=12, fontweight="bold")
    paths = _save_pub_figure(fig3, "Fig3_temporal_evidence_evolution")
    plt.close(fig3)
    figures.append({"id": "fig3", "title": "Temporal evidence evolution", "files": paths})

    fig4 = plt.figure(figsize=(11.2, 7.2))
    gs4 = fig4.add_gridspec(2, 2, width_ratios=[1.35, 0.9], wspace=0.58, hspace=0.34)
    ax = fig4.add_subplot(gs4[:, 0])
    ax.axis("off")
    if not pair_df.empty:
        shown = pair_df.sort_values("agent_prior", ascending=False).head(24)
        mat_nodes = list(dict.fromkeys(shown["material"].tolist()))
        perf_nodes = list(dict.fromkeys(shown["performance"].tolist()))
        mat_pos = {m: (0.12, 0.92 - i * (0.82 / max(len(mat_nodes) - 1, 1))) for i, m in enumerate(mat_nodes)}
        perf_pos = {p: (0.86, 0.88 - i * (0.72 / max(len(perf_nodes) - 1, 1))) for i, p in enumerate(perf_nodes)}
        for _, row in shown.iterrows():
            x1, y1 = mat_pos[row["material"]]
            x2, y2 = perf_pos[row["performance"]]
            width = 0.4 + 2.2 * row["agent_prior"]
            color = colors["red"] if row["risk_share"] > 0.35 else colors["teal"]
            ax.plot([x1, x2], [y1, y2], color=color, alpha=0.25 + 0.45 * row["agent_prior"], linewidth=width)
        for label, (x, y) in mat_pos.items():
            ax.scatter([x], [y], s=180, color=colors["blue"], alpha=0.82, edgecolors="white", linewidth=0.6)
            ax.text(x - 0.04, y, label, ha="right", va="center", fontsize=7.2)
        for label, (x, y) in perf_pos.items():
            ax.scatter([x], [y], s=210, color=colors["orange"], alpha=0.82, edgecolors="white", linewidth=0.6)
            ax.text(x + 0.04, y, label, ha="left", va="center", fontsize=7.2)
    ax.set_title("a  Evidence-weighted material-performance network", loc="left", fontsize=10, fontweight="bold")
    ax = fig4.add_subplot(gs4[0, 1])
    if not prior_df.empty:
        prior_mat = prior_df.sort_values("critic_risk", ascending=False).head(14)
        labels = prior_mat.apply(lambda row: f"{mat_label(row['material'])}\n{perf_label(row['performance'])}", axis=1).to_numpy()
        ax.barh(labels[::-1], prior_mat["critic_risk"][::-1], color=colors["red"])
        ax.set_xlim(0, 1)
        ax.tick_params(axis="y", labelsize=6.2)
    ax.set_title("b  Critic-agent risk queue", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("risk prior")
    ax = fig4.add_subplot(gs4[1, 1])
    if not prior_df.empty:
        heat_prior = _pd.DataFrame(0.0, index=materials, columns=perfs)
        for _, row in prior_df.iterrows():
            if row["material"] in heat_prior.index and row["performance"] in heat_prior.columns:
                heat_prior.loc[row["material"], row["performance"]] = row["retrieval_weight"]
        sns.heatmap(heat_prior, ax=ax, cmap="Purples", vmin=0, vmax=1, cbar_kws={"label": "retrieval weight"}, linewidths=0.45, linecolor="white")
        ax.set_xticklabels([perf_label(item) for item in heat_prior.columns], rotation=35, ha="right", fontsize=7)
        ax.set_yticklabels([mat_label(item) for item in heat_prior.index], rotation=0, fontsize=7)
    ax.set_title("c  Mechanism-agent retrieval priors", loc="left", fontsize=10, fontweight="bold")
    fig4.suptitle("How the evidence layer enters multi-agent reasoning", x=0.01, ha="left", fontsize=12, fontweight="bold")
    paths = _save_pub_figure(fig4, "Fig4_multi_agent_evidence_priors")
    plt.close(fig4)
    figures.append({"id": "fig4", "title": "Multi-agent evidence priors", "files": paths})

    fig5 = plt.figure(figsize=(10.8, 10.8))
    ax = fig5.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")
    chord_pairs = pair_df.sort_values("agent_prior", ascending=False).head(34) if not pair_df.empty else pair_df
    chord_materials = list(dict.fromkeys(chord_pairs["material"].tolist())) if not chord_pairs.empty else []
    chord_perfs = list(dict.fromkeys(chord_pairs["performance"].tolist())) if not chord_pairs.empty else []
    left_angles = _np.linspace(125, 235, max(len(chord_materials), 1)) * _np.pi / 180
    right_angles = _np.linspace(-55, 55, max(len(chord_perfs), 1)) * _np.pi / 180
    node_pos = {}
    radius = 1.0
    for label, angle in zip(chord_materials, left_angles):
        node_pos[("material", label)] = (radius * _np.cos(angle), radius * _np.sin(angle), angle)
    for label, angle in zip(chord_perfs, right_angles):
        node_pos[("performance", label)] = (radius * _np.cos(angle), radius * _np.sin(angle), angle)

    def bezier_points(p0, p1, pull=0.34, n=80):
        p0 = _np.asarray(p0, dtype=float)
        p1 = _np.asarray(p1, dtype=float)
        c0 = p0 * pull
        c1 = p1 * pull
        t = _np.linspace(0, 1, n)[:, None]
        return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c0 + 3 * (1 - t) * t ** 2 * c1 + t ** 3 * p1

    if not chord_pairs.empty:
        for _, row in chord_pairs.iterrows():
            left = node_pos.get(("material", row["material"]))
            right = node_pos.get(("performance", row["performance"]))
            if not left or not right:
                continue
            pts = bezier_points((left[0], left[1]), (right[0], right[1]), pull=0.22)
            is_risk = row.get("risk_share", 0) > 0.32
            color = colors["red"] if is_risk else colors["teal"]
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=color,
                alpha=0.16 + 0.58 * float(row.get("agent_prior", 0)),
                linewidth=0.45 + 4.6 * float(row.get("agent_prior", 0)),
                solid_capstyle="round",
                zorder=1,
            )
    for kind, labels, fill in [
        ("material", chord_materials, colors["blue"]),
        ("performance", chord_perfs, colors["orange"]),
    ]:
        for label in labels:
            x, y, angle = node_pos[(kind, label)]
            if kind == "material":
                count = next((item.get("count", 0) for item in payload.get("materials", []) if item.get("material") == label), 0)
            else:
                count = next((item.get("count", 0) for item in payload.get("performances", []) if item.get("performance") == label), 0)
            size = 110 + 8 * math.sqrt(max(count, 0))
            ax.scatter([x], [y], s=size, color=fill, alpha=0.9, edgecolors="white", linewidth=1.2, zorder=4)
            ha = "right" if kind == "material" else "left"
            tx = x + (-0.075 if kind == "material" else 0.075)
            label_short = mat_label(label) if kind == "material" else perf_label(label)
            ax.text(tx, y, label_short, ha=ha, va="center", fontsize=8.2, color="#202123", zorder=5)
    theta = _np.linspace(0, 2 * _np.pi, 360)
    ax.plot(_np.cos(theta), _np.sin(theta), color="#d7dce2", linewidth=0.8, zorder=0)
    ax.text(-0.95, 1.12, "Materials", ha="center", va="center", fontsize=10, fontweight="bold", color=colors["blue"])
    ax.text(0.95, 1.12, "Performance targets", ha="center", va="center", fontsize=10, fontweight="bold", color=colors["orange"])
    ax.text(0, -1.18, "Chord width = retrieval prior; green = supporting evidence, red = critic risk prior", ha="center", va="center", fontsize=8.2, color=colors["slate"])
    fig5.suptitle("Evidence-flow chord diagram for multi-agent concrete design", x=0.04, y=0.98, ha="left", fontsize=13, fontweight="bold")
    paths = _save_pub_figure(fig5, "Fig5_evidence_flow_chord")
    plt.close(fig5)
    figures.append({"id": "fig5", "title": "Evidence-flow chord diagram", "files": paths})

    fig6 = plt.figure(figsize=(11.2, 7.2))
    ax = fig6.add_subplot(111)
    stream_mats = [row["material"] for row in _top_items(payload.get("materials", []), "recent", 8)]
    if stream_mats:
        y_stack = _np.asarray([[mat_year.get(material, {}).get(year, 0) for year in year_keys] for material in stream_mats], dtype=float)
        y_smooth = y_stack.copy()
        for idx in range(y_smooth.shape[0]):
            series = _pd.Series(y_smooth[idx])
            y_smooth[idx] = series.rolling(3, center=True, min_periods=1).mean().to_numpy()
        palette = [colors.get(key, fallback) for key, fallback in [
            ("blue", "#1f77b4"), ("teal", "#1b9e77"), ("orange", "#d95f02"), ("purple", "#7b6fd0"),
            ("red", "#c44e52"), ("gold", "#b8860b"), ("mint", "#66c2a5"), ("slate", "#4c566a"),
        ]]
        ax.stackplot(years_num, y_smooth, labels=[mat_label(m) for m in stream_mats], colors=palette[:len(stream_mats)], alpha=0.82, baseline="wiggle")
        ax.legend(loc="upper left", ncol=4, fontsize=7, frameon=False)
    ax.set_title("Evidence river of material attention across 40,600 abstracts", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("publication year")
    ax.set_ylabel("smoothed evidence intensity")
    ax.text(0.01, 0.02, "Stream width = material-performance evidence volume; centred baseline highlights shifts in research attention.", transform=ax.transAxes, fontsize=8, color=colors["slate"])
    paths = _save_pub_figure(fig6, "Fig6_evidence_river_streamgraph")
    plt.close(fig6)
    figures.append({"id": "fig6", "title": "Evidence river streamgraph", "files": paths})

    fig7 = plt.figure(figsize=(11.2, 7.2))
    ax = fig7.add_subplot(111)
    ax.axis("off")
    if not pair_df.empty:
        alluvial = pair_df.sort_values("agent_prior", ascending=False).head(28).copy()
        alluvial["status"] = alluvial.apply(lambda row: "critic risk" if row.get("risk_share", 0) > 0.32 else "retrieval prior" if row.get("kg_chain_count", 0) > 0 else "gap mining", axis=1)
        mat_nodes = list(dict.fromkeys(alluvial["material"].tolist()))
        status_nodes = ["retrieval prior", "critic risk", "gap mining"]
        perf_nodes = list(dict.fromkeys(alluvial["performance"].tolist()))
        x_cols = [0.08, 0.50, 0.92]
        mat_pos = {m: (x_cols[0], 0.92 - i * (0.82 / max(len(mat_nodes) - 1, 1))) for i, m in enumerate(mat_nodes)}
        status_pos = {s: (x_cols[1], 0.78 - i * 0.24) for i, s in enumerate(status_nodes)}
        perf_pos = {p: (x_cols[2], 0.78 - i * (0.56 / max(len(perf_nodes) - 1, 1))) for i, p in enumerate(perf_nodes)}

        def curve(ax_, p0, p1, color, width, alpha):
            xs = _np.linspace(p0[0], p1[0], 90)
            t = _np.linspace(0, 1, 90)
            ys = (1 - t) * p0[1] + t * p1[1] + 0.035 * _np.sin(_np.pi * t) * (1 if p1[1] >= p0[1] else -1)
            ax_.plot(xs, ys, color=color, linewidth=width, alpha=alpha, solid_capstyle="round", zorder=1)

        for _, row in alluvial.iterrows():
            p0 = mat_pos[row["material"]]
            p1 = status_pos[row["status"]]
            p2 = perf_pos[row["performance"]]
            width = 0.6 + 5.0 * float(row.get("agent_prior", 0))
            color = colors["red"] if row["status"] == "critic risk" else colors["teal"] if row["status"] == "retrieval prior" else colors["gold"]
            curve(ax, p0, p1, color, width, 0.26 + 0.5 * float(row.get("agent_prior", 0)))
            curve(ax, p1, p2, color, width, 0.22 + 0.48 * float(row.get("agent_prior", 0)))
        for node_map, fill, align in [(mat_pos, colors["blue"], "right"), (status_pos, colors["purple"], "center"), (perf_pos, colors["orange"], "left")]:
            for label, (x, y) in node_map.items():
                ax.scatter([x], [y], s=220, color=fill, edgecolors="white", linewidth=1.0, zorder=3)
                tx = x - 0.035 if align == "right" else x + 0.035 if align == "left" else x
                ha = "right" if align == "right" else "left" if align == "left" else "center"
                text = mat_label(label) if label in material_labels else perf_label(label)
                ax.text(tx, y + (0.045 if align == "center" else 0), text, ha=ha, va="center", fontsize=8, color="#202123", zorder=4)
        ax.text(x_cols[0], 1.02, "Material evidence", ha="center", fontsize=10, fontweight="bold", color=colors["blue"])
        ax.text(x_cols[1], 1.02, "Agent use", ha="center", fontsize=10, fontweight="bold", color=colors["purple"])
        ax.text(x_cols[2], 1.02, "Design target", ha="center", fontsize=10, fontweight="bold", color=colors["orange"])
    fig7.suptitle("Alluvial routing map from abstract evidence to multi-agent actions", x=0.04, ha="left", fontsize=13, fontweight="bold")
    paths = _save_pub_figure(fig7, "Fig7_agent_alluvial_routing")
    plt.close(fig7)
    figures.append({"id": "fig7", "title": "Agent alluvial routing map", "files": paths})

    fig8 = plt.figure(figsize=(10.2, 10.2))
    ax = fig8.add_subplot(111, projection="polar")
    ax.set_theta_offset(_np.pi / 2)
    ax.set_theta_direction(-1)
    if not pair_df.empty:
        radial_mats = materials[:]
        radial_perfs = perfs[:]
        theta_edges = _np.linspace(0, 2 * _np.pi, len(radial_mats) + 1)
        r_edges = _np.linspace(0.18, 1.0, len(radial_perfs) + 1)
        matrix = _np.zeros((len(radial_perfs), len(radial_mats)))
        gap_matrix = _np.zeros_like(matrix)
        for _, row in pair_df.iterrows():
            if row["material"] in radial_mats and row["performance"] in radial_perfs:
                j = radial_mats.index(row["material"])
                i = radial_perfs.index(row["performance"])
                matrix[i, j] = math.log10(float(row.get("count", 0)) + 1)
                gap_matrix[i, j] = float(row.get("gap_score", 0))
        for i in range(len(radial_perfs)):
            for j in range(len(radial_mats)):
                theta = theta_edges[j]
                width = theta_edges[j + 1] - theta_edges[j]
                bottom = r_edges[i]
                height = r_edges[i + 1] - r_edges[i]
                val = matrix[i, j] / max(matrix.max(), 1)
                face = mpl.cm.YlGnBu(0.18 + 0.76 * val)
                ax.bar(theta, height, width=width * 0.94, bottom=bottom, align="edge", color=face, edgecolor="white", linewidth=0.8)
                if gap_matrix[i, j] > 0.45:
                    ax.scatter([theta + width / 2], [bottom + height / 2], s=18 + 45 * gap_matrix[i, j], facecolors="none", edgecolors=colors["red"], linewidths=1.1, zorder=4)
        ax.set_xticks((theta_edges[:-1] + theta_edges[1:]) / 2)
        ax.set_xticklabels([mat_label(m) for m in radial_mats], fontsize=8)
        ax.set_yticks((r_edges[:-1] + r_edges[1:]) / 2)
        ax.set_yticklabels([])
        ax.grid(color="#d7dce2", linewidth=0.6, alpha=0.8)
    ax.set_title("Radial evidence-gap matrix for concrete design priors", fontsize=13, fontweight="bold", pad=28)
    ax.text(0.5, -0.10, "Rings from centre: shrinkage, strength, flowability. Cell colour = log evidence volume; red rings = high-maturity gaps.", transform=ax.transAxes, ha="center", fontsize=8.2, color=colors["slate"])
    paths = _save_pub_figure(fig8, "Fig8_radial_evidence_gap_matrix")
    plt.close(fig8)
    figures.append({"id": "fig8", "title": "Radial evidence-gap matrix", "files": paths})

    payload["figures"] = figures
    ABSTRACT_LANDSCAPE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"updated_at": datetime.now().isoformat(timespec="seconds"), "figures": figures, "summary": payload.get("summary", {})}


def _extract_pdf_text(path):
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        try:
            import pdfplumber  # type: ignore

            with pdfplumber.open(str(path)) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception:
            return ""


def _ocr_pdf_text(path, max_pages=None):
    try:
        import fitz  # type: ignore
        import numpy as _np
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        engine = RapidOCR()
        document = fitz.open(str(path))
        texts = []
        page_count = len(document) if max_pages is None else min(len(document), max_pages)
        for index in range(page_count):
            page = document.load_page(index)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
            image = _np.frombuffer(pix.samples, dtype=_np.uint8).reshape(pix.height, pix.width, pix.n)
            result, _ = engine(image)
            if result:
                texts.append("\n".join(item[1] for item in result))
        return "\n".join(texts)
    except Exception:
        return ""


WATERMARK_PATTERNS = [
    re.compile(r"土小狐\s*www\.tuxiaohu\.com\s*免费下载规范图集", re.IGNORECASE),
    re.compile(r"www\.[a-z0-9\-]+\.(?:com|cn)", re.IGNORECASE),
]


DOMAIN_KEYWORDS = {
    "self_compacting_concrete": ["自密实", "scc", "self-compacting", "self compacting"],
    "pumped_concrete": ["泵送", "泵送混凝土", "pump", "pumped"],
    "high_performance_concrete": ["高性能混凝土", "高性能", "high performance"],
    "high_strength_concrete": ["高强混凝土", "高强", "high strength"],
    "reactive_powder_concrete": ["活性粉末", "rpc", "uhpc", "ultra-high"],
    "marine_concrete": ["水运", "海工", "港口", "marine", "harbor"],
    "durability_concrete": ["耐久", "耐久性", "抗渗", "氯离子", "碳化", "冻融", "durability"],
    "mix_design_concrete": ["配合比", "配合比设计", "mix design", "proportion"],
    "fiber_concrete": ["纤维混凝土", "钢纤维混凝土"],
    "shotcrete": ["喷射混凝土"],
    "mass_concrete": ["大体积混凝土"],
    "waterproof_concrete": ["防水混凝土", "地下防水", "waterproof concrete"],
    "pervious_concrete": ["透水混凝土"],
    "lightweight_concrete": ["轻骨料混凝土", "轻骨料", "泡沫混凝土", "lightweight concrete"],
    "general_concrete": ["普通混凝土", "混凝土", "concrete"],
}


METRIC_PATTERNS = {
    "slump_flow": [r"(?:坍落扩展度|扩展度)[^。；\n]{0,48}?(\d{3,4})\s*(?:mm|毫米)"],
    "t500": [r"T\s*500[^。；\n]{0,48}?(\d+(?:\.\d+)?)\s*s"],
    "flowability": [r"流动度[^。；\n]{0,48}?(\d{3,4})\s*(?:mm|毫米)"],
}


def _clean_extracted_text(text):
    raw_lines = [re.sub(r"\s+", " ", line).strip() for line in str(text or "").splitlines()]
    counts = {}
    for line in raw_lines:
        if line:
            counts[line] = counts.get(line, 0) + 1
    cleaned = []
    for line in raw_lines:
        if not line:
            continue
        if any(pattern.search(line) for pattern in WATERMARK_PATTERNS):
            continue
        if counts.get(line, 0) >= 4 and len(line) <= 80:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _extract_pdf_tables(path):
    tables = []
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(str(path)) as pdf:
            for page_index, page in enumerate(pdf.pages, 1):
                for table in page.extract_tables() or []:
                    rows = [[str(cell or "").strip() for cell in row] for row in table if row]
                    if rows:
                        tables.append({"page": page_index, "rows": rows[:40]})
    except Exception:
        return []
    return tables


def _detect_domains(text):
    lower = str(text or "").lower()
    return [domain for domain, tokens in DOMAIN_KEYWORDS.items() if any(token.lower() in lower for token in tokens)]


def _extract_normative_requirements(text, source_title="", tables=None):
    requirements = []
    searchable = str(text or "")
    for metric, patterns in METRIC_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, searchable, flags=re.IGNORECASE):
                value = float(match.group(1))
                requirements.append({
                    "metric": metric,
                    "value": value,
                    "unit": "mm" if metric in {"slump_flow", "flowability"} else ("s" if metric == "t500" else "grade"),
                    "source": source_title,
                    "evidence": searchable[max(0, match.start() - 60):match.end() + 80],
                })
    for match in re.finditer(r"C\s*(\d{2,3})", searchable, flags=re.IGNORECASE):
        value = int(match.group(1))
        evidence = searchable[max(0, match.start() - 80):match.end() + 100]
        if not (10 <= value <= 120):
            continue
        if not re.search(r"强度等级|最低强度|不应低于|混凝土", evidence):
            continue
        requirements.append({"metric": "strength_grade", "value": float(value), "unit": "grade", "source": source_title, "evidence": evidence})
    for table in tables or []:
        for row in table.get("rows", []):
            joined = " | ".join(row)
            if "坍落扩展度" in joined or "扩展度" in joined:
                for value in re.findall(r"(\d{3,4})\s*(?:mm|毫米)?", joined):
                    number = int(value)
                    if 300 <= number <= 900:
                        requirements.append({"metric": "slump_flow", "value": float(number), "unit": "mm", "source": source_title, "evidence": joined})
            if "T500" in joined.upper():
                for value in re.findall(r"(\d+(?:\.\d+)?)\s*s", joined, flags=re.IGNORECASE):
                    requirements.append({"metric": "t500", "value": float(value), "unit": "s", "source": source_title, "evidence": joined})
    dedup = {}
    for item in requirements:
        key = (item["metric"], item["value"], item["unit"], item["source"])
        dedup[key] = item
    return list(dedup.values())[:40]


def _extract_patent_requirements(text, source_title=""):
    items = []
    for match in re.finditer(r"(权利要求\s*\d+[^。；\n]{0,220})", str(text or "")):
        items.append({"type": "claim", "source": source_title, "text": match.group(1)})
    return items[:30]


def _extract_patent_metadata(text, source_title=""):
    body = str(text or "")
    patterns = {
        "patent_no": [r"(?:专利号|申请号)[:：]?\s*([A-Z]{0,3}\d[\dA-Z.\-]+)"],
        "applicant": [r"(?:申请人|专利权人)[:：]?\s*([^\n；。]{2,80})"],
        "publication_date": [r"(?:公开日|授权公告日|公告日)[:：]?\s*([12]\d{3}[年\-/.]\d{1,2}[月\-/.]\d{1,2}日?)"],
        "country": [r"(中国|中华人民共和国|CN|US|EP|WO)"],
    }
    meta = {"title": source_title}
    for key, candidates in patterns.items():
        for pattern in candidates:
            match = re.search(pattern, body, flags=re.IGNORECASE)
            if match:
                meta[key] = match.group(1).strip()
                break
    return meta


PERFORMANCE_TERMS = {
    "compressive strength": ["compressive strength", "strength", "抗压强度", "强度"],
    "flowability": ["flowability", "workability", "fluidity", "slump", "流动性", "工作性", "坍落度", "扩展度"],
    "durability": ["durability", "chloride", "permeability", "carbonation", "sulfate", "耐久", "氯离子", "渗透", "碳化", "硫酸盐"],
    "low carbon": ["low carbon", "co2", "carbon emission", "低碳", "碳排放", "减碳"],
    "shrinkage": [
        "shrinkage", "drying shrinkage", "autogenous shrinkage", "total shrinkage",
        "restrained shrinkage", "volume stability", "shrinkage", "shrinkage",
        "shrinkage control", "internal curing", "shrinkage reducing admixture", "sra",
        "收缩", "干燥收缩", "自收缩", "体积稳定", "低收缩", "收缩", "内养护", "收缩降低剂",
    ],
}

SHRINKAGE_EVIDENCE_TERMS = PERFORMANCE_TERMS["shrinkage"]
SHRINKAGE_EVIDENCE_CATEGORIES = {
    "drying_shrinkage": ["drying shrinkage", "restrained shrinkage", "干燥收缩", "约束收缩"],
    "autogenous_shrinkage": ["autogenous shrinkage", "self-desiccation", "internal relative humidity", "自收缩", "自干燥", "内部相对湿度"],
    "shrinkage": ["shrinkage", "shrinkage", "shrinkage control", "ring test", "收缩", "低收缩"],
    "internal_curing": ["internal curing", "prewetted", "lightweight aggregate", "superabsorbent polymer", "内养护", "预湿", "轻骨料"],
    "volume_stability": ["volume stability", "deformation", "expansion", "体积稳定", "变形", "膨胀"],
}


FEATURE_HINTS = [
    ("球形颗粒/滚珠效应", ["spherical", "ball bearing", "球形", "滚珠"]),
    ("微填充与孔结构细化", ["filler", "pore refinement", "dense", "microstructure", "微填充", "孔结构", "致密"]),
    ("火山灰/潜在水硬反应", ["pozzolanic", "hydration", "c-s-h", "c-a-s-h", "火山灰", "水化", "潜在水硬"]),
    ("高比表面积与需水量", ["specific surface", "water demand", "viscosity", "高比表面积", "需水", "黏度"]),
    ("PCE 分散与空间位阻", ["superplasticizer", "pce", "dispersion", "yield stress", "减水剂", "分散", "屈服应力"]),
    ("纤维桥联与收缩控制", ["fiber", "fibres", "bridging", "shrinkage", "volume stability", "flexural", "post-shrinkage", "纤维", "桥联", "收缩", "韧性", "抗弯"]),
    ("骨料级配粒径与界面效应", ["aggregate", "gradation", "particle size", "maximum size", "sand", "interfacial", "itz", "骨料", "级配", "粒径", "界面"]),
    ("低熟料替代与碳减排", ["cement replacement", "clinker", "co2", "carbon emission", "low-carbon", "低碳", "熟料", "替代", "碳排放"]),
    ("孔隙吸水与传输通道", ["water absorption", "sorptivity", "porosity", "permeability", "transport", "吸水", "孔隙", "渗透", "传输"]),
]


def _infer_feature_and_mechanism(text):
    lower = str(text or "").lower()
    matched = [label for label, tokens in FEATURE_HINTS if any(token.lower() in lower for token in tokens)]
    if matched:
        feature = "；".join(matched[:2])
    else:
        feature = "文本共现机理线索"
    mechanism = "；".join(matched[:3]) if matched else "需人工审阅确认具体作用机理"
    return feature, mechanism


def _extract_candidate_triples(text, source_title="", source_type="", limit=30):
    body = str(text or "")
    lower = body.lower()
    candidates = []
    material_hits = []
    for material, terms in MATERIAL_TERMS.items():
        if any(term.lower() in lower or term in body for term in terms):
            material_hits.append(material)
    performance_hits = []
    for performance, terms in PERFORMANCE_TERMS.items():
        if any(term.lower() in lower or term in body for term in terms):
            performance_hits.append(performance)
    if not material_hits or not performance_hits:
        return []
    feature, mechanism = _infer_feature_and_mechanism(body)
    for material in material_hits[:8]:
        for performance in performance_hits[:5]:
            terms = MATERIAL_TERMS.get(material, []) + PERFORMANCE_TERMS.get(performance, [])
            positions = [lower.find(term.lower()) for term in terms if lower.find(term.lower()) >= 0]
            anchor = min(positions) if positions else 0
            evidence = body[max(0, anchor - 180):anchor + 420].strip()
            confidence = 0.48
            if feature != "文本共现机理线索":
                confidence += 0.18
            if source_type == "literature":
                confidence += 0.08
            if source_type == "standard":
                confidence += 0.04
            candidates.append({
                "id": _stable_evidence_id("candidate_kg", source_title, material, feature, mechanism, performance, evidence[:160]),
                "material": material,
                "feature": feature,
                "mechanism": mechanism,
                "performance": performance,
                "relation": "candidate_support",
                "confidence": round(min(confidence, 0.92), 3),
                "status": "pending",
                "source": source_title,
                "source_type": source_type,
                "evidence": evidence[:800],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            })
    dedup = {}
    for item in candidates:
        dedup[(item["material"], item["feature"], item["mechanism"], item["performance"], item["source"])] = item
    return list(dedup.values())[:limit]


def _extract_candidate_triples_llm(text, source_title="", source_type="", limit=12):
    body = str(text or "")[:5000]
    if len(body.strip()) < 200:
        return []
    system_prompt = """
你是混凝土材料知识图谱抽取器。只输出 JSON 数组，不要解释。
从给定文献摘要、专利或规范片段中抽取可审计候选三元组，结构为：
[
  {
    "material": "材料名，尽量使用 cement/flyash/GGBFS/silicafume/metakaolin/limestone/water/superplasticizer 等规范名",
    "feature": "材料特征，如球形颗粒、细度、火山灰活性、比表面积、孔结构等",
    "mechanism": "形成性能的机理，不得只写影响",
    "performance": "性能，如 compressive strength/flowability/low carbon/shrinkage/shrinkage",
    "relation": "support/risk/tradeoff/constraint",
    "confidence": 0.0-1.0,
    "evidence": "原文中的最小证据片段"
  }
]
要求：
- 必须满足 材料 -> 特征 -> 机理 -> 性能 四段完整。
- 不能把简单共现当作机理；证据不足则不要抽取。
- 每条 evidence 必须来自输入原文。
- 最多输出 12 条。
""".strip()
    try:
        result = _chat_lm_json(
            system_prompt,
            {"source_title": source_title, "source_type": source_type, "text": body},
            max_tokens=1800,
            temperature=0.1,
        )
    except Exception:
        return []
    rows = result if isinstance(result, list) else result.get("triples", []) if isinstance(result, dict) else []
    triples = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        material = str(row.get("material") or "").strip()
        feature = str(row.get("feature") or "").strip()
        mechanism = str(row.get("mechanism") or "").strip()
        performance = str(row.get("performance") or "").strip()
        evidence = str(row.get("evidence") or "").strip()
        if not all([material, feature, mechanism, performance, evidence]):
            continue
        confidence = row.get("confidence", 0.72)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.72
        triples.append({
            "id": _stable_evidence_id("llm_candidate_kg", source_title, material, feature, mechanism, performance, evidence[:160]),
            "material": material,
            "feature": feature,
            "mechanism": mechanism,
            "performance": performance,
            "relation": str(row.get("relation") or "candidate_support"),
            "confidence": round(max(0.0, min(confidence, 0.98)), 3),
            "status": "pending",
            "source": source_title,
            "source_type": source_type,
            "extractor": "llm",
            "evidence": evidence[:800],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })
    return triples


def _extract_candidate_triples_hybrid(text, source_title="", source_type="", limit=30):
    llm_triples = _extract_candidate_triples_llm(text, source_title, source_type, limit=min(limit, 12))
    rule_triples = _extract_candidate_triples(text, source_title, source_type, limit=limit)
    merged = {}
    for item in [*llm_triples, *rule_triples]:
        if "extractor" not in item:
            item = {**item, "extractor": "rule_fallback"}
        key = (item.get("material"), item.get("feature"), item.get("mechanism"), item.get("performance"), item.get("source"))
        if key not in merged or item.get("extractor") == "llm":
            merged[key] = item
    return list(merged.values())[:limit]


def _standard_metadata_from_title(title):
    text = str(title or "")
    code_match = re.search(r"((?:GB/?T?|JGJ/?T?|JTG|T/CECS)\s*[_A-Z0-9\-]+)", text, flags=re.IGNORECASE)
    year_match = re.search(r"(20\d{2})", text)
    return {
        "standard_no": code_match.group(1).replace("_", " ") if code_match else "",
        "year": year_match.group(1) if year_match else "",
    }


def _stable_evidence_id(*parts):
    raw = "||".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _kg_chain_records():
    records = []
    for item in _load_recovered_kg_chains():
        route = [
            item.get("material", ""),
            item.get("feature", ""),
            item.get("mechanism", ""),
            item.get("consequence", ""),
            item.get("performance", ""),
        ]
        route_en = [
            item.get("material", ""),
            item.get("feature_en", ""),
            item.get("mechanism_en", ""),
            item.get("consequence_en", ""),
            item.get("performance_label_en") or item.get("performance_en", ""),
        ]
        body = " -> ".join(part for part in route if part) + f"；性能关系：{item.get('relation', '')}"
        body_en = " -> ".join(part for part in route_en if part) + f"; relation: {item.get('relation_en', '')}"
        records.append({
            "id": item.get("route_id") or _stable_evidence_id("kg", *route_en, item.get("relation_en") or item.get("relation")),
            "source_type": "kg",
            "title": f"{item.get('material')} / {item.get('performance_label_en') or item.get('performance_en') or item.get('performance')}",
            "text": f"{body}\n{body_en}",
            "route": route,
            "route_en": route_en,
            "material": item.get("material"),
            "performance": item.get("performance_en") or item.get("performance"),
            "relation": item.get("relation_en") or item.get("relation"),
            "tokens": _tokenize_evidence(" ".join([body, body_en])),
            "metadata": {
                "origin": str(RECOVERED_KG_APP.name),
                "performance_label": item.get("performance"),
                "lexical_unique_evidence_count": item.get(
                    "lexical_unique_evidence_count", 0
                ),
                "semantic_high_precision_count": item.get(
                    "semantic_high_precision_count", 0
                ),
                "expanded_unique_evidence_count": item.get(
                    "expanded_unique_evidence_count", 0
                ),
                "evidence_tiers": item.get("evidence_tiers", {}),
                "semantic_evidence_examples": item.get(
                    "semantic_evidence_examples", []
                ),
            },
        })
    route_extensions = []
    for item in route_extensions:
        route = [
            item.get("material_label") or item.get("material", ""),
            item.get("feature", ""),
            item.get("mechanism", ""),
            item.get("consequence", ""),
            item.get("performance", "shrinkage"),
        ]
        body = " -> ".join(part for part in route if part) + f"; relation: {item.get('relation', '')}; validation: {item.get('validation', '')}"
        records.append({
            "id": _stable_evidence_id("low_shrinkage_route", item.get("kg_id", ""), *route),
            "source_type": "kg",
            "title": f"{item.get('material_label') or item.get('material')} / {item.get('mechanism_family') or 'shrinkage-resistance route'}",
            "text": body,
            "route": route,
            "material": item.get("material"),
            "performance": item.get("performance", "shrinkage"),
            "relation": item.get("relation"),
            "tokens": _tokenize_evidence(" ".join([
                body,
                item.get("mechanism_family", ""),
                item.get("keywords", ""),
                item.get("validation", ""),
            ])),
            "metadata": {
                "origin": "low_shrinkage_kg_routes",
                "mechanism_family": item.get("mechanism_family"),
                "confidence_score": item.get("confidence_score"),
                "uncertainty_score": item.get("uncertainty_score"),
                "evidence_count": item.get("evidence_count"),
                "validation": item.get("validation"),
            },
        })
    deduped = []
    seen = set()
    for record in records:
        key = (
            record.get("source_type"),
            record.get("material"),
            tuple(record.get("route_en") or record.get("route") or []),
            record.get("relation"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _looks_like_wos_export(text):
    head = text[:2000]
    return "FN Clarivate Analytics Web of Science" in head and "\nER" in text and "\nAB " in text


def _parse_wos_records(text):
    records = []
    current = {}
    active_field = None
    multiline_fields = {"AU", "TI", "AB"}
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line == "ER":
            if current.get("TI") or current.get("AB"):
                records.append({
                    "title": " ".join(current.get("TI", [])).strip(),
                    "authors": current.get("AU", []),
                    "journal": " ".join(current.get("SO", [])).strip(),
                    "doi": " ".join(current.get("DI", [])).strip(),
                    "year": " ".join(current.get("PY", [])).strip(),
                    "abstract": " ".join(current.get("AB", [])).strip(),
                    "ut": " ".join(current.get("UT", [])).strip(),
                })
            current = {}
            active_field = None
            continue
        match = re.match(r"^([A-Z0-9]{2})\s(.*)$", line)
        if match:
            active_field = match.group(1)
            current.setdefault(active_field, []).append(match.group(2).strip())
            continue
        if line.startswith("   ") and active_field:
            addition = line.strip()
            if active_field in multiline_fields:
                current.setdefault(active_field, []).append(addition)
            elif current.get(active_field):
                current[active_field][-1] = f"{current[active_field][-1]} {addition}".strip()
    return records


def _asset_records():
    records = []
    if not SKILL_ASSETS_DIR.exists():
        return records
    for skill_dir in sorted(path for path in SKILL_ASSETS_DIR.iterdir() if path.is_dir()):
        for path in sorted(skill_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".pdf"}:
                continue
            records.extend(_records_for_asset(skill_dir.name, path))
    return records


def _load_standard_constraint_rows():
    if not STANDARD_CONSTRAINT_JSONL.exists():
        return []
    rows = []
    try:
        for line in STANDARD_CONSTRAINT_JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    except Exception:
        return []
    return rows


def _load_contextual_standard_rows():
    if not CONTEXTUAL_STANDARD_JSONL.exists():
        return []
    rows = []
    try:
        for line in CONTEXTUAL_STANDARD_JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except Exception:
        return []
    return rows


def _load_four_target_standard_rows():
    if not FOUR_TARGET_STANDARD_CSV.exists():
        return []
    try:
        with FOUR_TARGET_STANDARD_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _constraint_agent_evidence_payload():
    def contextual_target_stats(contextual_rows):
        target_items = {}
        for row in contextual_rows:
            if row.get("standard_status") == "future_or_superseded":
                continue
            target = row.get("target") or "Unknown"
            item = target_items.setdefault(
                target,
                {
                    "target": target,
                    "source_standards": set(),
                    "text_evidence": 0,
                    "numeric_records": 0,
                    "optimizable_records": 0,
                    "reference_only_records": 0,
                },
            )
            item["source_standards"].add(row.get("source_path") or row.get("source_file") or "")
            item["text_evidence"] += 1
            if any(str(row.get(key) or "").strip() for key in ("value", "lower", "upper")):
                item["numeric_records"] += 1
            if row.get("compile_status") == "optimizable":
                item["optimizable_records"] += 1
            if row.get("compile_status") == "reference_only":
                item["reference_only_records"] += 1
        return [
            {
                **{key: value for key, value in item.items() if key != "source_standards"},
                "source_standards": len({source for source in item["source_standards"] if source}),
            }
            for item in target_items.values()
        ]

    if CONSTRAINT_AGENT_PAYLOAD.exists():
        try:
            payload = json.loads(CONSTRAINT_AGENT_PAYLOAD.read_text(encoding="utf-8"))
            contextual_rows = _load_contextual_standard_rows()
            contextual_targets = contextual_target_stats(contextual_rows)
            payload["schema_version"] = 2
            payload["contextual_rules"] = {
                "rule_count": len(contextual_rows),
                "current_or_unspecified_rules": sum(
                    1 for row in contextual_rows if row.get("standard_status") != "future_or_superseded"
                ),
                "source_standards": len({row.get("source_path") for row in contextual_rows if row.get("source_path")}),
                "concrete_types": len({row.get("concrete_type") for row in contextual_rows if row.get("concrete_type")}),
                "conditional_rules": sum(1 for row in contextual_rows if row.get("condition_metric")),
                "targets": contextual_targets,
            }
            payload["targets_contextual"] = contextual_targets
            existing_targets = {item.get("target"): item for item in payload.get("targets", [])}
            for item in contextual_targets:
                if item.get("target") not in existing_targets:
                    payload.setdefault("targets", []).append(item)
            return payload
        except Exception:
            pass
    rows = _load_four_target_standard_rows()
    targets = {}
    country_groups = {}
    for row in rows:
        target = row.get("target") or "Unknown"
        group = row.get("country_group") or row.get("jurisdiction") or "Other"
        target_item = targets.setdefault(target, {"target": target, "text_evidence": 0, "numeric_records": 0, "sources": set()})
        target_item["text_evidence"] += 1
        if str(row.get("numeric_text") or "").strip():
            target_item["numeric_records"] += 1
        target_item["sources"].add(row.get("source_path") or row.get("source_file") or "")
        country_groups[group] = country_groups.get(group, 0) + 1
    contextual_rows = _load_contextual_standard_rows()
    contextual_targets = contextual_target_stats(contextual_rows)
    return {
        "schema_version": 2,
        "agent_role": "standard_constraint_evidence",
        "standard_count": len({row.get("source_path") for row in rows if row.get("source_path")}),
        "text_evidence_count": len(rows),
        "numeric_record_count": sum(1 for row in rows if str(row.get("numeric_text") or "").strip()),
        "country_groups": [{"country_group": key, "standards": value} for key, value in country_groups.items()],
        "targets": [
            {
                "target": item["target"],
                "source_standards": len(item["sources"]),
                "text_evidence": item["text_evidence"],
                "numeric_records": item["numeric_records"],
            }
            for item in targets.values()
        ],
        "group_target": [],
        "contextual_rules": {
            "rule_count": len(contextual_rows),
            "source_standards": len({row.get("source_path") for row in contextual_rows if row.get("source_path")}),
            "concrete_types": len({row.get("concrete_type") for row in contextual_rows if row.get("concrete_type")}),
            "conditional_rules": sum(1 for row in contextual_rows if row.get("condition_metric")),
            "targets": contextual_targets,
        },
        "targets_contextual": contextual_targets,
    }


def _standard_constraint_summary():
    if STANDARD_CONSTRAINT_SUMMARY.exists():
        try:
            return json.loads(STANDARD_CONSTRAINT_SUMMARY.read_text(encoding="utf-8"))
        except Exception:
            pass
    rows = _load_standard_constraint_rows()
    return {
        "source": str(STANDARD_SOURCE_DIR),
        "source_pdf_count": len(list(STANDARD_SOURCE_DIR.rglob("*.pdf"))) if STANDARD_SOURCE_DIR.exists() else 0,
        "processed_pdf_count": len({row.get("source_path") for row in rows}),
        "constraint_count": len(rows),
        "optimizable_count": sum(1 for row in rows if row.get("compile_status") == "optimizable"),
        "metrics": {},
        "domains": {},
    }


def _standard_constraint_visualization_payload(rows=None):
    rows = list(rows if rows is not None else _load_standard_constraint_rows())
    summary = _standard_constraint_summary()
    evidence_counts = _load_evidence_index().get("counts", {})
    abstract_status = _abstract_corpus_status()
    abstract_meta = abstract_status.get("corpus", {}) or {}
    shrinkage_meta = abstract_status.get("shrinkage_evidence", {}) or {}
    source_pdf_count = summary.get("source_pdf_count")
    if source_pdf_count is None:
        source_pdf_count = len(list(STANDARD_SOURCE_DIR.rglob("*.pdf"))) if STANDARD_SOURCE_DIR.exists() else 0
    metrics = {}
    domains = {}
    matrix = {}
    status_counts = {}
    optimizable_rows = []
    for row in rows:
        metric = row.get("metric") or "unknown"
        domain = _normalize_domain_name(row.get("domain") or "unknown")
        status = row.get("compile_status") or "unknown"
        metrics[metric] = metrics.get(metric, 0) + 1
        domains[domain] = domains.get(domain, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
        matrix_key = f"{domain}::{metric}"
        matrix[matrix_key] = matrix.get(matrix_key, 0) + 1
        if status == "optimizable":
            optimizable_rows.append(row)

    return {
        "source": summary.get("source") or str(STANDARD_SOURCE_DIR),
        "updated_at": summary.get("updated_at"),
        "source_pdf_count": source_pdf_count,
        "processed_pdf_count": summary.get("processed_pdf_count", len({row.get("source_path") for row in rows})),
        "constraint_count": len(rows),
        "optimizable_count": sum(1 for row in rows if row.get("compile_status") == "optimizable"),
        "evidence_counts": evidence_counts,
        "abstract_corpus": {
            "doc_count": abstract_meta.get("doc_count"),
            "with_abstract": abstract_meta.get("with_abstract"),
            "source_filename": abstract_meta.get("source_filename"),
            "year_min": abstract_meta.get("year_min"),
            "year_max": abstract_meta.get("year_max"),
            "shrinkage_doc_count": shrinkage_meta.get("shrinkage_doc_count"),
        },
        "metrics": metrics,
        "domains": domains,
        "compile_status": status_counts,
        "matrix": [
            {"domain": key.split("::", 1)[0], "metric": key.split("::", 1)[1], "count": value}
            for key, value in sorted(matrix.items())
        ],
        "optimizable_rows": optimizable_rows[:24],
        "constraint_agent": _constraint_agent_evidence_payload(),
    }


def _standard_constraint_text(row):
    bits = [
        row.get("source_file"),
        f"page {row.get('page')}" if row.get("page") else "",
        row.get("domain"),
        row.get("metric"),
        row.get("operator"),
        row.get("unit"),
        row.get("evidence"),
    ]
    return "；".join(str(item) for item in bits if item)


def _standard_constraint_records():
    records = []
    for index, row in enumerate(_load_standard_constraint_rows()):
        text = _standard_constraint_text(row)
        source_file = row.get("source_file") or "standard"
        records.append({
            "id": _stable_evidence_id("standard_constraint", source_file, row.get("page"), row.get("domain"), row.get("metric"), row.get("operator"), row.get("lower"), row.get("upper"), row.get("value"), row.get("evidence", "")[:160]),
            "source_type": "standard",
            "title": f"{source_file} | {row.get('metric')} | {row.get('domain')}",
            "text": text,
            "skill_id": "standard_constraints",
            "path": row.get("source_path") or "",
            "chunk_index": index,
            "tokens": _tokenize_evidence(text),
            "metadata": {
                "filename": source_file,
                "page": row.get("page"),
                "domain": row.get("domain"),
                "metric": row.get("metric"),
                "operator": row.get("operator"),
                "lower": row.get("lower"),
                "upper": row.get("upper"),
                "value": row.get("value"),
                "unit": row.get("unit"),
                "compile_status": row.get("compile_status"),
                "standard_status": row.get("standard_status"),
                "confidence": row.get("confidence"),
                "extraction_method": row.get("extraction_method"),
                "standard_constraint": True,
            },
        })
    return records


def _four_target_standard_records():
    records = []
    for index, row in enumerate(_load_four_target_standard_rows()):
        evidence = str(row.get("evidence") or "").strip()
        if not evidence:
            continue
        source_file = row.get("source_file") or "standard"
        target = row.get("target") or "standard constraint"
        metric = row.get("metric") or target
        numeric_text = str(row.get("numeric_text") or "").strip()
        text = "；".join(
            str(item)
            for item in [
                source_file,
                f"page {row.get('page')}" if row.get("page") else "",
                target,
                metric,
                numeric_text,
                evidence,
            ]
            if item
        )
        records.append({
            "id": _stable_evidence_id(
                "four_target_standard",
                source_file,
                row.get("page"),
                target,
                metric,
                evidence[:180],
            ),
            "source_type": "standard",
            "title": f"{source_file} | {target} | {metric}",
            "text": text,
            "skill_id": "standard_constraints",
            "path": row.get("source_path") or "",
            "chunk_index": index,
            "tokens": _tokenize_evidence(text),
            "metadata": {
                "filename": source_file,
                "page": row.get("page"),
                "target": target,
                "metric": metric,
                "requirement_type": row.get("requirement_type"),
                "operator": row.get("operator"),
                "lower": row.get("lower"),
                "upper": row.get("upper"),
                "value": row.get("value"),
                "unit": row.get("unit"),
                "numeric_text": numeric_text,
                "compile_status": row.get("compile_status"),
                "confidence": row.get("confidence"),
                "jurisdiction": row.get("jurisdiction"),
                "organization": row.get("organization"),
                "standard_constraint": True,
                "four_target_evidence": True,
            },
        })
    return records


def _contextual_standard_records():
    records = []
    for index, row in enumerate(_load_contextual_standard_rows()):
        evidence = str(row.get("evidence") or "").strip()
        if not evidence:
            continue
        source_file = row.get("source_file") or "standard"
        concrete_type = row.get("concrete_type") or "general_concrete"
        metric = row.get("metric") or row.get("target") or "constraint"
        condition = " ".join(
            str(item)
            for item in [
                row.get("condition_metric"),
                row.get("condition_operator"),
                row.get("condition_value"),
                row.get("condition_unit"),
            ]
            if item not in {None, ""}
        )
        text = "；".join(
            str(item)
            for item in [
                source_file,
                f"page {row.get('page')}" if row.get("page") else "",
                row.get("clause"),
                concrete_type,
                row.get("target"),
                metric,
                condition,
                evidence,
            ]
            if item
        )
        records.append({
            "id": _stable_evidence_id(
                "contextual_standard",
                source_file,
                row.get("page"),
                row.get("clause"),
                concrete_type,
                metric,
                row.get("operator"),
                row.get("lower"),
                row.get("upper"),
            ),
            "source_type": "standard",
            "title": f"{source_file} | {concrete_type} | {metric}",
            "text": text,
            "skill_id": "standard_constraints",
            "path": row.get("source_path") or "",
            "chunk_index": index,
            "tokens": _tokenize_evidence(text),
            "metadata": {
                "filename": source_file,
                "page": row.get("page"),
                "clause": row.get("clause"),
                "concrete_type": concrete_type,
                "required_contexts": row.get("required_contexts") or [],
                "target": row.get("target"),
                "metric": metric,
                "operator": row.get("operator"),
                "lower": row.get("lower"),
                "upper": row.get("upper"),
                "value": row.get("value"),
                "unit": row.get("unit"),
                "test_age_days": row.get("test_age_days"),
                "condition_metric": row.get("condition_metric"),
                "condition_operator": row.get("condition_operator"),
                "condition_value": row.get("condition_value"),
                "condition_unit": row.get("condition_unit"),
                "compile_status": row.get("compile_status"),
                "standard_status": row.get("standard_status"),
                "confidence": row.get("confidence"),
                "standard_constraint": True,
                "contextual_standard_constraint": True,
            },
        })
    return records


def _run_standard_constraint_extraction(source_dir=None, mode="precision", all_pdfs=True, limit=0):
    from standard_constraint_extractor import extract_constraints_from_pdf, likely_relevant_pdf
    from dataclasses import asdict

    source = Path(source_dir or STANDARD_SOURCE_DIR)
    if not source.exists():
        raise FileNotFoundError(f"规范 PDF 文件夹不存在：{source}")
    STANDARD_CONSTRAINT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(source.rglob("*.pdf"))
    if not all_pdfs:
        pdfs = [path for path in pdfs if likely_relevant_pdf(path)]
    if limit:
        pdfs = pdfs[: max(0, int(limit))]
    recall_mode = str(mode or "precision").lower() in {"recall", "broad", "扩展召回"}
    rows = []
    failures = []
    for pdf in pdfs:
        try:
            rows.extend(asdict(row) for row in extract_constraints_from_pdf(pdf, recall_mode=recall_mode))
        except Exception as exc:
            failures.append({"path": str(pdf), "error": str(exc)})
    with STANDARD_CONSTRAINT_JSONL.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = list(rows[0].keys()) if rows else [
        "source_file", "source_path", "page", "clause", "domain", "metric", "operator",
        "lower", "upper", "value", "unit", "evidence", "confidence", "extraction_method", "compile_status",
    ]
    with STANDARD_CONSTRAINT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": str(source),
        "mode": "recall" if recall_mode else "precision",
        "all_pdfs": bool(all_pdfs),
        "processed_pdf_count": len(pdfs),
        "failed_pdf_count": len(failures),
        "constraint_count": len(rows),
        "optimizable_count": sum(1 for row in rows if row.get("compile_status") == "optimizable"),
        "metrics": {},
        "domains": {},
        "outputs": {"jsonl": str(STANDARD_CONSTRAINT_JSONL), "csv": str(STANDARD_CONSTRAINT_CSV)},
        "failures": failures[:20],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    for row in rows:
        summary["metrics"][row.get("metric")] = summary["metrics"].get(row.get("metric"), 0) + 1
        summary["domains"][row.get("domain")] = summary["domains"].get(row.get("domain"), 0) + 1
    STANDARD_CONSTRAINT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _rebuild_evidence_index()
    return summary


def _records_for_asset(skill_id, path):
    records = []
    extraction = _asset_extraction(path)
    source_type = extraction["summary"]["source_type"]
    if source_type == "literature_wos":
        for index, article in enumerate(extraction["summary"].get("literature_records", [])):
            abstract = article.get("abstract", "").strip()
            body = "\n".join(part for part in [article.get("title", ""), article.get("journal", ""), abstract] if part)
            if not body:
                continue
            records.append({
                "id": _stable_evidence_id(skill_id, path.name, article.get("ut") or index, body[:120]),
                "source_type": "literature",
                "title": article.get("title") or path.stem,
                "text": body,
                "skill_id": skill_id,
                "path": str(path.relative_to(ROOT)),
                "chunk_index": index,
                "tokens": _tokenize_evidence(body),
                "metadata": {
                    "filename": path.name,
                    "authors": article.get("authors", []),
                    "journal": article.get("journal"),
                    "year": article.get("year"),
                    "doi": article.get("doi"),
                    "ut": article.get("ut"),
                    "record_type": "wos_article",
                },
            })
        return records
    text = extraction["text"]
    if not text.strip():
        text = f"{path.name} 已上传，但当前未提取到可索引正文。"
    for index, chunk in enumerate(_chunk_text(text)):
        records.append({
            "id": _stable_evidence_id(skill_id, path.name, index, chunk[:120]),
            "source_type": source_type,
            "title": path.stem,
            "text": chunk,
            "skill_id": skill_id,
            "path": str(path.relative_to(ROOT)),
            "chunk_index": index,
            "tokens": _tokenize_evidence(chunk),
            "metadata": {
                "filename": path.name,
                **(_standard_metadata_from_title(path.stem) if source_type == "standard" else {}),
                **(extraction["summary"].get("patent_metadata", {}) if source_type == "patent" else {}),
            },
        })
    return records


def _load_kg_candidates():
    if not KG_CANDIDATES_PATH.exists():
        return {"updated_at": None, "candidates": []}
    try:
        return json.loads(KG_CANDIDATES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"updated_at": None, "candidates": []}


def _save_kg_candidates(payload):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    KG_CANDIDATES_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _upsert_candidate_triples(skill_id, path, triples):
    payload = _load_kg_candidates()
    asset_path = str(Path(path).resolve().relative_to(ROOT))
    existing = [item for item in payload.get("candidates", []) if item.get("path") != asset_path]
    for item in triples or []:
        existing.append({**item, "skill_id": skill_id, "path": asset_path})
    payload["candidates"] = existing
    return _save_kg_candidates(payload)


def _candidate_kg_records():
    records = []
    for item in _load_kg_candidates().get("candidates", []):
        if item.get("status") != "approved":
            continue
        route = [item.get("material"), item.get("feature"), item.get("mechanism"), item.get("performance")]
        body = " -> ".join(part for part in route if part) + f"；证据：{item.get('evidence', '')}"
        records.append({
            "id": item["id"],
            "source_type": "auto_kg",
            "title": f"{item.get('material')} / {item.get('performance')} (audited)",
            "text": body,
            "route": route,
            "material": item.get("material"),
            "performance": item.get("performance"),
            "relation": item.get("relation", "candidate_support"),
            "skill_id": item.get("skill_id"),
            "path": item.get("path"),
            "tokens": _tokenize_evidence(body),
            "metadata": {"origin": item.get("source"), "audited": True, "confidence": item.get("confidence"), "extractor": item.get("extractor", "unknown")},
        })
    return records


def _refresh_candidate_kg_index():
    index = _load_evidence_index()
    records = [
        item for item in index.get("records", [])
        if item.get("source_type") != "auto_kg" and not (item.get("source_type") == "kg" and item.get("metadata", {}).get("audited"))
    ]
    records.extend(_candidate_kg_records())
    index["records"] = records
    index["updated_at"] = datetime.now().isoformat(timespec="seconds")
    index["counts"] = {
        "literature": sum(1 for item in records if item["source_type"] == "literature"),
        "standard": sum(1 for item in records if item["source_type"] == "standard"),
        "patent": sum(1 for item in records if item["source_type"] == "patent"),
        "kg": sum(1 for item in records if item["source_type"] == "kg"),
        "auto_kg": sum(1 for item in records if item["source_type"] == "auto_kg"),
    }
    EVIDENCE_INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def _upsert_asset_records(skill_id, path):
    index = _load_evidence_index()
    path = Path(path).resolve()
    asset_path = str(path.relative_to(ROOT))
    records = [item for item in index.get("records", []) if item.get("path") != asset_path]
    records.extend(_records_for_asset(skill_id, path))
    extraction = _asset_extraction(path)
    _upsert_candidate_triples(skill_id, path, extraction["summary"].get("candidate_triples", []))
    index["records"] = records
    index["updated_at"] = datetime.now().isoformat(timespec="seconds")
    index["counts"] = {
        "literature": sum(1 for item in records if item["source_type"] == "literature"),
        "standard": sum(1 for item in records if item["source_type"] == "standard"),
        "patent": sum(1 for item in records if item["source_type"] == "patent"),
        "kg": sum(1 for item in records if item["source_type"] == "kg"),
        "auto_kg": sum(1 for item in records if item["source_type"] == "auto_kg"),
    }
    EVIDENCE_INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def _remove_asset_records(path):
    index = _load_evidence_index()
    path = Path(path).resolve()
    asset_path = str(path.relative_to(ROOT))
    candidate_payload = _load_kg_candidates()
    candidate_payload["candidates"] = [item for item in candidate_payload.get("candidates", []) if item.get("path") != asset_path]
    _save_kg_candidates(candidate_payload)
    records = [item for item in index.get("records", []) if item.get("path") != asset_path]
    index["records"] = records
    index["updated_at"] = datetime.now().isoformat(timespec="seconds")
    index["counts"] = {
        "literature": sum(1 for item in records if item["source_type"] == "literature"),
        "standard": sum(1 for item in records if item["source_type"] == "standard"),
        "patent": sum(1 for item in records if item["source_type"] == "patent"),
        "kg": sum(1 for item in records if item["source_type"] == "kg"),
        "auto_kg": sum(1 for item in records if item["source_type"] == "auto_kg"),
    }
    EVIDENCE_INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def _asset_extraction(path, full_ocr=False):
    EVIDENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    text_cache = EVIDENCE_CACHE_DIR / f"{content_hash}.txt"
    summary_cache = EVIDENCE_CACHE_DIR / f"{content_hash}.json"
    if text_cache.exists() and summary_cache.exists():
        cached = json.loads(summary_cache.read_text(encoding="utf-8"))
        if cached.get("cache_version") == ASSET_CACHE_VERSION and (not full_ocr or cached.get("processing_complete")):
            return {
                "text": text_cache.read_text(encoding="utf-8", errors="ignore"),
                "summary": cached,
            }
    suffix = path.suffix.lower()
    lower_name = path.name.lower()
    if "专利" in path.name or "patent" in lower_name:
        source_type = "patent"
    elif suffix == ".pdf":
        source_type = "standard"
    else:
        source_type = "literature"
    if suffix == ".pdf":
        text = _clean_extracted_text(_extract_pdf_text(path))
        extraction_mode = "embedded_text"
        if len(text) < 1200:
            ocr_text = _clean_extracted_text(_ocr_pdf_text(path, max_pages=None if full_ocr else 3))
            if len(ocr_text) > len(text):
                text = ocr_text
                extraction_mode = "ocr_full" if full_ocr else "ocr_preview"
    else:
        raw_text = path.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff")
        if _looks_like_wos_export(raw_text):
            source_type = "literature_wos"
            text = raw_text.replace("\r\n", "\n")
            extraction_mode = "wos_export"
        else:
            text = _clean_extracted_text(raw_text)
            extraction_mode = "plain_text"
    chunks = _chunk_text(text)
    tables = _extract_pdf_tables(path) if suffix == ".pdf" else []
    domains = _detect_domains(f"{path.name}\n{text[:6000]}")
    normative_requirements = _extract_normative_requirements(text, path.stem, tables)
    patent_requirements = _extract_patent_requirements(text, path.stem) if source_type == "patent" else []
    patent_metadata = _extract_patent_metadata(text, path.stem) if source_type == "patent" else {}
    literature_records = _parse_wos_records(text) if source_type == "literature_wos" else []
    derived_triples = [
        {"subject": domain, "predicate": item["metric"], "object": f"{item['value']} {item['unit']}", "source": path.stem}
        for domain in domains
        for item in normative_requirements[:12]
    ]
    if literature_records:
        candidate_triples = []
        for article_index, article in enumerate(literature_records[:120]):
            source_title = article.get("title") or path.stem
            article_text = f"{article.get('title', '')}\n{article.get('abstract', '')}"
            if article_index < 12:
                candidate_triples.extend(_extract_candidate_triples_hybrid(article_text, source_title, "literature", limit=4))
            else:
                candidate_triples.extend(_extract_candidate_triples(article_text, source_title, "literature", limit=4))
        dedup_candidates = {}
        for item in candidate_triples:
            dedup_candidates[(item["material"], item["feature"], item["mechanism"], item["performance"], item["source"])] = item
        candidate_triples = list(dedup_candidates.values())[:80]
    else:
        candidate_triples = _extract_candidate_triples_hybrid(text[:20000], path.stem, source_type)
    summary = {
        "cache_version": ASSET_CACHE_VERSION,
        "processing_complete": suffix != ".pdf" or extraction_mode != "ocr_preview",
        "source_type": source_type,
        "extraction_mode": extraction_mode,
        "char_count": len(text),
        "chunk_count": len(chunks),
        "excerpt": (chunks[0] if chunks else text)[:1200],
        "tables": tables[:8],
        "table_count": len(tables),
        "domains": domains,
        "normative_requirements": normative_requirements[:20],
        "patent_requirements": patent_requirements,
        "patent_metadata": patent_metadata,
        "literature_records": literature_records[:5000],
        "literature_record_count": len(literature_records),
        "derived_triples": derived_triples,
        "candidate_triples": candidate_triples,
        "stages": [
            {"label": "接收文件", "detail": path.name},
            {"label": "读取正文", "detail": f"提取 {len(text)} 个字符"},
            {"label": "语义切块", "detail": f"生成 {len(chunks)} 个文本块"},
            {"label": "写入索引", "detail": f"归类为 {source_type}"},
            {"label": "可被检索", "detail": "进入统一证据索引"},
        ],
    }
    if source_type == "literature_wos":
        summary["stages"].insert(2, {"label": "解析记录", "detail": f"识别 {len(literature_records)} 篇 WoS 文献"})
    text_cache.write_text(text, encoding="utf-8")
    summary_cache.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    return {"text": text, "summary": summary}


def _asset_processing_summary(path):
    return _asset_extraction(path)["summary"]


def _run_document_job(job_id, path):
    job = DOCUMENT_JOBS[job_id]
    try:
        job.update({"status": "running", "stage": "ocr_and_layout", "progress": 20})
        extraction = _asset_extraction(path, full_ocr=True)
        job.update({"stage": "indexing", "progress": 80, "summary": extraction["summary"]})
        _upsert_asset_records(path.parent.name, path)
        job.update({"status": "completed", "stage": "completed", "progress": 100})
    except Exception as exc:
        job.update({"status": "failed", "stage": "failed", "error": str(exc)})


def _enqueue_document_job(path):
    job_id = _stable_evidence_id("job", path, datetime.now().isoformat())
    DOCUMENT_JOBS[job_id] = {
        "id": job_id,
        "asset_name": path.name,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(asyncio.to_thread(_run_document_job, job_id, path))
    return DOCUMENT_JOBS[job_id]


def _asset_text_window(path, offset=0, limit=4000):
    extraction = _asset_extraction(path)
    text = extraction["text"]
    offset = max(0, int(offset or 0))
    limit = max(400, min(int(limit or 4000), 12000))
    return {
        "offset": offset,
        "limit": limit,
        "total": len(text),
        "text": text[offset:offset + limit],
        "has_more": offset + limit < len(text),
    }


def _available_asset_path(asset_dir, preferred_name, content_hash):
    candidate = asset_dir / preferred_name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    return asset_dir / f"{stem}__{content_hash[:10]}{suffix}"


def _rebuild_evidence_index():
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    records = (
        _kg_chain_records()
        + _candidate_kg_records()
        + _asset_records()
        + _standard_constraint_records()
        + _four_target_standard_records()
        + _contextual_standard_records()
    )
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "records": records,
        "counts": {
            "literature": sum(1 for item in records if item["source_type"] == "literature"),
            "standard": sum(1 for item in records if item["source_type"] == "standard"),
            "patent": sum(1 for item in records if item["source_type"] == "patent"),
            "kg": sum(1 for item in records if item["source_type"] == "kg"),
            "auto_kg": sum(1 for item in records if item["source_type"] == "auto_kg"),
        },
    }
    EVIDENCE_INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _load_evidence_index():
    if not EVIDENCE_INDEX_PATH.exists():
        return _rebuild_evidence_index()
    try:
        return json.loads(EVIDENCE_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _rebuild_evidence_index()


def _evidence_score(record, query_tokens, query_terms):
    tokens = record.get("tokens", [])
    if not tokens:
        return 0.0, {}
    tf = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    lexical = sum(1 + math.log(tf[token]) for token in set(query_tokens) if token in tf)
    exact = sum(2.0 for term in query_terms if term and term in str(record.get("text", "")).lower())
    source_bonus = {"standard": 1.5, "literature": 1.0, "kg": 1.2, "auto_kg": 0.95, "patent": 0.9}.get(record.get("source_type"), 0.5)
    route_bonus = 0.0
    if record.get("source_type") in {"kg", "auto_kg"}:
        route_text = " ".join(record.get("route", [])).lower()
        route_bonus = sum(1.2 for term in query_terms if term in route_text)
    total = lexical + exact + source_bonus + route_bonus
    return total, {"lexical": round(lexical, 3), "exact": round(exact, 3), "source_bonus": source_bonus, "route_bonus": round(route_bonus, 3)}


def _grade_evidence(record, score):
    source = record.get("source_type")
    if source == "standard":
        tier = "A"
    elif source == "literature":
        tier = "B"
    else:
        tier = "B+"
    if score >= 8:
        confidence = "high"
    elif score >= 4:
        confidence = "medium"
    else:
        confidence = "exploratory"
    return {"tier": tier, "confidence": confidence}


def _hybrid_retrieve(requirements, limit=18):
    index = _load_evidence_index()
    query_terms = sorted(_kg_query_terms(requirements) | set(_tokenize_evidence(requirements.get("raw_request", ""))))
    query_tokens = _tokenize_evidence(" ".join(query_terms))
    ranked = []
    for record in index.get("records", []):
        score, components = _evidence_score(record, query_tokens, query_terms)
        if score <= 0:
            continue
        ranked.append({
            **record,
            "score": round(score, 4),
            "score_components": components,
            "grade": _grade_evidence(record, score),
        })
    ranked.sort(key=lambda item: item["score"], reverse=True)
    top = _balanced_evidence_selection(ranked, limit)
    kg_routes = [item for item in ranked if item.get("source_type") in {"kg", "auto_kg"}][:5]
    multi_hop_routes = []
    seen_routes = set()
    for item in kg_routes:
        route = tuple(item.get("route", []))
        if route and route not in seen_routes:
            seen_routes.add(route)
            multi_hop_routes.append({
                "evidence_id": item["id"],
                "route": item["route"],
                "relation": item.get("relation"),
                "score": item["score"],
            })
    return {
        "query_terms": query_terms,
        "candidates": ranked[:20],
        "selected": top,
        "multi_hop_routes": multi_hop_routes,
        "counts": index.get("counts", {}),
        "updated_at": index.get("updated_at"),
    }


def _role_query_terms(requirements, skill_id):
    terms = set(_kg_query_terms(requirements))
    skill = _read_skill(skill_id) or {}
    policy = skill.get("retrieval_policy", {})
    for item in policy.get("query_focus", []):
        terms.update(_tokenize_evidence(item))
    role_terms = {
        "requirement_analysis": ["规范", "术语", "要求", "强度等级", "坍落扩展度", "standard", "requirement"],
        "optimization_modeling": ["水胶比", "浆骨比", "配合比", "目标函数", "约束", "bolomey", "water binder", "paste aggregate", "optimization"],
        "mechanism_retrieval": ["机理", "微结构", "水化", "流动性", "收缩", "自收缩", "干燥收缩", "强度", "mechanism", "microstructure", "shrinkage", "flowability", "compressive strength"],
        "report_generation": ["验证", "实验", "风险", "结论", "引用", "收缩", "强度", "流动性", "evidence", "validation", "shrinkage", "strength", "flowability"],
    }
    terms.update(role_terms.get(skill_id, []))
    return {str(term).lower() for term in terms if term}


def _retrieve_for_skill(skill_id, requirements, limit=12):
    skill = _read_skill(skill_id) or {}
    policy = skill.get("retrieval_policy", {})
    primary_sources = list(policy.get("primary_sources", []))
    normalized_sources = ["literature" if item == "literature_wos" else item for item in primary_sources]
    if "kg" in normalized_sources and "auto_kg" not in normalized_sources:
        normalized_sources.append("auto_kg")
    index = _load_evidence_index()
    query_terms = sorted(_role_query_terms(requirements, skill_id) | set(_tokenize_evidence(requirements.get("raw_request", ""))))
    query_tokens = _tokenize_evidence(" ".join(query_terms))
    ranked = []
    for record in index.get("records", []):
        if normalized_sources and record.get("source_type") not in normalized_sources:
            continue
        score, components = _evidence_score(record, query_tokens, query_terms)
        if score <= 0:
            continue
        if record.get("source_type") in normalized_sources:
            score += 2.0
        ranked.append({
            **record,
            "score": round(score, 4),
            "score_components": components,
            "grade": _grade_evidence(record, score),
            "role": skill_id,
        })
    ranked.sort(key=lambda item: item["score"], reverse=True)
    selected = _balanced_evidence_selection(ranked, limit)
    routes = []
    seen = set()
    for item in ranked:
        if item.get("source_type") not in {"kg", "auto_kg"}:
            continue
        route = tuple(item.get("route", []))
        if route and route not in seen:
            seen.add(route)
            routes.append({"evidence_id": item["id"], "route": item["route"], "relation": item.get("relation"), "score": item["score"]})
        if len(routes) >= 5:
            break
    return {
        "role": skill_id,
        "policy": policy,
        "query_terms": query_terms,
        "candidates": ranked[:20],
        "selected": selected,
        "multi_hop_routes": routes,
        "counts": index.get("counts", {}),
        "updated_at": index.get("updated_at"),
    }


def _citations_from_evidence(evidence):
    citations = []
    for index, item in enumerate(evidence.get("selected", []), 1):
        citation = {
            "ref_no": index,
            "id": item["id"],
            "title": item["title"],
            "source_type": item["source_type"],
            "score": item.get("score"),
            "grade": item.get("grade"),
            "excerpt": item.get("text", "")[:320],
            "route": item.get("route", []),
            "metadata": item.get("metadata", {}),
            "role": evidence.get("role"),
        }
        citation["reference"] = _format_reference_entry(citation)
        citations.append(citation)
    return citations


def _role_evidence_bundle(requirements):
    roles = ["requirement_analysis", "optimization_modeling", "mechanism_retrieval", "report_generation"]
    bundle = {}
    all_selected = []
    for role in roles:
        evidence = _retrieve_for_skill(role, requirements, limit=12 if role != "report_generation" else 18)
        evidence["citations"] = _citations_from_evidence(evidence)
        bundle[role] = evidence
        all_selected.extend(evidence.get("selected", []))
    merged = []
    seen = set()
    for item in sorted(all_selected, key=lambda row: row.get("score", 0), reverse=True):
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        merged.append(item)
    merged_evidence = {
        "role": "merged",
        "policy": {"primary_sources": ["standard", "literature", "kg"]},
        "query_terms": sorted(set().union(*(set(v.get("query_terms", [])) for v in bundle.values()))),
        "candidates": merged[:30],
        "selected": merged[:18],
        "multi_hop_routes": bundle.get("mechanism_retrieval", {}).get("multi_hop_routes", []),
        "counts": _load_evidence_index().get("counts", {}),
        "updated_at": _load_evidence_index().get("updated_at"),
    }
    merged_evidence["citations"] = _citations_from_evidence(merged_evidence)
    bundle["merged"] = merged_evidence
    return bundle


def _balanced_evidence_selection(ranked, limit):
    quotas = {"standard": 5, "literature": 8, "kg": 5}
    selected = []
    used = set()
    used_titles = set()
    for source_type, quota in quotas.items():
        for item in (record for record in ranked if record.get("source_type") == source_type):
            if len([entry for entry in selected if entry.get("source_type") == source_type]) >= quota:
                break
            title_key = (item.get("source_type"), item.get("title"))
            if title_key in used_titles:
                continue
            selected.append(item)
            used.add(item["id"])
            used_titles.add(title_key)
    for item in ranked:
        if len(selected) >= limit:
            break
        title_key = (item.get("source_type"), item.get("title"))
        if item["id"] not in used and title_key not in used_titles:
            selected.append(item)
            used.add(item["id"])
            used_titles.add(title_key)
    return selected[:limit]


def _document_domain_hints(requirements):
    raw = requirements.get("raw_request", "")
    explicit = _detect_domains(raw)
    if float(requirements.get("target_mpa") or 0) >= 60:
        explicit.append("high_strength_concrete")
    if "high_flowability" in requirements.get("priorities", []) and "self_compacting" not in explicit:
        explicit.extend([])
    return sorted(set(explicit))


DOMAIN_EQUIVALENTS = {
    "self_compacting": "self_compacting_concrete",
    "pumpable": "pumped_concrete",
    "high_performance": "high_performance_concrete",
    "high_strength": "high_strength_concrete",
    "durability": "durability_concrete",
}


def _normalize_domain_name(domain):
    return DOMAIN_EQUIVALENTS.get(str(domain or ""), str(domain or ""))


def _normative_row_applies(row, domain_hints, requirements):
    row_domain = _normalize_domain_name(row.get("concrete_type") or row.get("domain"))
    hints = {_normalize_domain_name(item) for item in domain_hints if item}
    if not hints:
        hints = {"general_concrete", "mix_design_concrete"}
        if "high_flowability" in requirements.get("priorities", []):
            hints.update({"self_compacting_concrete", "pumped_concrete"})
    raw_contexts = row.get("required_contexts") or [row_domain]
    if isinstance(raw_contexts, str):
        raw_contexts = [item.strip() for item in raw_contexts.split(",") if item.strip()]
    required_contexts = {_normalize_domain_name(item) for item in raw_contexts}
    required_contexts.discard("general_concrete")
    required_contexts.discard("mix_design_concrete")
    if not required_contexts and row_domain in {"general_concrete", "mix_design_concrete"}:
        if row.get("required_contexts"):
            explicit_contexts = {
                _normalize_domain_name(item)
                for item in _detect_domains(requirements.get("raw_request", ""))
                if item not in {"general_concrete", "mix_design_concrete"}
            }
            return not explicit_contexts
        return True
    return required_contexts.issubset(hints)


def _contextual_condition_applies(row, requirements):
    metric = row.get("condition_metric")
    if not metric:
        return True
    if metric == "strength_grade":
        actual = float(requirements.get("target_mpa") or 0)
    else:
        return False
    try:
        threshold = float(row.get("condition_value"))
    except (TypeError, ValueError):
        return False
    operator = row.get("condition_operator")
    return {
        ">": actual > threshold,
        ">=": actual >= threshold,
        "<": actual < threshold,
        "<=": actual <= threshold,
        "==": actual == threshold,
    }.get(operator, False)


def _normative_row_to_requirement(row):
    metric = row.get("metric")
    normalized_metric = "flowability" if metric in {"slump", "slump_flow", "flowability"} else metric
    return {
        "metric": normalized_metric,
        "standard_metric": metric,
        "operator": row.get("operator"),
        "lower": row.get("lower"),
        "upper": row.get("upper"),
        "value": row.get("value"),
        "unit": row.get("unit"),
        "source": row.get("source_file"),
        "page": row.get("page"),
        "evidence": row.get("evidence"),
        "domain": _normalize_domain_name(row.get("concrete_type") or row.get("domain")),
        "concrete_type": _normalize_domain_name(row.get("concrete_type") or row.get("domain")),
        "concrete_type_zh": row.get("concrete_type_zh"),
        "concrete_type_en": row.get("concrete_type_en"),
        "target": row.get("target"),
        "clause": row.get("clause"),
        "test_age_days": row.get("test_age_days"),
        "condition_metric": row.get("condition_metric"),
        "condition_operator": row.get("condition_operator"),
        "condition_value": row.get("condition_value"),
        "condition_unit": row.get("condition_unit"),
        "applicability_text": row.get("applicability_text"),
        "required_contexts": row.get("required_contexts") or [],
        "compile_status": row.get("compile_status"),
        "confidence": row.get("confidence"),
        "extraction_method": row.get("extraction_method"),
        "standard_status": row.get("standard_status") or "current_or_unspecified",
    }


def _standard_source_family(source):
    compact = re.sub(r"[\s_／/]+", "", str(source or "").upper())
    compact = re.sub(r"\(OCR\)|OCR|（OCR）", "", compact)
    match = re.search(r"(?:GB|GBT|JGJ|JGJT|TB|TBT|JC|JCT|SLT?|ASTM|ACI|EN|ISO)\d{2,8}-\d{2,4}", compact)
    if match:
        return match.group(0)
    return re.sub(r"\.PDF$", "", compact)


def _compile_domain_requirements(requirements):
    domain_hints = _document_domain_hints(requirements)
    compiled = []
    contextual_rows = _load_contextual_standard_rows()
    source_rows = contextual_rows or _load_standard_constraint_rows()
    for row in source_rows:
        if row.get("standard_status") == "future_or_superseded":
            continue
        if row.get("compile_status") not in {"optimizable", "reference_only"}:
            continue
        if not _normative_row_applies(row, domain_hints, requirements):
            continue
        if not _contextual_condition_applies(row, requirements):
            continue
        compiled.append(_normative_row_to_requirement(row))
    if not compiled and domain_hints:
        index = _load_evidence_index()
        for record in index.get("records", []):
            path = record.get("path")
            if not path:
                continue
            asset_path = ROOT / path
            if not asset_path.exists():
                continue
            summary = _asset_processing_summary(asset_path)
            if not set(domain_hints) & set(summary.get("domains", [])):
                continue
            for item in summary.get("normative_requirements", []):
                compiled.append({**item, "domains": summary.get("domains", []), "asset": asset_path.name})
    dedup = {}
    for item in compiled:
        key = (
            item.get("metric"),
            item.get("standard_metric"),
            item.get("operator"),
            item.get("lower"),
            item.get("upper"),
            item.get("value"),
            item.get("unit"),
            _standard_source_family(item.get("source")),
            tuple(item.get("required_contexts") or []),
            item.get("condition_metric"),
            item.get("condition_operator"),
            item.get("condition_value"),
        )
        previous = dedup.get(key)
        if previous is None or float(item.get("confidence") or 0) > float(previous.get("confidence") or 0):
            dedup[key] = item
    rows = list(dedup.values())
    rows.sort(key=lambda item: (
        0 if item.get("compile_status") == "optimizable" else 1,
        -float(item.get("confidence") or 0),
        str(item.get("source") or ""),
    ))
    return rows[:40]


def _feature_safe_by_material(feature_map, material):
    for safe, meta in feature_map.items():
        if meta.get("material") == material:
            return safe
    return None


def _bolomey_water_binder_window(target_mpa, payload=None):
    payload = payload or {}
    wb_min = float(payload.get("water_binder_min", 0.18 if target_mpa >= 70 else 0.20))
    engineering_wb_max = float(payload.get("water_binder_max", 0.45))
    bolomey_a = float(payload.get("bolomey_a", 22.0))
    bolomey_b = float(payload.get("bolomey_b", 0.5))
    bolomey_factor = float(payload.get("bolomey_target_factor", 1.0))
    bolomey_wb_max = 1.0 / ((float(target_mpa) * bolomey_factor / bolomey_a) + bolomey_b)
    wb_max = min(engineering_wb_max, bolomey_wb_max)
    if wb_min >= wb_max:
        wb_min = max(0.12, wb_max - 0.03)
    return {
        "min": round(wb_min, 6),
        "max": round(wb_max, 6),
        "bolomey_max": round(bolomey_wb_max, 6),
        "engineering_max": engineering_wb_max,
        "bolomey_a": bolomey_a,
        "bolomey_b": bolomey_b,
        "bolomey_factor": bolomey_factor,
    }


def _expression_terms(expression):
    return [part.strip() for part in str(expression or "").split("+") if part.strip()]


def _evaluate_linear_expression(expression, solution):
    total = 0.0
    if not expression or not solution:
        return None
    for term in _expression_terms(expression):
        match = re.match(r"([+-]?\d+(?:\.\d+)?)\s*\*\s*([A-Za-z_]\w*)$", term)
        if not match:
            continue
        coef = float(match.group(1))
        name = match.group(2)
        value = solution.get(name)
        if isinstance(value, (int, float)):
            total += coef * float(value)
    return round(total, 4)


def _make_prediction_fn(model_package, feature_map, original_columns):
    model = (model_package or {}).get("model")
    selected_features = (model_package or {}).get("selected_features") or None
    surrogate_packages = _surrogate_packages()

    def _original_frame(candidate_df):
        original_df = pd.DataFrame()
        for safe, meta in feature_map.items():
            if safe in candidate_df:
                original_df[meta["original"]] = candidate_df[safe].to_numpy(dtype=float)
        return original_df.reindex(columns=original_columns, fill_value=0)

    def _predict_with_package(original_df, package, fallback_features=None):
        package = package or {}
        chosen_features = package.get("selected_features") or fallback_features or None
        feature_columns = package.get("feature_columns") or chosen_features or original_df.columns.tolist()
        source_df = original_df
        if fallback_features:
            source_df = original_df.reindex(columns=fallback_features, fill_value=0)
        try:
            matrix = ml_modeling.prepare_design_matrix(source_df, None, chosen_features, feature_columns)
        except Exception:
            matrix = source_df.reindex(columns=feature_columns, fill_value=0)
        return np.asarray(package["model"].predict(matrix), dtype=float)

    def prediction_fn(candidate_df):
        original_df = _original_frame(candidate_df)
        context = {"prediction": _predict_with_package(original_df, model_package)}
        for alias, payload in surrogate_packages.items():
            surrogate_package = payload.get("model_package") or {}
            if surrogate_package.get("model") is None:
                continue
            fallback_features = payload.get("feature_names") or surrogate_package.get("base_feature_columns") or None
            context[alias] = _predict_with_package(original_df, surrogate_package, fallback_features=fallback_features)
        context["ShrinkageRisk"] = _shrinkage_risk_index(candidate_df, feature_map)
        return context

    return prediction_fn


def _shrinkage_risk_index(candidate_df, feature_map):
    df = pd.DataFrame(candidate_df).copy()
    n = len(df)
    if n == 0:
        return np.asarray([], dtype=float)

    def values(material):
        safe = _feature_safe_by_material(feature_map, material)
        if safe and safe in df:
            return df[safe].to_numpy(dtype=float)
        return np.zeros(n, dtype=float)

    binder_materials = ["cement", "flyash", "GGBFS", "silicafume", "limestone", "metakaolin", "ricehuskash"]
    binder = np.zeros(n, dtype=float)
    for material in binder_materials:
        binder += values(material)
    water = values("water")
    cement = values("cement")
    silica = values("silicafume")
    flyash = values("flyash")
    slag = values("GGBFS")
    metakaolin = values("metakaolin")
    fine = values("fineaggregate")
    coarse = values("coarseaggregate")
    steel = values("steelfiber")
    pce = values("superplasticizer")
    aggregate = fine + coarse
    paste = binder + water + pce
    wb = np.divide(water, np.maximum(binder, 1e-9))
    paste_aggregate = np.divide(paste, np.maximum(aggregate, 1e-9))
    silica_frac = np.divide(silica, np.maximum(binder, 1e-9))
    cement_frac = np.divide(cement, np.maximum(binder, 1e-9))
    scm_frac = np.divide(flyash + slag, np.maximum(binder, 1e-9))
    mk_frac = np.divide(metakaolin, np.maximum(binder, 1e-9))

    autogenous = np.clip((0.34 - wb) / 0.16, 0, 1)
    paste_volume = np.clip((paste_aggregate - 0.30) / 0.30, 0, 1)
    binder_dosage = np.clip((binder - 360.0) / 260.0, 0, 1)
    ultrafine = np.clip(silica_frac / 0.12, 0, 1) + 0.45 * np.clip(mk_frac / 0.10, 0, 1)
    clinker = np.clip((cement_frac - 0.45) / 0.45, 0, 1)
    aggregate_restraint = np.clip((aggregate - 1500.0) / 500.0, 0, 1)
    scm_buffer = np.clip(scm_frac / 0.45, 0, 1)
    fiber_shrinkage_control = np.clip(steel / 1.5, 0, 1)

    risk = (
        0.30 * autogenous
        + 0.25 * paste_volume
        + 0.18 * binder_dosage
        + 0.13 * ultrafine
        + 0.08 * clinker
        - 0.10 * aggregate_restraint
        - 0.06 * scm_buffer
        - 0.05 * fiber_shrinkage_control
    )
    return np.clip(risk, 0.0, 1.0)


def _build_optimization_setup(requirements, feature_map, payload):
    bounds = _bounds_from_payload(payload, feature_map)
    by_name = {item["name"]: item for item in bounds}
    age = _feature_safe_by_material(feature_map, "age")
    if age and age in by_name:
        age_days = float(requirements.get("age_days", 28.0))
        by_name[age]["lower"] = age_days
        by_name[age]["upper"] = age_days
    for material in requirements.get("avoided_materials", []):
        safe = _feature_safe_by_material(feature_map, material)
        if safe and safe in by_name:
            by_name[safe]["lower"] = 0.0
            by_name[safe]["upper"] = 0.0
    practical_constraints = []
    explicitly_required = set(requirements.get("required_materials", []))
    explicitly_allowed = set(requirements.get("allowed_materials", []))
    requested_supplementary = explicitly_required & SUPPLEMENTARY_BINDERS
    allowed_supplementary = explicitly_allowed & SUPPLEMENTARY_BINDERS
    for safe, meta in feature_map.items():
        material = meta.get("material")
        if material in MANDATORY_MINIMUMS and safe in by_name:
            by_name[safe]["lower"] = max(float(by_name[safe]["lower"]), MANDATORY_MINIMUMS[material])
        if requested_supplementary and material in SUPPLEMENTARY_BINDERS and material not in requested_supplementary and safe in by_name:
            by_name[safe]["lower"] = 0.0
            by_name[safe]["upper"] = 0.0
        if not requested_supplementary and allowed_supplementary and material in SUPPLEMENTARY_BINDERS and material not in allowed_supplementary and safe in by_name:
            by_name[safe]["lower"] = 0.0
            by_name[safe]["upper"] = 0.0
        if material in PRACTICAL_MINIMUMS and safe in by_name and material not in requirements.get("avoided_materials", []):
            minimum = PRACTICAL_MINIMUMS[material]
            if material in explicitly_required:
                by_name[safe]["lower"] = max(float(by_name[safe]["lower"]), minimum)
            else:
                by_name[safe]["lower"] = max(0.0, float(by_name[safe]["lower"]))
                zero = ZERO_THRESHOLDS[material]
                practical_constraints.append(f"where({safe} <= {zero}, 0, {safe} - {minimum}) >= 0")
        if material == "steelfiber" and safe in by_name:
            by_name[safe]["lower"] = max(0.0, float(by_name[safe]["lower"]))
            by_name[safe]["upper"] = min(float(by_name[safe]["upper"]), 3.0)

    target_mpa = float(requirements["target_mpa"])
    carbon_expr = payload.get("carbon_expression") or _carbon_expression(feature_map)
    cost_expr = payload.get("cost_expression") or _cost_expression(feature_map)
    objectives = [
        {"name": "StrengthTarget", "expression": "prediction", "goal": "target", "target": target_mpa},
        {"name": "Carbon", "expression": carbon_expr, "goal": "min"},
    ]
    if cost_expr != "0":
        objectives.append({"name": "Cost", "expression": cost_expr, "goal": "min"})
    constraints = [f"prediction - {target_mpa} >= 0", *practical_constraints]
    binder_terms = [
        safe
        for safe, meta in feature_map.items()
        if meta.get("material") in {"cement", "flyash", "GGBFS", "silicafume", "limestone", "metakaolin", "ricehuskash"}
    ]
    binder_expr = " + ".join(binder_terms)
    water = _feature_safe_by_material(feature_map, "water")
    sand = _feature_safe_by_material(feature_map, "fineaggregate")
    volume_terms = []
    mass_terms = []
    for safe, meta in feature_map.items():
        material = meta.get("material")
        if material in MATERIAL_DENSITIES:
            density = float(payload.get("densities", {}).get(material, MATERIAL_DENSITIES[material]))
            volume_terms.append(f"{safe} / {density}")
            mass_terms.append(safe)
    steel = _feature_safe_by_material(feature_map, "steelfiber")
    air_content = float(payload.get("air_content", 0.02))
    absolute_volume_expr = " + ".join(volume_terms + ([f"{steel} / 100"] if steel else []) + [str(air_content)]) if volume_terms else None
    theoretical_density_expr = " + ".join(mass_terms) if mass_terms else None
    wb_window = _bolomey_water_binder_window(target_mpa, payload)
    if water and binder_expr:
        wb_min = wb_window["min"]
        wb_max = wb_window["max"]
        constraints.extend([
            f"{water} / ({binder_expr}) - {wb_min} >= 0",
            f"{wb_max} - {water} / ({binder_expr}) >= 0",
        ])
    if sand and binder_expr:
        bs_min = float(payload.get("binder_sand_min", 0.35))
        bs_max = float(payload.get("binder_sand_max", 0.80))
        constraints.extend([
            f"({binder_expr}) / {sand} - {bs_min} >= 0",
            f"{bs_max} - ({binder_expr}) / {sand} >= 0",
        ])
    if absolute_volume_expr:
        volume_tolerance = float(payload.get("absolute_volume_tolerance", 0.02))
        constraints.append(f"{volume_tolerance} - abs(({absolute_volume_expr}) - 1.0) >= 0")
    bolomey_a = wb_window["bolomey_a"]
    bolomey_b = wb_window["bolomey_b"]
    bolomey_expr = None
    bolomey_wb_max = None
    if water and binder_expr:
        bolomey_expr = f"{bolomey_a} * (({binder_expr}) / {water} - {bolomey_b})"
        bolomey_wb_max = wb_window["bolomey_max"]

    if "high_flowability" in requirements.get("priorities", []):
        flow_alias = _find_flowability_alias()
        if flow_alias:
            objectives.append({"name": "Flowability", "expression": flow_alias, "goal": "max"})
    if "low_shrinkage" in requirements.get("priorities", []):
        objectives.append({
            "name": "ShrinkageRisk",
            "expression": "ShrinkageRisk",
            "goal": "min",
            "evidence_level": "mechanistic_proxy",
            "note": "Mechanistic shrinkage-risk proxy derived from w/b, paste-to-aggregate ratio, binder dosage, ultrafine powders, aggregate restraint, SCM fraction, and fiber shrinkage-control terms; it is not a measured microstrain prediction.",
        })
    flow_alias = _find_flowability_alias()
    for spec in requirements.get("performance_constraints", []):
        if spec.get("metric") == "flowability" and flow_alias:
            if isinstance(spec.get("lower"), (int, float)):
                constraints.append(f"{flow_alias} - {float(spec['lower'])} >= 0")
            if isinstance(spec.get("upper"), (int, float)):
                constraints.append(f"{float(spec['upper'])} - {flow_alias} >= 0")
    compiled_normative_constraints = []
    for item in requirements.get("normative_requirements", []):
        metric = item.get("metric")
        if item.get("compile_status") != "optimizable":
            continue
        if metric not in {"slump", "slump_flow", "flowability"} or not flow_alias:
            continue
        expressions = []
        lower = item.get("lower")
        upper = item.get("upper")
        value = item.get("value")
        operator = item.get("operator")
        if isinstance(lower, (int, float)):
            expressions.append(f"{flow_alias} - {float(lower)} >= 0")
        elif operator in {">=", ">"} and isinstance(value, (int, float)):
            expressions.append(f"{flow_alias} - {float(value)} >= 0")
        if isinstance(upper, (int, float)):
            expressions.append(f"{float(upper)} - {flow_alias} >= 0")
        elif operator in {"<=", "<"} and isinstance(value, (int, float)):
            expressions.append(f"{float(value)} - {flow_alias} >= 0")
        for expression in expressions:
            if expression not in constraints:
                constraints.append(expression)
                compiled_normative_constraints.append({**item, "constraint": expression})

    return {
        "algorithm": payload.get("algorithm", "NSGA-II"),
        "bounds": bounds,
        "objectives": objectives,
        "constraints": constraints,
        "carbon_expression": carbon_expr,
        "cost_expression": cost_expr,
        "flowability_alias": _find_flowability_alias(),
        "shrinkage_risk_alias": "ShrinkageRisk" if "low_shrinkage" in requirements.get("priorities", []) else None,
        "pop_size": int(payload.get("pop_size", 72)),
        "generations": int(payload.get("generations", 36)),
        "compare_algorithms": payload.get("compare_algorithms") or ["MOEA/D", "MOPSO"],
        "standard_constraint_visualization": _standard_constraint_visualization_payload(requirements.get("normative_requirements", [])),
        "practical_dosage_policy": {
            "mandatory_minimums": MANDATORY_MINIMUMS,
            "optional_minimums": PRACTICAL_MINIMUMS,
            "zero_thresholds": ZERO_THRESHOLDS,
        },
        "material_policy": {
            "mode": requirements.get("material_mode", "exploratory"),
            "requested_supplementary_binders": sorted(requested_supplementary),
            "allowed_supplementary_binders": sorted(allowed_supplementary),
            "allowed_materials": requirements.get("allowed_materials", []),
            "forbidden_materials": requirements.get("avoided_materials", []),
        },
        "traditional_theory": {
            "densities": {material: float(payload.get("densities", {}).get(material, density)) for material, density in MATERIAL_DENSITIES.items()},
            "air_content": air_content,
            "absolute_volume_expression": absolute_volume_expr,
            "theoretical_density_expression": theoretical_density_expr,
            "absolute_volume_tolerance": float(payload.get("absolute_volume_tolerance", 0.02)),
            "bolomey_a": bolomey_a,
            "bolomey_b": bolomey_b,
            "bolomey_expression": bolomey_expr,
            "bolomey_target_factor": float(payload.get("bolomey_target_factor", 1.0)),
            "bolomey_wb_max": bolomey_wb_max,
            "water_binder_window": {"min": wb_window["min"], "max": wb_window["max"]},
            "target_mpa": target_mpa,
            "calibration_status": "project_calibrated" if payload.get("bolomey_calibrated") else "default_reference",
            "assumption_note": "当前 A、B0、密度与含气量来自默认工程先验；正式工程应用前应以本项目原材试验重新标定。"
            if not payload.get("bolomey_calibrated")
            else "鲍罗米参数已按项目数据标定。",
        },
        "constraint_status": _constraint_status(requirements, flow_alias),
        "unresolved_constraints": _constraint_status(requirements, flow_alias)["unresolved"],
        "compiled_normative_constraints": compiled_normative_constraints,
    }


def _kg_enhancement(requirements, feature_map, best_solution, shared_evidence=None):
    hits = _kg_hits(requirements.get("raw_request", ""), feature_map, best_solution)
    evidence = shared_evidence or _hybrid_retrieve(requirements)
    recommendations = []
    shrinkage_evidence = {}
    if "high_flowability" in requirements.get("priorities", []):
        recommendations.extend(FLOWABILITY_KG)
    if "low_shrinkage" in requirements.get("priorities", []):
        recommendations.extend(SHRINKAGE_KG)
        evidence_materials = set(requirements.get("required_materials") or [])
        evidence_materials.update(requirements.get("allowed_materials") or [])
        evidence_materials.update(item.get("material") for item in hits if item.get("material"))
        shrinkage_evidence = _shrinkage_evidence_for_materials(evidence_materials)
    if "low_carbon" in requirements.get("priorities", []):
        recommendations.append({
            "topic": "熟料替代与后期强度",
            "topic_en": "Clinker substitution and later-age strength",
            "mechanism": "矿渣、粉煤灰、硅灰通过潜在水硬/火山灰反应生成二次胶凝产物，降低熟料用量并细化孔结构。",
            "mechanism_en": "Slag, fly ash, and silica fume form secondary binding products through latent hydraulic or pozzolanic reactions, reducing clinker demand and refining pore structure.",
            "recommendation": "在满足目标强度的解中优先选择 Carbon 较低且矿物掺合料占比较高的方案，再通过 7d/28d 强度试验校准早期强度风险。",
            "recommendation_en": "Among solutions satisfying the target strength, prioritize lower-carbon mixtures with higher supplementary cementitious material fractions, then calibrate early-strength risk using 7 d and 28 d strength tests.",
        })
    if best_solution:
        silica = _feature_safe_by_material(feature_map, "silicafume")
        sp = _feature_safe_by_material(feature_map, "superplasticizer")
        if silica and sp and best_solution.get(silica, 0) > 0.6 * feature_map[silica]["upper"]:
            recommendations.append({
                "topic": "硅灰高掺量分散风险",
                "topic_en": "Dispersion risk at high silica-fume dosage",
                "mechanism": "硅灰高比表面积会显著提高外加剂需求，分散不足时流动性下降。",
                "mechanism_en": "The high specific surface area of silica fume substantially increases admixture demand; insufficient dispersion reduces flowability.",
                "recommendation": "建议试配中提高 PCE 梳形减水剂适配性测试，或以部分矿渣/粉煤灰微珠替代硅灰以降低黏度。",
                "recommendation_en": "In trial mixtures, strengthen PCE compatibility testing or partially replace silica fume with slag or fly-ash microspheres to reduce viscosity.",
            })
    target_performances = {"compressive strength", "flowability"}
    priorities = set(requirements.get("priorities", []))
    if "low_shrinkage" in priorities:
        target_performances.update({"shrinkage", "shrinkage"})
    if "low_carbon" in priorities:
        target_performances.add("low carbon")
    return {
        "hits": hits,
        "recommendations": recommendations[:6],
        "target_performances": sorted(target_performances),
        "query_terms": sorted(_kg_query_terms(requirements)),
        "evidence": evidence,
        "shrinkage_evidence_summary": shrinkage_evidence,
        "citations": evidence.get("citations") or _citations_from_evidence(evidence),
    }


def _summarize_solution(best, feature_map):
    if not best:
        return []
    rows = []
    for safe, meta in feature_map.items():
        value = best.get(safe)
        if isinstance(value, (int, float)):
            material = meta.get("material")
            zero_threshold = ZERO_THRESHOLDS.get(material)
            display_value = 0.0 if zero_threshold is not None and 0 <= float(value) <= zero_threshold else float(value)
            unit = meta.get("unit") or ("d" if meta.get("material") == "age" else "kg/m3")
            rows.append({
                "name": meta["original"],
                "value": round(display_value, 3),
                "unit": unit,
                "material": material,
            })
    return rows


def _solution_kpis(best, setup=None, feature_map=None):
    if not best:
        return {}
    kpis = {
        "strength_mpa": best.get("prediction") or best.get("StrengthTarget"),
        "carbon_kgco2e_m3": best.get("Carbon"),
        "cost_index": best.get("Cost"),
        "flowability": best.get("Flowability"),
        "shrinkage_risk_index": best.get("ShrinkageRisk"),
    }
    if kpis["shrinkage_risk_index"] is None and feature_map:
        try:
            kpis["shrinkage_risk_index"] = float(_shrinkage_risk_index(pd.DataFrame([best]), feature_map)[0])
        except Exception:
            kpis["shrinkage_risk_index"] = None
    if setup:
        if kpis["carbon_kgco2e_m3"] is None:
            kpis["carbon_kgco2e_m3"] = _evaluate_linear_expression(setup.get("carbon_expression"), best)
        if kpis["cost_index"] is None:
            kpis["cost_index"] = _evaluate_linear_expression(setup.get("cost_expression"), best)
    return {key: round(float(value), 4) for key, value in kpis.items() if isinstance(value, (int, float))}


def _mix_ratios(solution, feature_map):
    if not solution:
        return {}
    binder_materials = {"cement", *SUPPLEMENTARY_BINDERS}
    binder = 0.0
    water = 0.0
    aggregate = 0.0
    admixture = 0.0
    has_water = False
    for safe, meta in feature_map.items():
        value = solution.get(safe)
        if not isinstance(value, (int, float)):
            continue
        material = meta.get("material")
        if material in binder_materials:
            binder += float(value)
        elif material == "water":
            water += float(value)
            has_water = True
        elif material in {"fineaggregate", "coarseaggregate"}:
            aggregate += float(value)
        elif material == "superplasticizer":
            admixture += float(value)
    ratios = {"binder_total_kg_m3": round(binder, 4)}
    if has_water and binder > 0:
        ratios["water_binder_ratio"] = round(water / binder, 4)
    paste = binder + water + admixture
    if aggregate > 0:
        ratios["paste_aggregate_ratio"] = round(paste / aggregate, 4)
    return ratios


def _traditional_theory_checks(solution, feature_map, setup):
    if not solution:
        return {}
    theory = setup.get("traditional_theory", {}) if setup else {}
    densities = theory.get("densities", MATERIAL_DENSITIES)
    air_content = float(theory.get("air_content", 0.02))
    volume = air_content
    total_mass = 0.0
    components = []
    for safe, meta in feature_map.items():
        value = solution.get(safe)
        material = meta.get("material")
        if not isinstance(value, (int, float)):
            continue
        if material in densities:
            partial_volume = float(value) / float(densities[material])
            volume += partial_volume
            total_mass += float(value)
            components.append({"material": material, "mass": round(float(value), 4), "density": densities[material], "volume": round(partial_volume, 6)})
        elif material == "steelfiber":
            partial_volume = float(value) / 100.0
            volume += partial_volume
            components.append({"material": material, "mass": None, "density": None, "volume": round(partial_volume, 6)})
    ratios = _mix_ratios(solution, feature_map)
    bolomey_strength = None
    if ratios.get("water_binder_ratio"):
        a = float(theory.get("bolomey_a", 22.0))
        b = float(theory.get("bolomey_b", 0.5))
        bolomey_strength = a * ((1 / ratios["water_binder_ratio"]) - b)
    tolerance = float(theory.get("absolute_volume_tolerance", 0.02))
    target = theory.get("target_mpa")
    if target is None:
        target = solution.get("StrengthTarget") or solution.get("prediction")
    return {
        "components": components,
        "absolute_volume_m3": round(volume, 6),
        "absolute_volume_error": round(volume - 1.0, 6),
        "absolute_volume_pass": abs(volume - 1.0) <= tolerance,
        "theoretical_density_kg_m3": round(total_mass, 4),
        "bolomey_strength_mpa": round(bolomey_strength, 4) if bolomey_strength is not None else None,
        "bolomey_pass": bool(bolomey_strength is not None and target is not None and bolomey_strength >= float(target)),
        "air_content": air_content,
        "density_assumptions": densities,
        "bolomey_wb_max": theory.get("bolomey_wb_max"),
        "water_binder_window": theory.get("water_binder_window", {}),
        "calibration_status": theory.get("calibration_status", "default_reference"),
        "assumption_note": theory.get("assumption_note", "默认工程先验，待项目标定。"),
    }


def _passes_design_priors(solution, feature_map, setup):
    theory = _traditional_theory_checks(solution, feature_map, setup)
    ratios = _mix_ratios(solution, feature_map)
    wb = ratios.get("water_binder_ratio")
    window = (setup.get("traditional_theory") or {}).get("water_binder_window", {})
    wb_ok = wb is not None and window.get("min") is not None and window.get("max") is not None and window["min"] <= wb <= window["max"]
    return bool(wb_ok and theory.get("absolute_volume_pass") and theory.get("bolomey_pass"))


def _retain_prior_feasible_solutions(result, feature_map, setup):
    if not result:
        return result
    feasible = [item for item in result.get("solutions", []) if _passes_design_priors(item, feature_map, setup)]
    if not feasible:
        return {**result, "solutions": [], "best_solution": None, "prior_feasible_count": 0}
    return {**result, "solutions": feasible, "best_solution": feasible[0], "prior_feasible_count": len(feasible)}


def _representative_mix_schemes(result, feature_map, requirements, setup=None):
    solutions = result.get("solutions", []) if result else []
    if not solutions:
        return []
    schemes = []
    seen = set()

    def append_scheme(label, solution):
        signature = tuple(round(float(solution.get(safe, 0.0)), 3) for safe in feature_map)
        if signature in seen:
            return
        seen.add(signature)
        schemes.append({
            "label": label,
            "solution_table": _summarize_solution(solution, feature_map),
            "kpis": _solution_kpis(solution, setup, feature_map),
            "ratios": _mix_ratios(solution, feature_map),
            "traditional_theory": _traditional_theory_checks(solution, feature_map, setup or {}),
        })

    append_scheme("综合推荐方案", solutions[0])
    requested = set(requirements.get("required_materials", [])) & SUPPLEMENTARY_BINDERS
    carbon_ranked = sorted(
        [item for item in solutions if _solution_kpis(item, setup, feature_map).get("carbon_kgco2e_m3") is not None],
        key=lambda item: _solution_kpis(item, setup, feature_map).get("carbon_kgco2e_m3"),
    )
    strength_ranked = sorted(
        [item for item in solutions if _solution_kpis(item, setup, feature_map).get("strength_mpa") is not None],
        key=lambda item: _solution_kpis(item, setup, feature_map).get("strength_mpa"),
        reverse=True,
    )
    flow_ranked = sorted(
        [item for item in solutions if _solution_kpis(item, setup, feature_map).get("flowability") is not None],
        key=lambda item: _solution_kpis(item, setup, feature_map).get("flowability"),
        reverse=True,
    )
    shrinkage_ranked = sorted(
        [item for item in solutions if _solution_kpis(item, setup, feature_map).get("shrinkage_risk_index") is not None],
        key=lambda item: _solution_kpis(item, setup, feature_map).get("shrinkage_risk_index"),
    )
    if carbon_ranked:
        append_scheme("最低碳候选", carbon_ranked[0])
    if strength_ranked:
        append_scheme("最高强候选", strength_ranked[0])
    if flow_ranked:
        append_scheme("最高流动候选", flow_ranked[0])
    if "low_shrinkage" in requirements.get("priorities", []) and shrinkage_ranked:
        append_scheme("最低收缩风险候选", shrinkage_ranked[0])
    if requested:
        return schemes[:4]

    for material in ["flyash", "GGBFS", "silicafume", "limestone", "metakaolin", "ricehuskash"]:
        safe = _feature_safe_by_material(feature_map, material)
        minimum = PRACTICAL_MINIMUMS.get(material, 0.0)
        candidates = [item for item in solutions if safe and float(item.get(safe, 0.0) or 0.0) >= minimum]
        if candidates:
            append_scheme(f"{material} 主导方案", candidates[0])
        if len(schemes) >= 6:
            break
    return schemes[:6]


def _scheme_comparison(schemes):
    rows = []
    if not schemes:
        return rows
    carbon_values = [item["kpis"].get("carbon_kgco2e_m3") for item in schemes if item["kpis"].get("carbon_kgco2e_m3") is not None]
    strength_values = [item["kpis"].get("strength_mpa") for item in schemes if item["kpis"].get("strength_mpa") is not None]
    flow_values = [item["kpis"].get("flowability") for item in schemes if item["kpis"].get("flowability") is not None]
    shrinkage_values = [item["kpis"].get("shrinkage_risk_index") for item in schemes if item["kpis"].get("shrinkage_risk_index") is not None]
    for item in schemes:
        kpis = item.get("kpis", {})
        roles = []
        if kpis.get("carbon_kgco2e_m3") is not None and carbon_values and kpis["carbon_kgco2e_m3"] == min(carbon_values):
            roles.append("最低碳候选")
        if kpis.get("strength_mpa") is not None and strength_values and kpis["strength_mpa"] == max(strength_values):
            roles.append("最高强候选")
        if kpis.get("flowability") is not None and flow_values and kpis["flowability"] == max(flow_values):
            roles.append("最高流动候选")
        if kpis.get("shrinkage_risk_index") is not None and shrinkage_values and kpis["shrinkage_risk_index"] == min(shrinkage_values):
            roles.append("最低收缩风险候选")
        if item.get("label") == "综合推荐方案":
            roles.insert(0, "综合折中推荐")
        rows.append({
            "label": item.get("label"),
            "roles": roles or ["对照方案"],
            "strength_mpa": kpis.get("strength_mpa"),
            "carbon_kgco2e_m3": kpis.get("carbon_kgco2e_m3"),
            "cost_index": kpis.get("cost_index"),
            "flowability": kpis.get("flowability"),
            "shrinkage_risk_index": kpis.get("shrinkage_risk_index"),
            "water_binder_ratio": item.get("ratios", {}).get("water_binder_ratio"),
            "paste_aggregate_ratio": item.get("ratios", {}).get("paste_aggregate_ratio"),
        })
    return rows


def _design_trust_profile(solution, feature_map, setup, kg_payload=None):
    if not solution:
        return {}
    bounds = {item["name"]: item for item in setup.get("bounds", [])}
    margin_rows = []
    near_bounds = []
    for safe, meta in feature_map.items():
        value = solution.get(safe)
        bound = bounds.get(safe)
        if not isinstance(value, (int, float)) or not bound:
            continue
        lower = float(bound["lower"])
        upper = float(bound["upper"])
        span = max(upper - lower, 1e-9)
        normalized = (float(value) - lower) / span
        nearest_margin = min(abs(normalized), abs(1 - normalized))
        row = {
            "name": meta["original"],
            "material": meta.get("material"),
            "value": round(float(value), 4),
            "normalized_position": round(normalized, 4),
            "nearest_bound_margin": round(nearest_margin, 4),
        }
        margin_rows.append(row)
        if nearest_margin <= 0.05:
            near_bounds.append(row)
    kg_hits = kg_payload.get("hits", []) if isinstance(kg_payload, dict) else []
    kg_recs = kg_payload.get("recommendations", []) if isinstance(kg_payload, dict) else []
    if near_bounds:
        applicability = "边界邻近"
        confidence = "中"
    else:
        applicability = "变量边界内部"
        confidence = "较高"
    if len(near_bounds) >= 3:
        confidence = "中低"
    return {
        "applicability": applicability,
        "confidence": confidence,
        "near_bound_variables": near_bounds,
        "margin_rows": margin_rows,
        "evidence_levels": [
            {"label": "模型预测", "level": "定量", "description": "由上传模型直接输出的强度、流动性或相关目标值。"},
            {"label": "优化约束", "level": "可复现", "description": "由显式变量边界、材料策略和工程约束决定。"},
            {"label": "图谱机理", "level": "机理支持", "description": f"当前命中 {len(kg_hits)} 类材料机理，生成 {len(kg_recs)} 条建议。"},
            {"label": "实验验证", "level": "待确认", "description": "工作性、早期强度、收缩响应和实际原材适配仍需试验确认。"},
        ],
        "risk_flags": [
            *([f"{item['name']} 靠近搜索边界" for item in near_bounds[:4]]),
            *([] if setup.get("flowability_alias") else ["当前模型包未提供独立流动性预测器"]),
        ],
    }


def _method_baseline_comparison(requirements, optimization_summary, kg_payload=None):
    kpis = optimization_summary.get("kpis", {}) if isinstance(optimization_summary, dict) else {}
    objectives = [item.get("name") for item in optimization_summary.get("objectives", []) if isinstance(item, dict)]
    has_shrinkage = "low_shrinkage" in requirements.get("priorities", []) or "ShrinkageRisk" in objectives
    citation_count = len(kg_payload.get("citations", [])) if isinstance(kg_payload, dict) else 0
    kg_hit_count = len(kg_payload.get("hits", [])) if isinstance(kg_payload, dict) else 0
    shrinkage_note = (
        f"Low-shrinkage request is represented by a shrinkage-risk proxy ({kpis.get('shrinkage_risk_index')}) and KG-supported mechanism review."
        if has_shrinkage
        else "Shrinkage was not requested as a primary objective in this run."
    )
    return [
        {
            "method_id": "general_llm",
            "method": "General LLM direct design",
            "method_zh": "通用大语言模型直接给配方",
            "status": "reference_baseline",
            "basis": "No executable variable bounds, objective functions, model prediction, or verifiable mix-design equations are enforced.",
            "basis_zh": "不强制执行变量边界、目标函数、模型预测或可复现实用配合比公式。",
            "observed_or_expected_limitation": "Can produce plausible text but cannot guarantee feasibility, target-strength satisfaction, carbon accounting, or low-shrinkage trade-off control.",
            "observed_or_expected_limitation_zh": "可以生成看似合理的文字，但不能保证可行性、目标强度、碳核算或低收缩权衡控制。",
            "current_system_advantage": "This platform converts the request into structured constraints, runs a reproducible multi-objective optimizer, and attaches evidence for review.",
            "current_system_advantage_zh": "本平台将需求转为结构化约束，执行可复现多目标优化，并附带证据复审。",
        },
        {
            "method_id": "data_only_optimizer",
            "method": "Pure data-driven optimizer",
            "method_zh": "纯数据驱动优化器",
            "status": "reference_baseline",
            "basis": "Uses model predictions and Pareto search but has no material-mechanism retrieval or evidence-based risk audit.",
            "basis_zh": "使用模型预测和 Pareto 搜索，但没有材料机理检索和证据化风险审计。",
            "observed_or_expected_limitation": "May optimize strength or carbon numerically while missing mechanism conflicts such as ultrafine-powder shrinkage risk or SCM curing sensitivity.",
            "observed_or_expected_limitation_zh": "可能在数值上优化强度或碳排，但遗漏超细粉体收缩风险、SCM 养护敏感性等机理冲突。",
            "current_system_advantage": f"KG retrieval returned {kg_hit_count} material-mechanism groups and supports design interpretation beyond black-box prediction.",
            "current_system_advantage_zh": f"知识图谱命中 {kg_hit_count} 类材料机理，使设计解释不只依赖黑箱预测。",
        },
        {
            "method_id": "kg_free_agent",
            "method": "LLM-agent workflow without KG evidence",
            "method_zh": "无知识图谱证据的智能体流程",
            "status": "reference_baseline",
            "basis": "Agent roles can be separated, but retrieval, mechanism chains, and citation records are absent.",
            "basis_zh": "可以拆分智能体角色，但缺少检索、机理链和引用记录。",
            "observed_or_expected_limitation": "Reasoning is hard to audit and weak for publication-grade explanation and experimental follow-up planning.",
            "observed_or_expected_limitation_zh": "推理难以审计，不利于论文级解释和后续实验规划。",
            "current_system_advantage": f"The current run keeps {citation_count} evidence citations and passes them into the final report and reference panel.",
            "current_system_advantage_zh": f"当前运行保留 {citation_count} 条证据引用，并传入正式报告和引用弹窗。",
        },
        {
            "method_id": "kg_enhanced_data_driven_agent",
            "method": "Proposed KG-enhanced data-driven multi-agent system",
            "method_zh": "本文知识增强数据驱动多智能体系统",
            "status": "current_method",
            "basis": "Combines requirement parsing, ML prediction, constrained multi-objective optimization, engineering-theory checks, KG retrieval, and evidence-backed reporting.",
            "basis_zh": "融合需求解析、机器学习预测、约束多目标优化、工程理论校核、知识图谱检索和证据化报告。",
            "observed_or_expected_limitation": "Final adoption still requires trial mixing and target-age validation; shrinkage is currently ranked by a mechanistic proxy unless a measured shrinkage model is uploaded.",
            "observed_or_expected_limitation_zh": "最终工程采用仍需试拌和目标龄期验证；若未上传实测收缩模型，收缩目前按机理代理指标排序。",
            "current_system_advantage": shrinkage_note,
            "current_system_advantage_zh": "低收缩需求已进入目标函数、候选方案比较和机理证据链。" if has_shrinkage else "当前运行完成强度、碳排、方案比较和机理证据链整合。",
        },
    ]


def _design_evidence_chain(requirements, optimization_summary, kg_payload=None):
    kpis = optimization_summary.get("kpis", {}) if isinstance(optimization_summary, dict) else {}
    theory = optimization_summary.get("traditional_theory", {}) if isinstance(optimization_summary, dict) else {}
    ratios = optimization_summary.get("ratios", {}) if isinstance(optimization_summary, dict) else {}
    objectives = [item.get("name") for item in optimization_summary.get("objectives", []) if isinstance(item, dict)]
    citations = kg_payload.get("citations", []) if isinstance(kg_payload, dict) else []
    recommendations = kg_payload.get("recommendations", []) if isinstance(kg_payload, dict) else []
    refs = [item.get("ref_no") for item in citations[:6] if item.get("ref_no")]
    rec_topics = [item.get("topic") or item.get("material") for item in recommendations[:4] if item.get("topic") or item.get("material")]
    rows = [
        {
            "claim": "Target-strength feasibility",
            "claim_zh": "目标强度可行性",
            "value": f"{kpis.get('strength_mpa', '-')} MPa at {optimization_summary.get('age_days', requirements.get('age_days', '-'))} d",
            "source_type": "model_prediction + engineering_theory",
            "evidence": f"Uploaded model prediction plus Bolomey prior check; w/b = {ratios.get('water_binder_ratio', '-')}.",
            "evidence_zh": f"上传模型预测结合鲍罗米先验校核；水胶比 = {ratios.get('water_binder_ratio', '-')}。",
            "validation": "Confirm with 3 d, 7 d, and target-age compressive-strength tests.",
            "validation_zh": "通过 3 d、7 d 和目标龄期抗压强度试验确认。",
            "refs": refs,
        },
        {
            "claim": "Low-carbon objective",
            "claim_zh": "低碳目标",
            "value": f"{kpis.get('carbon_kgco2e_m3', '-')} kgCO2e/m3",
            "source_type": "objective_function + material_substitution",
            "evidence": "Carbon is minimized as an explicit optimization objective while SCM-related KG routes support clinker-reduction interpretation.",
            "evidence_zh": "碳排作为显式优化目标最小化，SCM 相关图谱链支持熟料替代解释。",
            "validation": "Audit the carbon factors of actual cement, SCMs, admixtures, and transport assumptions.",
            "validation_zh": "复核实际水泥、矿物掺合料、外加剂和运输假设的碳因子。",
            "refs": refs,
        },
        {
            "claim": "Low-shrinkage risk control",
            "claim_zh": "低收缩风险控制",
            "value": kpis.get("shrinkage_risk_index", "-"),
            "source_type": "mechanistic_proxy + knowledge_graph",
            "evidence": "ShrinkageRisk combines w/b, paste-to-aggregate ratio, binder dosage, ultrafine powder fraction, aggregate restraint, SCM buffering, and fiber shrinkage-control terms.",
            "evidence_zh": "ShrinkageRisk 综合水胶比、浆骨比、胶凝材料用量、超细粉体比例、骨料约束、SCM 缓冲和纤维控裂项。",
            "validation": "Measure drying shrinkage, autogenous shrinkage, and restrained-ring shrinkage when needed.",
            "validation_zh": "测试干燥收缩、自收缩，必要时做收缩试验。",
            "refs": refs,
        },
        {
            "claim": "Mechanistic recommendation support",
            "claim_zh": "机理建议支持",
            "value": " / ".join(rec_topics) if rec_topics else "-",
            "source_type": "knowledge_graph + retrieved_evidence",
            "evidence": f"Mechanism retrieval generated {len(recommendations)} recommendations and retained citation records for the reference panel.",
            "evidence_zh": f"机理检索生成 {len(recommendations)} 条建议，并保留引用记录用于弹窗追溯。",
            "validation": "Use the retrieved mechanisms to design trial-mix variables and ablation tests.",
            "validation_zh": "用检索到的机理设计试拌变量和消融实验。",
            "refs": refs,
        },
        {
            "claim": "Engineering consistency",
            "claim_zh": "工程一致性",
            "value": "absolute volume pass" if theory.get("absolute_volume_pass") else "absolute volume warning",
            "source_type": "traditional_mix_design_check",
            "evidence": f"Absolute volume = {theory.get('absolute_volume_m3', '-')} m3; theoretical density = {theory.get('theoretical_density_kg_m3', '-')} kg/m3.",
            "evidence_zh": f"绝对体积 = {theory.get('absolute_volume_m3', '-')} m3；理论容重 = {theory.get('theoretical_density_kg_m3', '-')} kg/m3。",
            "validation": "Check batching density, air content, aggregate moisture correction, and yield.",
            "validation_zh": "复核拌合容重、含气量、骨料含水率修正和出方量。",
            "refs": [],
        },
    ]
    if "ShrinkageRisk" not in objectives and "low_shrinkage" not in requirements.get("priorities", []):
        rows = [row for row in rows if row["claim"] != "Low-shrinkage risk control"]
    return rows


def _attach_design_audits(requirements, optimization_summary, kg_payload, setup=None, feature_map=None):
    setup = setup or {"bounds": []}
    feature_map = feature_map or STATE.get("feature_map") or {}
    optimization_summary["scheme_comparison"] = _scheme_comparison(optimization_summary.get("candidate_schemes", []))
    optimization_summary["trust_profile"] = _design_trust_profile(
        optimization_summary.get("best_solution", {}),
        feature_map,
        setup,
        kg_payload,
    )
    optimization_summary["method_baselines"] = _method_baseline_comparison(requirements, optimization_summary, kg_payload)
    optimization_summary["evidence_chain"] = _design_evidence_chain(requirements, optimization_summary, kg_payload)
    return optimization_summary


def _handoff_payload(requirements, setup, optimization_summary, kg_payload, report_requests=None):
    return {
        "requirement_to_optimizer": {
            "target_mpa": requirements.get("target_mpa"),
            "age_days": requirements.get("age_days"),
            "priorities": requirements.get("priorities", []),
            "mentioned_materials": requirements.get("mentioned_materials", []),
            "required_materials": requirements.get("required_materials", []),
            "allowed_materials": requirements.get("allowed_materials", []),
            "forbidden_materials": requirements.get("avoided_materials", []),
            "performance_constraints": requirements.get("performance_constraints", []),
            "material_mode": requirements.get("material_mode"),
        },
        "optimizer_to_kg": {
            "selected_algorithm": optimization_summary.get("best_algorithm"),
            "recommended_solution": optimization_summary.get("solution_table", []),
            "kpis": optimization_summary.get("kpis", {}),
            "candidate_scheme_count": len(optimization_summary.get("candidate_schemes", [])),
        },
        "kg_to_report": {
            "hit_count": len(kg_payload.get("hits", [])),
            "recommendation_count": len(kg_payload.get("recommendations", [])),
            "target_performances": kg_payload.get("target_performances", []),
        },
        "report_feedback": report_requests or {},
    }


def _requirement_trace(requirements):
    mapping = [
        {"token": f"{requirements['target_mpa']:g} MPa", "field": "目标强度", "value": f"{requirements['target_mpa']:g} MPa", "reason": "作为强度约束和强度目标函数的目标值"},
        {"token": f"{requirements['age_days']:g} 天", "field": "龄期", "value": f"{requirements['age_days']:g} d", "reason": "若模型存在 age/龄期变量，将固定为该值参与预测"},
    ]
    for priority in requirements.get("priorities", []):
        labels = {
            "low_carbon": "低碳优先",
            "high_flowability": "高流动性",
            "low_shrinkage": "低收缩",
            "low_shrinkage": "低收缩",
            "early_strength": "早强",
            "strength": "强度满足",
        }
        mapping.append({"token": labels.get(priority, priority), "field": "性能优先级", "value": labels.get(priority, priority), "reason": "用于决定目标函数和机理检索方向"})
    for material in requirements.get("mentioned_materials", []):
        mapping.append({"token": material, "field": "提及材料", "value": material, "reason": "仅用于语义理解和图谱检索，不自动改变配方边界"})
    for material in requirements.get("required_materials", []):
        mapping.append({"token": material, "field": "必用材料", "value": material, "reason": "作为后续优化的材料硬约束"})
    for material in requirements.get("allowed_materials", []):
        mapping.append({"token": material, "field": "允许材料", "value": material, "reason": "允许参与搜索，但不强制使用"})
    for material in requirements.get("avoided_materials", []):
        mapping.append({"token": material, "field": "禁用材料", "value": "0", "reason": "在优化边界中强制设为 0"})
    for spec in requirements.get("performance_constraints", []):
        if spec.get("metric") == "flowability":
            mapping.append({"token": spec.get("source", "流动性"), "field": "流动性约束", "value": f"{spec.get('lower', '-')} ~ {spec.get('upper', '-')}", "reason": "作为真实流动性预测器的数值约束"})
        if spec.get("metric") == "shrinkage":
            value = f"{spec.get('lower', '-')} ~ {spec.get('upper', '-')}" if spec.get("upper") is not None or spec.get("lower") is not None else spec.get("target", "low")
            mapping.append({"token": spec.get("source", "低收缩"), "field": "收缩约束", "value": value, "reason": "用于收缩风险识别、知识图谱检索和后续低收缩优化目标"})
    return mapping


def _select_plot_previews(result):
    plots = result.get("plots", {}) if result else {}
    wanted = [
        "Pareto Front",
        "Parallel Coordinates",
        "Progress History",
        "Spacing History",
        "Objective Correlation",
        "Objective Distribution",
        "Trade-off Matrix",
        "TOPSIS Screening",
        "Ranking Agreement",
        "Solution Score Heatmap",
        "Best Solution Profile",
        "Algorithm Performance Matrix",
        "Algorithm Ranking Agreement",
        "Algorithm Hypervolume History",
    ]
    previews = []
    for name in wanted:
        image = plots.get(name)
        if image:
            previews.append({"name": name, "image": image})
    return previews


def _optimization_trace(setup, kpis):
    return {
        "algorithm": setup.get("algorithm"),
        "compare_algorithms": setup.get("compare_algorithms", []),
        "objectives": setup.get("objectives", []),
        "constraints": setup.get("constraints", []),
        "bounds_count": len(setup.get("bounds", [])),
        "kpis": kpis,
        "material_policy": setup.get("material_policy", {}),
        "constraint_status": setup.get("constraint_status", {}),
        "traditional_theory": setup.get("traditional_theory", {}),
        "standard_constraint_visualization": setup.get("standard_constraint_visualization", {}),
    }


def _kg_trace(kg_payload):
    nodes = []
    links = []
    seen_nodes = set()
    seen_links = set()
    hit_materials = {hit.get("material") for hit in kg_payload.get("hits", [])}
    recovered_chains = _load_recovered_kg_chains()
    target_performances = set(kg_payload.get("target_performances") or [])
    query_terms = set(kg_payload.get("query_terms") or [])
    candidate_chains = [
        chain
        for chain in recovered_chains
        if chain["material"] in hit_materials and (not target_performances or chain["performance_en"] in target_performances)
    ]
    scored_chains = []
    for chain in candidate_chains:
        haystack = " ".join(
            [
                chain["material"],
                chain["feature"],
                chain["mechanism"],
                chain["consequence"],
                chain["performance"],
                chain["feature_en"],
                chain["mechanism_en"],
                chain["consequence_en"],
            ]
        ).lower()
        score = sum(1 for term in query_terms if term in haystack)
        if chain["material"] in query_terms:
            score += 3
        scored_chains.append((score, chain))
    per_material = {}
    for score, chain in sorted(scored_chains, key=lambda item: item[0], reverse=True):
        bucket = per_material.setdefault(chain["material"], [])
        if score > 0 and len(bucket) < 3:
            bucket.append(chain)
    if not any(per_material.values()):
        for _, chain in sorted(scored_chains, key=lambda item: item[0], reverse=True):
            bucket = per_material.setdefault(chain["material"], [])
            if len(bucket) < 2:
                bucket.append(chain)
    hit_chains = [chain for chains in per_material.values() for chain in chains]
    for idx, chain in enumerate(hit_chains):
        material = chain["material"]
        active = True
        chain_nodes = [
            (f"{material}-mat", "material", KG_CHAIN_LIBRARY.get(material, {}).get("label", material), KG_MATERIAL_LABEL_EN.get(material, material)),
            (f"{material}-feature-{idx}", "feature", chain["feature"], chain.get("feature_en")),
            (f"{material}-mechanism-{idx}", "mechanism", chain["mechanism"], chain.get("mechanism_en")),
            (f"{material}-consequence-{idx}", "consequence", chain["consequence"], chain.get("consequence_en")),
            (f"{material}-performance-{chain['performance_en']}", "performance", chain["performance"], chain.get("performance_label_en") or chain.get("performance_en")),
        ]
        for node_id, node_type, label, label_en in chain_nodes:
            if node_id not in seen_nodes:
                nodes.append({"id": node_id, "type": node_type, "label": label, "label_en": label_en or label, "material": material, "row": idx, "active": active})
                seen_nodes.add(node_id)
        for left, right, relation, relation_en in [
            (chain_nodes[0][0], chain_nodes[1][0], "has feature", "has feature"),
            (chain_nodes[1][0], chain_nodes[2][0], "causes", "causes"),
            (chain_nodes[2][0], chain_nodes[3][0], "produces", "produces"),
            (chain_nodes[3][0], chain_nodes[4][0], chain["relation"], chain.get("relation_en")),
        ]:
            link_key = (left, right, relation)
            if link_key not in seen_links:
                polarity = chain["relation"]
                evidence_role = "support"
                if any(token in polarity for token in ["降低", "不利", "风险", "损害"]):
                    evidence_role = "risk"
                links.append({"source": left, "target": right, "relation": relation, "relation_en": relation_en or relation, "material": material, "row": idx, "active": active, "performance": chain["performance"], "performance_en": chain.get("performance_en"), "polarity": polarity, "polarity_en": chain.get("relation_en"), "evidence_role": evidence_role})
                seen_links.add(link_key)
    for idx, item in enumerate(kg_payload.get("recommendations", [])[:4]):
        rec_id = f"recommendation-{idx}"
        nodes.append({"id": rec_id, "type": "recommendation", "label": item.get("topic", "建议"), "label_en": item.get("topic_en") or item.get("material") or "Recommendation"})
        if nodes:
            links.append({"source": nodes[0]["id"], "target": rec_id, "relation": "生成建议", "relation_en": "generates recommendation"})
    return {"nodes": nodes, "links": links}


def _recovered_kg_stats():
    chains = _load_recovered_kg_chains()
    materials = {chain["material"] for chain in chains}
    return {"chains": len(chains), "materials": len(materials)}


def _requirement_steps(requirements):
    return [
        {"title": "读取原始请求", "detail": requirements.get("raw_request", ""), "artifact": "raw_text"},
        {"title": "切分语义片段", "detail": f"得到 {len(requirements.get('raw_segments', []))} 个片段", "artifact": "segments"},
        {"title": "识别数值实体", "detail": f"目标强度 {requirements.get('target_mpa')} MPa；龄期 {requirements.get('age_days')} d", "artifact": "numeric_entities"},
        {"title": "识别材料意图", "detail": f"提及：{', '.join(requirements.get('mentioned_materials', [])) or '无'}；必用：{', '.join(requirements.get('required_materials', [])) or '无'}；允许：{', '.join(requirements.get('allowed_materials', [])) or '无'}；禁用：{', '.join(requirements.get('avoided_materials', [])) or '无'}", "artifact": "materials"},
        {
            "title": "确定配方策略",
            "detail": (
                "指定材料模式：未点名胶凝材料将被约束为 0"
                if requirements.get("material_mode") == "specified"
                else (
                    f"许可材料模式：仅在 {', '.join(requirements.get('allowed_materials', []))} 中搜索可选矿物掺合料"
                    if requirements.get("allowed_materials")
                    else "探索模式：自动比较多个矿物掺合料候选配方族"
                )
            ),
            "artifact": "material_policy",
        },
        {"title": "排序设计优先级", "detail": " > ".join(item["interpretation"] for item in requirements.get("priority_ranking", [])), "artifact": "priority_ranking"},
        {"title": "生成下游规格", "detail": "形成强度、龄期、优先级、材料约束与待验证项", "artifact": "design_spec"},
    ]


def _optimization_steps(setup, result, kpis):
    wb_window = setup.get("traditional_theory", {}).get("water_binder_window", {})
    return [
        {"title": "固定预测输入", "detail": "将用户指定龄期固定，不作为搜索变量", "artifact": "fixed_inputs"},
        {"title": "鲍罗米先验收缩", "detail": f"先反推水胶比可行区间 {wb_window.get('min')}~{wb_window.get('max')}，再进入多目标搜索", "artifact": "bolomey_prior"},
        {"title": "装配搜索变量", "detail": f"共装配 {len(setup.get('bounds', []))} 个变量边界", "artifact": "bounds"},
        {"title": "构造目标函数", "detail": f"目标函数 {len(setup.get('objectives', []))} 个", "artifact": "objectives"},
        {"title": "构造约束条件", "detail": "；".join(setup.get("constraints", [])), "artifact": "constraints"},
        {"title": "运行多目标算法", "detail": f"{setup.get('algorithm')} + 对比算法 {', '.join(setup.get('compare_algorithms', []))}", "artifact": "algorithm"},
        {"title": "生成 Pareto 解集", "detail": f"选定算法 {result.get('best_algorithm')}；传统先验复筛后保留 {result.get('prior_feasible_count', 0)} 个可行解", "artifact": "pareto"},
        {"title": "筛选推荐方案", "detail": f"预测强度 {kpis.get('strength_mpa', '-')} MPa；碳排放 {kpis.get('carbon_kgco2e_m3', '-')}", "artifact": "selected_solution"},
    ]


def _kg_steps(requirements, kg_payload):
    graph_stats = _kg_trace(kg_payload)
    source_stats = _recovered_kg_stats()
    evidence = kg_payload.get("evidence", {})
    return [
        {"title": "构造查询词", "detail": "、".join(requirements.get("priorities", [])), "artifact": "query_terms"},
        {"title": "统一证据召回", "detail": f"文献块 {evidence.get('counts', {}).get('literature', 0)}、规范块 {evidence.get('counts', {}).get('standard', 0)}、图谱链 {evidence.get('counts', {}).get('kg', 0)} 进入同一索引", "artifact": "unified_index"},
        {"title": "执行混合检索", "detail": f"召回候选 {len(evidence.get('candidates', []))} 条，选中高相关证据 {len(evidence.get('selected', []))} 条", "artifact": "hybrid_retrieval"},
        {"title": "抽取多跳路线", "detail": f"保留 {len(evidence.get('multi_hop_routes', []))} 条材料到性能的因果路线", "artifact": "multi_hop_routes"},
        {"title": "载入恢复图谱", "detail": f"底图包含 {source_stats['materials']} 类材料、{source_stats['chains']} 条因果链", "artifact": "full_graph"},
        {"title": "裁剪命中子图", "detail": f"本次保留 {len(graph_stats.get('nodes', []))} 个命中节点、{len(graph_stats.get('links', []))} 条命中关系", "artifact": "hit_subgraph"},
        {"title": "激活候选材料节点", "detail": "、".join(hit.get("material", "") for hit in kg_payload.get("hits", [])), "artifact": "material_nodes"},
        {"title": "沿因果链扩展", "detail": "材料 -> 特征 -> 机理 -> 后果 -> 性能", "artifact": "causal_expansion"},
        {"title": "识别正向支持", "detail": f"命中 {len(kg_payload.get('hits', []))} 类材料机理", "artifact": "supporting_paths"},
        {"title": "识别修正建议", "detail": f"生成 {len(kg_payload.get('recommendations', []))} 条机理反馈", "artifact": "recommendations"},
        {"title": "输出下一轮候选", "detail": "形成可进入实验或下一轮优化的材料调整建议", "artifact": "feedback"},
    ]


def _json_event(event, **payload):
    return json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n"


def _eventful_main_algorithm(bounds, objectives, constraints, algorithm_name, prediction_fn, pop_size, generations):
    opt = ml_optimization
    rng = np.random.default_rng(42)
    mode = opt.ALGORITHM_REGISTRY.get(algorithm_name, "nsga2")
    population = opt._initialize_population(bounds, pop_size, rng)
    velocity = np.zeros_like(population)
    archive_pop = population.copy()
    archive_evals = opt._evaluate_population(archive_pop, bounds, objectives, constraints, prediction_fn)
    yield {"phase": "population_initialized", "population_size": int(len(population)), "algorithm": algorithm_name}

    for generation in range(1, generations + 1):
        evals = opt._evaluate_population(population, bounds, objectives, constraints, prediction_fn)
        combined_pop = np.vstack([archive_pop, population])
        combined_eval = opt._evaluate_population(combined_pop, bounds, objectives, constraints, prediction_fn)
        archive_pop, archive_evals = opt._nondominated_archive(combined_pop, combined_eval)
        population, evals = opt._select_by_fronts(population, evals, min(pop_size, len(population)))
        yield {
            "phase": "generation_completed",
            "generation": generation,
            "generations": generations,
            "hypervolume": round(float(opt._approx_hypervolume(archive_evals["transformed"])), 5),
            "feasible_ratio": round(float(np.mean(evals["constraint_violation"] <= 1e-9)), 5),
            "nondominated": int(len(archive_evals["frame"])),
            "spacing": round(float(opt._spacing_metric(archive_evals["transformed"])), 5),
            "objective_names": archive_evals.get("objective_names", []),
            "pareto_points": np.asarray(archive_evals["raw_objectives"], dtype=float)[:80].round(5).tolist(),
        }
        if mode == "nsga2":
            offspring = opt._blend_crossover(population[rng.integers(0, len(population), size=len(population))], bounds, rng)
            population = np.vstack([population, offspring])
            population, _ = opt._select_by_fronts(population, opt._evaluate_population(population, bounds, objectives, constraints, prediction_fn), pop_size)
        elif mode == "de":
            trial = opt._de_step(population, bounds, rng)
            population = np.vstack([population, trial])
            population, _ = opt._select_by_fronts(population, opt._evaluate_population(population, bounds, objectives, constraints, prediction_fn), pop_size)
        elif mode == "mopso":
            leaders = archive_pop if len(archive_pop) else population
            population, velocity = opt._mopso_step(population, leaders, velocity, bounds, rng)
        elif mode == "moead":
            trial = opt._weighted_step(population, evals, bounds, rng)
            population = np.vstack([population, trial])
            population, _ = opt._select_by_fronts(population, opt._evaluate_population(population, bounds, objectives, constraints, prediction_fn), pop_size)
        elif mode == "weighted":
            population = opt._weighted_step(population, evals, bounds, rng)
        elif mode == "epsilon":
            population = opt._epsilon_constraint_step(population, evals, bounds, rng)
        elif mode == "anneal":
            temperature = max(0.02, 1 - generation / max(generations, 1))
            proposal, current_scores = opt._anneal_step(population, evals, bounds, temperature, rng)
            proposal_eval = opt._evaluate_population(proposal, bounds, objectives, constraints, prediction_fn)
            proposal_scores = proposal_eval["transformed"].sum(axis=1) + 10 * proposal_eval["constraint_violation"]
            accept = proposal_scores < current_scores + rng.random(len(current_scores)) * temperature
            population[accept] = proposal[accept]
        else:
            population = opt._initialize_population(bounds, pop_size, rng)


def _compact_for_lm(prompt, optimization, kg_hits, agents=None):
    opt = optimization or {}
    compact_agents = []
    for agent in agents or []:
        compact_agents.append({
            "name": agent.get("name"),
            "summary": agent.get("summary"),
        })
    solution = opt.get("solution_table", [])[:10]
    objectives = []
    for item in opt.get("objectives", [])[:6]:
        objectives.append({
            "name": item.get("name"),
            "goal": item.get("goal"),
            "expression": str(item.get("expression", ""))[:220],
            "target": item.get("target"),
        })
    kg_items = []
    if isinstance(kg_hits, dict):
        for hit in kg_hits.get("hits", [])[:5]:
            kg_items.append({"material": hit.get("material"), "evidence": str(hit.get("evidence", ""))[:120]})
        recommendations = [
            {
                "topic": item.get("topic"),
                "mechanism": str(item.get("mechanism", ""))[:120],
                "recommendation": str(item.get("recommendation", ""))[:160],
            }
            for item in kg_hits.get("recommendations", [])[:4]
        ]
        citations = [
            {
                "ref_no": item.get("ref_no"),
                "id": item.get("id"),
                "title": item.get("title"),
                "source_type": item.get("source_type"),
                "grade": item.get("grade"),
                "excerpt": str(item.get("excerpt", ""))[:180],
                "metadata": item.get("metadata", {}),
                "reference": _format_reference_entry(item),
            }
            for item in kg_hits.get("citations", [])[:18]
        ]
    else:
        kg_items = kg_hits[:5] if isinstance(kg_hits, list) else []
        recommendations = []
        citations = []
    compact = {
        "user_request": (prompt or "")[:500],
        "agents": compact_agents,
        "target_mpa": opt.get("target_mpa"),
        "age_days": opt.get("age_days"),
        "model_name": opt.get("model_name"),
        "best_algorithm": opt.get("best_algorithm"),
        "kpis": opt.get("kpis", {}),
        "ratios": opt.get("ratios", {}),
        "candidate_schemes": opt.get("candidate_schemes", []),
        "scheme_comparison": opt.get("scheme_comparison", []),
        "trust_profile": opt.get("trust_profile", {}),
        "method_baselines": opt.get("method_baselines", []),
        "evidence_chain": opt.get("evidence_chain", []),
        "traditional_theory": opt.get("traditional_theory", {}),
        "solution_table": solution,
        "objectives": objectives,
        "constraints": opt.get("constraints", [])[:4],
        "kg_hits": kg_items,
        "recommendations": recommendations,
        "citations": citations,
    }
    return compact


def _format_reference_entry(item):
    ref_no = item.get("ref_no")
    source_type = item.get("source_type")
    metadata = item.get("metadata", {}) or {}
    prefix = f"[{ref_no}] " if ref_no else ""
    if source_type == "literature":
        authors = "; ".join(metadata.get("authors", [])[:6])
        if len(metadata.get("authors", [])) > 6:
            authors += "; et al."
        title = item.get("title") or ""
        journal = metadata.get("journal") or ""
        year = metadata.get("year") or ""
        doi = metadata.get("doi") or ""
        entry = f"{prefix}{authors}. {title}[J]. {journal}, {year}."
        return entry + (f" DOI: {doi}." if doi else "")
    if source_type == "patent":
        applicant = metadata.get("applicant") or "未知申请人"
        title = item.get("title") or metadata.get("title") or "未命名专利"
        country = metadata.get("country") or "CN"
        patent_no = metadata.get("patent_no") or "未提取专利号"
        publication_date = metadata.get("publication_date") or "未提取公开日"
        return f"{prefix}{applicant}. {title}[P]. {country}: {patent_no}, {publication_date}."
    if source_type == "standard":
        title = item.get("title") or ""
        standard_no = metadata.get("standard_no") or ""
        year = metadata.get("year") or ""
        lead = f"{standard_no}. " if standard_no else ""
        tail = f", {year}" if year else ""
        return f"{prefix}{lead}{title}[S]{tail}."
    route = item.get("route", []) or []
    route_text = " -> ".join(route)
    if source_type == "auto_kg":
        extractor = metadata.get("extractor") or "hybrid"
        return prefix + f"{item.get('title')}[AKG]. 自动抽取知识图谱，抽取器：{extractor}" + (f": {route_text}." if route_text else ".")
    return prefix + f"{item.get('title')}[KG]" + (f": {route_text}." if route_text else ".")


def _ensure_reference_section(text, citations):
    body = str(text or "").rstrip()
    if not citations:
        return body
    if "参考文献与证据来源" in body:
        return body
    entries = "\n".join(_format_reference_entry(item) for item in citations if item)
    return f"{body}\n\n## 参考文献与证据来源\n{entries}"


def _fallback_answer(prompt, optimization, kg_hits, error_text, language="zh"):
    language = _normalize_language(language)
    opt = optimization or {}
    kpis = opt.get("kpis", {})
    rows = opt.get("solution_table", [])[:10]
    row_text = "\n".join(
        f"- {item.get('name')}: {item.get('value')} {item.get('unit', '')}".strip()
        for item in rows
    ) or ("- No displayable mix-design variables were returned." if language == "en" else "- 未获得可展示配合比变量。")
    recs = kg_hits.get("recommendations", []) if isinstance(kg_hits, dict) else []
    rec_text = "\n".join(
        f"- {item.get('topic')}: {item.get('recommendation')}"
        for item in recs[:4]
    ) or (
        "- Validate low water-to-binder ratio, clinker substitution by supplementary cementitious materials, PCE dispersion, and aggregate grading through trial mixing."
        if language == "en"
        else "- 建议围绕低水胶比、矿物掺合料替代熟料、PCE 分散和骨料级配进行试配验证。"
    )
    if language == "en":
        return (
            "LM Studio did not provide sufficient context, so a local rule-based technical summary was generated.\n\n"
            f"User request: {prompt}\n\n"
            "Key metrics:\n"
            f"- Target strength: {opt.get('target_mpa', '-') } MPa\n"
            f"- Curing age: {opt.get('age_days', '-') } d\n"
            f"- Predicted strength: {kpis.get('strength_mpa', '-') } MPa\n"
            f"- Carbon footprint: {kpis.get('carbon_kgco2e_m3', '-') } kgCO2e/m3\n"
            f"- Shrinkage-risk index: {kpis.get('shrinkage_risk_index', '-') }\n"
            f"- Cost index: {kpis.get('cost_index', '-') }\n"
            f"- Flowability: {kpis.get('flowability', 'No flowability predictor was provided by the model package') }\n\n"
            f"- Water-to-binder ratio: {opt.get('ratios', {}).get('water_binder_ratio', 'not available')}\n"
            f"- Paste-to-aggregate ratio: {opt.get('ratios', {}).get('paste_aggregate_ratio', 'not available')}\n\n"
            "Recommended mix design:\n"
            f"{row_text}\n\n"
            "Mechanistic recommendations:\n"
            f"{rec_text}\n\n"
            "Note: The system does not fabricate heuristic flowability proxies. If the uploaded model package does not include a flowability predictor, that objective is not converted into a numerical output."
            f"\n\nLM Studio error: {error_text}"
        )
    return (
        "LM Studio 当前上下文不足，已使用本地规则化总结输出。\n\n"
        f"用户需求：{prompt}\n\n"
        "关键指标：\n"
        f"- 目标强度：{opt.get('target_mpa', '-') } MPa\n"
        f"- 龄期：{opt.get('age_days', '-') } d\n"
        f"- 预测强度：{kpis.get('strength_mpa', '-') } MPa\n"
            f"- 碳排放：{kpis.get('carbon_kgco2e_m3', '-') } kgCO2e/m3\n"
            f"- 收缩风险指数：{kpis.get('shrinkage_risk_index', '-') }\n"
            f"- 成本指数：{kpis.get('cost_index', '-') }\n"
        f"- 流动性：{kpis.get('flowability', '模型未提供流动性输出') }\n\n"
        f"- 水胶比：{opt.get('ratios', {}).get('water_binder_ratio', '当前模型未提供')}\n"
        f"- 浆骨比：{opt.get('ratios', {}).get('paste_aggregate_ratio', '当前模型未提供')}\n\n"
        "推荐配合比：\n"
        f"{row_text}\n\n"
        "机理建议：\n"
        f"{rec_text}\n\n"
        "说明：系统不使用启发式流动性代理；若模型包未提供流动性预测器，则该目标不会被伪造为数值输出。"
        f"\n\nLM Studio 错误：{error_text}"
    )


def _chat_lm(system_prompt, user_payload, max_tokens=1800, temperature=0.25):
    body = {
        "model": LM_STUDIO_MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False) if not isinstance(user_payload, str) else user_payload,
            },
        ],
    }
    req = urllib.request.Request(
        LM_STUDIO_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _extract_json_object(text):
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("LLM did not return a JSON object")
    return json.loads(raw[start:end + 1])


def _chat_lm_json(system_prompt, user_payload, max_tokens=1400, temperature=0.1):
    return _extract_json_object(_chat_lm(system_prompt, user_payload, max_tokens=max_tokens, temperature=temperature))


def _normalize_materials(items):
    allowed = set(MATERIAL_TERMS)
    return [item for item in items or [] if item in allowed]


def _normalize_text_list(items):
    if items is None:
        return []
    if isinstance(items, str):
        value = items.strip()
        return [value] if value else []
    if not isinstance(items, list):
        return [str(items)]
    return [str(item).strip() for item in items if str(item).strip()]


def _normalize_performance_constraints(items):
    normalized = []
    for item in items or []:
        if not isinstance(item, dict) or item.get("metric") not in {"flowability", "shrinkage"}:
            continue
        metric = item.get("metric")
        lower = item.get("lower")
        upper = item.get("upper")
        lower = float(lower) if isinstance(lower, (int, float)) else None
        upper = float(upper) if isinstance(upper, (int, float)) else None
        if lower is not None and upper is not None and lower > upper:
            lower, upper = upper, lower
        if lower is None and upper is None and metric != "shrinkage":
            continue
        normalized.append(
            {
                "metric": metric,
                "lower": lower,
                "upper": upper,
                "unit": item.get("unit") or ("microstrain" if metric == "shrinkage" else "model_unit"),
                "source": str(item.get("source") or ""),
                **({"shrinkage_type": item.get("shrinkage_type") or "total", "age_days": item.get("age_days"), "target": item.get("target")} if metric == "shrinkage" else {}),
            }
        )
    return normalized


def _validated_requirement_spec(prompt, explicit_target=None):
    fallback = _analyze_user_request(prompt, explicit_target)
    proposal_prompt = """
你是高强低碳混凝土需求解析智能体。只输出一个 JSON 对象，不要输出 Markdown。
字段必须包括：
target_mpa(number), age_days(number), priorities(array), mentioned_materials(array), required_materials(array), allowed_materials(array), avoided_materials(array),
performance_constraints(array of {metric,lower,upper,unit,source}),
raw_segments(array of {text,role,interpretation}), rationale(array), open_questions(array)。
priorities 只允许 low_carbon, high_flowability, low_shrinkage, early_strength, strength。
materials 只允许 cement, flyash, GGBFS, silicafume, limestone, metakaolin, ricehuskash, water, superplasticizer, fineaggregate, coarseaggregate, steelfiber。
如果用户未明确提供，不要编造；可留空数组。龄期未给时可采用 28 d 并在 rationale 说明为默认值。
“提到某材料”不等于“要求必须使用某材料”；只有用户明确说要求使用、必须加入、采用、掺入等，才能放入 required_materials。
若用户只说允许使用某材料，放入 allowed_materials；若只是讨论某材料机理，放入 mentioned_materials 即可，不得改变配方边界。
请特别理解自然语言中的多样表达，不要只依赖固定词：
- “想用粉煤灰和硅灰”“希望配方里有粉煤灰与硅灰”“粉煤灰+硅灰体系”“优先采用粉煤灰、硅灰”通常表示 required_materials。
- “粉煤灰可以用”“允许矿渣参与”“可考虑硅灰”通常表示 allowed_materials，而不是 required_materials。
- “不要矿渣”“不想加矿粉”“避免使用硅灰”“无粉煤灰体系”表示 avoided_materials。
- “流动性大于500”“扩展度不小于500”“流动度控制在500到650”“坍落扩展度最好低于650”都要进入 performance_constraints。
- “低收缩”“控制干燥收缩”“降低自收缩”“56d 收缩小于400微应变”要识别为 low_shrinkage，并在 performance_constraints 中使用 metric=shrinkage；若有数值上限，upper 使用微应变数值，unit=microstrain，并标注 shrinkage_type=drying/autogenous/total。
- “低收缩”“收缩风险低”“控制收缩变形”“提高低收缩性”要识别为 low_shrinkage；若有收缩变形、收缩龄期或残余抗拉指标，应进入 performance_constraints。
冲突规则：
- 若同一材料同时被识别为 required 与 avoided，以 avoided 为准，并在 rationale 中说明冲突。
- 若用户只是在解释机理，例如“粉煤灰会改善流动性吗”，不得把粉煤灰放入 required_materials。
""".strip()
    try:
        proposal = _chat_lm_json(
            proposal_prompt,
            {
                "user_request": prompt,
                "non_binding_parser_hints": {
                    "candidate_material_mentions": fallback.get("mentioned_materials", []),
                    "candidate_material_requirements": fallback.get("required_materials", []),
                    "candidate_material_permissions": fallback.get("allowed_materials", []),
                    "candidate_material_avoidance": fallback.get("avoided_materials", []),
                    "candidate_performance_constraints": fallback.get("performance_constraints", []),
                },
                "instruction": "parser_hints 只是候选提示；请以用户真实语义为准，自主判断，不要机械照抄。",
            },
            max_tokens=1800,
        )
    except Exception:
        fallback["llm_spec_ok"] = False
        fallback["llm_spec"] = None
        fallback["semantic_interpreter"] = "rule_fallback"
        fallback["design_domains"] = _document_domain_hints(fallback)
        fallback["normative_requirements"] = _compile_domain_requirements(fallback)
        fallback["material_mode"] = "specified" if set(fallback["required_materials"]) & SUPPLEMENTARY_BINDERS else "exploratory"
        return fallback
    priorities = [item for item in proposal.get("priorities", []) if item in {"low_carbon", "high_flowability", "low_shrinkage", "early_strength", "strength"}]
    mentioned = _normalize_materials(proposal.get("mentioned_materials"))
    required = _normalize_materials(proposal.get("required_materials"))
    allowed_materials = _normalize_materials(proposal.get("allowed_materials"))
    avoided = _normalize_materials(proposal.get("avoided_materials"))
    performance_constraints = _normalize_performance_constraints(proposal.get("performance_constraints"))
    fallback_constraints = fallback.get("performance_constraints", [])
    seen_constraints = {(item.get("metric"), item.get("source"), item.get("upper"), item.get("lower")) for item in performance_constraints}
    for item in fallback_constraints:
        key = (item.get("metric"), item.get("source"), item.get("upper"), item.get("lower"))
        if key not in seen_constraints:
            performance_constraints.append(item)
            seen_constraints.add(key)
    for item in fallback.get("priorities", []):
        if item not in priorities and item in {"low_carbon", "high_flowability", "low_shrinkage", "early_strength", "strength"}:
            priorities.append(item)
    fallback_allowed = set(fallback.get("allowed_materials", []))
    fallback_required = set(fallback.get("required_materials", []))
    permission_demotions = sorted((set(required) & fallback_allowed) - fallback_required)
    if permission_demotions:
        required = [item for item in required if item not in permission_demotions]
        allowed_materials = list(dict.fromkeys([*allowed_materials, *permission_demotions]))
    avoided_set = set(avoided)
    conflicts = sorted(set(required) & avoided_set)
    required = [item for item in required if item not in avoided_set]
    allowed_materials = [item for item in allowed_materials if item not in avoided_set and item not in required]
    mentioned = list(dict.fromkeys([*mentioned, *required, *allowed_materials, *avoided]))
    spec = dict(fallback)
    spec.update(
        {
            "target_mpa": float(proposal.get("target_mpa") or fallback["target_mpa"]),
            "age_days": float(proposal.get("age_days") or fallback["age_days"]),
            "priorities": priorities or fallback["priorities"],
            "mentioned_materials": mentioned,
            "required_materials": required,
            "allowed_materials": allowed_materials,
            "avoided_materials": avoided,
            "performance_constraints": performance_constraints,
            "raw_segments": proposal.get("raw_segments") or fallback["raw_segments"],
            "rationale": _normalize_text_list(proposal.get("rationale")),
            "open_questions": _normalize_text_list(proposal.get("open_questions")),
            "validation_notes": [
                *([f"{item} 同时被识别为必用与禁用，程序按禁用优先处理。" for item in conflicts]),
                *([f"{item} 出现在“允许使用”语义中，程序已从必用材料校正为允许材料。" for item in permission_demotions]),
            ],
            "llm_spec_ok": True,
            "llm_spec": proposal,
            "semantic_interpreter": "llm_primary_with_program_validation",
            "parser_hints": {
                "mentioned_materials": fallback.get("mentioned_materials", []),
                "required_materials": fallback.get("required_materials", []),
                "allowed_materials": fallback.get("allowed_materials", []),
                "avoided_materials": fallback.get("avoided_materials", []),
                "performance_constraints": fallback.get("performance_constraints", []),
            },
        }
    )
    spec["design_domains"] = _document_domain_hints(spec)
    spec["normative_requirements"] = _compile_domain_requirements(spec)
    spec["material_mode"] = "specified" if set(spec["required_materials"]) & SUPPLEMENTARY_BINDERS else "exploratory"
    return spec


def _optimizer_llm_proposal(requirements, flow_alias):
    prompt = """
你是优化建模智能体。只输出 JSON，不要输出 Markdown。
请在允许的候选目标中选择目标函数，并给出排序权重建议：
allowed_objectives = StrengthTarget, Carbon, Cost, Flowability, ShrinkageRisk。
allowed_constraints = strength_target, water_binder_ratio, paste_aggregate_ratio, practical_minimums, fixed_age, user_performance_interval。
返回：
{"objectives":[...],"constraints":[...],"weights":{...},"rationale":[...]}
规则：StrengthTarget 必须存在；若用户要求高流动性且提供 flowability_alias 才能选择 Flowability；若用户要求低收缩且候选目标中包含 ShrinkageRisk，则可以选择 ShrinkageRisk，但必须理解为机理代理指标而非实测微应变预测；若用户给出流动性区间，必须选择 user_performance_interval；不得虚构目标。
""".strip()
    try:
        return _chat_lm_json(prompt, {"requirements": requirements, "flowability_alias": flow_alias}, max_tokens=1200)
    except Exception:
        return {"objectives": [], "constraints": [], "weights": {}, "rationale": [], "llm_ok": False}


def _apply_optimizer_proposal(setup, proposal):
    allowed = {item["name"]: item for item in setup["objectives"]}
    selected = [name for name in proposal.get("objectives", []) if name in allowed]
    if "StrengthTarget" not in selected:
        selected.insert(0, "StrengthTarget")
    if "ShrinkageRisk" in allowed and "ShrinkageRisk" not in selected:
        selected.append("ShrinkageRisk")
    if selected:
        setup["objectives"] = [allowed[name] for name in selected]
    setup["llm_proposal"] = {
        "objectives": selected or [item["name"] for item in setup["objectives"]],
        "constraints": proposal.get("constraints", []),
        "weights": proposal.get("weights", {}),
        "rationale": proposal.get("rationale", []),
    }
    return setup


def _mechanism_review_llm(requirements, kg_payload, optimization_summary):
    prompt = """
你是机理复审智能体。只输出 JSON，不要输出 Markdown。
判断知识图谱命中后是否应触发第二轮优化。
返回：
{
  "trigger_second_pass": boolean,
  "reason": string,
  "bound_adjustments":[{"material":string,"lower":number|null,"upper":number|null,"reason":string}],
  "requested_validations":[string]
}
只在当前推荐方案存在明确机理风险时才触发第二轮优化；材料名只能来自现有材料词表；数值调整必须保守。
""".strip()
    try:
        return _chat_lm_json(
            prompt,
            {"requirements": requirements, "kg": kg_payload, "optimization": optimization_summary},
            max_tokens=1400,
        )
    except Exception:
        return {"trigger_second_pass": False, "reason": "LLM review unavailable", "bound_adjustments": [], "requested_validations": []}


def _evidence_second_pass_review(requirements, kg_payload, optimization_summary, feature_map):
    selected = kg_payload.get("evidence", {}).get("selected", [])
    negative_routes = [
        item for item in selected
        if item.get("source_type") in {"kg", "auto_kg"}
        and any(token in str(item.get("relation", "")) for token in ["降低", "decrease"])
    ]
    if not negative_routes:
        return {"trigger_second_pass": False, "reason": "未发现足以触发二次优化的高相关负向机理链。", "bound_adjustments": [], "requested_validations": []}
    best_rows = optimization_summary.get("solution_table", [])
    by_material = {row.get("material"): row.get("value") for row in best_rows}
    adjustments = []
    validations = []
    high_flow = "high_flowability" in requirements.get("priorities", [])
    if high_flow and any(item.get("material") == "silicafume" for item in negative_routes) and by_material.get("silicafume", 0):
        safe = _feature_safe_by_material(feature_map, "silicafume")
        if safe:
            upper = round(float(by_material.get("silicafume", 0)) * 0.92, 4)
            adjustments.append({"material": "silicafume", "lower": None, "upper": upper, "reason": "高流动性目标下命中硅灰增黏负向机理链，保守压缩上界。"})
            validations.append("开展硅灰-PCE 适配和扩展度保持试验。")
    if high_flow and any(item.get("material") == "flyash" for item in selected if item.get("source_type") in {"kg", "auto_kg"}):
        validations.append("比较普通粉煤灰与粉煤灰微珠的流动性收益。")
    return {
        "trigger_second_pass": bool(adjustments),
        "reason": adjustments[0]["reason"] if adjustments else "存在负向机理证据，但当前未形成可安全执行的边界修正。",
        "bound_adjustments": adjustments,
        "requested_validations": validations,
        "evidence_ids": [item["id"] for item in negative_routes[:4]],
    }


def _merge_mechanism_reviews(llm_review, evidence_review):
    if evidence_review.get("trigger_second_pass"):
        merged = dict(llm_review or {})
        merged["trigger_second_pass"] = True
        merged["reason"] = evidence_review.get("reason") or merged.get("reason")
        merged["bound_adjustments"] = [
            *(llm_review or {}).get("bound_adjustments", []),
            *evidence_review.get("bound_adjustments", []),
        ]
        merged["requested_validations"] = list(dict.fromkeys([
            *(llm_review or {}).get("requested_validations", []),
            *evidence_review.get("requested_validations", []),
        ]))
        merged["evidence_ids"] = evidence_review.get("evidence_ids", [])
        merged["trigger_source"] = "hybrid_evidence"
        return merged
    review = dict(llm_review or {})
    review.setdefault("trigger_source", "llm_review")
    review.setdefault("evidence_ids", [])
    return review


def _apply_bound_adjustments(setup, feature_map, review):
    adjusted = json.loads(json.dumps(setup))
    by_name = {item["name"]: item for item in adjusted["bounds"]}
    applied = []
    for item in review.get("bound_adjustments", [])[:4]:
        safe = _feature_safe_by_material(feature_map, item.get("material"))
        if not safe or safe not in by_name:
            continue
        bound = by_name[safe]
        lower = item.get("lower")
        upper = item.get("upper")
        if isinstance(lower, (int, float)):
            bound["lower"] = max(bound["lower"], float(lower))
        if isinstance(upper, (int, float)):
            bound["upper"] = min(bound["upper"], float(upper))
        if bound["upper"] > bound["lower"]:
            applied.append(item)
    adjusted["mechanism_adjustments"] = applied
    return adjusted, applied


def _report_artifact_request_llm(prompt, requirements, optimization, kg_payload):
    system_prompt = """
你是总报告智能体的制图与补充材料规划模块。只输出 JSON。
返回：
{
 "requested_figures":[string],
 "requested_tables":[string],
 "requested_experiments":[string],
 "reasoning":[string]
}
内容必须紧扣当前设计任务，不要泛泛罗列。
""".strip()
    try:
        return _chat_lm_json(
            system_prompt,
            {"user_request": prompt, "requirements": requirements, "optimization": optimization, "kg": kg_payload},
            max_tokens=1200,
        )
    except Exception:
        return {"requested_figures": [], "requested_tables": [], "requested_experiments": [], "reasoning": []}


def _stream_chat_lm(system_prompt, user_payload, max_tokens=1800, temperature=0.25):
    body = {
        "model": LM_STUDIO_MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False) if not isinstance(user_payload, str) else user_payload,
            },
        ],
    }
    req = urllib.request.Request(
        LM_STUDIO_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = []
    finish_reason = None
    with urllib.request.urlopen(req, timeout=240) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choice = (data.get("choices") or [{}])[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = (choice.get("delta") or {}).get("content")
            if delta:
                chunks.append(delta)
                yield delta
    if finish_reason == "length":
        yield "\n\n[Output reached the model token limit. Increase max_tokens or request continuation for a complete verbatim agent transcript.]"
    if not chunks:
        text = _chat_lm(system_prompt, user_payload, max_tokens=max_tokens, temperature=temperature)
        for start in range(0, len(text), 48):
            yield text[start:start + 48]


def _run_agent_dialogues(prompt, requirements, optimization, kg_payload, agents, language="zh"):
    language = _normalize_language(language)
    compact = _compact_for_lm(prompt, optimization, kg_payload, agents)
    role_evidence = kg_payload.get("role_evidence") or _role_evidence_bundle(requirements)
    shared_evidence = _citations_from_evidence(role_evidence.get("merged", {}))
    compact["citations"] = shared_evidence
    outputs = {}
    ok = True
    try:
        outputs["requirement"] = _chat_lm(
            _load_skill_prompt("requirement_analysis", AGENT_PROMPTS["requirement"], language),
            {"user_request": prompt, "parsed_spec": requirements, "shared_evidence": role_evidence["requirement_analysis"].get("citations", []), "retrieval_policy": role_evidence["requirement_analysis"].get("policy", {})},
            max_tokens=10000,
        )
        outputs["optimizer"] = _chat_lm(
            _load_skill_prompt("optimization_modeling", AGENT_PROMPTS["optimizer"], language),
            {"requirement_agent_output": outputs["requirement"], "optimization_summary": compact, "shared_evidence": role_evidence["optimization_modeling"].get("citations", []), "retrieval_policy": role_evidence["optimization_modeling"].get("policy", {})},
            max_tokens=10000,
        )
        outputs["kg"] = _chat_lm(
            _load_skill_prompt("mechanism_retrieval", AGENT_PROMPTS["kg"], language),
            {
                "user_request": prompt,
                "optimization_summary": compact,
                "knowledge_graph_hits": compact.get("kg_hits"),
                "recommendations": compact.get("recommendations"),
                "shared_evidence": role_evidence["mechanism_retrieval"].get("citations", []),
                "retrieval_policy": role_evidence["mechanism_retrieval"].get("policy", {}),
            },
            max_tokens=10000,
        )
        final_prompt = _load_skill_prompt("report_generation", REPORT_SKILL_PROMPT, language)
        outputs["final"] = _chat_lm(
            final_prompt,
            {
                "user_request": prompt,
                "requirement_agent_output": outputs["requirement"],
                "optimizer_agent_output": outputs["optimizer"],
                "kg_agent_output": outputs["kg"],
                "structured_result": compact,
                "shared_evidence": shared_evidence,
                "role_evidence": {key: value.get("citations", []) for key, value in role_evidence.items() if key != "merged"},
            },
            max_tokens=24000,
            temperature=0.3,
        )
        outputs["final"] = _ensure_reference_section(outputs["final"], shared_evidence)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        ok = False
        outputs["error"] = str(exc)
        outputs["final"] = _fallback_answer(prompt, optimization, kg_payload, str(exc), language)
    return outputs, ok


@app.get("/api/status")
async def status():
    package = STATE.get("package")
    return {
        "status": "success",
        "ml_core_loaded": IMPORT_ERROR is None,
        "ml_core_error": IMPORT_ERROR,
        "model_loaded": bool(package),
        "package_name": STATE.get("package_name"),
        "package_type": package.get("package_type") if package else None,
        "lm_studio_url": LM_STUDIO_URL,
    }


@app.get("/api/skills")
async def list_skills(language: str = "zh"):
    language = _normalize_language(language)
    return {"status": "success", "skills": [_read_skill(skill_id, language) for skill_id in DEFAULT_SKILLS]}


@app.get("/api/skills/{skill_id}")
async def get_skill(skill_id: str, language: str = "zh"):
    skill = _read_skill(skill_id, _normalize_language(language))
    if not skill:
        return _json_error("Skill 不存在。", 404)
    return {"status": "success", "skill": skill}


def _skill_evaluation(skill):
    contract = skill.get("io_contract", {})
    retrieval = skill.get("retrieval_policy", {})
    toolchain = skill.get("toolchain", [])
    validators = skill.get("validators", [])
    reflection = skill.get("reflection_loop", [])
    benchmarks = skill.get("benchmark_cases", [])
    checks = [
        {"name": "输入契约", "passed": bool(contract.get("required_inputs")), "detail": f"{len(contract.get('required_inputs', []))} 项"},
        {"name": "输出契约", "passed": bool(contract.get("required_outputs")), "detail": f"{len(contract.get('required_outputs', []))} 项"},
        {"name": "证据路由", "passed": len(retrieval.get("primary_sources", [])) >= 2, "detail": " / ".join(retrieval.get("primary_sources", []))},
        {"name": "工具链", "passed": len(toolchain) >= 3, "detail": f"{len(toolchain)} 个工具"},
        {"name": "验证规则", "passed": len(validators) >= 4, "detail": f"{len(validators)} 条规则"},
        {"name": "反思闭环", "passed": len(reflection) >= 3, "detail": f"{len(reflection)} 条复核"},
        {"name": "基准样例", "passed": len(benchmarks) >= 3, "detail": f"{len(benchmarks)} 个 case"},
    ]
    score = round(100 * sum(1 for item in checks if item["passed"]) / len(checks), 1)
    return {"skill_id": skill["id"], "name": skill["name"], "score": score, "checks": checks}


@app.get("/api/skills/evaluate")
async def evaluate_skills():
    results = [_skill_evaluation(_read_skill(skill_id)) for skill_id in DEFAULT_SKILLS]
    return {
        "status": "success",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_score": round(sum(item["score"] for item in results) / max(len(results), 1), 1),
        "results": results,
    }


@app.get("/api/skill-evaluation")
async def skill_evaluation_alias():
    return await evaluate_skills()


@app.put("/api/skills/{skill_id}")
async def update_skill(skill_id: str, request: Request):
    skill = _read_skill(skill_id)
    if not skill:
        return _json_error("Skill 不存在。", 404)
    payload = await request.json()
    language = _normalize_language(payload.get("language"))
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        return _json_error("Skill 提示词不能为空。")
    if language == "en":
        skill["name_en"] = str(payload.get("name_en") or payload.get("name") or skill.get("name_en") or skill["name"]).strip()
        skill["description_en"] = str(payload.get("description_en") or payload.get("description") or skill.get("description_en") or skill["description"]).strip()
    else:
        skill["name"] = str(payload.get("name") or skill["name"]).strip()
        skill["description"] = str(payload.get("description") or skill["description"]).strip()
    skill["version"] = str(payload.get("version") or skill.get("version") or "1.0.0").strip()
    skill["architecture_version"] = str(payload.get("architecture_version") or skill.get("architecture_version") or "v2").strip()
    if language != "en":
        for key, fallback in {
            "io_contract": skill.get("io_contract", {}),
            "retrieval_policy": skill.get("retrieval_policy", {}),
            "toolchain": skill.get("toolchain", []),
            "validators": skill.get("validators", []),
            "reflection_loop": skill.get("reflection_loop", []),
            "benchmark_cases": skill.get("benchmark_cases", []),
        }.items():
            value = payload.get(key, fallback)
            skill[key] = value if isinstance(value, (dict, list)) else fallback
    skill["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _skill_meta_path(skill_id).write_text(
        json.dumps(
            {
                k: skill[k]
                for k in [
                    "id",
                    "name",
                    "name_en",
                    "agent",
                    "description",
                    "description_en",
                    "version",
                    "architecture_version",
                    "io_contract",
                    "retrieval_policy",
                    "toolchain",
                    "validators",
                    "reflection_loop",
                    "benchmark_cases",
                    "updated_at",
                ]
                if k in skill
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _skill_prompt_path_for_language(skill_id, language).write_text(prompt, encoding="utf-8")
    return {"status": "success", "skill": _read_skill(skill_id, language)}


@app.post("/api/skills/{skill_id}/assets/upload")
async def upload_skill_asset(skill_id: str, file: UploadFile = File(...)):
    if not _read_skill(skill_id):
        return _json_error("Skill 不存在。", 404)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".pdf"}:
        return _json_error("仅支持上传 TXT 摘要语料或 PDF 规范全文。")
    raw = await file.read()
    asset_dir = SKILL_ASSETS_DIR / skill_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    incoming_hash = hashlib.sha256(raw).hexdigest()
    for existing in asset_dir.iterdir():
        if existing.is_file() and hashlib.sha256(existing.read_bytes()).hexdigest() == incoming_hash:
            evidence = _upsert_asset_records(skill_id, existing)
            return {
                "status": "success",
                "duplicate": True,
                "asset": {"name": existing.name, "size": existing.stat().st_size},
                "processing": _asset_processing_summary(existing),
                "evidence_counts": evidence.get("counts", {}),
            }
    safe_name = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", Path(file.filename or f"asset{suffix}").name)
    out_path = _available_asset_path(asset_dir, safe_name, incoming_hash)
    out_path.write_bytes(raw)
    evidence = _upsert_asset_records(skill_id, out_path)
    job = _enqueue_document_job(out_path) if suffix == ".pdf" and not _asset_processing_summary(out_path).get("processing_complete") else None
    return {
        "status": "success",
        "asset": {"name": out_path.name, "size": out_path.stat().st_size},
        "duplicate": False,
        "processing": _asset_processing_summary(out_path),
        "job": job,
        "evidence_counts": evidence.get("counts", {}),
    }


@app.post("/api/skills/{skill_id}/assets/bulk-upload")
async def upload_skill_assets_bulk(skill_id: str, files: list[UploadFile] = File(...)):
    if not _read_skill(skill_id):
        return _json_error("Skill 不存在。", 404)
    asset_dir = SKILL_ASSETS_DIR / skill_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    imported = []
    skipped = []
    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix != ".pdf":
            skipped.append(file.filename or "")
            continue
        raw = await file.read()
        incoming_hash = hashlib.sha256(raw).hexdigest()
        duplicate = next(
            (
                existing for existing in asset_dir.iterdir()
                if existing.is_file() and hashlib.sha256(existing.read_bytes()).hexdigest() == incoming_hash
            ),
            None,
        )
        if duplicate:
            skipped.append(file.filename or "")
            continue
        relative_name = str(file.filename or f"asset-{len(imported) + 1}.pdf").replace("\\", "/")
        safe_name = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff/]+", "_", relative_name).strip("/")
        flattened_name = safe_name.replace("/", "__")
        out_path = _available_asset_path(asset_dir, flattened_name, incoming_hash)
        out_path.write_bytes(raw)
        processing = _asset_processing_summary(out_path)
        job = _enqueue_document_job(out_path) if not processing.get("processing_complete") else None
        imported.append({
            "name": out_path.name,
            "size": out_path.stat().st_size,
            "source_path": relative_name,
            "processing": processing,
            "job": job,
        })
    evidence = _load_evidence_index()
    for item in imported:
        evidence = _upsert_asset_records(skill_id, asset_dir / item["name"])
    return {
        "status": "success",
        "imported_count": len(imported),
        "skipped_count": len(skipped),
        "assets": imported,
        "skipped": skipped,
        "evidence_counts": evidence.get("counts", {}),
    }


@app.get("/api/skills/{skill_id}/assets/{asset_name}/preview")
async def preview_skill_asset(skill_id: str, asset_name: str, offset: int = 0, limit: int = 4000):
    if not _read_skill(skill_id):
        return _json_error("Skill 不存在。", 404)
    path = SKILL_ASSETS_DIR / skill_id / Path(asset_name).name
    if not path.exists():
        return _json_error("资产不存在。", 404)
    processing = dict(_asset_processing_summary(path))
    if processing.get("literature_records"):
        processing["literature_records"] = processing["literature_records"][:20]
    return {
        "status": "success",
        "asset": {"name": path.name, "size": path.stat().st_size},
        "processing": processing,
        "window": _asset_text_window(path, offset, limit),
    }


@app.get("/api/skills/{skill_id}/assets/{asset_name}/literature-records")
async def list_literature_records(skill_id: str, asset_name: str, page: int = 1, page_size: int = 10, q: str = ""):
    if not _read_skill(skill_id):
        return _json_error("Skill 不存在。", 404)
    path = SKILL_ASSETS_DIR / skill_id / Path(asset_name).name
    if not path.exists():
        return _json_error("资产不存在。", 404)
    records = _asset_processing_summary(path).get("literature_records", [])
    query = str(q or "").strip().lower()
    if query:
        records = [
            item for item in records
            if query in " ".join([
                item.get("title", ""),
                " ".join(item.get("authors", [])),
                item.get("journal", ""),
                item.get("doi", ""),
                item.get("year", ""),
                item.get("abstract", ""),
            ]).lower()
        ]
    page_size = max(5, min(int(page_size or 10), 50))
    total = len(records)
    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(int(page or 1), total_pages))
    start = (page - 1) * page_size
    return {
        "status": "success",
        "records": records[start:start + page_size],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "query": query,
    }


@app.get("/api/skills/{skill_id}/assets/{asset_name}/file")
async def open_skill_asset(skill_id: str, asset_name: str):
    if not _read_skill(skill_id):
        return _json_error("Skill 不存在。", 404)
    path = SKILL_ASSETS_DIR / skill_id / Path(asset_name).name
    if not path.exists():
        return _json_error("资产不存在。", 404)
    media_type = "application/pdf" if path.suffix.lower() == ".pdf" else "text/plain"
    return FileResponse(path, media_type=media_type)


@app.delete("/api/skills/{skill_id}/assets/{asset_name}")
async def delete_skill_asset(skill_id: str, asset_name: str):
    if not _read_skill(skill_id):
        return _json_error("Skill 不存在。", 404)
    path = SKILL_ASSETS_DIR / skill_id / Path(asset_name).name
    if not path.exists():
        return _json_error("资产不存在。", 404)
    path.unlink()
    evidence = _remove_asset_records(path)
    return {"status": "success", "deleted": path.name, "evidence_counts": evidence.get("counts", {})}


@app.delete("/api/skills/{skill_id}/assets")
async def delete_all_skill_assets(skill_id: str):
    if skill_id not in DEFAULT_SKILLS:
        return _json_error("Skill 不存在。", 404)
    asset_dir = SKILL_ASSETS_DIR / skill_id
    if not asset_dir.exists():
        return {"status": "success", "deleted_count": 0, "evidence_counts": _load_evidence_index().get("counts", {})}
    removed = []
    for path in list(asset_dir.iterdir()):
        if not path.is_file():
            continue
        removed.append(path)
        path.unlink(missing_ok=True)
    index = _load_evidence_index()
    removed_paths = {str(path.resolve().relative_to(ROOT)) for path in removed}
    index["records"] = [item for item in index.get("records", []) if item.get("path") not in removed_paths]
    index["updated_at"] = datetime.now().isoformat(timespec="seconds")
    index["counts"] = {
        "literature": sum(1 for item in index["records"] if item["source_type"] == "literature"),
        "standard": sum(1 for item in index["records"] if item["source_type"] == "standard"),
        "patent": sum(1 for item in index["records"] if item["source_type"] == "patent"),
        "kg": sum(1 for item in index["records"] if item["source_type"] == "kg"),
        "auto_kg": sum(1 for item in index["records"] if item["source_type"] == "auto_kg"),
    }
    EVIDENCE_INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    candidates = _load_kg_candidates()
    candidates["candidates"] = [item for item in candidates.get("candidates", []) if item.get("path") not in removed_paths]
    _save_kg_candidates(candidates)
    index = _refresh_candidate_kg_index()
    return {"status": "success", "deleted_count": len(removed), "evidence_counts": index.get("counts", {})}


@app.post("/api/skills/{skill_id}/assets/{asset_name}/reprocess")
async def reprocess_skill_asset(skill_id: str, asset_name: str):
    if not _read_skill(skill_id):
        return _json_error("Skill 不存在。", 404)
    path = SKILL_ASSETS_DIR / skill_id / Path(asset_name).name
    if not path.exists():
        return _json_error("资产不存在。", 404)
    return {"status": "success", "job": _enqueue_document_job(path)}


@app.get("/api/document-jobs")
async def list_document_jobs():
    return {"status": "success", "jobs": sorted(DOCUMENT_JOBS.values(), key=lambda item: item["created_at"], reverse=True)}


@app.get("/api/evidence/status")
async def evidence_status():
    index = _load_evidence_index()
    return {
        "status": "success",
        "updated_at": index.get("updated_at"),
        "counts": index.get("counts", {}),
        "total_records": len(index.get("records", [])),
    }


@app.post("/api/evidence/rebuild")
async def evidence_rebuild():
    index = _rebuild_evidence_index()
    return {
        "status": "success",
        "updated_at": index.get("updated_at"),
        "counts": index.get("counts", {}),
        "total_records": len(index.get("records", [])),
    }


@app.get("/api/standards/status")
async def standards_status():
    rows = _load_standard_constraint_rows()
    summary = _standard_constraint_summary()
    return {
        "status": "success",
        "source_dir": str(STANDARD_SOURCE_DIR),
        "summary": summary,
        "constraint_count": len(rows),
        "optimizable_count": sum(1 for row in rows if row.get("compile_status") == "optimizable"),
        "has_outputs": STANDARD_CONSTRAINT_JSONL.exists(),
        "outputs": {
            "jsonl": str(STANDARD_CONSTRAINT_JSONL),
            "csv": str(STANDARD_CONSTRAINT_CSV),
            "summary": str(STANDARD_CONSTRAINT_SUMMARY),
            "four_target_csv": str(FOUR_TARGET_STANDARD_CSV),
            "four_target_summary": str(FOUR_TARGET_STANDARD_SUMMARY),
            "constraint_agent": str(CONSTRAINT_AGENT_PAYLOAD),
        },
        "constraint_agent": _constraint_agent_evidence_payload(),
    }


@app.post("/api/standards/extract")
async def standards_extract(request: Request):
    body = await request.json()
    try:
        summary = await asyncio.to_thread(
            _run_standard_constraint_extraction,
            body.get("source_dir") or str(STANDARD_SOURCE_DIR),
            body.get("mode") or "precision",
            bool(body.get("all_pdfs", False)),
            _safe_int(body.get("limit"), 0),
        )
    except Exception as exc:
        return _json_error(f"规范约束抽取失败：{exc}", 500)
    return {"status": "success", "summary": summary, "evidence_counts": _load_evidence_index().get("counts", {})}


@app.get("/api/standards/constraints")
async def standards_constraints(limit: int = 80, domain: str = "", metric: str = "", compile_status: str = ""):
    rows = _load_standard_constraint_rows()
    if domain:
        rows = [row for row in rows if _normalize_domain_name(row.get("domain")) == _normalize_domain_name(domain)]
    if metric:
        rows = [row for row in rows if row.get("metric") == metric]
    if compile_status:
        rows = [row for row in rows if row.get("compile_status") == compile_status]
    rows.sort(key=lambda row: (
        0 if row.get("compile_status") == "optimizable" else 1,
        str(row.get("domain") or ""),
        str(row.get("source_file") or ""),
        int(row.get("page") or 0),
    ))
    limit = max(1, min(int(limit or 80), 500))
    return {"status": "success", "rows": rows[:limit], "total": len(rows), "summary": _standard_constraint_summary()}


@app.get("/api/standards/visualization")
async def standards_visualization():
    rows = _load_standard_constraint_rows()
    return {"status": "success", "visualization": _standard_constraint_visualization_payload(rows)}


@app.get("/api/standards/constraint-agent")
async def standards_constraint_agent():
    return {"status": "success", "constraint_agent": _constraint_agent_evidence_payload()}


@app.get("/api/standards/export/{kind}")
async def standards_export(kind: str):
    mapping = {
        "csv": STANDARD_CONSTRAINT_CSV,
        "jsonl": STANDARD_CONSTRAINT_JSONL,
        "summary": STANDARD_CONSTRAINT_SUMMARY,
        "four-target-csv": FOUR_TARGET_STANDARD_CSV,
        "four-target-summary": FOUR_TARGET_STANDARD_SUMMARY,
        "constraint-agent": CONSTRAINT_AGENT_PAYLOAD,
    }
    path = mapping.get(kind)
    if not path or not path.exists():
        return _json_error("导出文件不存在，请先运行规范抽取。", 404)
    return FileResponse(str(path), filename=path.name)


@app.get("/api/abstract-corpus/status")
async def abstract_corpus_status():
    return {"status": "success", **_abstract_corpus_status()}


@app.get("/api/abstract-corpus/embedding-config")
async def abstract_corpus_embedding_config():
    return {"status": "success", "config": _load_embedding_config(redact=False)}


@app.put("/api/abstract-corpus/embedding-config")
async def abstract_corpus_update_embedding_config(request: Request):
    body = await request.json()
    config = _save_embedding_config(body)
    return {"status": "success", "config": config}


@app.post("/api/abstract-corpus/embedding-test")
async def abstract_corpus_embedding_test(request: Request):
    body = await request.json()
    config = _load_embedding_config()
    if body:
        config.update({key: value for key, value in body.items() if key in config})
        config["batch_size"] = max(1, min(_safe_int(config.get("batch_size"), 32), 512))
        config["timeout_seconds"] = max(10, min(_safe_int(config.get("timeout_seconds"), 120), 900))
    try:
        result = await asyncio.to_thread(_test_embedding_config, config)
    except Exception as exc:
        return _json_error(f"Embedding 连接测试失败：{exc}", 500)
    return {"status": "success", "result": result}


@app.post("/api/abstract-corpus/upload")
async def upload_abstract_corpus(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".tsv", ".csv"}:
        return _json_error("仅支持 WOS TXT、TSV 或 CSV 摘要语料。")
    raw = await file.read()
    if not raw:
        return _json_error("上传文件为空。")
    ABSTRACT_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    ABSTRACT_CORPUS_RAW_PATH.write_bytes(raw)
    text = raw.decode("utf-8-sig", errors="ignore")
    docs = _parse_abstract_corpus(text, file.filename or "")
    docs = [doc for doc in docs if doc.get("title") or doc.get("abstract")]
    if not docs:
        return _json_error("没有从文件中解析到摘要记录。请确认包含 WOS plain text 或 TI/AB 表头。")
    meta = _write_abstract_docs(docs, file.filename or "", len(raw))
    return {"status": "success", "meta": meta, "sample": docs[:3]}


@app.post("/api/abstract-corpus/calibrate")
async def abstract_corpus_calibrate(request: Request):
    body = await request.json()
    top_k = max(3, min(_safe_int(body.get("top_k"), 8), 24))
    max_docs = max(0, _safe_int(body.get("max_docs"), 0))
    try:
        payload = await asyncio.to_thread(_calibrate_kg_with_abstracts, top_k, max_docs)
    except Exception as exc:
        return _json_error(f"证据校准失败：{exc}", 500)
    return {"status": "success", "summary": payload.get("summary", {}), "settings": payload.get("settings", {})}


@app.get("/api/abstract-corpus/calibration")
async def abstract_corpus_calibration(limit: int = 30, sort: str = "confidence"):
    if not ABSTRACT_CALIBRATION_PATH.exists():
        return _json_error("尚未运行证据校准。", 404)
    payload = json.loads(ABSTRACT_CALIBRATION_PATH.read_text(encoding="utf-8"))
    rows = payload.get("calibrations", [])
    if sort == "low_evidence":
        rows = sorted(rows, key=lambda item: (item.get("evidence_count", 0), -item.get("confidence_score", 0)))
    elif sort == "recent":
        rows = sorted(rows, key=lambda item: item.get("recent_evidence_count", 0), reverse=True)
    else:
        rows = sorted(rows, key=lambda item: item.get("confidence_score", 0), reverse=True)
    limit = max(5, min(int(limit or 30), 200))
    return {
        "status": "success",
        "updated_at": payload.get("updated_at"),
        "summary": payload.get("summary", {}),
        "settings": payload.get("settings", {}),
        "calibrations": rows[:limit],
        "total": len(payload.get("calibrations", [])),
    }


@app.post("/api/abstract-corpus/landscape")
async def abstract_corpus_landscape():
    try:
        payload = await asyncio.to_thread(_build_evidence_landscape)
    except Exception as exc:
        return _json_error(f"证据地形图计算失败：{exc}", 500)
    return {
        "status": "success",
        "updated_at": payload.get("updated_at"),
        "summary": payload.get("summary", {}),
        "top_gaps": payload.get("pairs", [])[:20],
        "agent_priors": payload.get("agent_priors", [])[:20],
    }


@app.get("/api/abstract-corpus/landscape")
async def abstract_corpus_landscape_get():
    if not ABSTRACT_LANDSCAPE_PATH.exists():
        return _json_error("尚未计算 evidence landscape。", 404)
    payload = json.loads(ABSTRACT_LANDSCAPE_PATH.read_text(encoding="utf-8"))
    return {
        "status": "success",
        "updated_at": payload.get("updated_at"),
        "summary": payload.get("summary", {}),
        "top_gaps": payload.get("pairs", [])[:50],
        "agent_priors": payload.get("agent_priors", [])[:50],
        "figures": payload.get("figures", []),
    }


@app.post("/api/abstract-corpus/shrinkage-evidence")
async def abstract_corpus_shrinkage_evidence():
    try:
        payload = await asyncio.to_thread(_build_shrinkage_evidence_summary)
    except Exception as exc:
        return _json_error(f"低收缩摘要证据分析失败：{exc}", 500)
    return {
        "status": "success",
        "updated_at": payload.get("updated_at"),
        "summary": payload.get("summary", {}),
        "materials": payload.get("materials", [])[:30],
        "exports": {
            "json": "evidence/abstract-corpus/shrinkage_evidence_summary.json",
            "csv": "evidence/abstract-corpus/shrinkage_material_evidence.csv",
        },
    }


@app.get("/api/abstract-corpus/shrinkage-evidence")
async def abstract_corpus_shrinkage_evidence_get():
    payload = _load_shrinkage_evidence_summary(build_if_missing=False)
    if not payload:
        return _json_error("尚未计算低收缩摘要证据分析。", 404)
    return {
        "status": "success",
        "updated_at": payload.get("updated_at"),
        "summary": payload.get("summary", {}),
        "materials": payload.get("materials", [])[:50],
        "examples": payload.get("examples", [])[:30],
        "exports": {
            "json": "evidence/abstract-corpus/shrinkage_evidence_summary.json",
            "csv": "evidence/abstract-corpus/shrinkage_material_evidence.csv",
        },
    }


@app.post("/api/abstract-corpus/figures")
async def abstract_corpus_figures():
    try:
        payload = await asyncio.to_thread(_generate_landscape_figures)
    except Exception as exc:
        return _json_error(f"图件生成失败：{exc}", 500)
    return {"status": "success", **payload}


@app.get("/api/abstract-corpus/figures")
async def abstract_corpus_figures_get():
    if not ABSTRACT_LANDSCAPE_PATH.exists():
        return _json_error("尚未生成图件。", 404)
    payload = json.loads(ABSTRACT_LANDSCAPE_PATH.read_text(encoding="utf-8"))
    return {"status": "success", "figures": payload.get("figures", []), "summary": payload.get("summary", {})}


@app.get("/api/abstract-corpus/figures/{filename}")
async def abstract_corpus_figure_file(filename: str):
    safe = Path(filename).name
    path = ABSTRACT_FIGURE_DIR / safe
    if not path.exists():
        return _json_error("图件不存在。", 404)
    media = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".pdf": "application/pdf",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media, filename=path.name)


@app.get("/api/abstract-corpus/lora")
async def abstract_corpus_lora_status():
    return {"status": "success", **_lora_status()}


@app.put("/api/abstract-corpus/lora/config")
async def abstract_corpus_lora_config(request: Request):
    body = await request.json()
    config = _save_lora_config(body)
    if LORA_DATASET_PATH.exists() or (LORA_DIR / "train_qwen3_lora.py").exists():
        _write_lora_training_script(base_model=config.get("base_model"), output_dir=config.get("output_dir"), config=config)
    return {"status": "success", "config": config, **_lora_status()}


@app.post("/api/abstract-corpus/lora/dataset")
async def abstract_corpus_lora_dataset(request: Request):
    body = await request.json()
    max_examples = max(300, min(_safe_int(body.get("max_examples"), 5000), 20000))
    base_model = str(body.get("base_model") or "Qwen/Qwen3-4B-Instruct-2507").strip() or "Qwen/Qwen3-4B-Instruct-2507"
    embedding_model = str(body.get("embedding_model") or "text-embedding-qwen3-embedding-0.6b").strip() or "text-embedding-qwen3-embedding-0.6b"
    output_dir = str(body.get("output_dir") or LORA_DEFAULT_OUTPUT_DIR).strip() or str(LORA_DEFAULT_OUTPUT_DIR)
    try:
        manifest = await asyncio.to_thread(_generate_lora_datasets, max_examples, base_model, embedding_model, output_dir)
    except Exception as exc:
        return _json_error(f"LoRA 数据集生成失败：{exc}", 500)
    return {"status": "success", "manifest": manifest, **_lora_status()}


@app.post("/api/abstract-corpus/lora/figures")
async def abstract_corpus_lora_figures():
    try:
        payload = await asyncio.to_thread(_generate_lora_figures)
    except Exception as exc:
        return _json_error(f"LoRA 图件生成失败：{exc}", 500)
    return {"status": "success", **payload}


@app.get("/api/abstract-corpus/lora/figures")
async def abstract_corpus_lora_figures_get():
    return {"status": "success", "figures": _list_lora_figures(), **_lora_status()}


@app.get("/api/abstract-corpus/lora/figures/{filename}")
async def abstract_corpus_lora_figure_file(filename: str):
    safe = Path(filename).name
    path = LORA_FIGURE_DIR / safe
    if not path.exists():
        return _json_error("LoRA 图件不存在。", 404)
    media = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".pdf": "application/pdf",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media, filename=path.name)


@app.get("/api/abstract-corpus/lora/export/{artifact}")
async def abstract_corpus_lora_export(artifact: str):
    mapping = {
        "train": LORA_DATASET_PATH,
        "validation": LORA_VALIDATION_PATH,
        "manifest": LORA_MANIFEST_PATH,
        "script": LORA_DIR / "train_qwen3_lora.py",
    }
    path = mapping.get(artifact)
    if not path or not path.exists():
        return _json_error("LoRA 文件不存在。", 404)
    media = "text/x-python" if path.suffix == ".py" else "application/jsonl" if path.suffix == ".jsonl" else "application/json"
    return FileResponse(path, media_type=media, filename=path.name)


@app.post("/api/abstract-corpus/discover-candidates")
async def abstract_corpus_discover_candidates(request: Request):
    body = await request.json()
    limit = max(20, min(_safe_int(body.get("limit"), 300), 1200))
    min_confidence = max(0.3, min(_safe_float(body.get("min_confidence"), 0.52), 0.95))
    max_docs = max(0, _safe_int(body.get("max_docs"), 0))
    try:
        payload = await asyncio.to_thread(_discover_abstract_candidates, limit, min_confidence, max_docs)
    except Exception as exc:
        return _json_error(f"候选关系发现失败：{exc}", 500)
    return {
        "status": "success",
        "updated_at": payload.get("updated_at"),
        "settings": payload.get("settings", {}),
        "count": len(payload.get("candidates", [])),
        "candidates": payload.get("candidates", [])[:50],
    }


@app.get("/api/abstract-corpus/candidates")
async def abstract_corpus_candidates(limit: int = 80):
    if not ABSTRACT_CANDIDATES_PATH.exists():
        return _json_error("尚未生成摘要候选 KG。", 404)
    payload = json.loads(ABSTRACT_CANDIDATES_PATH.read_text(encoding="utf-8"))
    limit = max(5, min(int(limit or 80), 500))
    return {
        "status": "success",
        "updated_at": payload.get("updated_at"),
        "settings": payload.get("settings", {}),
        "count": len(payload.get("candidates", [])),
        "candidates": payload.get("candidates", [])[:limit],
    }


@app.post("/api/abstract-corpus/merge-candidates")
async def abstract_corpus_merge_candidates():
    try:
        result = _merge_abstract_candidates_into_review_pool()
    except Exception as exc:
        return _json_error(str(exc), 400)
    return {"status": "success", **result}


@app.post("/api/abstract-corpus/run")
async def abstract_corpus_run(request: Request):
    body = await request.json()
    top_k = max(3, min(_safe_int(body.get("top_k"), 8), 24))
    max_docs = max(0, _safe_int(body.get("max_docs"), 0))
    try:
        calibration = await asyncio.to_thread(_calibrate_kg_with_abstracts, top_k, max_docs)
    except Exception as exc:
        return _json_error(f"摘要证据校准失败：{exc}", 500)
    return {
        "status": "success",
        "calibration_summary": calibration.get("summary", {}),
        "message": "已完成 Gold KG 的摘要证据校准；未生成候选 KG。",
    }


@app.get("/api/abstract-corpus/export/{artifact}")
async def abstract_corpus_export(artifact: str):
    mapping = {
        "docs": ABSTRACT_CORPUS_DOCS_PATH,
        "calibration": ABSTRACT_CALIBRATION_PATH,
        "candidates": ABSTRACT_CANDIDATES_PATH,
        "meta": ABSTRACT_CORPUS_META_PATH,
        "landscape": ABSTRACT_LANDSCAPE_PATH,
        "pairs": ABSTRACT_CORPUS_DIR / "material_performance_landscape.csv",
        "agent_priors": ABSTRACT_CORPUS_DIR / "agent_priors.csv",
        "kg_table": ABSTRACT_CORPUS_DIR / "kg_calibration_table.csv",
    }
    path = mapping.get(artifact)
    if not path or not path.exists():
        return _json_error("导出文件不存在。", 404)
    media_type = "application/jsonl" if path.suffix == ".jsonl" else "application/json"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/evidence/{evidence_id}")
async def evidence_detail(evidence_id: str):
    def enrich_record(record):
        record = dict(record)
        metadata = record.get("metadata", {}) or {}
        if record.get("path") and record.get("skill_id"):
            filename = Path(record["path"]).name
            record["file_url"] = f"/api/skills/{record['skill_id']}/assets/{urllib.parse.quote(filename)}/file"
        doi = metadata.get("doi") or record.get("doi")
        if doi:
            record["doi_url"] = f"https://doi.org/{doi}"
        return record

    index = _load_evidence_index()
    for record in index.get("records", []):
        if record.get("id") == evidence_id:
            return {"status": "success", "record": enrich_record(record)}
    for item in _load_kg_candidates().get("candidates", []):
        if item.get("id") == evidence_id:
            return {"status": "success", "record": enrich_record({**item, "source_type": "auto_kg_candidate", "title": f"{item.get('material')} -> {item.get('performance')}", "text": item.get("evidence", "")})}
    return _json_error("证据不存在。", 404)


@app.get("/api/kg-candidates")
async def list_kg_candidates(status: str = ""):
    payload = _load_kg_candidates()
    candidates = payload.get("candidates", [])
    if status:
        candidates = [item for item in candidates if item.get("status") == status]
    return {"status": "success", "updated_at": payload.get("updated_at"), "candidates": candidates}


@app.patch("/api/kg-candidates/{candidate_id}")
async def update_kg_candidate(candidate_id: str, request: Request):
    body = await request.json()
    status = str(body.get("status") or "").strip()
    if status not in {"pending", "approved", "rejected"}:
        return _json_error("status 只能为 pending、approved 或 rejected。")
    payload = _load_kg_candidates()
    found = False
    for item in payload.get("candidates", []):
        if item.get("id") == candidate_id:
            item["status"] = status
            item["review_note"] = str(body.get("review_note") or "")
            item["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
            found = True
            break
    if not found:
        return _json_error("候选三元组不存在。", 404)
    _save_kg_candidates(payload)
    index = _refresh_candidate_kg_index()
    return {"status": "success", "candidate_id": candidate_id, "candidate_status": status, "evidence_counts": index.get("counts", {})}


@app.post("/api/model/upload")
async def upload_model(file: UploadFile = File(...)):
    if IMPORT_ERROR:
        return _json_error(f"无法加载 testML 优化内核：{IMPORT_ERROR}", 500)
    raw = await file.read()
    try:
        package = model_io.load_model_package_from_bytes(raw)
    except Exception as exc:
        return _json_error(f"模型包读取失败：{exc}")
    model_package = package.get("main_model_package") if package.get("package_type") == "natureml_multiobjective_optimization" else package
    feature_columns = (model_package or {}).get("feature_columns") or (model_package or {}).get("selected_features") or []
    STATE["package"] = package
    STATE["package_name"] = file.filename
    STATE["feature_map"] = _make_feature_map(feature_columns)
    return {
        "status": "success",
        "package_type": package.get("package_type"),
        "model_name": (model_package or {}).get("model_name") or package.get("name"),
        "feature_columns": feature_columns,
        "variables": list(STATE["feature_map"].values()),
    }


@app.post("/api/agent/design")
async def design_agent(request: Request):
    if IMPORT_ERROR:
        return _json_error(f"无法加载 testML 优化内核：{IMPORT_ERROR}", 500)
    payload = await request.json()
    language = _normalize_language(payload.get("language"))
    prompt = payload.get("prompt", "")
    package = STATE.get("package")
    if not package:
        return _json_error("请先上传从 testML 导出的 .naturemlmodel 或 .naturemlopt 文件。")

    requirements = _validated_requirement_spec(prompt, payload.get("target_mpa"))
    target_mpa = float(requirements["target_mpa"])
    package_type = package.get("package_type")
    feature_map = STATE.get("feature_map") or {}

    if package_type == "natureml_multiobjective_optimization" and not package.get("main_model_package"):
        best = package.get("best_solution")
        imported_setup = _build_optimization_setup(requirements, feature_map, payload)
        imported_theory = _traditional_theory_checks(best, feature_map, imported_setup)
        if best and not _passes_design_priors(best, feature_map, imported_setup):
            wb = _mix_ratios(best, feature_map).get("water_binder_ratio")
            window = imported_setup.get("traditional_theory", {}).get("water_binder_window", {})
            return _json_error(
                f"当前上传的是仅含历史结果的优化包，不能按本次目标重新优化；该导入解 w/b={wb}，未通过当前鲍罗米先验区间 {window.get('min')}~{window.get('max')}。请上传包含 main_model_package 的模型/优化包后重新设计。"
            )
        role_evidence = _role_evidence_bundle(requirements)
        kg_payload = _kg_enhancement(requirements, feature_map, best, role_evidence["mechanism_retrieval"])
        kg_payload["role_evidence"] = role_evidence
        kpis = _solution_kpis(best, imported_setup, feature_map)
        optimization_summary = {
            "target_mpa": target_mpa,
            "age_days": requirements.get("age_days"),
            "best_algorithm": package.get("best_algorithm"),
            "best_solution": best,
            "solution_table": _summarize_solution(best, feature_map),
            "ratios": _mix_ratios(best, feature_map),
            "candidate_schemes": [],
            "traditional_theory": imported_theory,
            "kpis": kpis,
            "metrics": package.get("metrics", {}),
            "history": package.get("history", []),
            "comparison_metrics": package.get("comparison_metrics", []),
            "note": "当前上传的是优化结果包，已读取其中首选解；未重新运行优化。",
        }
        _attach_design_audits(requirements, optimization_summary, kg_payload, imported_setup, feature_map)
        agents = [
            {"id": "requirement", "name": "Requirement Parsing Agent" if language == "en" else "需求解析智能体", "status": "completed", "summary": (f"Identified target strength {target_mpa:g} MPa and curing age {requirements['age_days']:g} d." if language == "en" else f"识别目标强度 {target_mpa:g} MPa；龄期 {requirements['age_days']:g} d"), "details": {**requirements, "evidence": role_evidence["requirement_analysis"], "citations": role_evidence["requirement_analysis"].get("citations", [])}, "trace": _requirement_trace(requirements), "steps": _requirement_steps(requirements)},
            {"id": "optimizer", "name": "Optimization Modeling Agent" if language == "en" else "优化建模智能体", "status": "completed", "summary": ("Imported an existing optimization-result package without rerunning the search." if language == "en" else "读取已有优化结果包，未重新运行优化。"), "details": {**optimization_summary, "evidence": role_evidence["optimization_modeling"], "citations": role_evidence["optimization_modeling"].get("citations", [])}, "trace": {"kpis": kpis, "note": "imported_optimization"}, "steps": [{"title": "Read imported optimization package" if language == "en" else "读取导入优化包", "detail": "Existing recommended solution was loaded without rerunning the search." if language == "en" else "已读取既有推荐解，未重新执行搜索", "artifact": "imported_package"}]},
            {"id": "kg", "name": "Mechanism Retrieval Agent" if language == "en" else "机理检索智能体", "status": "completed", "summary": (f"Retrieved {len(kg_payload['hits'])} material-mechanism groups and generated {len(kg_payload['recommendations'])} recommendations." if language == "en" else f"命中 {len(kg_payload['hits'])} 类材料机理，生成 {len(kg_payload['recommendations'])} 条建议。"), "details": {**kg_payload, "evidence": role_evidence["mechanism_retrieval"], "citations": role_evidence["mechanism_retrieval"].get("citations", [])}, "trace": _kg_trace(kg_payload), "steps": _kg_steps(requirements, kg_payload)},
        ]
        dialogues, lm_ok = _run_agent_dialogues(prompt, requirements, optimization_summary, kg_payload, agents, language)
        for agent in agents:
            agent["dialogue"] = dialogues.get(agent["id"])
        return {
            "status": "success",
            "data": {
                "answer": dialogues.get("final"),
                "lm_ok": lm_ok,
                "agents": agents,
                "optimization": optimization_summary,
                "kg_hits": kg_payload["hits"],
                "recommendations": kg_payload["recommendations"],
            "visualizations": {"optimization_plots": []},
            "handoffs": _handoff_payload(requirements, imported_setup, optimization_summary, kg_payload),
            },
        }

    model_package = _model_package()
    model = (model_package or {}).get("model")
    if model is None:
        return _json_error("模型包中没有可调用的 model。请从 testML 导出 .naturemlmodel，或导出包含 main_model_package 的优化包。")

    if not feature_map:
        feature_columns = model_package.get("feature_columns") or model_package.get("selected_features") or []
        feature_map = _make_feature_map(feature_columns)
        STATE["feature_map"] = feature_map
    setup = _build_optimization_setup(requirements, feature_map, payload)
    setup = _apply_optimizer_proposal(setup, _optimizer_llm_proposal(requirements, setup.get("flowability_alias")))
    bounds = setup["bounds"]
    original_columns = [feature_map[item["name"]]["original"] for item in bounds]
    objectives = setup["objectives"]
    constraints = setup["constraints"]
    prediction_fn = _make_prediction_fn(model_package, feature_map, original_columns)

    try:
        result = ml_optimization.run_multiobjective_optimization(
            bounds,
            objectives,
            constraints,
            setup["algorithm"],
            prediction_fn,
            pop_size=setup["pop_size"],
            generations=setup["generations"],
            random_state=42,
            fmt="png",
            dpi=160,
            compare_algorithms=setup["compare_algorithms"],
        )
    except Exception as exc:
        return _json_error(f"多目标优化运行失败：{exc}")

    result = _retain_prior_feasible_solutions(result, feature_map, setup)
    if not result.get("best_solution"):
        wb_window = setup.get("traditional_theory", {}).get("water_binder_window", {})
        return _json_error(
            f"多目标优化未找到同时满足传统先验与机器学习约束的可行解。当前鲍罗米反推水胶比区间为 {wb_window.get('min')}~{wb_window.get('max')}；请检查原材边界、目标强度或放宽其他冲突约束。"
        )

    best = result.get("best_solution")
    role_evidence = _role_evidence_bundle(requirements)
    kg_payload = _kg_enhancement(requirements, feature_map, best, role_evidence["mechanism_retrieval"])
    kg_payload["role_evidence"] = role_evidence
    kpis = _solution_kpis(best, setup, feature_map)
    plot_previews = _select_plot_previews(result)
    optimization_summary = {
        "target_mpa": target_mpa,
        "age_days": requirements.get("age_days"),
        "model_name": model_package.get("model_name"),
        "best_algorithm": result.get("best_algorithm"),
        "metrics": result.get("metrics"),
        "best_solution": best,
        "solution_table": _summarize_solution(best, feature_map),
        "ratios": _mix_ratios(best, feature_map),
        "candidate_schemes": _representative_mix_schemes(result, feature_map, requirements, setup),
        "traditional_theory": _traditional_theory_checks(best, feature_map, setup),
        "kpis": kpis,
        "top_solutions": result.get("solutions", [])[:20],
        "history": result.get("history", []),
        "comparison_metrics": result.get("comparison_metrics", []),
        "prior_feasible_count": result.get("prior_feasible_count", 0),
        "objectives": objectives,
        "constraints": constraints,
        "constraint_status": setup.get("constraint_status", {}),
        "standard_constraint_visualization": setup.get("standard_constraint_visualization", {}),
        "compiled_normative_constraints": setup.get("compiled_normative_constraints", []),
        "llm_proposal": setup.get("llm_proposal", {}),
    }
    _attach_design_audits(requirements, optimization_summary, kg_payload, setup, feature_map)
    agents = [
        {
            "id": "requirement",
            "name": "Requirement Parsing Agent" if language == "en" else "需求解析智能体",
            "status": "completed",
            "summary": (
                f"Identified target strength {target_mpa:g} MPa, curing age {requirements['age_days']:g} d, and priorities: {', '.join(requirements['priorities'])}."
                if language == "en"
                else f"识别目标强度 {target_mpa:g} MPa；龄期 {requirements['age_days']:g} d；优先级：{', '.join(requirements['priorities'])}"
            ),
            "details": {**requirements, "evidence": role_evidence["requirement_analysis"], "citations": role_evidence["requirement_analysis"].get("citations", [])},
            "trace": _requirement_trace(requirements),
            "steps": _requirement_steps(requirements),
        },
        {
            "id": "optimizer",
            "name": "Optimization Modeling Agent" if language == "en" else "优化建模智能体",
            "status": "completed",
            "summary": (
                f"Generated {len(objectives)} objective functions and {len(constraints)} constraints, then ran {result.get('best_algorithm')}."
                if language == "en"
                else f"生成 {len(objectives)} 个目标函数、{len(constraints)} 个约束，并运行 {result.get('best_algorithm')}。"
            ),
            "details": {
                "algorithm": setup["algorithm"],
                "selected_algorithm": result.get("best_algorithm"),
                "objectives": objectives,
                "constraints": constraints,
                "bounds_count": len(bounds),
                "metrics": result.get("metrics"),
                "kpis": kpis,
                "constraint_status": setup.get("constraint_status", {}),
                "standard_constraint_visualization": setup.get("standard_constraint_visualization", {}),
                "compiled_normative_constraints": setup.get("compiled_normative_constraints", []),
                "traditional_theory": setup.get("traditional_theory", {}),
                "evidence": role_evidence["optimization_modeling"],
                "citations": role_evidence["optimization_modeling"].get("citations", []),
            },
            "trace": _optimization_trace(setup, kpis),
            "steps": _optimization_steps(setup, result, kpis),
        },
        {
            "id": "kg",
            "name": "Mechanism Retrieval Agent" if language == "en" else "机理检索智能体",
            "status": "completed",
            "summary": (
                f"Retrieved {len(kg_payload['hits'])} material-mechanism groups and generated {len(kg_payload['recommendations'])} optimization recommendations."
                if language == "en"
                else f"命中 {len(kg_payload['hits'])} 类材料机理，生成 {len(kg_payload['recommendations'])} 条优化建议。"
            ),
            "details": {**kg_payload, "evidence": role_evidence["mechanism_retrieval"], "citations": role_evidence["mechanism_retrieval"].get("citations", [])},
            "trace": _kg_trace(kg_payload),
            "steps": _kg_steps(requirements, kg_payload),
        },
    ]
    dialogues, lm_ok = _run_agent_dialogues(prompt, requirements, optimization_summary, kg_payload, agents, language)
    for agent in agents:
        agent["dialogue"] = dialogues.get(agent["id"])
    return {
        "status": "success",
        "data": {
            "answer": dialogues.get("final"),
            "lm_ok": lm_ok,
            "agents": agents,
            "optimization": optimization_summary,
            "kg_hits": kg_payload["hits"],
            "recommendations": kg_payload["recommendations"],
            "visualizations": {"optimization_plots": plot_previews},
            "handoffs": _handoff_payload(requirements, setup, optimization_summary, kg_payload),
        },
    }


@app.post("/api/agent/design/stream")
async def design_agent_stream(request: Request):
    payload = await request.json()
    language = _normalize_language(payload.get("language"))

    async def generate():
        if IMPORT_ERROR:
            yield _json_event("error", message=f"无法加载 testML 优化内核：{IMPORT_ERROR}")
            return
        prompt = payload.get("prompt", "")
        package = STATE.get("package")
        if not package:
            yield _json_event("error", message="请先上传从 testML 导出的 .naturemlmodel 文件。")
            return
        model_package = _model_package()
        model = (model_package or {}).get("model")
        if model is None:
            yield _json_event("error", message="流式模式需要包含可调用 model 的模型包。")
            return

        requirements = await asyncio.to_thread(_validated_requirement_spec, prompt, payload.get("target_mpa"))
        yield _json_event("requirement_started", raw_request=prompt)
        yield _json_event("requirement_llm_spec", spec=requirements.get("llm_spec"), validated=requirements)
        for index, segment in enumerate(requirements.get("raw_segments", []), 1):
            yield _json_event("requirement_segment", index=index, segment=segment)
            await asyncio.sleep(0.12)
        for index, item in enumerate(_requirement_trace(requirements), 1):
            yield _json_event("requirement_mapping", index=index, mapping=item)
            await asyncio.sleep(0.1)
        yield _json_event("requirement_completed", requirements=requirements, steps=_requirement_steps(requirements))

        feature_map = STATE.get("feature_map") or {}
        if not feature_map:
            feature_columns = model_package.get("feature_columns") or model_package.get("selected_features") or []
            feature_map = _make_feature_map(feature_columns)
            STATE["feature_map"] = feature_map
        setup = _build_optimization_setup(requirements, feature_map, payload)
        optimizer_proposal = await asyncio.to_thread(_optimizer_llm_proposal, requirements, setup.get("flowability_alias"))
        setup = _apply_optimizer_proposal(setup, optimizer_proposal)
        bounds = setup["bounds"]
        original_columns = [feature_map[item["name"]]["original"] for item in bounds]
        prediction_fn = _make_prediction_fn(model_package, feature_map, original_columns)

        yield _json_event("optimizer_started")
        yield _json_event("optimizer_llm_proposal", proposal=setup.get("llm_proposal", {}))
        yield _json_event("optimizer_bounds", bounds=bounds)
        await asyncio.sleep(0.12)
        for objective in setup["objectives"]:
            yield _json_event("optimizer_objective", objective=objective)
            await asyncio.sleep(0.12)
        for constraint in setup["constraints"]:
            yield _json_event("optimizer_constraint", constraint=constraint)
            await asyncio.sleep(0.12)
        for metric in _eventful_main_algorithm(
            bounds, setup["objectives"], setup["constraints"], setup["algorithm"], prediction_fn, setup["pop_size"], setup["generations"]
        ):
            yield _json_event("optimizer_progress", **metric)
            await asyncio.sleep(0)

        try:
            result = ml_optimization.run_multiobjective_optimization(
                bounds,
                setup["objectives"],
                setup["constraints"],
                setup["algorithm"],
                prediction_fn,
                pop_size=setup["pop_size"],
                generations=setup["generations"],
                random_state=42,
                fmt="png",
                dpi=160,
                compare_algorithms=setup["compare_algorithms"],
            )
        except Exception as exc:
            yield _json_event("error", message=f"多目标优化运行失败：{exc}")
            return

        result = _retain_prior_feasible_solutions(result, feature_map, setup)
        if not result.get("best_solution"):
            wb_window = setup.get("traditional_theory", {}).get("water_binder_window", {})
            yield _json_event("error", message=f"多目标优化未找到满足传统先验的可行解。当前鲍罗米反推水胶比区间为 {wb_window.get('min')}~{wb_window.get('max')}。")
            return

        best = result.get("best_solution")
        kpis = _solution_kpis(best, setup, feature_map)
        yield _json_event("optimizer_completed", kpis=kpis, best_algorithm=result.get("best_algorithm"))
        role_evidence = _role_evidence_bundle(requirements)
        kg_payload = _kg_enhancement(requirements, feature_map, best, role_evidence["mechanism_retrieval"])
        kg_payload["role_evidence"] = role_evidence
        kg_trace = _kg_trace(kg_payload)
        yield _json_event("kg_started", query_terms=kg_payload.get("query_terms", []))
        yield _json_event(
            "evidence_query",
            query_terms=kg_payload.get("evidence", {}).get("query_terms", []),
            counts=kg_payload.get("evidence", {}).get("counts", {}),
        )
        for item in kg_payload.get("evidence", {}).get("candidates", [])[:6]:
            yield _json_event(
                "evidence_candidate",
                evidence={"id": item["id"], "title": item["title"], "source_type": item["source_type"], "score": item["score"]},
            )
            await asyncio.sleep(0)
        yield _json_event(
            "evidence_ranked",
            selected=[
                {"id": item["id"], "title": item["title"], "source_type": item["source_type"], "score": item["score"], "grade": item["grade"]}
                for item in kg_payload.get("evidence", {}).get("selected", [])
            ],
            routes=kg_payload.get("evidence", {}).get("multi_hop_routes", []),
        )
        yield _json_event("kg_graph_loaded", graph=kg_trace)
        for node in kg_trace.get("nodes", []):
            if node.get("active"):
                yield _json_event("kg_node", node=node)
                await asyncio.sleep(0)
        for link in kg_trace.get("links", []):
            if link.get("active"):
                yield _json_event("kg_link", link=link)
                await asyncio.sleep(0)
        yield _json_event("kg_completed", recommendations=kg_payload.get("recommendations", []))

        optimization_summary = {
            "target_mpa": requirements["target_mpa"],
            "age_days": requirements.get("age_days"),
            "model_name": model_package.get("model_name"),
            "best_algorithm": result.get("best_algorithm"),
            "metrics": result.get("metrics"),
            "best_solution": best,
            "solution_table": _summarize_solution(best, feature_map),
            "ratios": _mix_ratios(best, feature_map),
            "candidate_schemes": _representative_mix_schemes(result, feature_map, requirements, setup),
            "traditional_theory": _traditional_theory_checks(best, feature_map, setup),
            "kpis": kpis,
            "top_solutions": result.get("solutions", [])[:20],
            "history": result.get("history", []),
            "comparison_metrics": result.get("comparison_metrics", []),
            "prior_feasible_count": result.get("prior_feasible_count", 0),
            "objectives": setup["objectives"],
            "constraints": setup["constraints"],
            "constraint_status": setup.get("constraint_status", {}),
            "standard_constraint_visualization": setup.get("standard_constraint_visualization", {}),
            "compiled_normative_constraints": setup.get("compiled_normative_constraints", []),
            "llm_proposal": setup.get("llm_proposal", {}),
        }
        _attach_design_audits(requirements, optimization_summary, kg_payload, setup, feature_map)
        llm_mechanism_review = await asyncio.to_thread(_mechanism_review_llm, requirements, kg_payload, optimization_summary)
        evidence_review = _evidence_second_pass_review(requirements, kg_payload, optimization_summary, feature_map)
        mechanism_review = _merge_mechanism_reviews(llm_mechanism_review, evidence_review)
        yield _json_event("kg_revision_decision", review=mechanism_review)
        if mechanism_review.get("trigger_second_pass"):
            revised_setup, applied = _apply_bound_adjustments(setup, feature_map, mechanism_review)
            if applied:
                yield _json_event("optimizer_revision_started", adjustments=applied)
                try:
                    revised_result = ml_optimization.run_multiobjective_optimization(
                        revised_setup["bounds"],
                        revised_setup["objectives"],
                        revised_setup["constraints"],
                        revised_setup["algorithm"],
                        prediction_fn,
                        pop_size=revised_setup["pop_size"],
                        generations=revised_setup["generations"],
                        random_state=84,
                        fmt="png",
                        dpi=160,
                        compare_algorithms=revised_setup["compare_algorithms"],
                    )
                    revised_result = _retain_prior_feasible_solutions(revised_result, feature_map, revised_setup)
                    if not revised_result.get("best_solution"):
                        yield _json_event("optimizer_revision_failed", message="机理二次优化未找到满足传统先验的可行解，保留首轮推荐方案。")
                        revised_result = None
                    if revised_result is None:
                        raise RuntimeError("second-pass-no-prior-feasible-solution")
                    result = revised_result
                    setup = revised_setup
                    best = result.get("best_solution")
                    kpis = _solution_kpis(best, setup, feature_map)
                    role_evidence = _role_evidence_bundle(requirements)
                    kg_payload = _kg_enhancement(requirements, feature_map, best, role_evidence["mechanism_retrieval"])
                    kg_payload["role_evidence"] = role_evidence
                    kg_trace = _kg_trace(kg_payload)
                    optimization_summary.update(
                        {
                            "best_algorithm": result.get("best_algorithm"),
                            "metrics": result.get("metrics"),
                            "best_solution": best,
                            "solution_table": _summarize_solution(best, feature_map),
                            "ratios": _mix_ratios(best, feature_map),
                            "candidate_schemes": _representative_mix_schemes(result, feature_map, requirements, setup),
                            "traditional_theory": _traditional_theory_checks(best, feature_map, setup),
                            "kpis": kpis,
                            "top_solutions": result.get("solutions", [])[:20],
                            "history": result.get("history", []),
                            "comparison_metrics": result.get("comparison_metrics", []),
                            "prior_feasible_count": result.get("prior_feasible_count", 0),
                            "objectives": setup["objectives"],
                            "constraints": setup["constraints"],
                            "constraint_status": setup.get("constraint_status", {}),
                            "standard_constraint_visualization": setup.get("standard_constraint_visualization", {}),
                            "compiled_normative_constraints": setup.get("compiled_normative_constraints", []),
                            "revision_pass": True,
                            "mechanism_adjustments": applied,
                        }
                    )
                    _attach_design_audits(requirements, optimization_summary, kg_payload, setup, feature_map)
                    yield _json_event("optimizer_revision_completed", kpis=kpis, adjustments=applied)
                except Exception as exc:
                    yield _json_event("optimizer_revision_failed", message=str(exc))
        report_requests = await asyncio.to_thread(_report_artifact_request_llm, prompt, requirements, optimization_summary, kg_payload)
        agents = [
            {"id": "requirement", "name": "Requirement Parsing Agent" if language == "en" else "需求解析智能体", "status": "completed", "summary": (f"Identified target strength {requirements['target_mpa']:g} MPa, curing age {requirements['age_days']:g} d, and priorities: {', '.join(requirements['priorities'])}." if language == "en" else f"识别目标强度 {requirements['target_mpa']:g} MPa；龄期 {requirements['age_days']:g} d；优先级：{', '.join(requirements['priorities'])}"), "details": {**requirements, "evidence": role_evidence["requirement_analysis"], "citations": role_evidence["requirement_analysis"].get("citations", [])}, "trace": _requirement_trace(requirements), "steps": _requirement_steps(requirements)},
            {"id": "optimizer", "name": "Optimization Modeling Agent" if language == "en" else "优化建模智能体", "status": "completed", "summary": (f"Generated {len(setup['objectives'])} objective functions and {len(setup['constraints'])} constraints, then ran {result.get('best_algorithm')}." if language == "en" else f"生成 {len(setup['objectives'])} 个目标函数、{len(setup['constraints'])} 个约束，并运行 {result.get('best_algorithm')}。"), "details": {"algorithm": setup["algorithm"], "selected_algorithm": result.get("best_algorithm"), "objectives": setup["objectives"], "constraints": setup["constraints"], "bounds_count": len(bounds), "metrics": result.get("metrics"), "kpis": kpis, "constraint_status": setup.get("constraint_status", {}), "standard_constraint_visualization": setup.get("standard_constraint_visualization", {}), "compiled_normative_constraints": setup.get("compiled_normative_constraints", []), "traditional_theory": setup.get("traditional_theory", {}), "evidence": role_evidence["optimization_modeling"], "citations": role_evidence["optimization_modeling"].get("citations", [])}, "trace": _optimization_trace(setup, kpis), "steps": _optimization_steps(setup, result, kpis)},
            {"id": "kg", "name": "Mechanism Retrieval Agent" if language == "en" else "机理检索智能体", "status": "completed", "summary": (f"Retrieved {len(kg_payload['hits'])} material-mechanism groups and generated {len(kg_payload['recommendations'])} optimization recommendations." if language == "en" else f"命中 {len(kg_payload['hits'])} 类材料机理，生成 {len(kg_payload['recommendations'])} 条优化建议。"), "details": {**kg_payload, "evidence": role_evidence["mechanism_retrieval"], "citations": role_evidence["mechanism_retrieval"].get("citations", []), "review": mechanism_review}, "trace": kg_trace, "steps": _kg_steps(requirements, kg_payload)},
        ]
        yield _json_event("report_started")
        compact = _compact_for_lm(prompt, optimization_summary, kg_payload, agents)
        compact["report_requests"] = report_requests
        merged_evidence = role_evidence.get("merged", {})
        shared_evidence = _citations_from_evidence(merged_evidence)
        compact["citations"] = shared_evidence
        compact["role_evidence_summary"] = {
            key: {
                "policy": value.get("policy", {}),
                "citations": value.get("citations", []),
                "selected_count": len(value.get("selected", [])),
            }
            for key, value in role_evidence.items()
            if key != "merged"
        }
        dialogues = {}
        lm_ok = True
        stream_specs = [
            ("requirement", "Requirement Parsing Agent" if language == "en" else "需求解析智能体", _load_skill_prompt("requirement_analysis", AGENT_PROMPTS["requirement"], language), {"user_request": prompt, "parsed_spec": requirements, "shared_evidence": role_evidence["requirement_analysis"].get("citations", []), "retrieval_policy": role_evidence["requirement_analysis"].get("policy", {})}, 10000, 0.25),
            ("optimizer", "Optimization Modeling Agent" if language == "en" else "优化建模智能体", _load_skill_prompt("optimization_modeling", AGENT_PROMPTS["optimizer"], language), {"requirement_agent_output": None, "optimization_summary": compact, "shared_evidence": role_evidence["optimization_modeling"].get("citations", []), "retrieval_policy": role_evidence["optimization_modeling"].get("policy", {})}, 10000, 0.25),
            (
                "kg",
                "Mechanism Retrieval Agent" if language == "en" else "机理检索智能体",
                _load_skill_prompt("mechanism_retrieval", AGENT_PROMPTS["kg"], language),
                {
                    "user_request": prompt,
                    "optimization_summary": compact,
                    "knowledge_graph_hits": compact.get("kg_hits"),
                    "recommendations": compact.get("recommendations"),
                    "shared_evidence": role_evidence["mechanism_retrieval"].get("citations", []),
                    "retrieval_policy": role_evidence["mechanism_retrieval"].get("policy", {}),
                },
                10000,
                0.25,
            ),
        ]
        try:
            for agent_id, agent_name, system_prompt, user_payload, max_tokens, temperature in stream_specs:
                if agent_id == "optimizer":
                    user_payload["requirement_agent_output"] = dialogues.get("requirement")
                yield _json_event("agent_output_started", agent_id=agent_id, agent_name=agent_name)
                parts = []
                for chunk in _stream_chat_lm(system_prompt, user_payload, max_tokens=max_tokens, temperature=temperature):
                    parts.append(chunk)
                    yield _json_event("agent_output_chunk", agent_id=agent_id, chunk=chunk)
                    await asyncio.sleep(0)
                dialogues[agent_id] = "".join(parts)
                yield _json_event("agent_output_completed", agent_id=agent_id)

            final_prompt = _load_skill_prompt("report_generation", REPORT_SKILL_PROMPT, language)
            yield _json_event("agent_output_started", agent_id="final", agent_name="Final Report Agent" if language == "en" else "总报告智能体")
            parts = []
            for chunk in _stream_chat_lm(
                final_prompt,
                {
                    "user_request": prompt,
                    "requirement_agent_output": dialogues.get("requirement"),
                    "optimizer_agent_output": dialogues.get("optimizer"),
                    "kg_agent_output": dialogues.get("kg"),
                    "structured_result": compact,
                    "shared_evidence": shared_evidence,
                    "role_evidence": {key: value.get("citations", []) for key, value in role_evidence.items() if key != "merged"},
                },
                max_tokens=24000,
                temperature=0.3,
            ):
                parts.append(chunk)
                yield _json_event("agent_output_chunk", agent_id="final", chunk=chunk)
                await asyncio.sleep(0)
            dialogues["final"] = "".join(parts)
            enriched_final = _ensure_reference_section(dialogues["final"], shared_evidence)
            appendix = enriched_final[len(dialogues["final"]):]
            if appendix:
                dialogues["final"] = enriched_final
                yield _json_event("agent_output_chunk", agent_id="final", chunk=appendix)
            yield _json_event("agent_output_completed", agent_id="final")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            lm_ok = False
            dialogues["error"] = str(exc)
            dialogues["final"] = _fallback_answer(prompt, optimization_summary, kg_payload, str(exc), language)
            yield _json_event("agent_output_chunk", agent_id="final", chunk=dialogues["final"])
        for agent in agents:
            agent["dialogue"] = dialogues.get(agent["id"])
        data = {
            "answer": dialogues.get("final"),
            "lm_ok": lm_ok,
            "agents": agents,
            "optimization": optimization_summary,
            "kg_hits": kg_payload["hits"],
            "recommendations": kg_payload["recommendations"],
            "visualizations": {"optimization_plots": _select_plot_previews(result)},
            "report_requests": report_requests,
            "mechanism_review": mechanism_review,
            "role_evidence": {
                key: {
                    "policy": value.get("policy", {}),
                    "citations": value.get("citations", []),
                    "selected": value.get("selected", []),
                    "query_terms": value.get("query_terms", []),
                }
                for key, value in role_evidence.items()
            },
            "handoffs": _handoff_payload(requirements, setup, optimization_summary, kg_payload, report_requests),
        }
        yield _json_event("done", data=data)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


class _JsonRequestShim:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


def _decode_testml_export_response(response):
    if isinstance(response, JSONResponse):
        try:
            return json.loads(response.body.decode("utf-8"))
        except Exception:
            return {"status": "error", "message": "Unable to decode export response."}
    return response


async def _build_testml_plot_workbook(context, plot_name, payload):
    if TESTML_MODULE is None:
        return {"status": "error", "message": TESTML_API_IMPORT_ERROR or "testML API is not loaded."}
    response = await TESTML_MODULE.export_plot_data(_JsonRequestShim({
        "context": context,
        "plot_name": plot_name,
        "payload": payload or {},
    }))
    return _decode_testml_export_response(response)


@app.post("/testml/api/export/plot_data")
async def export_testml_plot_data_with_metadata(request: Request):
    data = await request.json()
    result = await _build_testml_plot_workbook(
        data.get("context", "dataset"),
        data.get("plot_name", "Plot"),
        data.get("payload", {}) or {},
    )
    if result.get("status") != "success":
        return JSONResponse(status_code=400, content=result)
    return result


@app.post("/testml/api/export/zip")
async def export_testml_zip_bundle_with_plot_data(request: Request):
    if TESTML_MODULE is None:
        return JSONResponse(status_code=503, content={"status": "error", "message": TESTML_API_IMPORT_ERROR or "testML API is not loaded."})
    try:
        data = await request.json()
        bundle_name = TESTML_MODULE._slugify_filename(data.get("bundle_name", "NatureML_Export"))
        items = [item for item in data.get("items", []) if item and item.get("data_url")]
        if not items:
            return JSONResponse(status_code=400, content={"status": "error", "message": "No export items were provided."})

        zip_buffer = io.BytesIO()
        manifest = []
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            used_names = set()

            def unique_name(name):
                base, ext = os.path.splitext(name)
                candidate = name
                suffix = 2
                while candidate in used_names:
                    candidate = f"{base}_{suffix}{ext}"
                    suffix += 1
                used_names.add(candidate)
                return candidate

            for idx, item in enumerate(items, start=1):
                _, ext, image_payload = TESTML_MODULE._decode_data_url(item.get("data_url"))
                base_name = TESTML_MODULE._slugify_filename(item.get("name", f"plot_{idx}"))
                image_name = unique_name(f"{base_name}.{ext}")
                zf.writestr(image_name, image_payload)

                context = item.get("context", "dataset")
                plot_name = item.get("plot_name") or item.get("name") or f"plot_{idx}"
                payload = item.get("payload", {}) or {}
                workbook = await _build_testml_plot_workbook(context, plot_name, payload)
                workbook_name = None
                if workbook.get("status") == "success" and workbook.get("content_base64"):
                    workbook_name = unique_name(f"{base_name}_plot_data.xlsx")
                    zf.writestr(workbook_name, base64.b64decode(workbook["content_base64"]))
                else:
                    error_name = unique_name(f"{base_name}_plot_data_error.txt")
                    zf.writestr(error_name, workbook.get("message", "No workbook data available for this plot."))

                manifest.append({
                    "plot_name": plot_name,
                    "context": context,
                    "plot_type": TESTML_MODULE._plot_type_label(plot_name, context, payload),
                    "image_file": image_name,
                    "xlsx_file": workbook_name,
                })

            zf.writestr("plot_export_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

        return {
            "status": "success",
            "filename": f"{bundle_name}.zip",
            "content_base64": base64.b64encode(zip_buffer.getvalue()).decode("utf-8"),
        }
    except Exception as exc:
        return {"status": "error", "message": f"ZIP export failed: {str(exc)}"}


@app.post("/testml/api/eda/all")
async def export_testml_all_eda_with_plot_data(request: Request):
    if TESTML_MODULE is None:
        return JSONResponse(status_code=503, content={"status": "error", "message": TESTML_API_IMPORT_ERROR or "testML API is not loaded."})
    try:
        data = await request.json()
        df = TESTML_MODULE.SESSION_STATE.get("df")
        if df is None:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Dataset missing"})

        fmt = data.get("format", "png")
        dpi = int(data.get("dpi", 300))
        target_col = TESTML_MODULE.SESSION_STATE.get("target_col")
        columns = df.columns.tolist()
        feature_columns = [col for col in columns if col != target_col] or columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [col for col in numeric_columns if col != target_col]
        eda_module = TESTML_MODULE.eda_module
        items = []
        failures = []

        def collect(name, payload, builder):
            try:
                image = builder()
                if image:
                    items.append(plot_export := {
                        "name": name,
                        "data_url": image,
                        "context": "eda",
                        "plot_name": name,
                        "payload": payload,
                    })
            except Exception as exc:
                failures.append({"name": name, "message": str(exc)})

        for col in feature_columns:
            plot_types = ["hist", "density", "ecdf", "box", "violin", "strip"] if pd.api.types.is_numeric_dtype(df[col]) else ["hist"]
            for plot_type in plot_types:
                payload = {"col_x": col, "plot_type": plot_type}
                collect(
                    f"EDA_{plot_type}_{col}",
                    payload,
                    lambda col=col, plot_type=plot_type: eda_module.get_plot_distribution(df, col, plot_type, fmt, dpi),
                )

        feature_df = df[feature_columns]
        if len(numeric_features) >= 2:
            for method, plot_type in [("pearson", "corr"), ("spearman", "corr_spearman")]:
                collect(
                    f"EDA_{plot_type}_all_numeric_features",
                    {"plot_type": plot_type},
                    lambda method=method: eda_module.get_plot_correlation(feature_df, fmt, dpi, method),
                )
            collect(
                "EDA_pairplot_all_numeric_features",
                {"plot_type": "pair"},
                lambda: eda_module.get_plot_pairplot(df, fmt, dpi, target_col, max_cols=None),
            )
        if df[feature_columns].isna().sum().sum() > 0:
            collect(
                "EDA_missingness_all_features",
                {"plot_type": "missing"},
                lambda: eda_module.get_plot_missingness(feature_df, fmt, dpi),
            )

        if target_col:
            for col in feature_columns:
                collect(
                    f"EDA_target_{col}_vs_{target_col}",
                    {"col_x": col, "plot_type": "target"},
                    lambda col=col: eda_module.get_plot_target_relation(df, col, target_col, fmt, dpi),
                )

        for col_x, col_y in combinations(numeric_features, 2):
            for plot_type in ["scatter", "reg", "hex", "joint"]:
                collect(
                    f"EDA_{plot_type}_{col_x}_vs_{col_y}",
                    {"col_x": col_x, "col_y": col_y, "plot_type": plot_type},
                    lambda col_x=col_x, col_y=col_y, plot_type=plot_type: eda_module.get_plot_scatter(
                        df, col_x, col_y, fmt, dpi, plot_type, target_col
                    ),
                )

        if not items:
            return JSONResponse(status_code=400, content={"status": "error", "message": "No EDA plots could be generated.", "failures": failures})

        return await export_testml_zip_bundle_with_plot_data(_JsonRequestShim({
            "bundle_name": "EDA_All_Feature_Plots",
            "items": items,
        }))
    except Exception as exc:
        return {"status": "error", "message": f"EDA ZIP export failed: {str(exc)}"}


def _mount_testml_api_routes():
    global TESTML_API_IMPORT_ERROR, TESTML_MODULE
    main_path = TEST_ML / "main.py"
    if not main_path.is_file():
        TESTML_API_IMPORT_ERROR = f"testML main.py not found: {main_path}"
        return

    previous_cwd = os.getcwd()
    inserted_path = False
    try:
        if str(TEST_ML) not in sys.path:
            sys.path.insert(0, str(TEST_ML))
            inserted_path = True
        os.chdir(str(TEST_ML))
        spec = importlib.util.spec_from_file_location("testml_embedded_main", str(main_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load testML module from {main_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        TESTML_MODULE = module
        testml_router = APIRouter()
        for route in module.app.routes:
            if (
                isinstance(route, APIRoute)
                and route.path.startswith("/api/")
                and route.path not in {"/api/export/plot_data", "/api/export/zip", "/api/eda/all"}
            ):
                testml_router.routes.append(route)
        app.include_router(testml_router, prefix="/testml")
    except Exception as exc:
        TESTML_API_IMPORT_ERROR = str(exc)
    finally:
        os.chdir(previous_cwd)
        if inserted_path:
            try:
                sys.path.remove(str(TEST_ML))
            except ValueError:
                pass


_mount_testml_api_routes()


app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8060")))
