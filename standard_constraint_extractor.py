import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path


SOURCE_HINTS = [
    "自密实", "self-compacting", "self compacting", "scc",
    "泵送", "pump", "quality control", "质量控制",
    "配合比", "mix design", "proportion",
    "坍落", "slump", "flow", "流动",
    "compressive", "strength", "抗压", "强度",
    "高性能", "high performance", "高强", "high strength",
    "活性粉末", "reactive powder", "rpc", "uhpc", "ultra-high",
    "水运", "海工", "marine", "耐久", "durability",
    "c39", "c143", "c1611", "c1621", "c39m", "c143m",
    "普通混凝土配合比", "混凝土质量控制",
]

METRIC_ALIASES = {
    "compressive_strength": ["抗压强度", "立方体抗压", "compressive strength"],
    "strength_grade": ["强度等级", "strength class", "grade"],
    "slump": ["坍落度", "slump"],
    "slump_flow": ["坍落扩展度", "扩展度", "slump flow", "flow spread", "flow diameter"],
    "flowability": ["流动性", "流动度", "fluidity", "flowability"],
    "t500": ["t500", "t 500", "T500"],
}

DOMAIN_ALIASES = {
    "self_compacting_concrete": ["自密实", "self-compacting", "self compacting", "scc"],
    "pumped_concrete": ["泵送", "pump", "pumped"],
    "high_strength_concrete": ["高强", "high strength"],
    "high_performance_concrete": ["高性能", "high performance"],
    "reactive_powder_concrete": ["活性粉末", "reactive powder", "rpc", "uhpc", "ultra-high"],
    "marine_concrete": ["水运", "海工", "marine", "harbor", "港口"],
    "durability_concrete": ["耐久", "durability", "抗渗", "氯离子", "碳化", "冻融"],
    "mix_design_concrete": ["配合比", "mix design", "proportion"],
    "general_concrete": ["混凝土", "concrete"],
}

STANDARD_DOMAIN_LABELS = {
    "self_compacting_concrete": "自密实混凝土",
    "pumped_concrete": "泵送混凝土",
    "high_strength_concrete": "高强混凝土",
    "high_performance_concrete": "高性能混凝土",
    "reactive_powder_concrete": "活性粉末/UHPC",
    "marine_concrete": "水运/海工混凝土",
    "durability_concrete": "耐久性混凝土",
    "mix_design_concrete": "配合比设计",
    "general_concrete": "普通混凝土",
}

OPERATOR_ALIASES = [
    (">=", ["不应小于", "不宜小于", "不得小于", "应不小于", "大于等于", "不低于", "at least", "not less than", "minimum"]),
    ("<=", ["不应大于", "不宜大于", "不得大于", "应不大于", "小于等于", "不高于", "not greater than", "not more than", "maximum"]),
    ("range", ["～", "~", "-", "至", "到", "between", "from"]),
]


@dataclass
class Constraint:
    source_file: str
    source_path: str
    page: int | None
    clause: str
    domain: str
    metric: str
    operator: str
    lower: float | None
    upper: float | None
    value: float | None
    unit: str
    evidence: str
    confidence: float
    extraction_method: str
    compile_status: str


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_text_pages(path: Path) -> list[dict]:
    pages = []
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
                tables = []
                for table in page.extract_tables() or []:
                    rows = [" | ".join(str(cell or "").strip() for cell in row) for row in table if row]
                    tables.extend(row for row in rows if row.strip())
                pages.append({"page": i, "text": normalize_text(text), "tables": tables})
        if any(item["text"].strip() for item in pages):
            return pages
    except Exception:
        pass
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, 1):
            pages.append({"page": i, "text": normalize_text(page.extract_text() or ""), "tables": []})
    except Exception:
        return []
    return pages


def likely_relevant_pdf(path: Path) -> bool:
    lower = str(path).lower()
    return any(hint.lower() in lower for hint in SOURCE_HINTS)


def detect_domain(text: str, filename: str) -> str:
    hay = f"{filename}\n{text}".lower()
    for domain, aliases in DOMAIN_ALIASES.items():
        if any(alias.lower() in hay for alias in aliases):
            return domain
    return "general_concrete"


def detect_metric(text: str) -> str | None:
    lower = text.lower()
    for metric, aliases in METRIC_ALIASES.items():
        if any(alias.lower() in lower for alias in aliases):
            return metric
    return None


def detect_operator(text: str) -> str:
    lower = text.lower()
    for op, aliases in OPERATOR_ALIASES:
        if any(alias.lower() in lower for alias in aliases):
            return op
    return "mentioned"


def extract_clause(text: str) -> str:
    match = re.search(r"(?<![A-Z])(\d+(?:\.\d+){1,4})(?!\d)", text)
    return match.group(1) if match else ""


def clean_evidence(text: str, max_len: int = 520) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def context_windows(text: str) -> list[str]:
    keywords = [
        "自密实", "坍落扩展度", "扩展度", "坍落度", "T500", "流动度", "流动性",
        "抗压强度", "强度等级", "配制强度", "slump flow", "slump", "compressive strength",
        "self-compacting", "self compacting",
    ]
    windows = []
    for keyword in keywords:
        for match in re.finditer(re.escape(keyword), text, flags=re.IGNORECASE):
            start = max(0, match.start() - 220)
            end = min(len(text), match.end() + 260)
            windows.append(text[start:end])
    chunks = re.split(r"(?<=[。；;.!?])\s+|\n+", text)
    for chunk in chunks:
        if any(keyword.lower() in chunk.lower() for keyword in keywords):
            windows.append(chunk)
    dedup = []
    seen = set()
    for window in windows:
        key = re.sub(r"\s+", "", window)[:220]
        if key and key not in seen:
            seen.add(key)
            dedup.append(window)
    return dedup


def split_clauses(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = re.split(r"(?=(?:\d+(?:\.\d+){1,4}\s+))|(?<=[。；;.!?])\s+", text)
    cleaned = []
    for chunk in chunks:
        chunk = chunk.strip()
        if 20 <= len(chunk) <= 900:
            cleaned.append(chunk)
    return cleaned


def make_constraint(
    *,
    source_file: str,
    source_path: Path,
    page: int | None,
    text: str,
    domain: str,
    metric: str,
    operator: str,
    lower: float | None = None,
    upper: float | None = None,
    value: float | None = None,
    unit: str = "mm",
    confidence: float = 0.88,
    method: str = "high_precision_clause",
    compile_status: str = "optimizable",
) -> Constraint:
    return Constraint(
        source_file=source_file,
        source_path=str(source_path),
        page=page,
        clause=extract_clause(text),
        domain=domain,
        metric=metric,
        operator=operator,
        lower=lower,
        upper=upper,
        value=value,
        unit=unit,
        evidence=clean_evidence(text),
        confidence=confidence,
        extraction_method=method,
        compile_status=compile_status,
    )


def extract_high_precision_constraints(text: str, filename: str, path: Path, page: int | None, method: str) -> list[Constraint]:
    results = []
    text = normalize_text(text)
    for clause in split_clauses(text):
        domain = detect_domain(clause, filename)
        compact = re.sub(r"\s+", "", clause)
        compact = compact.replace("混凝土", "混凝土")

        for match in re.finditer(r"(自密实混凝土|self[- ]compacting concrete|SCC)[^。；;]{0,80}?(?:坍落扩展度|扩展度|slump[- ]flow)[^。；;]{0,30}?(?:不应小于|不宜小于|不得小于|不小于|not less than|minimum)\s*(\d{3})\s*(?:mm|毫米)?", clause, flags=re.IGNORECASE):
            value = float(match.group(2))
            if 450 <= value <= 900:
                results.append(make_constraint(source_file=filename, source_path=path, page=page, text=clause, domain="self_compacting_concrete", metric="slump_flow", operator=">=", lower=value, value=value, confidence=0.94, method=method))
        for match in re.finditer(r"(自密实混凝土|SCC)[^。；;]{0,80}?(?:坍落扩展度|扩展度)[^。；;]{0,30}?(?:不应小于|不宜小于|不得小于|不小于)(\d{3})(?:mm|毫米)?", compact, flags=re.IGNORECASE):
            value = float(match.group(2))
            if 450 <= value <= 900:
                results.append(make_constraint(source_file=filename, source_path=path, page=page, text=clause, domain="self_compacting_concrete", metric="slump_flow", operator=">=", lower=value, value=value, confidence=0.94, method=method))

        for match in re.finditer(r"(泵送高强混凝土|泵送混凝土|pumped concrete)[^。；;]{0,80}?(?:坍落扩展度|扩展度|slump[- ]flow)[^。；;]{0,30}?(?:不应小于|不宜小于|不得小于|不小于|not less than|minimum)\s*(\d{3})\s*(?:mm|毫米)?", clause, flags=re.IGNORECASE):
            value = float(match.group(2))
            if 350 <= value <= 900:
                results.append(make_constraint(source_file=filename, source_path=path, page=page, text=clause, domain="pumped_concrete", metric="slump_flow", operator=">=", lower=value, value=value, confidence=0.92, method=method))
        for match in re.finditer(r"(泵送高强混凝土|泵送混凝土)[^。；;]{0,80}?(?:坍落扩展度|扩展度)[^。；;]{0,30}?(?:不应小于|不宜小于|不得小于|不小于)(\d{3})(?:mm|毫米)?", compact, flags=re.IGNORECASE):
            value = float(match.group(2))
            if 350 <= value <= 900:
                results.append(make_constraint(source_file=filename, source_path=path, page=page, text=clause, domain="pumped_concrete", metric="slump_flow", operator=">=", lower=value, value=value, confidence=0.92, method=method))

        for match in re.finditer(r"(泵送混凝土|pumped concrete)[^。；;]{0,80}?(?:坍落度|slump)[^。；;]{0,30}?(?:不应大于|不宜大于|不得大于|不大于|not greater than|maximum)\s*(\d{2,3})\s*(?:mm|毫米)?", clause, flags=re.IGNORECASE):
            value = float(match.group(2))
            if 50 <= value <= 260:
                results.append(make_constraint(source_file=filename, source_path=path, page=page, text=clause, domain="pumped_concrete", metric="slump", operator="<=", upper=value, value=value, confidence=0.92, method=method))

        table_like = compact.replace("一", "-").replace("—", "-")
        if "坍落扩展度" in table_like or "slump-flow" in table_like.lower() or "slumpflow" in table_like.lower():
            for cls, lo, hi in re.findall(r"(SF\d)\s*(\d{3})\s*[～~\\-]\s*(\d{3})", table_like, flags=re.IGNORECASE):
                lower, upper = float(lo), float(hi)
                if 450 <= lower <= upper <= 900:
                    results.append(make_constraint(
                        source_file=filename,
                        source_path=path,
                        page=page,
                        text=f"{clause} | extracted_class={cls.upper()}",
                        domain="self_compacting_concrete",
                        metric="slump_flow",
                        operator="range",
                        lower=lower,
                        upper=upper,
                        unit="mm",
                        confidence=0.91,
                        method=method,
                    ))
            for cls, val in re.findall(r"(SF\d)\s*(?:≥|>=)\s*(\d{3})", table_like, flags=re.IGNORECASE):
                value = float(val)
                if 450 <= value <= 900:
                    results.append(make_constraint(
                        source_file=filename,
                        source_path=path,
                        page=page,
                        text=f"{clause} | extracted_class={cls.upper()}",
                        domain="self_compacting_concrete",
                        metric="slump_flow",
                        operator=">=",
                        lower=value,
                        value=value,
                        unit="mm",
                        confidence=0.89,
                        method=method,
                    ))

        if "扩展时间" in clause or "T500" in clause.upper() or "Tsoo" in clause:
            for cls, op, val in re.findall(r"(VS\d)\s*(≥|<=|≤|<|>|>=)\s*(\d+(?:\.\d+)?)", compact, flags=re.IGNORECASE):
                value = float(val)
                if 0.5 <= value <= 60:
                    normalized_op = ">=" if op in {"≥", ">="} else "<=" if op == "≤" else op
                    results.append(make_constraint(
                        source_file=filename,
                        source_path=path,
                        page=page,
                        text=f"{clause} | extracted_class={cls.upper()}",
                        domain="self_compacting_concrete",
                        metric="t500",
                        operator=normalized_op,
                        lower=value if normalized_op in {">=", ">"} else None,
                        upper=value if normalized_op in {"<=", "<"} else None,
                        value=value,
                        unit="s",
                        confidence=0.86,
                        method=method,
                        compile_status="reference_only",
                    ))

        if re.search(r"抗压强度|compressive strength", clause, flags=re.IGNORECASE):
            for match in re.finditer(r"(?<![A-Z/])C\s*(\d{2,3})(?![\dA-Za-z])", clause, flags=re.IGNORECASE):
                prefix = clause[max(0, match.start() - 18):match.start()]
                if re.search(r"(GB|JGJ|JC|ACI|ASTM|EN|ISO)\s*/?\s*$", prefix, flags=re.IGNORECASE):
                    continue
                value = float(match.group(1))
                if 10 <= value <= 120:
                    results.append(make_constraint(
                        source_file=filename,
                        source_path=path,
                        page=page,
                        text=clause,
                        domain=domain,
                        metric="strength_grade",
                        operator="class",
                        lower=value,
                        value=value,
                        unit="MPa",
                        confidence=0.70,
                        method=method,
                        compile_status="reference_only",
                    ))
            for match in re.finditer(r"(?:抗压强度|compressive strength)[^。；;]{0,80}?(?:不应小于|不宜小于|不得小于|不小于|不低于|not less than|minimum)\s*(\d{1,3}(?:\.\d+)?)\s*(?:MPa|兆帕)", clause, flags=re.IGNORECASE):
                value = float(match.group(1))
                if 5 <= value <= 200:
                    results.append(make_constraint(
                        source_file=filename,
                        source_path=path,
                        page=page,
                        text=clause,
                        domain=domain,
                        metric="compressive_strength",
                        operator=">=",
                        lower=value,
                        value=value,
                        unit="MPa",
                        confidence=0.84,
                        method=method,
                        compile_status="reference_only",
                    ))
    return results


def parse_numeric_constraint(window: str, filename: str, path: Path, page: int | None, method: str) -> list[Constraint]:
    metric = detect_metric(window)
    if not metric:
        return []
    domain = detect_domain(window, filename)
    op = detect_operator(window)
    constraints = []
    unit = "MPa" if metric in {"compressive_strength", "strength_grade"} else ("s" if metric == "t500" else "mm")

    if metric == "strength_grade":
        grade_matches = []
        for match in re.finditer(r"(?<![A-Z/])C\s*(\d{2,3})(?![\dA-Za-z])", window, flags=re.IGNORECASE):
            prefix = window[max(0, match.start() - 18):match.start()]
            if re.search(r"(GB|JGJ|JC|ACI|ASTM|EN|ISO)\s*/?\s*$", prefix, flags=re.IGNORECASE):
                continue
            if not re.search(r"强度等级|抗压强度|混凝土|等级|strength|grade|class", window, flags=re.IGNORECASE):
                continue
            value = float(match.group(1))
            if 10 <= value <= 120:
                grade_matches.append(value)
        for value in sorted(set(grade_matches)):
            constraints.append(Constraint(
                source_file=filename,
                source_path=str(path),
                page=page,
                clause=extract_clause(window),
                domain=domain,
                metric=metric,
                operator="class",
                lower=value,
                upper=None,
                value=value,
                unit="MPa",
                evidence=clean_evidence(window),
                confidence=0.58,
                extraction_method=method,
                compile_status="reference_only",
            ))
        return constraints

    range_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm|毫米|s|秒|MPa)?\s*(?:～|~|至|到|--?|—)\s*(\d+(?:\.\d+)?)\s*(mm|毫米|s|秒|MPa)?", window, flags=re.IGNORECASE)
    if range_match:
        lower = float(range_match.group(1))
        upper = float(range_match.group(2))
        if lower > upper:
            lower, upper = upper, lower
        constraints.append(Constraint(
            source_file=filename,
            source_path=str(path),
            page=page,
            clause=extract_clause(window),
            domain=domain,
            metric=metric,
            operator="range",
            lower=lower,
            upper=upper,
            value=None,
            unit=unit,
            evidence=clean_evidence(window),
            confidence=0.78,
            extraction_method=method,
            compile_status="optimizable" if metric in {"slump", "slump_flow", "flowability"} else "reference_only",
        ))
        return constraints

    number_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(mm|毫米|s|秒|MPa)?", window, flags=re.IGNORECASE)
    for raw_value, raw_unit in number_matches:
        value = float(raw_value)
        if metric in {"slump", "slump_flow", "flowability"} and not (50 <= value <= 900):
            continue
        if metric == "t500" and not (0.5 <= value <= 60):
            continue
        if metric == "compressive_strength" and not (5 <= value <= 200):
            continue
        lower = value if op == ">=" else None
        upper = value if op == "<=" else None
        constraints.append(Constraint(
            source_file=filename,
            source_path=str(path),
            page=page,
            clause=extract_clause(window),
            domain=domain,
            metric=metric,
            operator=op,
            lower=lower,
            upper=upper,
            value=value,
            unit=raw_unit.replace("毫米", "mm").replace("秒", "s") if raw_unit else unit,
            evidence=clean_evidence(window),
            confidence=0.72 if op in {">=", "<=", "range"} else 0.45,
            extraction_method=method,
            compile_status="optimizable" if metric in {"slump", "slump_flow", "flowability"} and op in {">=", "<=", "range"} else "reference_only",
        ))
    return constraints


def extract_constraints_from_pdf(path: Path, recall_mode: bool = False) -> list[Constraint]:
    pages = extract_text_pages(path)
    results = []
    for page in pages:
        text = page["text"]
        results.extend(extract_high_precision_constraints(text, path.name, path, page["page"], "high_precision_text"))
        if recall_mode:
            for window in context_windows(text):
                results.extend(parse_numeric_constraint(window, path.name, path, page["page"], "text_window"))
        for row in page.get("tables", []):
            results.extend(extract_high_precision_constraints(row, path.name, path, page["page"], "high_precision_table"))
            if recall_mode and detect_metric(row):
                results.extend(parse_numeric_constraint(row, path.name, path, page["page"], "table_row"))
    dedup = {}
    for item in results:
        key = (item.source_file, item.page, item.clause, item.domain, item.metric, item.operator, item.lower, item.upper, item.value, item.evidence[:120])
        old = dedup.get(key)
        if old is None or item.confidence > old.confidence:
            dedup[key] = item
    return sorted(dedup.values(), key=lambda x: (x.source_file, x.page or 0, x.metric, -(x.confidence)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(Path(__file__).resolve().parent / "data" / "standards"))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "evidence" / "standard-constraints"))
    parser.add_argument("--all-pdfs", action="store_true", help="Process all PDFs instead of filename-filtered PDFs.")
    parser.add_argument("--recall", action="store_true", help="Also include broad context-window extraction. Default is high precision only.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(source.rglob("*.pdf"))
    if not args.all_pdfs:
        pdfs = [path for path in pdfs if likely_relevant_pdf(path)]
    if args.limit:
        pdfs = pdfs[: args.limit]

    jsonl_path = out / "standard_strength_flow_constraints.jsonl"
    csv_path = out / "standard_strength_flow_constraints.csv"
    summary_path = out / "standard_extraction_summary.json"
    all_rows = []
    for index, pdf in enumerate(pdfs, 1):
        progress = f"[{index}/{len(pdfs)}] {pdf}"
        try:
            print(progress)
        except UnicodeEncodeError:
            print(progress.encode("ascii", errors="backslashreplace").decode("ascii"))
        try:
            all_rows.extend(extract_constraints_from_pdf(pdf, recall_mode=args.recall))
        except Exception as exc:
            print(f"  failed: {exc}")

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    fields = list(asdict(all_rows[0]).keys()) if all_rows else [field.name for field in Constraint.__dataclass_fields__.values()]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))
    summary = {
        "source": str(source),
        "processed_pdf_count": len(pdfs),
        "constraint_count": len(all_rows),
        "optimizable_count": sum(1 for row in all_rows if row.compile_status == "optimizable"),
        "metrics": {},
        "domains": {},
        "outputs": {"jsonl": str(jsonl_path), "csv": str(csv_path)},
    }
    for row in all_rows:
        summary["metrics"][row.metric] = summary["metrics"].get(row.metric, 0) + 1
        summary["domains"][row.domain] = summary["domains"].get(row.domain, 0) + 1
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
