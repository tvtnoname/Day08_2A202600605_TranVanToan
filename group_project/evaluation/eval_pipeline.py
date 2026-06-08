"""
RAG Evaluation Pipeline — Thành viên 3 (Evaluation Engineer).

Framework: DeepEval (LLM-as-a-judge). Giám khảo = router OpenAI-compatible của nhóm
(claude-haiku, temperature=0) → không cần OpenAI key thật. Xem eval_models.py.

Quy trình:
    1. Load golden_dataset.json (≥15 cặp Q&A).
    2. Với MỖI config (A/B), chạy RAG pipeline (src.rag_engine.generate_with_citation)
       trên từng câu hỏi → tạo LLMTestCase.
    3. Chấm 4 metrics: Faithfulness, Answer Relevancy, Contextual Recall, Contextual Precision.
    4. So sánh A/B (mặc định: hybrid_rerank vs hybrid_no_rerank).
    5. Phân tích worst performers, xuất results.md + eval_results.json.

Chạy:
    cd group_project
    python -m evaluation.eval_pipeline                # chạy toàn bộ
    python -m evaluation.eval_pipeline --limit 6      # chạy nhanh 6 câu
    python -m evaluation.eval_pipeline --configs hybrid_rerank dense_only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_DISABLE_PROGRESS_BAR", "YES")

EVAL_DIR = Path(__file__).resolve().parent
GROUP_DIR = EVAL_DIR.parent
sys.path.insert(0, str(GROUP_DIR))  # để import src.rag_engine / src.config

from src.config import RETRIEVAL_CONFIGS  # noqa: E402
from src.rag_engine import generate_with_citation  # noqa: E402

from deepeval.test_case import LLMTestCase  # noqa: E402
from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)

from evaluation.eval_models import JUDGE_MODEL, get_judge_model  # noqa: E402

GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_MD = EVAL_DIR / "results.md"
RESULTS_JSON = EVAL_DIR / "eval_results.json"

THRESHOLD = 0.7
METRIC_NAMES = ["faithfulness", "answer_relevancy", "contextual_recall", "contextual_precision"]
# A/B mặc định: có rerank vs không rerank.
DEFAULT_COMPARE = ["hybrid_rerank", "hybrid_no_rerank"]


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_metrics(judge):
    """Tạo 4 metric mới (dùng chung 1 judge)."""
    common = dict(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False)
    return {
        "faithfulness": FaithfulnessMetric(**common),
        "answer_relevancy": AnswerRelevancyMetric(**common),
        "contextual_recall": ContextualRecallMetric(**common),
        "contextual_precision": ContextualPrecisionMetric(**common),
    }


def evaluate_config(config_name: str, dataset: list[dict], judge) -> dict:
    """Chạy RAG theo 1 config + chấm 4 metrics cho toàn bộ dataset."""
    print(f"\n{'='*70}\n▶ CONFIG: {config_name}\n{'='*70}")
    metrics = build_metrics(judge)
    per_case = []

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        result = generate_with_citation(question, config_name=config_name)
        contexts = [c["content"] for c in result["sources"]]
        tc = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            expected_output=item.get("expected_answer", ""),
            retrieval_context=contexts,
        )

        scores, reasons = {}, {}
        for name, metric in metrics.items():
            try:
                metric.measure(tc)
                scores[name] = round(float(metric.score), 4)
                reasons[name] = metric.reason
            except Exception as e:  # 1 metric lỗi không làm hỏng cả run
                scores[name] = None
                reasons[name] = f"ERROR: {e}"

        valid = [v for v in scores.values() if v is not None]
        case_avg = round(sum(valid) / len(valid), 4) if valid else 0.0
        per_case.append({
            "question": question,
            "answer": result["answer"],
            "n_sources": len(result["sources"]),
            "scores": scores,
            "case_avg": case_avg,
            "reasons": reasons,
        })
        bar = " ".join(f"{n[:4]}={scores[n] if scores[n] is not None else 'ERR'}" for n in METRIC_NAMES)
        print(f"  [{i}/{len(dataset)}] avg={case_avg:.2f} | {bar} | {question[:45]}")

    # Trung bình từng metric trên toàn config.
    agg = {}
    for name in METRIC_NAMES:
        vals = [c["scores"][name] for c in per_case if c["scores"][name] is not None]
        agg[name] = round(sum(vals) / len(vals), 4) if vals else 0.0
    agg["overall"] = round(sum(agg[n] for n in METRIC_NAMES) / len(METRIC_NAMES), 4)

    return {"config_name": config_name, "aggregate": agg, "per_case": per_case}


def write_results_md(results: list[dict], dataset_size: int):
    lines = []
    lines.append("# RAG Evaluation Results\n")
    lines.append("- **Framework:** DeepEval (LLM-as-a-judge)")
    lines.append(f"- **Judge model:** `{JUDGE_MODEL}` (temperature=0)")
    lines.append(f"- **Golden dataset:** {dataset_size} câu hỏi")
    lines.append(f"- **Threshold pass:** {THRESHOLD}")
    lines.append("- **Metrics:** Faithfulness, Answer Relevancy, Contextual Recall, Contextual Precision\n")

    # Bảng tổng hợp / so sánh A/B.
    lines.append("## So sánh A/B (điểm trung bình)\n")
    header = "| Config | " + " | ".join(m.replace("_", " ").title() for m in METRIC_NAMES) + " | **Overall** |"
    sep = "|" + "---|" * (len(METRIC_NAMES) + 2)
    lines.append(header)
    lines.append(sep)
    for r in results:
        a = r["aggregate"]
        row = f"| `{r['config_name']}` | " + " | ".join(f"{a[m]:.3f}" for m in METRIC_NAMES) + f" | **{a['overall']:.3f}** |"
        lines.append(row)
    lines.append("")

    # Kết luận A/B.
    best = max(results, key=lambda r: r["aggregate"]["overall"])
    lines.append(f"**Cấu hình tốt nhất theo overall: `{best['config_name']}` ({best['aggregate']['overall']:.3f}).**\n")
    if len(results) >= 2:
        a, b = results[0], results[1]
        diff = a["aggregate"]["overall"] - b["aggregate"]["overall"]
        lines.append(
            f"> `{a['config_name']}` vs `{b['config_name']}`: chênh lệch overall = {diff:+.3f}. "
            + ("Reranking giúp cải thiện chất lượng." if diff > 0 else
               "Reranking không cải thiện (hoặc làm giảm) trên tập này — cần xem lại reranker.")
            + "\n"
        )

    # Worst performers cho config tốt nhất.
    lines.append(f"## Worst performers (config `{best['config_name']}`)\n")
    worst = sorted(best["per_case"], key=lambda c: c["case_avg"])[:3]
    for c in worst:
        lines.append(f"### ❌ avg={c['case_avg']:.2f} — {c['question']}")
        lines.append(f"- **Answer:** {c['answer'][:200].replace(chr(10), ' ')}...")
        weak = min(METRIC_NAMES, key=lambda m: c["scores"][m] if c["scores"][m] is not None else 1)
        lines.append(f"- **Metric yếu nhất:** {weak} = {c['scores'][weak]}")
        lines.append(f"- **Lý do:** {str(c['reasons'].get(weak, ''))[:240]}\n")

    # Đề xuất cải tiến.
    lines.append("## Phân tích & Đề xuất cải tiến\n")
    lines.append("- **Contextual Recall thấp** → retriever bỏ sót evidence: tăng `top_k`, cải thiện chunking, "
                 "hoặc bổ sung query expansion/HyDE.")
    lines.append("- **Contextual Precision thấp** → context lẫn nhiễu: siết rerank, giảm `top_k`, hoặc pre-filter theo metadata.")
    lines.append("- **Faithfulness thấp** → câu trả lời vượt chứng cứ: siết prompt grounding / abstention.")
    lines.append("- **Answer Relevancy thấp** → câu trả lời lan man: rút gọn, bám sát câu hỏi (giảm số câu trích).")
    lines.append("\n_(Báo cáo tự sinh bởi `eval_pipeline.py`. Thành viên 4 bổ sung phân tích sâu + sơ đồ kiến trúc.)_")

    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Viết báo cáo: {RESULTS_MD}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="giới hạn số câu hỏi (chạy nhanh)")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_COMPARE,
                        help=f"danh sách config để so sánh (có: {list(RETRIEVAL_CONFIGS)})")
    args = parser.parse_args()

    dataset = load_golden_dataset()
    if args.limit:
        dataset = dataset[: args.limit]
    print(f"Loaded {len(dataset)} test cases | Configs: {args.configs} | Judge: {JUDGE_MODEL}")

    judge = get_judge_model()
    results = [evaluate_config(cfg, dataset, judge) for cfg in args.configs]

    # Lưu kết quả thô (cho Thành viên 4 phân tích).
    RESULTS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Lưu kết quả thô: {RESULTS_JSON}")

    write_results_md(results, len(dataset))

    print("\n=== TÓM TẮT ===")
    for r in results:
        a = r["aggregate"]
        print(f"  {r['config_name']:20} overall={a['overall']:.3f} | " +
              " ".join(f"{m[:4]}={a[m]:.2f}" for m in METRIC_NAMES))


if __name__ == "__main__":
    main()
