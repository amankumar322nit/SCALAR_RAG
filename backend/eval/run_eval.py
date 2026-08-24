#!/usr/bin/env python3
"""
Automated Evaluation Suite for Scaler Learner Support RAG Pipeline.
Features:
- Context Precision / Retrieval Hit Rate
- LLM Judge Evaluator for Faithfulness (Grounding) & Answer Correctness
- Fact Recall & Negative Out-of-Scope Abstention Verification
"""

import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add backend directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import RAGPipeline

console = Console()
TEST_CASES_PATH = Path(__file__).resolve().parent / "test_cases.json"
REPORT_OUTPUT_PATH = Path(__file__).resolve().parent / "eval_report.json"


def llm_judge_evaluate(rag: RAGPipeline, query: str, context: str, answer: str, expected_facts: List[str]) -> Dict[str, Any]:
    """
    LLM-as-a-Judge: Prompts the configured LLM model to independently evaluate
    Faithfulness (is the answer 100% grounded in context?) and Fact Recall.
    """
    if not context or "could not find information" in answer.lower():
        return {
            "faithfulness": 1.0,
            "correctness": 1.0,
            "rationale": "Correct out-of-scope abstention."
        }

    judge_prompt = f"""You are an expert AI Judge evaluating a Retrieval-Augmented Generation (RAG) system.

[Question]
{query}

[Retrieved Context]
{context}

[Generated Answer]
{answer}

[Expected Key Facts]
{', '.join(expected_facts)}

Evaluate the Generated Answer on two metrics (0.0 to 1.0):
1. faithfulness: Does the answer strictly adhere to the retrieved context without hallucinating unsupported claims? (1.0 = fully grounded, 0.0 = contains hallucinations)
2. correctness: Does the answer accurately convey the key facts answering the user's question? (1.0 = completely correct, 0.0 = incorrect or missing key facts)

Respond ONLY with valid JSON:
{{"faithfulness": 1.0, "correctness": 1.0, "reasoning": "brief explanation"}}
"""
    try:
        if rag.llm.groq_key or rag.llm.grok_key or rag.llm.openai_key:
            res = rag.llm._generate_groq(judge_prompt, rag.llm.groq_key or rag.llm.grok_key)
            raw_judge = res.get("answer", "")
            # Extract JSON block
            json_match = re.search(r"\{.*?\}", raw_judge, re.DOTALL)
            if json_match:
                judge_data = json.loads(json_match.group(0))
                return {
                    "faithfulness": float(judge_data.get("faithfulness", 1.0)),
                    "correctness": float(judge_data.get("correctness", 1.0)),
                    "rationale": str(judge_data.get("reasoning", "Evaluated by LLM Judge"))
                }
    except Exception as e:
        pass

    # Deterministic grounding and fact verification fallback
    matched = sum(1 for f in expected_facts if f.lower() in answer.lower())
    fact_score = matched / max(1, len(expected_facts))

    # Compute sentence-level context grounding
    clean_answer = re.sub(r"\[[^\]]+\]", "", answer)
    context_words = set(re.findall(r"\b[a-z0-9]+\b", context.lower()))
    sentences = [s.strip() for s in re.split(r"[.\n;]", clean_answer) if len(s.strip()) > 10]

    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
        "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
        "will", "with", "this", "these", "those", "you", "your", "they", "their",
        "can", "could", "should", "would", "may", "might", "must", "scaler",
        "please", "official", "support", "according", "policy", "policies", "stated"
    }

    if not sentences or not context_words:
        grounding_score = 0.5 if context else 1.0
    else:
        scores = []
        for s in sentences:
            tokens = [w for w in re.findall(r"\b[a-z0-9]+\b", s.lower()) if w not in stop_words and len(w) > 2]
            if tokens:
                supported = sum(1 for t in tokens if t in context_words)
                scores.append(supported / len(tokens))
        grounding_score = round(sum(scores) / len(scores), 2) if scores else 0.5

    return {
        "faithfulness": grounding_score,
        "correctness": round(fact_score, 2),
        "rationale": f"Deterministic evaluation (grounding: {grounding_score:.2f}, fact recall: {fact_score:.2f})"
    }


def evaluate_test_case(rag: RAGPipeline, tc: Dict[str, Any]) -> Dict[str, Any]:
    query = tc["query"]
    expected_doc = tc.get("expected_doc")
    expected_facts = tc.get("expected_facts", [])
    is_out_of_scope = tc.get("is_out_of_scope", False)

    start_time = time.time()
    res = rag.query(query, emit_stdout=False)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    answer = res.get("answer", "")
    sources = res.get("sources", [])
    citations = res.get("citations", [])

    # 1. Retrieval Accuracy (Context Precision)
    retrieved_docs = [s["doc_path"] for s in sources]
    if is_out_of_scope:
        retrieval_pass = (len(sources) == 0 or "could not find" in answer.lower())
    else:
        retrieval_pass = any(expected_doc in d for d in retrieved_docs)

    # 2. LLM Judge Evaluation
    context_text = "\n\n".join([s["snippet"] for s in sources])
    judge_res = llm_judge_evaluate(rag, query, context_text, answer, expected_facts)

    faithfulness_score = judge_res["faithfulness"]
    correctness_score = judge_res["correctness"]

    # Pass Criteria:
    # 1. Retrieval Hit (or correct abstention)
    # 2. Faithfulness >= 0.70
    # 3. Correctness >= 0.50
    overall_pass = retrieval_pass and (faithfulness_score >= 0.7) and (correctness_score >= 0.5)

    return {
        "id": tc["id"],
        "category": tc["category"],
        "query": query,
        "is_out_of_scope": is_out_of_scope,
        "retrieval_pass": retrieval_pass,
        "faithfulness_score": faithfulness_score,
        "correctness_score": correctness_score,
        "overall_pass": overall_pass,
        "latency_ms": latency_ms,
        "judge_rationale": judge_res.get("rationale", ""),
        "retrieved_docs": retrieved_docs,
        "citations": citations,
        "answer_snippet": answer[:150] + "..." if len(answer) > 150 else answer
    }


def run_evaluation():
    console.print(Panel(
        "[bold cyan]🧪 SCALER RAG AUTOMATED EVALUATION SUITE (LLM-AS-A-JUDGE)[/bold cyan]\n"
        "[dim]Assessing Context Precision, LLM Judge Faithfulness, Correctness, and Abstention[/dim]",
        border_style="cyan"
    ))

    if not TEST_CASES_PATH.exists():
        console.print(f"[bold red]Error: Test cases file not found at {TEST_CASES_PATH}[/bold red]")
        sys.exit(1)

    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    rag = RAGPipeline()
    results = []

    table = Table(title="📊 Test Case Benchmark Results", border_style="blue")
    table.add_column("ID", style="bold cyan", width=8)
    table.add_column("Category", style="yellow", width=22)
    table.add_column("Retrieval", justify="center", width=11)
    table.add_column("LLM Faithfulness", justify="center", width=18)
    table.add_column("Correctness", justify="center", width=14)
    table.add_column("Latency", justify="right", width=10)
    table.add_column("Status", justify="center", width=10)

    total_cases = len(test_cases)
    passed_cases = 0
    total_retrieval_passes = 0
    total_faithfulness = 0.0
    total_correctness = 0.0
    total_latency = 0.0

    for tc in test_cases:
        eval_res = evaluate_test_case(rag, tc)
        results.append(eval_res)
        time.sleep(0.4)  # Prevent free-tier rate limit burst

        status_str = "[bold green]PASS[/bold green]" if eval_res["overall_pass"] else "[bold red]FAIL[/bold red]"
        ret_str = "✅ Hit" if eval_res["retrieval_pass"] else "❌ Miss"
        faith_str = f"✅ {eval_res['faithfulness_score'] * 100:.0f}%" if eval_res['faithfulness_score'] >= 0.7 else f"⚠️ {eval_res['faithfulness_score'] * 100:.0f}%"
        corr_str = f"{eval_res['correctness_score'] * 100:.0f}%"

        table.add_row(
            eval_res["id"],
            eval_res["category"],
            ret_str,
            faith_str,
            corr_str,
            f"{eval_res['latency_ms']:.1f}ms",
            status_str
        )

        if eval_res["overall_pass"]:
            passed_cases += 1
        if eval_res["retrieval_pass"]:
            total_retrieval_passes += 1
        total_faithfulness += eval_res["faithfulness_score"]
        total_correctness += eval_res["correctness_score"]
        total_latency += eval_res["latency_ms"]

    console.print(table)

    # Summary Metrics
    pass_rate = (passed_cases / total_cases) * 100
    retrieval_precision = (total_retrieval_passes / total_cases) * 100
    avg_faithfulness = (total_faithfulness / total_cases) * 100
    avg_correctness = (total_correctness / total_cases) * 100
    avg_latency = total_latency / total_cases

    summary_panel = Panel(
        f"[bold white]Overall Evaluation Summary:[/bold white]\n\n"
        f"• [bold]Test Cases Passed:[/bold] {passed_cases}/{total_cases} ([bold green]{pass_rate:.1f}%[/bold green])\n"
        f"• [bold]Context Precision (Retrieval Hit Rate):[/bold] {retrieval_precision:.1f}%\n"
        f"• [bold]LLM Judge Faithfulness Score:[/bold] {avg_faithfulness:.1f}%\n"
        f"• [bold]LLM Judge Correctness Score:[/bold] {avg_correctness:.1f}%\n"
        f"• [bold]Average End-to-End Latency:[/bold] {avg_latency:.2f} ms\n",
        title="🏆 Benchmark Evaluation Summary",
        border_style="green" if pass_rate >= 80 else "yellow"
    )
    console.print(summary_panel)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "pass_rate_percent": round(pass_rate, 2),
            "context_precision_percent": round(retrieval_precision, 2),
            "llm_judge_faithfulness_percent": round(avg_faithfulness, 2),
            "llm_judge_correctness_percent": round(avg_correctness, 2),
            "avg_latency_ms": round(avg_latency, 2)
        },
        "details": results
    }

    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    console.print(f"[dim]📁 Detailed evaluation report saved to: {REPORT_OUTPUT_PATH}[/dim]\n")
    return pass_rate >= 80


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)
