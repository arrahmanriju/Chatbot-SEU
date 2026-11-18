"""Export real answers for review and score completed human annotations."""
import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter, sleep

from evaluation.run import pipelines
from src.chain import run_chatbot
from src.service import ChatService


def export(output, model, lexical_only=False, pause=0, selected=None):
    corpus = Path(__file__).with_name("benchmark.json")
    cases = json.loads(corpus.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    completed = {(row["pipeline"], row["question"]) for row in rows}
    for name, (search, rewrite) in pipelines(lexical_only, model).items():
        if selected and name not in selected:
            continue
        for case in cases:
            if (name, case["question"]) in completed:
                continue
            evidence = []

            class RecordingRetriever:
                def search(self, question, limit):
                    matches = search(question, limit)
                    evidence.extend({"url": m.document.metadata.get("source_url"),
                                     "text": m.document.page_content[:2000]} for m in matches)
                    return matches

            service = ChatService(RecordingRetriever())
            service.rewrite = rewrite or (lambda question, history: question)
            history = case.get("history", [])
            start = perf_counter()
            if name == "original_dense":
                def retrieve(question, k):
                    matches = RecordingRetriever().search(question, k)
                    return [(match.document, match.dense_score) for match in matches]

                from types import SimpleNamespace
                answer = run_chatbot(case["question"], SimpleNamespace(
                    similarity_search_with_relevance_scores=retrieve), history)
                route, citations = None, []
            else:
                reply = service.chat(case["question"], history)
                answer, route, citations = reply.answer, reply.route, reply.citations
            rows.append({
                "pipeline": name, "question": case["question"], "history": history,
                "expected_route": case["route"], "expected_sources": case["sources"],
                "answer": answer, "route": route, "evidence": evidence, "citations": citations,
                "latency_ms": (perf_counter() - start) * 1000,
                "relevant": None, "factual_claims": None, "supported_claims": None,
            })
            checkpoint = output.with_suffix(output.suffix + ".tmp")
            checkpoint.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
            checkpoint.replace(output)
            if pause:
                sleep(pause)


def score(path, selected=None):
    rows = json.loads(path.read_text(encoding="utf-8"))
    for pipeline in sorted({row["pipeline"] for row in rows}):
        if selected and pipeline not in selected:
            continue
        group = [row for row in rows if row["pipeline"] == pipeline]
        for row in group:
            claims, supported = row["factual_claims"], row["supported_claims"]
            if type(row["relevant"]) is not bool or type(claims) is not int or type(supported) is not int:
                raise ValueError("Complete all human review fields before scoring.")
            if not 0 <= supported <= claims:
                raise ValueError("Supported claims must be between zero and total factual claims.")
        total = sum(row["factual_claims"] for row in group)
        supported = sum(row["supported_claims"] for row in group)
        print(json.dumps({
            "pipeline": pipeline, "answers": len(group),
            "answer_relevance": mean(row["relevant"] for row in group),
            "claim_faithfulness": supported / total if total else None,
            "hallucination_rate": (total - supported) / total if total else None,
            "mean_latency_ms": mean(row["latency_ms"] for row in group),
        }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["export", "score"])
    parser.add_argument("file", type=Path)
    parser.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--lexical-only", action="store_true")
    parser.add_argument("--pipeline", action="append", help="Run only the named pipeline(s).")
    parser.add_argument("--pause", type=float, default=0, help="Seconds between answer requests.")
    args = parser.parse_args()
    if args.pause < 0:
        parser.error("--pause cannot be negative")
    if args.mode == "export":
        export(args.file, args.reranker, args.lexical_only, args.pause, args.pipeline)
    else:
        score(args.file, args.pipeline)
