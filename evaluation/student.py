"""Compare the archive-only and student-support pipelines on held-out queries."""
import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter, sleep

from src.hybrid import HybridRetriever
from src.loader import read_records
from src.service import ChatService
from src.store import DEFAULT_PERSIST_DIR, load_index
from src.student import DATA, classify_intents


def run(baseline, index, live_output=None):
    cases = [row for row in read_records(DATA / "evaluation_dataset.jsonl") if row["split"] == "test"]
    sources = {row["id"]: row["source_url"] for row in read_records(DATA / "verified_student_sources.jsonl")}
    output = []
    for name, path in (("archive", baseline), ("student", index)):
        retriever = HybridRetriever(load_index(path))
        for language in ("all", "en", "bn", "banglish"):
            selected = [row for row in cases if language == "all" or row["language"] == language]
            recalls, reciprocal, latencies, intents = [], [], [], []
            for row in selected:
                expected = {sources[key] for key in row["expected_source_ids"]}
                if name == "student":
                    predicted = classify_intents(row["question"])
                    intents.append(bool(predicted and predicted[0] == row["category"]))
                if not expected:
                    continue
                start = perf_counter()
                ranked = [match.document.metadata.get("source_url")
                          for match in retriever.search(row["question"], 5)]
                latencies.append((perf_counter() - start) * 1000)
                recalls.append(len(set(ranked) & expected) / len(expected))
                reciprocal.append(next((1 / position for position, url in enumerate(ranked, 1)
                                        if url in expected), 0))
            print(json.dumps({"pipeline": name, "language": language, "cases": len(selected),
                              "retrieval_cases": len(recalls), "recall_at_5": round(mean(recalls), 4),
                              "mrr_at_5": round(mean(reciprocal), 4),
                              "category_accuracy": round(mean(intents), 4) if intents else None,
                              "mean_retrieval_ms": round(mean(latencies), 2)}), flush=True)
        if live_output:
            class RecordingRetriever:
                def __init__(self):
                    self.evidence = []

                def search_program_fees(self, programs):
                    return retriever.search_program_fees(programs)

                def search(self, question, limit):
                    matches = retriever.search(question, limit)
                    self.evidence.extend(
                        {"url": item.document.metadata.get("source_url"), "text": item.document.page_content}
                        for item in matches
                    )
                    return matches

            for row in cases:
                if row["intent"] not in {"admission_contact", "duplicate_certificate"}:
                    continue
                recording = RecordingRetriever()
                service = ChatService(recording, student_support=name == "student")
                try:
                    result = service.chat(row["question"])
                except Exception as exc:
                    output.append({"pipeline": name, "case": row, "error": type(exc).__name__})
                else:
                    output.append({"pipeline": name, "case": row, "route": result.route, "answer": result.answer,
                                   "citations": result.citations, "evidence": recording.evidence,
                                   "latency_ms": result.elapsed_ms, "correct": None, "grounded": None})
                live_output.parent.mkdir(exist_ok=True, parents=True)
                live_output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
                sleep(3)


def score(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    for name in sorted({row["pipeline"] for row in rows}):
        group = [row for row in rows if row["pipeline"] == name and "answer" in row]
        if not group or any(type(row[key]) is not bool for row in group for key in ("correct", "grounded")):
            raise ValueError("Review the correct and grounded fields for every completed answer.")
        factual = [row for row in group if row["route"] == "university"]
        print(json.dumps({"pipeline": name, "completed": len(group),
                          "correctness": mean(row["correct"] for row in group),
                          "groundedness": mean(row["grounded"] for row in factual) if factual else None,
                          "unsupported_answer_rate": mean(not row["grounded"] for row in factual) if factual else None,
                          "mean_response_ms": round(mean(row["latency_ms"] for row in group), 2)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_PERSIST_DIR.parent / "student-baseline")
    parser.add_argument("--index", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--live-output", type=Path)
    parser.add_argument("--score", type=Path)
    args = parser.parse_args()
    if args.score:
        score(args.score)
    else:
        run(args.baseline, args.index, args.live_output)
