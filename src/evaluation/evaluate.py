import json
import re
import sys
from collections import defaultdict

sys.path.insert(0, ".")

def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_f1(prediction, gold):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return int(pred_tokens == gold_tokens)

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def exact_match(prediction, gold):
    return normalize_answer(prediction) == normalize_answer(gold)

def evaluate_pipeline(pipeline, test_file, output_file):
    with open(test_file) as f:
        test_data = json.load(f)

    results_by_type = defaultdict(lambda: {"correct": 0, "total": 0, "f1_sum": 0})
    errors = []
    all_results = []

    print(f"Evaluating {len(test_data)} questions...")

    for i, qa in enumerate(test_data):
        print(f"  [{i+1}/{len(test_data)}] {qa['question'][:60]}...")
        result = pipeline.answer(qa["question"])

        predicted = result.get("natural_answer", result.get("answer", ""))
        gold = qa["answer"]

        em = exact_match(predicted, gold)
        f1 = compute_f1(predicted, gold)

        qtype = qa["type"]
        results_by_type[qtype]["total"] += 1
        results_by_type[qtype]["f1_sum"] += f1
        if em:
            results_by_type[qtype]["correct"] += 1

        record = {
            "question": qa["question"],
            "gold": gold,
            "predicted": predicted,
            "em": em,
            "f1": f1,
            "type": qtype,
            "confidence": result["confidence"],
            "source": result["source"]
        }
        all_results.append(record)

        if not em:
            errors.append(record)

    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    total_em = sum(r["correct"] for r in results_by_type.values())
    total_n = sum(r["total"] for r in results_by_type.values())
    total_f1 = sum(r["f1_sum"] for r in results_by_type.values())

    print(f"\nOverall: EM={total_em/total_n:.4f}, F1={total_f1/total_n:.4f} ({total_em}/{total_n})")

    print(f"\nBy Question Type:")
    for qtype, stats in sorted(results_by_type.items()):
        em_rate = stats["correct"] / stats["total"]
        f1_rate = stats["f1_sum"] / stats["total"]
        print(f"  {qtype:25s}: EM={em_rate:.4f}, F1={f1_rate:.4f} ({stats['correct']}/{stats['total']})")

    print(f"\nError Analysis ({len(errors)} errors):")
    error_sources = defaultdict(int)
    for e in errors:
        error_sources[e["source"]] += 1
    for source, count in sorted(error_sources.items(), key=lambda x: -x[1]):
        print(f"  Source '{source}': {count} errors")

    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "overall_em": total_em / total_n,
                "overall_f1": total_f1 / total_n,
                "by_type": {
                    k: {"em": v["correct"]/v["total"], "f1": v["f1_sum"]/v["total"]}
                    for k, v in results_by_type.items()
                }
            },
            "all_results": all_results,
            "errors": errors
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    from src.pipeline.kbqa_pipeline import ScholarlyKBQA, DEFAULT_CONFIG
    pipeline = ScholarlyKBQA(DEFAULT_CONFIG)
    evaluate_pipeline(
        pipeline,
        "data/golden_qa/test.json",
        "results/evaluation_results.json"
    )
