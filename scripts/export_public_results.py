"""Export aggregate experiment metrics without prompts, responses or API payloads."""
import json
from pathlib import Path

SOURCE = Path("results")
TARGET = Path("public_results/metrics.json")


def read(name):
    return json.loads((SOURCE / name).read_text())


def interval(metric):
    return [round(float(value), 6) for value in metric["ci95"]]


def gsm_row(name, steps, status):
    metric = read(f"{name}.metrics.json")
    n = metric["n"]
    return {
        "name": name,
        "training_steps": steps,
        "training_status": status,
        "correct": round(metric["accuracy"]["mean"] * n),
        "n": n,
        "accuracy": round(metric["accuracy"]["mean"], 6),
        "accuracy_ci95": interval(metric["accuracy"]),
        "invalid_answer_rate": round(metric["invalid_answer_rate"], 6),
        "mean_completion_tokens": round(metric["mean_completion_length"], 3),
    }


def tool_row(name, status):
    metric = read(f"{name}.tool-metrics.json")
    return {
        "name": name,
        "training_steps": metric["training_step"],
        "training_status": status,
        "correct": round(metric["exact_match"]["mean"] * metric["n"]),
        "n": metric["n"],
        "exact_match": round(metric["exact_match"]["mean"], 6),
        "exact_match_ci95": interval(metric["exact_match"]),
        "call_required_correct": round(
            metric["call_exact_match"]["mean"] * metric["call_count"]
        ),
        "call_required_n": metric["call_count"],
        "format_correct": round(
            metric["format_accuracy"]["mean"] * metric["n"]
        ),
        "no_call_abstentions": round(
            metric["no_call_abstention"]["mean"] * metric["no_call_count"]
        ),
        "no_call_n": metric["no_call_count"],
        "mean_completion_tokens": round(metric["mean_length"], 3),
        "truncated": metric["truncated_count"],
    }


def main():
    validation = read("reward_validation.json")
    pilot = read("summary.json")
    expanded = read("expanded_comparison.json")
    tool = read("tool_comparison.json")
    validation_metrics = {}
    for dimension in ("helpfulness", "correctness", "coherence"):
        value = validation["metrics"][dimension]
        validation_metrics[dimension] = {
            "pearson": round(value["pearson"], 6),
            "spearman": round(value["spearman"], 6),
            "mae_normalized": round(value["mae"], 6),
            "pairwise_accuracy": round(value["pairwise"]["mean"], 6),
            "pairwise_ci95": interval(value["pairwise"]),
            "pairwise_n": value["pairwise"]["n"],
        }
    tool_names = (
        ("tool-reference-step1300", "completed"),
        ("tool-jev-step1300", "completed_after_resume"),
        ("tool-jev-step850", "intermediate_checkpoint"),
        ("tool-jev-no-answer-step1250", "checkpoint_from_interrupted_run"),
    )
    tool_metrics = [read(f"{name}.tool-metrics.json") for name, _ in tool_names]
    if len({m["dataset_fingerprint"] for m in tool_metrics}) != 1:
        raise ValueError("ToolRL metrics are not from the same filtered dataset")
    out = {
        "schema_version": 1,
        "scope": "aggregate_metrics_only",
        "model": "Qwen3.5-4B text backbone",
        "training_seed": 42,
        "jev_returned_model": validation["models"],
        "reward_validation": {
            "dataset": "HelpSteer2 validation",
            "responses": validation["n"],
            "prompts": validation["prompts"],
            "metrics": validation_metrics,
        },
        "gsm8k_smoke": {
            "dataset": "GSM8K test first 16 examples",
            "evaluation_n": 16,
            "purpose": "integration_check_only",
            "models": [
                gsm_row("smoke-base", 0, "base_model"),
                gsm_row("smoke-reference", 2, "completed"),
                gsm_row("smoke-jev", 2, "completed"),
            ],
        },
        "gsm8k_pilot": {
            "dataset": "GSM8K test",
            "evaluation_n": 1319,
            "greedy_max_tokens": 512,
            "models": [
                gsm_row("base", 0, "base_model"),
                gsm_row("reference", 144, "completed"),
                gsm_row("jev", 144, "completed"),
            ],
            "paired_jev_minus_reference": {
                "difference": round(
                    pilot["paired_jev_minus_reference"]["mean"], 6
                ),
                "ci95": interval(pilot["paired_jev_minus_reference"]),
            },
        },
        "gsm8k_expanded": {
            "dataset": "GSM8K test",
            "evaluation_n": 1319,
            "greedy_max_tokens": 512,
            "models": [
                gsm_row("expanded-reference-seed42", 1602, "completed"),
                gsm_row(
                    "expanded-jev-seed42-step1300",
                    1300,
                    "checkpoint_from_interrupted_run",
                ),
            ],
            "paired_jev_minus_reference": {
                "difference": round(
                    expanded["paired_jev_minus_reference"]["mean"], 6
                ),
                "ci95": interval(expanded["paired_jev_minus_reference"]),
                "equal_training_steps": False,
            },
        },
        "toolrl": {
            "dataset": "ToolRL rlla_4k test",
            "source_test_rows": 80,
            "evaluated_rows": 79,
            "excluded_overlength": 1,
            "call_required_n": 71,
            "no_call_n": 8,
            "greedy_max_tokens": 1024,
            "official_bfcl": False,
            "tool_execution": False,
            "models": [tool_row(name, status) for name, status in tool_names],
            "paired_jev1300_minus_reference1300": {
                "difference": round(
                    tool["paired_jev_minus_reference"]["mean"], 6
                ),
                "ci95": interval(tool["paired_jev_minus_reference"]),
                "jev_only_correct": tool["paired_jev_minus_reference"][
                    "jev_only_correct"
                ],
                "reference_only_correct": tool["paired_jev_minus_reference"][
                    "reference_only_correct"
                ],
                "mcnemar_exact_p": tool["paired_jev_minus_reference"][
                    "mcnemar_exact_p"
                ],
            },
        },
        "source_files": [
            "results/reward_validation.json",
            "results/summary.json",
            "results/expanded_comparison.json",
            "results/tool_comparison.json",
            "results/*.metrics.json",
            "results/*.tool-metrics.json",
        ],
    }
    TARGET.parent.mkdir(exist_ok=True)
    TARGET.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
