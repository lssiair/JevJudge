"""ToolRL-style deterministic rewards; calls are parsed, never executed.

Format: 0/1. Correctness: -3..3 using tool-name multiset IoU plus
parameter-name IoU and exact parameter-value matches. As described by
GDPO appendix C, repeated tool names use optimal one-to-one assignment.
No-call references receive correctness 0, following ToolRL's recipe.
"""
import json
import re
from collections import Counter
from scipy.optimize import linear_sum_assignment
from src.rewards.reference_reward import completion_text

VERSION = "toolrl-gdpo-optimal-v1"

def parse_calls(text):
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S)
    if text.count("<tool_call>") != len(blocks) or text.count("</tool_call>") != len(blocks):
        raise ValueError("Unclosed tool call")
    calls = []
    for block in blocks:
        if not block.strip():
            raise ValueError("Empty tool-call block")
        for line in block.strip().split("\n"):
            if not line.strip():
                continue
            call = json.loads(line)
            if not isinstance(call, dict) or set(call) != {"name", "parameters"}:
                raise ValueError("Expected tool name and parameters")
            if not isinstance(call["name"], str) or not isinstance(call["parameters"], dict):
                raise ValueError("Invalid tool-call schema")
            calls.append(call)
    return calls

def iou(left, right):
    a, b = Counter(left), Counter(right)
    intersect = sum((a & b).values())
    union = sum((a | b).values())
    return intersect / union if union else 1.0

def format_score(text, reference):
    pattern = r"<think>.*?</think>"
    for tag in ("tool_call", "response"):
        expected = int(f"<{tag}>" in reference)
        if text.count(f"<{tag}>") != expected or text.count(f"</{tag}>") != expected:
            return 0.0
        if expected:
            pattern += rf"\s*<{tag}>.*?</{tag}>"
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return 0.0
    return float(re.fullmatch(pattern, text.strip(), re.S) is not None)

def tool_scores(completion, reference):
    text = completion_text(completion).strip()
    truth = parse_calls(reference)  # Invalid labels must fail, never become rewards.
    fmt = format_score(text, reference)
    try:
        predicted = parse_calls(text)
    except (ValueError, TypeError):
        return dict(format_reward=fmt, correctness_reward=-3.0 if truth else 0.0,
                    exact_match=0.0, reward=fmt-3.0 if truth else fmt)
    canonical = lambda calls: Counter(json.dumps(c, sort_keys=True, ensure_ascii=False) for c in calls)
    exact = float(canonical(truth) == canonical(predicted) and (bool(truth) or fmt == 1))
    if not truth:
        correctness = 0.0
    else:
        score = iou([c["name"] for c in truth], [c["name"] for c in predicted])
        # Dummy columns permit unmatched labels; assignment prevents reusing a prediction.
        import numpy as np
        matrix = np.zeros((len(truth), max(len(truth), len(predicted))))
        for i, gt in enumerate(truth):
            for j, candidate in enumerate(predicted):
                if gt["name"] == candidate["name"]:
                    a, b = gt["parameters"], candidate["parameters"]
                    matrix[i,j] = iou(a.keys(), b.keys()) + sum(k in b and b[k] == v for k,v in a.items())
        row, col = linear_sum_assignment(-matrix)
        score += float(matrix[row,col].sum())
        maximum = 1 + sum(1 + len(c["parameters"]) for c in truth)
        correctness = 6 * score / maximum - 3
    return dict(format_reward=fmt, correctness_reward=correctness, exact_match=exact,
                reward=fmt+correctness)

def tool_reference_reward(prompts, completions, reference, **kwargs):
    if len(completions) != len(reference):
        raise ValueError("Completion/reference count mismatch")
    return [tool_scores(c,r)["reward"] for c,r in zip(completions,reference)]
