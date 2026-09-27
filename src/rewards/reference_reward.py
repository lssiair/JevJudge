"""Independent numerical final-answer evaluator. Never execute model output."""
import re
from fractions import Fraction

NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+"

def completion_text(completion):
    if isinstance(completion, str):
        return completion
    return "\n".join(x["content"] for x in completion if x.get("role") == "assistant")

def numeric(value):
    value = value.strip().replace(",", "").replace("$", "").replace("−", "-")
    value = re.sub(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", value)
    value = value.strip().rstrip(".").strip()
    try:
        if value.endswith("%"):
            return Fraction(value[:-1].strip()) / 100
        if "/" in value:
            a, b = value.split("/")
            return Fraction(a.strip()) / Fraction(b.strip())
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None

def extract_answer(text):
    text = completion_text(text)
    # Remove analysis so truncated thinking cannot count as a final answer.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return None
    text = text.replace("−", "-")
    boxes = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    if boxes:
        return numeric(boxes[-1])
    if "####" in text:
        text = text.rsplit("####", 1)[1].strip()
    else:
        finals = re.findall(r"(?:final answer|the answer is|answer:)\s*[:=]?\s*(.*)", text, re.I)
        text = finals[-1] if finals else text.strip().split("\n")[-1]
    # Only unambiguous single numeric values on the final line are accepted.
    values = re.findall(r"(?:" + NUMBER + r")\s*/\s*(?:" + NUMBER + r")|(?:" + NUMBER + r")\s*%?", text)
    if len(values) != 1:
        return None
    return numeric(values[0])

def reference_reward(prompts, completions, reference, **kwargs):
    if len(completions) != len(reference):
        raise ValueError("Completion/reference count mismatch")
    return [float((a := extract_answer(c)) is not None and a == numeric(str(r)))
            for c, r in zip(completions, reference)]
