"""Exact numeric outcome reward for GSM8K and DAPO; no model-generated code executes."""
from decimal import Decimal, InvalidOperation
import json
import re


def final_answer(text):
    # Miles preserves tokenizer EOS markers in sample.response.
    text = re.sub(r'(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>)\s*$', '', text)
    candidates = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        for end in range(match.end(), len(text)):
            depth += (text[end] == "{") - (text[end] == "}")
            if depth == 0:
                candidates.append((match.start(), text[match.end():end]))
                break
    for match in re.finditer(r"(?im)^\s*(?:\*\*)?(?:final\s+)?answer\s*:\s*(?:\*\*)?\s*(.+)$", text):
        candidates.append((match.start(), match.group(1)))
    if not candidates:
        return None
    value = max(candidates)[1].strip().strip("*$ ").replace(r"\,", "").replace(r"\!", "")
    # A boxed answer on the Answer line is handled by the later box position.
    value = value.removesuffix(".").strip()
    if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", value):
        return None
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None


def score(text, answer):
    value = final_answer(text)
    return float(value is not None and value == Decimal(str(answer).replace(",", "")))


async def reward(args, sample, **kwargs):
    label = json.loads(sample.label) if isinstance(sample.label, str) else sample.label
    return score(sample.response, label["answer"])
