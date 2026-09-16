"""Math-only rewards; thinking tokens never count as a final answer."""
import json
import re
from fractions import Fraction


def final_answer(response, require_thinking_end=True):
    if "</think>" in response:
        return response.rsplit("</think>", 1)[1].strip()
    return "" if require_thinking_end or "<think>" in response else response.strip()


def gsm8k_score(response, answer):
    candidates = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)", response)
    if not candidates:
        candidates = re.findall(r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*\}", response)
    if not candidates:
        candidates = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?", response[-300:])
    try:
        return float(bool(candidates) and Fraction(candidates[-1].replace(",", "")) == Fraction(str(answer).replace(",", "")))
    except (ValueError, ZeroDivisionError):
        return 0.0


def dapo_score(response, answer):
    from miles.rollout.rm_hub.math_dapo_utils import is_correct_minerva, last_boxed_only_string, remove_boxed

    # Respect the official dataset's requested Answer: format. An explicit
    # incorrect final answer must never be rescued by an earlier boxed value.
    if re.search(r"(?i)Answer\s*:", response):
        correct, _ = is_correct_minerva(response, str(answer), answer_pattern=r"(?i)Answer[^\S\n]*:[^\S\n]*([^\n]*)")
        return float(correct)
    boxed = last_boxed_only_string(response)
    if boxed is None:
        return 0.0
    correct, _ = is_correct_minerva("Answer: " + remove_boxed(boxed), str(answer))
    return float(correct)


def score(response, label, require_thinking_end=True):
    answer = final_answer(response, require_thinking_end=require_thinking_end)
    if not answer:
        return 0.0
    parsed = json.loads(label) if isinstance(label, str) else label
    if isinstance(parsed, dict) and parsed.get("task") == "gsm8k":
        return gsm8k_score(answer, parsed["answer"])
    if isinstance(parsed, dict) and parsed.get("task") == "dapo":
        return dapo_score(answer, parsed["answer"])
    from miles.rollout.rm_hub.deepscaler import _grade_boxed_solution
    truth = parsed["answer"] if isinstance(parsed, dict) else parsed
    return float(_grade_boxed_solution(answer, str(truth)))


async def reward(args, sample, **kwargs):
    if isinstance(sample, list):
        return [await reward(args, item, **kwargs) for item in sample]
    require_end = getattr(args, "apply_chat_template_kwargs", {}).get("enable_thinking", False)
    return score(sample.response, sample.label, require_thinking_end=require_end)
