"""Deterministic binary verifiers; no generated code is executed."""
import ast
import json
import re
from collections import Counter
from fractions import Fraction


def gsm8k_score(response, answer):
    candidates = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)", response)
    if not candidates:
        candidates = re.findall(r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*\}", response)
    if not candidates:
        # Like VERL's flexible GSM8K extractor, accept an unmarked final number.
        # Correct solutions should not need to learn an answer delimiter first.
        candidates = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?", response[-300:])
    try:
        return float(bool(candidates) and Fraction(candidates[-1].replace(",", "")) == Fraction(str(answer).replace(",", "")))
    except (ValueError, ZeroDivisionError):
        return 0.0


def countdown_score(response, numbers, target):
    response = response.replace("×", "*").replace("÷", "/")
    matches = re.findall(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    if matches:
        expression = matches[-1].strip()
    else:
        boxed = re.findall(r"\\boxed\{([^{}]*)\}", response)
        if boxed:
            expression = boxed[-1].strip()
        else:
            cleaned = re.sub(r"<\|[^>]+\|>", "", response).replace("×", "*").replace("÷", "/")
            # Newlines delimit attempts: merging consecutive equation lines
            # incorrectly rejects an otherwise valid final expression.
            candidates = re.findall(r"[\d() \t+*/.=\-]+", cleaned)
            candidates = [candidate.strip().rstrip(".") for candidate in candidates if re.search(r"\d", candidate)]
            if not candidates:
                return 0.0
            expression = candidates[-1]
    if "=" in expression:
        parts = expression.split("=")
        if len(parts) != 2:
            return 0.0
        expression, claimed_result = (part.strip() for part in parts)
        try:
            if Fraction(claimed_result) != Fraction(target):
                return 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0
    if len(expression) > 200 or not re.fullmatch(r"[\d\s+*/().-]+", expression):
        return 0.0
    used = []

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            used.append(node.value)
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError("Unsupported expression")

    try:
        value = visit(ast.parse(expression, mode="eval").body)
        return float(Counter(used) == Counter(numbers) and value == Fraction(target))
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError, OverflowError):
        return 0.0


def score(response, label):
    label = json.loads(label) if isinstance(label, str) else label
    if label["task"] == "gsm8k":
        return gsm8k_score(response, label["answer"])
    if label["task"] == "countdown":
        return countdown_score(response, label["numbers"], label["target"])
    raise ValueError(f"Unknown task: {label['task']}")


async def reward(args, sample, **kwargs):
    if isinstance(sample, list):
        return [score(item.response, item.label) for item in sample]
    return score(sample.response, sample.label)
