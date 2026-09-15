"""Deterministic reward functions; never execute model-produced code."""
import ast
from collections import Counter
from fractions import Fraction
import json
import re


def final_box(text):
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def normalize_arithmetic(expression):
    """Translate a small, explicit LaTeX arithmetic subset; never evaluate TeX."""
    expression = expression.replace(r"\left", "").replace(r"\right", "")
    for source, target in ((r"\times", "*"), (r"\cdot", "*"), (r"\div", "/")):
        expression = expression.replace(source, target)

    def braced(start):
        while start < len(expression) and expression[start].isspace():
            start += 1
        if start == len(expression) or expression[start] != "{":
            raise ValueError("Fraction arguments must be braced")
        depth = 1
        for end in range(start + 1, len(expression)):
            depth += (expression[end] == "{") - (expression[end] == "}")
            if depth == 0:
                return expression[start+1:end], end+1
        raise ValueError("Unbalanced fraction")

    match = re.search(r"\\(?:dfrac|tfrac|frac)", expression)
    if match:
        numerator, middle = braced(match.end())
        denominator, end = braced(middle)
        replacement = "((" + normalize_arithmetic(numerator) + ")/(" + normalize_arithmetic(denominator) + "))"
        return normalize_arithmetic(expression[:match.start()] + replacement + expression[end:])
    return expression


def countdown_score(text, nums, target):
    expression = final_box(text)
    if not expression or len(expression) > 200:
        return 0.0
    try:
        expression = normalize_arithmetic(expression)
        if not re.fullmatch(r"[0-9+*/()\s-]+", expression):
            return 0.0
        root = ast.parse(expression, mode="eval")
        used = []

        def evaluate(node):
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and type(node.value) is int:
                used.append(node.value)
                return Fraction(node.value)
            if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
                value = evaluate(node.operand)
                return -value if isinstance(node.op, ast.USub) else value
            if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
                a, b = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add): return a + b
                if isinstance(node.op, ast.Sub): return a - b
                if isinstance(node.op, ast.Mult): return a * b
                return a / b
            raise ValueError("Only literal numbers and + - * / are allowed")

        value = evaluate(root)
        return float(Counter(used) == Counter(nums) and value == target)
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError, OverflowError):
        return 0.0


def integer_score(text, answer):
    value = final_box(text)
    if value is None:
        return 0.0
    value = value.replace(r"\,", "").replace(",", "").strip()
    if not re.fullmatch(r"[+-]?\d+", value):
        return 0.0
    return float(int(value) == int(answer))


async def reward(args, sample, **kwargs):
    label = json.loads(sample.label) if isinstance(sample.label, str) else sample.label
    if label["task"] == "countdown":
        return countdown_score(sample.response, label["nums"], label["target"])
    if label["task"] == "deepmath":
        return integer_score(sample.response, label["answer"])
    raise ValueError(label["task"])
