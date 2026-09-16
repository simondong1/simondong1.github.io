"""Check that rewards use final answers, official DAPO format, and correct labels."""
from math_rewards import score


def main():
    cases = [
        ("dapo", "</think>Answer: 42", "42", 1),
        ("dapo", "</think>Answer: 41", "42", 0),
        ("dapo", "<think>Answer: 42</think>Answer: 41", "42", 0),
        ("dapo", "<think>Answer: 42", "42", 0),
        ("dapo", "Answer: 42", "42", 0),
        ("dapo", r"</think>\boxed{42}", "42", 1),
        ("dapo", r"</think>\boxed{42}. Answer: 41", "42", 0),
        ("dapo", "</think>Answer: 41\nAnswer: 42", "42", 1),
        ("dapo", "</think>Answer: 42\nAnswer: ", "42", 0),
        ("dapo", "</think>Answer: -1", "-1", 1),
        ("dapo", "</think>Answer: 1,024", "1024", 1),
        ("dapo", r"</think>Answer: $\boxed{42}$", "42", 1),
        ("dapo", "</think>42 appeared in the calculation", "42", 0),
        ("gsm8k", "</think>#### 1,024", "1024", 1),
        ("gsm8k", "</think>She has 42 apples.", "42", 1),
        ("gsm8k", r"</think>\boxed{42}", "42", 1),
        ("gsm8k", "</think>#### 41", "42", 0),
        ("gsm8k", "<think>#### 42", "42", 0),
        ("gsm8k", "<think>#### 42</think>I cannot answer.", "42", 0),
        ("gsm8k", "</think>#### 1/2", "0.5", 1),
        ("gsm8k", "</think>#### 1/0", "42", 0),
    ]
    for task, response, answer, expected in cases:
        actual = score(response, {"task": task, "answer": answer})
        assert actual == expected, (task, response, actual, expected)
    assert score("Answer: 42", {"task": "dapo", "answer": "42"}, require_thinking_end=False) == 1
    print(f"Passed {len(cases) + 1} math reward cases")


if __name__ == "__main__":
    main()
