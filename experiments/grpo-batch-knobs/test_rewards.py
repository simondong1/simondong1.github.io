from study_rewards import countdown_score, gsm8k_score


def test_numeric_answers_and_format():
    assert gsm8k_score("We get 1234.\n#### 1,234", "1234") == 1
    assert gsm8k_score(r"Final: \boxed{3.5}", "3.5") == 1
    assert gsm8k_score("#### 7/2", "3.5") == 1
    assert gsm8k_score("#### 4\nCorrection: #### 5", "4") == 0
    assert gsm8k_score("Therefore the final answer is 10 weeks.", "10") == 1
    assert gsm8k_score("The answer is unknown.", "1234") == 0
    assert gsm8k_score("#### 7/0", "0") == 0


def test_countdown_enforces_multiset_and_exact_arithmetic():
    assert countdown_score("<answer>(3+3)*4</answer>", [3, 3, 4], 24) == 1
    assert countdown_score("<answer>8/(3-8/3)</answer>", [8, 3, 8, 3], 24) == 1
    assert countdown_score("<answer>24</answer>", [3, 3, 4], 24) == 0
    assert countdown_score("<answer>3*8</answer>", [3, 3, 4], 24) == 0
    assert countdown_score("<answer>3*3*3</answer>", [3, 3, 4], 27) == 0
    assert countdown_score("<answer>2**3</answer>", [2, 3], 8) == 0
    assert countdown_score("<answer>8//3</answer>", [8, 3], 2) == 0
    assert countdown_score("<answer>4/(3-3)</answer>", [3, 3, 4], 0) == 0
    assert countdown_score("<answer>__import__('os').system('id')</answer>", [3, 3, 4], 24) == 0


def test_countdown_accepts_correct_untagged_equations_but_checks_both_sides():
    assert countdown_score("(98 - 87) * 2 = 22<|im_end|>", [87, 2, 98], 22) == 1
    assert countdown_score("(3 + 3) * 4", [3, 3, 4], 24) == 1
    assert countdown_score(r"\boxed{(3 + 3) * 4}", [3, 3, 4], 24) == 1
    assert countdown_score("(85 - 44) * 2 + 57 = 59", [44, 57, 85, 2], 59) == 0
    assert countdown_score("<answer>(3+3)*4 = 25</answer>", [3, 3, 4], 24) == 0
    assert countdown_score("(3+3)*4 = 24\nFinal answer: 25", [3, 3, 4], 24) == 0
    assert countdown_score("3+3 = 6\n(3+3)*4 = 24", [3, 3, 4], 24) == 1
    assert countdown_score("<answer>(3+3)×4</answer>", [3, 3, 4], 24) == 1
