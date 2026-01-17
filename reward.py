import re
from typing import List, Optional, Union

# 常量定义
_SOLUTION_CLIP_CHARS = 300
_STRICT_ANSWER_PATTERN = r"#### (\-?[0-9\.\,]+)"
_FLEXIBLE_ANSWER_PATTERN = r"(\-?[0-9\.\,]+)"
_INVALID_ANSWERS = ["", "."]

# 奖励分数常量
REWARD_CORRECT = 1.0
REWARD_FORMAT_CORRECT = 0.3
REWARD_NO_ANSWER = -0.3
REWARD_WRONG = 0.0


def extract_solution(solution_str: str, method: str = "strict") -> Optional[str]:
    """
    从模型回答中提取最终答案

    Args:
        solution_str: 模型回答字符串
        method: 解析方法，可选 "strict" 或 "flexible"

    Returns:
        提取的答案字符串，如果无法提取则返回None

    Note:
        - 只匹配回答的最后300个字符以优化性能
        - 建议使用默认的strict模式以测试模型格式
    """
    if method not in ["strict", "flexible"]:
        raise ValueError(
            f"method must be 'strict' or 'flexible', got {method}")

    if not solution_str:
        return None

    # 优化：只在最后300个字符中匹配，避免在长字符串上慢速的正则匹配
    # 对于数学问题，最终答案通常在回答的末尾
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    try:
        if method == "strict":
            # 严格模式：匹配 "#### 数字" 格式，测试模型的格式化能力
            solutions = re.findall(_STRICT_ANSWER_PATTERN, solution_str)
            if not solutions:
                return None

            # 取最后一个匹配项，并清理格式
            final_answer = solutions[-1].replace(",", "").replace("$", "")

        elif method == "flexible":
            # 灵活模式：匹配任何数字
            answers = re.findall(_FLEXIBLE_ANSWER_PATTERN, solution_str)
            if not answers:
                return None

            # 从后往前找到第一个有效的数字
            final_answer = None
            for answer in reversed(answers):
                if answer not in _INVALID_ANSWERS:
                    final_answer = answer
                    break

        return final_answer

    except re.error as e:
        # 处理正则表达式错误
        print(f"正则表达式匹配错误: {e}")
        return None


def compute_score(
    solution_str: str,
    ground_truth: str,
    method: str = "strict",
    format_score: float = REWARD_FORMAT_CORRECT,
    score: float = REWARD_CORRECT
) -> float:
    """
    GSM8K评分函数 - 计算奖励分数

    Args:
        solution_str: 模型回答
        ground_truth: 真实答案
        method: 解析答案的方式，可选 'strict' 或 'flexible'
        format_score: 格式正确但答案错误的分数
        score: 答案正确的分数

    Returns:
        奖励分数

    Note:
        - 无法提取答案时返回 REWARD_NO_ANSWER (-0.3)
        - 答案正确时返回 score (默认 1.0)
        - 答案错误但格式正确时返回 format_score (默认 0.3)
    """
    if not solution_str or not ground_truth:
        return REWARD_NO_ANSWER

    answer = extract_solution(solution_str=solution_str, method=method)

    if answer is None:
        return REWARD_NO_ANSWER
    elif answer == ground_truth:
        return score
    else:
        return format_score


# 可选：添加辅助函数用于调试和测试
def validate_answer_format(answer: str) -> bool:
    """
    验证答案格式是否正确（用于调试）

    Args:
        answer: 要验证的答案字符串

    Returns:
        如果格式正确返回True，否则返回False
    """
    if not answer:
        return False

    # 检查是否包含数学答案的常见格式
    has_number = re.search(r"\d", answer) is not None
    has_format = "####" in answer

    return has_number or has_format


def trl_reward_fn(
    prompts: List[str],
    completions: List[str],
    solution: List[str],
    **kwargs
) -> List[float]:
    """
    TRL库自定义的奖励函数 - 注意所有字段均为列表输入

    Args:
        prompts: 数据的prompt输入（当前未使用，保留用于未来扩展）
        completions: 模型回答列表
        solution: 正确答案列表
        **kwargs: 其他参数（保留用于未来扩展）

    Returns:
        奖励分数列表

    Raises:
        ValueError: 当输入列表长度不一致时

    Example:
        >>> prompts = ["问题1", "问题2"]
        >>> completions = ["答案是#### 42", "答案是#### 24"]
        >>> solution = ["42", "24"]
        >>> rewards = trl_reward_fn(prompts, completions, solution)
        >>> print(rewards)  # [1.0, 1.0]
    """
    # 验证输入参数
    if not (len(completions) == len(solution)):
        raise ValueError(
            f"completions和solution列表长度必须相同，"
            f"当前长度: completions={len(completions)}, solution={len(solution)}"
        )

    if not completions:
        return []

    rewards = []

    try:
        for model_answer, gt in zip(completions, solution):
            # 使用严格模式计算分数
            score = compute_score(
                solution_str=model_answer,
                ground_truth=gt,
                method="strict",
                format_score=REWARD_FORMAT_CORRECT,
                score=REWARD_CORRECT
            )
            rewards.append(score)

    except Exception as e:
        # 如果计算过程中出现错误，为所有样本返回最低分数
        print(f"奖励计算错误: {e}")
        rewards = [REWARD_NO_ANSWER] * len(completions)

    return rewards


# 添加测试函数
if __name__ == "__main__":
    # 测试用例
    test_cases = [
        ("答案是#### 42", "42"),
        ("计算结果是#### 3.14", "3.14"),
        ("最终答案：24", "24"),
        ("没有答案", "42"),
        ("", "42"),
    ]

    print("测试 extract_solution 函数:")
    for answer, expected in test_cases:
        extracted = extract_solution(answer)
        print(f"输入: '{answer}' -> 提取: {extracted}, 期望: {expected}")

    print("\n测试 compute_score 函数:")
    for answer, expected in test_cases:
        score = compute_score(answer, expected)
        print(f"输入: '{answer}' -> 分数: {score}, 期望答案: {expected}")

    print("\n测试 trl_reward_fn 函数:")
    prompts = ["问题1", "问题2", "问题3"]
    completions = ["答案是#### 42", "结果是#### 3.14", "最终答案：24"]
    solutions = ["42", "3.14", "24"]
    rewards = trl_reward_fn(prompts, completions, solutions)
    print(f"奖励分数: {rewards}")
