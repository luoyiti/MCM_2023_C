import math
from collections import Counter
from wordle_game_simulate import get_feedback

# 若你的反馈是字符串形式，可用这个映射
COLOR2INT = {
    "gray": 0,
    "yellow": 1,
    "green": 2
}

def normalize_feedback(feedback):
    """
    将反馈统一成 tuple[int] 形式，如 (0,2,1,0,0)
    """
    if isinstance(feedback, dict):
        # 如果反馈是字典形式 {位置: 'green'/'yellow'/'gray'}
        # 转换为按位置排序的元组
        feedback_list = [None] * len(feedback)
        for pos, color in feedback.items():
            feedback_list[pos] = COLOR2INT[color]
        return tuple(feedback_list)
    elif isinstance(feedback, (list, tuple)) and len(feedback) > 0 and isinstance(feedback[0], str):
        # 如果反馈是列表形式 ['gray', 'green', ...]
        return tuple(COLOR2INT[c] for c in feedback)
    else:
        # 如果已经是数字形式
        return tuple(feedback)


def feedback_entropy(guess, possible_words, get_feedback):
    """
    计算在当前候选答案集合 possible_words 下，
    猜测 guess 所对应的反馈信息熵

    Parameters
    ----------
    guess : str
        本轮猜测词
    possible_words : list[str]
        当前仍可能的真实答案集合
    get_feedback : function
        Wordle 官方反馈函数

    Returns
    -------
    entropy : float
        Shannon entropy (log2)
    """
    feedback_counter = Counter()

    for true_word in possible_words:
        fb = get_feedback(guess, true_word)
        fb = normalize_feedback(fb)
        feedback_counter[fb] += 1

    total = len(possible_words)
    entropy = 0.0

    for count in feedback_counter.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


# 测试函数
if __name__ == "__main__":
    # 从 wordle_game_simulate.py 导入 get_feedback 函数
    from wordle_game_simulate import get_feedback
    
    # 测试用例
    test_words = ["crane", "slate", "trace", "adieu", "raise"]
    possible_answers = ["crane", "slate", "trace"]
    
    # 测试每个猜测词的熵
    for guess in test_words:
        entropy = feedback_entropy(guess, possible_answers, get_feedback)
        print(f"猜测词 '{guess}' 在候选答案 {possible_answers} 下的反馈熵: {entropy:.4f}")
    
    # 测试边界情况
    print("\n边界测试:")
    # 只有一个可能答案
    single_answer = ["crane"]
    entropy = feedback_entropy("crane", single_answer, get_feedback)
    print(f"只有一个可能答案时的熵: {entropy:.4f}")
    
    # 所有答案都会产生相同反馈的情况
    similar_answers = ["crane", "crane"]  # 重复词会产生相同反馈
    entropy = feedback_entropy("aaaaa", similar_answers, get_feedback)
    print(f"所有答案产生相同反馈时的熵: {entropy:.4f}")