# 最优规则
# 第一次随机选择词
# 之后在前一次信息反馈的基础上再选择词
import random
from collections import Counter
from tqdm import trange
import math
import os
import csv

def calculate_entropy(guess, possible_words):
    """
    计算一个猜测词的熵值。
    通过模拟每个候选词对该猜测词的反馈来计算熵。
    """
    feedback_distribution = Counter()

    # 模拟所有可能的真实词
    for true_word in possible_words:
        feedback = get_feedback(guess, true_word)
        feedback_tuple = tuple(feedback.values())  # 转换反馈为元组，方便计数
        feedback_distribution[feedback_tuple] += 1

    # 计算熵
    total = sum(feedback_distribution.values())
    entropy = 0
    for count in feedback_distribution.values():
        probability = count / total
        entropy -= probability * math.log(probability, 2)  # 以 2 为底计算信息熵

    return entropy

def get_feedback(guess, true_word):
    """
    返回 feedback 字典: {位置: 'green'/'yellow'/'gray'}
    
    规则（与真实 Wordle 一致）：
    1. 先标 green（位置和字母都对）
    2. 再标 yellow（字母对但位置不对），但 yellow 数量不能超过答案中剩余的该字母数量
    3. 剩下的标 gray
    """
    feedback = ['gray'] * len(guess)
    answer_counts = Counter(true_word)
    
    # 第一遍：标 green
    for i, (g, t) in enumerate(zip(guess, true_word)):
        if g == t:
            feedback[i] = 'green'
            answer_counts[g] -= 1  # 消耗掉一个
    
    # 第二遍：标 yellow
    for i, g in enumerate(guess):
        if feedback[i] == 'gray' and answer_counts[g] > 0:
            feedback[i] = 'yellow'
            answer_counts[g] -= 1  # 消耗掉一个
    
    return {i: feedback[i] for i in range(len(guess))}

def fit_green(accept_word, words_now):
    """
    根据已确定的绿色字母过滤候选词：
    - accept_word 是长度为 5 的列表，如 ['m', '?', 'n', '?', 'y']
    - 仅当 word 在所有已知绿色位置上都与 accept_word 一致时保留
    """
    return [
        word
        for word in words_now
        if all(
            accept_word[i] == '?' or word[i] == accept_word[i]
            for i in range(len(accept_word))
        )
    ]


def fit_yellow(yellow_info, words_now):
    """
    根据黄色字母过滤候选词：
    - yellow_info: {letter: [不能出现的位置列表]}
    - 候选词必须：
      1. 包含所有黄色字母
      2. 这些字母不能出现在已知的"错误位置"上
    """
    for letter, forbidden_positions in yellow_info.items():
        # 必须包含该字母
        words_now = [w for w in words_now if letter in w]
        # 该字母不能出现在 forbidden_positions 中的任何位置
        words_now = [w for w in words_now if all(w[pos] != letter for pos in forbidden_positions)]
    return words_now


def fit_gray(gray_letters, yellow_letters, green_letters, words_now):
    """
    根据灰色字母过滤候选词：
    - gray_letters: 确定不在答案中的字母集合
    - 但要注意：如果某字母已经是 yellow 或 green，不能简单排除
      （因为可能是"多余的那个"被标灰，但答案中确实有这个字母）
    - 所以只排除那些"纯灰"的字母（既不是 yellow 也不是 green）
    """
    # 只排除那些确定不在答案中的字母
    pure_gray = gray_letters - yellow_letters - green_letters
    return [w for w in words_now if all(letter not in w for letter in pure_gray)]

def optimal_rule(true_word, words_5):
    """
    模拟 Wordle 游戏：
    - 第一次随机选择一个词
    - 之后根据反馈过滤候选词，再随机选择
    - 返回猜中所需的次数（1-7，7 表示失败）
    - 对于词的选择都完全随机，不考虑词频和熵
    """
    # 初始化
    yellow_info = {}           # {letter: [forbidden_positions]}
    gray_letters = set()       # 灰色字母集合
    yellow_letters = set()     # 出现过的黄色字母集合
    green_letters = set()      # 出现过的绿色字母集合
    accept_word = ['?'] * 5    # 已确定的绿色位置
    words_now = list(words_5)  # 当前候选词列表
    
    # 第一次随机选择
    guessed_word = random.choice(words_now)
    
    for try_num in range(1, 8):  # 最多 7 次（1-6 正常，7 表示失败）
        # 检查是否猜中
        if guessed_word == true_word:
            return try_num
        
        if try_num == 7:
            # 第 7 次还没猜中，返回 7 表示失败
            return 7
        
        # 获取反馈
        feedback = get_feedback(guessed_word, true_word)
        
        # 更新约束信息
        for pos, color in feedback.items():
            guess_char = guessed_word[pos]
            
            if color == 'green':
                accept_word[pos] = guess_char
                green_letters.add(guess_char)
                
            elif color == 'yellow':
                yellow_letters.add(guess_char)
                # 记录这个字母不能出现在这个位置
                if guess_char not in yellow_info:
                    yellow_info[guess_char] = []
                yellow_info[guess_char].append(pos)
                
            elif color == 'gray':
                gray_letters.add(guess_char)
        
        # 过滤候选词
        words_now = fit_green(accept_word, words_now)
        words_now = fit_yellow(yellow_info, words_now)
        words_now = fit_gray(gray_letters, yellow_letters, green_letters, words_now)
        
        # 移除已经猜过的词
        if guessed_word in words_now:
            words_now.remove(guessed_word)
        
        # 检查是否还有候选词
        if not words_now:
            return 7  # 没有候选词了，失败
        
        # 随机选择下一个词
        guessed_word = random.choice(words_now)
    
    return 7

def optimal_rule_according_to_frequency(true_word, words_5):
    """
    在前规则的基础上，每次都优先选择词频最高的词
    表示人们都会把最先想到的词考虑在内
    """
    # 初始化
    yellow_info = {}           # {letter: [forbidden_positions]}
    gray_letters = set()       # 灰色字母集合
    yellow_letters = set()     # 出现过的黄色字母集合
    green_letters = set()      # 出现过的绿色字母集合
    accept_word = ['?'] * 5    # 已确定的绿色位置
    words_now = list(words_5)  # 当前候选词列表
    random_pick_prob = [0.2, 0.3, 0.5]
    
    # 第一次随机选择
    top_k = max(1, int(len(words_now) * random.choice(random_pick_prob)))
    guessed_word = random.choice(words_now[:top_k])
    
    for try_num in range(1, 8):  # 最多 7 次（1-6 正常，7 表示失败）
        # 检查是否猜中
        if guessed_word == true_word:
            return try_num
        
        if try_num == 7:
            # 第 7 次还没猜中，返回 7 表示失败
            return 7
        
        # 获取反馈
        feedback = get_feedback(guessed_word, true_word)
        
        # 更新约束信息
        for pos, color in feedback.items():
            guess_char = guessed_word[pos]
            
            if color == 'green':
                accept_word[pos] = guess_char
                green_letters.add(guess_char)
                
            elif color == 'yellow':
                yellow_letters.add(guess_char)
                # 记录这个字母不能出现在这个位置
                if guess_char not in yellow_info:
                    yellow_info[guess_char] = []
                yellow_info[guess_char].append(pos)
                
            elif color == 'gray':
                gray_letters.add(guess_char)
        
        # 过滤候选词
        words_now = fit_green(accept_word, words_now)
        words_now = fit_yellow(yellow_info, words_now)
        words_now = fit_gray(gray_letters, yellow_letters, green_letters, words_now)
        
        # 移除已经猜过的词
        if guessed_word in words_now:
            words_now.remove(guessed_word)
        
        # 检查是否还有候选词
        if not words_now:
            return 7  # 没有候选词了，失败
        
        # 随机选择下一个词
        top_k = max(1, int(len(words_now) * random.choice(random_pick_prob)))
        guessed_word = random.choice(words_now[:top_k])
    
    return 7

def optimal_rule_according_to_entropy(true_word, words_5):
    """
    模拟 Wordle 游戏：
    选择一个根据熵值计算的最优猜测词。
    """
    # 初始化
    yellow_info = {}           # {letter: [forbidden_positions]}
    gray_letters = set()       # 灰色字母集合
    yellow_letters = set()     # 出现过的黄色字母集合
    green_letters = set()      # 出现过的绿色字母集合
    accept_word = ['?'] * 5    # 已确定的绿色位置
    words_now = list(words_5)  # 当前候选词列表
    
    # 第一次随机选择
    guessed_word = random.choice(words_now)
    
    for try_num in range(1, 8):  # 最多 7 次（1-6 正常，7 表示失败）
        # 检查是否猜中
        if guessed_word == true_word:
            return try_num
        
        if try_num == 7:
            # 第 7 次还没猜中，返回 7 表示失败
            return 7
        
        # 获取反馈
        feedback = get_feedback(guessed_word, true_word)
        
        # 更新约束信息
        for pos, color in feedback.items():
            guess_char = guessed_word[pos]
            
            if color == 'green':
                accept_word[pos] = guess_char
                green_letters.add(guess_char)
                
            elif color == 'yellow':
                yellow_letters.add(guess_char)
                # 记录这个字母不能出现在这个位置
                if guess_char not in yellow_info:
                    yellow_info[guess_char] = []
                yellow_info[guess_char].append(pos)
                
            elif color == 'gray':
                gray_letters.add(guess_char)
        
        # 过滤候选词
        words_now = fit_green(accept_word, words_now)
        words_now = fit_yellow(yellow_info, words_now)
        words_now = fit_gray(gray_letters, yellow_letters, green_letters, words_now)
        
        # 移除已经猜过的词
        if guessed_word in words_now:
            words_now.remove(guessed_word)
        
        # 检查是否还有候选词
        if not words_now:
            return 7  # 没有候选词了，失败
        
        # 计算每个候选词的熵值，并选择熵值最大的词作为下一个猜测
        entropy_values = {word: calculate_entropy(word, words_now) for word in words_now}
        guessed_word = max(entropy_values, key=entropy_values.get)
    
    return 7

def run_optimal_rule_for_word(args):
    word, words_5 = args
    res = []
    for _ in range(10000):
        res.append(optimal_rule(word, words_5))
    return word, res

def run_optimal_rule_according_to_frequency_for_word(args):
    word, words_5 = args
    res = []
    for _ in range(10000):
        res.append(optimal_rule_according_to_frequency(word, words_5))
    return word, res

def run_optiam_rule_according_to_entropy_for_word(args):
    word, words_5 = args
    res = []
    for _ in range(10000):
        res.append(optimal_rule_according_to_entropy(word, words_5))
    return word, res

if __name__ == '__main__':
    results = []
    from tqdm import tqdm
    from wordfreq import top_n_list
    from multiprocessing import Pool, cpu_count
    import pandas as pd

    # 获取词频最高的 500000 个英文单词
    top5000 = top_n_list("en", 500000)

    # 筛选长度为 5 的
    words_5 = [w for w in top5000 if len(w) == 5 if "'" not in w][:20000]
    print("从" + str(len(words_5)) + "个5字母单词中选择答案词进行模拟")
    
    # 多线程作业
    df = pd.read_csv('data/mcm_processed_data.csv', low_memory=False)
    
    words = df['word'].tolist()

    # 计算时间较长，采用多线程机制
    with Pool(cpu_count()) as p:
        total = len(words)
        for word, res in tqdm(
            p.imap_unordered(run_optiam_rule_according_to_entropy_for_word, [(w, words_5) for w in words]),
            total = total
        ):
            results.append((word, res))
        

    # 将结果保存到表格中：每一行对应一个 word，一列存放 10000 次结果的列表
    out_df = pd.DataFrame({
        'word': [w for w, _ in results],
        'results': [vals for _, vals in results]
    })
    
    # 计算平均值
    out_df['mean_simulate_entropy'] = out_df['results'].apply(lambda x: sum(x)/len(x))
    out_df['1_try_simulate_entropy'] = out_df['results'].apply(lambda xs: sum(1 for v in xs if v == 1) / len(xs) if xs else 0)
    out_df['2_try_simulate_entropy'] = out_df['results'].apply(lambda xs: sum(1 for v in xs if v == 2) / len(xs) if xs else 0)
    out_df['3_try_simulate_entropy'] = out_df['results'].apply(lambda xs: sum(1 for v in xs if v == 3) / len(xs) if xs else 0)
    out_df['4_try_simulate_entropy'] = out_df['results'].apply(lambda xs: sum(1 for v in xs if v == 4) / len(xs) if xs else 0)
    out_df['5_try_simulate_entropy'] = out_df['results'].apply(lambda xs: sum(1 for v in xs if v == 5) / len(xs) if xs else 0)
    out_df['6_try_simulate_entropy'] = out_df['results'].apply(lambda xs: sum(1 for v in xs if v == 6) / len(xs) if xs else 0)
    out_df['7_try_simulate_entropy'] = out_df['results'].apply(lambda xs: sum(1 for v in xs if v == 7) / len(xs) if xs else 0)

    out_df.to_csv("sum_manly_results_entropy.csv", index=False)
    
    df = df.merge(out_df[['word', 'mean_simulate_entropy', '1_try_simulate_entropy',
                          '2_try_simulate_entropy', '3_try_simulate_entropy', '4_try_simulate_entropy',
                          '5_try_simulate_entropy', '6_try_simulate_entropy', '7_try_simulate_entropy']], on='word', how='left')
    df.to_csv("mcm_processed_data.csv", index=False)
    print("已完成")
    

    