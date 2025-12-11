import pandas as pd
import nltk
from wordfreq import zipf_frequency

# 下载 nltk 需要的词性标注资源 (第一次运行需要)
print("正在下载语言学资源...")
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

def get_pos(word):
    """获取单词的词性 (Noun/Verb/Adj)"""
    # NLTK 返回的是 [('WORD', 'TAG')]
    try:
        tag = nltk.pos_tag([word], tagset='universal')[0][1]
        return tag
    except:
        return 'NOUN' # 默认兜底

# 1. 读取你现有的数据
input_file = 'data_with_features.xlsx'
print(f"正在读取 {input_file}...")
df = pd.read_excel(input_file)
df.columns = [c.strip() for c in df.columns] # 清理列名空格

# 2. 添加【词频】特征 (Word Frequency)
# zipf_frequency 返回一个 0-8 的数值，数值越大越常用
# 比如 'the' 是 7.x, 'eerie' 可能只有 2.x
print("正在计算单词词频...")
df['word_freq'] = df['word'].apply(lambda x: zipf_frequency(str(x), 'en'))

# 3. 添加【词性】特征 (Part of Speech)
print("正在识别单词词性...")
df['pos_tag'] = df['word'].apply(lambda x: get_pos(str(x)))
# 将词性转换为 One-Hot 编码 (0/1)
# 这样模型才能处理：is_noun, is_verb, is_adj
df['is_noun'] = (df['pos_tag'] == 'NOUN').astype(int)
df['is_verb'] = (df['pos_tag'] == 'VERB').astype(int)
df['is_adj']  = (df['pos_tag'] == 'ADJ').astype(int)

# 4. 保存为新文件
output_file = 'data_final.csv'
df.to_csv(output_file, index=False)
print(f"\n✅ 成功！新数据已保存为: {output_file}")
print(f"新增了 4 列特征: word_freq, is_noun, is_verb, is_adj")