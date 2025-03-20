import os
import re
import glob
import json
import pickle
import numpy as np
from tokenizers import Tokenizer
from collections import defaultdict

# 定义要保留的常用符号
COMMON_PUNCTUATION = r'，。！？、；：‘’“”（）【】《》〈〉『』「」﹃﹄〔〕…—～‖·\.,!?;:()\[\]{}<>\"\'-_+=*/\\@#$%^&~`|'

def is_valid_line(line):
    """检查行是否符合要求"""
    # 去除空行
    if not line.strip():
        return False
    
    # 去除字符数过少的行 (这里假设少于5个字符为"过少"，可以根据需要调整)
    if len(line.strip()) < 20:
        return False
    
    # 去除字符数大于2048的行
    if len(line.strip()) > 2048:
        return False
    
    # 去除以数字开头的行
    if re.match(r'^\d', line.strip()):
        return False
    
    return True

def clean_line(line):
    """清理行中的非中英文字符（保留常用符号）"""
    # 保留中文、英文字母、数字和常用符号
    pattern = f'[^\u4e00-\u9fff\u3040-\u30ffA-Za-z0-9{COMMON_PUNCTUATION}]'
    return re.sub(pattern, '', line)

def process_files():
    """处理Data文件夹中的所有TXT文件"""
    if not os.path.exists('Data'):
        print("错误: Data文件夹不存在")
        return
    
    # 获取所有TXT文件路径
    txt_files = glob.glob(os.path.join('Data', '*.txt'))
    if not txt_files:
        print("错误: Data文件夹中没有找到TXT文件")
        return
    
    valid_lines = set()
    
    # 处理每个文件
    for file_path in txt_files:
        print(f"处理文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if is_valid_line(line):
                    cleaned_line = clean_line(line.strip())
                    if cleaned_line:  # 确保清理后的行不为空
                        valid_lines.add(cleaned_line)
    
    # 保存处理后的数据
    with open('TrainData.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_lines))
    
    print(f"处理完成: 保存了 {len(valid_lines)} 行数据到 TrainData.txt")
    return len(valid_lines)

def build_token_tree():
    """
    使用分词器将TrainData.txt中的文本转换为token树，
    [BOS]为根节点，每条路径以[EOS]结束，计算每个节点子节点的概率
    """
    if not (os.path.exists('TrainData.txt') and os.path.exists('tokenizer.json')):
        print("错误: TrainData.txt或tokenizer.json不存在，无法构建token树")
        return
    
    print("开始构建token树...")
    # 加载分词器
    tokenizer = Tokenizer.from_file("tokenizer.json")
    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")
    
    # 初始化树结构
    # tree结构: {token_id: {next_token_id: count, ...}, ...}
    tree = defaultdict(lambda: defaultdict(int))
    
    # 读取并处理每行文本
    line_count = 0
    with open('TrainData.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # 对文本进行分词
            encoding = tokenizer.encode(line)
            token_ids = encoding.ids
            
            # 确保每个序列以BOS开始，以EOS结束
            if token_ids[0] != bos_token_id:
                token_ids = [bos_token_id] + token_ids
            if token_ids[-1] != eos_token_id:
                token_ids.append(eos_token_id)
            
            # 构建树：统计每个token后面接哪些token及其次数
            for i in range(len(token_ids) - 1):
                current_token = token_ids[i]
                next_token = token_ids[i + 1]
                tree[current_token][next_token] += 1
            
            line_count += 1
            if line_count % 1000 == 0:
                print(f"已处理 {line_count} 行文本")
    
    print(f"总共处理了 {line_count} 行文本")
    
    # 计算每个节点子节点的概率
    prob_tree = {}
    for token_id, next_tokens in tree.items():
        total_count = sum(next_tokens.values())
        prob_tree[token_id] = {}
        
        # 计算子节点概率
        for next_token, count in next_tokens.items():
            prob = count / total_count
            # 处理浮点数精度问题
            prob = round(prob, 6)  # 保留6位小数
            prob_tree[token_id][next_token] = prob
        
        # 规范化概率，确保总和为1.0（处理浮点数误差）
        prob_sum = sum(prob_tree[token_id].values())
        if abs(prob_sum - 1.0) > 1e-6:  # 如果总和与1.0有显著差异
            # 找到最大概率的子节点，将误差加到它上面
            max_prob_token = max(prob_tree[token_id], key=lambda k: prob_tree[token_id][k])
            prob_tree[token_id][max_prob_token] += (1.0 - prob_sum)
    
    # 存储树结构
    token_tree_file = 'token_tree.pkl'
    with open(token_tree_file, 'wb') as f:
        pickle.dump(prob_tree, f)
    
    # 为了便于调试和检查，也保存一个可读的JSON版本
    readable_tree = {}
    for token_id, next_tokens in prob_tree.items():
        token = tokenizer.id_to_token(token_id) if token_id < tokenizer.get_vocab_size() else str(token_id)
        readable_tree[token] = {}
        for next_token_id, prob in next_tokens.items():
            next_token = tokenizer.id_to_token(next_token_id) if next_token_id < tokenizer.get_vocab_size() else str(next_token_id)
            readable_tree[token][next_token] = prob
    
    with open('token_tree_readable.json', 'w', encoding='utf-8') as f:
        json.dump(readable_tree, f, ensure_ascii=False, indent=2)
    
    print(f"token树已保存到 {token_tree_file} 和 token_tree_readable.json")
    return prob_tree

if __name__ == "__main__":
    total_lines = process_files()
    print(f"预处理完成，总共保存了 {total_lines} 行有效数据")
    
    # 检查是否存在TrainData.txt和tokenizer.json，如果存在则构建token树
    if os.path.exists('TrainData.txt') and os.path.exists('tokenizer.json'):
        build_token_tree()
    else:
        print("未发现分词器或训练数据，跳过token树构建")
