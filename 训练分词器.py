from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os

def train_tokenizer(data_path='TrainData.txt', vocab_size=15000, min_frequency=5):
    """
    训练一个BPE分词器
    
    参数:
        data_path (str): 训练数据文件路径
        vocab_size (int): 词汇表大小
        min_frequency (int): 词元最小出现频率
    """
    if not os.path.exists(data_path):
        print(f"错误: 找不到训练数据文件 {data_path}")
        return None
    
    # 读取文件行数
    with open(data_path, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    
    print(f"开始训练分词器，使用 {line_count} 行文本数据")
    
    # 初始化分词器
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # 设置训练器
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
    )
    
    # 设置预分词器
    tokenizer.pre_tokenizer = Whitespace()
    
    # 从文件训练
    tokenizer.train(files=[data_path], trainer=trainer)
    
    # 添加后处理器来处理序列（添加BOS和EOS标记）
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]"))
        ]
    )
    
    # 保存分词器
    tokenizer.save("tokenizer.json")
    print(f"分词器训练完成，词汇表大小: {tokenizer.get_vocab_size()}")
    print(f"分词器已保存到 tokenizer.json")
    
    # 测试分词器
    test_tokenization(tokenizer)
    
    return tokenizer

def test_tokenization(tokenizer):
    """测试分词器在一些示例文本上的效果"""
    test_sentences = [
        "这是一个测试句子，用来检验分词器的效果。",
        "Transformer模型在自然语言处理领域取得了巨大成功。",
        "混合中英文句子testing the tokenizer's performance。",
        "小明是一个好孩子，他喜欢学习，喜欢帮助别人。",
        "英伟达的DLSS4.0技术在游戏领域取得了显著的提升。",
        "AI技术的发展正在改变我们的生活，让我们更加便捷地获取信息。",
        "这个模型在处理长文本时表现出色，能够保持较高的准确率。",
        "人工智能在医疗领域的应用正在逐步深入，为患者提供更精准的诊断和治疗方案。",
        "随着技术的不断进步，自动驾驶汽车将变得更加安全可靠。",
        "量子计算机的研究进展迅速，有望在未来解决更多复杂问题。"
    ]
    
    print("\n分词器测试:")
    for sentence in test_sentences:
        output = tokenizer.encode(sentence)
        print(f"\n原文: {sentence}")
        print(f"分词结果: {output.tokens}")
        print(f"ID: {output.ids[:30]}{'...' if len(output.ids) > 30 else ''}")

if __name__ == "__main__":
    print("开始训练分词器...")
    
    # 检查训练数据是否存在
    if not os.path.exists('TrainData.txt'):
        print("错误: 找不到TrainData.txt文件，请先运行数据预处理脚本")
    else:
        tokenizer = train_tokenizer('TrainData.txt')
