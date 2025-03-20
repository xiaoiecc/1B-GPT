import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import Tokenizer
from tqdm import tqdm
import pickle
import random

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print("使用GPU进行训练")
else:
    print("未检测到GPU，使用CPU进行训练")

# 超参数
vocab_size = 15000
max_seq_len = 2048
d_model = 8
num_layers = 12
num_heads = 8
d_ff = 8
num_experts = 8
dropout_rate = 0.05
EPOCHS = 3
global_batch_size = 2
learning_rate = 1e-4
sample_interval = 100
micro_batch_size = 4
samples_per_epoch = 50000

def cosine_positional_encoding(max_seq_len, d_model):
    pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
    denominator = torch.pow(10000.0, dim / d_model)
    pe = torch.cos(pos / denominator)
    return pe.to(device)

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, activation=nn.GELU):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_ff = d_ff
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                activation(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        x = inputs.contiguous().view(-1, self.d_model)  # (batch*seq, d_model)
        gate_weights = self.softmax(self.gate(x))        # (batch*seq, num_experts)
        # 得到各专家的输出，并叠加
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (batch*seq, num_experts, d_model)
        gate_weights = gate_weights.unsqueeze(-1)  # (batch*seq, num_experts, 1)
        output = torch.sum(gate_weights * expert_outputs, dim=1)  # (batch*seq, d_model)
        return output.view(batch_size, seq_len, self.d_model)

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts, dropout_rate=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.moe = MoELayer(d_model, d_ff, num_experts)

    def forward(self, x, mask=None):
        # nn.MultiheadAttention要求输入形状(seq_len, batch, embed_dim)
        x_t = x.transpose(0, 1)
        attn_output, _ = self.mha(x_t, x_t, x_t, attn_mask=mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(x_t + attn_output)
        out1 = out1.transpose(0, 1)
        moe_output = self.moe(out1)
        moe_output = self.dropout2(moe_output)
        out2 = self.norm2(out1 + moe_output)
        return out2

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, num_layers, num_heads, d_ff, num_experts, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = cosine_positional_encoding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.dec_layers = nn.ModuleList(
            [TransformerDecoderBlock(d_model, num_heads, d_ff, num_experts, dropout_rate) for _ in range(num_layers)]
        )
        self.final_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x shape: (batch, seq_len)
        seq_len = x.size(1)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = x + self.pos_encoding[:seq_len, :]
        x = self.dropout(x)
        # 构建因果mask (上三角填充0)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('0'), diagonal=1)
        for dec_layer in self.dec_layers:
            x = dec_layer(x, mask=mask)
        logits = self.final_layer(x)
        return logits

print("加载分词器...")
tokenizer = Tokenizer.from_file("tokenizer.json")

def load_token_tree():
    print("加载token树...")
    if not os.path.exists('token_tree.pkl'):
        raise FileNotFoundError("错误: token_tree.pkl不存在，请先运行数据预处理.py构建token树")
    with open('token_tree.pkl', 'rb') as f:
        token_tree = pickle.load(f)
    print(f"Token树包含 {len(token_tree)} 个键")
    return token_tree

def generate_sequence_from_tree(token_tree, max_length=512):
    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("分词器中未定义 [BOS] 或 [EOS] 标记")
    sequence = [bos_token_id]
    current_token = bos_token_id
    while len(sequence) < max_length:
        if current_token not in token_tree or not token_tree[current_token]:
            sequence.append(eos_token_id)
            break
        next_tokens = list(token_tree[current_token].keys())
        probabilities = list(token_tree[current_token].values())
        prob_sum = sum(probabilities)
        if abs(prob_sum - 1.0) > 1e-10:
            probabilities = [p/prob_sum for p in probabilities]
        try:
            next_token = np.random.choice(next_tokens, p=probabilities)
        except ValueError as e:
            print(f"警告: 节点 {current_token} 的概率和有问题 ({e})，使用均匀分布")
            next_token = np.random.choice(next_tokens)
        sequence.append(next_token)
        if next_token == eos_token_id:
            break
        current_token = next_token
    if len(sequence) < 2:
        sequence = [bos_token_id, eos_token_id]
    if sequence[-1] != eos_token_id:
        sequence.append(eos_token_id)
    seq_array = np.array(sequence, dtype=np.int64)
    if len(seq_array) < max_seq_len:
        seq_array = np.pad(seq_array, (0, max_seq_len - len(seq_array)), mode='constant', constant_values=0)
    else:
        seq_array = seq_array[:max_seq_len]
    return seq_array

def generate_micro_batch(token_tree, micro_batch_size, max_seq_len):
    sequences = []
    for _ in range(micro_batch_size):
        seq = generate_sequence_from_tree(token_tree, max_length=max_seq_len-1)
        sequences.append(seq)
    # 按照警告修改
    numpy_array = np.array(sequences)
    return torch.tensor(numpy_array, dtype=torch.long, device=device)

# 使用CrossEntropyLoss并忽略padding(0)
criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

# 初始化模型、优化器
model = TransformerDecoder(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    num_experts=num_experts,
    dropout_rate=dropout_rate
).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def compute_loss(logits, targets):
    # logits: (batch, seq_len, vocab_size) => (batch*seq_len, vocab_size)
    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1)
    loss = criterion(logits, targets)
    # 除以非pad的数量
    non_pad = (targets != 0).sum().float()
    return loss / non_pad

def train_step_with_accumulation(micro_batches, num_accumulation_steps):
    if num_accumulation_steps == 0:
        print("警告：num_accumulation_steps 为 0，无法计算损失。")
        return 0.0
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for micro_batch in micro_batches:
        inp = micro_batch[:, :-1]
        tar = micro_batch[:, 1:]
        logits = model(inp)
        loss = compute_loss(logits, tar)
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    return total_loss / num_accumulation_steps

def generate_text(model, tokenizer, prompt, gen_length=20):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer.encode(prompt)
        token_ids = encoding.ids
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        for _ in range(gen_length):
            if input_ids.size(1) > max_seq_len:
                input_ids = input_ids[:, -max_seq_len:]
            logits = model(input_ids)
            next_logits = logits[:, -1, :]
            probs = torch.softmax(next_logits, dim=-1)
            max_prob = probs.max(dim=-1, keepdim=True)[0]
            candidate_mask = probs >= 0.7 * max_prob
            masked_probs = torch.where(candidate_mask, probs, torch.zeros_like(probs))
            total = masked_probs.sum(dim=-1, keepdim=True)
            normalized_probs = torch.where(total == 0, probs, masked_probs / total)
            normalized_probs_np = normalized_probs.cpu().numpy()[0]
            next_token_id = int(np.random.choice(np.arange(vocab_size), p=normalized_probs_np))
            eos_token_id = tokenizer.token_to_id("[EOS]")
            if next_token_id == eos_token_id:
                break
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
        generated_ids = input_ids.cpu().numpy()[0].tolist()
        generated_text = tokenizer.decode(generated_ids)
    return generated_text

# 主训练循环
print("开始训练...")
token_tree = load_token_tree()
step = 0
num_accumulation_steps = global_batch_size // micro_batch_size
batch_count = samples_per_epoch // global_batch_size

for epoch in range(EPOCHS):
    start_time = time.time()
    total_loss = 0.0
    progress_bar = tqdm(range(batch_count), desc=f"Epoch {epoch+1}")
    for _ in progress_bar:
        micro_batches = []
        for _ in range(num_accumulation_steps):
            micro_batch = generate_micro_batch(token_tree, micro_batch_size, max_seq_len)
            micro_batches.append(micro_batch)
        batch_loss = train_step_with_accumulation(micro_batches, num_accumulation_steps)
        total_loss += batch_loss
        progress_bar.set_postfix(loss=f"{batch_loss:.4f}")
        if step % sample_interval == 0:
            sample_prompts = ["今天", "人工智能", "天气"]
            for sp in sample_prompts:
                gen_text = generate_text(model, tokenizer, sp, gen_length=20)
                print(f"生成 (提示 '{sp}')：{gen_text}")
            print("\n从token树随机游走生成:")
            tree_seq = generate_sequence_from_tree(token_tree, max_length=30)
            tree_text = tokenizer.decode(tree_seq.tolist())
            print(f"随机游走: {tree_text}")
        step += 1
    epoch_loss = total_loss / batch_count
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}, Time: {time.time() - start_time:.2f} secs")

torch.save(model.state_dict(), "transformer_decoder_moe_weights.pth")
print("训练完成，模型权重已保存到 transformer_decoder_moe_weights.pth")