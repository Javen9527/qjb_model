      
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import List, Union
from fastapi import FastAPI  # 用于部署API服务

# --------------------------- 1. 模型核心结构（训练/推理/部署共用） ---------------------------
# 工具函数：掩码生成
def create_padding_mask(seq, device):
    """
    为什么要这样设计？
    注意力计算中，Q、K、V 的形状是[batch_size, num_heads, seq_len, d_k]，掩码需要与之匹配才能广播（broadcast）。
    最终掩码形状[batch_size, 1, 1, seq_len]可以通过广播适配[batch_size, num_heads, seq_len, seq_len]的注意力权重矩阵，实现对所有头、所有位置的 PAD 掩盖。
    """
    # 步骤1：标记PAD位置（PAD=0）
    # seq是输入序列，形状为[batch_size, seq_len]（例如：[[1,2,0,0], [3,0,0,0]]）
    # (seq == 0)会生成布尔矩阵，PAD位置为True，其他为False：[[False,False,True,True], [False,True,True,True]]
    # .float()将布尔值转为0.0（非PAD）和1.0（PAD）：[[0.,0.,1.,1.], [0.,1.,1.,1.]]
    mask = (seq == 0).float()
    
    # 步骤2：增加维度，适配注意力计算的形状
    # 原mask形状为[batch_size, seq_len]，通过两次unsqueeze增加两个维度：
    # unsqueeze(1) → [batch_size, 1, seq_len]
    # unsqueeze(2) → [batch_size, 1, 1, seq_len]（最终形状）
    mask = mask.unsqueeze(1).unsqueeze(2)
    
    # 步骤3：将掩码移动到指定设备（CPU/GPU）
    return mask.to(device)

def create_look_ahead_mask(size, device):
    """
    为什么要这样设计？
    解码器的任务是 “自回归生成”（如翻译时从左到右生成目标语言），必须保证预测时 “看不到未来的词”。
    下三角矩阵的反转（上三角为 1）刚好能标记 “未来位置”，与注意力分数矩阵相加后，未来位置的分数被压制，模型无法关注。
    """
    # 步骤1：生成下三角矩阵（对角线及以下为1，以上为0）
    # torch.ones(1, 1, size, size)生成全1矩阵，形状为[1,1,size,size]
    # torch.tril(...)取下三角部分（保留对角线及以下）：
    # 例如size=3时，结果为：
    # [[[[1.,1.,1.],
    #    [0.,1.,1.],
    #    [0.,0.,1.]]]]
    tril = torch.tril(torch.ones(1, 1, size, size, device=device))
    
    # 步骤2：反转矩阵（下三角为0，上三角为1）
    # 1 - tril 后，上三角（未来token位置）变为1.0，下三角（历史token位置）变为0.0：
    # [[[[0.,1.,1.],
    #    [0.,0.,1.],
    #    [0.,0.,0.]]]]
    mask = 1 - tril
    
    return mask

# 位置编码层
class PositionalEncoding(nn.Module):
    """
    位置编码的必要性
    Transformer 没有 RNN/LSTM 等循环结构，无法通过时序顺序隐式捕捉位置信息。位置编码的作用是：
    - 为每个位置分配一个独特的向量，让模型知道 “token 在序列中的位置”（如第 1 个词和第 3 个词的位置编码不同）；
    - 通过正弦 / 余弦函数生成的位置编码具有 “相对位置不变性”（即位置i和i+k的编码差异固定），便于模型学习相对位置关系
    """
    def __init__(self, d_model, max_len=5000, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pe = torch.zeros(max_len, d_model, device=self.device)
        position = torch.arange(0, max_len, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=self.device) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"序列长度 {seq_len} 超过位置编码最大长度 {self.max_len}")
        return x + self.pe[:seq_len, :].unsqueeze(0)

# 多头注意力层
class MultiHeadAttention(nn.Module):
    """
    多头注意力的核心价值
    1. 并行捕捉多维度依赖：每个注意力头可以关注序列中不同的模式（如一个头关注语法结构，另一个头关注语义关联），比单头注意力更全面。
    2. 降低计算复杂度：将d_model拆分为num_heads个d_k，单个头的计算复杂度从O(d_model²)降为O(d_k²)，整体复杂度不变但并行性更好。
    3. 增强模型表达能力：通过多个头的组合，模型能学习到更丰富的特征表示，提升对长序列依赖关系的捕捉能力。
    """
    def __init__(self, d_model, num_heads, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads  # 注意力头数（如8头）
        self.d_model = d_model      # 模型总维度（如512）
        
        # 检查维度合法性：总维度必须能被头数整除（每个头的维度需相同）
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} 必须能被 num_heads {num_heads} 整除")
        self.d_k = d_model // num_heads  # 单个头的维度（如512/8=64）
        
        # 定义Q、K、V的线性投影层（输入输出维度均为d_model）
        self.wq = nn.Linear(d_model, d_model).to(self.device)  # Q的投影矩阵
        self.wk = nn.Linear(d_model, d_model).to(self.device)  # K的投影矩阵
        self.wv = nn.Linear(d_model, d_model).to(self.device)  # V的投影矩阵
        self.wo = nn.Linear(d_model, d_model).to(self.device)  # 多头头结果拼接后的输出投影

    def split_heads(self, x, batch_size):
        # x形状：[batch_size, seq_len, d_model]（如[32, 10, 512]）
        # 步骤1：reshape为[batch_size, seq_len, num_heads, d_k]（如[32, 10, 8, 64]）
        # 步骤2：transpose(1, 2)交换维度1和2 → [batch_size, num_heads, seq_len, d_k]（如[32, 8, 10, 64]）
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        核心逻辑：通过 Q 与 K 的相似度计算注意力权重，再用权重对 V 加权求和，得到每个位置的 “关注结果”。
        """
        # q/k/v形状：[batch_size, num_heads, seq_len_q, d_k]（查询序列）和[..., seq_len_k, d_k]（键值序列）
        # 步骤1：计算Q与K的相似度（点积）
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # 形状：[batch, num_heads, seq_len_q, seq_len_k]
        # 例：查询序列长度10，键序列长度15 → 结果为10×15的注意力权重矩阵
        
        # 步骤2：缩放（防止维度d_k过大导致点积值过大，softmax后梯度消失）
        scaled_logits = matmul_qk / math.sqrt(self.d_k)  # 除以√d_k
        
        # 步骤3：应用掩码（掩盖无效位置，如PAD或未来token）
        if mask is not None:
            scaled_logits += (mask * -1e9)  # 无效位置加一个极大负数，softmax后接近0
        
        # 步骤4：计算注意力权重（归一化）
        attn_weights = F.softmax(scaled_logits, dim=-1)  # 沿最后一维（seq_len_k）归一化
        
        # 步骤5：加权求和（用注意力权重对V加权）
        output = torch.matmul(attn_weights, v)  # 形状：[batch, num_heads, seq_len_q, d_k]
        return output, attn_weights

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)  # 获取批次大小
        
        # 步骤1：线性投影（将输入特征映射到Q、K、V空间）
        q, k, v = self.wq(q), self.wk(k), self.wv(v)  # 形状均为[batch, seq_len, d_model]
        
        # 步骤2：拆分多头（每个头独立计算注意力）
        q, k, v = self.split_heads(q, batch_size), self.split_heads(k, batch_size), self.split_heads(v, batch_size)
        # 形状均为[batch, num_heads, seq_len, d_k]
        
        # 步骤3：计算缩放点积注意力（多头并行）
        scaled_attn, _ = self.scaled_dot_product_attention(q, k, v, mask)  # [batch, num_heads, seq_len, d_k]
        
        # 步骤4：拼接多头结果（将多个头的输出合并为总维度）
        # 先transpose(1, 2) → [batch, seq_len, num_heads, d_k]
        # 再reshape → [batch, seq_len, num_heads*d_k] = [batch, seq_len, d_model]
        scaled_attn = scaled_attn.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        
        # 步骤5：输出投影（将拼接后的结果映射回d_model维度）
        output = self.wo(scaled_attn)  # [batch, seq_len, d_model]
        return output

# 前馈网络层
class PointWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, device=None):
        """
        为什么需要前馈网络？
        在编码器层和解码器层中，前馈网络总是跟在注意力机制之后，形成 “注意力子层→前馈网络子层” 的结构，且每个子层都配有残差连接和层归一化
        注意力机制擅长捕捉序列中不同位置的依赖关系（如 “谁与谁相关”），但缺乏对单个位置特征的深度加工能力。前馈网络的作用是：
        - 增强非线性表达：通过 ReLU 激活函数引入非线性，让模型能学习更复杂的特征模式；
        - 维度扩展与压缩：先将d_model升维到d_ff（如 512→2048），再降维回d_model，相当于在高维空间中对特征进行更精细的调整，提升模型的拟合能力；
        - 与注意力机制互补：注意力负责 “捕捉关联”，前馈网络负责 “加工单个特征”，两者结合让模型既能关注全局依赖，又能深入挖掘局部特征。
        
        关键参数：
        - d_model：模型的核心维度（与注意力机制的输入 / 输出维度一致，如 512）；
        - d_ff：前馈网络中间层的维度（通常是d_model的 4 倍，如 2048），用于临时扩展特征维度，增强表达能力；
        - nn.Sequential：按顺序执行的网络容器，这里封装了 “升维→非线性激活→降维” 的流程。
        """
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 定义前馈网络的核心结构（序列式网络）
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff).to(self.device),  # 第一层线性变换：升维
            nn.ReLU().to(self.device),                 # 激活函数：引入非线性
            nn.Linear(d_ff, d_model).to(self.device)   # 第二层线性变换：降维
        )

    def forward(self, x):
        return self.net(x)

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rate=0.1, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 多头自注意力模块（Q=K=V，关注输入序列内部的依赖关系）
        self.mha = MultiHeadAttention(d_model, num_heads, self.device)
        
        # 2. 前馈网络模块（对注意力输出进行非线性特征变换）
        self.ffn = PointWiseFFN(d_model, d_ff, self.device)
        
        # 3. 层归一化（稳定训练时的数值分布，加速收敛）
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6).to(self.device)  # 用于注意力子层
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6).to(self.device)  # 用于前馈网络子层
        
        # 4. Dropout（防止过拟合，训练时随机丢弃部分神经元）
        self.dropout1 = nn.Dropout(rate).to(self.device)  # 用于注意力输出
        self.dropout2 = nn.Dropout(rate).to(self.device)  # 用于前馈网络输出

    def forward(self, x, mask):
        """
        编码器层的前向传播包含两个核心子层：多头自注意力子层和前馈网络子层，每个子层都遵循 “计算→Dropout→残差连接→层归一化” 的流程
        1. 为什么用 “自注意力”？
            自注意力（Q=K=V=x）让序列中的每个 token 都能关注到其他所有 token（受掩码限制），从而捕捉全局上下文依赖。例如：
            在机器翻译中，输入 “我吃了苹果”，“苹果” 需要关注 “吃” 才能正确翻译；
            在文本分类中，关键词需要关注整个句子的语境。
        2. Dropout 的作用
            训练时随机将部分特征值设为 0，防止模型过度依赖某些局部特征，从而缓解过拟合。推理时（model.eval()）会自动关闭。
        3. 残差连接（Residual Connection）的作用
            残差连接的公式是 x + sublayer_output（子层输出与输入相加）：
            解决深层网络的 “梯度消失” 问题：梯度可以直接通过残差路径回传，无需经过复杂的子层变换；
            保留原始特征：子层专注于学习 “增量特征”（输入与目标的差异），而非从头学习所有特征，降低学习难度。
        4. 层归一化（LayerNorm）的作用
            层归一化对每个样本的d_model维度进行归一化（均值为 0，方差为 1）：
            稳定训练过程中的数值分布：注意力和前馈网络的计算可能导致特征值波动，归一化能将其拉回合理范围；
            加速收敛：避免因数值过大 / 过小导致的梯度爆炸或训练停滞。
        """
        # 输入x形状：[batch_size, seq_len, d_model]（序列特征）
        # mask形状：[batch_size, 1, 1, seq_len]（填充掩码，掩盖PAD token）
        
        # -------------------------- 1. 多头自注意力子层 --------------------------
        # 计算自注意力：Q=K=V=x（关注输入序列内部的依赖关系）
        attn_output = self.mha(x, x, x, mask)  # 形状：[batch_size, seq_len, d_model]
        
        # 应用Dropout（训练时随机丢弃部分特征）
        attn_output = self.dropout1(attn_output)
        
        # 残差连接（x + 注意力输出）+ 层归一化
        # 残差连接：缓解深层网络梯度消失问题，保留原始特征
        # 层归一化：对每个样本的seq_len维度做归一化，稳定数值范围
        out1 = self.layernorm1(x + attn_output)  # 形状：[batch_size, seq_len, d_model]
        
        # -------------------------- 2. 前馈网络子层 --------------------------
        # 前馈网络处理：对每个位置的特征进行非线性变换
        ffn_output = self.ffn(out1)  # 形状：[batch_size, seq_len, d_model]
        
        # 应用Dropout
        ffn_output = self.dropout2(ffn_output)
        
        # 残差连接（out1 + 前馈网络输出）+ 层归一化
        out2 = self.layernorm2(out1 + ffn_output)  # 形状：[batch_size, seq_len, d_model]
        
        return out2  # 输出传给下一个编码器层

# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rate=0.1, device=None):
        super().__init__()
        """
        与编码器层的对比
        解码器层比编码器层多一个注意力模块（mha2），这是因为解码器需要同时处理两种依赖关系：
        - 目标序列内部的依赖（如生成句子时的语法连贯性）；
        - 目标序列与输入序列的依赖（如翻译时 “目标词” 与 “源词” 的对应关系）。
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 第一个多头注意力（解码器自注意力）：关注目标序列自身的依赖关系
        self.mha1 = MultiHeadAttention(d_model, num_heads, self.device)
        
        # 2. 第二个多头注意力（编解码注意力）：关注编码器输出的输入序列上下文
        self.mha2 = MultiHeadAttention(d_model, num_heads, self.device)
        
        # 3. 前馈网络：对特征进行非线性变换
        self.ffn = PointWiseFFN(d_model, d_ff, self.device)
        
        # 4. 三个层归一化：分别用于两个注意力子层和前馈网络子层
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6).to(self.device)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6).to(self.device)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6).to(self.device)
        
        # 5. 三个Dropout：分别用于两个注意力输出和前馈网络输出
        self.dropout1 = nn.Dropout(rate).to(self.device)
        self.dropout2 = nn.Dropout(rate).to(self.device)
        self.dropout3 = nn.Dropout(rate).to(self.device)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        """
        解码器层的前向传播包含三个核心子层：解码器自注意力子层、编解码注意力子层和前馈网络子层，每个子层同样遵循 “计算→Dropout→残差连接→层归一化” 的流程
        """
        # 输入参数：
        # x：解码器输入（已生成的目标序列特征），形状[batch_size, target_seq_len, d_model]
        # enc_output：编码器输出（输入序列的上下文特征），形状[batch_size, input_seq_len, d_model]
        # look_ahead_mask：前瞻掩码（掩盖解码器自注意力中的未来token）
        # padding_mask：填充掩码（掩盖输入序列中的PAD token）
        
        # -------------------------- 1. 解码器自注意力子层 --------------------------
        # 自注意力：Q=K=V=x（关注已生成的目标序列内部依赖）
        # 应用前瞻掩码，确保生成第i个token时只能看到1~i个历史token
        attn1 = self.mha1(x, x, x, look_ahead_mask)  # 形状[batch_size, target_seq_len, d_model]
        
        attn1 = self.dropout1(attn1)  # Dropout防止过拟合
        out1 = self.layernorm1(x + attn1)  # 残差连接+层归一化
        
        # -------------------------- 2. 编解码注意力子层 --------------------------
        # 编解码注意力：Q=out1（解码器特征），K=V=enc_output（编码器特征）
        # 作用：让解码器关注输入序列中与当前生成内容相关的部分（如翻译时对齐源词和目标词）
        # 应用填充掩码，掩盖输入序列中的PAD token
        attn2 = self.mha2(out1, enc_output, enc_output, padding_mask)  # 形状[batch_size, target_seq_len, d_model]
        
        attn2 = self.dropout2(attn2)  # Dropout
        out2 = self.layernorm2(out1 + attn2)  # 残差连接+层归一化
        
        # -------------------------- 3. 前馈网络子层 --------------------------
        # 对编解码注意力的输出进行非线性特征变换
        ffn_output = self.ffn(out2)  # 形状[batch_size, target_seq_len, d_model]
        
        ffn_output = self.dropout3(ffn_output)  # Dropout
        out3 = self.layernorm3(out2 + ffn_output)  # 残差连接+层归一化
        
        return out3  # 输出传给下一个解码器层

# 编码器
class Encoder(nn.Module):
    """
    结构: “词嵌入 + 位置编码 + 多层编码器层” 
    编码器的输出（enc_output）是包含输入序列全局上下文的特征向量，有两个核心作用：
    - 作为解码器中 “编解码注意力” 的K和V，让解码器在生成目标序列时能关注输入序列的相关信息（如翻译时对齐源词和目标词）；
    - 为整个模型提供输入序列的语义表示，是连接 “输入理解” 和 “输出生成” 的桥梁。
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, max_len=5000, rate=0.1, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.d_model = d_model  # 模型核心维度（贯穿整个编码器）
        
        # 1. 词嵌入层：将输入token ID转换为稠密向量
        self.embedding = nn.Embedding(input_vocab_size, d_model).to(self.device)
        # input_vocab_size：输入词汇表大小（如10000表示有10000个不同的token）
        
        # 2. 位置编码层：为序列注入位置信息（Transformer无循环结构，需显式编码位置）
        self.pos_encoding = PositionalEncoding(d_model, max_len, self.device)
        # max_len：支持的最大序列长度（超过会报错）
        
        # 3. 多层编码器层：堆叠num_layers个EncoderLayer，逐步提取深层特征
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, rate, self.device) 
            for _ in range(num_layers)
        ])
        
        # 4. Dropout层：对嵌入+位置编码后的特征进行随机丢弃，防止过拟合
        self.dropout = nn.Dropout(rate).to(self.device)

    def forward(self, x, mask):
        """
        编码器的前向传播流程可分为 4 步：词嵌入→位置编码→Dropout→多层编码器层处理，最终输出包含全局上下文的特征向量。
        """
        # 输入参数：
        # x：输入序列的token ID，形状[batch_size, input_seq_len]（如[[1,5,3,0], [2,7,0,0]]）
        # mask：填充掩码（padding mask），形状[batch_size, 1, 1, input_seq_len]（掩盖PAD token）
        
        # 步骤1：词嵌入 + 缩放
        # 词嵌入：将token ID转换为d_model维向量，形状[batch_size, input_seq_len, d_model]
        # 缩放：乘以√d_model（原论文技巧，平衡嵌入向量和位置编码的数值范围）
        x = self.embedding(x) * math.sqrt(self.d_model)  # 形状：[batch, seq_len, d_model]
        
        # 步骤2：添加位置编码
        # 位置编码：为每个位置注入独特的位置信息，形状与x一致
        x = self.pos_encoding(x)  # 形状不变：[batch, seq_len, d_model]
        
        # 步骤3：应用Dropout（训练时随机丢弃部分特征）
        x = self.dropout(x)
        
        # 步骤4：通过多层编码器层提取特征
        # 每层编码器层都会对x进行加工（捕捉更复杂的上下文依赖）
        for layer in self.layers:
            x = layer(x, mask)  # 每层输入输出形状均为[batch, seq_len, d_model]
        
        # 输出：经过多层处理的上下文特征向量，将作为解码器的输入之一
        return x  # 形状：[batch_size, input_seq_len, d_model]

# 解码器
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, target_vocab_size, max_len=5000, rate=0.1, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.d_model = d_model  # 模型核心维度（与编码器保持一致）
        
        # 1. 词嵌入层：将目标序列的token ID转换为稠密向量
        self.embedding = nn.Embedding(target_vocab_size, d_model).to(self.device)
        # target_vocab_size：目标词汇表大小（如翻译任务中目标语言的词汇量）
        
        # 2. 位置编码层：为目标序列注入位置信息（同编码器，确保模型感知序列顺序）
        self.pos_encoding = PositionalEncoding(d_model, max_len, self.device)
        
        # 3. 多层解码器层：堆叠num_layers个DecoderLayer，逐步处理生成逻辑
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, rate, self.device) 
            for _ in range(num_layers)
        ])
        
        # 4. Dropout层：对嵌入+位置编码后的特征进行随机丢弃，防止过拟合
        self.dropout = nn.Dropout(rate).to(self.device)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        """
        解码器的输出（dec_output）是经过多层处理的目标序列特征，最终会通过一个线性层（final_layer）映射到目标词汇表，得到每个位置的预测概率分布
        """
        # 输入参数：
        # x：目标序列的token ID（如翻译中的部分目标译文），形状[batch_size, target_seq_len]
        # enc_output：编码器输出的上下文特征，形状[batch_size, input_seq_len, d_model]
        # look_ahead_mask：前瞻掩码（用于解码器自注意力，掩盖未来token）
        # padding_mask：填充掩码（用于编解码注意力，掩盖输入序列的PAD token）
        
        # 步骤1：词嵌入 + 缩放
        # 词嵌入：将目标序列的token ID转换为d_model维向量，形状[batch_size, target_seq_len, d_model]
        # 缩放：同编码器，平衡嵌入向量和位置编码的数值范围
        x = self.embedding(x) * math.sqrt(self.d_model)  # 形状：[batch, target_seq_len, d_model]
        
        # 步骤2：添加位置编码
        # 为目标序列注入位置信息（确保模型知道生成的词在序列中的位置）
        x = self.pos_encoding(x)  # 形状不变：[batch, target_seq_len, d_model]
        
        # 步骤3：应用Dropout（训练时随机丢弃部分特征）
        x = self.dropout(x)
        
        # 步骤4：通过多层解码器层处理生成逻辑
        # 每层解码器层会同时关注目标序列自身和编码器输出的上下文
        for layer in self.layers:
            x = layer(x, enc_output, look_ahead_mask, padding_mask)  # 形状保持不变
        
        # 输出：经过多层处理的目标序列特征，后续将通过线性层映射到目标词汇表
        return x  # 形状：[batch_size, target_seq_len, d_model]

# 完整Transformer模型
class Transformer(nn.Module):
    """
    与训练 / 推理的关联
    - 训练阶段：final_output会与真实目标序列（[I, eat, apples]）通过交叉熵损失函数计算损失，反向传播优化整个模型的参数（包括编码器、解码器、最终层）；
    - 推理阶段：模型采用 “自回归生成”：从<BOS>开始，每次用final_output的概率分布选择下一个词，拼接后作为新的tar输入解码器，直到生成<EOS>（结束符）。
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size,
                 pe_input=5000, pe_target=5000, rate=0.1, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 1. 初始化编码器（处理输入序列，提取上下文特征）
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, pe_input, rate, self.device)
        # 2. 初始化解码器（生成目标序列，结合编码器输出）
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, target_vocab_size, pe_target, rate, self.device)
        # 3. 最终输出层（将解码器输出映射到目标词汇表）
        self.final_layer = nn.Linear(d_model, target_vocab_size).to(self.device)

    def forward(self, inp, tar, enc_pad_mask, look_ahead_mask, dec_pad_mask):
        """
        完整计算流程：输入序列→编码器处理→解码器处理→输出预测，是模型 “工作” 的核心逻辑
        """
        # 输入参数：
        # inp：输入序列的token ID，形状[batch_size, input_seq_len]（如中文句子的ID序列）
        # tar：目标序列的token ID（右移一位的训练输入），形状[batch_size, target_seq_len]（如英文句子的ID序列）
        # enc_pad_mask：编码器填充掩码，掩盖输入序列的PAD token
        # look_ahead_mask：解码器前瞻掩码，掩盖目标序列的未来token
        # dec_pad_mask：解码器填充掩码，掩盖输入序列的PAD token（用于编解码注意力）
        
        # 步骤1：编码器处理输入序列，输出上下文特征
        enc_output = self.encoder(inp, enc_pad_mask)  # 形状：[batch_size, input_seq_len, d_model]
        
        # 步骤2：解码器结合编码器输出和目标序列，生成中间特征
        dec_output = self.decoder(tar, enc_output, look_ahead_mask, dec_pad_mask)  # 形状：[batch_size, target_seq_len, d_model]
        
        # 步骤3：最终层映射到目标词汇表，输出logits（未归一化的概率）
        final_output = self.final_layer(dec_output)  # 形状：[batch_size, target_seq_len, target_vocab_size]
        
        return final_output

# --------------------------- 2. 训练模块（含模型保存） ---------------------------
class TransformerTrainer:
    def __init__(self, config: dict, save_dir: str = "models"):
        """初始化训练器"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)  # 创建模型保存目录

        # 1. 初始化Transformer模型（参数从config读取，确保与任务匹配）
        self.model = Transformer(
            num_layers=config["num_layers"],
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            input_vocab_size=config["input_vocab_size"],
            target_vocab_size=config["target_vocab_size"],
            device=self.device
        )
        # 2. 初始化损失函数：交叉熵损失（忽略PAD token，PAD=0）
        # 为什么忽略PAD？PAD是填充符号（无实际意义），计算损失时需排除，避免影响模型学习
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD token
        # 3. 初始化优化器：Adam优化器（Transformer原论文推荐配置）
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),  # 优化模型所有可训练参数
            lr=config.get("lr", 1e-4),  # 学习率（默认1e-4，可从config自定义）
            betas=(0.9, 0.98),  # 动量参数（原论文配置，加速收敛）
            eps=1e-9  # 防止分母为0的微小值
        )

    def generate_sample_data(self, batch_size: int):
        """生成模拟训练数据（实际需替换为真实数据集）"""
        input_seq_len = self.config["max_input_seq_len"]
        target_seq_len = self.config["max_output_seq_len"]
        inp = torch.randint(1, self.config["input_vocab_size"], (batch_size, input_seq_len), device=self.device)
        tar = torch.randint(1, self.config["target_vocab_size"], (batch_size, target_seq_len), device=self.device)
        return inp, tar

    def train(self, num_epochs: int = 5, batch_size: int = 64, num_batches: int = 100):
        """训练模型并保存"""
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for _ in range(num_batches):
                # 1. 准备数据
                inp, tar = self.generate_sample_data(batch_size)
                tar_input = tar[:, :-1]  # 解码器输入（右偏移）
                tar_label = tar[:, 1:]   # 标签（与输出对齐）

                # 2. 生成掩码
                enc_pad_mask = create_padding_mask(inp, self.device)
                dec_pad_mask = create_padding_mask(inp, self.device)
                look_ahead_mask = create_look_ahead_mask(tar_input.size(1), self.device)
                dec_target_pad_mask = create_padding_mask(tar_input, self.device)
                combined_mask = torch.max(dec_target_pad_mask, look_ahead_mask)

                # 3. 前向传播
                outputs = self.model(inp, tar_input, enc_pad_mask, combined_mask, dec_pad_mask)

                # 4. 计算损失
                loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), tar_label.reshape(-1))
                total_loss += loss.item()

                # 5. 反向传播与优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 打印每轮损失
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # 保存模型（训练完成后）
        model_path = os.path.join(self.save_dir, f"transformer_{num_epochs}epochs.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"模型已保存至: {model_path}")
        return model_path

# --------------------------- 3. 推理模块（含模型加载） ---------------------------
class SimpleTokenizer:
    """分词器（实际需替换为训练时使用的Tokenizer）"""
    def __init__(self, vocab_size: int, pad=0, bos=1, eos=2):
        self.vocab_size = vocab_size
        self.pad, self.bos, self.eos = pad, bos, eos
        self.id2token = {i: f"token_{i}" for i in range(vocab_size)}
        self.id2token.update({pad: "<PAD>", bos: "<BOS>", eos: "<EOS>"})
        self.token2id = {v: k for k, v in self.id2token.items()}

    def text_to_ids(self, text: str, max_len: int) -> torch.Tensor:
        """文本转Token ID"""
        tokens = text.split()[:max_len-2]  # 预留BOS和EOS
        ids = [self.bos] + [self.token2id.get(t, 3) for t in tokens] + [self.eos]
        ids += [self.pad] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def ids_to_text(self, ids: torch.Tensor) -> str:
        """Token ID转文本"""
        ids = ids.squeeze().tolist()
        return " ".join([self.id2token[id] for id in ids if id not in [self.pad, self.bos, self.eos]])

class TransformerInfer:
    def __init__(self, model_path: str, config: dict):
        """初始化推理器（加载模型）"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self.model = Transformer(
            num_layers=config["num_layers"],
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            input_vocab_size=config["input_vocab_size"],
            target_vocab_size=config["target_vocab_size"],
            device=self.device
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # 切换推理模式

        # 初始化分词器
        self.tokenizer = SimpleTokenizer(vocab_size=config["input_vocab_size"])
        print(f"模型加载完成: {model_path}（设备: {self.device}）")

    def infer(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """推理接口（支持单条/批量输入）"""
        if isinstance(text, list):
            return [self._infer_single(t) for t in text]
        return self._infer_single(text)

    def _infer_single(self, text: str) -> str:
        """单条文本推理"""
        # 1. 预处理：文本→Token ID
        inp_ids = self.tokenizer.text_to_ids(text, self.config["max_input_seq_len"]).to(self.device)
        enc_pad_mask = create_padding_mask(inp_ids, self.device)

        # 2. 编码器输出
        with torch.no_grad():
            enc_output = self.model.encoder(inp_ids, enc_pad_mask)

        # 3. 解码器自回归生成
        dec_input = torch.tensor([[self.tokenizer.bos]], dtype=torch.long, device=self.device)
        for _ in range(self.config["max_output_seq_len"] - 1):
            seq_len = dec_input.size(1)
            look_ahead_mask = create_look_ahead_mask(seq_len, self.device)
            with torch.no_grad():
                output = self.model(inp_ids, dec_input, enc_pad_mask, look_ahead_mask, enc_pad_mask)
            next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(1)
            dec_input = torch.cat([dec_input, next_token], dim=1)
            if next_token.item() == self.tokenizer.eos:
                break

        # 4. 后处理：Token ID→文本
        return self.tokenizer.ids_to_text(dec_input)

# --------------------------- 4. 部署模块（FastAPI服务） ---------------------------
def run_api_service(model_path: str, config: dict, host: str = "0.0.0.0", port: int = 8000):
    """启动API服务"""
    app = FastAPI(title="Transformer推理服务")
    # 全局加载推理器（避免重复加载模型）
    infer_engine = TransformerInfer(model_path, config)

    @app.post("/infer")
    def api_infer(input_text: str) -> dict:
        """推理API接口"""
        output_text = infer_engine.infer(input_text)
        return {"input": input_text, "output": output_text}

    @app.post("/batch_infer")
    def api_batch_infer(input_texts: List[str]) -> dict:
        """批量推理API接口"""
        output_texts = infer_engine.infer(input_texts)
        return {"inputs": input_texts, "outputs": output_texts}

    print(f"API服务启动: http://{host}:{port}/docs")
    import uvicorn
    uvicorn.run(app, host=host, port=port)

# --------------------------- 5. 全流程示例（训练→保存→加载→推理→部署） ---------------------------
if __name__ == "__main__":
    # 配置参数（训练/推理/部署必须一致）
    model_config = {
        "num_layers": 2,          # 编码器/解码器层数
        "d_model": 128,           # 嵌入维度
        "num_heads": 8,           # 注意力头数
        "d_ff": 512,              # 前馈网络维度
        "input_vocab_size": 1000, # 输入词汇表大小
        "target_vocab_size": 1000,# 输出词汇表大小
        "max_input_seq_len": 32,  # 输入最大长度
        "max_output_seq_len": 30, # 输出最大长度
        "lr": 1e-4                # 学习率
    }

    # 步骤1: 训练模型（5轮）并保存
    trainer = TransformerTrainer(model_config)
    model_path = trainer.train(num_epochs=5)  # 保存路径: models/transformer_5epochs.pth

    # 步骤2: 加载模型并推理
    infer_engine = TransformerInfer(model_path, model_config)
    print("\n===== 推理示例 =====")
    print(infer_engine.infer("test input"))  # 单条推理
    print(infer_engine.infer(["batch test 1", "batch test 2"]))  # 批量推理

    # 步骤3: 启动API服务（注释掉可跳过部署）
    # run_api_service(model_path, model_config)

    