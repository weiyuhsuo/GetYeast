import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import yaml
import os
import logging
import math
from torch.nn.init import trunc_normal_

logger = logging.getLogger(__name__)

# ========== Softplus激活函数模块 ==========
class softSoftplus(nn.Module):
    """
    Softplus激活函数：softplus(x) = (1/β) * log(1 + exp(βx))
    
    作用：
    - 平滑的ReLU替代方案，处处可导
    - 输出始终为正数，适合表达量预测（表达量不能为负）
    - beta控制激活曲线的陡峭程度
    - threshold用于数值稳定性（当输入很大时用线性近似）
    
    参数：
    - beta: 控制增长率的超参数（默认1）
    - threshold: 数值稳定性阈值（默认20）
    """
    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.beta = beta       # 控制激活函数陡峭程度的参数
        self.threshold = threshold  # 数值稳定性阈值
        
    def forward(self, x):
        """
        前向传播
        
        参数：
        - x: 输入张量
        
        返回：
        - 经过softplus激活后的张量
        """
        return torch.nn.functional.softplus(x, beta=self.beta, threshold=self.threshold)

class RegionEmbed(nn.Module):
    """
    区域嵌入模块 - 将原始特征映射到高维向量空间
    
    作用：
    - 将原始特征（motif + accessibility + condition）映射到768维向量
    - 这是模型的第一层，为后续的Transformer编码做准备
    - 使用线性变换 + 截断正态分布初始化
    - 特征维度由配置文件指定（SC: 545维, KM: 242维等）
    
    输入形状：(batch_size, sequence_length, num_features)
    输出形状：(batch_size, sequence_length, 768)
    """
    def __init__(self, config: Dict):
        super().__init__()
        # 输入特征维度：由配置文件指定（例如SC: 470 motif + 1 accessibility + 74 condition = 545）
        self.num_features = config['num_features']
        # 嵌入维度：768（BERT-base的标准维度）
        self.embed_dim = config['embed_dim']
        
        # 验证输入特征数必须大于0
        assert self.num_features > 0, f"RegionEmbed输入特征数必须大于0，实际为{self.num_features}"
        
        # 线性投影层：num_features维 -> 768维
        self.embed = nn.Linear(self.num_features, self.embed_dim)
        
        # 使用截断正态分布初始化权重（标准差0.02）
        # 这是BERT和Vision Transformer常用的初始化方法
        trunc_normal_(self.embed.weight, std=0.02)
        if self.embed.bias is not None:
            # 偏置初始化为0
            nn.init.constant_(self.embed.bias, 0)
            
    def forward(self, x):
        """
        前向传播：执行特征嵌入
        
        参数：
        - x: 输入张量，形状为 (B, N, num_features)，其中B是批次大小，N是序列长度
        
        返回：
        - 嵌入后的张量，形状为 (B, N, 768)
        """
        # 验证输入维度
        assert x.shape[-1] == self.num_features, \
            f"RegionEmbed输入特征维数应为{self.num_features}，实际为{x.shape[-1]}"
        return self.embed(x)

class Attention(nn.Module):
    """
    多头自注意力机制（Multi-Head Self-Attention）
    
    作用：
    - Transformer的核心组件，让模型能够关注序列中不同位置的信息
    - 多头机制允许模型同时关注不同类型的依赖关系
    - 每个头学习不同的表示子空间
    
    计算公式：
    Attention(Q, K, V) = softmax(QK^T / √d_k) * V
    
    参数：
    - dim: 输入维度（通常为768）
    - num_heads: 注意力头数（通常为12）
    - qkv_bias: 是否在QKV投影中添加偏置
    - attn_drop: 注意力权重的dropout率
    - proj_drop: 输出投影的dropout率
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个头的维度（768/12=64）
        
        # 缩放因子：1/√d_k，用于防止点积值过大导致softmax饱和
        self.scale = qk_scale or head_dim ** -0.5

        # 将Q、K、V投影合并为一个线性层（768 -> 2304，即768*3）
        # 这是效率优化，一次计算三个投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力权重dropout
        self.proj = nn.Linear(dim, dim)  # 输出投影层
        self.proj_drop = nn.Dropout(proj_drop)  # 输出dropout

    def forward(self, x):
        """
        前向传播：计算多头注意力
        
        参数：
        - x: 输入张量，形状为 (B, N, C)，其中B=批次，N=序列长度，C=特征维度
        
        返回：
        - 注意力输出，形状为 (B, N, C)
        """
        B, N, C = x.shape  # B=批次大小，N=序列长度，C=特征维度768
        
        # 计算QKV并重塑为多头形式
        # qkv形状：(B, N, 3, 12, 64) -> 转置为 (3, B, 12, N, 64)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别提取Q、K、V

        # 计算注意力分数：Q @ K^T / √d_k
        # attn形状：(B, 12, N, N)，表示每个头对所有位置的注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 应用softmax得到归一化的注意力权重
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # 应用dropout

        # 用注意力权重对V进行加权求和
        # 然后将多头结果拼接并投影
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 重塑回 (B, N, C)
        x = self.proj(x)  # 输出投影
        x = self.proj_drop(x)  # 输出dropout
        return x

class Mlp(nn.Module):
    """
    多层感知机（MLP/Feed-Forward Network）
    
    作用：
    - Transformer块中的前馈神经网络
    - 为每个位置独立地应用非线性变换
    - 通常包含两层：扩展层（维度扩大4倍）+ 压缩层（恢复原维度）
    
    结构：
    - 线性层1：768 -> 3072（扩展）
    - GELU激活函数
    - Dropout
    - 线性层2：3072 -> 768（压缩）
    - Dropout
    
    这是"Point-wise Feed-Forward Networks"的实现
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 默认输出维度等于输入维度
        out_features = out_features or in_features
        # 默认隐藏层是输入维度的4倍（Transformer标准配置）
        hidden_features = hidden_features or in_features
        
        # 第一层：扩展到hidden_features维度
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数：GELU（Gaussian Error Linear Unit）
        # 比ReLU更平滑，在Transformer中表现更好
        self.act = act_layer()
        # 第二层：压缩回输出维度
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)  # Dropout正则化

    def forward(self, x):
        """
        前向传播：执行MLP变换
        
        参数：
        - x: 输入张量，形状为 (B, N, C)
        
        返回：
        - MLP输出，形状为 (B, N, C)
        """
        x = self.fc1(x)      # (B, N, 768) -> (B, N, 3072)
        x = self.act(x)      # GELU激活
        x = self.drop(x)     # Dropout正则化
        x = self.fc2(x)      # (B, N, 3072) -> (B, N, 768)
        x = self.drop(x)     # Dropout正则化
        return x

class Block(nn.Module):
    """
    Transformer编码器块
    
    完整结构（Pre-LayerNorm）：
    1. Layer Norm -> Multi-Head Attention -> 残差连接
    2. Layer Norm -> MLP -> 残差连接
    
    作用：
    - Transformer的基础单元，包含自注意力机制和前馈网络
    - 每个块都允许模型学习序列内的复杂依赖关系
    - 使用残差连接避免梯度消失问题
    - 使用Layer Norm提供稳定的训练
    
    关键特性：
    - Pre-LayerNorm：先归一化再计算，更稳定
    - 残差连接：x = x + f(norm(x))
    - Dropout正则化：防止过拟合
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # 第一个Layer Norm：用于注意力层前的归一化
        self.norm1 = norm_layer(dim)
        
        # 多头自注意力层
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        # Drop path：随机深度（Stochastic Depth），这里简化为Identity（不使用）
        self.drop_path = nn.Identity()
        
        # 第二个Layer Norm：用于MLP前的归一化
        self.norm2 = norm_layer(dim)
        
        # MLP的隐藏层维度：通常是输入维度的4倍（mlp_ratio=4）
        mlp_hidden_dim = int(dim * mlp_ratio)  # 768 * 4 = 3072
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        前向传播：执行一个完整的Transformer块
        
        参数：
        - x: 输入张量，形状为 (B, N, C)
        
        返回：
        - Transformer块输出，形状为 (B, N, C)
        
        流程：
        1. x -> LayerNorm -> Attention -> 残差连接 -> x'
        2. x' -> LayerNorm -> MLP -> 残差连接 -> x''
        """
        # 第一个子层：自注意力
        x = x + self.drop_path(self.attn(self.norm1(x)))  # 残差连接
        # 第二个子层：MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 残差连接
        return x

class GETTransformer(nn.Module):
    """
    GET Transformer编码器
    
    作用：
    - 由12个Transformer块堆叠而成（类似BERT-base）
    - 逐层提取和传递序列信息
    - 最后一层使用Layer Norm进行最终归一化
    
    架构：
    - 输入嵌入层 + 12个Transformer块 + 输出归一化
    - 每一层都进行自注意力计算和MLP变换
    - 通过残差连接让信息逐层传递
    
    参数量：
    - 总共约106M参数（类似BERT-base规模）
    """
    def __init__(self, config: Dict):
        super().__init__()
        # 嵌入维度：768
        self.embed_dim = config['embed_dim']
        # 注意力头数：12
        self.num_heads = config['num_heads']
        # Transformer层数：12
        self.num_layers = config['num_layers']
        # Dropout率：0.1
        self.dropout = config['dropout']
        
        # 构建12个Transformer块的堆叠
        # 每个块都是独立的：自注意力 + MLP
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim,      # 768
                num_heads=self.num_heads, # 12头
                mlp_ratio=4,             # MLP隐藏层是输入层的4倍（3072）
                qkv_bias=True,           # 使用QKV偏置
                drop=self.dropout,       # Dropout率0.1
                attn_drop=self.dropout,  # 注意力Dropout率0.1
                act_layer=nn.GELU,       # 使用GELU激活函数
                norm_layer=nn.LayerNorm  # 使用Layer Norm
            )
            for i in range(self.num_layers)  # 重复12次
        ])
        
        # 最终归一化层（Pre-LN架构的最后一层）
        self.norm = nn.LayerNorm(self.embed_dim)
        
        # 初始化所有权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """
        权重初始化函数
        
        策略：
        - 线性层：截断正态分布（std=0.02），偏置初始化为0
        - Layer Norm：偏置=0，权重=1
        """
        if isinstance(m, nn.Linear):
            # 线性层权重：使用截断正态分布初始化（标准差0.02）
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 偏置初始化为0
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # LayerNorm偏置：初始化为0
            nn.init.constant_(m.bias, 0)
            # LayerNorm权重：初始化为1
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        前向传播：执行12层Transformer编码
        
        参数：
        - x: 输入张量，形状为 (B, N, 768)
        
        返回：
        - 编码后的张量，形状为 (B, N, 768)
        
        流程：
        - 通过12个Transformer块逐层处理
        - 最后一层使用Layer Norm归一化
        """
        # 逐层通过Transformer块
        for blk in self.blocks:
            x = blk(x)  # 每个块：Attention + MLP + 残差连接
        
        # 最终归一化
        x = self.norm(x)
        return x

class ExpressionHead(nn.Module):
    """
    表达预测头（Expression Prediction Head）
    
    作用：
    - 将Transformer编码器的输出映射到表达量预测
    - 预测正链和负链的表达水平（输出2维）
    - 简单的线性层，将768维特征压缩到2维
    
    结构：
    - 线性层：768 -> 2
    - 使用特殊的初始化：权重乘以0.001，使初始预测值很小
    - 这样模型在训练初期不会产生太大的预测值
    
    输出：
    - (batch_size, sequence_length, 2)：正链表达量和负链表达量
    """
    def __init__(self, config: Dict):
        super().__init__()
        # 输入维度：768（编码器输出维度）
        self.embed_dim = config['embed_dim']
        # 输出维度：2（正链和负链的表达量）
        self.output_dim = config['output_dim']
        
        # 单层线性投影：768维 -> 2维
        self.head = nn.Linear(self.embed_dim, self.output_dim)
        
        # 使用特殊的初始化方式
        # 1. 先用截断正态分布初始化（std=0.02）
        trunc_normal_(self.head.weight, std=0.02)
        # 2. 将权重缩小1000倍（乘以0.001），避免初始预测过大
        self.head.weight.data.mul_(0.001)
        if self.head.bias is not None:
            # 偏置也缩小1000倍
            self.head.bias.data.mul_(0.001)
    
    def forward(self, x):
        """
        前向传播：预测表达量
        
        参数：
        - x: 输入张量，形状为 (B, N, 768)
        
        返回：
        - 表达量预测，形状为 (B, N, 2)
        """
        return self.head(x)

class LoRALinear(nn.Module):
    """
    LoRA（Low-Rank Adaptation）线性层适配器
    
    作用：
    - 参数高效微调技术：不直接训练大型模型的权重，而是训练小的低秩矩阵
    - 冻结原始模型权重（节省内存且避免破坏预训练知识）
    - 只训练两个小矩阵A和B，参数量远小于原模型
    
    原理：
    - 原始变换：W * x
    - LoRA变换：W * x + (B * A * x) * (alpha/rank)
    - 其中A的形状为(rank, in_features)，B的形状为(out_features, rank)
    - 当rank << min(in_features, out_features)时，参数量大幅减少
    
    优点：
    - 显著减少可训练参数（例如：768*768需要589K参数，rank=4只需6K参数）
    - 保持模型性能的同时实现快速微调
    - 非常适合迁移学习场景
    """

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: int = 16):
        super().__init__()
        self.base = base              # 原始线性层（将被冻结）
        self.rank = rank              # LoRA的秩（低秩矩阵的维度）
        self.alpha = alpha            # LoRA的缩放因子

        # 冻结原始权重：不参与梯度更新
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # LoRA 参数矩阵
        # A: (rank, in_features) - 降维矩阵
        self.lora_A = nn.Parameter(torch.zeros(rank, self.base.in_features))
        # B: (out_features, rank) - 升维矩阵
        self.lora_B = nn.Parameter(torch.zeros(self.base.out_features, rank))

        # 初始化LoRA矩阵
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # A用Kaiming初始化
        nn.init.zeros_(self.lora_B)  # B初始化为0（训练从零开始）

        # LoRA的缩放因子：alpha/rank
        self.scaling = self.alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：基础变换 + LoRA变换
        
        参数：
        - x: 输入张量
        
        返回：
        - 原始输出 + LoRA调整
        
        计算过程：
        - result = W * x（原始输出）
        - lora_update = (x @ A^T @ B^T) * (alpha/rank)（LoRA调整）
        - 最终输出 = result + lora_update
        """
        result = self.base(x)  # 原始变换
        lora_update = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling  # LoRA调整
        return result + lora_update

    @property
    def weight(self):
        """返回基础权重，用于兼容性"""
        return self.base.weight
    
    @property
    def bias(self):
        """返回基础偏置，用于兼容性"""
        return self.base.bias
    
    @property
    def in_features(self):
        """返回输入特征数"""
        return self.base.in_features
    
    @property
    def out_features(self):
        """返回输出特征数"""
        return self.base.out_features

class YeastModel(nn.Module):
    """
    酵母基因表达预测模型（Yeast Gene Expression Prediction Model）
    
    完整架构：
    1. 区域嵌入层：将原始特征（由配置指定维度）映射到768维
    2. CLS Token：学习序列级别的全局表示
    3. Transformer编码器：12层，提取序列依赖关系
    4. 表达预测头：输出正链和负链的表达量
    
    数据流：
    Input (B, 1, num_features) 
    -> Region Embed (B, 1, 768)
    -> + CLS Token (B, 2, 768)
    -> Transformer × 12 (B, 2, 768)
    -> Remove CLS (B, 1, 768)
    -> Output Head (B, 1, 2)
    -> Softplus (B, 1, 2)
    
    参数量：约106M（类似BERT-base）
    
    注意：特征维度由配置文件指定（SC: 545维, KM: 242维等）
    
    可选功能：
    - LoRA微调：只训练少量低秩参数，实现参数高效微调
    """
    def __init__(self, cfg: Dict, use_lora=False, lora_rank=4, lora_alpha=16, lora_layers: Optional[List[str]] = None):
        super().__init__()
        # LoRA配置参数
        self.use_lora = use_lora           # 是否使用LoRA微调
        self.lora_rank = lora_rank         # LoRA的秩（默认4）
        self.lora_alpha = lora_alpha       # LoRA的缩放因子（默认16）
        
        # 区域嵌入层：将358维原始特征映射到768维向量空间
        self.region_embed = RegionEmbed(cfg['region_embed'])
        
        # Transformer编码器：12层，768维，12个注意力头
        self.encoder = GETTransformer(cfg['encoder'])
        
        # 表达预测头：将768维特征映射到2维输出（正链和负链的表达量）
        self.head_exp = ExpressionHead(cfg['head_exp'])
        
        # CLS Token（Classification Token）：学习全局序列表示
        # 这是一个可学习的参数，用于聚合整个序列的信息
        # 形状：(1, 1, 768)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg['encoder']['embed_dim']))
        
        # 损失函数：均方误差损失（Mean Squared Error）
        self.loss_fn = nn.MSELoss(reduction='mean')
        
        # 初始化所有权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        权重初始化函数
        
        策略：
        - 线性层：截断正态分布初始化（std=0.02），偏置=0
        - Layer Norm：权重=1.0，偏置=0
        """
        if isinstance(m, nn.Linear):
            # 线性层权重：截断正态分布初始化
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 线性层偏置：初始化为0
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # Layer Norm偏置：初始化为0
            nn.init.constant_(m.bias, 0)
            # Layer Norm权重：初始化为1（保持归一化效果）
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播：执行完整的基因表达预测
        
        参数：
        - x: 输入字典，包含'motif_features'字段
           形状：(B, 1, num_features)，特征维度由配置指定
        
        返回：
        - predictions: 表达量预测，形状为 (B, 1, 2)
          第一维是正链表达量，第二维是负链表达量
        
        完整流程：
        1. 输入原始特征（维度由配置指定）
        2. 区域嵌入：num_features -> 768维
        3. 添加CLS token：序列变长（1个token -> 2个token）
        4. Transformer编码：提取序列依赖关系
        5. 移除CLS token：保留原始序列token
        6. 预测头：768 -> 2维
        7. Softplus激活：确保输出为正数（表达量不能为负）
        """
        # 提取特征输入（维度由配置指定，例如SC: 545维 = 470 motif + 1 accessibility + 74 condition）
        all_feat = x['motif_features']  # 形状：(B, 1, num_features)
        
        # 区域嵌入：将特征映射到768维向量空间
        x = self.region_embed(all_feat)  # 形状：(B, 1, 768)
        
        # 添加CLS token（分类令牌）
        # CLS token是一个可学习的特殊token，用于学习全局序列表示
        B, N, C = x.shape  # B=批次大小，N=序列长度（1），C=特征维度（768）
        cls_tokens = self.cls_token.expand(B, -1, -1)  # 扩展到批次大小
        x = torch.cat((cls_tokens, x), dim=1)  # 拼接：[CLS, SEQ] 形状：(B, 2, 768)
        
        # Transformer编码：通过12层Transformer块提取序列依赖关系
        # CLS token会与序列token进行注意力交互，学习全局信息
        x = self.encoder(x)  # 形状：(B, 2, 768)
        
        # 移除CLS token：我们只需要序列token的表示
        x = x[:, 1:]  # 取出序列部分，形状：(B, 1, 768)
        
        # 表达预测头：将768维特征映射到2维输出（正链和负链的表达量）
        predictions = self.head_exp(x)  # 形状：(B, 1, 2)
        
        # Softplus激活函数：确保预测的表达量为正数
        # Softplus(x) = log(1 + exp(x))，输出始终大于0
        # 这符合生物学意义：基因表达量不能为负
        predictions = nn.Softplus()(predictions)
        
        return predictions

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失函数
        
        参数：
        - predictions: 模型预测的表达量，形状为 (B, N, 2)
          第一维是正链表达量，第二维是负链表达量
        - targets: 真实表达量，形状为 (B, N, 2)
        
        返回：
        - loss: 均方误差损失（MSE Loss）
        
        注：
        - 使用MSE损失是回归任务的经典选择
        - 对所有样本和维度求平均（reduction='mean'）
        """
        return self.loss_fn(predictions, targets)

    def _inject_lora(self, module: nn.Module, rank: int, alpha: int):
        """
        递归地将模块中的所有Linear层替换为LoRA版本
        
        作用：
        - 深度遍历模块树，找到所有nn.Linear层
        - 将每个Linear层替换为LoRALinear包装器
        - 实现参数高效的微调
        
        参数：
        - module: 要处理的模块（通常是encoder或head_exp）
        - rank: LoRA的秩
        - alpha: LoRA的缩放因子
        """
        # 遍历模块的所有子模块
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # 如果子模块是Linear层，替换为LoRALinear
                setattr(module, name, LoRALinear(child, rank, alpha))
            else:
                # 如果不是Linear层，递归处理它的子模块
                self._inject_lora(child, rank, alpha)
    
    def inject_lora_adapters(self, lora_layers: Optional[List[str]] = None):
        """
        注入LoRA适配器到指定层
        
        作用：
        - 为模型的指定层（如encoder、head_exp）注入LoRA适配器
        - 实现参数高效的微调，只需要训练少量参数
        - 适合在预训练模型上进行任务特定的微调
        
        参数：
        - lora_layers: 要应用LoRA的层名称列表
                      如果不指定，默认应用到encoder和head_exp
        
        使用场景：
        - 在预训练模型上进行微调时使用
        - 节省显存和计算资源
        - 只需要训练rank×2的额外参数（相对于原始参数）
        """
        # 如果不使用LoRA，直接返回
        if not self.use_lora:
            return
            
        logger.info(f"注入 LoRA 适配器 (rank={self.lora_rank}, alpha={self.lora_alpha})")
        
        # 确定目标层：如果未指定，使用默认的encoder和head_exp
        target_layers = lora_layers if lora_layers else ['encoder', 'head_exp']
        
        # 为每个目标层注入LoRA
        for layer_name in target_layers:
            module = getattr(self, layer_name, None)
            if module is not None:
                # 递归替换该层中的所有Linear层为LoRA版本
                self._inject_lora(module, self.lora_rank, self.lora_alpha)
                logger.info(f"已为 {layer_name} 注入 LoRA 适配器")
            else:
                logger.warning(f"LoRA 目标层 {layer_name} 不存在，跳过注入")

def create_model(config_path: str):
    """
    从配置文件创建模型实例
    
    功能：
    - 读取YAML配置文件
    - 解析模型架构参数和LoRA配置
    - 创建并返回YeastModel实例
    
    参数：
    - config_path: 配置文件路径（通常是yeast_training.yaml）
    
    返回：
    - YeastModel实例
    
    配置文件结构：
    - model.model: 模型架构参数（region_embed, encoder, head_exp等）
    - training.use_lora: 是否使用LoRA
    - training.lora_rank: LoRA的秩
    - training.lora_alpha: LoRA的缩放因子
    - training.lora_layers: 应用LoRA的层列表
    """
    # 读取YAML配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建模型实例
    # 传入model.model字段作为模型配置
    return YeastModel(
        cfg=config['model']['model'],                       # 模型架构配置
        use_lora=config['training'].get('use_lora', False), # 是否使用LoRA
        lora_rank=config['training'].get('lora_rank', 4),   # LoRA秩
        lora_alpha=config['training'].get('lora_alpha', 16), # LoRA缩放因子
        lora_layers=config['training'].get('lora_layers', None)  # LoRA应用层
    )

# YeastModel的cfg参数应为如下结构：
# cfg = {
#   'region_embed': {'num_features': 545, 'embed_dim': 768},  # SC: 470 motif + 1 accessibility + 74 condition
#   # 或 {'num_features': 242, 'embed_dim': 768},  # KM: 235 motif + 1 accessibility + 6 condition
#   'encoder': {'embed_dim': 768, 'num_heads': 12, 'num_layers': 12, 'dropout': 0.1},
#   'head_exp': {'embed_dim': 768, 'output_dim': 2},  # 正负链表达
#   ...
# } 