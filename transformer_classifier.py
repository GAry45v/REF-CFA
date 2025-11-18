# 文件: transformer_classifier.py
import torch
import torch.nn as nn

# class ModifiedMultiheadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, batch_first=True):
#         super(ModifiedMultiheadAttention, self).__init__()
#         self.batch_first = batch_first
#         self.num_heads = num_heads  # 显式添加 num_heads 属性
#         self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        
#         # 添加 qkv_same_embed_dim 属性
#         self._qkv_same_embed_dim = True  # 需要将其设置为 True，因为 MultiheadAttention 默认期望它们相同
    
#     def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        # 计算多头自注意力
        # attn_output, attn_weights = self.multihead_attention(query, key, value, key_padding_mask, need_weights, attn_mask)
        
        # # 返回注意力权重
        # return attn_output, attn_weights




# class ClassificationTransformer(nn.Module):
#     def __init__(self, input_dim, projection_dim, num_heads, num_layers, num_classes):
#         """
#         初始化Transformer分类器 (稳定版)。
#         """
#         super(ClassificationTransformer, self).__init__()
        
#         self.projection = nn.Linear(input_dim, projection_dim)
        
#         encoder_layer = nn.TransformerEncoderLayer(d_model=projection_dim, nhead=num_heads, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         self.fc = nn.Linear(projection_dim, num_classes)

#     def forward(self, x):
#         """
#         前向传播。
#         """
#         # 1. 应用投影层进行降维
#         projected_x = self.projection(x)
        
#         # 2. 通过Transformer编码器 (直接、稳定地调用)
#         encoded_x = self.transformer_encoder(projected_x)
        
#         # 3. 使用序列的平均值进行池化
#         pooled_x = encoded_x.mean(dim=1)
        
#         # 4. 通过全连接层得到分类结果
#         output = self.fc(pooled_x)
        
#         # 返回编码后的序列和第一个自注意力层的静态权重用于分析
#         # 注意：这里返回的是 encoded_x，它对于分析范数更有用
#         return output, encoded_x

# 文件: transformer_classifier.py
import torch
import torch.nn as nn

class ClassificationTransformer(nn.Module):
    def __init__(self, input_dim, projection_dim, num_heads, num_layers, num_classes, seq_length):
        """
        初始化Transformer分类器 (升级版：使用 CLS Token 和位置编码)。
        """
        super(ClassificationTransformer, self).__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        # --- 新增 1: CLS Token ---
        # 这是一个可学习的参数，作为序列的“班长”
        self.cls_token = nn.Parameter(torch.randn(1, 1, projection_dim))
        
        # --- 新增 2: 位置编码 ---
        # 为每个位置（包括CLS Token的位置）创建一个可学习的编码
        self.positional_embedding = nn.Parameter(torch.randn(1, seq_length + 1, projection_dim))
        
        self.projection = nn.Linear(input_dim, projection_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=projection_dim, nhead=num_heads, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(projection_dim, num_classes)

    def forward(self, x):
        """
        前向传播。
        x 的形状: [batch_size, seq_length, input_dim]
        """

        x_normalized = self.input_norm(x)
        # 1. 应用投影层进行降维
        projected_x = self.projection(x_normalized)
        
        # 2. 准备 CLS Token 并拼接到序列最前面
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # 为批次中的每个样本复制一份CLS Token
        x_with_cls = torch.cat((cls_tokens, projected_x), dim=1) # 拼接到序列的第0个位置
        
        # 3. 添加位置编码
        x_with_pos = x_with_cls + self.positional_embedding
        
        # 4. 通过 Transformer 编码器
        encoded_x = self.transformer_encoder(x_with_pos)
        
        # 5. 只取出 CLS Token 对应的输出 (序列的第0个位置)
        cls_output = encoded_x[:, 0]
        
        # 6. 通过全连接层得到分类结果
        output = self.fc(cls_output)
        
        # 返回最终输出和编码后的完整序列 (用于 Grad-CAM 分析)
        return output, encoded_x


# class ClassificationTransformer(nn.Module):
#     def __init__(self, input_dim, projection_dim, num_heads, num_layers, num_classes):
#         """
#         初始化Transformer分类器。
#         """
#         super(ClassificationTransformer, self).__init__()
        
#         self.projection = nn.Linear(input_dim, projection_dim)
        
#         # 使用自定义的 MultiheadAttention 层
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=projection_dim, 
#             nhead=num_heads, 
#             batch_first=True  # 确保 batch_first=True
#         )
        
#         # 替换原有的 self-attention 层
#         encoder_layer.self_attn = ModifiedMultiheadAttention(
#             embed_dim=projection_dim,
#             num_heads=num_heads,
#             batch_first=True  # 传递 batch_first 参数
#         )

#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         self.fc = nn.Linear(projection_dim, num_classes)

#     def forward(self, x):
#         """
#         前向传播。
#         """
#         # 1. 应用投影层进行降维
#         projected_x = self.projection(x)
        
#         # 2. 通过Transformer编码器 (直接、稳定地调用)
#         encoded_x = self.transformer_encoder(projected_x)
        
#         # 3. 使用序列的平均值进行池化
#         pooled_x = encoded_x.mean(dim=1)
        
#         # 4. 通过全连接层得到分类结果
#         output = self.fc(pooled_x)
        
#         # 返回编码后的序列
#         return output, encoded_x


    