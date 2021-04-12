from __future__ import unicode_literals, print_function, division


import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from torch.nn import Parameter
from torch.nn.utils import weight_norm
from utils.masking import TriangularCausalMask,ProbMask


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class Causal_Conv(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride=1, dilation=1, padding=None, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(Causal_Conv, self).__init__()
        padding = padding or (kernel_size-1) * dilation
        self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


        self.net = nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)
        # self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        # self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        # res = x if self.downsample is None else self.downsample(x)
        return out


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        """

        :param queries:张量，维度是(B,L,H,E),其中B是Batch_size,L是序列长度，H:header数，E：特征维度
        :param keys:张量，维度是(B,S,H,D),其中B是Batch_size,S是序列长度，H:header数，D：特征维度
        :param values:张量，维度是(B,S,H,D),其中B是Batch_size,S是序列长度，H:header数，D：特征维度
        :param attn_mask:掩码

        :return:content:返回加权得到的values
        A:注意力分布
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) # scores:(b,h,l,s)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1)) # A:(b,h,l,s)
        V = torch.einsum("bhls,bshd->blhd", A, values) # V:(b,l,h,d)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,kernel_size=3, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        # self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.query_projection = Causal_Conv(d_model,d_keys * n_heads,kernel_size)
        # self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        # self.key_projection = Causal_Conv(d_model,d_keys,kernel_size)
        self.key_projection = self.query_projection
        # self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Conv1d(d_model, d_values * n_heads,1)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1) # 张量：(B,L,H,d_keys)
        keys = self.key_projection(keys).view(B, S, H, -1) # 张量：(B,S,H,d_keys)
        values = self.value_projection(values).view(B, S, H, -1) # 张量：(B,S,H,d_keys)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        ) # out:(b,l,h,d),attn:(b,h,l,s)
        out = out.view(B, L, -1) # out:(b,l,h*d)

        return self.out_projection(out), attn #(b,l,d_model),attn:(b,h,l,s)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).double().to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.view(B, H, L_Q, -1)
        keys = keys.view(B, H, L_K, -1)
        values = values.view(B, H, L_K, -1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn

class multi_task_Attention(nn.Module):
    def __init__(self,d_model,attention_dropout=0.1, output_attention=False):
        super(multi_task_Attention).__init__()
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.W = Parameter(torch.Tensor(d_model,d_model))
        self.scale = d_model

    def forward(self,H):
        """

        :param H:张量，维度是(B,L,nums,d)，其中B为Batch_size，L为序列长度，nums为任务数，d为特征维度
        :return:
        """
        scale = self.scale
        y = torch.matmul(H,self.W) # y:(B,L,nums,d)
        scores = torch.matmul( y, torch.transpose(H,dim0=2,dim1=3))  # scores:(b,l,n,n)
        A = self.dropout(torch.softmax(scale * scores, dim=-1)) # A:(b,l,n,n)
        V = torch.matmul(A,H) # V:(b,l,n,d)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ResidualAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, layer_nums = 0, attention_dropout=0.1, output_attention=True):
        super(ResidualAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.layer_nums = layer_nums
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask,A):
        """

        :param queries:张量，维度是(B,L,H,E),其中B是Batch_size,L是序列长度，H:header数，E：特征维度
        :param keys:张量，维度是(B,S,H,D),其中B是Batch_size,S是序列长度，H:header数，D：特征维度
        :param values:张量，维度是(B,S,H,D),其中B是Batch_size,S是序列长度，H:header数，D：特征维度
        :param attn_mask:掩码
        :param A:上一层得到的注意力分布

        :return:content:返回加权得到的values
        A:注意力分布

        A_{n}=\frac{Q_nK_n^T}{\sqrt{d}}+A_{n-1}
        Attention(Q_n,K_n,V_n)=softmax(A_{n})V_n
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) # scores:(b,h,l,s)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        if self.layer_nums == 0:
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
        else:
            A = self.dropout(torch.softmax(scale * scores + A, dim=-1))  # A:(b,h,l,s)

        V = torch.einsum("bhls,bshd->blhd", A, values) # V:(b,l,h,d)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)











