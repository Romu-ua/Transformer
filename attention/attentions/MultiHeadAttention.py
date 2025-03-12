import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple

from ScaledDotProductAttention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
	"""
	"Attention is all you need"で提案された multi-head attention
	単一のアテンション機構 d_model 次元の key, value, query に適用するのではなく、
	これらを h 回にわたって異なる学習可能な線形変換(W_k, W_v, W_q)を通して、d_head 次元へ投影する

	この結果得られた h 個の head の出力を結合し、最終的に線形変換 W_o を適用することで、最終的な出力を得る
	Multi-head attention により、
	異なる表現サブスペースや異なる位置情報に対して同時に注意を向けることが可能になる

	Args:
		d_model (int): key, value, query の次元 (default: 512)
		num_heads (int): attention head の数 (default: 8)

	Inputs: query, key, value, mask
		query (batch, q_len, d_model)
		key (batch, k_len, d_model)
		value (batch, v_len, d_model)
			これらは以下３通りのソースから供給される
			case1: デコーダ前のレイヤー
			case2: 入力埋め込み
			case3: 出力埋め込み
		mask: マスクする index を含むテンソル
			  マスクを適用することで、不要な部分をゼロにする

	Returns: output, attn
		output (batch, output_len, dimensions): アテンションを適用した後の出力特徴量を含むテンソル
		attn (batch * num_heads, v_len): エンコーダの出力に対するアテンションを含むテンソル
	"""

	def __init__(self, d_model: int = 512, num_heads: int = 8):
		super().__init__()

		assert d_model % num_heads == 0, "d_model % num_heads should be zero."

		self.d_head = int(d_model / num_heads)
		self.num_heads = num_heads
		self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
		self.query_proj = nn.Linear(d_model, self.d_head * num_heads)  # 元の特徴量の次元数（入力）→ Multi-Head Attention 用に変換する次元数（出力）
		self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
		self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

	def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
		batch_size = value.size(0)

		query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
		key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)
		value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)

		query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
		key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)
		value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)

		if mask is not None:
			mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

		context, attn = self.scaled_dot_attn(query, key, value, mask)

		context = context.view(self.num_heads, batch_size, -1, self.d_head)
		context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)

		return context, attn

def main():
	batch_size = 2
	seq_len_q = 10
	seq_len_kv = 15
	d_model = 512
	num_heads = 8

	query = torch.randn(batch_size, seq_len_q, d_model)
	key = torch.randn(batch_size, seq_len_kv, d_model)
	value = torch.randn(batch_size, seq_len_kv, d_model)

	mask = torch.zeros(batch_size, seq_len_q, seq_len_kv)
	mask[:, :, 5:] = 1
	mask = mask.bool()

	mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

	output, attn = mha(query, key, value, mask)

	print("Output shape:", output.shape)  # (batch_size, seq_len_q, d_model)
	print("Attention shape:", attn.shape) # (batch_size * num_heads, seq_len_kv)

if __name__ == "__main__":
	main()
