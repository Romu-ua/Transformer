import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple # コードの可読性のため

class ScaledDotProductAttention(nn.Module):
	"""
	Scaled Dot-Product Attention は "Attention is all you need"で提案された
	queryと全てのkeyの内積を計算し、次元の平方根で割ってスケールを調整する
	その後ソフトマックスを適用し、valueに対する重みを得る

	Args: dim, mask
		dim (int): attentionの次元
		mask (torch.Tensor): マスクするインデックスを含むテンソル

	Inputs: query, key, value, mask
		query (batch, q_len, d_model): 行列積を計算後のq
		key (batch, k_len, d_model): 行列積を計算後のk
		value (batch, v_len, d_model): 行列を計算後のv
		mask: マスクするインデックスを含むテンソル

	Returns: context, attn
		context: アテンション機構から得られる文脈ベクトル
		attn: エンコーダの出力に対するアテンションを含むテンソル attn = softmax(QK^T/√d)
	"""
	def __init__(self, dim: int):
		super().__init__() # 元コードと異なるが、python3ではこちらの方がシンプル
		self.sqrt_dim = np.sqrt(dim)

	def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) ->Tuple[Tensor, Tensor]:
		score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

		if mask is not None:
			score.masked_fill_(mask.view(score.size()), -float('Inf'))

		attn = F.softmax(score, -1)
		context = torch.bmm(attn, value)
		return context, attn


if __name__ == "__main__":
	# demo実行

	batch_size = 2
	q_len = 10
	k_len = 15
	d_model = 64

	query = torch.randn(batch_size, q_len, d_model)
	key = torch.randn(batch_size, k_len, d_model)
	value = torch.randn(batch_size, k_len, d_model)

	mask = torch.zeros(batch_size, q_len, k_len)
	mask[:, :, 5:] = 1 # ６番目以降をマスク
	mask = mask.bool()

	attention = ScaledDotProductAttention(d_model)
	context, attn = attention(query, key, value, mask)

	print("Context Shape: ", context.shape)
	print("Attention Shape: ", attn.shape)

	print("Context", context)


	# masked_fill_()について
	print("-------maked_fill_()について-------")
	x = torch.randn(4, 4)
	print("Before masked_fill_: ")
	print(x)

	mask = torch.tensor([
		[1, 0, 1, 0],
		[0, 1, 0, 1],
		[1, 0, 1, 0],
		[0, 1, 0, 1],
	], dtype=torch.bool)

	x.masked_fill_(mask, -float('inf'))

	print("After masked_fill_:")
	print(x)

	# masked_fill_とmasked_fillの違い
	# masked_fill_ : 元のTensorを変更する
	# masked_fill  : 元のTensorを変更せずにあたらしいTensorを返す

	# なぜ-infを入れるのか？
	# 	softmaxはe^xを計算するから、e^{-inf} = 0になり、マスクの部分が無視されるから
