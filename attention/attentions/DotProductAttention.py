import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple

class DotProductAttention(nn.Module):
	"""
	softmax(query * value^T)のみの計算
	"""
	def __init__(self, hidden_dim):
		super().__init__()

	def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
		batch_size, hidden_dim, input_size = query.size(0), query.size(2), query.size(1)

		score = torch.bmm(query, value.transpose(1, 2))
		attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
		context = torch.bmm(attn, value)

		return context, attn
