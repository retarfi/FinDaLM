import re

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, moe_type: str, intermediate_size: int, num_experts: int):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, num_experts)
        if re.match(r"top\d+", moe_type):
            self._k = int(re.match(r"top(\d+)", moe_type).group(1))
            self.moe_type = "top"
        elif moe_type == "dense":
            self.moe_type = moe_type
        else:
            raise ValueError()

    def forward(self, inputs):
        hidden_states = self.dense(inputs)
        weight = self._topk(hidden_states)
        return weight

    def _topk(self, inputs):
        if self.moe_type == "top":
            bool_topk = (
                F.one_hot(
                    inputs.topk(self._k, sorted=True).indices,
                    num_classes=inputs.size(2),
                )
                .sum(dim=2)
                .to(torch.bool)
            )
            weight = torch.where(bool_topk, inputs, -torch.inf).softmax(dim=2)
        elif self.moe_type == "dense":
            weight = inputs.softmax(dim=2)
        return weight


class Skipper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs
