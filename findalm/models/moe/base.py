import torch.nn as nn

MOE_TYPES: tuple[str] = ("top2-skip", "top2", "top1", "dense")


def confirm_same_weights(model1: nn.Module, model2: nn.Module) -> bool:
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True
