import pytest
import torch
import torch.nn.functional as F

from findalm.models.moe.router import Router

top2_w: torch.tensor = torch.tensor([-torch.inf, -torch.inf, 3.0, 4.0]).softmax(dim=0)
dense_w: torch.tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).softmax(dim=0)


@pytest.mark.parametrize(
    "moe_type, expected",
    [
        ("top1", F.one_hot(torch.tensor([[3, 2], [0, 0]])).to(torch.float)),
        (
            "top2",
            torch.stack(
                [
                    torch.stack([top2_w, top2_w[[0, 2, 3, 1]]]),
                    torch.stack([top2_w[[3, 2, 1, 0]], top2_w[[3, 1, 0, 2]]]),
                ]
            ),
        ),
        (
            "dense",
            torch.stack(
                [
                    torch.stack([dense_w, dense_w[[0, 2, 3, 1]]]),
                    torch.stack([dense_w[[3, 2, 1, 0]], dense_w[[3, 1, 0, 2]]]),
                ]
            ),
        ),
    ],
)
def test_router_topk(moe_type: str, expected: torch.tensor) -> None:
    inputs: torch.tensor = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 4.0, 2.0]],
            [[4.0, 3.0, 2.0, 1.0], [4.0, 2.0, 1.0, 3.0]],
        ]
    )
    r = Router(moe_type, intermediate_size=10, num_experts=inputs.size(2))
    weight = r._topk(inputs)
    assert torch.allclose(weight, expected)
