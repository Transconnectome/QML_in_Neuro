"""파라미터 수를 맞춘 고전 비교 모델.

이 챌린지의 핵심은 **양자 모델과 학습 파라미터 수가 비슷한 고전 모델**과 비교하는
것입니다. 논문이 비교한 고전 트랜스포머는 1.68M~11.2M개였고 양자 모델은 22,157개라서,
적은 표본에서 생긴 성능 차이가 구조 때문인지 단순히 모델 크기 때문인지 구분할 수
없었습니다.

여기에는 참고용 기준 모델 세 개와, 임의의 구조를 목표 파라미터 수에 맞춰 주는
`match_param_budget` 유틸리티가 들어 있습니다. 참가자는 이 중 하나를 그대로 쓰거나,
직접 설계한 구조를 `match_param_budget`으로 맞춰서 비교하면 됩니다.

모든 모델의 입력은 `(batch, n_timesteps, feature_dim)` 입니다.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _pick_nhead(d_model: int) -> int:
    for h in (8, 6, 4, 3, 2, 1):
        if d_model % h == 0:
            return h
    return 1


class MeanLinear(nn.Module):
    """시간 축으로 평균을 낸 뒤 선형 층 하나. 가장 단순한 하한 기준입니다."""

    def __init__(self, feature_dim: int, output_dim: int = 1, dropout: float = 0.1, **_):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, output_dim)

    def forward(self, x):
        return self.fc(self.dropout(x.mean(dim=1))).squeeze(-1)


class SmallTransformer(nn.Module):
    """폭을 줄인 고전 self-attention 트랜스포머.

    위치 임베딩은 기본적으로 쓰지 않습니다 (`learn_pos=True`로 켤 수 있습니다).
    시계열 길이가 360이면 위치 임베딩만으로 파라미터가 수만 개 늘어나기 때문에,
    파라미터 예산을 attention 쪽에 쓰도록 기본값을 정했습니다.
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 42,
        n_layers: int = 1,
        ff_mult: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
        n_timesteps: int | None = None,
        learn_pos: bool = False,
        **_,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos = (
            nn.Parameter(torch.zeros(1, n_timesteps, d_model))
            if (learn_pos and n_timesteps)
            else None
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=_pick_nhead(d_model),
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x):
        h = self.input_proj(x)
        if self.pos is not None:
            h = h + self.pos
        h = self.encoder(h)
        return self.head(h.mean(dim=1)).squeeze(-1)


class Conv1dBaseline(nn.Module):
    """시간 축 1D 합성곱 두 층. attention 없이 지역적 시간 패턴만 봅니다."""

    def __init__(
        self,
        feature_dim: int,
        width: int = 22,
        kernel_size: int = 5,
        output_dim: int = 1,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(feature_dim, width, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size, padding=pad),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, output_dim)

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        return self.head(self.dropout(h.mean(dim=2))).squeeze(-1)


#: `--model` 로 고를 수 있는 고전 모델과, 파라미터 수를 조절할 인자 이름
CLASSICAL_MODELS: dict[str, tuple[type[nn.Module], str, tuple[int, int]]] = {
    "mean_linear": (MeanLinear, "", (0, 0)),
    "small_transformer": (SmallTransformer, "d_model", (4, 512)),
    "conv1d": (Conv1dBaseline, "width", (2, 512)),
}


def match_param_budget(
    factory: Callable[[int], nn.Module],
    target_params: int,
    lo: int,
    hi: int,
) -> tuple[nn.Module, int, int]:
    """`factory(width)` 의 파라미터 수가 목표에 가장 가까워지는 width를 찾습니다.

    이진 탐색으로 목표를 넘지 않는 가장 큰 width를 찾고, 그 다음 값과 비교해
    목표에 더 가까운 쪽을 고릅니다.

    Returns
    -------
    (model, width, n_params)
    """
    best_below = None
    a, b = lo, hi
    while a <= b:
        mid = (a + b) // 2
        n = count_trainable_parameters(factory(mid))
        if n <= target_params:
            best_below = (mid, n)
            a = mid + 1
        else:
            b = mid - 1

    if best_below is None:
        width = lo
        return factory(width), width, count_trainable_parameters(factory(width))

    width, n_below = best_below
    n_above_width = width + 1
    if n_above_width <= hi:
        n_above = count_trainable_parameters(factory(n_above_width))
        if abs(n_above - target_params) < abs(n_below - target_params):
            return factory(n_above_width), n_above_width, n_above
    return factory(width), width, n_below


def build_classical(
    name: str,
    feature_dim: int,
    n_timesteps: int,
    output_dim: int,
    dropout: float,
    param_budget: int | None = None,
    **kwargs,
) -> tuple[nn.Module, int]:
    """고전 비교 모델을 만듭니다. param_budget을 주면 폭을 맞춰 줍니다."""
    if name not in CLASSICAL_MODELS:
        raise KeyError(f"모르는 모델: {name}. 사용 가능: {sorted(CLASSICAL_MODELS)}")

    cls, width_arg, (lo, hi) = CLASSICAL_MODELS[name]
    common = dict(
        feature_dim=feature_dim,
        n_timesteps=n_timesteps,
        output_dim=output_dim,
        dropout=dropout,
        **kwargs,
    )

    if not width_arg or param_budget is None:
        model = cls(**common)
        return model, count_trainable_parameters(model)

    model, width, n_params = match_param_budget(
        lambda w: cls(**{**common, width_arg: w}), param_budget, lo, hi
    )
    print(f"  [파라미터 맞춤] {name}: {width_arg}={width} -> 학습 파라미터 {n_params:,}개 (목표 {param_budget:,})")
    return model, n_params
