"""Quantum Time-series Transformer (PennyLane implementation).

논문: Park et al. (2025), "Resting-state fMRI Analysis using Quantum Time-series
Transformer", IEEE QCE 2025. https://arxiv.org/abs/2509.00711

모델 흐름
---------
1. 시점마다 ROI 특징 벡터를 선형 변환해 회로 회전 각도로 바꿉니다.
2. 각 시점의 회로를 LCU 계수로 섞습니다 (mix_coeffs).
3. QSVT 다항식 계수로 비선형 변환을 줍니다 (poly_coeffs).
4. feed-forward 회로를 통과시킨 뒤 PauliX/Y/Z 기대값을 측정합니다.
5. 측정값을 선형 층에 넣어 최종 출력을 만듭니다.

입력 규약
---------
`forward`는 `(batch, n_timesteps, feature_dim)` 모양을 받습니다.
저장된 ROI 시계열 배열 `(시간, 뇌 영역)`과 축 순서가 같습니다.
"""

import scipy.constants  # noqa: F401  (scipy<1.13 에서 pennylane import 실패를 막습니다)

import pennylane as qml
import torch


def sim14_circuit(params, wires, layers=1):
    """Sim et al. (2019) circuit 14. 1D/2D(배치) 파라미터를 모두 받습니다."""
    is_batched = params.ndim == 2
    idx = 0

    def take():
        nonlocal idx
        angle = params[:, idx] if is_batched else params[idx]
        idx += 1
        return angle

    for _ in range(layers):
        for i in range(wires):
            qml.RY(take(), wires=i)
        for i in range(wires - 1, -1, -1):
            qml.CRX(take(), wires=[i, (i + 1) % wires])
        for i in range(wires):
            qml.RY(take(), wires=i)
        for i in [wires - 1] + list(range(wires - 1)):
            qml.CRX(take(), wires=[i, (i - 1) % wires])


def apply_lcu_mixing(base_states, unitary_params, qnode_state, coeffs):
    """시점별 회로를 통과시킨 뒤 LCU 계수로 결합합니다.

    모든 (배치 × 시점) 조합을 하나의 큰 배치로 묶어 QNode를 한 번만 호출합니다.
    """
    bsz, n_timesteps, _ = unitary_params.shape
    n_amplitudes = base_states.shape[1]

    flat_params = unitary_params.reshape(bsz * n_timesteps, -1)
    repeated = base_states.repeat_interleave(n_timesteps, dim=0)

    evolved = qnode_state(initial_state=repeated, params=flat_params)
    evolved = evolved.reshape(bsz, n_timesteps, n_amplitudes).to(torch.complex64)

    return torch.einsum("bti,bt->bi", evolved, coeffs.to(torch.complex64))


def apply_qsvt_polynomial(base_states, unitary_params, qnode_state, lcu_coeffs, poly_coeffs):
    """QSVT 다항식 변환을 적용합니다."""
    acc = poly_coeffs[0] * base_states
    working = base_states
    for c in poly_coeffs[1:]:
        working = apply_lcu_mixing(working, unitary_params, qnode_state, lcu_coeffs)
        acc = acc + c * working
    return acc / torch.linalg.vector_norm(poly_coeffs, ord=1)


class QuantumTSTransformer(torch.nn.Module):
    """Quantum Time-series Transformer.

    Parameters
    ----------
    n_qubits : int
        큐빗 수. 논문 설정은 10입니다.
    n_timesteps : int
        시계열 길이.
    degree : int
        QSVT 다항식 차수. 논문 설정은 2입니다.
    n_ansatz_layers : int
        circuit 14 층 수. 논문 설정은 3입니다.
    feature_dim : int
        시점당 특징 수 (HCP-MMP1 180 파셀레이션이면 180).
    output_dim : int
        출력 차원. 이진 분류와 회귀 모두 1입니다.
    dropout : float
        입력 dropout 비율.
    device : torch.device
    """

    def __init__(
        self,
        n_qubits: int,
        n_timesteps: int,
        degree: int,
        n_ansatz_layers: int,
        feature_dim: int,
        output_dim: int,
        dropout: float,
        device,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_timesteps = n_timesteps
        self.degree = degree
        self.n_ansatz_layers = n_ansatz_layers
        self.device = device

        self.n_rots = 4 * n_qubits * n_ansatz_layers
        self.qff_n_rots = 4 * n_qubits

        self.feature_projection = torch.nn.Linear(feature_dim, self.n_rots)
        self.dropout = torch.nn.Dropout(dropout)
        self.rot_sigm = torch.nn.Sigmoid()
        self.output_ff = torch.nn.Linear(3 * n_qubits, output_dim)

        self.poly_coeffs = torch.nn.Parameter(torch.rand(degree + 1))
        self.mix_coeffs = torch.nn.Parameter(torch.rand(n_timesteps, dtype=torch.complex64))
        self.qff_params = torch.nn.Parameter(torch.rand(self.qff_n_rots))

        self.dev = qml.device("default.qubit", wires=n_qubits)

        n_wires = n_qubits
        n_layers = n_ansatz_layers

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _timestep_state(initial_state, params):
            qml.StatePrep(initial_state, wires=range(n_wires))
            sim14_circuit(params, wires=n_wires, layers=n_layers)
            return qml.state()

        self.timestep_state_qnode = _timestep_state

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _qff_expval(initial_state, params):
            qml.StatePrep(initial_state, wires=range(n_wires))
            sim14_circuit(params, wires=n_wires, layers=1)
            obs = (
                [qml.PauliX(i) for i in range(n_wires)]
                + [qml.PauliY(i) for i in range(n_wires)]
                + [qml.PauliZ(i) for i in range(n_wires)]
            )
            return [qml.expval(o) for o in obs]

        self.qff_qnode_expval = _qff_expval

    def forward(self, x):
        """x: (batch, n_timesteps, feature_dim) -> (batch,) 또는 (batch, output_dim)."""
        bsz = x.shape[0]

        timestep_params = self.rot_sigm(self.feature_projection(self.dropout(x)))

        base_states = torch.zeros(
            bsz, 2**self.n_qubits, dtype=torch.complex64, device=x.device
        )
        base_states[:, 0] = 1.0

        mixed = apply_qsvt_polynomial(
            base_states,
            timestep_params,
            self.timestep_state_qnode,
            self.mix_coeffs.repeat(bsz, 1),
            self.poly_coeffs,
        )

        norm = torch.linalg.vector_norm(mixed, dim=1, keepdim=True)
        mixed = mixed / (norm + 1e-9)

        exps = torch.stack(self.qff_qnode_expval(initial_state=mixed, params=self.qff_params), dim=1)
        out = self.output_ff(exps.float())
        return out.squeeze(-1)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
