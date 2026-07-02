from __future__ import annotations

import torch
from torch import nn


class RaBELValueEncoder(nn.Module):
    """Exponent-aware radial-basis scalar encoder from LimiX."""

    def __init__(
        self,
        embedding_size: int,
        *,
        num_kernels: int = 64,
        sigma: float = 1.0,
        center_min: float = 0.0,
        center_max: float = 10.0,
        exponent_digits: int = 1,
        exponent_embedding_size: int = 32,
    ):
        super().__init__()
        if num_kernels <= 1:
            raise ValueError("num_kernels must be greater than one.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        if exponent_digits <= 0:
            raise ValueError("exponent_digits must be positive.")

        self.embedding_size = embedding_size
        self.num_kernels = num_kernels
        self.exponent_digits = exponent_digits
        self.register_buffer(
            "centers",
            torch.linspace(center_min, center_max, steps=num_kernels),
        )
        self.register_buffer("sigma", torch.tensor(float(sigma)))

        self.sign_embedding = nn.Embedding(2, exponent_embedding_size)
        self.exponent_sign_embedding = nn.Embedding(2, exponent_embedding_size)
        self.exponent_digit_embedding = nn.Embedding(10, exponent_embedding_size)
        control_size = (exponent_digits + 2) * exponent_embedding_size
        self.gate = nn.Sequential(
            nn.Linear(control_size, 4 * exponent_embedding_size),
            nn.GELU(),
            nn.Linear(4 * exponent_embedding_size, 2 * num_kernels),
        )
        self.basis_norm = nn.LayerNorm(num_kernels)
        self.projection = nn.Linear(num_kernels, embedding_size)
        self.output_norm = nn.LayerNorm(embedding_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.sign_embedding.weight)
        nn.init.orthogonal_(self.exponent_sign_embedding.weight)
        nn.init.orthogonal_(self.exponent_digit_embedding.weight)
        nn.init.xavier_uniform_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)
        nn.init.zeros_(self.gate[2].weight)
        gate_bias = self.gate[2].bias.view(2, self.num_kernels)
        nn.init.zeros_(gate_bias[0])
        nn.init.zeros_(gate_bias[1])
        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def _exponent_context(
        self,
        values: torch.Tensor,
        exponents: torch.Tensor,
    ) -> torch.Tensor:
        sign = self.sign_embedding((values < 0).to(torch.long))
        exponent_sign = self.exponent_sign_embedding(
            (exponents < 0).to(torch.long)
        )
        abs_exponents = exponents.abs()
        digits = []
        for power in reversed(range(self.exponent_digits)):
            digit = (abs_exponents // (10**power)) % 10
            digits.append(self.exponent_digit_embedding(digit))
        return torch.cat([sign, exponent_sign, *digits], dim=-1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != 1:
            raise ValueError(
                "RaBELValueEncoder expects a final scalar dimension, got "
                f"{tuple(values.shape)}"
            )

        original_dtype = values.dtype
        x = values.squeeze(-1).to(torch.float32)
        abs_x = x.abs()
        is_zero = abs_x == 0
        safe_abs_x = torch.where(is_zero, torch.ones_like(abs_x), abs_x)
        max_exponent = 10**self.exponent_digits - 1
        exponents = torch.floor(torch.log10(safe_abs_x)).clamp(
            -max_exponent,
            max_exponent,
        ).to(torch.long)
        mantissa = abs_x / torch.pow(
            x.new_tensor(10.0),
            exponents.to(torch.float32),
        )
        mantissa = torch.where(is_zero, torch.zeros_like(mantissa), mantissa)

        centers = self.centers.to(device=x.device, dtype=torch.float32)
        sigma = self.sigma.to(device=x.device, dtype=torch.float32)
        squared_distance = (mantissa.unsqueeze(-1) - centers) ** 2
        basis = torch.exp(-squared_distance / (2.0 * sigma.square()))

        context = self._exponent_context(x, exponents)
        gamma_beta = self.gate(context.to(dtype=self.gate[0].weight.dtype))
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        basis = basis.to(dtype=gamma.dtype)
        basis = basis * torch.sigmoid(gamma) + torch.tanh(beta)
        basis = self.basis_norm(basis)
        output = self.output_norm(self.projection(basis))
        return output.to(dtype=original_dtype)


__all__ = ["RaBELValueEncoder"]
