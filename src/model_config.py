"""Unified model configuration for KV cache, Mamba state, and FLOP calculations.

Instead of hardcoding model architecture parameters across strategies and config,
a single ``ModelConfig`` captures everything *model-specific* the simulator
needs.  Hardware-specific quantities (GPU compute throughput, PCIe bandwidth)
are passed separately at simulation/metric time — they are not stored on
``ModelConfig``.  Pre-defined configurations are available via
``ModelConfig.from_name()``.

Naming convention: ``flop`` denotes a *count* of floating-point operations
(extensive quantity, e.g. ``prefill_flop``); ``flops`` denotes operations
*per second* (rate / throughput, e.g. the ``gpu_flops`` argument passed to
``compute_run_metrics``).

Usage::

    model = ModelConfig.from_name("qwen3.5-27b")
    print(model.kv_bytes_per_token)       # bytes of KV cache per token
    print(model.prefill_flop(2048))       # total prefill FLOP for 2048 tokens
    print(model.mamba_state_token_equiv)  # recurrent state cost in token-equivalent units
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


# ---------------------------------------------------------------------------
# Low-level FLOP-count / memory helpers (per-layer, used by eviction
# strategies for *relative* scoring — these assume MHA and standard 4×D
# FFN for simplicity; ModelConfig methods use the precise per-model
# formulas).  All helpers return *counts*, not rates.
# ---------------------------------------------------------------------------

def _attn_flop(l: int, d: int) -> float:
    """Attention block FLOP count (MHA approximation): 8*L*D^2 + 4*L^2*D."""
    return 8 * l * d ** 2 + 4 * l ** 2 * d


def _mlp_flop(l: int, d: int) -> float:
    """MLP block FLOP count (non-gated, I=4D approximation): 16*L*D^2."""
    return 16 * l * d ** 2


def _mamba1_flop(l: int, d: int, n: int) -> float:
    """Mamba-1 layer FLOP count: 12*L*D^2 + 16*L*D*N + 10*L*D."""
    return 12 * l * d ** 2 + 16 * l * d * n + 10 * l * d


def _kvs_size(l: int, d: int) -> float:
    """KV cache size in bytes for one attention layer: 2*L*D*2 (fp16)."""
    return 2 * l * d * 2


def _mamba_state_size(d: int, n: int, conv_kernel: int = 4, expand: int = 2) -> float:
    """SSM + conv state size in bytes for one SSM layer (fp16)."""
    return d * n * 2 + (expand * d + 2 * n) * conv_kernel * 2


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Architecture parameters for one model, sufficient for cache simulation.

    Parameters
    ----------
    name:
        Human-readable identifier (used in CLI ``--model`` flag).
    num_attn_layers:
        Number of full-attention layers.
    num_mlp_layers:
        Number of MLP / feed-forward layers.
    d_model:
        Hidden dimension.
    num_ssm_layers:
        Number of SSM / linear-attention layers (0 for pure transformer).
        Both Mamba SSM and linear-attention layers are modelled as
        recurrent layers with a fixed-size state.
    ssm_state_dim:
        SSM state dimension *n* (ignored when ``num_ssm_layers == 0``).
    ssm_conv_kernel:
        Mamba conv1d kernel size.
    ssm_expand:
        Mamba inner-dimension expansion factor.
    dtype_bytes:
        Bytes per element (2 for fp16/bf16, 4 for fp32).
    kv_channels:
        Total KV channel width per layer = ``num_kv_heads * head_dim``.
        ``0`` means MHA (kv_channels = d_model).  Set explicitly for GQA
        models where ``num_kv_heads < num_q_heads``.
    intermediate_size:
        MLP intermediate / FFN hidden dimension.
        ``0`` means use the classic ``4 * d_model`` assumption.
    mlp_gated:
        If True the MLP uses a gated architecture (gate + up + down =
        3 projections); otherwise 2 projections (up + down).
    """

    name: str
    num_attn_layers: int
    num_mlp_layers: int
    d_model: int
    num_ssm_layers: int = 0
    ssm_state_dim: int = 0
    ssm_conv_kernel: int = 4
    ssm_expand: int = 2
    dtype_bytes: int = 2
    kv_channels: int = 0
    intermediate_size: int = 0
    mlp_gated: bool = False

    # -- derived helpers --------------------------------------------------

    @property
    def effective_kv_channels(self) -> int:
        """KV channel width per layer (accounts for GQA)."""
        return self.kv_channels if self.kv_channels > 0 else self.d_model

    @property
    def effective_intermediate_size(self) -> int:
        return self.intermediate_size if self.intermediate_size > 0 else 4 * self.d_model

    # -- KV cache --------------------------------------------------------

    @property
    def kv_bytes_per_token(self) -> int:
        """Total KV cache bytes per token across all attention layers."""
        return self.num_attn_layers * 2 * self.effective_kv_channels * self.dtype_bytes

    def kv_cache_bytes(self, seqlen: int) -> float:
        """Total KV cache bytes for *seqlen* tokens across all attention layers."""
        kv = self.effective_kv_channels
        return self.num_attn_layers * 2 * seqlen * kv * self.dtype_bytes

    def kv_cache_bytes_one_layer(self, seqlen: int) -> float:
        """KV cache bytes for *seqlen* tokens in one attention layer."""
        kv = self.effective_kv_channels
        return 2 * seqlen * kv * self.dtype_bytes

    # -- Recurrent state (Mamba / linear attention) ----------------------

    @property
    def mamba_state_bytes_per_layer(self) -> float:
        """Bytes for one recurrent state snapshot in one SSM layer."""
        if self.num_ssm_layers == 0:
            return 0.0
        return _mamba_state_size(
            self.d_model, self.ssm_state_dim,
            self.ssm_conv_kernel, self.ssm_expand,
        )

    @property
    def mamba_state_bytes_total(self) -> float:
        """Bytes for one recurrent state snapshot across all SSM layers."""
        return self.num_ssm_layers * self.mamba_state_bytes_per_layer

    @property
    def mamba_state_token_equiv(self) -> int:
        """Recurrent state cost expressed in KV-cache token-equivalent units.

        Returns 0 for pure-transformer models (no SSM layers).
        """
        bpt = self.kv_bytes_per_token
        if bpt == 0 or self.num_ssm_layers == 0:
            return 0
        return max(1, int(self.mamba_state_bytes_total / bpt))

    # -- FLOP counts -----------------------------------------------------

    def _attn_flop_layer(self, seqlen: int) -> float:
        """One attention layer prefill FLOP count (GQA-aware).

        Q/O projections: d_model → d_model (2 * 2 * L * D²).
        K/V projections: d_model → kv_channels (2 * 2 * L * D * kv).
        Attention scores + value: 4 * L² * D (Q has d_model total dims).
        """
        d = self.d_model
        kv = self.effective_kv_channels
        return (4 * d + 4 * kv) * seqlen * d + 4 * seqlen ** 2 * d

    def _mlp_flop_layer(self, seqlen: int) -> float:
        """One MLP layer prefill FLOP count.

        Non-gated (up + down): 2 * 2 * L * D * I = 4 * L * D * I.
        Gated (gate + up + down): 3 * 2 * L * D * I = 6 * L * D * I.
        """
        d = self.d_model
        i = self.effective_intermediate_size
        factor = 6 if self.mlp_gated else 4
        return factor * seqlen * d * i

    def _ssm_flop_layer(self, seqlen: int) -> float:
        """One SSM / linear-attention layer prefill FLOP count."""
        if self.num_ssm_layers == 0:
            return 0.0
        return _mamba1_flop(seqlen, self.d_model, self.ssm_state_dim)

    def prefill_flop(self, seqlen: int) -> float:
        """Total FLOP count for a full prefill of *seqlen* tokens (no cache)."""
        total = float(
            self.num_attn_layers * self._attn_flop_layer(seqlen)
            + self.num_mlp_layers * self._mlp_flop_layer(seqlen)
        )
        if self.num_ssm_layers > 0 and self.ssm_state_dim > 0:
            total += self.num_ssm_layers * self._ssm_flop_layer(seqlen)
        return total

    def incremental_prefill_flop(self, total_len: int, cached_prefix_len: int) -> float:
        """FLOP count to compute suffix tokens given a cached prefix.

        For attention layers the suffix tokens still attend to the full
        sequence (including cached prefix keys), so the cost is not simply
        ``prefill_flop(suffix_len)``.
        """
        if cached_prefix_len <= 0:
            return self.prefill_flop(total_len)
        if cached_prefix_len >= total_len:
            return 0.0

        d = self.d_model
        kv = self.effective_kv_channels
        suffix = total_len - cached_prefix_len

        # Attention: QKV+O projection for suffix + attend to all total_len keys
        attn = self.num_attn_layers * (
            (4 * d + 4 * kv) * suffix * d + 4 * suffix * total_len * d
        )
        # MLP: only suffix tokens
        mlp = self.num_mlp_layers * self._mlp_flop_layer(suffix)
        # SSM: only suffix tokens (state loaded from checkpoint)
        ssm = 0.0
        if self.num_ssm_layers > 0 and self.ssm_state_dim > 0:
            ssm = self.num_ssm_layers * _mamba1_flop(suffix, d, self.ssm_state_dim)

        return attn + mlp + ssm

    # -- Capacity conversion ---------------------------------------------

    def gb_to_token_capacity(self, gb: float) -> int:
        """Convert cache capacity in GB to a token budget for this model."""
        if gb <= 0 or gb == float("inf"):
            raise ValueError("gb must be positive and finite")
        return int(gb * (1024 ** 3) / self.kv_bytes_per_token)

    # -- Factory ---------------------------------------------------------

    @classmethod
    def from_name(cls, name: str) -> ModelConfig:
        """Look up a pre-defined model by name.

        Raises ``ValueError`` if the name is not in the registry.
        """
        cfg = MODEL_REGISTRY.get(name)
        if cfg is None:
            available = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(
                f"Unknown model {name!r}. Available: {available}"
            )
        return cfg

    @classmethod
    def list_models(cls) -> list[str]:
        """Return sorted list of registered model names."""
        return sorted(MODEL_REGISTRY)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, ModelConfig] = {}


def _register(*configs: ModelConfig) -> None:
    for c in configs:
        MODEL_REGISTRY[c.name] = c


_register(
    # ── Qwen3.5-27B (Qwen/Qwen3.5-27B) ────────────────────────────────
    # Hybrid: 48 linear-attention + 16 full-attention layers (pattern: 3 linear + 1 full).
    # GQA: 24 Q heads, 4 KV heads, head_dim=256 → kv_channels = 4*256 = 1024.
    # Gated MLP (SiLU): intermediate_size=17408.
    # Linear-attention layers modelled as SSM-like recurrent layers.
    ModelConfig(
        name="qwen3.5-27b",
        num_attn_layers=16,
        num_ssm_layers=48,
        num_mlp_layers=64,
        d_model=5120,
        kv_channels=1024,       # 4 KV heads × 256 head_dim
        intermediate_size=17408,
        mlp_gated=True,
        ssm_state_dim=128,      # linear_key_head_dim
        ssm_conv_kernel=4,      # linear_conv_kernel_dim
    ),
    # ── Jamba 1.5 Mini (AI21) ──────────────────────────────────────────
    # Hybrid Mamba + Attention: 48 SSM, 16 attn, 64 MLP, d=4096, n=128.
    ModelConfig(
        name="jamba-1.5-mini",
        num_attn_layers=16,
        num_mlp_layers=64,
        d_model=4096,
        num_ssm_layers=48,
        ssm_state_dim=128,
    ),
    # ── Generic 28-layer transformer ───────────────────────────────────
    # Matches the original KV_BYTES_PER_TOKEN_DEFAULT = 2*28*4096*2.
    ModelConfig(
        name="transformer-28l-4096d",
        num_attn_layers=28,
        num_mlp_layers=28,
        d_model=4096,
    ),
    # ── Llama 3.1 8B ──────────────────────────────────────────────────
    # 32 layers, d=4096, GQA 8 KV heads × 128 head_dim, gated MLP I=14336.
    ModelConfig(
        name="llama-3.1-8b",
        num_attn_layers=32,
        num_mlp_layers=32,
        d_model=4096,
        kv_channels=1024,       # 8 KV heads × 128 head_dim
        intermediate_size=14336,
        mlp_gated=True,
    ),
    # ── Llama 3.1 70B ─────────────────────────────────────────────────
    # 80 layers, d=8192, GQA 8 KV heads × 128 head_dim, gated MLP I=28672.
    ModelConfig(
        name="llama-3.1-70b",
        num_attn_layers=80,
        num_mlp_layers=80,
        d_model=8192,
        kv_channels=1024,       # 8 KV heads × 128 head_dim
        intermediate_size=28672,
        mlp_gated=True,
    ),
    # ── Qwen3 0.6B ────────────────────────────────────────────────────
    # 28 layers, d=1024, GQA 2 KV heads × 128, gated MLP I=3072.
    ModelConfig(
        name="qwen3-0.6b",
        num_attn_layers=28,
        num_mlp_layers=28,
        d_model=1024,
        kv_channels=256,        # 2 KV heads × 128 head_dim
        intermediate_size=3072,
        mlp_gated=True,
    ),
)

# Default model used when no --model flag is provided.
DEFAULT_MODEL = MODEL_REGISTRY["qwen3.5-27b"]
