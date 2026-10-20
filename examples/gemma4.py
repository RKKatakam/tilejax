"""Gemma 4 (E2B, text only) inference on autokernel kernels: weights, a plain-JAX reference, prefill, KV-cache
decoding and greedy generation with the whole decode loop on the device.

The model (Hugging Face transformers' modeling_gemma4.py; E2B: d_model 1536, 35 layers, vocab 262144):

    e = embed[ids] * sqrt(D)                                       (the multiplier rounded to bf16 first)
    ple = (rmsnorm_P(e @ W_ple * D**-0.5) * w_ple + embed_ple[ids] * sqrt(P)) * 2**-0.5    (B, S, L, P)
    h = e;  per layer l (sliding-window or full attention; the last 20 share another layer's keys/values):
        n = rmsnorm(h) * w_in
        q = rope(rmsnorm_h(n @ Wq) * q_norm)
        k = rope(rmsnorm_h(n @ Wk) * k_norm), v = rmsnorm_h(n @ Wv)      non-shared layers; a KV-shared layer
                                      uses those of the last non-shared layer of its type (13: sliding, 14: full)
        a = softmax(q k^T + mask) v                  scale 1; 8 query heads per kv head; sliding: 512 keys
        h = h + rmsnorm(a @ Wo) * w_post_attn
        m = rmsnorm(h) * w_pre_ffn;  h = h + rmsnorm((gelu(m @ Wg) * (m @ Wu)) @ Wd) * w_post_ffn
        h = h + rmsnorm((gelu(h @ Wpg) * ple[:, :, l]) @ Wpp) * w_post_ple
        h = h * layer_scalar[l]
    logits = tanh((rmsnorm(h) * w_final @ embed.T) / 30) * 30

Sliding layers have head_dim 256 and RoPE theta 1e4 over the whole head; full layers head_dim 512 and the
"proportional" RoPE (theta 1e6; only the first 64 of the 256 frequencies turn, the rest are 0). KV-shared
layers have twice as wide MLPs (12288).

WEIGHTS. `load(path)` reads the checkpoint's 600 text tensors (`model.language_model.*`) from
model.safetensors with a minimal reader of the format (an 8-byte header length, a JSON header of names,
dtypes, shapes and byte offsets, then the raw little-endian data: bf16 is read as uint16 and viewed as
ml_dtypes' bfloat16) and packs them once into the layouts the kernels read: every nn.Linear weight (out,
in) transposed to (in, out), norm weights as (1, n) rows, the layer scalar as a (1, D) row. The tied
embedding (V, D) stays as is: the LM head reads it transposed in place. The per-layer-embedding table
(V, 35 * 256: 4.7 GB) stays on the device (the text weights total 9.3 GB of the v5e's 16 GB) and each
step gathers only its tokens' rows. KV-shared layers' unused k/v projections are not loaded.

REFERENCE. `reference_forward` is the model written straight from the definition in plain JAX, rounding
to bf16 where Hugging Face's bf16 model does (between ops; norms and softmax in f32).

FAST PATH (gemma4_kernels.py's kernels, plain-JAX glue for embedding gathers, the PLE inputs, RoPE
tables and cache writes). Every layer is three kernels:
  1. `qkv_projection` (`q_projection` on KV-shared layers): input RMSNorm + projections + q/k/v norms +
     RoPE. Rows are tokens: prefill passes all B * S tokens as one sequence of rows (1, B * S, D) with
     each row's RoPE tables, so its row tiles span prompts (128-token prompts would be too few rows per
     tile to reuse the weights); decode passes the B new tokens the same way.
  2. attention: prefill, `make_attention(head_dim, window)` (flash attention over each prompt's keys);
     decode, decode.py's `gqa_decode` over the KV cache (one tile per sequence, a runtime mask at its
     length). Sliding layers' caches are RING BUFFERS of the last 512 positions (position p in slot
     p % 512; the softmax doesn't care about order), full layers' caches hold every position; KV-shared
     layers read layer 13's / 14's caches, which hold the new token's key already (written before).
  3. `post_attention`: Wo, norms, residuals, the GELU MLP, the per-layer-embedding gate, projection and
     norm, the layer scalar, on rows (all B tokens in one tile when decoding: every weight read once).
The final norm + tied LM head + soft-cap is `lm_head` (logits) or, for greedy decoding, `lm_head_argmax`
(token ids only).

`generate` runs prefill, then the decode steps in ONE jitted `lax.fori_loop` (no host round trip per
token; the caches are donated and updated in place), and reports the prefill time and decode tokens/s.

Run with `uv run python examples/gemma4.py`; without a TPU it runs a tiny random model in the interpreter.
"""

from __future__ import annotations

import functools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from decode import DecodeTiles, gqa_decode
from gemma4_kernels import (
    AttentionTiles,
    HeadTiles,
    PostTiles,
    QkvTiles,
    lm_head,
    lm_head_argmax,
    make_attention,
    post_attention,
    q_projection,
    qkv_projection,
)

B16, F32, I32 = jnp.bfloat16, jnp.float32, jnp.int32
EPS = 1e-6
LANES = 128
SLIDING, FULL = "sliding_attention", "full_attention"
E2B_TYPES = tuple(FULL if layer % 5 == 4 else SLIDING for layer in range(35))


@dataclass(frozen=True)
class Config:
    """Gemma 4's text decoder. `layer_types[l]` is SLIDING or FULL; the last `kv_shared_layers` layers use
    the keys and values of the last earlier layer of their type (and, with `double_wide_mlp`, MLPs of 2 *
    ffn hidden units)."""

    layers: int = 35
    d_model: int = 1536
    heads: int = 8
    kv_heads: int = 1
    head_dim: int = 256  # sliding layers
    global_head_dim: int = 512  # full-attention layers
    ffn: int = 6144
    vocab: int = 262144
    ple_dim: int = 256  # hidden_size_per_layer_input
    kv_shared_layers: int = 20
    layer_types: tuple[str, ...] = E2B_TYPES
    double_wide_mlp: bool = True
    window: int = 512
    sliding_theta: float = 1e4
    full_theta: float = 1e6
    full_rotary_fraction: float = 0.25
    softcap: float = 30.0

    @property
    def first_shared(self) -> int:
        return self.layers - self.kv_shared_layers

    def is_full(self, layer: int) -> bool:
        return self.layer_types[layer] == FULL

    def is_shared(self, layer: int) -> bool:
        return layer >= self.first_shared

    def kv_layer(self, layer: int) -> int:
        """The layer whose keys and values `layer` attends to: itself, or for a KV-shared layer the last
        non-shared layer of its type."""
        if not self.is_shared(layer):
            return layer
        kind = self.layer_types[layer]
        return max(i for i in range(self.first_shared) if self.layer_types[i] == kind)

    def dh(self, layer: int) -> int:
        return self.global_head_dim if self.is_full(layer) else self.head_dim

    def ffn_of(self, layer: int) -> int:
        return self.ffn * (2 if self.double_wide_mlp and self.is_shared(layer) and self.first_shared > 0 else 1)

    def rope(self, layer: int) -> tuple[int, float, float]:
        """(head_dim, theta, rotary fraction) of a layer's RoPE."""
        if self.is_full(layer):
            return self.global_head_dim, self.full_theta, self.full_rotary_fraction
        return self.head_dim, self.sliding_theta, 1.0

    def ring(self, max_len: int) -> int:
        """Positions a sliding layer's cache holds: the window (or every position, if fewer)."""
        return min(self.window, max_len)


jax.tree_util.register_static(Config)

E2B = Config()


@dataclass(frozen=True)
class Tiles:
    """Tunables for the kernels a model call runs (None: tuned on a TPU, the defaults elsewhere)."""

    qkv: QkvTiles | None = None
    attention: AttentionTiles | None = None
    decode: DecodeTiles | None = None
    post: PostTiles | None = None
    head: HeadTiles | None = None


jax.tree_util.register_static(Tiles)


def _call(kernel: Any, *args: Any, tiles: Any = None) -> Any:
    return kernel(*args) if tiles is None else kernel(*args, tiles)


# --------------------------------------------------------------------------- weights


def config_from_json(path: str | Path) -> Config:
    """The text model's Config from a Hugging Face config.json (its `text_config`, or the file itself)."""
    raw = json.loads(Path(path).read_text())
    c = raw.get("text_config", raw)
    if c.get("attention_k_eq_v") or c.get("enable_moe_block") or c.get("num_global_key_value_heads"):
        raise ValueError("only Gemma 4's dense, k != v text models are supported")
    rope = c["rope_parameters"]
    types = list(c["layer_types"][: c["num_hidden_layers"]])
    types[-1] = FULL  # as transformers forces it
    return Config(
        layers=c["num_hidden_layers"],
        d_model=c["hidden_size"],
        heads=c["num_attention_heads"],
        kv_heads=c["num_key_value_heads"],
        head_dim=c["head_dim"],
        global_head_dim=c.get("global_head_dim") or c["head_dim"],
        ffn=c["intermediate_size"],
        vocab=c["vocab_size"],
        ple_dim=c["hidden_size_per_layer_input"],
        kv_shared_layers=c.get("num_kv_shared_layers", 0),
        layer_types=tuple(types),
        double_wide_mlp=bool(c.get("use_double_wide_mlp", False)),
        window=c["sliding_window"],
        sliding_theta=rope[SLIDING]["rope_theta"],
        full_theta=rope[FULL]["rope_theta"],
        full_rotary_fraction=rope[FULL].get("partial_rotary_factor", 1.0),
        softcap=c.get("final_logit_softcapping") or 0.0,
    )


_DTYPES = {"BF16": ml_dtypes.bfloat16, "F16": np.float16, "F32": np.float32, "I32": np.int32, "I64": np.int64}


def read_safetensors(path: str | Path, prefix: str = "") -> dict[str, np.ndarray]:
    """The tensors of a .safetensors file whose names start with `prefix` (the prefix removed), as NumPy
    arrays mapped from the file (no copy). The format: an 8-byte little-endian header length n, n bytes of
    JSON {name: {"dtype", "shape", "data_offsets": [begin, end]}} (offsets into the data after the header),
    then the raw little-endian tensor data."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    data = np.memmap(path, np.uint8, mode="r", offset=8 + n)
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__" or not name.startswith(prefix):
            continue
        begin, end = info["data_offsets"]
        dtype = np.dtype(_DTYPES[info["dtype"]])
        tensors[name[len(prefix) :]] = data[begin:end].view(dtype).reshape(info["shape"])
    return tensors


def _row(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).reshape(1, -1)


def _t(x: np.ndarray) -> np.ndarray:
    """An nn.Linear weight (out, in) as (in, out)."""
    return np.ascontiguousarray(np.asarray(x).T)


def pack(tensors: dict[str, np.ndarray], config: Config) -> dict[str, Any]:
    """Hugging Face's text weights (names after `model.language_model.`) in the kernels' layouts, on the
    device."""
    put = jax.device_put
    layers = []
    for i in range(config.layers):

        def w(name: str) -> np.ndarray:
            return tensors[f"layers.{i}.{name}"]  # noqa: B023 (used at once)

        layer = {
            "attn_norm": put(_row(w("input_layernorm.weight"))),
            "w_q": put(_t(w("self_attn.q_proj.weight"))),
            "q_norm": put(_row(w("self_attn.q_norm.weight"))),
            "wo": put(_t(w("self_attn.o_proj.weight"))),
            "post_attn_norm": put(_row(w("post_attention_layernorm.weight"))),
            "pre_ffn_norm": put(_row(w("pre_feedforward_layernorm.weight"))),
            "w_gate": put(_t(w("mlp.gate_proj.weight"))),
            "w_up": put(_t(w("mlp.up_proj.weight"))),
            "w_down": put(_t(w("mlp.down_proj.weight"))),
            "post_ffn_norm": put(_row(w("post_feedforward_layernorm.weight"))),
            "w_ple_gate": put(_t(w("per_layer_input_gate.weight"))),
            "w_ple_proj": put(_t(w("per_layer_projection.weight"))),
            "post_ple_norm": put(_row(w("post_per_layer_input_norm.weight"))),
            "layer_scale": put(np.full((1, config.d_model), np.asarray(w("layer_scalar")).reshape(()))),
        }
        if not config.is_shared(i):
            layer |= {
                "w_k": put(_t(w("self_attn.k_proj.weight"))),
                "w_v": put(_t(w("self_attn.v_proj.weight"))),
                "k_norm": put(_row(w("self_attn.k_norm.weight"))),
            }
        layers.append(layer)
    return {
        "config": config,
        "embed": put(tensors["embed_tokens.weight"]),
        "embed_ple": put(tensors["embed_tokens_per_layer.weight"]),
        "ple_proj": put(_t(tensors["per_layer_model_projection.weight"])),
        "ple_norm": put(_row(tensors["per_layer_projection_norm.weight"])),
        "final_norm": put(_row(tensors["norm.weight"])),
        "layers": layers,
    }


def load(path: str | Path) -> dict[str, Any]:
    """The text model of a Gemma 4 checkpoint: `path` a directory holding config.json and model.safetensors
    (or the .safetensors file, config.json beside it). bf16 weights on the default device."""
    path = Path(path)
    directory, file = (path, path / "model.safetensors") if path.is_dir() else (path.parent, path)
    config = config_from_json(directory / "config.json")
    return pack(read_safetensors(file, "model.language_model."), config)


def init_params(config: Config, seed: int = 0, dtype: Any = B16) -> dict[str, Any]:
    """Random weights in `load`'s layout (norm weights around 1, q and k norms' around dh**-0.25, layer
    scalars in [0.5, 1])."""
    c, d, p = config, config.d_model, config.ple_dim
    keys = iter(jax.random.split(jax.random.key(seed), 8 + 20 * c.layers))

    def normal(shape: tuple[int, ...], scale: float) -> jax.Array:
        return (jax.random.normal(next(keys), shape) * scale).astype(dtype)

    def norm(n: int, scale: float = 1.0) -> jax.Array:
        return ((1 + 0.1 * jax.random.normal(next(keys), (1, n))) * scale).astype(dtype)

    layers = []
    for i in range(c.layers):
        dh, f = c.dh(i), c.ffn_of(i)
        hd, kd = c.heads * dh, c.kv_heads * dh
        layer = {
            "attn_norm": norm(d),
            "w_q": normal((d, hd), d**-0.5),
            "q_norm": norm(dh, dh**-0.25),  # scores q . k of about unit size (attention's scale is 1)
            "wo": normal((hd, d), hd**-0.5),
            "post_attn_norm": norm(d),
            "pre_ffn_norm": norm(d),
            "w_gate": normal((d, f), d**-0.5),
            "w_up": normal((d, f), d**-0.5),
            "w_down": normal((f, d), f**-0.5),
            "post_ffn_norm": norm(d),
            "w_ple_gate": normal((d, p), d**-0.5),
            "w_ple_proj": normal((p, d), p**-0.5),
            "post_ple_norm": norm(d),
            "layer_scale": jnp.full((1, d), 0.5 + 0.5 * jax.random.uniform(next(keys)), dtype),
        }
        if not c.is_shared(i):
            layer |= {"w_k": normal((d, kd), d**-0.5), "w_v": normal((d, kd), d**-0.5), "k_norm": norm(dh, dh**-0.25)}
        layers.append(layer)
    return {
        "config": c,
        "embed": normal((c.vocab, d), 1.0),
        "embed_ple": normal((c.vocab, c.layers * p), 1.0),
        "ple_proj": normal((d, c.layers * p), d**-0.5),
        "ple_norm": norm(p),
        "final_norm": norm(d),
        "layers": layers,
    }


# --------------------------------------------------------------------------- shared pieces (plain JAX)


def _dot(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.dot(a, b, preferred_element_type=F32)


def _rms(x: jax.Array, gamma: jax.Array | None = None) -> jax.Array:
    """RMSNorm over the last dim in f32 (times a (1, n) weight if given), f32."""
    xf = x.astype(F32)
    out = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + EPS)
    return out if gamma is None else out * gamma.astype(F32).reshape(-1)


def rope_tables(positions: jax.Array, head_dim: int, theta: float, fraction: float = 1.0) -> tuple[jax.Array, ...]:
    """RoPE's f32 cos and sin, (len(positions), head_dim), rotate-half convention over the whole head: the
    first int(fraction * head_dim / 2) frequencies theta ** (-2 i / head_dim) turn, the rest are 0
    (Gemma 4's "proportional" RoPE; fraction 1 is the default RoPE)."""
    turning = int(fraction * head_dim // 2)
    inv_freq = theta ** (-jnp.arange(0, 2 * turning, 2, dtype=F32) / head_dim)
    inv_freq = jnp.concatenate([inv_freq, jnp.zeros(head_dim // 2 - turning, F32)])
    angles = positions.astype(F32).reshape(-1, 1) * jnp.concatenate([inv_freq, inv_freq])[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def embed_tokens(params: dict[str, Any], ids: jax.Array) -> jax.Array:
    """embed[ids] * sqrt(D), the multiplier rounded to the weights' dtype first (as transformers does)."""
    table = params["embed"]
    return table[ids] * jnp.asarray(params["config"].d_model ** 0.5, table.dtype)


def per_layer_inputs(params: dict[str, Any], ids: jax.Array, e: jax.Array) -> jax.Array:
    """The per-layer embeddings (..., L, P): the token's rows of the PLE table (scaled by sqrt(P)) plus the
    normed projection of its embedding e (scaled by D**-0.5), times 2**-0.5; bf16 where transformers is."""
    c = params["config"]
    table = params["embed_ple"]
    dtype = table.dtype
    token = (table[ids] * jnp.asarray(c.ple_dim**0.5, dtype)).reshape(*ids.shape, c.layers, c.ple_dim)
    context = _dot(e, params["ple_proj"]).astype(dtype) * jnp.asarray(c.d_model**-0.5, dtype)
    context = _rms(context.reshape(*ids.shape, c.layers, c.ple_dim), params["ple_norm"]).astype(dtype)
    return (context + token) * jnp.asarray(2**-0.5, dtype)


def _softcap(logits: jax.Array, cap: float) -> jax.Array:
    return jnp.tanh(logits / cap) * cap if cap else logits


# --------------------------------------------------------------------------- the reference: plain JAX


def _heads_rope(t: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotate-half RoPE of t (B, S, heads, dh) with (S, dh) tables, in f32."""
    half = t.shape[-1] // 2
    rotated = jnp.concatenate([-t[..., half:], t[..., :half]], axis=-1)
    return t * cos[None, :, None, :] + rotated * sin[None, :, None, :]


def _attention(q: jax.Array, k: jax.Array, v: jax.Array, window: int | None) -> jax.Array:
    """Causal (sliding-window) GQA attention, scale 1: q (B, S, H, dh), k and v (B, S, Hkv, dh)."""
    b, s, h, dh = q.shape
    kv_heads = k.shape[2]
    qg = q.reshape(b, s, kv_heads, h // kv_heads, dh)
    scores = jnp.einsum("bqgrd,bkgd->bgrqk", qg, k, preferred_element_type=F32)
    rows, cols = jnp.arange(s)[:, None], jnp.arange(s)[None, :]
    visible = cols <= rows if window is None else (cols <= rows) & (rows - cols < window)
    p = jax.nn.softmax(jnp.where(visible, scores, -jnp.inf), axis=-1).astype(v.dtype)
    out = jnp.einsum("bgrqk,bkgd->bqgrd", p, v, preferred_element_type=F32)
    return out.reshape(b, s, h * dh).astype(q.dtype)


def reference_forward(params: dict[str, Any], ids: jax.Array) -> jax.Array:
    """Logits (B, S, vocab), f32, for token ids (B, S) at positions 0 .. S - 1: the model as defined, in plain
    JAX, rounded to bf16 between ops as transformers' bf16 model is."""
    c: Config = params["config"]
    b, s = ids.shape
    dtype = params["embed"].dtype
    x = embed_tokens(params, ids)
    ple = per_layer_inputs(params, ids, x)
    positions = jnp.arange(s)
    kv = {}
    for i, layer in enumerate(params["layers"]):
        dh, theta, fraction = c.rope(i)
        cos, sin = rope_tables(positions, dh, theta, fraction)
        n = _rms(x, layer["attn_norm"]).astype(dtype)
        q = _dot(n, layer["w_q"]).astype(dtype).reshape(b, s, c.heads, dh)
        q = _heads_rope(_rms(q, layer["q_norm"]).astype(dtype).astype(F32), cos, sin).astype(dtype)
        if not c.is_shared(i):
            k = _dot(n, layer["w_k"]).astype(dtype).reshape(b, s, c.kv_heads, dh)
            k = _heads_rope(_rms(k, layer["k_norm"]).astype(dtype).astype(F32), cos, sin).astype(dtype)
            v = _rms(_dot(n, layer["w_v"]).astype(dtype).reshape(b, s, c.kv_heads, dh)).astype(dtype)
            kv[i] = (k, v)
        k, v = kv[c.kv_layer(i)]
        a = _attention(q, k, v, None if c.is_full(i) else c.window)
        a = _rms(_dot(a, layer["wo"]).astype(dtype), layer["post_attn_norm"]).astype(dtype)
        x = x + a
        m = _rms(x, layer["pre_ffn_norm"]).astype(dtype)
        hidden = jax.nn.gelu(_dot(m, layer["w_gate"]).astype(dtype).astype(F32), approximate=True).astype(dtype)
        hidden = hidden * _dot(m, layer["w_up"]).astype(dtype)
        y = _rms(_dot(hidden, layer["w_down"]).astype(dtype), layer["post_ffn_norm"]).astype(dtype)
        x = x + y
        g = jax.nn.gelu(_dot(x, layer["w_ple_gate"]).astype(dtype).astype(F32), approximate=True).astype(dtype)
        g = g * ple[:, :, i, :]
        p = _rms(_dot(g, layer["w_ple_proj"]).astype(dtype), layer["post_ple_norm"]).astype(dtype)
        x = (x + p) * layer["layer_scale"][0, 0]
    n = _rms(x, params["final_norm"]).astype(dtype)
    logits = jnp.einsum("bsd,vd->bsv", n, params["embed"], preferred_element_type=F32).astype(dtype)
    return _softcap(logits.astype(F32), c.softcap)


# --------------------------------------------------------------------------- the model on kernels


def _tables(config: Config, positions: jax.Array) -> dict[str, tuple[jax.Array, jax.Array]]:
    """RoPE tables at `positions` for each layer type."""
    first = {kind: config.layer_types.index(kind) for kind in set(config.layer_types)}
    return {kind: rope_tables(positions, *config.rope(layer)) for kind, layer in first.items()}


def _post_attention(x: jax.Array, attn: jax.Array, ple: jax.Array, layer: dict[str, Any], tiles: Tiles) -> jax.Array:
    names = ("wo", "post_attn_norm", "pre_ffn_norm", "w_gate", "w_up", "w_down", "post_ffn_norm")
    names += ("w_ple_gate", "w_ple_proj", "post_ple_norm", "layer_scale")
    return _call(post_attention, x, attn, ple, *(layer[n] for n in names), tiles=tiles.post)


def _projections(x: jax.Array, layer: dict[str, Any], shared: bool, cos: jax.Array, sin: jax.Array, tiles: Tiles):
    """q (and k, v unless `shared`) of rows x (1, M, D) whose RoPE tables are cos, sin (M, dh)."""
    if shared:
        return _call(q_projection, x, layer["attn_norm"], layer["w_q"], layer["q_norm"], cos, sin, tiles=tiles.qkv)
    weights = (layer["attn_norm"], layer["w_q"], layer["w_k"], layer["w_v"], layer["q_norm"], layer["k_norm"])
    return _call(qkv_projection, x, *weights, cos, sin, tiles=tiles.qkv)


def prefill_hidden(params: dict[str, Any], ids: jax.Array, tiles: Tiles = Tiles()) -> tuple[jax.Array, dict]:
    """The last layer's output (B, S, D) for ids (B, S) at positions 0 .. S - 1, and each non-shared
    layer's keys and values {layer: (k, v)}, (B, S, Hkv * dh) each (RoPE applied: what a cache holds)."""
    c: Config = params["config"]
    b, length = ids.shape
    s = -(-length // LANES) * LANES  # the kernels' blocks are multiples of 128 positions: pad (causal: harmless)
    ids = jnp.pad(ids, [(0, 0), (0, s - length)])
    m = b * s
    e = embed_tokens(params, ids)
    ple = per_layer_inputs(params, ids, e).reshape(m, c.layers, c.ple_dim)
    positions = jnp.tile(jnp.arange(s), b)  # each row's position: the tokens are one sequence of B * S rows
    tables = _tables(c, positions)
    x = e.reshape(m, c.d_model)
    kv = {}
    for i, layer in enumerate(params["layers"]):
        dh = c.dh(i)
        cos, sin = tables[c.layer_types[i]]
        out = _projections(x[None], layer, c.is_shared(i), cos, sin, tiles)
        q = out if c.is_shared(i) else out[0]
        if not c.is_shared(i):
            kv[i] = (out[1].reshape(b, s, -1), out[2].reshape(b, s, -1))
        k, v = kv[c.kv_layer(i)]
        attention = make_attention(dh, None if c.is_full(i) else c.window)
        attn = _call(attention, q.reshape(b, s, -1), k, v, tiles=tiles.attention)
        x = _post_attention(x, attn.reshape(m, -1), ple[:, i, :], layer, tiles)
    kv = {i: (k[:, :length], v[:, :length]) for i, (k, v) in kv.items()}
    return x.reshape(b, s, c.d_model)[:, :length], kv


def forward(params: dict[str, Any], ids: jax.Array, tiles: Tiles = Tiles()) -> jax.Array:
    """Soft-capped logits (B, S, vocab), bf16, for ids (B, S), with the kernels."""
    b, s = ids.shape
    x, _ = prefill_hidden(params, ids, tiles)
    x = x.reshape(b * s, -1)
    return _call(lm_head, x, params["final_norm"], params["embed"], tiles=tiles.head).reshape(b, s, -1)


def empty_caches(config: Config, batch: int, max_len: int, dtype: Any = B16) -> list[dict[str, jax.Array]]:
    """Zero KV caches for the non-shared layers: (B, ring, Hkv * dh) for sliding layers, (B, max_len, Hkv *
    dh) for full ones."""
    caches = []
    for i in range(config.first_shared):
        n = max_len if config.is_full(i) else config.ring(max_len)
        shape = (batch, n, config.kv_heads * config.dh(i))
        caches.append({"k": jnp.zeros(shape, dtype), "v": jnp.zeros(shape, dtype)})
    return caches


def fill_caches(config: Config, kv: dict, lengths: jax.Array, max_len: int) -> list[dict[str, jax.Array]]:
    """Caches holding prompts' keys and values kv {layer: (k, v)} (B, S, ...), prompt b's first lengths[b]
    positions: full layers at their positions, sliding layers' rings slot s holding the last position p <
    lengths[b] with p % ring == s (slots past a short prompt hold don't-cares, which decoding overwrites
    before it reads them)."""
    caches = []
    for i in range(config.first_shared):
        k, v = kv[i]
        s = k.shape[1]
        if config.is_full(i):
            pad = [(0, 0), (0, max_len - s), (0, 0)]
            caches.append({"k": jnp.pad(k, pad), "v": jnp.pad(v, pad)})
            continue
        ring = config.ring(max_len)
        slots = jnp.arange(ring)[None, :]
        last = lengths[:, None] - 1
        position = jnp.where(slots <= last, slots + ring * ((last - slots) // ring), slots)
        position = jnp.minimum(position, s - 1)[:, :, None]
        caches.append({"k": jnp.take_along_axis(k, position, 1), "v": jnp.take_along_axis(v, position, 1)})
    return caches


def prefill(
    params: dict[str, Any], ids: jax.Array, lengths: jax.Array, max_len: int, tiles: Tiles = Tiles()
) -> tuple[jax.Array, list[dict[str, jax.Array]], jax.Array]:
    """Prompts ids (B, S), prompt b being its first lengths[b] tokens (the rest padding): the logits (B, V)
    at each prompt's last token, the KV caches (`empty_caches`' shapes) and the lengths."""
    c: Config = params["config"]
    x, kv = prefill_hidden(params, ids, tiles)
    last = jnp.take_along_axis(x, (lengths - 1)[:, None, None], axis=1)[:, 0]
    logits = _call(lm_head, last, params["final_norm"], params["embed"], tiles=tiles.head)
    return logits, fill_caches(c, kv, lengths, max_len), lengths


def decode_hidden(
    params: dict[str, Any], tokens: jax.Array, caches: list[dict[str, jax.Array]], lengths: jax.Array, tiles: Tiles
) -> tuple[jax.Array, list[dict[str, jax.Array]]]:
    """One decode step up to the final norm: the last layer's output (B, D) for the new tokens (B,), token b
    at position lengths[b], and the caches with the new keys and values written."""
    c: Config = params["config"]
    b = tokens.shape[0]
    e = embed_tokens(params, tokens)
    ple = per_layer_inputs(params, tokens, e)
    tables = _tables(c, lengths)
    rows = jnp.arange(b)
    x, caches = e, list(caches)
    for i, layer in enumerate(params["layers"]):
        dh, full = c.dh(i), c.is_full(i)
        cos, sin = tables[c.layer_types[i]]
        out = _projections(x[None], layer, c.is_shared(i), cos, sin, tiles)
        q = out if c.is_shared(i) else out[0]
        if not c.is_shared(i):
            cache = caches[i]
            slot = lengths if full else lengths % cache["k"].shape[1]
            k_cache = cache["k"].at[rows, slot].set(out[1][0], mode="drop")
            v_cache = cache["v"].at[rows, slot].set(out[2][0], mode="drop")
            caches[i] = {"k": k_cache, "v": v_cache}
        cache = caches[c.kv_layer(i)]
        seen = lengths + 1 if full else jnp.minimum(lengths + 1, cache["k"].shape[1])
        attn = _call(gqa_decode, q.reshape(b, 1, c.heads, dh), cache["k"], cache["v"], seen, tiles=tiles.decode)
        x = _post_attention(x, attn.reshape(b, -1), ple[:, i, :], layer, tiles)
    return x, caches


def decode_logits(params, tokens, caches, lengths, tiles: Tiles = Tiles()):
    """One decode step: soft-capped logits (B, V) for the new tokens (B,), the caches, lengths + 1."""
    x, caches = decode_hidden(params, tokens, caches, lengths, tiles)
    logits = _call(lm_head, x, params["final_norm"], params["embed"], tiles=tiles.head)
    return logits, caches, lengths + 1


def decode_step(params, tokens, caches, lengths, tiles: Tiles = Tiles()):
    """One greedy decode step: the next tokens (B,), the caches, lengths + 1."""
    x, caches = decode_hidden(params, tokens, caches, lengths, tiles)
    best = _call(lm_head_argmax, x, params["final_norm"], params["embed"], tiles=tiles.head)
    return best[:, 0], caches, lengths + 1


def decode_loop(params, tokens, caches, lengths, steps: int, tiles: Tiles = Tiles()):
    """`steps` greedy decode steps in one `lax.fori_loop`: the tokens generated (B, steps), the caches and
    the lengths after them."""

    def body(t, carry):
        token, caches, lengths, out = carry
        token, caches, lengths = decode_step(params, token, caches, lengths, tiles)
        return token, caches, lengths, jax.lax.dynamic_update_slice(out, token[:, None], (0, t))

    out = jnp.zeros((tokens.shape[0], steps), I32)
    _, caches, lengths, out = jax.lax.fori_loop(0, steps, body, (tokens, caches, lengths, out))
    return out, caches, lengths


# --------------------------------------------------------------------------- generation


class Timings(NamedTuple):
    prefill_ms: float
    decode_ms: float
    decode_tokens_per_s: float  # B * (new tokens - 1) / decode time
    tokens_per_s: float  # B * new tokens / (prefill + decode time)


@dataclass
class Generator:
    """Greedy generation for prompts of one shape, compiled once (`compile`), then run any number of
    times: prefill, then every decode step in one jitted loop, the caches donated."""

    params: dict[str, Any]
    batch: int
    prompt_len: int
    new_tokens: int
    tiles: Tiles = field(default_factory=Tiles)

    def __post_init__(self) -> None:
        self.max_len = -(-(self.prompt_len + self.new_tokens) // 128) * 128
        self._prefill = jax.jit(functools.partial(prefill, max_len=self.max_len, tiles=self.tiles))
        steps = self.new_tokens - 1
        self._decode = jax.jit(functools.partial(decode_loop, steps=steps, tiles=self.tiles), donate_argnums=2)
        self._compiled: tuple[Any, Any] | None = None

    def compile(self, prompts: jax.Array, lengths: jax.Array) -> None:
        """Trace (tuning every kernel on a TPU) and compile prefill and the decode loop."""
        prefill_c = self._prefill.lower(self.params, prompts, lengths).compile()
        _, caches, lengths = jax.eval_shape(self._prefill, self.params, prompts, lengths)
        first = jax.ShapeDtypeStruct((self.batch,), I32)
        decode_c = self._decode.lower(self.params, first, caches, lengths).compile()
        self._compiled = (prefill_c, decode_c)

    def __call__(self, prompts: jax.Array, lengths: jax.Array | None = None) -> tuple[jax.Array, Timings]:
        """Greedy tokens (B, new_tokens) after each prompt, and the timings of this run."""
        lengths = jnp.full((self.batch,), self.prompt_len, I32) if lengths is None else lengths
        if self._compiled is None:
            self.compile(prompts, lengths)
        prefill_c, decode_c = self._compiled
        start = time.perf_counter()
        logits, caches, lengths = prefill_c(self.params, prompts, lengths)
        first = jax.block_until_ready(jnp.argmax(logits, axis=-1).astype(I32))
        middle = time.perf_counter()
        rest, _, _ = jax.block_until_ready(decode_c(self.params, first, caches, lengths))
        end = time.perf_counter()
        b, n = self.batch, self.new_tokens
        timings = Timings(
            prefill_ms=(middle - start) * 1e3,
            decode_ms=(end - middle) * 1e3,
            decode_tokens_per_s=b * (n - 1) / max(end - middle, 1e-9),
            tokens_per_s=b * n / (end - start),
        )
        return jnp.concatenate([first[:, None], rest], axis=1), timings


def generate(
    params: dict[str, Any],
    prompts: jax.Array,
    max_new_tokens: int,
    lengths: jax.Array | None = None,
    tiles: Tiles = Tiles(),
) -> tuple[jax.Array, Timings]:
    """Greedy decoding: `max_new_tokens` tokens (B, max_new_tokens) after each prompt (B, S) (prompt b its
    first lengths[b] tokens; all S by default), and the timings (compilation excluded)."""
    b, s = prompts.shape
    return Generator(params, b, s, max_new_tokens, tiles)(prompts, lengths)


TINY = Config(
    layers=5,
    d_model=256,
    heads=2,
    kv_heads=1,
    head_dim=128,
    global_head_dim=256,
    ffn=512,
    vocab=512,
    ple_dim=128,
    kv_shared_layers=2,
    layer_types=(SLIDING, SLIDING, FULL, SLIDING, FULL),
    window=128,
)


if __name__ == "__main__":
    params = init_params(TINY)
    prompts = jax.random.randint(jax.random.key(1), (2, 192), 0, TINY.vocab)
    lengths = jnp.array([192, 150], I32)
    tokens, timings = generate(params, prompts, 8, lengths)
    print("greedy tokens:", tokens.tolist())
    print(timings)
    ours = jax.jit(forward)(params, prompts)
    want = jax.jit(reference_forward)(params, prompts)
    print("max |kernels - plain JAX| logits:", float(jnp.max(jnp.abs(ours.astype(F32) - want))))
