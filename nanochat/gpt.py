"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


# A Python-equivalent reference implementation (ignoring performance optimizations)
def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    if attn_mask is not None:
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))

    if enable_gqa:
        num_groups = query.size(-3) // key.size(-3)
        key = key.repeat_interleave(num_groups, dim=-3)
        value = value.repeat_interleave(num_groups, dim=-3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if dropout_p > 0.0:
        attn_weight = F.dropout(attn_weight, p=dropout_p)
    return attn_weight @ value


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.embd = config.embd
        self.head_dim = self.embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)
        Tk = k.size(2)

        enable_gqa = self.n_head != self.n_kv_head
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=enable_gqa
            )
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=False, enable_gqa=enable_gqa
            )
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            # This seems the chunk inference situation?
            attn_mask = torch.zeros(
                (Tq, Tk), dtype=torch.boo, device=q.device
            )  # True = keep False = mask
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tk), dtype=torch.bool, device=q.device)
            )
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa
            )

        # contiguous() is needed for view() call, reshape doesn't require.
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        # For DDP, we want vocab_size divisible by world_size. Also, there are potential performance benefits, see:
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print(
                f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} to be divisible by {pad_vocab_size_to}"
            )
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
                ),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, biase=False)
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weight(self):
        pass
    
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        pass
    
    def get_device(self):
        return self.transformer.wte.weight.device
    
    def estimate_flops(self):
        pass
    
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matri_lr=0.02, weight_decay=0.0):
        pass
    
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        pass
    
    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        pass