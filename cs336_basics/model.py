import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class Linear(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 device: torch.device=None, 
                 dtype:torch.dtype=None):
        super().__init__()

        self.W = nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros((out_features, in_features), dtype=dtype, device=device)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")
    

class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 device: torch.device=None, 
                 dtype: torch.dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros((num_embeddings, embedding_dim), dtype=dtype, device=device)))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_select = F.one_hot(token_ids, num_classes=self.num_embeddings).float()
        return einops.einsum(token_select, self.embedding, "... num_embeddings, num_embeddings embedding_dim -> ... embedding_dim")


class RMSNorm(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 eps: float=1e-5, 
                 device=None, 
                 dtype=None):
        super().__init__()

        self.g = nn.Parameter(torch.zeros(d_model, dtype=dtype, device=device))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_x  = torch.sqrt(torch.sum(torch.square(x), dim=-1) / self.d_model + torch.scalar_tensor(self.eps).broadcast_to(x.shape[:-1]))
        result = einops.einsum(torch.div(x, einops.repeat(rms_x, "... -> ... dim", dim=x.shape[-1])), self.g, "... d_model, d_model -> ... d_model")
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 d_ff: int, 
                 device: torch.device=None, 
                 dtype: torch.dtype=None):
        super().__init__()
        self.d_model = d_model
        self.ffn = d_ff # int(d_model * 8 / 3)
        self.w1 = Linear(self.d_model, self.ffn, device, dtype)
        self.w2 = Linear(self.ffn, self.d_model, device, dtype)
        self.w3 = Linear(self.d_model, self.ffn, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_1 = self.w1(x)
        silu = einops.einsum(res_1, torch.sigmoid(res_1), "... d, ... d -> ... d")
        res_2 = self.w3(x)
        silu = einops.einsum(silu, res_2, "... d, ... d -> ... d")
        res_3 = self.w2(silu)
        return res_3


def silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.exp(-x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, 
                 theta: float, 
                 d_k: int, 
                 max_seq_len: int, 
                 dtype: torch.dtype=None,
                 device: torch.device=None):
        super().__init__()
        rotation_matrix = torch.zeros(max_seq_len, d_k, d_k, device=device, dtype=dtype)
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                angle = i / torch.pow(torch.tensor(theta), 2*k / d_k)
                cos, sin = torch.cos(angle), torch.sin(angle)
                rotation_matrix[i, 2*k, 2*k] = cos
                rotation_matrix[i, 2*k, 2*k+1] = -sin
                rotation_matrix[i, 2*k+1, 2*k] = sin 
                rotation_matrix[i, 2*k+1, 2*k+1] = cos

        self.register_buffer("R", rotation_matrix, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        # Understanding of einops.einsum wrong
        r = self.R
        if token_positions is not None:
            r = r[token_positions]
        else:
            seq_len = x.shape[-2]
            r = r[:seq_len]
        r = einops.repeat(r, "seq_num d_k d_v -> batch seq_num d_k d_v", batch=x.shape[0])
        return torch.matmul(r, x.unsqueeze(-1)).squeeze()


def softmax(v: torch.Tensor, dim: int) -> torch.Tensor:
    exp_v = torch.exp(v - torch.max(v, dim=dim, keepdim=True).values)
    return torch.div(exp_v, torch.sum(exp_v, dim=dim, keepdim=True))


def scaled_dot_product_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    k_q = einops.einsum(queries, keys, "... n d, ... m d -> ... n m") / torch.sqrt(torch.tensor(queries.shape[-1], dtype=torch.float32))
    if mask is not None:
        # Here we should use masked_fill and not torch.where and then multiply because there will be problems of sign.
        k_q = k_q.masked_fill(mask == 0, float('-inf'))
    res = einops.einsum(softmax(k_q, dim=-1), values, "... n m, ... m d -> ... n d")
    return res


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 d_k: int, 
                 d_v: int, 
                 d_in: int, 
                 d_model: int, 
                 num_heads: int, 
                 rope: nn.Module=None, 
                 dtype: torch.dtype=None, 
                 device: torch.device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros(d_k, d_in, dtype=dtype, device=device)))
        self.W_K = nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros(d_k, d_in, dtype=dtype, device=device)))
        self.W_V = nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros(d_v, d_in, dtype=dtype, device=device)))
        self.W_O = nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros(d_model, d_v, dtype=dtype, device=device)))

        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        # Understanding of einops.rearrange wrong. Reshape is not transpose.
        # w_q = einops.einsum(self.W_Q, x, "d_k d_in, ... seq_len d_in -> ... seq_len d_k")
        # w_k = einops.einsum(self.W_K, x, "d_k d_in, ... seq_len d_in -> ... seq_len d_k")
        # w_v = einops.einsum(self.W_V, x, "d_v d_model, ... seq_len d_in -> ... seq_len d_v")
        # w_q = einops.rearrange(w_q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        # w_k = einops.rearrange(w_k, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        # w_v = einops.rearrange(w_v, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)
        w_q = einops.rearrange(self.W_Q, "(h d_k) d_in -> h d_k d_in", h=self.num_heads)
        w_k = einops.rearrange(self.W_K, "(h d_k) d_in -> h d_k d_in", h=self.num_heads)
        w_v = einops.rearrange(self.W_V, "(h d_v) d_in -> h d_v d_in", h=self.num_heads)
        w_q = einops.einsum(w_q, x, "h d_k d_in, ... seq_len d_in -> ... h seq_len d_k")
        w_k = einops.einsum(w_k, x, "h d_k d_in, ... seq_len d_in -> ... h seq_len d_k")
        w_v = einops.einsum(w_v, x, "h d_v d_in, ... seq_len d_in -> ... h seq_len d_v")

        if self.rope is not None:
            w_q = self.rope(w_q)
            w_k = self.rope(w_k)

        casual_masking = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float32, device=x.device))
        multi_head = einops.rearrange(scaled_dot_product_attention(w_q, w_k, w_v, casual_masking), "... num_heads seq_len d -> ... seq_len (num_heads d)")
        return einops.einsum(self.W_O, multi_head, "d_model d_v, ... seq_len d_v -> ... seq_len d_model")


class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_k: int, 
                 d_v: int, 
                 d_in: int, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 rope: nn.Module, 
                 dtype: torch.dtype=None, 
                 device: torch.device=None):
        super().__init__()
        self.rms_norm_1 = RMSNorm(d_model, dtype=dtype, device=device)
        self.multi_head_att = MultiHeadSelfAttention(d_k, d_v, d_in, d_model, num_heads, rope, dtype, device)

        self.rms_norm_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_1 = self.multi_head_att(self.rms_norm_1(x)) + x
        y_2 = self.ffn(self.rms_norm_2(y_1)) + y_1
        return y_2


class TransformerLM(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 context_len: int, 
                 d_k: int, 
                 d_v: int, 
                 d_in: int, 
                 d_model: int, 
                 d_ff: int, 
                 num_heads: int, 
                 rope_theta:float,
                 num_layers: int, 
                 device: torch.device=None, 
                 dtype: torch.dtype=None):
        super().__init__()
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_len)
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        self.tf_blocks = [
            TransformerBlock(d_k, d_v, d_in, d_model, num_heads, d_ff, self.rope, dtype, device)
              for _ in range(num_layers)]
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No softmax in the end
        y = self.embedding(x)
        for i in range(len(self.tf_blocks)):
            y = self.tf_blocks[i](y)
        y = self.linear(self.norm(y))
        return y
